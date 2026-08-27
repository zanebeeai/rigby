"""On-disk registry of ingested robots.

Filesystem-backed rather than Postgres-backed. A robot record is a handful of
files that a person may reasonably want to open, diff, or copy between machines,
and keeping the uploaded source next to the model measured from it makes a
morphology traceable without a database session. The Postgres schema is where
*primitives* live, because those need vector lookup and a release lifecycle.

Writes are atomic: a reader sees either the previous complete record or the next
one, never a half-written robot.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .contracts import DirectionV1, RobotAssetManifestV1, RobotMorphologyV1
from .config import base_tree_fingerprint
from .errors import RobotNotFoundError
from .morphology import confirm_frame
from .pipeline import IngestedRobot, ingest_robot


ROBOT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_MANIFEST = "manifest.json"
_MORPHOLOGY = "morphology.json"
_MODEL = "robot.xml"
_PROVENANCE = "provenance.json"


def normalize_robot_id(raw: str) -> str:
    """Fold an uploaded filename into a safe identifier.

    Also the reason a robot id can never contain a path separator: ids are used
    as directory names, and an id like ``../../etc`` would escape the store.
    """

    candidate = re.sub(r"[^a-z0-9_-]+", "_", raw.strip().lower()).strip("_-")
    if not candidate or not ROBOT_ID.match(candidate):
        raise ValueError(f"cannot derive a robot id from {raw!r}")
    return candidate


@dataclass(frozen=True, slots=True)
class RobotRecord:
    robot_id: str
    manifest: RobotAssetManifestV1
    morphology: RobotMorphologyV1
    model_path: Path
    provenance: dict

    @property
    def frame_confirmed(self) -> bool:
        return self.morphology.intrinsic_frame.source.value == "operator_confirmed"

    def summary(self) -> dict:
        scale = self.morphology.scale
        frame = self.morphology.intrinsic_frame
        return {
            "robot_id": self.robot_id,
            "morphology_class": self.morphology.morphology_class.value,
            "source_format": self.manifest.source_format,
            "dof": len(self.manifest.dofs),
            "effectors": [
                {
                    "name": effector.name,
                    "kind": effector.kind.value,
                    "can_grasp": effector.can_grasp,
                    "aperture_m": effector.max_aperture_m,
                }
                for effector in self.morphology.effectors
            ],
            "scale": {
                "reach_radius_m": scale.reach_radius_m,
                "characteristic_length_m": scale.characteristic_length_m,
                "neutral_speed_mps": scale.neutral_speed_mps,
                "payload_kg": scale.payload_kg,
                "total_mass_kg": scale.total_mass_kg,
            },
            "intrinsic_frame": {
                "front": [frame.front.x, frame.front.y, frame.front.z],
                "up": [frame.up.x, frame.up.y, frame.up.z],
                "source": frame.source.value,
                "confidence": frame.confidence,
                "evidence": list(frame.evidence),
            },
            "sites": [
                {"name": site.name, "semantic": site.semantic.value, "body": site.body}
                for site in self.morphology.sites
            ],
        }



def _copy_referenced_assets(
    mjcf_xml: str, source_root: Path, destination: Path
) -> list[str]:
    """Copy every mesh and texture the model names, keeping its own layout.

    A stored model that names ``meshes/kr6_agilus/link_1.stl`` and does not carry
    that file is a model that compiles nowhere. Registration used to keep only
    the XML, so every mesh-bearing robot registered successfully and then failed
    to load -- the API listed seven robots out of eleven and the four missing
    ones were exactly the ones with geometry worth looking at.

    The relative path is preserved rather than flattened, because it is what the
    XML says. Anything resolving outside the source directory is skipped: the
    loader's sandbox already refuses those, and a copy step is not the place to
    quietly re-admit one.
    """

    from xml.etree import ElementTree as ET

    try:
        root = ET.fromstring(mjcf_xml)
    except ET.ParseError:  # pragma: no cover - the model just compiled
        return []

    source_root = source_root.resolve()
    copied: list[str] = []
    seen: set[str] = set()
    for element in root.iter():
        if element.tag not in ("mesh", "texture", "hfield"):
            continue
        reference = element.get("file")
        if not reference or reference in seen:
            continue
        seen.add(reference)

        origin = (source_root / reference).resolve()
        if not origin.is_file():
            continue
        try:
            origin.relative_to(source_root)
        except ValueError:
            continue

        target = destination / reference
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, origin.read_bytes())
        copied.append(reference)

    return sorted(copied)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class RobotRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, robot_id: str) -> Path:
        if not ROBOT_ID.match(robot_id):
            raise ValueError(f"unsafe robot id: {robot_id!r}")
        return self.root / robot_id

    def exists(self, robot_id: str) -> bool:
        return (self.directory(robot_id) / _MANIFEST).is_file()

    def list_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                entry.name
                for entry in self.root.iterdir()
                if entry.is_dir() and (entry / _MANIFEST).is_file()
            )
        )

    # -- writing -----------------------------------------------------------

    def register_source(
        self, source_path: Path, *, robot_id: str | None = None
    ) -> RobotRecord:
        """Ingest a robot from a file on disk and store what was measured."""

        identity = normalize_robot_id(robot_id or source_path.parent.name)
        ingested = ingest_robot(source_path, robot_id=identity)
        return self._store(identity, ingested, source_path)

    def register_bytes(
        self, payload: bytes, *, filename: str, robot_id: str | None = None
    ) -> RobotRecord:
        """Ingest an uploaded body of bytes.

        Written to a scratch directory first so ingest works on a real path and
        the sandboxing in the loader applies unchanged.
        """

        identity = normalize_robot_id(robot_id or Path(filename).stem)
        suffix = Path(filename).suffix or ".urdf"
        scratch = Path(tempfile.mkdtemp(prefix="rigby-upload-"))
        try:
            staged = scratch / identity / f"robot{suffix}"
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(payload)
            ingested = ingest_robot(staged, robot_id=identity)
            return self._store(identity, ingested, staged)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _store(
        self, robot_id: str, ingested: IngestedRobot, source_path: Path
    ) -> RobotRecord:
        directory = self.directory(robot_id)
        directory.mkdir(parents=True, exist_ok=True)

        source_name = f"source{source_path.suffix or '.urdf'}"
        _atomic_write(directory / source_name, source_path.read_bytes())
        _atomic_write(directory / _MODEL, ingested.mjcf_xml.encode("utf-8"))
        copied = _copy_referenced_assets(
            ingested.mjcf_xml, source_path.parent, directory
        )
        _atomic_write(
            directory / _MANIFEST, ingested.manifest.canonical_json().encode("utf-8")
        )
        _atomic_write(
            directory / _MORPHOLOGY,
            ingested.morphology.canonical_json().encode("utf-8"),
        )

        provenance = {
            "robot_id": robot_id,
            "source_file": source_name,
            "asset_files": copied,
            "source_sha256": ingested.manifest.source_asset.sha256,
            "mjcf_sha256": ingested.manifest.mjcf.sha256,
            "manifest_sha256": ingested.manifest.content_hash(),
            "morphology_sha256": ingested.morphology.content_hash(),
            "base_tree": base_tree_fingerprint().as_dict(),
            "integrity": {
                "body_count": ingested.integrity.body_count,
                "actuated_joint_count": ingested.integrity.actuated_joint_count,
                "collider_count": ingested.integrity.collider_count,
                "total_mass_kg": ingested.integrity.total_mass_kg,
            },
        }
        _atomic_write(
            directory / _PROVENANCE,
            json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8"),
        )
        return self.load(robot_id)

    # -- reading -----------------------------------------------------------

    def load(self, robot_id: str) -> RobotRecord:
        directory = self.directory(robot_id)
        manifest_path = directory / _MANIFEST
        if not manifest_path.is_file():
            raise RobotNotFoundError(robot_id)

        manifest = RobotAssetManifestV1.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        morphology = RobotMorphologyV1.model_validate_json(
            (directory / _MORPHOLOGY).read_text(encoding="utf-8")
        )
        provenance = json.loads((directory / _PROVENANCE).read_text(encoding="utf-8"))
        return RobotRecord(
            robot_id=robot_id,
            manifest=manifest,
            morphology=morphology,
            model_path=directory / _MODEL,
            provenance=provenance,
        )

    # -- operator confirmation --------------------------------------------

    def confirm_front(self, robot_id: str, front: DirectionV1 | None) -> RobotRecord:
        """Record that a person accepted or corrected the proposed front axis.

        The one place a human is genuinely required. A bare arm on a pedestal has
        no derivable front, and deictic schemas resolve against it, so the choice
        has to come from someone who knows how the robot is installed.
        """

        record = self.load(robot_id)
        confirmed = confirm_frame(record.morphology.intrinsic_frame, front=front)
        morphology = record.morphology.model_copy(
            update={"intrinsic_frame": confirmed}
        )
        manifest = record.manifest.model_copy(update={"morphology": morphology})

        directory = self.directory(robot_id)
        _atomic_write(
            directory / _MORPHOLOGY, morphology.canonical_json().encode("utf-8")
        )
        _atomic_write(directory / _MANIFEST, manifest.canonical_json().encode("utf-8"))

        provenance = dict(record.provenance)
        provenance["manifest_sha256"] = manifest.content_hash()
        provenance["morphology_sha256"] = morphology.content_hash()
        provenance["frame_confirmed"] = True
        _atomic_write(
            directory / _PROVENANCE,
            json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8"),
        )
        return self.load(robot_id)
