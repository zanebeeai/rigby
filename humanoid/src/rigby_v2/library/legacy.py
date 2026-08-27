"""Safe import of v1 result folders as non-retrievable legacy candidates."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from rigby_core.artifacts import ArtifactStore
from rigby_core.hashing import content_hash, hash_file, sha256_bytes

from .models import AnimationRecord, AnimationRecordInput, AnimationStatus
from .repository import LibraryRepository


_JSON_FILES = (
    "program.json",
    "scene.json",
    "clip.json",
    "metrics.json",
    "provenance.json",
    "request.json",
)
_SAFE_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_INDEX_BINDING_KEYS = (
    "legacy_source_id",
    "legacy_source_index_sha256",
    "legacy_index_entry_sha256",
    "legacy_source_index_schema",
)


@dataclass(frozen=True)
class LegacyArchiveCandidate:
    source_id: str
    folder: Path
    source: AnimationRecordInput


class LegacySink(Protocol):
    def import_legacy(self, source: AnimationRecordInput) -> AnimationRecord: ...


class LegacyRepositorySink:
    """Legacy-only sink that does not register or mutate embedding namespaces."""

    def __init__(
        self,
        repository: LibraryRepository,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.repository = repository
        self.id_factory = id_factory or (lambda: str(uuid4()))

    def import_legacy(self, source: AnimationRecordInput) -> AnimationRecord:
        existing = self.repository.find_legacy_by_lineage(source.lineage_id)
        if existing is not None:
            expected = AnimationRecord.from_input(
                existing.record_id,
                source,
                status=AnimationStatus.LEGACY_CANDIDATE,
            )
            if existing == expected:
                return existing
            if existing.evaluation.get("requires_hydration") is True and not source.evaluation.get(
                "requires_hydration", False
            ):
                hydrated_source = prepare_hydrated_legacy_source(existing, source)
                hydrated = AnimationRecord.from_input(
                    existing.record_id,
                    hydrated_source,
                    status=AnimationStatus.LEGACY_CANDIDATE,
                )
                self.repository.replace_legacy_record(existing.record_id, hydrated)
                return hydrated
            raise ValueError("legacy lineage already exists with different content")
        record = AnimationRecord.from_input(
            self.id_factory(),
            source,
            status=AnimationStatus.LEGACY_CANDIDATE,
        )
        self.repository.insert_record(record)
        return record


def prepare_hydrated_legacy_source(
    deferred: AnimationRecord, source: AnimationRecordInput
) -> AnimationRecordInput:
    """Bind verified full-folder content back to its deferred index provenance."""

    if deferred.evaluation.get("requires_hydration") is not True:
        raise ValueError("legacy record is not awaiting hydration")
    if source.lineage_id != deferred.provenance.get("lineage_id"):
        raise ValueError("hydrated source lineage differs from deferred metadata")
    bindings = {}
    for key in _INDEX_BINDING_KEYS:
        value = deferred.provenance.get(key)
        if value is None:
            raise ValueError("deferred legacy record is missing archive-index binding")
        bindings[key] = value
    return replace(
        source,
        evaluation={
            **source.evaluation,
            "requires_hydration": False,
            "hydration_verified": True,
            "requires_resimulation": True,
            "requires_evaluation": True,
        },
        provenance={
            **source.provenance,
            **bindings,
            "deferred_metadata_record_id": deferred.record_id,
        },
    )


class DeferredLegacyIndexImporter:
    """Quarantine index metadata without reading or copying result payload files."""

    def __init__(
        self,
        root: Path,
        *,
        index_name: str = "index.json",
        maximum_index_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        raw_root = Path(root)
        if raw_root.is_symlink():
            raise ValueError("legacy archive root and index name must be safe")
        self.root = raw_root.resolve()
        if (
            not self.root.is_dir()
            or self.root.is_symlink()
            or not index_name
            or "/" in index_name
            or "\\" in index_name
        ):
            raise ValueError("legacy archive root and index name must be safe")
        raw_index_path = self.root / index_name
        if raw_index_path.is_symlink():
            raise ValueError("legacy archive index must be a safe immediate-child file")
        self.index_path = raw_index_path.resolve()
        if (
            self.index_path.parent != self.root
            or not self.index_path.is_file()
            or self.index_path.is_symlink()
        ):
            raise ValueError("legacy archive index must be a safe immediate-child file")
        if maximum_index_bytes <= 0 or self.index_path.stat().st_size > maximum_index_bytes:
            raise ValueError("legacy archive index exceeds the import size limit")
        self.maximum_index_bytes = maximum_index_bytes

    def _safe_folder(self, source_id: str) -> Path:
        if not _SAFE_SOURCE_ID.fullmatch(source_id) or source_id in {".", ".."}:
            raise ValueError(f"unsafe legacy index result ID: {source_id!r}")
        raw_folder = self.root / source_id
        if raw_folder.is_symlink():
            raise ValueError(
                f"legacy index result {source_id!r} must name a real immediate-child folder"
            )
        folder = raw_folder.resolve()
        if (
            folder.parent != self.root
            or not folder.is_dir()
            or folder.is_symlink()
            or folder.name != source_id
        ):
            raise ValueError(
                f"legacy index result {source_id!r} must name a real immediate-child folder"
            )
        return folder

    def _validated_candidates(self) -> tuple[LegacyArchiveCandidate, ...]:
        encoded = self.index_path.read_bytes()
        if len(encoded) > self.maximum_index_bytes:
            raise ValueError("legacy archive index exceeds the import size limit")
        source_index_sha256 = sha256_bytes(encoded)
        try:
            document = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError("legacy archive index is invalid JSON") from error
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            raise ValueError("legacy archive index must contain a results array")
        source_schema = str(document.get("schema_version", "unknown"))
        seen = set()
        candidates = []
        for entry in document["results"]:
            if not isinstance(entry, dict):
                raise ValueError("legacy archive index entries must be objects")
            source_id = entry.get("id")
            if not isinstance(source_id, str):
                raise ValueError("legacy archive index entry has no string ID")
            if source_id in seen:
                raise ValueError(f"legacy archive index contains duplicate ID {source_id!r}")
            seen.add(source_id)
            folder = self._safe_folder(source_id)
            prompt = entry.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"legacy index result {source_id!r} has no prompt")
            prompt = prompt.strip()
            intent = str(entry.get("intent", "unknown"))
            entry_sha256 = content_hash(entry)
            pointer = {
                "archive_directory_name": self.root.name,
                "source_folder_id": source_id,
                "source_index_name": self.index_path.name,
                "source_index_sha256": source_index_sha256,
                "index_entry_sha256": entry_sha256,
            }
            source = AnimationRecordInput(
                schema_version="1.0",
                split="legacy",
                prompt_text=prompt,
                program={
                    "schema_version": "legacy-index-metadata-v1",
                    "source_text": prompt,
                    "intent": intent,
                    "metadata_only": True,
                },
                world={"legacy_archive_pointer": pointer, "metadata_only": True},
                motion={"metadata_only": True, "hydrated": False},
                evidence={
                    "legacy_archive_pointer": pointer,
                    "indexed_success": entry.get("success"),
                    "indexed_metrics": entry.get("metrics", {}),
                },
                labels={"intent": intent, "legacy_metadata_only": True},
                evaluation={
                    "certified": False,
                    "requires_hydration": True,
                    "hydration_verified": False,
                    "requires_resimulation": True,
                    "requires_evaluation": True,
                },
                provenance={
                    "legacy_source_id": source_id,
                    "legacy_source_index_sha256": source_index_sha256,
                    "legacy_index_entry_sha256": entry_sha256,
                    "legacy_source_index_schema": source_schema,
                    "legacy_metadata_only": True,
                },
                license_id="rigby-poc-local-generated",
                rig_id="legacy-index-unhydrated",
                lineage_id=f"legacy:{source_id}",
                compact_example=f"Deferred legacy prompt: {prompt}\nIntent: {intent}",
            )
            candidates.append(LegacyArchiveCandidate(source_id, folder, source))
        return tuple(candidates)

    def scan(self) -> Iterator[LegacyArchiveCandidate]:
        yield from self._validated_candidates()

    def import_all(
        self,
        service: LegacySink,
        *,
        limit: int | None = None,
    ) -> tuple[AnimationRecord, ...]:
        if limit is not None and limit < 0:
            raise ValueError("legacy import limit cannot be negative")
        candidates = self._validated_candidates()
        selected = candidates if limit is None else candidates[:limit]
        return tuple(service.import_legacy(candidate.source) for candidate in selected)


class LegacyArchiveImporter:
    def __init__(
        self,
        root: Path,
        *,
        artifacts: ArtifactStore,
        maximum_json_bytes: int = 32 * 1024 * 1024,
        maximum_animation_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self.root = root.resolve()
        self.artifacts = artifacts
        self.maximum_json_bytes = maximum_json_bytes
        self.maximum_animation_bytes = maximum_animation_bytes
        if maximum_json_bytes <= 0 or maximum_animation_bytes <= 0:
            raise ValueError("legacy import size limits must be positive")

    def _json(self, folder: Path, name: str) -> dict[str, Any]:
        path = (folder / name).resolve()
        if folder not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"legacy result is missing safe file {name!r}")
        if path.stat().st_size > self.maximum_json_bytes:
            raise ValueError(f"legacy file {name!r} exceeds the import size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"legacy file {name!r} must contain a JSON object")
        return value

    def candidate(self, folder: Path) -> LegacyArchiveCandidate:
        resolved = folder.resolve()
        if self.root not in resolved.parents or resolved.parent != self.root:
            raise ValueError("legacy result folder must be an immediate archive child")
        if resolved.is_symlink() or not resolved.is_dir():
            raise ValueError("legacy result folder must be a real directory")
        documents = {name: self._json(resolved, name) for name in _JSON_FILES}
        animation = (resolved / "animation.glb").resolve()
        if (
            resolved not in animation.parents
            or not animation.is_file()
            or animation.is_symlink()
        ):
            raise ValueError("legacy result is missing a safe animation.glb")
        if animation.stat().st_size > self.maximum_animation_bytes:
            raise ValueError("legacy animation exceeds the import size limit")
        animation_reference = self.artifacts.put_file(
            animation,
            media_type="model/gltf-binary",
            filename=f"{resolved.name}.legacy.glb",
        )
        file_hashes = {
            name: hash_file(resolved / name) for name in (*_JSON_FILES, "animation.glb")
        }
        bundle_hash = content_hash(file_hashes)
        program = documents["program.json"]
        scene = documents["scene.json"]
        provenance = documents["provenance.json"]
        prompt = str(program.get("source_text", "")).strip()
        if not prompt:
            raise ValueError("legacy program has no source text")
        source_id = resolved.name
        hand = str(program.get("hand", "unspecified"))
        limbs = () if hand == "unspecified" else (f"{hand}_hand",)
        primitives = program.get("primitives", [])
        if not isinstance(primitives, list):
            raise ValueError("legacy program primitives must be a list")
        primitive_kinds = tuple(
            sorted(
                {
                    str(item.get("kind"))
                    for item in primitives
                    if isinstance(item, Mapping) and item.get("kind")
                }
            )
        )
        objects = scene.get("objects", [])
        affordances = tuple(
            sorted(
                {
                    str(item.get("kind"))
                    for item in objects
                    if isinstance(item, Mapping) and item.get("kind")
                }
            )
        )
        rig = scene.get("rig", {})
        rig_id = str(
            provenance.get("rig_id")
            or (rig.get("id") if isinstance(rig, Mapping) else "")
        ).strip()
        if not rig_id:
            raise ValueError("legacy result has no rig identity")
        compact = (
            f"Legacy prompt: {prompt}\n"
            f"Intent: {program.get('intent', 'unknown')}\n"
            f"Primitives: {', '.join(primitive_kinds) or 'none'}"
        )
        source = AnimationRecordInput(
            schema_version="1.0",
            split="legacy",
            prompt_text=prompt,
            program=program,
            world=scene,
            motion={
                "legacy_clip_sha256": file_hashes["clip.json"],
                "legacy_clip_schema": documents["clip.json"].get("schema_version"),
            },
            evidence={
                "legacy_animation": animation_reference.model_dump(mode="json"),
                "metrics": documents["metrics.json"],
            },
            labels={
                "intent": program.get("intent", "unknown"),
                "hand": hand,
                "primitive_kinds": primitive_kinds,
            },
            evaluation={
                "certified": False,
                "requires_hydration": False,
                "hydration_verified": True,
                "requires_resimulation": True,
                "requires_evaluation": True,
            },
            provenance={
                **provenance,
                "legacy_source_id": source_id,
                "legacy_bundle_sha256": bundle_hash,
                "legacy_file_sha256": file_hashes,
            },
            license_id="rigby-poc-local-generated",
            rig_id=rig_id,
            lineage_id=f"legacy:{source_id}",
            compact_example=compact,
            object_affordances=affordances,
            limbs=limbs,
            contact_requirements=primitive_kinds,
        )
        return LegacyArchiveCandidate(source_id, resolved, source)

    def scan(self) -> Iterator[LegacyArchiveCandidate]:
        for folder in sorted(self.root.iterdir(), key=lambda item: item.name):
            if folder.is_dir() and not folder.name.startswith("."):
                yield self.candidate(folder)

    def import_all(
        self,
        service: LegacySink,
        *,
        limit: int | None = None,
    ) -> tuple[AnimationRecord, ...]:
        if limit is not None and limit < 0:
            raise ValueError("legacy import limit cannot be negative")
        records = []
        for candidate in self.scan():
            if limit is not None and len(records) >= limit:
                break
            records.append(service.import_legacy(candidate.source))
        return tuple(records)
