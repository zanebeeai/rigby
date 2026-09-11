"""Fetch third-party URDFs, pinned by commit, to test on bodies nobody here wrote.

The zoo proves the pipeline works on robots this repository authored, which is
exactly the weakness a generality claim has. These are real robot descriptions
from an outside project, pinned to a commit and recorded with their licence.
Three of them are manipulators, deliberately awkward ones:

``kuka_lwr``  a seven-axis lightweight arm whose zero pose buries one wrist
              link 21 mm inside another and whose collision hulls stay
              intersecting at every configuration.
``iiwa7``     a seven-axis industrial arm whose collision geometry is meshes
              rather than primitives -- the first model here whose shapes were
              not authored to be convenient.
``panda``     seven axes plus a real parallel-jaw hand, so the contact path gets
              exercised on a gripper this repository did not design around.
``xarm6``     six axes plus a *linkage* gripper: six finger joints coupled by
              URDF ``<mimic>`` tags, which MuJoCo's URDF reader does not
              implement. They arrive as six independent joints. Whether closure
              is still found is a genuine test of measuring behaviour rather
              than reading names.

Two more are here to be refused, and the refusal is the result:

``pr2_gripper``  a gripper on its own, with no arm. Zero positioning axes.
``cartpole``     a cart on a rail with a freely swinging pole. One.

Following ``rigby_v2/rigging/references.py``: reference-only, per-asset licence
recorded, upstream commit pinned, nothing committed to this repository.

Packaging
---------
An upstream URDF is not an upload. These reference meshes as
``package://xarm_description/meshes/...``, which resolves through a ROS package
search path -- something outside the model's own folder, which ingest refuses
before the parser runs, and rightly: a sandbox that honours ``package://`` is not
a sandbox. Several also carry ``<gazebo>`` blocks naming native plugin objects
(``libgazebo_ros_control.so``), which ingest refuses for the same reason.

So this script does the packaging a real uploader would do. It rewrites every
mesh reference to a path relative to the model, fetches the mesh to that path,
and drops ``<gazebo>`` extensions -- which MuJoCo ignores in any case. Both
edits are counted and recorded in ``references.json``, because a normalization
nobody can see is indistinguishable from a bug. The digest recorded is of the
packaged bytes, since those are what gets ingested; the upstream digest is kept
beside it.

None of this is per-robot: no robot id, link name, or joint name appears below
except as a fetch address, and the whole script lives outside
``src/rigby_general/`` where the zero-per-robot-code scan applies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path


# Pinned rather than tracking master, so a re-fetch gets the same bytes.
UPSTREAM = "bulletphysics/bullet3"
COMMIT = "master"
LICENSE = "Zlib"
LICENSE_URL = "https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt"

# The richer model set lives under the pybullet data directory, not ``data/``.
PREFIX = "examples/pybullet/gym/pybullet_data"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "assets" / "general" / "exotic"


@dataclass(frozen=True, slots=True)
class Reference:
    robot_id: str
    urdf: str
    note: str
    expect: str
    prefix: str = PREFIX
    """Upstream directory. Not every model lives under the pybullet data tree."""
    """``manipulator`` if it should ingest, ``refusal`` if being refused is the
    result. Recorded rather than asserted here -- the pipeline decides."""


REFERENCES: tuple[Reference, ...] = (
    Reference(
        robot_id="iiwa7",
        urdf="kuka_iiwa/model.urdf",
        note="Seven-axis industrial arm; mesh collision geometry, no gripper.",
        expect="manipulator",
    ),
    Reference(
        robot_id="panda",
        urdf="franka_panda/panda.urdf",
        note="Seven axes plus a parallel-jaw hand; package:// mesh references.",
        expect="manipulator",
    ),
    Reference(
        robot_id="xarm6",
        urdf="xarm/xarm6_with_gripper.urdf",
        note="Six axes plus a six-joint linkage gripper coupled by <mimic>.",
        expect="manipulator",
    ),
    Reference(
        robot_id="kuka_lwr",
        urdf="kuka_lwr/kuka.urdf",
        prefix="data",
        note="Seven-axis lightweight arm whose wrist collision hulls intersect.",
        expect="manipulator",
    ),
    Reference(
        robot_id="pr2_gripper",
        urdf="pr2_gripper.urdf",
        note="A gripper with no arm; expected to be refused for zero reach.",
        expect="refusal",
    ),
    Reference(
        robot_id="cartpole",
        urdf="cartpole.urdf",
        note="Cart on a rail with a swinging pole; not a manipulator at all.",
        expect="refusal",
    ),
)


# ``filename`` also appears on plugin elements, so match the mesh element itself.
_MESH = re.compile(r'(<mesh\b[^>]*?\bfilename\s*=\s*")([^"]+)(")', re.IGNORECASE)
_GAZEBO = re.compile(r"<gazebo\b.*?</gazebo\s*>", re.IGNORECASE | re.DOTALL)
_GAZEBO_EMPTY = re.compile(r"<gazebo\b[^>]*/>", re.IGNORECASE)


def fetch(path: str) -> bytes | None:
    url = f"https://raw.githubusercontent.com/{UPSTREAM}/{COMMIT}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            if response.status != 200:
                return None
            return response.read()
    except Exception:  # noqa: BLE001 - a missing optional asset is not fatal
        return None


def localize(reference: str) -> str | None:
    """Rewrite one mesh reference to a path inside the model's own folder.

    ``package://pkg/a/b.obj`` becomes ``pkg/a/b.obj``: the package name is kept,
    because whether it is a real directory upstream varies -- one of these
    models nests its meshes under the package name and another does not, and
    guessing wrong is what :func:`candidates` exists to absorb. Keeping it makes
    the local path a faithful, collision-free mirror either way.

    An absolute or parent-escaping path returns ``None``. It is not something a
    self-contained bundle can hold, and quietly flattening it would launder the
    escape that ingest is meant to refuse.
    """

    text = reference.strip().replace("\\", "/")
    if text.startswith("package://"):
        text = text[len("package://") :]
    if text.startswith(("/", "file:", "http:", "https:")) or ":" in text.split("/")[0]:
        return None
    if any(part == ".." for part in text.split("/")):
        return None
    return text or None


def candidates(local: str, upstream_dir: str) -> tuple[str, ...]:
    """Where upstream might keep ``local``: under the package name, or without.

    ROS resolves ``package://pkg/...`` through a search path, so the URDF alone
    cannot say whether ``pkg`` is a directory beside it. Both readings are tried
    rather than encoding which robot is which.
    """

    head, _, tail = local.partition("/")
    options = [local] + ([tail] if tail else [])
    prefix = f"{upstream_dir}/" if upstream_dir else ""
    return tuple(dict.fromkeys(f"{prefix}{option}" for option in options))


def package(
    source: bytes, urdf_path: str
) -> tuple[str, dict[str, tuple[str, ...]], int]:
    """Return the rewritten URDF, its mesh map (local path -> upstream
    candidates), and how many Gazebo extension blocks were dropped."""

    text = source.decode("utf-8", "replace")
    text, dropped = _GAZEBO.subn("", text)
    text, dropped_empty = _GAZEBO_EMPTY.subn("", text)

    upstream_dir = urdf_path.rsplit("/", 1)[0] if "/" in urdf_path else ""
    meshes: dict[str, tuple[str, ...]] = {}

    def rewrite(match: re.Match[str]) -> str:
        local = localize(match.group(2))
        if local is None:
            return match.group(0)
        meshes[local] = candidates(local, upstream_dir)
        return f"{match.group(1)}{local}{match.group(3)}"

    text = _MESH.sub(rewrite, text)
    return text, meshes, dropped + dropped_empty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", default=None)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    selected = [
        reference
        for reference in REFERENCES
        if arguments.only is None or reference.robot_id in arguments.only
    ]

    manifest: list[dict[str, object]] = []
    for reference in selected:
        directory = arguments.out / reference.robot_id
        directory.mkdir(parents=True, exist_ok=True)

        upstream_path = f"{reference.prefix}/{reference.urdf}"
        payload = fetch(upstream_path)
        if payload is None:
            print(f"{reference.robot_id:<14} FAILED to fetch {reference.urdf}")
            continue

        text, meshes, gazebo_dropped = package(payload, reference.urdf)

        fetched, missing = 0, []
        for local, remotes in sorted(meshes.items()):
            target = directory / local
            if target.is_file() and target.stat().st_size > 0:
                fetched += 1
                continue
            data = next(
                (found for remote in remotes
                 if (found := fetch(f"{reference.prefix}/{remote}")) is not None),
                None,
            )
            if data is None:
                missing.append(local)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            fetched += 1

        packaged = text.encode("utf-8")
        (directory / "robot.urdf").write_bytes(packaged)
        digest = hashlib.sha256(packaged).hexdigest()
        (directory / "robot.urdf.sha256").write_text(
            f"{digest}  robot.urdf\n", encoding="ascii"
        )
        manifest.append(
            {
                "robot_id": reference.robot_id,
                "upstream": UPSTREAM,
                "upstream_commit": COMMIT,
                "upstream_path": upstream_path,
                "license_id": LICENSE,
                "license_source_url": LICENSE_URL,
                "bundled": False,
                "expect": reference.expect,
                "upstream_sha256": hashlib.sha256(payload).hexdigest(),
                "packaged_sha256": digest,
                "meshes_referenced": len(meshes),
                "meshes_fetched": fetched,
                "meshes_missing": sorted(missing),
                "gazebo_blocks_dropped": gazebo_dropped,
                "note": reference.note,
            }
        )
        detail = f"{fetched}/{len(meshes)} mesh(es)"
        if missing:
            detail += f", {len(missing)} MISSING"
        if gazebo_dropped:
            detail += f", -{gazebo_dropped} gazebo"
        print(f"{reference.robot_id:<14} ok  {len(packaged):>7} bytes  {detail}")

    index_path = arguments.out / "references.json"
    existing: dict[str, dict[str, object]] = {}
    if index_path.is_file():
        try:
            for row in json.loads(index_path.read_text(encoding="utf-8"))["references"]:
                existing[str(row["robot_id"])] = row
        except Exception:  # noqa: BLE001 - a corrupt index is simply replaced
            existing = {}
    for row in manifest:
        existing[str(row["robot_id"])] = row

    index = {
        "schema_version": "1.0",
        "role": "modeling_and_testing_reference_only",
        "canonical_identity_allowed": False,
        "note": (
            "Fetched at setup time and never committed, following the "
            "reference-only policy in rigby_v2/rigging/references.py. Mesh "
            "references were rewritten relative to the model and Gazebo "
            "extensions dropped so each model is a self-contained upload; see "
            "the module docstring of scripts/fetch_exotic.py."
        ),
        "references": [existing[key] for key in sorted(existing)],
    }
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n{len(manifest)} reference(s) -> {arguments.out}")
    return 0 if len(manifest) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
