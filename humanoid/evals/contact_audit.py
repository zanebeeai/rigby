"""Measure where the hand passes through the world, against the real surface.

Every contact number this project has published came from pivots -- a wrist
offset, a knuckle centroid, a fingertip distance. Pivots are not what touches.
The rig's palm surface sits 37.6 mm palmar of its metacarpal-head plane, so a
measure taken on the skeleton reports clearance while a viewer watches fingers
disappear into a block.

This audit skins the hand out of the rig's own GLB and measures the signed
distance from that surface to each body in the scene. It is deliberately a
script and not an analyzer:

* ``analysis`` carries a 300 ms wall-clock bound with about 72 ms of headroom,
  and skinning 2,499 vertices per frame does not fit inside it;
* an analyzer would add a metric key, which moves ``metrics_sha256`` on all 47
  cases and needs a re-bless before the first number can be read.

Run read-only against any tree, including ``main``, and nothing has to be
blessed to believe the answer.

    python -m evals.contact_audit
    python -m evals.contact_audit --case object-throw-far --verbose

The support surface is read from the manifest like every other body: the
``table`` object in ``SceneManifest`` is the one the compiler plans against,
the physics proxy builds, and the viewer draws. A scene without one is refused
rather than silently measured against nothing, because hand-through-table is
the largest overlap a viewer sees during a pickup.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from rigby_poc.compiler import compile_motion
from rigby_poc.hand_mesh import box_signed_distance, hand_mesh
from rigby_poc.models import CompileRequest, Hand, MotionProgram, SceneManifest

CASES = Path(__file__).resolve().parent / "corpus" / "cases"

#: Depth below which surfaces are touching rather than overlapping. Contact has
#: to be allowed to happen or nothing can ever be held.
TOUCH_TOLERANCE_M = 0.002


@dataclass(frozen=True)
class Body:
    """One collidable box in the world."""

    name: str
    centre: np.ndarray
    half_extents: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class Overlap:
    """The worst interpenetration found against one body."""

    body: str
    frames: int
    depth_m: float
    time_s: float
    phase: str


def _phase_at(clip: Any, time_s: float) -> str:
    for span in clip.metrics.get("phase_ranges_s", []) or []:
        if span["start_s"] - 1e-9 <= time_s <= span["end_s"] + 1e-9:
            return str(span["kind"])
    return "-"


def _scene_bodies(frame: Any, scene: SceneManifest) -> list[Body]:
    """Every scene object at this frame, the support surface included."""

    bodies: list[Body] = []
    for item in scene.objects:
        pose = frame.objects.get(item.id)
        if pose is None:
            continue
        bodies.append(
            Body(
                name=item.id,
                centre=np.asarray(pose.translation.as_list(), dtype=float),
                half_extents=np.asarray(
                    item.dimensions_m.as_list(), dtype=float
                )
                * 0.5,
                rotation=Rotation.from_quat(pose.rotation.as_list()).as_matrix(),
            )
        )
    return bodies


def audit_case(case_id: str) -> list[Overlap]:
    """Per-body interpenetration for one corpus case."""

    root = CASES / case_id
    scene = SceneManifest.model_validate(json.loads((root / "scene.json").read_text()))
    if scene.support_surface() is None:
        raise SystemExit(f"{case_id}: scene has no support surface to measure against")
    program = MotionProgram.model_validate(
        json.loads((root / "program.json").read_text())
    )
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    if not clip.success or not clip.frames:
        return []

    hands = {program.hand, *program.hands} or {program.hand}
    meshes = {hand: hand_mesh(hand.value) for hand in hands if isinstance(hand, Hand)}

    worst: dict[str, Overlap] = {}
    for frame in clip.frames:
        bodies = _scene_bodies(frame, scene)
        for hand, mesh in meshes.items():
            points = mesh.world(frame.bones)
            for body in bodies:
                depth = -float(
                    box_signed_distance(
                        points, body.centre, body.rotation, body.half_extents
                    ).min()
                )
                if depth <= TOUCH_TOLERANCE_M:
                    continue
                key = f"{hand.value}:{body.name}"
                previous = worst.get(key)
                worst[key] = Overlap(
                    body=key,
                    frames=(previous.frames if previous else 0) + 1,
                    depth_m=max(depth, previous.depth_m if previous else 0.0),
                    time_s=(
                        frame.time_s
                        if previous is None or depth > previous.depth_m
                        else previous.time_s
                    ),
                    phase=(
                        _phase_at(clip, frame.time_s)
                        if previous is None or depth > previous.depth_m
                        else previous.phase
                    ),
                )
    return sorted(worst.values(), key=lambda item: -item.depth_m)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="case id (repeatable)")
    parser.add_argument("--json", type=Path, help="write the findings here")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="list every body, not just the ones that overlap",
    )
    arguments = parser.parse_args(argv)

    case_ids = arguments.case or sorted(
        path.name for path in CASES.iterdir() if path.is_dir()
    )
    findings: dict[str, list[dict[str, Any]]] = {}
    clean = 0
    for case_id in case_ids:
        overlaps = audit_case(case_id)
        findings[case_id] = [
            {
                "body": item.body,
                "frames": item.frames,
                "depth_mm": round(item.depth_m * 1000, 2),
                "time_s": round(item.time_s, 3),
                "phase": item.phase,
            }
            for item in overlaps
        ]
        if not overlaps:
            clean += 1
            if arguments.verbose:
                print(f"{case_id:38} clean")
            continue
        print(f"{case_id:38} " + "; ".join(
            f"{item.body} {item.depth_m * 1000:.1f}mm x{item.frames} ({item.phase})"
            for item in overlaps
        ))

    print(f"\n{clean} of {len(case_ids)} cases show no interpenetration "
          f"beyond {TOUCH_TOLERANCE_M * 1000:.0f} mm.")
    if arguments.json:
        arguments.json.write_text(json.dumps(findings, indent=2), encoding="utf-8")
        print(f"wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
