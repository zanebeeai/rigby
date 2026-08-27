"""Ground one schema program on every zoo robot and compile each result.

The end-to-end check for phase 2: the same body-neutral program, grounded against
six very different bodies, must produce six valid trajectories -- and one
identical role-normalized hash.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rigby_core.motion.compiler import compile_motion_program

from rigby_general.contracts import DirectionV1
from rigby_general.grounding import ground
from rigby_general.morphology import confirm_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.schema.inventory import afforded_entries, load_inventory
from rigby_general.schema.program import (
    BindingRole,
    BoundaryCondition,
    Concurrency,
    Dimensionality,
    MannerV1,
    MotionSchemaProgramV1,
    ReferenceFrame,
    RegionV1,
    Remove,
    RoleBindingV1,
    SegmentLinkV1,
    SegmentV1,
)


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"


def reach_and_retract() -> MotionSchemaProgramV1:
    """Reach out to a distal point, then come back. Two segments, one boundary."""

    inventory = load_inventory()
    reach = inventory.by_id("reach_to_point").schema
    retract = inventory.by_id("retract_from_point").schema

    return MotionSchemaProgramV1(
        program_id="probe-reach-retract",
        source_text="reach out in front of you and then come back",
        segments=(
            SegmentV1(
                segment_id="reach",
                motion_schema=reach,
                figure=RoleBindingV1(role=BindingRole.PRIMARY_EFFECTOR),
                ground=RoleBindingV1(role=BindingRole.BASE),
                region=RegionV1(remove=Remove.DISTAL, dimensionality=Dimensionality.POINT),
                frame=ReferenceFrame.ABSOLUTE,
                manner=MannerV1(),
                boundary=BoundaryCondition.TERMINUS,
            ),
            SegmentV1(
                segment_id="retract",
                motion_schema=retract,
                figure=RoleBindingV1(role=BindingRole.PRIMARY_EFFECTOR),
                ground=RoleBindingV1(role=BindingRole.BASE),
                region=RegionV1(remove=Remove.DISTAL, dimensionality=Dimensionality.POINT),
                frame=ReferenceFrame.ABSOLUTE,
                manner=MannerV1(),
                boundary=BoundaryCondition.TERMINUS,
            ),
        ),
        links=(
            SegmentLinkV1(
                from_segment="reach", to_segment="retract", relation=Concurrency.SEQUENCE
            ),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-frames", action="store_true")
    arguments = parser.parse_args()

    inventory = load_inventory()
    schema_program = reach_and_retract()
    hashes: set[str] = set()
    failures = 0

    for directory in sorted(ZOO_ROOT.iterdir()):
        if not (directory / "robot.urdf").is_file():
            continue
        robot = ingest_robot(directory / "robot.urdf", robot_id=directory.name)
        manifest = robot.manifest
        morphology = robot.morphology

        if arguments.confirm_frames:
            confirmed = confirm_frame(
                morphology.intrinsic_frame, front=DirectionV1(x=1.0, y=0.0, z=0.0)
            )
            morphology = morphology.model_copy(update={"intrinsic_frame": confirmed})
            manifest = manifest.model_copy(update={"morphology": morphology})

        afforded = afforded_entries(inventory, morphology)
        try:
            grounded = ground(
                schema_program, manifest, robot.finalized.model, inventory
            )
        except Exception as error:  # noqa: BLE001 - developer probe
            failures += 1
            print(f"{directory.name:<18} GROUND FAILED {type(error).__name__}: {error}")
            continue

        hashes.add(schema_program.role_normalized_hash())

        try:
            trajectory = compile_motion_program(
                grounded.program, robot.finalized.model, manifest, sample_hz=240
            )
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"{directory.name:<18} COMPILE FAILED {type(error).__name__}: {error}")
            continue

        travel = float(
            np.linalg.norm(
                np.asarray(grounded.program.tracks[0].keyframes[-1].position.x)
                - np.asarray(grounded.program.tracks[0].keyframes[0].position.x)
            )
        )
        print(
            f"{directory.name:<18} afforded={len(afforded):>2}/{len(inventory.entries)}"
            f"  duration={grounded.program.duration_s:5.2f}s"
            f"  frames={trajectory.qpos.shape[0]:>5}"
            f"  reach={morphology.scale.reach_radius_m:.3f}m"
            f"  waypoints={grounded.waypoint_count}"
        )

    print()
    print(f"distinct role-normalized hashes across robots: {len(hashes)}")
    if len(hashes) != 1:
        failures += 1
        print("  SCHEMA INVARIANCE VIOLATED -- the semantic layer depends on the body")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
