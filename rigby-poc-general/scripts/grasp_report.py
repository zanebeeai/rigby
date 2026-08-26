"""Attempt a pick on every gripper-bearing robot, and render the ones that hold.

Reports the measurement behind each verdict rather than a pass or fail, because
the interesting information in a failed grasp is *which* claim failed: opposition
never achieved, the block never rose, it rose and fell, or the jaws ended up
inside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rigby_general.contact import attempt_grasp
from rigby_general.grounding.grounder import figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes import build_grasp_scene

import render_demo


ROOT = Path(__file__).resolve().parents[1]
ZOO_ROOT = ROOT / "assets" / "general" / "zoo"
MEDIA = ROOT / "docs" / "media"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=Path, default=MEDIA)
    arguments = parser.parse_args()

    rows: list[dict[str, object]] = []
    certified = 0
    attempted = 0

    for directory in sorted(ZOO_ROOT.iterdir()):
        if not (directory / "robot.urdf").is_file():
            continue
        robot = ingest_robot(directory / "robot.urdf", robot_id=directory.name)
        grippers = robot.morphology.grasping_effectors
        if not grippers:
            print(f"{directory.name:<18} no gripper; nothing to attempt")
            continue

        effector = grippers[0]
        chain = next(
            item
            for item in robot.morphology.chains
            if item.chain_id == effector.chain_id
        )
        site = figure_site_for(robot.manifest, effector.chain_id)
        frame = build_workspace_frame(
            robot.finalized.model, robot.morphology, chain, figure_site=site
        )
        scene = build_grasp_scene(robot.manifest, robot.mjcf_xml, effector, frame)
        result = attempt_grasp(robot.manifest, scene, effector, frame)

        attempted += 1
        certified += int(result.certified)
        print(
            f"{directory.name:<18} {'HELD ' if result.certified else 'failed'} "
            f"block={2 * scene.block_half_extent_m * 1000:5.1f}mm "
            f"lift={result.lift_height_m * 1000:6.1f}mm "
            f"(needs {0.8 * 2 * scene.block_half_extent_m * 1000:5.1f}) "
            f"grip={result.peak_force_n:6.1f}N "
            f"pen={result.max_penetration_m * 1000:5.2f}mm "
            f"{[violation.code for violation in result.violations]}"
        )
        rows.append(
            {
                "robot_id": directory.name,
                "certified": result.certified,
                "effector": effector.kind.value,
                "block_size_m": round(2 * scene.block_half_extent_m, 5),
                "block_mass_kg": round(scene.block_mass_kg, 5),
                "lift_height_m": round(result.lift_height_m, 5),
                "grip_force_n": round(result.peak_force_n, 3),
                "penetration_m": round(result.max_penetration_m, 5),
                "opposition_achieved": result.opposition_achieved,
                "violations": [violation.code for violation in result.violations],
            }
        )

        if arguments.render and result.certified:
            scale = robot.morphology.scale
            centre = np.array(
                [
                    scene.block_position_m[0],
                    scene.block_position_m[1],
                    scene.block_position_m[2],
                ]
            )
            frames = render_demo.render_trace(
                scene.model,
                result.qpos,
                result.times_s,
                centre=centre,
                reach=scale.reach_radius_m * 0.45,
            )
            path = arguments.out / f"{directory.name}-pick-up-the-block.gif"
            render_demo.write_gif(frames, path)
            print(f"{'':18} -> {path.name}")

    (arguments.out / "grasp-report.json").write_text(
        json.dumps({"attempts": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n{certified}/{attempted} grasps certified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
