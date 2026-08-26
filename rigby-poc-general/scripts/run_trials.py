"""Every gripper against every authored world, and what actually happened.

Two stages, and the split is the point. Admission is decided from ingest
measurements alone, in milliseconds, and answers whether the pairing is even
attemptable. Only what survives is simulated.

That ordering is what an authored world buys. The derived probe places its block
inside the envelope by construction, so it can never report "too far" -- here a
robot can be turned away before anything moves, with the shortfall in
millimetres.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rigby_general.contact.grasp import attempt_grasp
from rigby_general.contact.task import build_task_scene, bystander_motion
from rigby_general.grounding.grounder import figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes import (
    admit_object,
    available_environments,
    load_environment,
)


ROOT = Path(__file__).resolve().parents[1]
ROOTS = (ROOT / "assets" / "general" / "zoo", ROOT / "assets" / "general" / "exotic")


def discover() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in ROOTS:
        if not root.is_dir():
            continue
        for directory in sorted(root.iterdir()):
            source = directory / "robot.urdf"
            if source.is_file():
                found[directory.name] = source
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "trials.json")
    parser.add_argument("--environment", nargs="*", default=None)
    arguments = parser.parse_args()

    prepared = {}
    for robot_id, source in discover().items():
        try:
            robot = ingest_robot(source, robot_id=robot_id)
        except Exception as error:  # noqa: BLE001 - a refused robot has no trials
            print(f"{robot_id:<18} not ingestable: {type(error).__name__}")
            continue
        grippers = robot.morphology.grasping_effectors
        if not grippers:
            print(f"{robot_id:<18} no gripper; nothing to attempt")
            continue
        effector = grippers[0]
        chain = next(
            item
            for item in robot.morphology.chains
            if item.chain_id == effector.chain_id
        )
        frame = build_workspace_frame(
            robot.finalized.model,
            robot.morphology,
            chain,
            figure_site=figure_site_for(robot.manifest, effector.chain_id),
        )
        prepared[robot_id] = (robot, effector, frame)

    rows: list[dict] = []
    names = arguments.environment or list(available_environments())
    for environment_id in names:
        environment = load_environment(environment_id)
        print(f"\n=== {environment_id} ===")
        for robot_id, (robot, effector, frame) in sorted(prepared.items()):
            for item in environment.objects:
                admission = admit_object(robot.manifest, effector, frame, item)
                row = {
                    "environment": environment_id,
                    "robot": robot_id,
                    "object": item.name,
                    "admission": admission.as_dict(),
                    "attempted": False,
                    "held": False,
                    "failure": None,
                    "bystanders": [],
                }
                if not admission.admitted:
                    rows.append(row)
                    print(
                        f"  {robot_id:<17} {item.name:<8} refused: {admission.code}"
                    )
                    continue

                started = time.perf_counter()
                try:
                    scene = build_task_scene(
                        robot.manifest, robot.mjcf_xml, environment, item.name
                    )
                    result = attempt_grasp(robot.manifest, scene, effector, frame)
                except Exception as error:  # noqa: BLE001 - report, do not hide
                    row["failure"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
                    print(f"  {robot_id:<17} {item.name:<8} ERROR {row['failure'][:54]}")
                    continue

                row["attempted"] = True
                row["held"] = bool(result.certified)
                row["failure"] = None if result.certified else result.violations[0].code
                row["lift_m"] = round(result.lift_height_m, 4)
                row["carry_offset_m"] = round(result.carry_offset_m, 4)
                row["penetration_m"] = round(result.max_penetration_m, 5)
                row["elapsed_s"] = round(time.perf_counter() - started, 2)
                row["bystanders"] = [
                    {
                        "name": report.name,
                        "moved_m": round(report.moved_m, 4),
                        "disturbed": report.disturbed,
                    }
                    for report in bystander_motion(
                        scene.model, result.qpos, environment, item.name
                    )
                ]
                verdict = "HELD" if result.certified else row["failure"]
                nudged = [b["name"] for b in row["bystanders"] if b["disturbed"]]
                print(
                    f"  {robot_id:<17} {item.name:<8} {verdict}"
                    + (f"  (disturbed {', '.join(nudged)})" if nudged else "")
                )
                rows.append(row)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    attempted = sum(1 for r in rows if r["attempted"])
    held = sum(1 for r in rows if r["held"])
    print(
        f"\n{len(rows)} pairings, {attempted} admitted and attempted, {held} held"
        f"  ->  {arguments.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
