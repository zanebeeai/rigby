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
from rigby_general.trace import RunTrace, TraceStore
from rigby_general.viewer import build_scene, sample_track


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


def _robot_summary(manifest) -> dict:
    from rigby_general.run import _robot_summary as summary

    return summary(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "trials.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--environment", nargs="*", default=None)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="skip the geometry export; the studio then has no playback for these",
    )
    arguments = parser.parse_args()

    store = TraceStore(arguments.results)
    viewer_root = arguments.results / "viewer" / "env"
    viewer_root.mkdir(parents=True, exist_ok=True)
    # One scene per robot-and-world; the objects in it differ only by which one
    # is being attempted, so the geometry is shared across its attempts.
    exported: dict[str, dict] = {}

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
                trace = RunTrace(
                    prompt=f"pick up the {item.name} in the {environment_id}",
                    robot_id=robot_id,
                    kind="environment_trial",
                )
                trace.robot = _robot_summary(robot.manifest)
                trace.trial = {
                    "environment": environment_id,
                    "description": environment.description,
                    "object": item.name,
                    "object_span_m": round(item.span_m, 4),
                    "object_mass_kg": round(item.mass_kg, 4),
                    "object_position_m": [round(v, 4) for v in item.position_m],
                    "admission": admission.as_dict(),
                }
                trace.record(
                    "admission",
                    "ok" if admission.admitted else "refused",
                    0.0,
                    admission.reason,
                    **admission.as_dict(),
                )

                if not admission.admitted:
                    trace.failure = {
                        "stage": "admission",
                        "code": admission.code,
                        "detail": admission.reason,
                    }
                    store.write(trace)
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
                    trace.record(
                        "scene", "refused", 0.0, str(error)[:300],
                        error=type(error).__name__,
                    )
                    trace.failure = {
                        "stage": "scene",
                        "code": "scene_not_buildable",
                        "detail": str(error)[:300],
                    }
                    store.write(trace)
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
                trace.accepted = bool(result.certified)
                trace.record(
                    "scene", "ok", 0.0,
                    f"{item.name} at {item.position_m}, "
                    f"{item.span_m * 1000:.0f} mm across, "
                    f"{item.mass_kg * 1000:.0f} g -- authored, not derived",
                    fixtures=[f.name for f in environment.fixtures],
                    objects=[o.name for o in environment.objects],
                )
                trace.record(
                    "grasp_gates",
                    "ok" if result.certified else "refused",
                    row["elapsed_s"] * 1000.0,
                    "held" if result.certified else row["failure"],
                    lift_m=row["lift_m"],
                    carry_offset_m=row["carry_offset_m"],
                    penetration_m=row["penetration_m"],
                    violations=[v.code for v in result.violations],
                )
                trace.grasp = {
                    "certified": bool(result.certified),
                    "lift_height_m": row["lift_m"],
                    "carry_offset_m": row["carry_offset_m"],
                    "penetration_m": row["penetration_m"],
                    "grip_force_n": round(result.peak_force_n, 3),
                    "opposition_achieved": bool(result.opposition_achieved),
                    "violations": [
                        {"code": v.code, "detail": v.detail} for v in result.violations
                    ],
                }
                trace.trial["bystanders"] = row["bystanders"]
                if not result.certified:
                    trace.failure = {
                        "stage": "grasp_gates",
                        "code": row["failure"],
                        "detail": result.violations[0].detail,
                    }
                trace.elapsed_seconds = row["elapsed_s"]
                store.write(trace)

                if not arguments.no_viewer:
                    key = f"{environment_id}.{robot_id}"
                    entry = exported.setdefault(
                        key,
                        {
                            "environment": environment_id,
                            "robot_id": robot_id,
                            "scene": build_scene(scene.model),
                            "rest_qpos": [
                                round(float(v), 7)
                                for v in scene.model.qpos0
                            ],
                            "attempts": [],
                        },
                    )
                    entry["attempts"].append(
                        {
                            "trace_id": trace.trace_id,
                            "object": item.name,
                            "held": bool(result.certified),
                            "failure": row["failure"],
                            "track": sample_track(
                                result.times_s, result.qpos, max_frames=90
                            ),
                        }
                    )

                verdict = "HELD" if result.certified else row["failure"]
                nudged = [b["name"] for b in row["bystanders"] if b["disturbed"]]
                print(
                    f"  {robot_id:<17} {item.name:<8} {verdict}"
                    + (f"  (disturbed {', '.join(nudged)})" if nudged else "")
                )
                rows.append(row)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    for key, entry in exported.items():
        (viewer_root / f"{key}.json").write_text(
            json.dumps(entry, separators=(",", ":")), encoding="utf-8"
        )
    if exported:
        size = sum(
            (viewer_root / f"{k}.json").stat().st_size for k in exported
        ) / 1024
        print(f"{len(exported)} playable scenes ({size:.0f} KB) -> {viewer_root}")

    attempted = sum(1 for r in rows if r["attempted"])
    held = sum(1 for r in rows if r["held"])
    print(
        f"\n{len(rows)} pairings, {attempted} admitted and attempted, {held} held"
        f"  ->  {arguments.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
