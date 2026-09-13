"""Execute the registered G06 transfer roster and record every outcome.

Two tracks, never pooled. The capability-normalized track scales the frozen
fixture to each gripper-bearing body (distances by reach, the cube by
aperture, mass by volume) and runs the canonical transfer once per body: the
prerequisite G06 asks every enabled body to pass. The strict fixed-world track
runs the fixture unchanged, on the bodies the independent feasibility map
classed feasible, over the registered seeds: one attempt per seed, the cube
displaced, its mass and friction scaled by the registered perturbation draws.
Infeasible bodies are attempted once anyway, so their typed refusal is on
record beside the map's reason. Every trial keeps its full physical trace
locally; the canonical trials and every failure are sealed as replayable
bundles and rendered to full-duration video. No API or model calls.

    python any-robot/scripts/g06_transfer_campaign.py --out docs/results/g06-campaign \
        --local any-robot/results/g06-campaign [--bodies a,b] [--seeds N]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from rigby_core.evidence import write_bundle
from rigby_core.simulation.recording import PhysicsRecorder, replay_physics

from rigby_general.contact.placement import PlacementGoal
from rigby_general.contact.transfer import TransferResult, attempt_transfer, collision_policy, transfer_scene_from_environment
from rigby_general.evidence.capture import json_bytes, source_provenance
from rigby_general.evidence.render import render_bundle
from rigby_general.grounding.grounder import figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import EnvironmentV1, SceneObjectV1


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL_DIR = ROOT / "assets/general/research-protocols/g06-transfer-v1"
PROTOCOL = "rigby.transfer-trial/1"


def load_registration() -> tuple[EnvironmentV1, PlacementGoal, dict, dict, dict]:
    registration = json.loads((PROTOCOL_DIR / "registration.json").read_bytes())
    for name in ("environment.json", "goal.json", "feasibility-map.json", "roster.json"):
        digest = hashlib.sha256((PROTOCOL_DIR / name).read_bytes()).hexdigest()
        if registration["files"][name] != digest:
            raise SystemExit(f"{name} does not match its registration; refusing to run")
    env = EnvironmentV1.model_validate_json((PROTOCOL_DIR / "environment.json").read_bytes())
    goal_json = json.loads((PROTOCOL_DIR / "goal.json").read_bytes())
    goal = PlacementGoal(region_minimum_m=tuple(goal_json["region_minimum_m"]), region_maximum_m=tuple(goal_json["region_maximum_m"]),
                         dwell_s=goal_json["dwell_s"], maximum_linear_speed_mps=goal_json["maximum_linear_speed_mps"],
                         maximum_angular_speed_radps=goal_json["maximum_angular_speed_radps"])
    feasibility = json.loads((PROTOCOL_DIR / "feasibility-map.json").read_bytes())
    roster = json.loads((PROTOCOL_DIR / "roster.json").read_bytes())
    return env, goal, feasibility, roster, registration


def perturbed(env: EnvironmentV1, draw: dict) -> EnvironmentV1:
    """The fixture with the cube displaced and its mass and friction scaled by
    a registered draw; the fixtures themselves never move."""

    cube = env.objects[0]
    moved = SceneObjectV1(
        name=cube.name, size_m=cube.size_m, mass_kg=round(cube.mass_kg * draw["mass_multiplier"], 9),
        position_m=(cube.position_m[0] + draw["translation_m"][0], cube.position_m[1] + draw["translation_m"][1], cube.position_m[2] + draw["translation_m"][2]),
        friction=round(cube.friction * draw["friction_multiplier"], 6), rgba=cube.rgba,
    )
    return env.model_copy(update={"objects": (moved,)})


def normalized(env: EnvironmentV1, goal: PlacementGoal, robot, reference_reach_m: float, reference_aperture_m: float) -> tuple[EnvironmentV1, PlacementGoal, dict]:
    """The capability-normalized world: the frozen fixture as it would have
    been authored for this body, using the environment's own scaling rule."""

    effector = robot.morphology.grasping_effectors[0]
    authored = env.model_copy(update={"authored_for_reach_m": reference_reach_m, "authored_for_aperture_m": reference_aperture_m})
    scaled = authored.scaled_to(robot.morphology.scale.reach_radius_m, float(effector.max_aperture_m), robot.morphology.scale.payload_kg)
    span = robot.morphology.scale.reach_radius_m / reference_reach_m
    return scaled, goal.scaled(span), {"length_factor": span, "reference_reach_m": reference_reach_m, "reference_aperture_m": reference_aperture_m,
                                       "cube_span_m": scaled.objects[0].span_m, "cube_mass_kg": scaled.objects[0].mass_kg}


def summarize(result: TransferResult) -> dict:
    return {
        "certified": result.certified, "failed_gate": result.failed_gate,
        "violations": [asdict(v) for v in result.violations],
        "phases": [asdict(p) for p in result.phases], "duration_s": result.duration_s,
        "lift_height_m": result.lift_height_m, "hold_s": result.hold_s, "carry_offset_max_m": result.carry_offset_max_m,
        "peak_force_n": result.peak_force_n, "max_penetration_m": result.max_penetration_m,
        "opposition_achieved": result.opposition_achieved, "placement_dwell_s": result.placement_dwell_s,
        "placement_success": result.placement_success, "placed_inside": result.placed_inside, "released": result.released,
        "unexpected_contacts": [list(p) for p in result.unexpected_contacts],
        "robot_fixture_contacts": [list(p) for p in result.robot_fixture_contacts],
        "collision_policy": result.collision_policy, "path_seed": result.path_seed,
    }


def run_one(robot, source: Path, env: EnvironmentV1, goal: PlacementGoal, *, record: bool) -> tuple[TransferResult, PhysicsRecorder | None, object, float]:
    effector = robot.morphology.grasping_effectors[0]
    chain = next(c for c in robot.morphology.chains if c.chain_id == effector.chain_id)
    frame = build_workspace_frame(robot.finalized.model, robot.morphology, chain, figure_site=figure_site_for(robot.manifest, effector.chain_id))
    scene = transfer_scene_from_environment(robot.manifest, robot.mjcf_xml, env, object_name="cube", destination_fixture="platform", goal=goal, asset_root=source.parent)
    recorder = PhysicsRecorder(scene.model) if record else None
    started = time.perf_counter()
    result = attempt_transfer(robot.manifest, scene, effector, frame, recorder=recorder)
    return result, recorder, scene, time.perf_counter() - started


def seal(destination: Path, *, label: str, robot, source: Path, scene, env: EnvironmentV1, goal: PlacementGoal, result: TransferResult,
         recorder: PhysicsRecorder | None, track: str, trial: dict, caption: str) -> dict:
    """A replayable bundle in the G01 layout, from the recorder's physical record."""

    model = scene.model
    if recorder is None or not recorder.rows["time_s"]:
        record = None
    else:
        record = recorder.finish()
    if record is None:
        data = mujoco.MjData(model)
        rec = PhysicsRecorder(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        rec.capture(data, np.zeros(model.nu), control_time_s=0.0)
        record = rec.finish()
        status = "pre_execution_refusal"
        replay = {"recorded_controls": {"agrees": True, "note": "no motion was executed"}, "agrees": True}
    else:
        status = "success" if result.certified else "runtime_failure"
        replay = {"recorded_controls": replay_physics(model, record)}
        replay["agrees"] = bool(replay["recorded_controls"]["agrees"])
    provenance, archive = source_provenance()
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    outcome = {"status": status, "scope": "contact transfer in an authored world; independent placement evaluator", **summarize(result),
               "refusal": {"stage": "path", "code": result.failed_gate, "detail": result.violations[0].detail} if status == "pre_execution_refusal" else None,
               "actual_physics_duration_s": float(record.arrays["time_s"][-1]), "physical_steps": len(record.arrays["state"]) - 1}
    task = {"goal": "G06", "protocol": PROTOCOL, "track": track, "label": label, "trial": trial,
            "environment": env.model_dump(mode="json"), "placement_goal": asdict(goal),
            "environment_sha256": hashlib.sha256(json_bytes(env.model_dump(mode="json"))).hexdigest(),
            "observation_contract": {"policy": "fully_observed_model_based_baseline", "inputs": ["joint encoders", "model parameters", "contact forces on the gripper"], "vlm": False},
            "interventions": [], "retry_limit": 0, "attempts": 1,
            "limits": "Manifest joint/actuator limits; closure force bounded by the object's needs and half the weakest closure actuator; penetration 4 mm; no relaxed thresholds.",
            "clock_disclosure": {"physics_timestep_s": float(model.opt.timestep), "phase_timing_on_native_physics_time": True}}
    payloads = {
        "model.mjb": buffer.tobytes(), "model.xml": scene.scene.xml.encode("utf-8"),
        "robot.urdf": source.read_bytes(), "robot.json": json_bytes(robot.manifest.model_dump(mode="json")),
        "world.json": json_bytes({"mode": track, "environment_id": env.environment_id, "object_count": len(env.objects), "timestep_s": model.opt.timestep,
                                  "gravity": model.opt.gravity.tolist(), "collision_policy": collision_policy(model), "fault": None}),
        "task.json": json_bytes(task), "outcome.json": json_bytes(outcome),
        "execution.json": json_bytes({"phases": [asdict(p) for p in result.phases], "active_skill_tree": "Sequence(approach, descend, close, lift, hold, carry, lower, release, retreat, dwell)",
                                      "contact_and_sensor_timestamp_semantics": "Initial forward solve, then the previous integration interval; see contact_sample_time_s."}),
        "trace.npz": record.to_bytes(),
        "controller.json": json_bytes({"arm": "rigby_general.gates.control.ComputedTorqueController", "closure": "rigby_general.contact.closure.ClosureController"}),
        "repeats.json": json_bytes({"count": 1, "recorded_control_replay": replay, "note": "one attempt per trial; the recorded controls replay to the recorded states"}),
        "source.json": json_bytes(provenance), "source.zip": archive,
    }
    scale = robot.morphology.scale
    centre = [float(v) for v in (np.asarray(env.objects[0].position_m) + np.asarray(env.fixtures[1].position_m)) / 2.0]
    metadata = {"goal": "G06", "protocol": PROTOCOL, "robot_id": label, "rig_id": robot.manifest.rig_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "outcome": status, "fault": False,
                "simulation_duration_s": float(record.arrays["time_s"][-1]), "reference_duration_s": result.duration_s,
                "reference_clock_matches_physics": True, "caption": caption, "trace_sha256": record.content_hash(),
                "world_sha256": hashlib.sha256(payloads["world.json"]).hexdigest(),
                "camera": {"centre": centre, "reach": max(0.35, 0.45 * scale.reach_radius_m)}, "task_site": "scene_block_center",
                "controller": "computed torque + contact-driven closure", "observation": "model + joint encoders + gripper contact"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, "outcome": status, "trace_sha256": metadata["trace_sha256"],
            "simulation_duration_s": metadata["simulation_duration_s"], "replay_agrees": replay["agrees"]}


def save_trace(local: Path, result: TransferResult) -> str | None:
    if not len(result.times_s):
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, time_s=result.times_s, qpos=result.qpos, ctrl=result.ctrl, demand=result.demand,
                        object_position_m=result.object_position_m, grip_force_n=result.grip_force_n)
    local.write_bytes(stream.getvalue())
    return hashlib.sha256(local.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--bodies")
    parser.add_argument("--seeds", type=int, help="run only the first N registered seeds (engineering smoke; never a scored run)")
    parser.add_argument("--render-successes", type=int, default=5, help="how many successful fixed-world trials per body to render in full")
    args = parser.parse_args()
    env, goal, feasibility, roster, registration = load_registration()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a campaign writes only to a fresh destination")
    args.out.mkdir(parents=True)
    args.local.mkdir(parents=True, exist_ok=True)
    wanted = set(args.bodies.split(",")) if args.bodies else None
    classes = {b["zoo_id"]: b for b in feasibility["bodies"]}
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                  "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__,
                  "registration_sha256": registration["registration_sha256"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "seed_limit": args.seeds, "scored": args.seeds is None, "generation_calls": 0}
    (args.out / "provenance.json").write_bytes(json_bytes(provenance))
    bodies_out = []
    for body in roster["bodies"]:
        zoo_id = body["zoo_id"]
        if wanted and zoo_id not in wanted:
            continue
        source = REPO / body["source_urdf"]
        robot = ingest_robot(source, robot_id=zoo_id)
        if robot.manifest.rig_id != body["rig_id"]:
            raise SystemExit(f"{zoo_id}: intake no longer matches the roster")
        out = args.out / zoo_id
        out.mkdir()
        entry = {"zoo_id": zoo_id, "rig_id": robot.manifest.rig_id, "feasibility_class": classes[zoo_id]["class"], "feasibility_reasons": classes[zoo_id]["reasons"]}
        if not robot.morphology.grasping_effectors:
            entry["normalized"] = {"attempted": False, "reason": "no grasping effector"}
            entry["fixed"] = {"attempted": False, "reason": "no grasping effector"}
            (out / "trials.json").write_bytes(json_bytes(entry))
            bodies_out.append(entry)
            print(json.dumps({"body": zoo_id, "skipped": "no grasping effector"}), flush=True)
            continue

        # -- capability-normalized canonical ---------------------------------
        scaled_env, scaled_goal, normalization = normalized(env, goal, robot, roster["normalization"]["reference_reach_m"], roster["normalization"]["reference_aperture_m"])
        result, recorder, scene, wall = run_one(robot, source, scaled_env, scaled_goal, record=True)
        sealed = seal(out / "normalized" / "physical", label=f"{zoo_id}-normalized", robot=robot, source=source, scene=scene, env=scaled_env, goal=scaled_goal,
                      result=result, recorder=recorder, track="capability_normalized", trial={"kind": "canonical_normalized", "normalization": normalization},
                      caption=f"{zoo_id} | transfer | capability-normalized fixture (x{normalization['length_factor']:.2f})")
        media = render_bundle(out / "normalized" / "physical", out / "normalized" / "media", expected_digest=sealed["sha256"])
        entry["normalized"] = {"attempted": True, **summarize(result), "wall_seconds": wall, "normalization": normalization, "bundle": sealed, "media_sha256": media["sha256"], "frames": media["frame_count"]}
        print(json.dumps({"body": zoo_id, "normalized": result.certified, "gate": result.failed_gate, "wall": round(wall, 1)}), flush=True)

        # -- strict fixed world --------------------------------------------
        fixed = {"attempted": True, "class": classes[zoo_id]["class"], "trials": []}
        result, recorder, scene, wall = run_one(robot, source, env, goal, record=True)
        sealed = seal(out / "fixed-canonical" / "physical", label=f"{zoo_id}-fixed", robot=robot, source=source, scene=scene, env=env, goal=goal,
                      result=result, recorder=recorder, track="strict_fixed_world", trial={"kind": "canonical_fixed"},
                      caption=f"{zoo_id} | transfer | strict fixed world, nominal cube")
        media = render_bundle(out / "fixed-canonical" / "physical", out / "fixed-canonical" / "media", expected_digest=sealed["sha256"])
        fixed["canonical"] = {**summarize(result), "wall_seconds": wall, "bundle": sealed, "media_sha256": media["sha256"], "frames": media["frame_count"]}
        print(json.dumps({"body": zoo_id, "fixed_canonical": result.certified, "gate": result.failed_gate, "wall": round(wall, 1)}), flush=True)
        seeds = body["seeds"] if classes[zoo_id]["class"] == "feasible" else body["seeds"][:1]
        if args.seeds is not None:
            seeds = seeds[: args.seeds]
        rendered_successes = 0
        for draw in seeds:
            trial_env = perturbed(env, draw)
            result, recorder, scene, wall = run_one(robot, source, trial_env, goal, record=True)
            row = {"seed": draw["seed"], "draw": draw, **summarize(result), "wall_seconds": wall}
            row["trace_file_sha256"] = save_trace(args.local / zoo_id / f"seed-{draw['seed']:03d}.npz", result)
            keep = (not result.certified) or rendered_successes < args.render_successes
            if keep:
                sealed = seal(out / "fixed" / f"seed-{draw['seed']:03d}" / "physical", label=f"{zoo_id}-seed{draw['seed']:03d}", robot=robot, source=source, scene=scene,
                              env=trial_env, goal=goal, result=result, recorder=recorder, track="strict_fixed_world", trial={"kind": "seeded_fixed", "seed": draw["seed"], "draw": draw},
                              caption=f"{zoo_id} | transfer | strict fixed world | seed {draw['seed']}" + ("" if result.certified else f" | {result.failed_gate}"))
                media = render_bundle(out / "fixed" / f"seed-{draw['seed']:03d}" / "physical", out / "fixed" / f"seed-{draw['seed']:03d}" / "media", expected_digest=sealed["sha256"])
                row["bundle"] = sealed
                row["media_sha256"] = media["sha256"]
                if result.certified:
                    rendered_successes += 1
            fixed["trials"].append(row)
            print(json.dumps({"body": zoo_id, "seed": draw["seed"], "certified": result.certified, "gate": result.failed_gate, "wall": round(wall, 1)}), flush=True)
        fixed["successes"] = sum(1 for t in fixed["trials"] if t["certified"])
        fixed["count"] = len(fixed["trials"])
        fixed["failure_taxonomy"] = {}
        for t in fixed["trials"]:
            if not t["certified"]:
                fixed["failure_taxonomy"][t["failed_gate"]] = fixed["failure_taxonomy"].get(t["failed_gate"], 0) + 1
        entry["fixed"] = fixed
        (out / "trials.json").write_bytes(json_bytes(entry))
        bodies_out.append({k: entry[k] for k in ("zoo_id", "rig_id", "feasibility_class")} | {
            "normalized_certified": entry["normalized"]["certified"], "normalized_gate": entry["normalized"]["failed_gate"],
            "fixed_canonical_certified": fixed["canonical"]["certified"], "fixed_successes": fixed["successes"], "fixed_count": fixed["count"],
            "fixed_failure_taxonomy": fixed["failure_taxonomy"]})
    summary = {"goal": "G06", "provenance": provenance, "registration_sha256": registration["registration_sha256"],
               "finished_at_utc": datetime.now(timezone.utc).isoformat(), "bodies": bodies_out}
    (args.out / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(bodies_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
