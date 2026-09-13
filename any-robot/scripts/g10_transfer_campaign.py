"""Run the registered G10 protocol: the closed-loop TransferObject skill on every episode, and keep every outcome.

One root skill definition runs on every episode: the neutral
``transfer_object`` tree bound to the body, its verifications decided by
the G09 conditionals from the declared sensors, its retries bounded at
three per subgoal, its cap 120 s of physics. Nominal episodes perturb the
cube by the registered draw; disturbed episodes add the declared push,
pull or occluder through the step hook. Every episode is one continuous
physics record; every failed, undecided or interrupted episode and the
first five successes per body and class are sealed as replayable bundles
in the local results tree and rendered in full under --out; every episode
keeps its row, its leaf calls, its verdict trail and its oracle judgement. A completion is
counted only when the skill's own verdict is success; whether the object
is then actually placed is judged again from the full recorded state by
the independent oracle, and a success the oracle denies is a false
completion. No API or model calls.

    python any-robot/scripts/g10_transfer_campaign.py --out docs/results/g10-campaign --local any-robot/results/g10-campaign
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from rigby_core.skills import Belief, Interrupt, execute
from rigby_core.skills.examples import transfer_object_library
from rigby_core.simulation.recording import PhysicsRecord

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import render_bundle
from rigby_general.scenes.environment import SceneObjectV1
from rigby_general.sensing import load_policy
from rigby_general.skills import TransferObjectRuntime, TransferObjectSession, disturbance_named, predicates_for_transfer, seal_tree_run
from rigby_general.skills.transfer_runtime import SimulationClock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as corpus_module  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROTOCOL = "rigby.transfer-object-episode/1"
RENDER_SUCCESSES = 5


def perturbed(env, draw: dict):
    cube = env.objects[0]
    moved = SceneObjectV1(name=cube.name, size_m=cube.size_m, mass_kg=round(cube.mass_kg * draw["mass_multiplier"], 9),
                          position_m=(cube.position_m[0] + draw["translation_m"][0], cube.position_m[1] + draw["translation_m"][1], cube.position_m[2] + draw["translation_m"][2]),
                          friction=round(cube.friction * draw["friction_multiplier"], 6), rgba=cube.rgba)
    return env.model_copy(update={"objects": (moved,)})


def oracle_placed(model: mujoco.MjModel, record: PhysicsRecord, goal, *, dwell_s: float) -> dict:
    """The placement judged from the full recorded state over the final
    dwell: the object's whole geometry inside the region on every sample,
    no robot geom touching it, its speeds under the goal's limits."""

    arrays = record.arrays
    times = arrays["time_s"]
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "scene_block_geom")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scene_block_free")
    dof = int(model.jnt_dofadr[joint])
    low, high = np.asarray(goal.region_minimum_m), np.asarray(goal.region_maximum_m)
    data = mujoco.MjData(model)
    start = int(max(0, np.searchsorted(times, times[-1] - dwell_s)))
    worst = {"inside": True, "released": True, "still": True, "samples": 0}
    for index in range(start, len(times)):
        data.qpos[:] = arrays["qpos"][index]
        mujoco.mj_kinematics(model, data)
        centre = np.array(data.geom_xpos[geom], dtype=float)
        half = np.abs(np.array(data.geom_xmat[geom], dtype=float).reshape(3, 3)) @ np.array(model.geom_size[geom], dtype=float)
        if not (np.all(centre - half >= low) and np.all(centre + half <= high)):
            worst["inside"] = False
        lo, hi = int(arrays["contact_offsets"][index]), int(arrays["contact_offsets"][index + 1])
        for (a, b), row, wrench in zip(arrays["contact_geom"][lo:hi], arrays["contact_geometry"][lo:hi], arrays["contact_wrench"][lo:hi]):
            if geom not in (int(a), int(b)):
                continue
            other = int(b) if int(a) == geom else int(a)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ""
            if int(model.geom_bodyid[other]) == 0 or name.startswith(("env_fixture_", "scene_", "env_object_", "g10_")):
                continue
            if float(row[0]) <= 0.0 or abs(float(wrench[0])) > 0.0:
                worst["released"] = False
        qvel = arrays["qvel"][index]
        if float(np.linalg.norm(qvel[dof: dof + 3])) > goal.maximum_linear_speed_mps or float(np.linalg.norm(qvel[dof + 3: dof + 6])) > goal.maximum_angular_speed_radps:
            worst["still"] = False
        worst["samples"] += 1
    worst["dwell_s"] = float(times[-1] - times[start])
    worst["placed"] = bool(worst["inside"] and worst["released"] and worst["still"] and worst["dwell_s"] >= dwell_s - 0.01)
    return worst


def attempts_of(record) -> dict:
    outer = None
    inner = []
    for node in record.root.walk():
        if node.skill_id == "place_until_placed":
            outer = node.evidence.get("attempts", node.attempts)
        if node.skill_id == "acquire_until_held":
            inner.append(node.evidence.get("attempts", node.attempts))
    return {"place_until_placed": outer, "acquire_until_held": inner, "leaf_runs": {k: v for k, v in {}.items()}}


def run_episode(entry: dict, env, goal, policy, library, *, seed_label: str):
    session = TransferObjectSession.open(entry["zoo_id"], REPO / "any-robot/assets/general/zoo" / entry["zoo_id"] / "robot.urdf", perturbed(env, entry["draw"]), goal, policy,
                                         configuration_name=corpus_module.CONFIGURATION, disturbance=disturbance_named(entry["class"]), seed_label=seed_label)
    tree = library.expand("transfer_object", {"object": "cube", "destination": "platform", "effector": session.effector.chain_id})
    interrupt = Interrupt()
    runtime = TransferObjectRuntime(session, interrupt)
    started = time.perf_counter()
    record = execute(tree, library, runtime, predicates_for_transfer(), clock=SimulationClock(session), interrupt=interrupt, belief=Belief())
    return session, runtime, tree, record, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--bodies")
    parser.add_argument("--limit", type=int, help="run only the first N episodes per body and class (engineering smoke; never a scored run)")
    args = parser.parse_args()
    corpus = corpus_module.load_registration()
    registration = json.loads((corpus_module.PROTOCOL / "registration.json").read_bytes())
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a campaign writes only to a fresh destination")
    args.out.mkdir(parents=True)
    args.local.mkdir(parents=True, exist_ok=True)
    env, goal = corpus_module.environment(), corpus_module.registered_goal()
    policy = load_policy(corpus_module.G09 / "policy.json")
    library = transfer_object_library()
    wanted = set(args.bodies.split(",")) if args.bodies else None
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                  "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__,
                  "registration_sha256": registration["registration_sha256"], "library_sha256": corpus["library_sha256"], "policy_sha256": corpus["policy_sha256"],
                  "episode_cap_s": corpus["episode_cap_s"], "retry_budget": corpus["retry_budget"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "limit": args.limit, "scored": args.limit is None and wanted is None, "generation_calls": 0}
    (args.out / "provenance.json").write_bytes(json_bytes(provenance))
    rows = []
    counters: dict[tuple[str, str], int] = {}
    rendered_successes: dict[tuple[str, str], int] = {}
    for entry in corpus["episodes"]:
        body, kind = entry["zoo_id"], entry["class"]
        if wanted and body not in wanted:
            continue
        counters[(body, kind)] = counters.get((body, kind), 0) + 1
        if args.limit is not None and counters[(body, kind)] > args.limit:
            continue
        session, runtime, tree, record, wall = run_episode(entry, env, goal, policy, library, seed_label=entry["episode_id"])
        physical = session.recorder.finish() if session.recorder.rows["time_s"] else None
        oracle = oracle_placed(session.model, physical, goal, dwell_s=goal.dwell_s) if physical is not None else {"placed": False, "samples": 0}
        skill_success = record.verdict.value == "success"
        physics_s = float(session.time_s)
        row = {"episode_id": entry["episode_id"], "zoo_id": body, "class": kind, "seed": entry["seed"], "draw": entry["draw"], "verdict": record.verdict.value, "root_reason": record.root.reason,
               "interrupted": record.interrupted, "interrupt_reason": record.interrupt_reason, "physics_s": physics_s, "within_cap": physics_s <= corpus["episode_cap_s"] + 1e-9,
               "wall_seconds": wall, "attempts": attempts_of(record), "leaf_calls": runtime.calls, "verdict_trail": runtime.verdicts, "events": session.events,
               "disturbance": session.disturbance.log if session.disturbance is not None else [], "skill_success": skill_success, "oracle": oracle,
               "false_completion": bool(skill_success and not oracle["placed"]), "recovered": bool(kind != "nominal" and skill_success),
               "physical_steps": session.steps, "effector": session.effector.chain_id}
        keep = not skill_success or rendered_successes.get((body, kind), 0) < RENDER_SUCCESSES
        if keep:
            bundle_dir = args.local / body / kind / f"seed-{entry['seed']:03d}" / "physical"
            caption = f"{body} | transfer_object | {kind} | seed {entry['seed']} | {record.verdict.value}" + (f" | {record.root.reason}" if record.root.reason else "")
            sealed = seal_tree_run(bundle_dir, session=session, library=library, tree=tree, record=record, label=entry["episode_id"], caption=caption, runtime_calls=runtime.calls,
                                   goal="G10", protocol=PROTOCOL,
                                   task_extra={"class": kind, "seed": entry["seed"], "draw": entry["draw"], "placement_goal": {"region_minimum_m": list(goal.region_minimum_m), "region_maximum_m": list(goal.region_maximum_m),
                                               "dwell_s": goal.dwell_s, "maximum_linear_speed_mps": goal.maximum_linear_speed_mps, "maximum_angular_speed_radps": goal.maximum_angular_speed_radps},
                                               "episode_cap_s": corpus["episode_cap_s"], "retry_budget": corpus["retry_budget"], "sensor_configuration": corpus["sensor_configuration"],
                                               "policy_sha256": corpus["policy_sha256"], "disturbance": corpus["disturbances"].get(kind), "registration_sha256": registration["registration_sha256"]},
                                   outcome_extra={"skill_success": skill_success, "oracle_placement": oracle, "false_completion": row["false_completion"], "attempts": row["attempts"],
                                                  "disturbance_log": row["disturbance"], "events": session.events, "verdict_trail": runtime.verdicts, "within_cap": row["within_cap"]},
                                   observation={"policy": "declared_sensors_with_oracle_labels_kept_apart", "inputs": ["joint encoders", "grasp point through the model", "contact force per opposition group", "front camera by ray visibility"], "vlm": False})
            media = render_bundle(bundle_dir, args.out / body / kind / f"seed-{entry['seed']:03d}" / "media", expected_digest=sealed["sha256"])
            row["bundle"] = sealed
            row["media_sha256"] = media["sha256"]
            row["frames"] = media["frame_count"]
            if skill_success:
                rendered_successes[(body, kind)] = rendered_successes.get((body, kind), 0) + 1
        rows.append(row)
        print(json.dumps({"episode": entry["episode_id"], "verdict": record.verdict.value, "reason": record.root.reason, "physics_s": round(physics_s, 2), "oracle_placed": oracle["placed"],
                          "false_completion": row["false_completion"], "attempts": row["attempts"], "wall": round(wall, 1), "kept": keep}), flush=True)
        (args.out / "trials.json").write_bytes(json_bytes(rows))
    summary = {"goal": "G10", "provenance": provenance, "registration_sha256": registration["registration_sha256"], "finished_at_utc": datetime.now(timezone.utc).isoformat(), "per_body": {}}
    for body in corpus["bodies"]:
        per = {}
        for kind in corpus["classes"]:
            subset = [r for r in rows if r["zoo_id"] == body and r["class"] == kind]
            if not subset:
                continue
            per[kind] = {"episodes": len(subset), "successes": sum(r["skill_success"] for r in subset), "false_completions": sum(r["false_completion"] for r in subset),
                         "within_cap": sum(r["within_cap"] for r in subset), "verdicts": {v: sum(1 for r in subset if r["verdict"] == v) for v in ("success", "failure", "unknown", "interrupted")},
                         "reasons": {}, "physics_s": {"mean": float(np.mean([r["physics_s"] for r in subset])), "max": float(np.max([r["physics_s"] for r in subset]))},
                         "max_attempts": {"place_until_placed": max((r["attempts"]["place_until_placed"] or 0) for r in subset), "acquire_until_held": max((max(r["attempts"]["acquire_until_held"], default=0)) for r in subset)}}
            for r in subset:
                if not r["skill_success"]:
                    per[kind]["reasons"][r["root_reason"] or r["verdict"]] = per[kind]["reasons"].get(r["root_reason"] or r["verdict"], 0) + 1
        summary["per_body"][body] = per
    summary["false_completions"] = sum(r["false_completion"] for r in rows)
    summary["episodes"] = len(rows)
    (args.out / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary["per_body"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
