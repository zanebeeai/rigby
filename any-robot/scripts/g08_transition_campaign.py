"""Execute the registered G08 transition corpus and record every composition.

Every case is two skills composed through the boundary check: the first
runs, the boundary it leaves is measured and held against the second's
initiation set, a path repair runs and the boundary is verified again
within the budget, and the second runs only if the boundary is compatible.
Every rendered case's replayable bundle is sealed under --local with its
digest in the row; its full-duration video, frame map and summary go under
--out. A feasible case succeeds when the first skill certifies on its own gates,
the boundary is compatible after any repair, the second skill certifies,
and no joint gate fires on a transition or on the second skill. An
injected case is handled correctly when it is rejected before the second
skill runs, or repaired and verified compatible before it runs. Every case
is one continuous physics record; the feasible successes are rendered five
per body and every failure and every injected case in full. No API or
model calls.

    python any-robot/scripts/g08_transition_campaign.py --out docs/results/g08-campaign \
        --local any-robot/results/g08-campaign [--cases a,b] [--limit N]
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
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from rigby_core.evidence import write_bundle
from rigby_core.simulation.recording import PhysicsRecorder, replay_physics

from rigby_general.contact.transfer import collision_policy, transfer_scene_from_environment
from rigby_general.evidence.capture import json_bytes, source_provenance
from rigby_general.evidence.render import render_bundle
from rigby_general.grounding.grounder import _collision_guard, figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.transitions import CompositionRecord, compose, joint_move_skill, transfer_skill


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import g06_transfer_campaign as g06  # noqa: E402

PROTOCOL_DIR = ROOT / "assets/general/research-protocols/g08-transitions-v1"
PROTOCOL = "rigby.transition-composition/1"


def load_corpus() -> tuple[dict, dict, object, object]:
    registration = json.loads((PROTOCOL_DIR / "registration.json").read_bytes())
    g06_dir = ROOT / "assets/general/research-protocols/g06-transfer-v1"
    for name, path in (("corpus.json", PROTOCOL_DIR / "corpus.json"), ("g06/environment.json", g06_dir / "environment.json"), ("g06/goal.json", g06_dir / "goal.json")):
        if registration["files"][name] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise SystemExit(f"{name} does not match its registration; refusing to run")
    corpus = json.loads((PROTOCOL_DIR / "corpus.json").read_bytes())
    env, goal, feasibility, roster, g06_registration = g06.load_registration()
    if g06_registration["registration_sha256"] != corpus["g06_registration_sha256"]:
        raise SystemExit("the G06 fixture no longer matches the one the corpus was registered against")
    return corpus, registration, env, goal


class Body:
    def __init__(self, zoo_id: str, env, goal):
        self.zoo_id = zoo_id
        self.source = REPO / f"any-robot/assets/general/zoo/{zoo_id}/robot.urdf"
        self.robot = ingest_robot(self.source, robot_id=zoo_id)
        self.manifest, self.morphology = self.robot.manifest, self.robot.morphology
        self.effector = self.morphology.grasping_effectors[0]
        chain = next(c for c in self.morphology.chains if c.chain_id == self.effector.chain_id)
        self.frame = build_workspace_frame(self.robot.finalized.model, self.morphology, chain, figure_site=figure_site_for(self.manifest, self.effector.chain_id))
        self.env, self.goal = env, goal
        self.scene = transfer_scene_from_environment(self.manifest, self.robot.mjcf_xml, env, object_name="cube", destination_fixture="platform", goal=goal, asset_root=self.source.parent)
        self.model = self.scene.model
        self.guard = _collision_guard(self.manifest, self.model)

    def skill(self, spec: dict):
        name = spec["skill"]
        if name == "joint_move":
            return joint_move_skill(self.model, self.manifest, self.effector, self.frame, self.guard, skill_id="joint_move", targets=dict(spec["targets"]),
                                    stop_fraction=spec.get("stop_fraction"), keeps_resources=tuple(spec.get("keeps_resources", ())))
        if name == "transfer":
            return transfer_skill(self.model, self.manifest, self.scene, self.effector, self.frame, skill_id="transfer")
        if name == "acquire_carry":
            return transfer_skill(self.model, self.manifest, self.scene, self.effector, self.frame, skill_id="acquire_carry", phase_range=("approach", "carry"))
        if name == "place":
            return transfer_skill(self.model, self.manifest, self.scene, self.effector, self.frame, skill_id="place", phase_range=("lower", "dwell"))
        if name == "return_home":
            from rigby_general.contact.grasp import _scene_rest_qpos
            from rigby_general.transitions import arm_joint_names

            arm = arm_joint_names(self.model, self.effector, self.frame)
            rest = _scene_rest_qpos(self.model, self.manifest)
            dofs = {d.joint: d for d in self.manifest.dofs}
            targets = {dofs[n].name: float(rest[int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])]) for n in arm}
            return joint_move_skill(self.model, self.manifest, self.effector, self.frame, self.guard, skill_id="return_home", targets=targets)
        raise ValueError(f"unknown skill {name!r}")


def run_case(body: Body, case: dict, *, validate: bool = True, record: bool = True) -> tuple[CompositionRecord, PhysicsRecorder | None, float]:
    recorder = PhysicsRecorder(body.model) if record else None
    started = time.perf_counter()
    result = compose(body.model, body.manifest, body.effector, body.frame, body.skill(case["first"]), body.skill(case["second"]), recorder, guard=body.guard,
                     validate=validate, max_repairs=2, belief_age_s=float(case.get("belief_age_s", 0.0)))
    return result, recorder, time.perf_counter() - started


def handled_correctly(case: dict, result: CompositionRecord) -> tuple[bool, str]:
    """Whether an injected case was rejected before the second skill, or
    repaired and verified compatible before it; a feasible case succeeds
    as the corpus defines success."""

    if case["kind"] == "feasible":
        return result.composed_success, "composed_success" if result.composed_success else "composed_failure"
    if result.rejected and result.second is None:
        return True, f"rejected:{result.rejection}"
    if result.repairs and result.re_verified and result.compatible_before_second:
        return True, "repaired_then_reverified:" + ",".join(r.kind.value for r in result.repairs)
    if not result.verdicts:
        return False, "first_skill_did_not_run"
    if result.verdicts[0].compatible:
        return False, "boundary_was_compatible_nothing_to_inject"
    return False, "second_ran_without_a_repaired_boundary"


def status_of(case: dict, result: CompositionRecord) -> str:
    if result.composed_success:
        return "success"
    if result.rejected and result.second is None:
        return "rejected"
    return "runtime_failure"


def seal(destination: Path, *, body: Body, case: dict, result: CompositionRecord, recorder: PhysicsRecorder | None, caption: str, label: str, corpus_sha256: str) -> dict:
    model = body.model
    executed = recorder is not None and bool(recorder.rows["time_s"])
    if executed:
        physical = recorder.finish()
        replay = {"recorded_controls": replay_physics(model, physical)}
        replay["agrees"] = bool(replay["recorded_controls"]["agrees"])
    else:
        data = mujoco.MjData(model)
        from rigby_general.contact.grasp import _scene_rest_qpos

        data.qpos[:] = _scene_rest_qpos(model, body.manifest)
        mujoco.mj_forward(model, data)
        rec = PhysicsRecorder(model)
        rec.capture(data, np.zeros(model.nu), control_time_s=0.0)
        physical = rec.finish()
        replay = {"recorded_controls": {"agrees": True, "note": "no motion was executed"}, "agrees": True}
    status = status_of(case, result) if executed else "pre_execution_refusal"
    provenance, archive = source_provenance()
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    summary = result.summary()
    ok, handling = handled_correctly(case, result)
    outcome = {"status": status, "scope": "two certified skills composed through the boundary check; repairs on physics; the second skill only after a compatible boundary",
               "case_id": case["case_id"], "kind": case["kind"], "expect": case["expect"], "handled_correctly": ok, "handling": handling, **summary,
               "actual_physics_duration_s": float(physical.arrays["time_s"][-1] - physical.arrays["time_s"][0]) if executed else 0.0,
               "physical_steps": (len(physical.arrays["state"]) - 1) if executed else 0,
               "refusal": None if executed else {"stage": "path", "code": result.rejection or "not_executed", "detail": result.first.detail}}
    task = {"goal": "G08", "protocol": PROTOCOL, "label": label, "case": case, "corpus_sha256": corpus_sha256, "validated": result.validated,
            "environment": body.env.model_dump(mode="json"),
            "observation_contract": {"policy": "fully_observed_model_based_baseline", "inputs": ["joint encoders", "model parameters", "contact forces on the gripper"], "vlm": False},
            "interventions": [], "retry_limit": 0, "attempts": 1, "repair_budget": 2,
            "limits": "Manifest joint/actuator limits; initiation margins 2 percent of range and 5 percent of velocity limit; belief age 5 s; penetration 4 mm; no relaxed thresholds.",
            "clock_disclosure": {"physics_timestep_s": float(model.opt.timestep), "phase_timing_on_native_physics_time": True}}
    payloads = {
        "model.mjb": buffer.tobytes(), "model.xml": body.scene.scene.xml.encode("utf-8"),
        "robot.urdf": body.source.read_bytes(), "robot.json": json_bytes(body.manifest.model_dump(mode="json")),
        "world.json": json_bytes({"mode": "strict_fixed_world", "environment_id": body.env.environment_id, "object_count": len(body.env.objects), "timestep_s": model.opt.timestep,
                                  "gravity": model.opt.gravity.tolist(), "collision_policy": collision_policy(model), "fault": None}),
        "task.json": json_bytes(task), "outcome.json": json_bytes(outcome),
        "execution.json": json_bytes({"composition": summary, "active_skill_tree": f"Sequence({case['first']['skill']}, [boundary check, repairs], {case['second']['skill']})"}),
        "trace.npz": physical.to_bytes(),
        "controller.json": json_bytes({"composition": "rigby_general.transitions.compose", "arm": "rigby_general.gates.control.ComputedTorqueController", "closure": "rigby_general.contact.closure.ClosureController"}),
        "repeats.json": json_bytes({"count": 1, "recorded_control_replay": replay, "note": "one run of the composition; the recorded controls replay to the recorded states across every boundary"}),
        "source.json": json_bytes(provenance), "source.zip": archive,
    }
    scale = body.morphology.scale
    centre = [float(v) for v in (np.asarray(body.env.objects[0].position_m) + np.asarray(body.env.fixtures[1].position_m)) / 2.0]
    metadata = {"goal": "G08", "protocol": PROTOCOL, "robot_id": label, "rig_id": body.manifest.rig_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "outcome": status, "fault": False,
                "simulation_duration_s": float(physical.arrays["time_s"][-1] - physical.arrays["time_s"][0]) if executed else 0.0,
                "reference_duration_s": float(physical.arrays["time_s"][-1] - physical.arrays["time_s"][0]) if executed else 0.0,
                "reference_clock_matches_physics": True, "caption": caption, "trace_sha256": physical.content_hash(),
                "world_sha256": hashlib.sha256(payloads["world.json"]).hexdigest(),
                "camera": {"centre": centre, "reach": max(0.35, 0.45 * scale.reach_radius_m)}, "task_site": "scene_block_center",
                "controller": "composition through the boundary check over computed torque + contact-driven closure", "observation": "model + joint encoders + gripper contact"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, "outcome": status, "trace_sha256": metadata["trace_sha256"],
            "simulation_duration_s": metadata["simulation_duration_s"], "replay_agrees": replay["agrees"], "executed": executed}


def save_trace(local: Path, recorder: PhysicsRecorder | None) -> str | None:
    if recorder is None or not recorder.rows["time_s"]:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(recorder.finish().to_bytes())
    return hashlib.sha256(local.read_bytes()).hexdigest()


def row_for(case: dict, result: CompositionRecord, wall: float) -> dict:
    ok, handling = handled_correctly(case, result)
    return {"case_id": case["case_id"], "zoo_id": case["zoo_id"], "kind": case["kind"], "expect": case["expect"], "handled_correctly": ok, "handling": handling,
            "status": status_of(case, result), "wall_seconds": wall, **result.summary()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--cases", help="comma-separated case ids to run (engineering smoke; never a scored run)")
    parser.add_argument("--limit", type=int, help="run only the first N feasible and N injected cases (engineering smoke; never a scored run)")
    parser.add_argument("--render-successes", type=int, default=5)
    args = parser.parse_args()
    corpus, registration, env, goal = load_corpus()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a campaign writes only to a fresh destination")
    args.out.mkdir(parents=True)
    args.local.mkdir(parents=True, exist_ok=True)
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                  "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__,
                  "registration_sha256": registration["registration_sha256"], "corpus_sha256": registration["files"]["corpus.json"],
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "scored": args.cases is None and args.limit is None, "generation_calls": 0}
    (args.out / "provenance.json").write_bytes(json_bytes(provenance))
    wanted = set(args.cases.split(",")) if args.cases else None
    feasible = corpus["feasible_cases"][: args.limit] if args.limit else corpus["feasible_cases"]
    injected = corpus["injected_cases"][: args.limit] if args.limit else corpus["injected_cases"]
    bodies: dict[str, Body] = {}
    rows = {"feasible": [], "injected": []}
    rendered_successes: dict[str, int] = {}
    for group, cases in (("feasible", feasible), ("injected", injected)):
        for case in cases:
            if wanted and case["case_id"] not in wanted:
                continue
            body = bodies.setdefault(case["zoo_id"], Body(case["zoo_id"], env, goal))
            result, recorder, wall = run_case(body, case)
            row = row_for(case, result, wall)
            row["trace_file_sha256"] = save_trace(args.local / group / f"{case['case_id']}.npz", recorder)
            keep = group == "injected" or not result.composed_success or rendered_successes.get(case["zoo_id"], 0) < args.render_successes
            if keep:
                # A composition's record runs to several megabytes; every
                # sealed bundle lives in the local results tree with its digest
                # in the row, and its full-duration video, frame map and summary
                # go under --out beside the rows.
                bundle_dir = args.local / group / case["case_id"] / "physical"
                sealed = seal(bundle_dir, body=body, case=case, result=result, recorder=recorder, label=f"{case['case_id']}", corpus_sha256=registration["files"]["corpus.json"],
                              caption=f"{case['zoo_id']} | {case['first']['skill']} then {case['second']['skill']} | {case['kind']} | {row['handling']}")
                media = render_bundle(bundle_dir, args.out / group / case["case_id"] / "media", expected_digest=sealed["sha256"])
                row["bundle"] = sealed
                row["media_sha256"] = media["sha256"]
                if result.composed_success and group == "feasible":
                    rendered_successes[case["zoo_id"]] = rendered_successes.get(case["zoo_id"], 0) + 1
            rows[group].append(row)
            print(json.dumps({"case": case["case_id"], "kind": case["kind"], "handled": row["handled_correctly"], "handling": row["handling"], "status": row["status"],
                              "repairs": [r["kind"] for r in row["repairs"]], "wall": round(wall, 1)}), flush=True)
    (args.out / "trials.json").write_bytes(json_bytes(rows))
    per_body = {}
    for row in rows["feasible"]:
        entry = per_body.setdefault(row["zoo_id"], {"feasible": 0, "successes": 0, "failure_taxonomy": {}})
        entry["feasible"] += 1
        if row["handled_correctly"]:
            entry["successes"] += 1
        else:
            code = (row["second"]["gate"] if row["second"] else None) or row["rejection"] or (row["gate_violations"][0]["code"] if row["gate_violations"] else None) or row["first"]["gate"] or "unknown"
            entry["failure_taxonomy"][code] = entry["failure_taxonomy"].get(code, 0) + 1
    injected_summary = {}
    for row in rows["injected"]:
        entry = injected_summary.setdefault(row["kind"], {"count": 0, "handled_correctly": 0, "handling": {}})
        entry["count"] += 1
        entry["handled_correctly"] += int(row["handled_correctly"])
        key = row["handling"].split(":")[0]
        entry["handling"][key] = entry["handling"].get(key, 0) + 1
    costs = [row["cost"] for row in rows["feasible"] + rows["injected"] if row["repairs"]]
    summary = {"goal": "G08", "provenance": provenance, "registration_sha256": registration["registration_sha256"], "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "feasible": {"count": len(rows["feasible"]), "successes": sum(1 for r in rows["feasible"] if r["handled_correctly"]), "per_body": per_body},
               "injected": {"count": len(rows["injected"]), "handled_correctly": sum(1 for r in rows["injected"] if r["handled_correctly"]), "per_kind": injected_summary},
               "transition_cost": {"repaired_cases": len(costs), "physics_s_mean": float(np.mean([c["physics_s"] for c in costs])) if costs else 0.0,
                                   "physics_s_max": float(max((c["physics_s"] for c in costs), default=0.0)), "joint_travel_rad_mean": float(np.mean([c["joint_travel_rad"] for c in costs])) if costs else 0.0}}
    (args.out / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps({k: summary[k] for k in ("feasible", "injected", "transition_cost")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
