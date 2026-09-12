"""Run an externally pinned benchmark through the declared observation boundary."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
from rigby_core.hashing import canonical_json_bytes, content_hash, hash_file
from rigby_core.observations import PolicyWorker, PolicyWorkerError

from ..pipeline import ingest_robot
from .evaluator import IndependentEvaluator
from .protocol import load_registration, public_variant
from .sensors import SensorAdapter
from .world import BenchmarkBodyManifestV1, BenchmarkRefusal, compile_world, normalize_world, validate_binding, verify_world

ROOT = Path(__file__).resolve().parents[4]


def run_trial(*, registration: Path, expected_registration_sha256: str, robot_source: Path,
              split_id: str, seed: int, mode: str, policy: str, destination: Path,
              probe_steps: int | None = None) -> dict:
    """One bounded engineering episode; no research score is assigned here.

    No retry is hidden inside this runner. One invocation consumes one attempt;
    future hierarchy executors must debit their retries from the registered budget.
    Confirmatory scoring additionally needs G01 physical replay integration and
    a predeclared trial roster/feasibility map. This runner does not bypass them.
    Raw evaluator state is written only after the acting worker has closed.
    Success at the exact simulation cap is eligible; a step beyond it is not.
    Clock comparisons allow 1e-12 seconds for floating-point accumulation only.
    """
    if probe_steps is not None and (type(probe_steps) is not int or probe_steps <= 0):
        raise ValueError("Probe steps must be a positive integer")
    started = time.monotonic()
    protocol = load_registration(registration, expected_registration_sha256=expected_registration_sha256)
    if mode not in protocol.modes:
        raise BenchmarkRefusal("unregistered_mode", mode)
    if protocol.world.sensor_policy.mode != "declared_sensors":
        raise BenchmarkRefusal("diagnostic_policy_required", "This runner accepts sensor-only workers; fully observed inputs use the explicit diagnostic API")
    authored, variant = public_variant(protocol, split_id=split_id, seed=seed)
    robot = ingest_robot(robot_source)
    body = BenchmarkBodyManifestV1(robot=robot.manifest)
    normalization = None
    world = authored
    if mode == "capability_normalized":
        world, normalization = normalize_world(authored, robot.morphology.scale.reach_radius_m / authored.reference_length_m)
    semantics = {"predicate": "all_objects_placed", "objects": list(authored.goal.object_names), "region": "declared_destination", "require_release": True}
    validate_binding(mode=mode, authored=authored, resolved=world, requested_semantics=semantics,
                     bound_semantics=semantics, normalization=normalization)
    compiled = compile_world(world, body=body, robot_xml=robot.finalized.mjcf_xml, asset_root=robot_source.parent)
    destination.mkdir(parents=True, exist_ok=False)
    data = mujoco.MjData(compiled.model)
    # World object qpos comes first; map each scalar robot joint explicitly.
    for joint_name in robot.manifest.actuator_order:
        joint = mujoco.mj_name2id(compiled.model, mujoco.mjtObj.mjOBJ_JOINT, "robot/"+joint_name)
        source_joint = mujoco.mj_name2id(robot.finalized.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if compiled.model.jnt_type[joint] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise BenchmarkRefusal("unsupported_initialization", "Register multi-coordinate joint initialization explicitly")
        data.qpos[compiled.model.jnt_qposadr[joint]] = robot.manifest.rest_qpos[robot.finalized.model.jnt_qposadr[source_joint]]
    mujoco.mj_forward(compiled.model, data)
    inputs = {"schema": "benchmark.trial-input.v1", "registration_sha256": expected_registration_sha256,
              "protocol_sha256": content_hash(protocol), "split": variant, "mode": mode,
              "normalization": normalization, "requested_semantics": semantics, "bound_semantics": semantics,
              "world_sha256": compiled.world_sha256, "model_sha256": compiled.compiled_model_sha256,
              "body_manifest_sha256": content_hash(body), "source_robot_sha256": hash_file(robot_source),
              "policy": policy, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "attempts": 1, "probe_steps": probe_steps, "scored": False,
              "run_kind": "protocol_probe" if probe_steps is not None else "engineering_episode",
              "feasibility": "unassessed_kept_in_all_attempted_denominator",
              "observation_mode": world.sensor_policy.mode}
    for name, value in (("trial-input.json", inputs), ("body-manifest.json", body), ("world-manifest.json", compiled.manifest)):
        (destination / name).write_bytes(canonical_json_bytes(value))
    (destination / "model.xml").write_text(compiled.xml, encoding="utf-8", newline="\n")
    evaluator = IndependentEvaluator(compiled, expected_world_sha256=compiled.world_sha256)
    rows, actions, observations = [], [], []
    status, failure = "running", None
    steps, sequence, next_observation = 0, 0, 0.0
    last_assessment = None
    try:
        with SensorAdapter(compiled) as sensors, PolicyWorker(
            policy=policy, declaration=sensors.declaration, action_size=compiled.model.nu,
            policy_paths=(ROOT / "core/src", ROOT / "any-robot/src"), timeout_s=min(10.0, protocol.budget.maximum_wall_s),
        ) as worker:
            (destination / "sensor-declaration.json").write_bytes(canonical_json_bytes(sensors.declaration))
            while True:
                if time.monotonic()-started >= protocol.budget.maximum_wall_s:
                    status = "wall_budget_exhausted"
                    break
                if data.time > protocol.budget.maximum_simulation_s + 1e-12:
                    status = "simulation_budget_exhausted"
                    break
                last_assessment = evaluator.evaluate(data)
                rows.append({"time_s": float(data.time), "conditions_met": last_assessment.conditions_met,
                             "dwell_s": last_assessment.dwell_elapsed_s, "root_success": last_assessment.root_success})
                if time.monotonic()-started >= protocol.budget.maximum_wall_s:
                    status = "wall_budget_exhausted"
                    break
                if last_assessment.root_success:
                    status = "root_success"
                    break
                if data.time + compiled.model.opt.timestep > protocol.budget.maximum_simulation_s + 1e-12:
                    status = "simulation_budget_exhausted"
                    break
                if probe_steps is not None and steps >= probe_steps:
                    status = "protocol_probe_complete"
                    break
                if data.time + 1e-9 >= next_observation:
                    packet = sensors.observe(data, sequence=sequence)
                    # Evidence retains sensor hashes; evaluator traces are private
                    # until after this process boundary is closed.
                    observations.append({"sequence": sequence, "time_s": float(data.time), "sha256": packet.content_hash()})
                    if sequence == 0:
                        (destination / "initial-policy-observation.json").write_bytes(canonical_json_bytes(packet))
                    remaining = protocol.budget.maximum_wall_s - (time.monotonic()-started)
                    if remaining <= 0:
                        status = "wall_budget_exhausted"
                        break
                    worker.timeout_s = min(10.0, remaining)
                    command = np.asarray(worker.act(packet).values, dtype=float)
                    limited = compiled.model.actuator_ctrllimited.astype(bool)
                    if np.any(command[limited] < compiled.model.actuator_ctrlrange[limited, 0]) or np.any(command[limited] > compiled.model.actuator_ctrlrange[limited, 1]):
                        raise BenchmarkRefusal("action_limit", "Policy command exceeds a declared actuator range")
                    data.ctrl[:] = command
                    actions.append({"time_s": float(data.time), "values": command.tolist()})
                    sequence += 1
                    next_observation = sequence / world.sensor_policy.frequency_hz
                mujoco.mj_step(compiled.model, data)
                steps += 1
    except (BenchmarkRefusal, PolicyWorkerError) as error:
        status = "execution_refusal"
        failure = {"code": getattr(error, "code", type(error).__name__), "detail": str(error)}
    verify_world(compiled, compiled.world_sha256)
    report = {"schema": "benchmark.trial-result.v1", "inputs_sha256": content_hash(inputs),
              "status": status, "failure": failure, "mode": mode, "world_sha256": compiled.world_sha256,
              "root_success": status == "root_success", "scored": False,
              "run_kind": inputs["run_kind"],
              "attempts": 1, "physical_steps": steps, "elapsed_simulation_s": float(data.time),
              "elapsed_wall_s": time.monotonic()-started, "policy_observations": observations,
              "actions": actions, "evaluations": rows,
              "worker_closed_before_privileged_output": True,
              "physical_capability_claim": "none; unscored infrastructure result"}
    if last_assessment is not None:
        (destination / "final-evaluator-truth.json").write_bytes(canonical_json_bytes(last_assessment.truth))
    (destination / "result.json").write_bytes(canonical_json_bytes(report))
    return report
