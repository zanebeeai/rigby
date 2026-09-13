"""Record one prompt on one body as a replayable evidence bundle.

The G01 canonical capture is fixed to the public zoo, the canonical prompt and
the measured rest pose. A composition trial varies all three: the body arrives
already ingested (through whichever intake the campaign registered), the prompt
carries a pace, and the start state may be displaced. What does not vary is the
bundle: the same payload layout as the canonical capture, so ``replay_bundle``
and ``render_bundle`` read a trial exactly as they read the baseline.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_core.evidence import write_bundle
from rigby_core.simulation.recording import PhysicsRecord, PhysicsRecorder, replay_physics

from ..bake.enumerate import build_candidate
from ..bake.runner import _attempt
from ..config import base_tree_fingerprint
from ..gates.certify import GatePolicy, evaluate_gates, simulate
from ..gates.control import ControllerConfig
from ..pipeline import IngestedRobot
from ..planner import OfflineSchemaPlanner
from ..primitives import PrimitiveRecord
from ..run import RunResult, answer
from ..schema.inventory import afforded_entries, load_inventory
from .capture import REPEATS, ROOT, json_bytes, source_provenance


PROTOCOL = "rigby.composition-trial/1"


def run_prompt(
    robot: IngestedRobot,
    prompt: str,
    *,
    start_qpos: np.ndarray | None = None,
    repeats: int = REPEATS,
) -> tuple[list[RunResult], list[PrimitiveRecord], list, object]:
    """Bake the prompt's leaves fresh, then answer the prompt ``repeats`` times.

    Returns the runs, the certified leaf records, the refused leaf bakes and the
    requested schema program (``None`` when planning itself refused).
    """

    inventory = load_inventory()
    model = robot.finalized.model
    manifest = robot.manifest
    if start_qpos is not None:
        manifest = manifest.model_copy(
            update={"rest_qpos": tuple(float(value) for value in np.asarray(start_qpos, dtype=float))}
        )
    planner = OfflineSchemaPlanner(inventory)
    afforded = afforded_entries(inventory, robot.morphology)
    records: list[PrimitiveRecord] = []
    failures: list = []
    try:
        program = planner.plan(prompt, afforded=afforded)
    except Exception:  # noqa: BLE001 - the answer path reports the typed refusal
        program = None
    if program is not None:
        for segment in program.segments:
            entry = inventory.by_binding_key(
                f"{segment.motion_schema.canonical_key}|{segment.figure.role.value}->{segment.ground.role.value}"
            )
            result = _attempt(
                build_candidate(entry, segment.region.remove), manifest, model, inventory,
                policy=None, fingerprint=base_tree_fingerprint().sha256,
            )
            (records if isinstance(result, PrimitiveRecord) else failures).append(result)
    runs = [
        answer(prompt, manifest, model, inventory, tuple(records), tuple(failures), planner=planner)
        for _ in range(repeats)
    ]
    return runs, records, failures, program


def capture_prompt(
    robot: IngestedRobot,
    prompt: str,
    destination: Path,
    *,
    label: str,
    source_urdf: Path,
    start_qpos: np.ndarray | None = None,
    goal: str = "G05",
    protocol_reference: dict | None = None,
    extra_payloads: dict[str, bytes] | None = None,
    caption: str | None = None,
) -> dict:
    """Bake, answer, record and seal one trial. Mirrors the canonical capture.

    ``label`` names the trial in the bundle metadata (the public body id and
    the variant); ``robot`` may carry a hashed structural rig id.
    """

    model, manifest = robot.finalized.model, robot.manifest
    start = np.asarray(manifest.rest_qpos if start_qpos is None else start_qpos, dtype=float)
    runs, records, failures, program = run_prompt(robot, prompt, start_qpos=start)
    run = runs[0]
    discrete_outcomes = [
        {"accepted": r.accepted, "failure_code": r.failure_code, "failure_stage": r.failure_stage,
         "semantic_hash": r.schema_program.role_normalized_hash() if r.schema_program else None}
        for r in runs
    ]
    reference_hashes = [
        PhysicsRecord({name: getattr(r.trajectory, name) for name in ("times_s", "qpos", "qvel", "qacc")}).content_hash()
        if r.trajectory is not None else None for r in runs
    ]
    if len(set(reference_hashes)) != 1:
        raise ValueError("Independent prompt runs produced different reference data")
    started = manifest.model_copy(update={"rest_qpos": tuple(float(v) for v in start)})
    physical, rollouts, replay_reports = [], [], []
    for repeat_run in runs:
        recorder = PhysicsRecorder(model)
        if repeat_run.trajectory is None:
            data = mujoco.MjData(model)
            data.qpos[:] = start
            mujoco.mj_forward(model, data)
            recorder.capture(data, np.zeros(model.nu), control_time_s=0.0)
        else:
            rollouts.append(simulate(
                model, started, repeat_run.trajectory,
                site_name=repeat_run.bound.grounded.figure_sites[0], recorder=recorder,
            ))
        record = recorder.finish()
        physical.append(record)
        replay_reports.append(replay_physics(model, record))
    hashes = [record.content_hash() for record in physical]
    repeated = len(set(hashes)) == 1 and all(d == discrete_outcomes[0] for d in discrete_outcomes)
    violations = list(evaluate_gates(model, started, rollouts[0], GatePolicy())) if rollouts else []
    status = "pre_execution_refusal" if not rollouts else "runtime_failure" if violations else "success"
    if not repeated or not all(r["agrees"] for r in replay_reports):
        raise ValueError("Repeated recordings or saved-control physics replay disagree")
    if rollouts and status == "success" and not run.accepted:
        raise ValueError("Recorded success conflicts with the pipeline result")
    if rollouts:
        for repeat_run, rollout in zip(runs, rollouts, strict=True):
            for name in ("qpos", "qvel", "ctrl", "demand"):
                if not np.array_equal(getattr(rollout, name), getattr(repeat_run.certification.trace, name)):
                    raise ValueError(f"The observer changed the corresponding prompt run's {name}")

    provenance, source_archive = source_provenance()
    xml = robot.finalized.mjcf_xml
    world = {
        "mode": "free_space_composition", "object_count": 0,
        "scope": "No manipulation environment; free-space reach/return composition on one body.",
        "world_geom_ids": np.flatnonzero(model.geom_bodyid == 0).tolist(),
        "gravity": model.opt.gravity.tolist(), "timestep_s": model.opt.timestep,
        "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
        "fault": None,
        "authoritative_complete_geometry_and_physics": "model.mjb and model.xml",
    }
    outcome = {
        "status": status, "scope": "free-space motion; no object task",
        "physical_steps": len(physical[0].arrays["state"]) - 1,
        "violations": [asdict(v) for v in violations],
        "refusal": {"stage": run.failure_stage, "code": run.failure_code, "detail": run.failure_detail} if not rollouts else None,
        "refused_leaf_bakes": [asdict(f) for f in failures],
        "nominal_pipeline_result": run.summary(),
        "fault": None,
        "reference_duration_s": run.duration_s,
        "actual_physics_duration_s": float(physical[0].arrays["time_s"][-1]),
        "region_substitutions": run.bound.substitutions if run.bound else None,
        "leaf_duration_scales": [
            float(b.record.measurements.get("duration_scale", 1.0)) for b in run.bound.bindings if b.record
        ] if run.bound else None,
        "tracking_error_m": float(rollouts[0].tracking_error_m.max()) if rollouts else None,
        "unexpected_contacts": [list(pair) for pair in rollouts[0].unexpected_contacts] if rollouts else None,
    }
    task = {
        "goal": goal, "protocol": PROTOCOL, "prompt": prompt, "label": label,
        "protocol_reference": protocol_reference,
        "requested_semantics": program.model_dump(mode="json") if program is not None else None,
        "bound_semantics": run.bound.program.model_dump(mode="json") if run.bound else None,
        "grounded_program": run.bound.grounded.program.model_dump(mode="json") if run.bound else None,
        "observation_contract": {"policy": "fully_observed_model_based_baseline", "inputs": ["joint encoders", "model parameters", "compiled reference"], "vlm": False},
        "gate_policy": asdict(GatePolicy()), "seed": 0, "model_rng": "none",
        "start_qpos": start.tolist(),
        "start_is_measured_rest": bool(np.array_equal(start, np.asarray(robot.manifest.rest_qpos, dtype=float))),
        "interventions": [],
        "retry_limit": 0, "reference_duration_s": run.duration_s,
        "limits": "Existing manifest joint/actuator limits and free-space gates, including the self-collision gate; no relaxed thresholds.",
        "clock_disclosure": {
            "physics_timestep_s": float(model.opt.timestep),
            "reference_sampled_on_native_physics_time": True,
            "interpretation": "Certification samples the reference at MjData.time; recorded times are physics times.",
        },
    }
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    payloads = {
        "model.mjb": buffer.tobytes(), "model.xml": xml.encode("utf-8"),
        "robot.urdf": source_urdf.read_bytes(), "robot.json": json_bytes(manifest.model_dump(mode="json")),
        "world.json": json_bytes(world), "task.json": json_bytes(task),
        "outcome.json": json_bytes(outcome), "execution.json": json_bytes({
            "pipeline": run.trace.to_json(), "nominal_repeat_outcomes": discrete_outcomes,
            "active_skill_tree": "flat sequence of the requested schema segments",
            "baked_primitives": [r.to_json() for r in records],
            "refused_bakes": [asdict(f) for f in failures],
            "contact_and_sensor_timestamp_semantics": "Initial forward solve, then the previous integration interval; see contact_sample_time_s.",
            "user_input_semantics": "External forces, mocap, equality and userdata schedule; no intermediate qpos/state resets during replay.",
        }),
        "trace.npz": physical[0].to_bytes(),
        "controller.json": json_bytes({"class": "rigby_general.gates.control.ComputedTorqueController", "config": asdict(ControllerConfig())}),
        "reference.json": json_bytes(run.trajectory.payload() if run.trajectory else None),
        "repeats.json": json_bytes({
            "count": REPEATS, "physical_hashes": hashes, "agree": repeated,
            "independent_prompt_runs": True, "one_recording_per_prompt_run": True,
            "reference_data_hashes": reference_hashes,
            "pipeline_trace_hashes": [r.certification.trace.content_hash() if r.certification else None for r in runs],
            "recorded_control_replays": replay_reports, "tolerance": {"absolute": 0.0, "relative": 0.0},
        }),
        "source.json": json_bytes(provenance), "source.zip": source_archive,
        "uv.lock": (ROOT / "uv.lock").read_bytes(),
    }
    payloads.update(extra_payloads or {})
    scale = robot.morphology.scale
    metadata = {
        "goal": goal, "protocol": PROTOCOL, "robot_id": label, "rig_id": manifest.rig_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": status, "fault": False, "simulation_duration_s": float(physical[0].arrays["time_s"][-1]),
        "reference_duration_s": run.duration_s,
        "reference_clock_matches_physics": True,
        "caption": caption or prompt,
        "trace_sha256": hashes[0], "world_sha256": hashlib.sha256(payloads["world.json"]).hexdigest(),
        "camera": {"centre": [scale.workspace_centroid_m.x, scale.workspace_centroid_m.y, scale.workspace_centroid_m.z], "reach": scale.reach_radius_m},
        "task_site": run.bound.grounded.figure_sites[0] if run.bound else None,
        "controller": "computed torque", "observation": "model + joint encoders",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, **metadata, "outcome_record": outcome}
