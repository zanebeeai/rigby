"""Record the canonical public-zoo probe without borrowing stored libraries."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_core.evidence import verify_bundle, write_bundle
from rigby_core.motion.trajectory import CandidateTrajectoryV1, ContactPlateauV1
from rigby_core.simulation.recording import PhysicsRecord, PhysicsRecorder, replay_physics

from ..bake.enumerate import build_candidate
from ..bake.runner import _attempt
from ..config import base_tree_fingerprint
from ..contracts import RobotAssetManifestV1
from ..gates.certify import GatePolicy, evaluate_gates, simulate
from ..gates.control import ComputedTorqueController, ControllerConfig
from ..pipeline import ingest_robot
from ..planner import OfflineSchemaPlanner
from ..primitives import PrimitiveRecord
from ..run import answer
from ..schema.inventory import INVENTORY_PATH, afforded_entries, load_inventory


PROMPT = "reach out as far as you can and then come back"
ROOT = Path(__file__).resolve().parents[4]
REPEATS = 3
PROTOCOL = "rigby.baseline-replay/1"


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def source_provenance() -> tuple[dict, bytes]:
    """Save relevant package sources, including untracked development modules.

    No environment files, sealed evaluation assets or arbitrary working-tree
    files are collected. The archive is provenance, not executable input to the
    replay command, which uses the installed, explicitly versioned controller.
    """
    paths = sorted([
        *ROOT.joinpath("core/src").rglob("*.py"),
        *ROOT.joinpath("any-robot/src").rglob("*.py"),
        ROOT / "core/pyproject.toml", ROOT / "any-robot/pyproject.toml",
        ROOT / "pyproject.toml", INVENTORY_PATH,
    ])
    hashes = {}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            name = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "tracked_patch_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)).hexdigest(),
        "package_snapshot_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "package_source_sha256": hashes,
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "scipy", "pydantic", "pillow")},
        "generation_calls": 0,
    }, stream.getvalue()


def _trajectory(payload: dict) -> CandidateTrajectoryV1:
    value = dict(payload)
    value["contact_plateaus"] = tuple(ContactPlateauV1(**p) for p in value["contact_plateaus"])
    value["quaternion_qpos_adrs"] = tuple(value["quaternion_qpos_adrs"])
    return CandidateTrajectoryV1(**value)


def capture_canonical(robot_id: str, destination: Path, *, fault: bool = False) -> dict:
    source = ROOT / "any-robot/assets/general/zoo" / robot_id / "robot.urdf"
    # Names must designate a member of the public development zoo, never a path.
    if source not in sorted((ROOT / "any-robot/assets/general/zoo").glob("*/robot.urdf")):
        raise ValueError("Choose a public zoo robot ID")
    inventory = load_inventory()
    robot = ingest_robot(source, robot_id=robot_id)
    model, manifest = robot.finalized.model, robot.manifest
    planner = OfflineSchemaPlanner(inventory)
    program = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
    records, failures = [], []
    for segment in program.segments:
        entry = inventory.by_binding_key(
            f"{segment.motion_schema.canonical_key}|{segment.figure.role.value}->{segment.ground.role.value}"
        )
        result = _attempt(
            build_candidate(entry, segment.region.remove), manifest, model, inventory,
            policy=None, fingerprint=base_tree_fingerprint().sha256,
        )
        (records if isinstance(result, PrimitiveRecord) else failures).append(result)
    runs = [answer(PROMPT, manifest, model, inventory, tuple(records), tuple(failures), planner=planner) for _ in range(REPEATS)]
    run = runs[0]
    discrete_outcomes = [
        {"accepted": r.accepted, "failure_code": r.failure_code, "failure_stage": r.failure_stage,
         "semantic_hash": r.schema_program.role_normalized_hash() if r.schema_program else None}
        for r in runs
    ]
    xml = robot.finalized.mjcf_xml
    if fault:
        if run.trajectory is None:
            raise ValueError("A runtime fault example needs a compiled nominal trajectory")
        tree = ET.fromstring(xml)
        for actuator in tree.find("actuator"):
            actuator.set("gainprm", "0")
        xml = ET.tostring(tree, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml)

    reference_hashes = [
        PhysicsRecord({name: getattr(r.trajectory, name) for name in ("times_s", "qpos", "qvel", "qacc")}).content_hash()
        if r.trajectory is not None else None for r in runs
    ]
    if len(set(reference_hashes)) != 1:
        raise ValueError("Independent prompt runs produced different reference data")
    physical, rollouts, replay_reports = [], [], []
    for repeat_run in runs:
        recorder = PhysicsRecorder(model)
        if repeat_run.trajectory is None:
            data = mujoco.MjData(model)
            data.qpos[:] = manifest.rest_qpos
            mujoco.mj_forward(model, data)
            recorder.capture(data, np.zeros(model.nu), control_time_s=0.0)
        else:
            rollouts.append(simulate(
                model, manifest, repeat_run.trajectory,
                site_name=repeat_run.bound.grounded.figure_sites[0], recorder=recorder,
            ))
        record = recorder.finish()
        physical.append(record)
        replay_reports.append(replay_physics(model, record))
    hashes = [record.content_hash() for record in physical]
    repeated = len(set(hashes)) == 1 and all(d == discrete_outcomes[0] for d in discrete_outcomes)
    violations = list(evaluate_gates(model, manifest, rollouts[0], GatePolicy())) if rollouts else []
    status = "pre_execution_refusal" if not rollouts else "runtime_failure" if violations else "success"
    if not repeated or not all(r["agrees"] for r in replay_reports):
        raise ValueError("Repeated recordings or saved-control physics replay disagree")
    if fault and status != "runtime_failure":
        raise ValueError("The declared actuation-loss diagnostic did not fail a physical gate")
    if not fault and rollouts:
        if status == "success" and not run.accepted:
            raise ValueError("Recorded success conflicts with the pipeline result")
        for repeat_run, rollout in zip(runs, rollouts, strict=True):
            for name in ("qpos", "qvel", "ctrl", "demand"):
                if not np.array_equal(getattr(rollout, name), getattr(repeat_run.certification.trace, name)):
                    raise ValueError(f"The observer changed the corresponding prompt run's {name}")

    provenance, source_archive = source_provenance()
    world = {
        "mode": "free_space_baseline", "object_count": 0,
        "scope": "No manipulation environment; this is not G02 fixed-world transfer.",
        "world_geom_ids": np.flatnonzero(model.geom_bodyid == 0).tolist(),
        "gravity": model.opt.gravity.tolist(), "timestep_s": model.opt.timestep,
        "integrator": int(model.opt.integrator), "solver": int(model.opt.solver),
        "fault": "zero_actuator_gain" if fault else None,
        "authoritative_complete_geometry_and_physics": "model.mjb and model.xml",
    }
    outcome = {
        "status": status, "scope": "canonical free-space motion; no object task",
        "physical_steps": len(physical[0].arrays["state"]) - 1,
        "violations": [asdict(v) for v in violations],
        "refusal": {"stage": run.failure_stage, "code": run.failure_code, "detail": run.failure_detail} if not rollouts else None,
        "nominal_pipeline_result": run.summary(),
        "fault": world["fault"],
        "reference_duration_s": run.duration_s,
        "actual_physics_duration_s": float(physical[0].arrays["time_s"][-1]),
    }
    task = {
        "goal": "G01", "protocol": PROTOCOL, "prompt": PROMPT,
        "requested_semantics": program.model_dump(mode="json"),
        "bound_semantics": run.bound.program.model_dump(mode="json") if run.bound else None,
        "grounded_program": run.bound.grounded.program.model_dump(mode="json") if run.bound else None,
        "observation_contract": {"policy": "fully_observed_model_based_baseline", "inputs": ["joint encoders", "model parameters", "compiled reference"], "vlm": False},
        "gate_policy": asdict(GatePolicy()), "seed": 0, "model_rng": "none",
        "interventions": ["Actuator gains set to zero before simulation (declared diagnostic)"] if fault else [],
        "retry_limit": 0, "reference_duration_s": run.duration_s,
        "limits": "Existing manifest joint/actuator limits and free-space gates; no relaxed thresholds.",
        "clock_disclosure": {
            "reference_sample_interval_s": 1.0 / 240,
            "physics_timestep_s": float(model.opt.timestep),
            "matching_clocks": 1.0 / 240 == float(model.opt.timestep),
            "interpretation": "The existing loop advances its reference at 240 Hz independently of mj_step's timestep. Evidence preserves this behavior and renders actual physics time. Timing fidelity is not established by passing the existing gates.",
        },
    }
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    payloads = {
        "model.mjb": buffer.tobytes(), "model.xml": xml.encode("utf-8"),
        "robot.urdf": source.read_bytes(), "robot.json": json_bytes(manifest.model_dump(mode="json")),
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
    scale = robot.morphology.scale
    metadata = {
        "goal": "G01", "protocol": PROTOCOL, "robot_id": robot_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": status, "fault": fault, "simulation_duration_s": float(physical[0].arrays["time_s"][-1]),
        "reference_duration_s": run.duration_s,
        "reference_clock_matches_physics": 1.0 / 240 == float(model.opt.timestep),
        "trace_sha256": hashes[0], "world_sha256": hashlib.sha256(payloads["world.json"]).hexdigest(),
        "camera": {"centre": [scale.workspace_centroid_m.x, scale.workspace_centroid_m.y, scale.workspace_centroid_m.z], "reach": scale.reach_radius_m},
        "task_site": run.bound.grounded.figure_sites[0] if run.bound else None,
        "controller": "computed torque", "observation": "model + joint encoders",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, **metadata}


def replay_bundle(root: Path, expected_digest: str | None = None) -> dict:
    bundle = verify_bundle(root, expected_digest)
    source = json.loads((root / "source.json").read_bytes())
    if source["dependencies"]["mujoco"] != mujoco.__version__ or source["machine"] != platform.machine() or source["platform"] != platform.platform():
        raise ValueError("Exact replay requires the recorded MuJoCo version, architecture and platform")
    for name in ("numpy", "scipy", "pydantic"):
        if importlib.metadata.version(name) != source["dependencies"][name]:
            raise ValueError(f"Exact replay requires the recorded {name} version")
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    record = PhysicsRecord.from_bytes((root / "trace.npz").read_bytes())
    if record.content_hash() != bundle["metadata"]["trace_sha256"]:
        raise ValueError("Recorded array content hash does not match its metadata")
    result = {"recorded_controls": replay_physics(model, record)}
    reference = json.loads((root / "reference.json").read_bytes())
    if reference is not None:
        trajectory = _trajectory(reference)
        controller = ComputedTorqueController(model, ControllerConfig(**json.loads((root / "controller.json").read_bytes())["config"]))
        result["recomputed_controller"] = replay_physics(
            model, record, controller=lambda data, i: controller.compute(data, trajectory.sample(record.arrays["control_time_s"][i])),
        )
    else:
        result["controller_replay"] = "not applicable: no execution occurred"
    result["agrees"] = all(v["agrees"] for v in result.values() if isinstance(v, dict))
    result["task_outcome"] = json.loads((root / "outcome.json").read_bytes())["status"]
    return result
