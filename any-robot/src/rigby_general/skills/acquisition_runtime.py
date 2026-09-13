"""Acquisition bound to bodies: the frozen problems' worlds, single-shot evaluation, the search loop, the held-out test.

The neutral search (``rigby_core.skills.acquisition``) proposes parameter
vectors of the contact transfer family -- how the closure advances, detects
contact and squeezes, how the arm tracks, how much slower than the declared
joint speeds the moving phases run -- and this module runs them. Every
evaluation is the G06 transfer primitive, single-shot, every hard gate on:
no retry, no observation loop, nothing that could hide a parameter's
quality behind a second attempt. The development draws are the only
episodes the search sees; the held-out draws are run once, with the vector
the search settled on, and never inform it.

A world is the registered G10 world with the change the problem declares:
the cube at half its registered friction, or the cube grown and made
heavier, or nothing changed at all for a body the fixed world defeated.
Sealing follows the G06 layout for a single transfer, so a held-out trial
replays and renders like any other evidence.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rigby_core.evidence import write_bundle
from rigby_core.simulation.recording import replay_physics
from rigby_core.skills import (
    AcquisitionOutcomeV1,
    AcquisitionProblemV1,
    AcquisitionStatus,
    AttemptV1,
    EpisodeOutcomeV1,
    EvolutionSearch,
    ProblemKind,
    attempts_digest,
    classify,
)

from ..contact.closure import ClosureConfig
from ..contact.placement import PlacementGoal
from ..contact.transfer import TransferResult, attempt_transfer, collision_policy
from ..evidence.capture import json_bytes, source_provenance
from ..gates.control import ControllerConfig
from ..scenes.environment import EnvironmentV1
from ..sensing import load_policy
from .skill_store import goal_for, perturbed
from .transfer_object import TransferObjectSession


PROTOCOL = "rigby.acquisition-trial/1"


# -- parameters ------------------------------------------------------------------------


def configs_of(parameters: dict[str, float]) -> tuple[ControllerConfig, ClosureConfig, float]:
    """A parameter vector into the three things the transfer takes."""

    arm = {k[len("arm."):]: float(v) for k, v in parameters.items() if k.startswith("arm.")}
    closure = {k[len("closure."):]: float(v) for k, v in parameters.items() if k.startswith("closure.")}
    controller = ControllerConfig(**arm) if arm else ControllerConfig()
    return controller, ClosureConfig(**closure) if closure else ClosureConfig(), float(parameters.get("duration_scale", 1.0))


# -- worlds -----------------------------------------------------------------------------------


def problem_world(problem: AcquisitionProblemV1, base: EnvironmentV1, changes: dict[str, Any]) -> EnvironmentV1:
    """The registered world with the problem's declared change applied."""

    cube = base.objects[0]
    change = changes.get(problem.problem_id, {})
    updates: dict[str, Any] = {}
    if "friction_scale" in change:
        updates["friction"] = round(cube.friction * float(change["friction_scale"]), 6)
    if "object_half_size_m" in change:
        half = float(change["object_half_size_m"])
        updates["size_m"] = (half, half, half)
        updates["position_m"] = (cube.position_m[0], cube.position_m[1], cube.position_m[2] + (half - cube.size_m[2]))
        updates["mass_kg"] = round(cube.mass_kg * (half / cube.size_m[2]) ** 3 * float(change.get("mass_scale", 1.0)), 9)
    elif "mass_scale" in change:
        updates["mass_kg"] = round(cube.mass_kg * float(change["mass_scale"]), 9)
    if not updates:
        return base
    return base.model_copy(update={"environment_id": f"{base.environment_id}+{problem.problem_id}", "objects": (cube.model_copy(update=updates),)})


# -- one single-shot episode -----------------------------------------------------------------------


def run_single_shot(body: str, source: Path, environment: EnvironmentV1, goal: PlacementGoal, policy: dict, parameters: dict[str, float], *, seed_label: str, record: bool = False):
    """The G06 transfer primitive once, every gate on, under the given parameters."""

    controller, closure, duration_scale = configs_of(parameters)
    session = TransferObjectSession.open(body, source, environment, goal, policy, seed_label=seed_label, controller_config=controller, closure_config=closure, duration_scale=duration_scale)
    started = time.perf_counter()
    result = attempt_transfer(session.robot.manifest, session.scene, session.effector, session.frame, recorder=session.recorder if record else None,
                              controller_config=controller, closure_config=closure, duration_scale=duration_scale)
    wall = time.perf_counter() - started
    return session, result, wall


def outcome_of(seed: int, result: TransferResult, wall: float) -> EpisodeOutcomeV1:
    physics = float(result.times_s[-1] - result.times_s[0]) if result.executed and len(result.times_s) else 0.0
    return EpisodeOutcomeV1(seed=seed, certified=bool(result.certified), failed_gate=result.failed_gate, physics_s=physics, wall_s=float(wall),
                            measurements={"peak_force_n": float(result.peak_force_n), "max_penetration_m": float(result.max_penetration_m), "lift_height_m": float(result.lift_height_m),
                                          "hold_s": float(result.hold_s), "placement_dwell_s": float(result.placement_dwell_s)})


def evaluate(problem: AcquisitionProblemV1, source: Path, environment: EnvironmentV1, goal: PlacementGoal, policy: dict, parameters: dict[str, float], draws: list[dict], *, label: str,
             stop_on_failure: bool = True) -> list[EpisodeOutcomeV1]:
    outcomes = []
    for draw in draws:
        world = perturbed(environment, draw)
        _, result, wall = run_single_shot(problem.body, source, world, goal, policy, parameters, seed_label=f"{label}-{draw['seed']}")
        outcomes.append(outcome_of(int(draw["seed"]), result, wall))
        if stop_on_failure and not result.certified:
            break
    return outcomes


# -- the search loop ----------------------------------------------------------------------------------


def acquire(problem: AcquisitionProblemV1, source: Path, environment: EnvironmentV1, goal: PlacementGoal, policy: dict, draws_by_seed: dict[int, dict], *, search_seed: int,
            on_attempt=None) -> tuple[list[AttemptV1], AttemptV1 | None, EvolutionSearch, dict]:
    """Propose, run the development draws, observe; confirm a full pass; stop at the ceiling."""

    search = EvolutionSearch(problem, seed=search_seed)
    development = [draws_by_seed[s] for s in problem.development_seeds]
    confirmation = [draws_by_seed[s] for s in problem.confirmation_seeds]
    attempts: list[AttemptV1] = []
    best: AttemptV1 | None = None
    physics_total = 0.0
    wall_total = 0.0
    ceiling_hit = None
    started = time.perf_counter()
    for index in range(problem.ceiling.attempts):
        # The ceiling's minutes are simulator-worker minutes: the wall time the
        # worker spent in physics, not the simulated seconds it produced.
        if wall_total / 60.0 >= problem.ceiling.worker_minutes:
            ceiling_hit = "worker_minutes"
            break
        parameters = search.propose()
        episodes = evaluate(problem, source, environment, goal, policy, parameters, development, label=f"{problem.problem_id}-a{index:03d}")
        certified = sum(1 for e in episodes if e.certified)
        confirm: list[EpisodeOutcomeV1] = []
        if certified == len(development) and confirmation:
            confirm = evaluate(problem, source, environment, goal, policy, parameters, confirmation, label=f"{problem.problem_id}-a{index:03d}-confirm")
        physics = sum(e.physics_s for e in episodes) + sum(e.physics_s for e in confirm)
        wall = sum(e.wall_s for e in episodes) + sum(e.wall_s for e in confirm)
        physics_total += physics
        wall_total += wall
        attempt = AttemptV1(index=index, parameters=parameters, episodes=tuple(episodes), certified=certified, of=len(development), confirmation=tuple(confirm), sigma=search.sigma,
                            physics_s=physics, wall_s=wall)
        improved = search.observe(parameters, attempt.score)
        attempt = attempt.model_copy(update={"improved": improved})
        attempts.append(attempt)
        if best is None or attempt.score > best.score:
            best = attempt
        if on_attempt is not None:
            on_attempt(attempt, wall_total)
        if certified == len(development) and confirm and all(e.certified for e in confirm):
            break
    else:
        ceiling_hit = "attempts"
    if ceiling_hit is None and not (best is not None and best.confirmation and all(e.certified for e in best.confirmation)):
        ceiling_hit = "worker_minutes"
    budget = {"attempts": len(attempts), "physics_minutes": physics_total / 60.0, "worker_minutes": wall_total / 60.0, "wall_minutes": (time.perf_counter() - started) / 60.0, "ceiling_hit": ceiling_hit,
              "ceiling": {"attempts": problem.ceiling.attempts, "worker_minutes": problem.ceiling.worker_minutes}}
    return attempts, best, search, budget


def summarize(problem: AcquisitionProblemV1, attempts: list[AttemptV1], best: AttemptV1 | None, search: EvolutionSearch, budget: dict, *, accepted: bool, limiting: str = "") -> AcquisitionOutcomeV1:
    confirmed = best if (best is not None and best.confirmation and all(e.certified for e in best.confirmation) and best.certified == best.of) else None
    status, kind, changed = classify(problem, confirmed, accepted=accepted)
    return AcquisitionOutcomeV1(problem_id=problem.problem_id, status=status, kind=kind, attempts=len(attempts), physics_minutes=float(budget["physics_minutes"]), wall_minutes=float(budget["wall_minutes"]),
                                ceiling_hit=budget["ceiling_hit"] if confirmed is None else None, best_attempt=None if best is None else best.index, best_parameters=dict(best.parameters) if best else {},
                                parameters_changed=changed, limiting_capability=limiting if confirmed is None else "", provenance=search.provenance(), attempts_sha256=attempts_digest(attempts))


# -- sealing a single-shot trial -----------------------------------------------------------------------


def seal_single_shot(destination: Path, *, session: TransferObjectSession, result: TransferResult, label: str, caption: str, parameters: dict[str, float], task_extra: dict | None = None) -> dict:
    """A replayable bundle in the G06 single-transfer layout, goal G12."""

    model = session.model
    recorder = session.recorder
    if recorder.rows["time_s"]:
        record = recorder.finish()
        status = "success" if result.certified else "runtime_failure"
        replay = {"recorded_controls": replay_physics(model, record)}
        replay["agrees"] = bool(replay["recorded_controls"]["agrees"])
    else:
        from rigby_core.simulation.recording import PhysicsRecorder

        data = mujoco.MjData(model)
        rec = PhysicsRecorder(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        rec.capture(data, np.zeros(model.nu), control_time_s=0.0)
        record = rec.finish()
        status = "pre_execution_refusal"
        replay = {"recorded_controls": {"agrees": True, "note": "no motion was executed"}, "agrees": True}
    provenance, archive = source_provenance()
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    controller, closure, duration_scale = configs_of(parameters)
    outcome = {"status": status, "scope": "one contact transfer, every hard gate on, under searched parameters; independent placement evaluator",
               "certified": bool(result.certified), "failed_gate": result.failed_gate, "phases": [p.name for p in result.phases], "lift_height_m": float(result.lift_height_m), "hold_s": float(result.hold_s),
               "placement_dwell_s": float(result.placement_dwell_s), "max_penetration_m": float(result.max_penetration_m), "peak_force_n": float(result.peak_force_n), "path_seed": result.path_seed,
               "violations": [{"code": v.code, "detail": v.detail[:300], "measured": float(v.measured), "limit": float(v.limit)} for v in result.violations],
               "refusal": {"stage": "path", "code": result.failed_gate, "detail": result.violations[0].detail} if status == "pre_execution_refusal" and result.violations else None,
               "actual_physics_duration_s": float(record.arrays["time_s"][-1]), "physical_steps": len(record.arrays["state"]) - 1}
    environment = session.environment
    task = {"goal": "G12", "protocol": PROTOCOL, "label": label, "environment": environment.model_dump(mode="json"), "placement_goal": asdict(session.goal),
            "environment_sha256": hashlib.sha256(json_bytes(environment.model_dump(mode="json"))).hexdigest(), "parameters": parameters,
            "controller_config": asdict(controller), "closure_config": asdict(closure), "duration_scale": duration_scale,
            "observation_contract": {"policy": "fully_observed_model_based_baseline", "inputs": ["joint encoders", "model parameters", "contact forces on the gripper"], "vlm": False},
            "interventions": [], "retry_limit": 0, "attempts": 1, "manual_trajectory_edits": 0,
            "limits": "Manifest joint/actuator limits; closure force bounded by the object's needs and the ceiling fraction; penetration 4 mm; no relaxed thresholds.",
            "clock_disclosure": {"physics_timestep_s": float(model.opt.timestep), "phase_timing_on_native_physics_time": True}}
    task.update(task_extra or {})
    payloads = {
        "model.mjb": buffer.tobytes(), "model.xml": session.scene.scene.xml.encode("utf-8"),
        "robot.urdf": session.source.read_bytes(), "robot.json": json_bytes(session.robot.manifest.model_dump(mode="json")),
        "world.json": json_bytes({"mode": "strict_fixed_world", "environment_id": environment.environment_id, "object_count": len(environment.objects), "timestep_s": model.opt.timestep,
                                  "gravity": model.opt.gravity.tolist(), "collision_policy": collision_policy(model), "fault": None}),
        "task.json": json_bytes(task), "outcome.json": json_bytes(outcome),
        "execution.json": json_bytes({"phases": [asdict(p) for p in result.phases], "active_skill_tree": "Sequence(approach, turn, descend, close, lift, hold, carry, lower, release, retreat, dwell)"}),
        "trace.npz": record.to_bytes(),
        "controller.json": json_bytes({"arm": "rigby_general.gates.control.ComputedTorqueController", "arm_config": asdict(controller), "closure": "rigby_general.contact.closure.ClosureController",
                                       "closure_config": asdict(closure), "duration_scale": duration_scale}),
        "repeats.json": json_bytes({"count": 1, "recorded_control_replay": replay, "note": "one attempt per trial; the recorded controls replay to the recorded states"}),
        "source.json": json_bytes(provenance), "source.zip": archive,
    }
    scale = session.robot.morphology.scale
    centre = [float(v) for v in (np.asarray(environment.objects[0].position_m) + np.asarray(environment.fixtures[1].position_m)) / 2.0]
    metadata = {"goal": "G12", "protocol": PROTOCOL, "robot_id": label, "rig_id": session.robot.manifest.rig_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "outcome": status, "fault": False, "simulation_duration_s": float(record.arrays["time_s"][-1]), "reference_duration_s": float(result.duration_s),
                "reference_clock_matches_physics": True, "caption": caption, "trace_sha256": record.content_hash(), "world_sha256": hashlib.sha256(payloads["world.json"]).hexdigest(),
                "camera": {"centre": centre, "reach": max(0.35, 0.45 * scale.reach_radius_m)}, "task_site": "scene_block_center",
                "controller": "computed torque + contact-driven closure under searched parameters", "observation": "model + joint encoders + gripper contact"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, "outcome": status, "trace_sha256": metadata["trace_sha256"], "simulation_duration_s": metadata["simulation_duration_s"], "replay_agrees": replay["agrees"]}


__all__ = ["PROTOCOL", "acquire", "configs_of", "evaluate", "outcome_of", "problem_world", "run_single_shot", "seal_single_shot", "summarize"]
