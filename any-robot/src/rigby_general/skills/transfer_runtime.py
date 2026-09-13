"""Bind the neutral skill contract's leaves to the transfer primitive on physics.

The core executor knows nothing about bodies; this module is what a body
supplies. A session is one body in one authored world with one physics
record: every primitive leaf continues the world the last one left -- the
arm where it stands, the object where it lies, the clock where it stood --
so a tree that tries a second manipulator after the first lost the object
tries it against the object where it actually went, not a reset scene. A
primitive is the G06 transfer, whichever manipulator the node names; an
observation reads either the scene as it is (inventory) or what the
independent placement evaluator recorded during the last transfer's dwell
(placement), and says nothing when there was no dwell to read.

Predicates are functions of the belief: the object's pose is known once
the inventory has been read; the object is placed when the evaluator saw it
resting, released and still inside the destination; the object is in reach
when some manipulator's measured envelope contains it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rigby_core.evidence import write_bundle
from rigby_core.simulation.recording import PhysicsRecorder, replay_physics
from rigby_core.skills import Belief, ExecutionRecordV1, Interrupt, LeafContext, LeafOutcome, SkillLibraryV1, TaskTreeV1, Verdict

from ..contact.placement import PlacementEvaluator, PlacementGoal
from ..contact.transfer import TransferResult, TransferStart, attempt_transfer, collision_policy, transfer_scene_from_environment
from ..contracts import EffectorV1
from ..evidence.capture import json_bytes, source_provenance
from ..grounding.grounder import figure_site_for
from ..grounding.workspace import WorkspaceFrame, build_workspace_frame
from ..pipeline import ingest_robot
from ..scenes.block import block_qpos_address
from ..scenes.environment import EnvironmentV1


PROTOCOL = "rigby.skill-tree-run/1"


@dataclass
class SimulationClock:
    """The executor's clock is the physics clock of the session."""

    session: "BodySession"

    def now(self) -> float:
        return self.session.time_s


@dataclass
class BodySession:
    robot: Any
    source: Path
    environment: EnvironmentV1
    goal: PlacementGoal
    scene: Any
    effectors: dict[str, EffectorV1]
    frames: dict[str, WorkspaceFrame]
    recorder: PhysicsRecorder
    state: TransferStart | None = None
    time_s: float = 0.0
    results: list[tuple[str, TransferResult]] = field(default_factory=list)

    @classmethod
    def open(cls, zoo_id: str, source: Path, environment: EnvironmentV1, goal: PlacementGoal) -> "BodySession":
        robot = ingest_robot(source, robot_id=zoo_id)
        effectors = {e.chain_id: e for e in robot.morphology.grasping_effectors}
        if not effectors:
            raise ValueError(f"{zoo_id} has no grasping effector; a transfer tree cannot be bound to it")
        frames = {}
        for chain_id, effector in effectors.items():
            chain = next(c for c in robot.morphology.chains if c.chain_id == chain_id)
            frames[chain_id] = build_workspace_frame(robot.finalized.model, robot.morphology, chain, figure_site=figure_site_for(robot.manifest, chain_id))
        scene = transfer_scene_from_environment(robot.manifest, robot.mjcf_xml, environment, object_name="cube", destination_fixture="platform", goal=goal, asset_root=source.parent)
        return cls(robot=robot, source=source, environment=environment, goal=goal, scene=scene, effectors=effectors, frames=frames, recorder=PhysicsRecorder(scene.model))

    @property
    def model(self) -> mujoco.MjModel:
        return self.scene.model

    def current_qpos(self) -> np.ndarray:
        if self.state is not None:
            return np.array(self.state.qpos, dtype=float)
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        for dof in self.robot.manifest.dofs:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)
            finalized = self.robot.finalized.model
            data.qpos[int(self.model.jnt_qposadr[joint])] = self.robot.manifest.rest_qpos[int(finalized.jnt_qposadr[mujoco.mj_name2id(finalized, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)])]
        return np.array(data.qpos, dtype=float)

    def object_position(self) -> np.ndarray:
        address = block_qpos_address(self.model)
        return self.current_qpos()[address: address + 3]

    @property
    def last_result(self) -> TransferResult | None:
        return self.results[-1][1] if self.results else None


@dataclass
class TransferRuntime:
    """The leaves: ``transfer`` on any manipulator the session has, and the
    two observations. ``stop_at_s`` interrupts the running transfer at that
    physics time and tells the executor."""

    session: BodySession
    interrupt: Interrupt
    stop_at_s: float | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def run_primitive(self, context: LeafContext) -> LeafOutcome:
        node = context.node
        session = self.session
        effector_id = node.arguments["effector"]
        if effector_id not in session.effectors:
            return LeafOutcome(Verdict.FAILURE, f"no_such_effector:{effector_id}")

        def should_stop(now: float) -> bool:
            if context.should_stop():
                return True
            if self.stop_at_s is not None and now >= self.stop_at_s:
                self.interrupt.request(f"stopped from outside at {now:.3f} s of physics")
                return True
            return False

        result = attempt_transfer(session.robot.manifest, session.scene, session.effectors[effector_id], session.frames[effector_id],
                                  recorder=session.recorder, resume=session.state, should_stop=should_stop)
        session.results.append((node.node_id, result))
        if result.executed:
            session.state = result.continuation()
            session.time_s = result.final_time_s
        facts: dict[str, Any] = {}
        if result.executed and not result.interrupted and any(p.name == "dwell" for p in result.phases):
            facts["placed:cube:platform"] = bool(result.placement_success)
        self.calls.append({"node": node.node_id, "effector": effector_id, "executed": result.executed, "certified": result.certified, "gate": result.failed_gate,
                           "phases": [p.name for p in result.phases], "duration_s": result.duration_s, "interrupted": result.interrupted})
        if result.interrupted:
            verdict = Verdict.INTERRUPTED
        elif result.certified:
            verdict = Verdict.SUCCESS
        else:
            verdict = Verdict.FAILURE
        evidence = {"effector": effector_id, "executed": result.executed, "failed_gate": result.failed_gate, "phases": [p.name for p in result.phases],
                    "lift_height_m": result.lift_height_m, "hold_s": result.hold_s, "placement_dwell_s": result.placement_dwell_s,
                    "max_penetration_m": result.max_penetration_m, "peak_force_n": result.peak_force_n, "path_seed": result.path_seed,
                    "physics_time_s": [float(result.times_s[0]), float(result.times_s[-1])] if result.executed else None}
        return LeafOutcome(verdict, result.failed_gate or "", facts=facts, evidence=evidence)

    def observe(self, context: LeafContext) -> dict[str, Any] | None:
        node = context.node
        session = self.session
        if node.skill_id == "observe_inventory":
            position = session.object_position()
            evaluator = PlacementEvaluator(session.model, session.goal, object_geom="scene_block_geom", object_joint="scene_block_free")
            data = mujoco.MjData(session.model)
            data.qpos[:] = session.current_qpos()
            if session.state is not None:
                data.qvel[:] = session.state.qvel
            mujoco.mj_forward(session.model, data)
            sample = evaluator.assess(data)
            in_reach = any(frame.contains(position, margin=1.0) for frame in session.frames.values())
            self.calls.append({"node": node.node_id, "observed": "inventory", "object_position_m": [float(v) for v in position], "in_reach": in_reach})
            return {"known:cube": True, "pose:cube": [float(v) for v in position], "reach:cube": bool(in_reach),
                    "placed:cube:platform": bool(sample.whole_geometry_inside and sample.released)}
        if node.skill_id == "observe_placement":
            last = session.last_result
            if last is None or not last.executed or last.interrupted or not any(p.name == "dwell" for p in last.phases):
                self.calls.append({"node": node.node_id, "observed": "placement", "answer": None})
                return None
            self.calls.append({"node": node.node_id, "observed": "placement", "answer": bool(last.placement_success)})
            return {"placed:cube:platform": bool(last.placement_success), "placement_dwell_s": float(last.placement_dwell_s), "released:cube": bool(last.released)}
        return None


def predicates_for(session: BodySession):
    return {
        "object_known": lambda belief, args: belief.get(f"known:{args[0]}"),
        "object_placed": lambda belief, args: belief.get(f"placed:{args[0]}:{args[1]}"),
        "object_in_reach": lambda belief, args: belief.get(f"reach:{args[0]}"),
    }


def seal_tree_run(destination: Path, *, session: BodySession, library: SkillLibraryV1, tree: TaskTreeV1, record: ExecutionRecordV1,
                  label: str, caption: str, runtime_calls: list[dict[str, Any]]) -> dict:
    """One replayable bundle in the G01 layout for a whole tree run: the
    continuous physics record across every leaf, the library and tree by
    content, and the execution record with every node's verdict."""

    model = session.model
    executed = bool(session.recorder.rows["time_s"])
    if executed:
        physical = session.recorder.finish()
        replay = {"recorded_controls": replay_physics(model, physical)}
        replay["agrees"] = bool(replay["recorded_controls"]["agrees"])
        status = {"success": "success", "failure": "runtime_failure", "unknown": "undecided", "interrupted": "interrupted"}[record.verdict.value]
    else:
        data = mujoco.MjData(model)
        data.qpos[:] = session.current_qpos()
        mujoco.mj_forward(model, data)
        recorder = PhysicsRecorder(model)
        recorder.capture(data, np.zeros(model.nu), control_time_s=0.0)
        physical = recorder.finish()
        replay = {"recorded_controls": {"agrees": True, "note": "no motion was executed"}, "agrees": True}
        status = "pre_execution_refusal" if record.verdict is Verdict.FAILURE else record.verdict.value
    provenance, archive = source_provenance()
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    leaves = [{"node": node_id, "certified": r.certified, "failed_gate": r.failed_gate, "interrupted": r.interrupted, "executed": r.executed,
               "phases": [{"name": p.name, "start_s": p.start_s, "end_s": p.end_s, "note": p.note} for p in r.phases],
               "lift_height_m": r.lift_height_m, "hold_s": r.hold_s, "placement_dwell_s": r.placement_dwell_s, "placement_success": r.placement_success,
               "max_penetration_m": r.max_penetration_m, "peak_force_n": r.peak_force_n, "path_seed": r.path_seed, "collision_policy": r.collision_policy}
              for node_id, r in session.results]
    outcome = {"status": status, "scope": "a recursive skill tree executed on physics; verdicts by the neutral executor, placement by the independent evaluator",
               "verdict": record.verdict.value, "root_reason": record.root.reason, "interrupted": record.interrupted, "interrupt_reason": record.interrupt_reason,
               "leaves": leaves, "actual_physics_duration_s": float(physical.arrays["time_s"][-1]) if executed else 0.0,
               "physical_steps": (len(physical.arrays["state"]) - 1) if executed else 0,
               "refusal": None if executed else {"stage": "path", "code": next((r.failed_gate for _, r in session.results), record.root.reason),
                                                 "detail": next((r.violations[0].detail for _, r in session.results if r.violations), record.root.reason)}}
    task = {"goal": "G07", "protocol": PROTOCOL, "label": label, "library_id": library.library_id, "library_sha256": library.content_hash(),
            "tree_sha256": tree.content_hash(), "root": tree.root.skill_id, "arguments": dict(tree.root.arguments), "max_depth": tree.max_depth,
            "environment": session.environment.model_dump(mode="json"),
            "observation_contract": {"policy": "fully_observed_model_based_baseline", "inputs": ["joint encoders", "model parameters", "contact forces on the gripper"], "vlm": False},
            "interventions": [], "retry_limit": 0, "attempts": 1,
            "limits": "Manifest joint/actuator limits; closure force bounded; penetration 4 mm; every leaf timeout from the library; no relaxed thresholds.",
            "clock_disclosure": {"physics_timestep_s": float(model.opt.timestep), "executor_clock": "physics time of the session", "phase_timing_on_native_physics_time": True}}
    payloads = {
        "model.mjb": buffer.tobytes(), "model.xml": session.scene.scene.xml.encode("utf-8"),
        "robot.urdf": session.source.read_bytes(), "robot.json": json_bytes(session.robot.manifest.model_dump(mode="json")),
        "world.json": json_bytes({"mode": "strict_fixed_world", "environment_id": session.environment.environment_id, "object_count": len(session.environment.objects),
                                  "timestep_s": model.opt.timestep, "gravity": model.opt.gravity.tolist(), "collision_policy": collision_policy(model), "fault": None}),
        "task.json": json_bytes(task), "outcome.json": json_bytes(outcome),
        "library.json": library.model_dump_json(indent=2).encode("utf-8"), "tree.json": tree.model_dump_json(indent=2).encode("utf-8"),
        "execution.json": json_bytes({"record": json.loads(record.model_dump_json()), "runtime_calls": runtime_calls,
                                      "active_skill_tree": "the expanded tree in tree.json; verdicts per node in record"}),
        "trace.npz": physical.to_bytes(),
        "controller.json": json_bytes({"executor": "rigby_core.skills.execute", "arm": "rigby_general.gates.control.ComputedTorqueController", "closure": "rigby_general.contact.closure.ClosureController"}),
        "repeats.json": json_bytes({"count": 1, "recorded_control_replay": replay, "note": "one run of the tree; the recorded controls replay to the recorded states"}),
        "source.json": json_bytes(provenance), "source.zip": archive,
    }
    scale = session.robot.morphology.scale
    centre = [float(v) for v in (np.asarray(session.environment.objects[0].position_m) + np.asarray(session.environment.fixtures[1].position_m)) / 2.0]
    metadata = {"goal": "G07", "protocol": PROTOCOL, "robot_id": label, "rig_id": session.robot.manifest.rig_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "outcome": status, "fault": False,
                "simulation_duration_s": float(physical.arrays["time_s"][-1]) if executed else 0.0, "reference_duration_s": record.ended_s,
                "reference_clock_matches_physics": True, "caption": caption, "trace_sha256": physical.content_hash(),
                "world_sha256": hashlib.sha256(payloads["world.json"]).hexdigest(),
                "camera": {"centre": centre, "reach": max(0.35, 0.45 * scale.reach_radius_m)}, "task_site": "scene_block_center",
                "controller": "neutral skill executor over computed torque + contact-driven closure", "observation": "model + joint encoders + gripper contact"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, "outcome": status, "trace_sha256": metadata["trace_sha256"],
            "simulation_duration_s": metadata["simulation_duration_s"], "replay_agrees": replay["agrees"], "executed": executed}
