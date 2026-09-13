"""Transitions on physics: hold still, or move the joints, from where the body is.

Both run under the same computed-torque controller and closure the skills
run under, continue the world the last skill left (full integration state,
clock and all), and are recorded on the same physics record, so a repaired
composition is one continuous replayable run. A joint move is planned as a
quintic in joint space between where the joints are and where the checker
says they must go, and is refused before it starts if the self-collision
guard finds any sample of that path inside the body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np
from rigby_core.motion.timing import QuinticSegment
from rigby_core.simulation.controller import ControlTarget
from rigby_core.simulation.recording import STATE_SPEC, PhysicsRecorder

from ..contact.closure import ClosureController
from ..contact.transfer import TransferStart
from ..contracts import EffectorV1, RobotAssetManifestV1
from ..gates.control import ComputedTorqueController, ControllerConfig
from ..grounding import ik


GUARD_SAMPLES = 24
MIN_MOVE_S = 0.5
MAX_SETTLE_S = 3.0
MOVE_SPEED_FRACTION = 0.35
"""Of each joint's velocity limit: how fast a repair move travels at most."""
BRAKE_RATE = 4.0
"""Per second of each joint's velocity limit: the deceleration a settle may
ask for. A boundary reached still moving is brought to rest along a quintic
that starts at the velocity the joint actually has and ends at zero, over
the time that rate needs; a reference frozen at the current pose would ask
the controller for an impulse, and the joints crossed their velocity limits
in the few milliseconds it took to deliver it."""


@dataclass
class Executed:
    executed: bool
    start: TransferStart | None
    physics_s: float
    joint_travel_rad: float
    peak_speed_fraction: float
    interrupted: bool = False
    refusal: str | None = None
    detail: str = ""
    times_s: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    qpos: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)
    qvel: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)


def _begin(model: mujoco.MjModel, manifest: RobotAssetManifestV1, resume: TransferStart | None) -> mujoco.MjData:
    data = mujoco.MjData(model)
    if resume is None:
        from ..contact.grasp import _scene_rest_qpos

        data.qpos[:] = _scene_rest_qpos(model, manifest)
        mujoco.mj_forward(model, data)
    elif resume.state is not None:
        mujoco.mj_setState(model, data, np.asarray(resume.state, dtype=float), STATE_SPEC)
        mujoco.mj_forward(model, data)
        mujoco.mj_setState(model, data, np.asarray(resume.state, dtype=float), STATE_SPEC)
    else:
        data.qpos[:] = resume.qpos
        data.qvel[:] = resume.qvel
        data.time = resume.time_s
        mujoco.mj_forward(model, data)
    return data


def _capture(recorder: PhysicsRecorder | None, data: mujoco.MjData, command: np.ndarray, demand: np.ndarray, resume: TransferStart | None) -> None:
    if recorder is None:
        return
    recorded = recorder.rows["time_s"]
    now = float(data.time)
    if resume is not None and recorded and now <= recorded[-1] + 1e-12:
        recorder.amend_last_action(command, demand=demand)
    else:
        recorder.capture(data, command, control_time_s=now, demand=demand)


Reference = Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]
"""Elapsed seconds to the arm joints' reference position, velocity and acceleration."""


def reference_of(segment: QuinticSegment, duration_s: float) -> Reference:
    def planned(elapsed: float):
        sample = segment.sample(min(max(elapsed, 0.0), duration_s))
        return np.asarray(sample.position, dtype=float), np.asarray(sample.velocity, dtype=float), np.asarray(sample.acceleration, dtype=float)
    return planned


def track(model: mujoco.MjModel, manifest: RobotAssetManifestV1, effector: EffectorV1, arm_joints: tuple[str, ...], resume: TransferStart | None,
          recorder: PhysicsRecorder | None, planned: Reference, duration_s: float, *, holding: bool,
          stop_when: Callable[[mujoco.MjData], bool] | None = None, should_stop: Callable[[float], bool] | None = None,
          on_step: Callable[[mujoco.MjData], None] | None = None, controller_config: ControllerConfig | None = None) -> Executed:
    """Follow ``planned(elapsed)`` -- position, velocity and acceleration of
    the arm joints -- for ``duration_s`` of physics (or until ``stop_when``),
    the closure holding if ``holding``. The velocity and acceleration reach
    the controller: a reference that begins at the velocity the arm has is
    followed, not fought."""

    data = _begin(model, manifest, resume)
    if on_step is not None:
        on_step(data)
    dt = float(model.opt.timestep)
    controller = ComputedTorqueController(model, controller_config or ControllerConfig())
    closure = ClosureController(model, manifest, effector, object_geoms=frozenset({"scene_block_geom"}))
    arm_adr = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm_joints])
    arm_dof = np.array([int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm_joints])
    grip_adr = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in effector.grip_joints], dtype=int)
    limits = np.array([next(d.velocity_limit for d in manifest.dofs if d.joint == n) for n in arm_joints], dtype=float)
    base = np.array(data.qpos, dtype=float)
    started = float(data.time)
    times, qpos_log, qvel_log = [], [], []
    travel = 0.0
    peak_fraction = 0.0
    interrupted = False
    previous = np.array(data.qpos[arm_adr], dtype=float)
    target = mujoco.MjData(model)
    steps = max(1, int(round(duration_s / dt)))
    for step in range(steps + 1):
        now = float(data.time)
        elapsed = now - started
        if should_stop is not None and should_stop(now):
            interrupted = True
            break
        report = closure.step(data, dt, closing=holding)
        target.qpos[:] = base
        position, velocity, acceleration = planned(min(elapsed, duration_s))
        target.qpos[arm_adr] = position
        target_qvel = np.zeros(model.nv)
        target_qacc = np.zeros(model.nv)
        target_qvel[arm_dof] = velocity
        target_qacc[arm_dof] = acceleration
        if holding:
            target.qpos[grip_adr] = data.qpos[grip_adr]
        else:
            commanded = closure.commanded_qpos()
            target.qpos[grip_adr] = [commanded.get(n, float(data.qpos[a])) for n, a in zip(effector.grip_joints, grip_adr)]
        # Everything that is not the arm or the gripper is left where it is.
        others = np.ones(model.nq, dtype=bool)
        others[arm_adr] = False
        others[grip_adr] = False
        target.qpos[others] = data.qpos[others]
        command = controller.compute(data, ControlTarget(qpos=np.array(target.qpos), qvel=target_qvel, qacc=target_qacc))
        if holding:
            for joint_name, force in closure.force_commands().items():
                actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_motor")
                if actuator >= 0:
                    command[actuator] = float(np.clip(force, model.actuator_forcerange[actuator][0], model.actuator_forcerange[actuator][1]))
        times.append(now)
        qpos_log.append(np.array(data.qpos, dtype=float))
        qvel_log.append(np.array(data.qvel, dtype=float))
        current = np.array(data.qpos[arm_adr], dtype=float)
        travel += float(np.abs(current - previous).sum())
        previous = current
        peak_fraction = max(peak_fraction, float(np.max(np.abs(np.array(data.qvel[arm_dof], dtype=float)) / limits)))
        _capture(recorder, data, command, controller.last_demand, resume if step == 0 else None)
        if step == steps or (stop_when is not None and elapsed > 0.0 and stop_when(data)):
            break
        data.ctrl[:] = command
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(data)
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    start = TransferStart(qpos=np.array(data.qpos, dtype=float), qvel=np.array(data.qvel, dtype=float), time_s=float(data.time), state=state)
    return Executed(executed=True, start=start, physics_s=float(data.time) - started, joint_travel_rad=travel, peak_speed_fraction=peak_fraction,
                    interrupted=interrupted, times_s=np.asarray(times), qpos=np.asarray(qpos_log), qvel=np.asarray(qvel_log))


def settle(model, manifest, effector, arm_joints, resume: TransferStart, recorder, *, speed_fraction: float, holding: bool, max_s: float = MAX_SETTLE_S) -> Executed:
    """Brake the arm to rest from the velocity it has, then hold until every
    joint is under the speed limit."""

    arm_adr = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm_joints])
    dof = np.array([int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm_joints])
    limits = np.array([next(d.velocity_limit for d in manifest.dofs if d.joint == n) for n in arm_joints], dtype=float)
    q0 = np.array(resume.qpos, dtype=float)[arm_adr]
    v0 = np.array(resume.qvel, dtype=float)[dof]
    brake_s = max(0.1, float(np.max(np.abs(v0) / (BRAKE_RATE * limits))))
    # A quintic from (q0, v0) to rest lands where its own shape takes it;
    # v0 * t / 2 is where a uniform deceleration would stop.
    segment = QuinticSegment(q0, q0 + 0.5 * v0 * brake_s, brake_s, start_velocity=v0, end_velocity=np.zeros_like(v0))

    planned = reference_of(segment, brake_s)

    def still(data: mujoco.MjData) -> bool:
        return float(data.time) - resume.time_s >= brake_s and bool(np.all(np.abs(np.array(data.qvel[dof], dtype=float)) <= 0.5 * speed_fraction * limits))

    result = track(model, manifest, effector, arm_joints, resume, recorder, planned, max_s, holding=holding, stop_when=still)
    result.detail = f"braked over {brake_s:.3f} s then " + ("settled" if result.physics_s < max_s - 1e-9 else "held for the full allowance")
    return result


def joint_move(model, manifest, effector, arm_joints, resume: TransferStart, recorder, targets: dict[str, float], guard: "ik.CollisionGuard | None", *,
               holding: bool, speed_fraction: float = MOVE_SPEED_FRACTION, on_step: Callable[[mujoco.MjData], None] | None = None,
               controller_config: ControllerConfig | None = None) -> Executed:
    """A quintic move of the named joints to their targets, the rest of the
    arm held; refused typed if any sample of the straight joint-space path
    puts guarded bodies inside one another."""

    dofs = {dof.name: dof for dof in manifest.dofs}
    by_joint = {dof.joint: dof for dof in manifest.dofs}
    arm_adr = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm_joints])
    start_arm = np.array(resume.qpos, dtype=float)[arm_adr]
    end_arm = start_arm.copy()
    for index, name in enumerate(arm_joints):
        dof = by_joint[name]
        if dof.name in targets:
            end_arm[index] = float(targets[dof.name])
        elif name in targets:
            end_arm[index] = float(targets[name])
    if guard is not None:
        full = np.array(resume.qpos, dtype=float)
        for fraction in np.linspace(0.0, 1.0, GUARD_SAMPLES):
            sample = full.copy()
            sample[arm_adr] = start_arm + fraction * (end_arm - start_arm)
            inside = ik.penetrations(model, guard, sample)
            if inside:
                first, second, depth = inside[0]
                return Executed(executed=False, start=resume, physics_s=0.0, joint_travel_rad=0.0, peak_speed_fraction=0.0,
                                refusal="self_collision_path", detail=f"{first} inside {second} by {depth * 1000:.1f} mm at {fraction:.2f} of the move")
    limits = np.array([by_joint[n].velocity_limit for n in arm_joints], dtype=float)
    dof = np.array([int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm_joints])
    v0 = np.array(resume.qvel, dtype=float)[dof]
    duration = max(MIN_MOVE_S, float(np.max(np.abs(end_arm - start_arm) / (speed_fraction * limits))), float(np.max(np.abs(v0) / (BRAKE_RATE * limits))))
    segment = QuinticSegment(start_arm, end_arm, duration, start_velocity=v0, end_velocity=np.zeros_like(v0))
    return track(model, manifest, effector, arm_joints, resume, recorder, reference_of(segment, duration), duration, holding=holding, on_step=on_step, controller_config=controller_config)
