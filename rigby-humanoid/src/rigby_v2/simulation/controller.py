"""Bounded feedback control for native MuJoCo simulations.

The controller intentionally supports only direct, stateless joint motors.  A
model with a more complicated transmission must be adapted explicitly instead
of silently receiving physically meaningless controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


class ControllerConfigurationError(ValueError):
    """Raised when a model cannot be controlled by this controller safely."""


@dataclass(frozen=True)
class ControlTarget:
    """Desired generalized state at one physics step."""

    qpos: FloatArray
    qvel: FloatArray
    qacc: FloatArray

    @classmethod
    def stationary(cls, qpos: npt.ArrayLike, nv: int) -> "ControlTarget":
        return cls(
            qpos=np.asarray(qpos, dtype=np.float64).copy(),
            qvel=np.zeros(nv, dtype=np.float64),
            qacc=np.zeros(nv, dtype=np.float64),
        )


class TargetProvider(Protocol):
    """Returns a complete generalized target for a simulation time."""

    def sample(self, time_s: float) -> ControlTarget: ...


@dataclass(frozen=True)
class ConstantTarget:
    target: ControlTarget

    def sample(self, time_s: float) -> ControlTarget:
        del time_s
        return self.target


@dataclass(frozen=True)
class LinearKeyframeTrajectory:
    """Deterministic keyframe interpolation for controller targets.

    This provider is deliberately simple: it linearly interpolates supplied
    generalized positions and velocities and computes a piecewise-constant
    acceleration.  Free/ball quaternion entries should remain constant; the
    controller never actuates those coordinates.
    """

    times_s: FloatArray
    qpos: FloatArray
    qvel: FloatArray
    qacc: FloatArray | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64)
        qpos = np.asarray(self.qpos, dtype=np.float64)
        qvel = np.asarray(self.qvel, dtype=np.float64)
        qacc = (
            None
            if self.qacc is None
            else np.asarray(self.qacc, dtype=np.float64)
        )
        if times.ndim != 1 or len(times) < 1:
            raise ValueError("times_s must contain at least one keyframe")
        if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("times_s must be finite and strictly increasing")
        if qpos.ndim != 2 or qpos.shape[0] != len(times):
            raise ValueError("qpos must have one row per keyframe")
        if qvel.ndim != 2 or qvel.shape[0] != len(times):
            raise ValueError("qvel must have one row per keyframe")
        if np.any(~np.isfinite(qpos)) or np.any(~np.isfinite(qvel)):
            raise ValueError("trajectory values must be finite")
        if qacc is not None and (
            qacc.shape != qvel.shape or np.any(~np.isfinite(qacc))
        ):
            raise ValueError("qacc must be finite and match qvel")
        object.__setattr__(self, "times_s", times.copy())
        object.__setattr__(self, "qpos", qpos.copy())
        object.__setattr__(self, "qvel", qvel.copy())
        object.__setattr__(self, "qacc", None if qacc is None else qacc.copy())

    def sample(self, time_s: float) -> ControlTarget:
        t = float(np.clip(time_s, self.times_s[0], self.times_s[-1]))
        if len(self.times_s) == 1:
            return ControlTarget(
                self.qpos[0].copy(),
                self.qvel[0].copy(),
                (
                    np.zeros(self.qvel.shape[1], dtype=np.float64)
                    if self.qacc is None
                    else self.qacc[0].copy()
                ),
            )
        right = int(np.searchsorted(self.times_s, t, side="right"))
        right = min(max(right, 1), len(self.times_s) - 1)
        left = right - 1
        span = float(self.times_s[right] - self.times_s[left])
        alpha = (t - float(self.times_s[left])) / span
        qpos = (1.0 - alpha) * self.qpos[left] + alpha * self.qpos[right]
        qvel = (1.0 - alpha) * self.qvel[left] + alpha * self.qvel[right]
        qacc = (
            (self.qvel[right] - self.qvel[left]) / span
            if self.qacc is None
            else (1.0 - alpha) * self.qacc[left] + alpha * self.qacc[right]
        )
        return ControlTarget(qpos=qpos, qvel=qvel, qacc=qacc)


@dataclass(frozen=True)
class PDGains:
    kp: float = 80.0
    kd: float = 8.0
    max_pd_torque: float = 100.0
    max_control: float = 150.0

    def __post_init__(self) -> None:
        values = (self.kp, self.kd, self.max_pd_torque, self.max_control)
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("controller gains and bounds must be finite and positive")


@dataclass(frozen=True)
class StandingControlConfig:
    """Task-space targets for deterministic free-root standing control."""

    support_foot_bodies: tuple[str, ...]
    pelvis_body: str
    torso_body: str
    correction_joint_names: tuple[str, ...] = ()
    com_kp: float = 180.0
    com_kd: float = 30.0
    pelvis_kp: float = 120.0
    pelvis_kd: float = 20.0
    torso_kp: float = 90.0
    torso_kd: float = 14.0
    foot_kp: float = 240.0
    foot_kd: float = 32.0
    max_task_force: float = 300.0
    max_task_torque: float = 120.0
    max_generalized_correction: float = 80.0
    posture_kp: float = 80.0
    posture_kd: float = 16.0
    inverse_rcond: float = 1e-7

    def __post_init__(self) -> None:
        if not self.support_foot_bodies:
            raise ValueError("standing control requires at least one support foot")
        if len(set(self.support_foot_bodies)) != len(self.support_foot_bodies):
            raise ValueError("support foot body names must be unique")
        if not self.pelvis_body or not self.torso_body:
            raise ValueError("pelvis and torso body names are required")
        if len(set(self.correction_joint_names)) != len(self.correction_joint_names):
            raise ValueError("standing correction joint names must be unique")
        values = (
            self.com_kp,
            self.com_kd,
            self.pelvis_kp,
            self.pelvis_kd,
            self.torso_kp,
            self.torso_kd,
            self.foot_kp,
            self.foot_kd,
            self.max_task_force,
            self.max_task_torque,
            self.max_generalized_correction,
            self.posture_kp,
            self.posture_kd,
            self.inverse_rcond,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("standing gains and bounds must be finite and positive")


@dataclass(frozen=True)
class _MotorMap:
    actuator_id: int
    qpos_adr: int
    dof_adr: int
    transmission_gain: float
    control_low: float
    control_high: float


class InverseDynamicsPDController:
    """Inverse-dynamics feed-forward plus bounded joint-space PD.

    All inverse-dynamics work happens in a scratch ``MjData``.  The active
    simulation state is therefore only changed by ``mj_step`` after its one
    allowed initialization write.
    """

    def __init__(self, model: mujoco.MjModel, gains: PDGains | None = None) -> None:
        self._model = model
        self.gains = gains or PDGains()
        self._scratch = mujoco.MjData(model)
        self._feed_forward = np.zeros(model.nv, dtype=np.float64)
        self._inertial_force = np.zeros(model.nv, dtype=np.float64)
        self._motors = self._build_motor_map(model)

    @staticmethod
    def _build_motor_map(model: mujoco.MjModel) -> tuple[_MotorMap, ...]:
        if model.nu == 0:
            raise ControllerConfigurationError("model has no actuators")
        maps: list[_MotorMap] = []
        seen_dofs: set[int] = set()
        for actuator_id in range(model.nu):
            if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ControllerConfigurationError(
                    f"actuator {actuator_id} is not a direct joint transmission"
                )
            if model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE:
                raise ControllerConfigurationError(
                    f"actuator {actuator_id} has unsupported activation dynamics"
                )
            if model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED:
                raise ControllerConfigurationError(
                    f"actuator {actuator_id} does not use fixed gain"
                )
            if model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE:
                raise ControllerConfigurationError(
                    f"actuator {actuator_id} has state-dependent bias"
                )
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            joint_type = model.jnt_type[joint_id]
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ControllerConfigurationError(
                    f"actuator {actuator_id} targets a multi-DOF joint"
                )
            dof_adr = int(model.jnt_dofadr[joint_id])
            if dof_adr in seen_dofs:
                raise ControllerConfigurationError(
                    f"multiple actuators target DOF {dof_adr}; provide an explicit adapter"
                )
            seen_dofs.add(dof_adr)
            gain = float(model.actuator_gainprm[actuator_id, 0])
            gear = float(model.actuator_gear[actuator_id, 0])
            transmission_gain = gain * gear
            if not np.isfinite(transmission_gain) or abs(transmission_gain) < 1e-12:
                raise ControllerConfigurationError(
                    f"actuator {actuator_id} has a zero or invalid transmission gain"
                )
            if bool(model.actuator_ctrllimited[actuator_id]):
                low, high = (float(value) for value in model.actuator_ctrlrange[actuator_id])
            else:
                low, high = -np.inf, np.inf
            maps.append(
                _MotorMap(
                    actuator_id=actuator_id,
                    qpos_adr=int(model.jnt_qposadr[joint_id]),
                    dof_adr=dof_adr,
                    transmission_gain=transmission_gain,
                    control_low=low,
                    control_high=high,
                )
            )
        return tuple(maps)

    def compute(
        self,
        data: mujoco.MjData,
        target: ControlTarget,
        extra_generalized_torque: npt.ArrayLike | None = None,
    ) -> FloatArray:
        qpos = np.asarray(target.qpos, dtype=np.float64)
        qvel = np.asarray(target.qvel, dtype=np.float64)
        qacc = np.asarray(target.qacc, dtype=np.float64)
        if qpos.shape != (self._model.nq,):
            raise ValueError(f"target qpos must have shape ({self._model.nq},)")
        if qvel.shape != (self._model.nv,) or qacc.shape != (self._model.nv,):
            raise ValueError(f"target qvel/qacc must have shape ({self._model.nv},)")
        if np.any(~np.isfinite(qpos)) or np.any(~np.isfinite(qvel)) or np.any(~np.isfinite(qacc)):
            raise ValueError("target state must be finite")
        extra = np.zeros(self._model.nv, dtype=np.float64)
        if extra_generalized_torque is not None:
            extra = np.asarray(extra_generalized_torque, dtype=np.float64)
            if extra.shape != (self._model.nv,) or np.any(~np.isfinite(extra)):
                raise ValueError(f"extra generalized torque must have shape ({self._model.nv},)")

        # Calculate feed-forward forces away from the authoritative MjData.
        self._scratch.qpos[:] = data.qpos
        self._scratch.qvel[:] = data.qvel
        self._scratch.time = data.time
        # A free-root contact system does not have a unique unconstrained
        # inverse-dynamics solution for a prescribed zero acceleration.  Calling
        # ``mj_inverse`` here used MuJoCo's constraint inverse and produced a
        # fictitious ~141 N m demand on every canonical leg motor at neutral
        # rest.  Use the well-defined unconstrained terms M*qacc + bias instead;
        # the standing layer supplies contact-consistent feedback and the free
        # root remains unactuated.
        mujoco.mj_forward(self._model, self._scratch)
        mujoco.mj_mulM(
            self._model,
            self._scratch,
            self._inertial_force,
            qacc,
        )
        self._feed_forward[:] = self._scratch.qfrc_bias + self._inertial_force

        control = np.zeros(self._model.nu, dtype=np.float64)
        for motor in self._motors:
            position_error = qpos[motor.qpos_adr] - float(data.qpos[motor.qpos_adr])
            velocity_error = qvel[motor.dof_adr] - float(data.qvel[motor.dof_adr])
            pd_torque = self.gains.kp * position_error + self.gains.kd * velocity_error
            pd_torque = float(
                np.clip(pd_torque, -self.gains.max_pd_torque, self.gains.max_pd_torque)
            )
            feed_forward = float(self._feed_forward[motor.dof_adr])
            command = (feed_forward + pd_torque + extra[motor.dof_adr]) / motor.transmission_gain
            low = max(motor.control_low, -self.gains.max_control)
            high = min(motor.control_high, self.gains.max_control)
            control[motor.actuator_id] = np.clip(command, low, high)
        return control


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ControllerConfigurationError(f"standing body {name!r} does not exist")
    return int(body_id)


def _body_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[FloatArray, FloatArray]:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        0,
    )
    return velocity[:3], velocity[3:]


def _orientation_error(current: FloatArray, target: FloatArray) -> FloatArray:
    """Small-angle world-frame error between two rotation matrices."""

    return 0.5 * sum(
        (np.cross(current[:, axis], target[:, axis]) for axis in range(3)),
        start=np.zeros(3, dtype=np.float64),
    )


class WholeBodyStandingController:
    """Layer deterministic COM, pelvis, torso, and support-foot tasks over PD.

    Task errors are converted to generalized forces through MuJoCo Jacobians.
    Only the actuated entries are consumed by the bounded direct-motor base
    controller; the free root remains genuinely unactuated.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        initial_data: mujoco.MjData,
        base: InverseDynamicsPDController,
        config: StandingControlConfig,
    ) -> None:
        self._model = model
        self._base = base
        self.config = config
        self._pelvis_id = _body_id(model, config.pelvis_body)
        self._torso_id = _body_id(model, config.torso_body)
        self._foot_ids = tuple(_body_id(model, name) for name in config.support_foot_bodies)
        root_id = self._pelvis_id
        while int(model.body_parentid[root_id]) > 0:
            root_id = int(model.body_parentid[root_id])
        if model.body_jntnum[root_id] < 1:
            raise ControllerConfigurationError("standing hierarchy has no free-root body")
        root_joint = int(model.body_jntadr[root_id])
        if model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ControllerConfigurationError("standing hierarchy root is not a free joint")
        self._root_id = root_id
        self._root_joint = root_joint
        self._root_qpos_adr = int(model.jnt_qposadr[root_joint])
        self._root_dof_adr = int(model.jnt_dofadr[root_joint])
        self._target_com = initial_data.subtree_com[root_id].copy()
        self._target_pelvis = initial_data.xpos[self._pelvis_id].copy()
        self._target_root_rotation = initial_data.xmat[root_id].reshape(3, 3).copy()
        self._target_torso_rotation = initial_data.xmat[self._torso_id].reshape(3, 3).copy()
        self._target_feet = tuple(initial_data.xpos[body_id].copy() for body_id in self._foot_ids)
        self._correction_mask = np.ones(model.nv, dtype=np.float64)
        if config.correction_joint_names:
            self._correction_mask.fill(0.0)
            for name in config.correction_joint_names:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0:
                    raise ControllerConfigurationError(
                        f"standing correction joint {name!r} does not exist"
                    )
                joint_type = model.jnt_type[joint_id]
                width = 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    raise ControllerConfigurationError(
                        "standing correction joints must be actuated non-free joints"
                    )
                dof_adr = int(model.jnt_dofadr[joint_id])
                self._correction_mask[dof_adr : dof_adr + width] = 1.0

        # Floating-base inverse dynamics must solve actuator efforts and
        # physical support reactions together.  S maps actual motor controls
        # to generalized force.  Six wrench columns per support foot represent
        # forces that MuJoCo's real contact constraints must provide; they are
        # never applied as controls or hidden root forces.
        self._actuation = np.zeros((model.nv, model.nu), dtype=np.float64)
        for motor in base._motors:
            self._actuation[motor.dof_adr, motor.actuator_id] = (
                motor.transmission_gain
            )
        self._has_distal_balance_actuator = any(
            "ankle" in (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    int(model.actuator_trnid[motor.actuator_id, 0]),
                )
                or ""
            )
            for motor in base._motors
        )
        self._underactuated_scratch = mujoco.MjData(model)
        self._inverse_matrix = np.zeros(
            (model.nv, model.nu + 6 * len(self._foot_ids)), dtype=np.float64
        )
        self._inverse_matrix[:, : model.nu] = self._actuation
        self._desired_qacc = np.zeros(model.nv, dtype=np.float64)
        self._inertial_force = np.zeros(model.nv, dtype=np.float64)
        self._generalized_demand = np.zeros(model.nv, dtype=np.float64)

    def _linear_task(
        self,
        data: mujoco.MjData,
        body_id: int,
        target: FloatArray,
        kp: float,
        kd: float,
    ) -> FloatArray:
        jacobian = np.zeros((3, self._model.nv), dtype=np.float64)
        rotational = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self._model, data, jacobian, rotational, body_id)
        _, velocity = _body_velocity(self._model, data, body_id)
        force = kp * (target - data.xpos[body_id]) - kd * velocity
        force = np.clip(force, -self.config.max_task_force, self.config.max_task_force)
        return jacobian.T @ force

    def _correction(self, data: mujoco.MjData) -> FloatArray:
        correction = np.zeros(self._model.nv, dtype=np.float64)

        # Keep the whole-hierarchy COM over its calibrated support location.
        com_jacobian = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacSubtreeCom(self._model, data, com_jacobian, self._root_id)
        root_angular_velocity, root_linear_velocity = _body_velocity(
            self._model, data, self._root_id
        )
        del root_angular_velocity
        com_error = self._target_com - data.subtree_com[self._root_id]
        com_force = self.config.com_kp * com_error - self.config.com_kd * root_linear_velocity
        com_force = np.clip(
            com_force, -self.config.max_task_force, self.config.max_task_force
        )
        correction += com_jacobian.T @ com_force

        correction += self._linear_task(
            data,
            self._pelvis_id,
            self._target_pelvis,
            self.config.pelvis_kp,
            self.config.pelvis_kd,
        )

        torso_jacobian_pos = np.zeros((3, self._model.nv), dtype=np.float64)
        torso_jacobian_rot = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacBody(
            self._model,
            data,
            torso_jacobian_pos,
            torso_jacobian_rot,
            self._torso_id,
        )
        torso_angular_velocity, _ = _body_velocity(self._model, data, self._torso_id)
        torso_rotation = data.xmat[self._torso_id].reshape(3, 3)
        rotation_error = _orientation_error(torso_rotation, self._target_torso_rotation)
        torque = (
            self.config.torso_kp * rotation_error
            - self.config.torso_kd * torso_angular_velocity
        )
        torque = np.clip(
            torque, -self.config.max_task_torque, self.config.max_task_torque
        )
        correction += torso_jacobian_rot.T @ torque

        for body_id, target in zip(self._foot_ids, self._target_feet, strict=True):
            correction += self._linear_task(
                data,
                body_id,
                target,
                self.config.foot_kp,
                self.config.foot_kd,
            )
        return np.clip(
            correction,
            -self.config.max_generalized_correction,
            self.config.max_generalized_correction,
        ) * self._correction_mask

    def compute(self, data: mujoco.MjData, target: ControlTarget) -> FloatArray:
        qpos = np.asarray(target.qpos, dtype=np.float64)
        qvel = np.asarray(target.qvel, dtype=np.float64)
        qacc = np.asarray(target.qacc, dtype=np.float64)
        if qpos.shape != (self._model.nq,):
            raise ValueError(f"target qpos must have shape ({self._model.nq},)")
        if qvel.shape != (self._model.nv,) or qacc.shape != (self._model.nv,):
            raise ValueError(f"target qvel/qacc must have shape ({self._model.nv},)")
        if any(np.any(~np.isfinite(item)) for item in (qpos, qvel, qacc)):
            raise ValueError("target state must be finite")

        if not self._has_distal_balance_actuator:
            # Compatibility path for deliberately minimal smoke bipeds whose
            # feet have no ankle DOFs.  They cannot realize a two-foot wrench
            # distribution through distal actuation, so retain the earlier
            # bounded hip/knee constraint-inverse controller for this explicit
            # model class.  Production canonical rigs always take the audited
            # contact-consistent path below.
            scratch = self._underactuated_scratch
            scratch.qpos[:] = data.qpos
            scratch.qvel[:] = data.qvel
            scratch.qacc[:] = qacc
            scratch.time = data.time
            mujoco.mj_inverse(self._model, scratch)
            correction = self._correction(data)
            control = np.zeros(self._model.nu, dtype=np.float64)
            for motor in self._base._motors:
                position_error = qpos[motor.qpos_adr] - data.qpos[motor.qpos_adr]
                velocity_error = qvel[motor.dof_adr] - data.qvel[motor.dof_adr]
                pd_torque = np.clip(
                    self._base.gains.kp * position_error
                    + self._base.gains.kd * velocity_error,
                    -self._base.gains.max_pd_torque,
                    self._base.gains.max_pd_torque,
                )
                command = (
                    scratch.qfrc_inverse[motor.dof_adr]
                    + pd_torque
                    + correction[motor.dof_adr]
                ) / motor.transmission_gain
                low = max(motor.control_low, -self._base.gains.max_control)
                high = min(motor.control_high, self._base.gains.max_control)
                control[motor.actuator_id] = np.clip(command, low, high)
            return control

        self._desired_qacc[:] = qacc
        root_angular_velocity, root_linear_velocity = _body_velocity(
            self._model, data, self._root_id
        )
        root_position_error = self._target_pelvis - data.xpos[self._pelvis_id]
        root_rotation = data.xmat[self._root_id].reshape(3, 3)
        root_rotation_error = _orientation_error(
            root_rotation, self._target_root_rotation
        )
        root = self._root_dof_adr
        self._desired_qacc[root : root + 3] = (
            self.config.pelvis_kp * root_position_error
            - self.config.pelvis_kd * root_linear_velocity
        )
        self._desired_qacc[root + 3 : root + 6] = (
            self.config.torso_kp * root_rotation_error
            - self.config.torso_kd * root_angular_velocity
        )
        for motor in self._base._motors:
            position_error = qpos[motor.qpos_adr] - data.qpos[motor.qpos_adr]
            velocity_error = qvel[motor.dof_adr] - data.qvel[motor.dof_adr]
            self._desired_qacc[motor.dof_adr] = (
                qacc[motor.dof_adr]
                + self.config.posture_kp * position_error
                + self.config.posture_kd * velocity_error
            )
        np.clip(
            self._desired_qacc,
            -self.config.max_task_force,
            self.config.max_task_force,
            out=self._desired_qacc,
        )

        mujoco.mj_mulM(
            self._model,
            data,
            self._inertial_force,
            self._desired_qacc,
        )
        self._generalized_demand[:] = self._inertial_force + data.qfrc_bias

        column = self._model.nu
        for body_id in self._foot_ids:
            linear = np.zeros((3, self._model.nv), dtype=np.float64)
            angular = np.zeros((3, self._model.nv), dtype=np.float64)
            mujoco.mj_jacBody(
                self._model,
                data,
                linear,
                angular,
                body_id,
            )
            self._inverse_matrix[:, column : column + 3] = linear.T
            self._inverse_matrix[:, column + 3 : column + 6] = angular.T
            column += 6

        solution = np.linalg.lstsq(
            self._inverse_matrix,
            self._generalized_demand,
            rcond=self.config.inverse_rcond,
        )[0]
        control = solution[: self._model.nu].copy()
        for motor in self._base._motors:
            low = max(motor.control_low, -self._base.gains.max_control)
            high = min(motor.control_high, self._base.gains.max_control)
            control[motor.actuator_id] = np.clip(
                control[motor.actuator_id], low, high
            )
        return control
