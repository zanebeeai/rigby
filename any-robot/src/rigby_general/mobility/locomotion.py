"""Analytic locomotion controllers for the three mobile bodies, and the body-neutral navigator that drives them.

Each controller is hand-authored for its body's mechanics and says so in
its provenance: a trot for the legged dog (diagonal pairs, position
servos tracking foot targets through an analytic leg IK), a pitch
balance for the wheeled biped (velocity servos on the wheels driven by
the torso's pitch, pitch rate and speed, the legs held), and a tripod
crawl for the octopus (alternating groups of three tentacles lifting,
swinging forward, pressing down and sweeping back). Every command goes
through the body's own actuators; nothing sets the base's pose or
velocity, applies a hidden wrench or props the body up. The navigator
above them knows only a base pose and a drive interface (forward speed,
turn rate): it steers to waypoints, stops within the radius, holds, and
turns around, the same for a walker, a roller and a crawler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .ingest import MobileBody


def quat_to_euler(quat) -> tuple[float, float, float]:
    """roll, pitch, yaw (x-y-z, radians) from a MuJoCo (w, x, y, z) quaternion."""

    w, x, y, z = (float(v) for v in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class BaseState:
    position: np.ndarray
    yaw: float
    pitch: float
    roll: float
    velocity: np.ndarray
    """world-frame linear velocity of the base"""
    angular: np.ndarray
    """body-frame angular velocity of the base"""


class Locomotor:
    """A body's drive: `control(data, v, omega)` returns the actuator command that pursues a forward speed v (m/s) and a turn rate omega (rad/s)."""

    provenance = "analytic, hand-authored per body; no training, no adaptation, no shared policy"
    max_speed_mps = 0.2
    min_speed_mps = 0.0
    """The slowest forward speed the drive makes way at; the navigator never asks for less while it still has distance to cover."""
    max_turn_radps = 0.6

    def __init__(self, body: MobileBody, model: mujoco.MjModel) -> None:
        self.body = body
        self.model = model
        self.declaration = body.declaration
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.declaration["base_body"])
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.declaration["root_joint"])
        self.root_qpos = int(model.jnt_qposadr[root])
        self.root_dof = int(model.jnt_dofadr[root])
        self.actuator_of = {}
        for a in range(model.nu):
            joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a][0]))
            self.actuator_of[joint] = a
        self.qpos_of = {j["name"]: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j["name"])]) for j in self.declaration["joints"]}
        self.dof_of = {j["name"]: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j["name"])]) for j in self.declaration["joints"]}
        self.stance = dict(self.declaration["stances"][self.working_stance()]["joints"])
        self.control_hz = 500.0

    def working_stance(self) -> str:
        return self.declaration["working_stance"]

    def smooth(self, v: float, omega: float, dt: float, tau_s: float = 0.35) -> tuple[float, float]:
        """The commanded speed and turn rate eased towards their targets with a first-order lag, so a gait starts and stops without a lurch."""

        if not hasattr(self, "_v"):
            self._v, self._omega = 0.0, 0.0
        alpha = 1.0 if dt <= 0.0 else min(1.0, dt / tau_s)
        self._v += alpha * (v - self._v)
        self._omega += alpha * (omega - self._omega)
        return self._v, self._omega

    def base_state(self, data: mujoco.MjData) -> BaseState:
        q = data.qpos[self.root_qpos + 3: self.root_qpos + 7]
        roll, pitch, yaw = quat_to_euler(q)
        velocity = np.array(data.qvel[self.root_dof: self.root_dof + 3], dtype=float)
        angular = np.array(data.qvel[self.root_dof + 3: self.root_dof + 6], dtype=float)
        return BaseState(position=np.array(data.xpos[self.base], dtype=float), yaw=yaw, pitch=pitch, roll=roll, velocity=velocity, angular=angular)

    def hold(self) -> np.ndarray:
        control = np.zeros(self.model.nu)
        for joint, value in self.stance.items():
            if joint in self.actuator_of:
                control[self.actuator_of[joint]] = value
        return control

    def reset(self, data: mujoco.MjData) -> None:
        pass

    def control(self, data: mujoco.MjData, v: float, omega: float) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------
# the dog: a trot
# --------------------------------------------------------------------------------------------------


THIGH_M, SHANK_M = 0.20, 0.21


def leg_ik(x: float, z: float) -> tuple[float, float]:
    """Hip pitch and knee for a foot at (x forward, z down, in the hip frame): the analytic two-link solution
    in the dog's convention (positive hip pitch swings the foot back, negative knee folds the shank forward)."""

    d = max(1e-6, min(math.hypot(x, z), THIGH_M + SHANK_M - 1e-3))
    cos_b = (d * d - THIGH_M * THIGH_M - SHANK_M * SHANK_M) / (2.0 * THIGH_M * SHANK_M)
    knee = -math.acos(max(-1.0, min(1.0, cos_b)))
    psi = math.atan2(-x, -z)
    gamma = math.atan2(SHANK_M * math.sin(knee), THIGH_M + SHANK_M * math.cos(knee))
    return psi - gamma, knee


def leg_fk(hip: float, knee: float) -> tuple[float, float]:
    ux, uz = -math.sin(hip), -math.cos(hip)
    vx, vz = -math.sin(hip + knee), -math.cos(hip + knee)
    return THIGH_M * ux + SHANK_M * vx, THIGH_M * uz + SHANK_M * vz


class DogTrot(Locomotor):
    """Diagonal pairs alternate at a fixed period; a stance foot slides back
    at the body speed, a swing foot lifts and comes forward; turning is a
    stride difference between the left and right legs. Hip roll stays at
    its rest; nothing but the twelve leg servos moves the body."""

    provenance = "analytic, hand-authored: trot gait over an analytic leg IK, position servos"
    max_speed_mps = 0.35
    min_speed_mps = 0.16
    """Below this the strides are a few centimetres and the trot marks time."""
    max_turn_radps = 0.8
    PERIOD_S = 0.44
    DUTY = 0.62
    LIFT_M = 0.05
    FOOT_Z_M = -0.335
    TRACK_M = 0.32
    LEGS = {"fl": 0.0, "hr": 0.0, "fr": 0.5, "hl": 0.5}
    SIDE = {"fl": 1.0, "hl": 1.0, "fr": -1.0, "hr": -1.0}

    def __init__(self, body: MobileBody, model: mujoco.MjModel) -> None:
        super().__init__(body, model)
        self.phase = 0.0
        self.last_time = None
        self.moving = False

    def reset(self, data: mujoco.MjData) -> None:
        self.phase = 0.0
        self.last_time = float(data.time)
        self.moving = False
        self._v, self._omega = 0.0, 0.0

    def control(self, data: mujoco.MjData, v: float, omega: float) -> np.ndarray:
        now = float(data.time)
        dt = 0.0 if self.last_time is None else now - self.last_time
        self.last_time = now
        v = float(np.clip(v, -self.max_speed_mps, self.max_speed_mps))
        omega = float(np.clip(omega, -self.max_turn_radps, self.max_turn_radps))
        v, omega = self.smooth(v, omega, dt)
        want = abs(v) > 0.01 or abs(omega) > 0.02
        if want:
            self.moving = True
            self.phase = (self.phase + dt / self.PERIOD_S) % 1.0
        elif self.moving:
            # finish the cycle so every foot is down before standing still
            self.phase = (self.phase + dt / self.PERIOD_S) % 1.0
            if self.phase < dt / self.PERIOD_S * 1.5:
                self.moving = False
                self.phase = 0.0
        control = self.hold()
        # turning in place, the stride difference is half again as large: strides of two centimetres mostly slip, and the turn would take a full minute
        turn_gain = 1.5 - 0.5 * min(1.0, abs(v) / 0.1)
        for leg, offset in self.LEGS.items():
            side = self.SIDE[leg]
            v_leg = v - omega * side * self.TRACK_M / 2.0 * turn_gain
            stride = v_leg * self.DUTY * self.PERIOD_S
            p = (self.phase + offset) % 1.0 if self.moving else 0.0
            swing = 1.0 - self.DUTY
            if self.moving and p < swing:
                s = p / swing
                x = -stride / 2.0 + stride * s
                z = self.FOOT_Z_M + self.LIFT_M * math.sin(math.pi * s)
            elif self.moving:
                s = (p - swing) / self.DUTY
                x = stride / 2.0 - stride * s
                z = self.FOOT_Z_M
            else:
                x, z = 0.0, self.FOOT_Z_M
            hip, knee = leg_ik(x, z)
            control[self.actuator_of[f"{leg}_hip_pitch"]] = hip
            control[self.actuator_of[f"{leg}_knee"]] = float(np.clip(knee, -2.4, -0.15))
            control[self.actuator_of[f"{leg}_hip_roll"]] = 0.0
        return control


# --------------------------------------------------------------------------------------------------
# the wheeled biped: a pitch balance on the wheels
# --------------------------------------------------------------------------------------------------


class WheeledBalance(Locomotor):
    """The legs hold the standing stance; the wheels balance the torso as an
    inverted pendulum. The wheel servos are velocity servos, so a torque is
    commanded through them by feeding the measured wheel speed forward: the
    servo's force is kv times the difference, which is the torque asked for
    until it saturates. The torque law is a pitch and pitch-rate regulator
    about a desired lean; the lean is how speed is changed (lean forward to
    go, lean back to stop), and a slow integral finds the pitch at which the
    body's mass, arm included, sits over the axle. Turn rate is a torque
    difference between the wheels, closed on the gyro's yaw rate. The leg
    servos are undamped springs in the model, and the wheel torque reacts
    through the knee and the hip, so the legs are held with a feed-forward
    of that torque (target plus torque over kp, which keeps the joint on
    its target while the spring carries the load) and damped through the
    same servos; without both, the legs flex in series with the pitch loop
    and the balance rings against the torque limit."""

    provenance = "analytic, hand-authored: wheeled inverted-pendulum balance (pitch regulator about a speed-commanding lean) through velocity servos on the wheels, legs held"
    max_speed_mps = 0.4
    min_speed_mps = 0.05
    max_turn_radps = 0.8
    WHEEL_R_M = 0.08
    TRACK_M = 0.52
    KV = 1.5
    """The wheel servo's velocity gain (N m s / rad), from the model: a torque tau is asked for by commanding the measured speed plus tau / KV."""
    TORQUE_MAX = 12.0
    K_PITCH = 80.0
    K_RATE = 10.0
    K_LEAN = 0.28
    LEAN_MAX = 0.16
    K_INT = 0.06
    INT_GATE_MPS = 0.2
    K_FEEDFORWARD = 0.8
    """Fraction of the acceleration lean (atan(a / g)) fed forward from the eased command."""
    K_TURN = 1.0
    """Turn torque per rad/s of yaw-rate error, closed on the gyro: an open differential would spin the body up without limit."""
    KD_LEG = 6.0
    """Leg damping (N m s / rad) applied through the position servos as a target offset of -KD_LEG / kp times the joint speed."""

    def __init__(self, body: MobileBody, model: mujoco.MjModel) -> None:
        super().__init__(body, model)
        self.gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_gyro")
        self.gyro_adr = int(model.sensor_adr[self.gyro]) if self.gyro >= 0 else None
        self.wheel_dof = {side: int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wheel_spin")]) for side in ("left", "right")}
        self.wheel_body = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wheel") for side in ("left", "right")}
        self.leg_joints = {side: [f"{side}_hip_pitch", f"{side}_knee"] for side in ("left", "right")}
        self.kp_of = {joint: float(model.actuator_gainprm[self.actuator_of[joint]][0]) for side in self.leg_joints for joint in self.leg_joints[side]}
        self.pitch_offset = 0.0
        self.last_time = None

    def axle_velocity(self, data: mujoco.MjData) -> np.ndarray:
        """World velocity of the axle (the mean of the two wheel centres): the speed the balance regulates. The torso's own
        velocity carries the pitch rate times its height above the axle, which would feed the pitch loop back into the speed loop."""

        velocity = np.zeros(6)
        total = np.zeros(3)
        for side in ("left", "right"):
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.wheel_body[side], velocity, 0)
            total += velocity[3:]
        return total / 2.0

    def equilibrium_pitch(self, data: mujoco.MjData) -> float:
        """The pitch at which the whole body's centre of mass stands over the wheel axle, read off the model in its current joint pose."""

        mujoco.mj_forward(self.model, data)
        com = np.array(data.subtree_com[self.base], dtype=float)
        axle = np.mean([data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wheel")] for side in ("left", "right")], axis=0)
        state = self.base_state(data)
        forward = np.array([math.cos(state.yaw), math.sin(state.yaw), 0.0])
        ahead = float(np.dot(com - axle, forward))
        above = float(com[2] - axle[2])
        # the body is pitched by state.pitch already; the offset is the extra lean that would put the mass over the axle
        return float(state.pitch - math.atan2(ahead, max(above, 1e-3)))

    def reset(self, data: mujoco.MjData) -> None:
        self.pitch_offset = float(np.clip(self.equilibrium_pitch(data), -0.3, 0.3))
        self.last_time = float(data.time)
        self._v, self._omega = 0.0, 0.0

    def control(self, data: mujoco.MjData, v: float, omega: float) -> np.ndarray:
        now = float(data.time)
        dt = 0.0 if self.last_time is None else now - self.last_time
        self.last_time = now
        v = float(np.clip(v, -self.max_speed_mps, self.max_speed_mps))
        omega = float(np.clip(omega, -self.max_turn_radps, self.max_turn_radps))
        previous = float(self._v) if hasattr(self, "_v") else 0.0
        v, omega = self.smooth(v, omega, dt, tau_s=0.6)
        acceleration = 0.0 if dt <= 0.0 else (v - previous) / dt
        state = self.base_state(data)
        pitch = state.pitch
        rate = float(data.sensordata[self.gyro_adr + 1]) if self.gyro_adr is not None else float(state.angular[1])
        forward = np.array([math.cos(state.yaw), math.sin(state.yaw)])
        speed = float(np.dot(self.axle_velocity(data)[:2], forward))
        # lean forward to gain speed, back to lose it; and learn, slowly, where upright really is for this load
        # the lean that produces the commanded acceleration (atan(a / g)) is fed forward, so a stop begins as the command falls and not after the speed error has grown
        lean = float(np.clip(self.K_LEAN * (v - speed) + self.K_FEEDFORWARD * math.atan2(acceleration, 9.81), -self.LEAN_MAX, self.LEAN_MAX))
        if abs(v - speed) < self.INT_GATE_MPS:
            # the integral learns only near the commanded speed: during a transient (a push, a hard start) it would wind up and the speed would overshoot on the far side
            self.pitch_offset = float(np.clip(self.pitch_offset + self.K_INT * (v - speed) * dt, -0.3, 0.3))
        error = pitch - (self.pitch_offset + lean)
        torque = float(np.clip(self.K_PITCH * error + self.K_RATE * rate, -self.TORQUE_MAX, self.TORQUE_MAX))
        yaw_rate = float(data.sensordata[self.gyro_adr + 2]) if self.gyro_adr is not None else float(state.angular[2])
        turn = float(np.clip(self.K_TURN * (omega - yaw_rate), -4.0, 4.0))
        control = self.hold()
        for side, sign in (("left", -1.0), ("right", 1.0)):
            wheel_speed = float(data.qvel[self.wheel_dof[side]])
            asked = float(np.clip(torque + sign * turn, -self.TORQUE_MAX, self.TORQUE_MAX))
            control[self.actuator_of[f"{side}_wheel_spin"]] = float(np.clip(wheel_speed + asked / self.KV, -25.0, 25.0))
            # the wheel torque reacts through this leg's knee and hip: hold them on target under it, and damp them
            for joint in self.leg_joints[side]:
                kp = self.kp_of[joint]
                a = self.actuator_of[joint]
                control[a] = float(np.clip(self.stance[joint] + asked / kp - self.KD_LEG / kp * float(data.qvel[self.dof_of[joint]]), self.model.actuator_ctrlrange[a][0], self.model.actuator_ctrlrange[a][1]))
        return control


# --------------------------------------------------------------------------------------------------
# the octopus: a tripod crawl
# --------------------------------------------------------------------------------------------------


class OctopusCrawl(Locomotor):
    """Two groups of three tentacles alternate: a lifted group curls up and
    swings its tips forward while the other, pressed down, sweeps its tips
    back and pushes the mantle along. Direction comes from the sweep's
    sense on each side (a tentacle's yaw moves its tip along the body's
    axis by the sine of its mounting angle); turning is a sweep amplitude
    difference between the left and right sides."""

    provenance = "analytic, hand-authored: tripod crawl of six segmented tentacles, position servos on every joint"
    max_speed_mps = 0.12
    min_speed_mps = 0.04
    max_turn_radps = 0.5
    PERIOD_S = 1.4
    SWEEP_RAD = 0.55
    LIFT_RAD = 0.45
    PRESS_RAD = 0.18
    ANGLES = {0: math.pi / 6, 1: math.pi / 2, 2: 5 * math.pi / 6, 3: 7 * math.pi / 6, 4: 3 * math.pi / 2, 5: 11 * math.pi / 6}
    GROUP = {0: 0.0, 2: 0.0, 4: 0.0, 1: 0.5, 3: 0.5, 5: 0.5}

    def __init__(self, body: MobileBody, model: mujoco.MjModel) -> None:
        super().__init__(body, model)
        self.phase = 0.0
        self.last_time = None
        self.moving = False

    def reset(self, data: mujoco.MjData) -> None:
        self.phase = 0.0
        self.last_time = float(data.time)
        self.moving = False
        self._v, self._omega = 0.0, 0.0

    def control(self, data: mujoco.MjData, v: float, omega: float) -> np.ndarray:
        now = float(data.time)
        dt = 0.0 if self.last_time is None else now - self.last_time
        self.last_time = now
        v = float(np.clip(v, -self.max_speed_mps, self.max_speed_mps))
        omega = float(np.clip(omega, -self.max_turn_radps, self.max_turn_radps))
        v, omega = self.smooth(v, omega, dt, tau_s=0.5)
        want = abs(v) > 0.005 or abs(omega) > 0.02
        if want:
            self.moving = True
            self.phase = (self.phase + dt / self.PERIOD_S) % 1.0
        elif self.moving:
            self.phase = (self.phase + dt / self.PERIOD_S) % 1.0
            if self.phase < dt / self.PERIOD_S * 1.5:
                self.moving = False
                self.phase = 0.0
        control = self.hold()
        if not self.moving:
            return control
        gain = v / self.max_speed_mps
        turn = omega / self.max_turn_radps
        for index, angle in self.ANGLES.items():
            side = 1.0 if math.sin(angle) > 0 else -1.0  # +1 left, -1 right
            # moving: the outer side sweeps more, as a differential drive; in place: the two sides sweep in opposite senses (the sign is the opposite of the moving case, where a larger sweep on the right turns the body left)
            amplitude = self.SWEEP_RAD * (gain * (1.0 - 0.8 * turn * side) - 0.8 * turn * side * (1.0 if abs(gain) < 0.05 else 0.0))
            p = (self.phase + self.GROUP[index]) % 1.0
            prefix = f"t{index}_"
            if p < 0.5:
                # lifted: curl up, swing the tip forward (yaw sense: -sign(sin angle) moves the tip towards +x)
                s = p / 0.5
                yaw = -side * amplitude * (-1.0 + 2.0 * s)
                pitch0 = self.stance[f"{prefix}pitch_0"] - self.LIFT_RAD * math.sin(math.pi * s)
                curl = -0.35 * math.sin(math.pi * s)
            else:
                # pressed: sweep the tip back, pushing the mantle forward
                s = (p - 0.5) / 0.5
                yaw = -side * amplitude * (1.0 - 2.0 * s)
                pitch0 = self.stance[f"{prefix}pitch_0"] + self.PRESS_RAD
                curl = 0.0
            control[self.actuator_of[f"{prefix}yaw_0"]] = float(np.clip(self.stance[f"{prefix}yaw_0"] + yaw, -1.2, 1.2))
            control[self.actuator_of[f"{prefix}pitch_0"]] = float(np.clip(pitch0, -1.4, 1.4))
            for seg in (1, 2, 3):
                control[self.actuator_of[f"{prefix}pitch_{seg}"]] = float(np.clip(self.stance[f"{prefix}pitch_{seg}"] + curl, -1.4, 1.4))
        return control


def make_locomotor(body: MobileBody, model: mujoco.MjModel) -> Locomotor:
    kind = body.declaration["base_kind"]
    if kind == "legged":
        return DogTrot(body, model)
    if kind == "wheeled":
        return WheeledBalance(body, model)
    if kind == "crawling":
        return OctopusCrawl(body, model)
    raise ValueError(f"no locomotor for base kind {kind!r}")


# --------------------------------------------------------------------------------------------------
# the navigator: waypoints, the same for every body
# --------------------------------------------------------------------------------------------------


@dataclass
class Navigator:
    """Steer the base through waypoints: turn towards the next, drive when
    facing it, stop inside the radius and hold for the dwell, then the
    next. Knows nothing of legs, wheels or tentacles."""

    waypoints: list[tuple[float, float]]
    radius_m: float = 0.3
    release_m: float = 0.45
    """Once inside the radius the body counts as arrived until it drifts past this: a gait settling at the edge does not lose the waypoint."""
    dwell_s: float = 2.0
    k_turn: float = 1.5
    facing_rad: float = math.radians(35.0)
    slow_m: float = 0.8
    """The run-in over which the speed eases off towards the waypoint's edge, so a body whose speed lags its command (a balancing one) arrives at a crawl and stops short of what stands behind the waypoint."""
    index: int = 0
    arrived_at: float | None = None
    turn_sign: float = 0.0
    """the direction committed to while the target is behind: the wrapped heading error flips sign across the seam, the body should not"""
    log: list[dict] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.index >= len(self.waypoints)

    def command(self, state: BaseState, now: float, locomotor: Locomotor) -> tuple[float, float]:
        if self.done:
            return 0.0, 0.0
        target = np.array(self.waypoints[self.index])
        delta = target - state.position[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= self.radius_m or (self.arrived_at is not None and distance <= self.release_m):
            if self.arrived_at is None:
                self.arrived_at = now
                self.log.append({"event": "arrived", "waypoint": self.index, "time_s": now, "distance_m": distance})
            if now - self.arrived_at >= self.dwell_s:
                self.log.append({"event": "released", "waypoint": self.index, "time_s": now, "distance_m": distance})
                self.index += 1
                self.arrived_at = None
                self.turn_sign = 0.0
            return 0.0, 0.0
        if self.arrived_at is not None:
            self.log.append({"event": "drifted", "waypoint": self.index, "time_s": now, "distance_m": distance})
        self.arrived_at = None
        heading = math.atan2(float(delta[1]), float(delta[0]))
        error = wrap(heading - state.yaw)
        if abs(error) > math.pi - 0.35 and self.turn_sign != 0.0:
            error = self.turn_sign * abs(error)
        self.turn_sign = math.copysign(1.0, error) if abs(error) > 0.05 else 0.0
        omega = float(np.clip(self.k_turn * error, -locomotor.max_turn_radps, locomotor.max_turn_radps))
        if abs(error) > self.facing_rad:
            return 0.0, omega
        speed = locomotor.max_speed_mps * min(1.0, (distance - self.radius_m) / self.slow_m) * max(0.2, math.cos(error))
        return max(locomotor.min_speed_mps, speed), omega


__all__ = ["BaseState", "DogTrot", "Locomotor", "Navigator", "OctopusCrawl", "WheeledBalance", "leg_fk", "leg_ik", "make_locomotor", "quat_to_euler", "wrap"]
