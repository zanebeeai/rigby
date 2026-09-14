"""Retrieve and deliver on a mobile body: approach the object, stabilise, acquire it, carry it, place it, return; one contract, three bodies.

The task is the course's retrieve goal. Its phases are the same for a
walker, a roller and a crawler, and so is the rule that scores them:
the cube at rest inside the tray's rim, the body back on the start pad
and stable, no fall. What differs per body is read from its declaration
and manifest: which limb manipulates, where the body has to stand to
reach (its grasp pose beside the station and the tray, found by the
manipulator's own reach), and which limbs bear it while it carries.
Every step is recorded; the root is placed once at the start and never
written again; the object is never written. A disturbance is a push on
the base, a force on the held object, or a limb fought, applied through
the physics step as recorded user input.

The resource schedule is explicit: each phase names the limbs it uses
for support and the limb it uses to hold. A limb holding the object is
excluded from the drive for as long as it holds (the octopus crawls on
five tentacles); the invariant that a holding limb bears no ground
contact is checked from the recorded contacts at every step, and a
violation is a fault the trial reports.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from rigby_core.simulation.recording import PhysicsRecorder

from .ingest import MobileBody
from .locomotion import Locomotor, Navigator, make_locomotor, quat_to_euler, wrap
from .manipulation import Manipulator, manipulators_of
from .trials import FALL_TILT_DEG, STABLE_SPEED_MPS, STABLE_WINDOW_S, course_world_xml
from .validate import Run, floor_contacts, place, robot_bodies
from .world import Course


PHASES = ("approach", "stabilize", "acquire", "carry", "place", "return")
CUBE_BODY = "course_cube"
TRAY_CENTRE = (3.4, 2.2)
TRAY_FLOOR_Z = 0.07
TRAY_INNER_HALF = 0.19
RIM_TOP_Z = 0.11
START = (0.0, 0.0)
STATION_CUBE = (3.4, 0.0, 0.075)


@dataclass(frozen=True)
class RetrieveDisturbance:
    kind: str
    """push (on the base) | object (a force on the held object) | support (a limb fought)"""
    detail: str
    phase: str
    """the phase during which it lands"""
    offset_s: float
    duration_s: float
    magnitude: float
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    target: str = ""

    def as_json(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "phase": self.phase, "offset_s": self.offset_s, "duration_s": self.duration_s, "magnitude": self.magnitude, "direction": list(self.direction), "target": self.target}


@dataclass
class GraspPoses:
    """Where a body stands to reach the station and the tray: waypoints ending beside each, facing along its edge, and the creep that brings the target into reach."""

    station_waypoints: list[tuple[float, float]]
    tray_waypoints: list[tuple[float, float]]
    return_waypoints: list[tuple[float, float]]
    grasp_radius_m: float
    """how close the base creeps to the object before it settles to reach"""
    tray_radius_m: float
    """how close the base creeps to the tray's centre before it settles to place: closer than at the station, so the arm comes down over the rim rather than resting on it"""
    manipulation_stance: str | None
    """the stance the body settles into to reach (None: it reaches from its working stance)"""


def grasp_poses(body: MobileBody) -> GraspPoses:
    """Body-specific approach routes derived from the base kind and the manipulator's reach: a walker stands beside the platform
    with its feet clear of it and lies down to reach, a crawler faces it from its reach distance, a roller parks beside it."""

    kind = body.declaration["base_kind"]
    stances = body.declaration["stances"]
    holdable = next((n for n, v in stances.items() if v["statically_stable"] and n != body.declaration["working_stance"]), None)
    if kind == "crawling":
        # the crawler faces each target from its reach distance; its front tentacles' tips stop short of the platform's sides; it returns over the ramp, which it climbs
        return GraspPoses(station_waypoints=[(2.0, 0.0), (2.85, 0.0)], tray_waypoints=[(2.3, 2.2), (2.85, 2.2)], return_waypoints=[(1.2, 2.2), (0.0, 0.0)], grasp_radius_m=0.58, tray_radius_m=0.58, manipulation_stance=None)
    if kind == "wheeled":
        # the roller stays on its balance to reach: parked, its wheels stand a third of a metre ahead of the torso and the arm's mount is higher than its reach allows either way
        return GraspPoses(station_waypoints=[(2.0, 0.0), (2.7, 0.0)], tray_waypoints=[(3.4, 1.2), (3.4, 1.7)], return_waypoints=[(2.6, 1.2), (0.0, 0.0)], grasp_radius_m=0.52, tray_radius_m=0.46, manipulation_stance=None)
    # the walker faces each target head-on and stops half a metre from it, its front feet a hand short of the platform, and reaches standing:
    # its arm mount rides a fifth of a metre ahead of the torso, and the second version's arm reaches the cube and the tray floor from there,
    # coming down steeply enough to clear the tray's rim; it returns by the flat route, its trot having no gait for the ramp
    return GraspPoses(station_waypoints=[(2.0, 0.0), (2.7, 0.0)], tray_waypoints=[(3.4, 1.2), (3.4, 1.7)], return_waypoints=[(2.6, 1.2), (0.0, 0.0)], grasp_radius_m=0.52, tray_radius_m=0.47, manipulation_stance=holdable)


@dataclass
class PhaseRecord:
    phase: str
    started_s: float
    ended_s: float = 0.0
    success: bool = False
    reason: str = ""
    attempts: int = 1
    support: tuple[str, ...] = ()
    holding: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class RetrieveResult:
    success: bool
    reason: str
    phases: list[PhaseRecord]
    fell: bool
    fall_time_s: float | None
    duration_s: float
    cap_s: float
    energy_j: float
    distance_m: float
    collisions: int
    cube_in_tray: bool
    cube_final: list[float]
    at_start: bool
    stable_at_end: bool
    hold_events: list[dict]
    invariant_violations: list[dict]
    schedule: list[dict]
    disturbance_log: list[dict]
    control_latency_s: dict
    contacts_at_end: tuple[str, ...]
    run: Run
    samples: int
    actuation: dict
    support: dict
    max_object_slip_m: float
    """the largest offset of the held object from the grasp site while carrying"""


class RetrieveSession:
    """One retrieve trial on physics: the body in the course with the cube on the station."""

    GRASP_TOLERANCE_M = 0.012
    """How far the grasp site may miss the grasp point: within the jaw's clearance around a 30 mm cube."""

    def __init__(self, body: MobileBody, course: Course, *, seed: int, cap_s: float, jitter_xy_m: float = 0.10, jitter_yaw_deg: float = 10.0, disturbance: RetrieveDisturbance | None = None, settle_s: float = 1.5, retry_budget: int = 2) -> None:
        self.body = body
        self.course = course
        self.cap_s = cap_s
        self.settle_s = settle_s
        self.retry_budget = retry_budget
        self.disturbance = disturbance
        rng = np.random.default_rng(seed)
        self.world_xml = course_world_xml(body, course, None)
        self.model = mujoco.MjSpec.from_string(self.world_xml).compile()
        self.data = mujoco.MjData(self.model)
        self.locomotor: Locomotor = make_locomotor(body, self.model)
        self.manipulator: Manipulator = manipulators_of(body, self.model)[0]
        stance = body.declaration["stances"][self.locomotor.working_stance()]["joints"]
        place(self.model, self.data, body, stance, yaw_deg=float(rng.uniform(-jitter_yaw_deg, jitter_yaw_deg)))
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, body.declaration["root_joint"])
        self.root_qpos = int(self.model.jnt_qposadr[root])
        self.root_dof = int(self.model.jnt_dofadr[root])
        self.data.qpos[self.root_qpos] += float(rng.uniform(-jitter_xy_m, jitter_xy_m))
        self.data.qpos[self.root_qpos + 1] += float(rng.uniform(-jitter_xy_m, jitter_xy_m))
        mujoco.mj_forward(self.model, self.data)
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body.declaration["base_body"])
        self.robot = robot_bodies(self.model, body.declaration["base_body"])
        self.cube = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
        self.cube_dof = int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{CUBE_BODY}_free")])
        self.hazards = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("course_corridor_left", "course_corridor_right", "course_station", "course_tray_plinth", "course_tray", "course_tray_rim_n", "course_tray_rim_s", "course_tray_rim_e", "course_tray_rim_w")}
        self.hazards.discard(-1)
        self.arm_bodies = {b for b in self.robot if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("arm_", "torso", "mantle"))} | {self.base}
        self.arm_bodies -= set(self.manipulator.finger_bodies) | {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "arm_palm")}
        self.limb_bodies = {limb: self._limb_bodies(limb) for limb in body.declaration["limbs"]}
        self.recorder = PhysicsRecorder(self.model)
        self.locomotor.reset(self.data)
        self.arm_targets: dict[str, float] = {j: stance[j] for j in self.manipulator.joints if j in stance}
        self.grip_targets: dict[str, float] = self.manipulator.open_targets()
        self.holding_limb = ""
        self.phase = "settle"
        self.phase_started = 0.0
        self.phases: list[PhaseRecord] = []
        self.schedule: list[dict] = []
        self.hold_events: list[dict] = []
        self.violations: list[dict] = []
        self.disturbance_log: list[dict] = []
        self.energy = 0.0
        self.collisions = 0
        self.fell = False
        self.fall_time = None
        self.positions: list[np.ndarray] = []
        self.latencies: list[float] = []
        self.speeds: list[float] = []
        self.window = int(STABLE_WINDOW_S / self.model.opt.timestep)
        self.max_slip = 0.0
        self.step_count = 0
        limit = np.where(self.model.actuator_forcelimited.astype(bool), np.abs(self.model.actuator_forcerange).max(axis=1), np.inf)
        self.force_limit = limit
        self.saturated = np.zeros(self.model.nu, dtype=np.int64)
        self.peak_force = np.zeros(self.model.nu)
        self.contact_counts: dict[str, int] = {}
        self.contact_samples = 0
        self.poses = grasp_poses(body)
        self.support_dofs: list[int] = []
        if disturbance is not None and disturbance.kind == "support" and disturbance.target in body.declaration["limbs"]:
            self.support_dofs = [int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in body.declaration["limbs"][disturbance.target]]
        self.disturbance_started: float | None = None
        self.debug_moves: list | None = None
        self.unheld_steps = 0

    # -- bookkeeping -----------------------------------------------------------------------------------
    def _limb_bodies(self, limb: str) -> set[int]:
        bodies = set()
        for joint in self.body.declaration["limbs"][limb]:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if j >= 0:
                bodies.add(int(self.model.jnt_bodyid[j]))
        # every body below the limb's joints
        grown = True
        while grown:
            grown = False
            for b in range(self.model.nbody):
                if b not in bodies and int(self.model.body_parentid[b]) in bodies:
                    bodies.add(b)
                    grown = True
        return bodies

    @property
    def now(self) -> float:
        return float(self.data.time)

    def base_xy(self) -> np.ndarray:
        return np.array(self.data.xpos[self.base][:2], dtype=float)

    def tilt_deg(self) -> float:
        up = np.zeros(3)
        mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), self.data.xquat[self.base])
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def support_set(self) -> tuple[str, ...]:
        declared = self.body.declaration["support_members"]
        if not self.holding_limb:
            return tuple(declared)
        excluded = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.limb_bodies[self.holding_limb]}
        return tuple(m for m in declared if m not in excluded)

    def begin(self, phase: str, attempt: int = 1) -> PhaseRecord:
        self.phase = phase
        self.phase_started = self.now
        record = PhaseRecord(phase=phase, started_s=self.now, attempts=attempt, support=self.support_set(), holding=self.holding_limb)
        self.phases.append(record)
        self.schedule.append({"phase": phase, "attempt": attempt, "from_s": self.now, "support": list(record.support), "holding": self.holding_limb or None, "drive_excludes": sorted(self.locomotor.excluded_limbs)})
        return record

    def end(self, record: PhaseRecord, success: bool, reason: str = "", **detail) -> bool:
        record.ended_s = self.now
        record.success = success
        record.reason = reason
        record.detail.update(detail)
        self.schedule[-1]["to_s"] = self.now
        self.schedule[-1]["success"] = success
        return success

    # -- the physics step ------------------------------------------------------------------------------
    ARM_DAMPING = 0.03
    """Damping through the manipulator's position servos, as a target offset of -ARM_DAMPING times the joint speed (kd = ARM_DAMPING * kp): the
    model's limbs are undamped springs, and a tentacle or an arm moved in small steps would otherwise ring."""

    def control(self, v: float, omega: float) -> np.ndarray:
        control = self.locomotor.control(self.data, v, omega)
        for joint, value in self.arm_targets.items():
            a = self.manipulator.actuator_of.get(joint)
            if a is not None:
                dof = int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)])
                damped = value - self.ARM_DAMPING * float(self.data.qvel[dof])
                control[a] = float(np.clip(damped, self.model.actuator_ctrlrange[a][0], self.model.actuator_ctrlrange[a][1]))
        for joint, value in self.grip_targets.items():
            a = self.manipulator.actuator_of.get(joint)
            if a is not None:
                control[a] = float(np.clip(value, self.model.actuator_ctrlrange[a][0], self.model.actuator_ctrlrange[a][1]))
        return control

    GENTLE = {"crawling": 0.6, "legged": 1.0, "wheeled": 1.0}
    """The fraction of the drive's speed and turn rate used while an object is held, per base kind: a crawler that rocks shakes the object out of its pincer; a trot below its floor marks time, so the floor is kept."""

    def step(self, v: float, omega: float) -> None:
        data = self.data
        now = self.now
        if self.holding_limb:
            gentle = self.GENTLE.get(self.body.declaration["base_kind"], 1.0)
            if gentle < 1.0:
                v = (max(self.locomotor.min_speed_mps, gentle * v) if v > 0.0 else gentle * v)
                omega = 0.6 * gentle * omega
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0
        d = self.disturbance
        if d is not None and self.disturbance_started is None and self.phase == d.phase and now - self.phase_started >= d.offset_s:
            self.disturbance_started = now
            self.disturbance_log.append({"event": "on", "time_s": now, "phase": self.phase, "kind": d.kind, "detail": d.detail})
        if d is not None and self.disturbance_started is not None and now < self.disturbance_started + d.duration_s:
            if d.kind == "push":
                data.xfrc_applied[self.base, :3] = np.array(d.direction) * d.magnitude
            elif d.kind == "object":
                data.xfrc_applied[self.cube, :3] = np.array(d.direction) * d.magnitude
            elif d.kind == "support":
                for dof in self.support_dofs:
                    data.qfrc_applied[dof] = d.magnitude * math.sin(2 * math.pi * 3.0 * (now - self.disturbance_started))
        elif d is not None and self.disturbance_started is not None and self.disturbance_log[-1]["event"] == "on":
            self.disturbance_log.append({"event": "off", "time_s": now})
        started = time.perf_counter()
        control = self.control(v, omega)
        self.latencies.append(time.perf_counter() - started)
        self.recorder.capture(data, control, control_time_s=now)
        data.ctrl[:] = control
        mujoco.mj_step(self.model, data)
        self.step_count += 1
        speed = float(np.linalg.norm(data.qvel[self.root_dof: self.root_dof + 2]))
        self.speeds.append(speed)
        if len(self.speeds) > self.window:
            self.speeds.pop(0)
        self.energy += float(np.sum(np.abs(data.qfrc_actuator * data.qvel))) * self.model.opt.timestep
        force = np.abs(data.actuator_force)
        self.saturated += (force >= 0.98 * self.force_limit).astype(np.int64)
        self.peak_force = np.maximum(self.peak_force, force)
        if self.step_count % 10 == 0:
            self.positions.append(self.base_xy())
            self.contact_samples += 1
            touching = floor_contacts(self.model, data, 0, self.robot)
            for name in touching:
                self.contact_counts[name] = self.contact_counts.get(name, 0) + 1
            if self.holding_limb:
                # the limb's links must bear no ground while it holds; its palm and fingers touch the object and the fixture it is placed on by design
                held_bodies = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.limb_bodies[self.holding_limb]} - {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.manipulator.finger_bodies} - {"arm_palm", f"{self.manipulator.limb}_palm", "t0_pincer", "t5_pincer"}
                grounded = sorted(set(touching) & held_bodies)
                if grounded:
                    self.violations.append({"time_s": self.now, "phase": self.phase, "limb": self.holding_limb, "members_on_ground": grounded})
            for i in range(data.ncon):
                c = data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in self.hazards and int(self.model.geom_bodyid[g2]) in self.arm_bodies) or (g2 in self.hazards and int(self.model.geom_bodyid[g1]) in self.arm_bodies):
                    self.collisions += 1
                    break
        if self.holding_limb:
            self.max_slip = max(self.max_slip, float(np.linalg.norm(data.xpos[self.cube] - data.site_xpos[self.manipulator.site])))
        if self.tilt_deg() > FALL_TILT_DEG and not self.fell:
            self.fell = True
            self.fall_time = self.now

    def out_of_time(self) -> bool:
        return self.now >= self.cap_s

    def still_holding(self) -> bool:
        """The hold, debounced: the fingers off the object (or the object away from the site) for half a second running means it is lost; a flicker of contact does not."""

        offset = float(np.linalg.norm(self.data.xpos[self.cube] - self.data.site_xpos[self.manipulator.site]))
        if offset < 0.06:
            # the object rides with the site: held, whatever the finger contacts say from step to step (a wedged object's contacts flicker)
            self.unheld_steps = 0
            return True
        self.unheld_steps += 1
        return self.unheld_steps < 250 and offset < 0.10

    def stable(self) -> bool:
        return len(self.speeds) == self.window and max(self.speeds) < STABLE_SPEED_MPS and self.tilt_deg() < 15.0

    # -- locomotion phases -----------------------------------------------------------------------------
    def navigate(self, waypoints: list[tuple[float, float]], *, budget_s: float) -> tuple[bool, str, list[dict]]:
        navigator = Navigator(waypoints=list(waypoints))
        deadline = self.now + budget_s
        while not self.out_of_time() and self.now < deadline:
            state = self.locomotor.base_state(self.data)
            v, omega = navigator.command(state, self.now, self.locomotor)
            self.step(v, omega)
            if self.fell:
                return False, f"fell at {self.fall_time:.1f} s", navigator.log
            if self.holding_limb and not self.still_holding():
                self.hold_events.append({"event": "lost", "time_s": self.now, "phase": self.phase})
                return False, f"hold lost at {self.now:.1f} s", navigator.log
            if navigator.done and self.stable():
                return True, "", navigator.log
        return False, "time cap" if self.out_of_time() else "phase budget", navigator.log

    def face(self, target: np.ndarray, *, budget_s: float = 20.0, within_rad: float = math.radians(8.0)) -> bool:
        """Turn in place until the base faces the target."""

        deadline = self.now + budget_s
        while not self.out_of_time() and self.now < deadline:
            state = self.locomotor.base_state(self.data)
            delta = target[:2] - state.position[:2]
            bearing = wrap(math.atan2(float(delta[1]), float(delta[0])) - state.yaw)
            if abs(bearing) < within_rad:
                self.hold_still(0.4, require_stable=False)
                return True
            omega = float(np.clip(1.5 * bearing, -self.locomotor.max_turn_radps, self.locomotor.max_turn_radps))
            for _ in range(25):
                self.step(0.0, omega)
            if self.fell:
                return False
        return False

    def creep_into_reach(self, target: np.ndarray, *, budget_s: float, radius_m: float | None = None, approach_from: tuple[float, float] | None = None) -> tuple[bool, str, float]:
        """Bring the base to the stand-off point the given radius from the target on the approach axis (from the last route waypoint by default),
        then face the target: the navigator does the bringing, to a tight radius, and a creep closes any gap left; the body ends on the fixture's
        axis with its feet clear of the fixture's corners before it settles to reach."""

        radius = self.poses.grasp_radius_m if radius_m is None else radius_m
        origin = np.array(approach_from if approach_from is not None else self.base_xy(), dtype=float)
        direction = target[:2] - origin
        direction = direction / max(1e-9, float(np.linalg.norm(direction)))
        stand_off = target[:2] - radius * direction
        deadline = self.now + budget_s
        # drive at a point beyond the stand-off on the axis (a heading reference that stays ahead), and stop on crossing the stand-off line
        beyond = stand_off + 0.4 * direction
        navigator = Navigator(waypoints=[(float(beyond[0]), float(beyond[1]))], radius_m=0.05, release_m=0.15, dwell_s=0.5, slow_m=0.5)
        crossed = False
        while not self.out_of_time() and self.now < deadline:
            state = self.locomotor.base_state(self.data)
            along = float(np.dot(state.position[:2] - stand_off, direction))
            if along >= -0.02:
                crossed = True
                break
            v, omega = navigator.command(state, self.now, self.locomotor)
            if navigator.done:
                break
            self.step(v, omega)
            if self.fell:
                return False, f"fell at {self.fall_time:.1f} s", float(np.linalg.norm(target[:2] - self.base_xy()))
        if not crossed:
            return False, ("time cap" if self.out_of_time() else f"could not reach the stand-off line ({float(np.linalg.norm(stand_off - self.base_xy())):.2f} m off)"), float(np.linalg.norm(target[:2] - self.base_xy()))
        for _ in range(3):
            self.hold_still(0.6, require_stable=False)
            bearing = wrap(math.atan2(float(target[1] - self.base_xy()[1]), float(target[0] - self.base_xy()[0])) - self.locomotor.base_state(self.data).yaw)
            if abs(bearing) > math.radians(10.0) and not self.face(target):
                return False, "could not turn to face the target", float(np.linalg.norm(target[:2] - self.base_xy()))
            distance = float(np.linalg.norm(target[:2] - self.base_xy()))
            if distance <= radius + 0.06:
                return True, "", distance
            # a gap is left (a body that drifts back turning in place): creep straight in
            creep_deadline = self.now + 15.0
            while not self.out_of_time() and self.now < creep_deadline:
                state = self.locomotor.base_state(self.data)
                distance = float(np.linalg.norm(target[:2] - state.position[:2]))
                if distance <= radius + 0.03:
                    break
                for _ in range(50):
                    self.step(self.locomotor.min_speed_mps, 0.0)
                if self.fell:
                    return False, f"fell at {self.fall_time:.1f} s", distance
        distance = float(np.linalg.norm(target[:2] - self.base_xy()))
        return distance <= radius + 0.08, "" if distance <= radius + 0.08 else f"stands {distance:.2f} m from the target, beyond its reach of {radius + 0.08:.2f} m", distance

    def settle_into(self, stance: str | None, seconds: float = 1.5) -> None:
        """Ease the body's joints into a stance and hold it there: lying down to reach, or rising to carry."""

        if stance is None:
            self.locomotor.posture = None
            return
        targets = dict(self.body.declaration["stances"][stance]["joints"])
        for joint in list(targets):
            if joint in self.manipulator.joints or joint in self.manipulator.grip_joints:
                del targets[joint]
        start = {j: float(self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]]) for j in targets}
        if self.body.declaration["base_kind"] == "wheeled" and self.locomotor.posture is None:
            # a balancing body sits down with its balance running: the legs ease to the parked angles while the wheels keep the axle under the
            # mass, until the tail skid touches; only then does the posture take over and the drives brake
            held = dict(self.locomotor.stance)
            steps = max(1, int(3.0 / self.model.opt.timestep))
            for k in range(1, steps + 1):
                f = k / steps
                for j in targets:
                    if j in self.locomotor.stance:
                        self.locomotor.stance[j] = start[j] + f * (targets[j] - start[j])
                self.step(0.0, 0.0)
                if self.fell or self.out_of_time():
                    self.locomotor.stance = held
                    return
                if self.base in {int(self.model.geom_bodyid[int(self.data.contact[i].geom1)]) for i in range(self.data.ncon)} | {int(self.model.geom_bodyid[int(self.data.contact[i].geom2)]) for i in range(self.data.ncon)}:
                    break
            self.locomotor.stance = held
            self.locomotor.posture = dict(targets)
            return
        steps = max(1, int(seconds / self.model.opt.timestep))
        for k in range(1, steps + 1):
            f = k / steps
            self.locomotor.posture = {j: start[j] + f * (targets[j] - start[j]) for j in targets}
            self.step(0.0, 0.0)
            if self.fell or self.out_of_time():
                return
        self.locomotor.posture = dict(targets)

    def rise(self) -> bool:
        """Back to the working stance with the drive in charge again."""

        working = self.locomotor.working_stance()
        if self.locomotor.posture is None:
            return True
        self.settle_into(working, seconds=1.5)
        self.locomotor.posture = None
        self.locomotor.reset(self.data)
        self.hold_still(1.5, require_stable=False)
        return not self.fell

    def hold_still(self, seconds: float, *, require_stable: bool = True) -> bool:
        deadline = self.now + seconds
        while self.now < deadline and not self.out_of_time():
            self.step(0.0, 0.0)
            if self.fell:
                return False
        return self.stable() if require_stable else True

    # -- manipulation ----------------------------------------------------------------------------------
    def move_arm(self, targets: dict[str, float], seconds: float) -> None:
        """Position-servo targets eased from where they are to `targets` over `seconds`, the base holding."""

        start = dict(self.arm_targets)
        steps = max(1, int(seconds / self.model.opt.timestep))
        for k in range(1, steps + 1):
            s = k / steps
            self.arm_targets = {j: start.get(j, targets[j]) + s * (targets[j] - start.get(j, targets[j])) for j in targets}
            self.step(0.0, 0.0)
            if self.fell or self.out_of_time():
                return

    def reach(self, target: np.ndarray, seconds: float, *, down: float, correct: bool = True) -> tuple[bool, float]:
        """Solve, move the servos there, then measure where the site actually came to rest and correct once for the servos' sag."""

        solution = self.manipulator.solve(self.data, target, seed=self.arm_targets, down=down)
        self.move_arm(solution.joints, seconds)
        if solution.residual_m > 0.03 or not correct:
            return solution.reached, solution.residual_m
        actual = self.manipulator.site_position(self.data)
        error = np.asarray(target, dtype=float) - actual
        if float(np.linalg.norm(error)) > 0.004:
            corrected = self.manipulator.solve(self.data, np.asarray(target, dtype=float) + error, seed=self.arm_targets, down=down)
            if corrected.residual_m < 0.03:
                self.move_arm(corrected.joints, max(0.4, seconds * 0.4))
        measured = float(np.linalg.norm(np.asarray(target, dtype=float) - self.manipulator.site_position(self.data)))
        return measured < self.GRASP_TOLERANCE_M, measured

    def lift_clear(self, height: float) -> None:
        """Raise the grasp site straight up to `height` (or 8 cm above where it is, whichever is higher) before moving over an object."""

        if self.body.declaration["base_kind"] == "crawling":
            # a tentacle lying on the floor is raised whole first (its segments would drag if only the tip rose): the root pitched up, the rest straight
            raised = {j: 0.0 for j in self.manipulator.joints}
            raised[self.manipulator.joints[1]] = -0.9
            self.move_arm(raised, 1.5)
            self.hold_still(0.5, require_stable=False)
            return
        site = self.manipulator.site_position(self.data)
        self.move_site(np.array([site[0], site[1], max(height, site[2] + 0.08)]), steps=40)

    def object_touches(self, body: int) -> bool:
        """Whether the object touches `body` now."""

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = int(self.model.geom_bodyid[int(c.geom1)]), int(self.model.geom_bodyid[int(c.geom2)])
            if (b1 == self.cube and b2 == body) or (b2 == self.cube and b1 == body):
                return True
        return False

    def limb_touches(self, body: int) -> bool:
        """Whether the manipulating limb, fingers aside, touches `body` now: a link, palm or pincer body coming down on the object is the descent's end; the fingers are meant to brush it."""

        limb = self.limb_bodies[self.manipulator.limb] - set(self.manipulator.finger_bodies)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = int(self.model.geom_bodyid[int(c.geom1)]), int(self.model.geom_bodyid[int(c.geom2)])
            if (b1 in limb and b2 == body) or (b2 in limb and b1 == body):
                return True
        return False

    def move_site(self, target: np.ndarray, *, step_m: float = 0.02, seconds_per_step: float = 0.25, steps: int = 30, down: float = 0.0, stop_on_contact_with: int | None = None, want: np.ndarray | None = None) -> float:
        """Move the grasp site along a straight line to `target` in small closed-loop steps on the limb's current branch: each step aims at the
        next point on the line plus the servos' measured miss (capped), so the site neither sweeps sideways nor droops into what lies under it."""

        target = np.asarray(target, dtype=float)
        stalled = 0
        for _ in range(steps):
            site = self.manipulator.site_position(self.data)
            remaining = target - site
            distance = float(np.linalg.norm(remaining))
            if distance < 0.004 or self.fell or self.out_of_time():
                break
            point = site + remaining * min(1.0, step_m / distance)
            commanded = self.manipulator.site_position_of(self.data, self.arm_targets)
            sag = commanded - site
            norm = float(np.linalg.norm(sag))
            if norm > 0.06:
                sag = sag * (0.06 / norm)
            solution = self.manipulator.solve(self.data, point + sag, seed=self.arm_targets, down=down, single_start=True, want=want)
            self.move_arm(solution.joints, seconds_per_step)
            moved = float(np.linalg.norm(self.manipulator.site_position(self.data) - site))
            if stop_on_contact_with is not None and self.limb_touches(stop_on_contact_with):
                break
            if self.debug_moves is not None:
                arm_touch = []
                for i in range(self.data.ncon):
                    c = self.data.contact[i]
                    names = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)), mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)))
                    if any(n and n.startswith(("arm_", "t0_")) for n in names):
                        arm_touch.append(names)
                self.debug_moves.append({"site": np.round(site, 4).tolist(), "aim": np.round(point, 4).tolist(), "sag": np.round(sag, 4).tolist(), "solve_cm": round(solution.residual_m * 100, 2), "reached": np.round(self.manipulator.site_position(self.data), 4).tolist(), "q": {j: round(v, 3) for j, v in solution.joints.items()}, "actual_q": {j: round(float(self.data.qpos[a]), 3) for j, a in zip(self.manipulator.joints, self.manipulator.qpos_adr)}, "cube": np.round(self.data.xpos[self.cube], 4).tolist(), "touch": arm_touch})
            # blocked: the site no longer follows the aim although the servos are asked further; pressing on would only push what lies there
            stalled = stalled + 1 if moved < 0.003 and distance > 0.01 else 0
            if stalled >= 2:
                break
        return float(np.linalg.norm(target - self.manipulator.site_position(self.data)))

    def descend(self, target: np.ndarray, *, onto: int | None = None, want: np.ndarray | None = None) -> float:
        """Straight to `target` along the approach (down for a jaw, from the side for a pincer), or until the limb first touches `onto`."""

        return self.move_site(target, step_m=0.01, seconds_per_step=0.25, steps=24, down=0.2, stop_on_contact_with=onto, want=want)

    def pincer_approach(self, target: np.ndarray) -> np.ndarray:
        """A pincer comes at its target horizontally from the tentacle's root."""

        root = np.array(self.data.xpos[int(self.model.jnt_bodyid[self.manipulator.joint_ids[0]])], dtype=float)
        direction = np.array([target[0] - root[0], target[1] - root[1], 0.0])
        direction = direction / max(1e-9, float(np.linalg.norm(direction)))
        # pitched down a little, so the pincer's body rides above the fingers and clears the platform the object stands on
        tilted = direction + np.array([0.0, 0.0, -0.35])
        return tilted / float(np.linalg.norm(tilted))

    def grip(self, close: bool, seconds: float = 0.8) -> None:
        self.grip_targets = self.manipulator.closed_targets() if close else self.manipulator.open_targets()
        self.hold_still(seconds, require_stable=False)

    def acquire(self, record: PhaseRecord) -> bool:
        cube = np.array(self.data.xpos[self.cube], dtype=float)
        down = 0.3
        offset = 0.02 if self.manipulator.slides else 0.0
        self.grip(False, 0.4)
        if self.manipulator.slides:
            want = None
            above = cube + np.array([0.0, 0.0, 0.06 + offset])
        else:
            # a pincer comes at the object from the side at its own height, the fingers open around it
            want = self.pincer_approach(cube)
            level = np.array([want[0], want[1], 0.0]) / max(1e-9, float(np.linalg.norm(want[:2])))
            above = cube - 0.10 * level + np.array([0.0, 0.0, 0.005])
        # up first, then over on a straight path: the jaw rests low in the stance and would sweep into the object on a joint-space move
        self.lift_clear(above[2] + 0.04)
        residual = self.move_site(above, down=down, want=want)
        if residual > 0.03:
            solution = self.manipulator.solve(self.data, above, seed=self.arm_targets, down=down, want=want)
            self.move_arm(solution.joints, 1.0)
            residual = float(np.linalg.norm(above - self.manipulator.site_position(self.data)))
        if residual > 0.03:
            return self.end(record, False, f"object out of reach: pre-grasp residual {residual * 100:.1f} cm", residual_m=residual)
        grasp = cube + np.array([0.0, 0.0, offset]) if self.manipulator.slides else cube - 0.01 * level + np.array([0.0, 0.0, 0.005])
        # the approach ends where the jaw meets the object: a link, palm or pincer body touching it, the fingers being meant to brush it
        self.descend(grasp, onto=self.cube, want=want)
        miss = grasp - self.manipulator.site_position(self.data)
        lateral = float(np.linalg.norm(miss[:2]))
        moved = float(np.linalg.norm(np.array(self.data.xpos[self.cube][:2]) - cube[:2]))
        if lateral > self.GRASP_TOLERANCE_M or abs(float(miss[2])) > 0.03 or moved > (0.015 if self.manipulator.slides else 0.045):
            self.reach(above, 0.8, down=down, correct=False)
            return self.end(record, False, f"grasp miss {lateral * 100:.1f} cm across, {miss[2] * 100:.1f} cm up; the object moved {moved * 100:.1f} cm", residual_m=float(np.linalg.norm(miss)), object_moved_m=moved)
        self.grip(True, 0.8)
        contacts = self.manipulator.finger_contacts(self.data, self.cube)
        if not all(contacts):
            self.grip(False, 0.3)
            return self.end(record, False, f"fingers did not both close on the object ({contacts})", contacts=list(contacts))
        self.holding_limb = self.manipulator.limb
        self.locomotor.excluded_limbs = {self.manipulator.limb}
        self.hold_events.append({"event": "acquired", "time_s": self.now, "phase": self.phase})
        lifted = self.manipulator.site_position(self.data) + np.array([0.0, 0.0, 0.15])
        self.move_site(lifted, step_m=0.01, seconds_per_step=0.2, steps=40)
        if not self.manipulator.holding(self.data, self.cube):
            self.holding_limb = ""
            self.locomotor.excluded_limbs = set()
            self.hold_events.append({"event": "lost", "time_s": self.now, "phase": self.phase})
            return self.end(record, False, "the object did not come up with the jaw")
        self.tuck()
        if not self.manipulator.holding(self.data, self.cube):
            self.holding_limb = ""
            self.locomotor.excluded_limbs = set()
            self.hold_events.append({"event": "lost", "time_s": self.now, "phase": self.phase})
            return self.end(record, False, "the object was lost while tucking")
        if not self.rise():
            return self.end(record, False, "fell rising with the object")
        if not self.manipulator.holding(self.data, self.cube):
            self.holding_limb = ""
            self.locomotor.excluded_limbs = set()
            self.hold_events.append({"event": "lost", "time_s": self.now, "phase": self.phase})
            return self.end(record, False, "the object was lost rising")
        return self.end(record, True, cube_lifted_to=float(self.data.xpos[self.cube][2]))

    def tuck(self) -> None:
        """The held object brought in front of and above the base, where it rides during the carry."""

        if self.body.declaration["base_kind"] == "crawling":
            # the tentacle folds back over the mantle: the held object rides above the mantle's centre, where the body's rocking moves it least,
            # rather than out to the side on a free-standing tentacle; nothing of the tentacle is near the floor
            joints = list(self.manipulator.joints)
            tuck = {j: 0.0 for j in joints}
            for name, value in zip(joints[1::2], (-1.0, -1.0, -0.9, -0.6)):
                tuck[name] = value
            self.move_arm(tuck, 2.0)
            self.hold_still(0.5, require_stable=False)
            # then the pincer comes down until the object rests on the mantle: pinched from above and bearing on the body, it rides on three
            # supports and cannot swing out when the mantle rocks
            for _ in range(10):
                if self.object_touches(self.base):
                    break
                site = self.manipulator.site_position(self.data)
                solution = self.manipulator.solve(self.data, site + np.array([0.0, 0.0, -0.01]), seed=self.arm_targets, down=0.0, single_start=True)
                self.move_arm(solution.joints, 0.3)
            self.hold_still(0.3, require_stable=False)
            return
        yaw = self.locomotor.base_state(self.data).yaw
        base = np.array(self.data.xpos[self.base], dtype=float)
        local = np.array([0.30, 0.0, 0.12])
        target = base + np.array([math.cos(yaw) * local[0] - math.sin(yaw) * local[1], math.sin(yaw) * local[0] + math.cos(yaw) * local[1], local[2]])
        self.move_site(target, steps=40)

    def place(self, record: PhaseRecord) -> bool:
        offset = 0.02 if self.manipulator.slides else 0.01
        target = np.array([TRAY_CENTRE[0], TRAY_CENTRE[1], TRAY_FLOOR_Z + 0.015 + offset])
        self.settle_into(self.poses.manipulation_stance)
        if not self.hold_still(2.5):
            return self.end(record, False, "fell" if self.fell else "not stable settling to place")
        want = None if self.manipulator.slides else self.pincer_approach(target)
        self.lift_clear(RIM_TOP_Z + 0.12)
        above = target + np.array([0.0, 0.0, 0.10])
        # out over the tray on the best branch the arm has (the carry pose folds the arm back over the torso; a straight-line move from
        # there keeps that branch and brings the wrist down on the rim), then a short straight correction
        solution = self.manipulator.solve(self.data, above, down=0.3, want=want)
        self.move_arm(solution.joints, 1.5)
        residual = self.move_site(above, steps=12, down=0.2, want=want)
        if residual > 0.03:
            state = self.locomotor.base_state(self.data)
            return self.end(record, False, f"tray out of reach: residual {residual * 100:.1f} cm", residual_m=residual, base=[round(float(state.position[0]), 3), round(float(state.position[1]), 3), round(math.degrees(state.yaw), 1)], distance_to_tray_m=round(float(np.linalg.norm(np.array(TRAY_CENTRE) - state.position[:2])), 3), site=[round(float(v), 3) for v in self.manipulator.site_position(self.data)], contacts=list(floor_contacts(self.model, self.data, 0, self.robot)))
        if not self.manipulator.holding(self.data, self.cube):
            self.holding_limb = ""
            self.locomotor.excluded_limbs = set()
            self.hold_events.append({"event": "lost", "time_s": self.now, "phase": self.phase})
            return self.end(record, False, "the object was lost over the tray")
        self.move_site(target, step_m=0.01, seconds_per_step=0.25, steps=40, down=0.2, want=want)
        self.grip(False, 0.6)
        self.holding_limb = ""
        self.locomotor.excluded_limbs = set()
        self.hold_events.append({"event": "released", "time_s": self.now, "phase": self.phase})
        self.move_site(target + np.array([0.0, 0.0, 0.15]))
        self.arm_targets = {j: self.body.declaration["stances"][self.locomotor.working_stance()]["joints"][j] for j in self.manipulator.joints}
        self.hold_still(1.5, require_stable=False)
        inside, detail = self.cube_in_tray()
        if not self.rise():
            return self.end(record, False, "fell rising after the placement", cube=[round(float(v), 4) for v in self.data.xpos[self.cube]])
        return self.end(record, inside, "" if inside else f"the object is not at rest inside the tray ({detail})", cube=[round(float(v), 4) for v in self.data.xpos[self.cube]])

    def cube_in_tray(self) -> tuple[bool, str]:
        p = np.array(self.data.xpos[self.cube], dtype=float)
        v = float(np.linalg.norm(self.data.qvel[self.cube_dof: self.cube_dof + 3]))
        inside = abs(p[0] - TRAY_CENTRE[0]) < TRAY_INNER_HALF and abs(p[1] - TRAY_CENTRE[1]) < TRAY_INNER_HALF and TRAY_FLOOR_Z < p[2] < TRAY_FLOOR_Z + 0.06
        return bool(inside and v < 0.02), f"at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.3f}), speed {v:.3f} m/s"

    # -- the task --------------------------------------------------------------------------------------
    def run(self) -> RetrieveResult:
        # the settle: the controller runs, no command
        while self.now < self.settle_s:
            self.step(0.0, 0.0)
        poses = self.poses
        reason = ""
        success = True
        down = 0.3

        def phase_loop(name: str, run_once) -> bool:
            for attempt in range(1, self.retry_budget + 2):
                record = self.begin(name, attempt)
                if run_once(record):
                    return True
                if self.fell or self.out_of_time() or "out of reach" in record.reason and attempt >= 2:
                    return False
            return False

        def approach(record: PhaseRecord) -> bool:
            if not self.rise():
                return self.end(record, False, "fell rising to approach")
            cube = np.array(self.data.xpos[self.cube], dtype=float)
            waypoints = list(poses.station_waypoints) if self.holding_limb == "" and abs(cube[0] - STATION_CUBE[0]) < 0.2 and abs(cube[1] - STATION_CUBE[1]) < 0.2 else [(float(cube[0]) - 0.45 * math.cos(self.locomotor.base_state(self.data).yaw), float(cube[1]) - 0.45 * math.sin(self.locomotor.base_state(self.data).yaw))]
            # run-up waypoints already passed are dropped: a retry does not walk back to the start of its run-up
            here = self.base_xy()
            while len(waypoints) > 1 and float(np.linalg.norm(np.array(waypoints[-1]) - here)) < float(np.linalg.norm(np.array(waypoints[0]) - here)) + 0.2:
                waypoints.pop(0)
            ok, why, log = self.navigate(waypoints, budget_s=0.3 * self.cap_s)
            if not ok:
                return self.end(record, False, why, navigator_log=log)
            ok, why, distance = self.creep_into_reach(cube, budget_s=40.0, approach_from=tuple(waypoints[-2]) if len(waypoints) > 1 else None)
            return self.end(record, ok, why, navigator_log=log, distance_to_object_m=distance, object_pose_source="the simulator's object state stands in for perception: the mobile bodies carry no camera")

        def stabilize(record: PhaseRecord) -> bool:
            self.settle_into(poses.manipulation_stance)
            ok = self.hold_still(2.0)
            return self.end(record, ok, "" if ok else ("fell" if self.fell else "not stable after 2 s"), stance=poses.manipulation_stance or self.locomotor.working_stance(), tilt_deg=self.tilt_deg(), contacts=list(floor_contacts(self.model, self.data, 0, self.robot)))

        def carry(record: PhaseRecord) -> bool:
            ok, why, log = self.navigate(poses.tray_waypoints, budget_s=0.5 * self.cap_s)
            if not ok:
                if "hold lost" in why:
                    self.holding_limb = ""
                    self.locomotor.excluded_limbs = set()
                return self.end(record, False, why, navigator_log=log)
            target = np.array([TRAY_CENTRE[0], TRAY_CENTRE[1], TRAY_FLOOR_Z + 0.015 + (0.02 if self.manipulator.slides else 0.01)])
            ok, why, residual = self.creep_into_reach(target, budget_s=40.0, radius_m=poses.tray_radius_m, approach_from=tuple(poses.tray_waypoints[-2]) if len(poses.tray_waypoints) > 1 else None)
            if ok and not self.manipulator.holding(self.data, self.cube):
                self.holding_limb = ""
                self.locomotor.excluded_limbs = set()
                self.hold_events.append({"event": "lost", "time_s": self.now, "phase": self.phase})
                return self.end(record, False, "hold lost while lining up on the tray", navigator_log=log)
            return self.end(record, ok, why, navigator_log=log, reach_residual_m=residual, max_slip_m=self.max_slip)

        def go_home(record: PhaseRecord) -> bool:
            ok, why, log = self.navigate(poses.return_waypoints, budget_s=self.cap_s)
            at_start = float(np.linalg.norm(self.base_xy() - np.array(START))) <= 0.45
            return self.end(record, ok and at_start, why if not ok else ("" if at_start else "not on the start pad"), navigator_log=log)

        # approach -> stabilize -> acquire, with recovery: a failed acquire or a lost hold sends the body back to the object where it is
        acquired = False
        for attempt in range(1, self.retry_budget + 2):
            if not phase_loop("approach", approach):
                reason = f"approach: {self.phases[-1].reason}"
                success = False
                break
            if not phase_loop("stabilize", stabilize):
                reason = f"stabilize: {self.phases[-1].reason}"
                success = False
                break
            record = self.begin("acquire", attempt)
            if self.acquire(record):
                acquired = True
                break
            if self.fell or self.out_of_time() or "out of reach" in record.reason:
                reason = f"acquire: {record.reason}"
                success = False
                break
        if success and not acquired:
            success, reason = False, f"acquire: {self.phases[-1].reason}"
        delivered = False
        if success:
            for attempt in range(1, self.retry_budget + 2):
                record = self.begin("carry", attempt)
                if not carry(record):
                    if self.fell or self.out_of_time():
                        reason = f"carry: {record.reason}"
                        success = False
                        break
                    # the hold was lost: back to the object where it lies
                    if not phase_loop("approach", approach) or not phase_loop("stabilize", stabilize):
                        reason = f"recovery after carry: {self.phases[-1].reason}"
                        success = False
                        break
                    record2 = self.begin("acquire", attempt + 1)
                    if not self.acquire(record2):
                        reason = f"re-acquire: {record2.reason}"
                        success = False
                        break
                    continue
                placed = False
                for place_attempt in range(1, self.retry_budget + 2):
                    record = self.begin("place", place_attempt)
                    if self.place(record):
                        placed = True
                        break
                    if self.fell or self.out_of_time() or "out of reach" in record.reason or not self.holding_limb:
                        break
                    # still holding: the settle or the reach failed, try the placement again
                if placed:
                    delivered = True
                    break
                if self.fell or self.out_of_time() or "out of reach" in record.reason or self.holding_limb:
                    reason = f"place: {record.reason}"
                    success = False
                    break
                # the cube is not in the tray: find it and try again
                if not phase_loop("approach", approach) or not phase_loop("stabilize", stabilize):
                    reason = f"recovery after place: {self.phases[-1].reason}"
                    success = False
                    break
                record2 = self.begin("acquire", attempt + 1)
                if not self.acquire(record2):
                    reason = f"re-acquire: {record2.reason}"
                    success = False
                    break
            if success and not delivered:
                success, reason = False, f"place: {self.phases[-1].reason}"
        if success:
            record = self.begin("return", 1)
            if not go_home(record):
                success, reason = False, f"return: {record.reason}"
        # the end: 2 s of stillness, then the verdict
        if success:
            self.hold_still(2.0)
        record = self.recorder.finish()
        in_tray, detail = self.cube_in_tray()
        at_start = float(np.linalg.norm(self.base_xy() - np.array(START))) <= 0.45
        stable = self.stable()
        final = bool(success and in_tray and at_start and stable and not self.fell)
        if success and not final:
            reason = "cube not at rest inside the tray at the end" if not in_tray else ("not on the start pad at the end" if not at_start else "not stable at the end")
        distance = float(sum(np.linalg.norm(self.positions[i + 1] - self.positions[i]) for i in range(len(self.positions) - 1))) if len(self.positions) > 1 else 0.0
        latencies = np.asarray(self.latencies)
        steps_run = max(1, self.step_count)
        actuation = {"provenance": self.locomotor.provenance + "; manipulation: numerical reach on the declared grasp site, position servos", "controller": type(self.locomotor).__name__, "control_hz": 1.0 / float(self.model.opt.timestep),
                     "saturation_fraction": {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a): round(float(self.saturated[a]) / steps_run, 4) for a in range(self.model.nu)},
                     "peak_force": {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a): round(float(self.peak_force[a]), 3) for a in range(self.model.nu)},
                     "external_inputs": "none" if self.disturbance is None else f"{self.disturbance.kind}: {self.disturbance.detail} (recorded as user input, replayed exactly)", "root_writes": "none after placement", "object_writes": "none", "artificial_support": "none"}
        declared = list(self.body.declaration["support_members"])
        support = {"declared_contact_fraction": {m: round(self.contact_counts.get(m, 0) / max(1, self.contact_samples), 4) for m in declared},
                   "undeclared_contact_fraction": {m: round(c / max(1, self.contact_samples), 4) for m, c in sorted(self.contact_counts.items()) if m not in declared}, "samples": self.contact_samples}
        run = Run(record=record, floor_geom=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"), contacts_at_end=floor_contacts(self.model, self.data, 0, self.robot), base_height_m=float(self.data.xpos[self.base][2]), tilt_deg=self.tilt_deg(), settled=stable, upright=self.tilt_deg() < 15.0, heights=[])
        return RetrieveResult(success=final, reason=reason, phases=self.phases, fell=self.fell, fall_time_s=self.fall_time, duration_s=self.now, cap_s=self.cap_s, energy_j=self.energy, distance_m=distance, collisions=self.collisions, cube_in_tray=in_tray,
                              cube_final=[round(float(v), 4) for v in self.data.xpos[self.cube]], at_start=at_start, stable_at_end=stable, hold_events=self.hold_events, invariant_violations=self.violations, schedule=self.schedule, disturbance_log=self.disturbance_log,
                              control_latency_s={"mean": float(latencies.mean()), "p99": float(np.percentile(latencies, 99)), "max": float(latencies.max())}, contacts_at_end=run.contacts_at_end, run=run, samples=len(record.arrays["time_s"]),
                              actuation=actuation, support=support, max_object_slip_m=self.max_slip)


def run_retrieve(body: MobileBody, course: Course, *, seed: int, cap_s: float, jitter_xy_m: float = 0.10, jitter_yaw_deg: float = 10.0, disturbance: RetrieveDisturbance | None = None, settle_s: float = 1.5, retry_budget: int = 2) -> RetrieveResult:
    return RetrieveSession(body, course, seed=seed, cap_s=cap_s, jitter_xy_m=jitter_xy_m, jitter_yaw_deg=jitter_yaw_deg, disturbance=disturbance, settle_s=settle_s, retry_budget=retry_budget).run()


__all__ = ["PHASES", "GraspPoses", "PhaseRecord", "RetrieveDisturbance", "RetrieveResult", "RetrieveSession", "grasp_poses", "run_retrieve"]
