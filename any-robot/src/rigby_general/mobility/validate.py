"""Settling and recovery tests for a mobile body, on physics, recorded.

A stance is validated by dropping the body two centimetres onto a level
floor in that stance with every position servo holding it, and reading
what happens: whether it comes to rest, whether its base stays upright,
which members touch the floor at rest (against the ones its author
declared), and how tall it stands. A recoverable initial state is one the
body is released from -- rolled, pitched or dropped -- and comes back from
to its stance on its own, servos holding, nothing else acting on it. A
trial marked unsupported is one the body is not meant to survive without a
controller; running it anyway, and keeping the clip, is how the claim is
kept honest. Every run is a physics record the evidence layer can seal
and replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from rigby_core.simulation.recording import PhysicsRecord, PhysicsRecorder

from .contracts import RecoveryTrialV1, StanceMeasurementV1
from .ingest import MobileBody


SETTLE_S = 4.0
STILL_SPEED_MPS = 0.02
STILL_HEIGHT_M = 0.01
UPRIGHT_TILT_DEG = 15.0
RELEASE_HEIGHT_M = 0.02


@dataclass
class Run:
    record: PhysicsRecord
    floor_geom: int
    contacts_at_end: tuple[str, ...]
    base_height_m: float
    tilt_deg: float
    settled: bool
    upright: bool
    heights: list[float]


def stance_controls(model: mujoco.MjModel, joints: dict[str, float]) -> np.ndarray:
    """Position servos to their stance targets; velocity servos (wheels) to zero."""

    control = np.zeros(model.nu)
    for a in range(model.nu):
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a][0]))
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if name.endswith("_servo") and joint in joints:
            control[a] = joints[joint]
    return control


def place(model: mujoco.MjModel, data: mujoco.MjData, body: MobileBody, joints: dict[str, float], *, roll_deg: float = 0.0, pitch_deg: float = 0.0, drop_m: float = 0.0, yaw_deg: float = 0.0) -> None:
    """The body in the stance, its lowest point RELEASE_HEIGHT_M (+ drop) above the floor, rolled and pitched as asked."""

    mujoco.mj_resetData(model, data)
    for name, value in joints.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = value
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, body.declaration["root_joint"])
    adr = int(model.jnt_qposadr[root])
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.radians([roll_deg, pitch_deg, yaw_deg]), "xyz")
    data.qpos[adr + 3: adr + 7] = quat
    mujoco.mj_forward(model, data)
    lowest = min(float(data.geom_xpos[g][2] - model.geom_rbound[g]) for g in range(model.ngeom) if model.geom_bodyid[g] != 0)
    data.qpos[adr + 2] += RELEASE_HEIGHT_M + drop_m - lowest
    mujoco.mj_forward(model, data)


def _tilt_deg(model: mujoco.MjModel, data: mujoco.MjData, base: int) -> float:
    up = np.zeros(3)
    mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), data.xquat[base])
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def _ground(model: mujoco.MjModel, geom: int) -> bool:
    """A static geom: the floor, a pad, a platform, a ramp -- anything of the world body."""

    return int(model.geom_bodyid[geom]) == 0


def robot_bodies(model: mujoco.MjModel, base_body: str) -> frozenset[int]:
    """The base body and everything hanging from it: the robot, as opposed to a course object."""

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    found = {base}
    changed = True
    while changed:
        changed = False
        for b in range(1, model.nbody):
            if b not in found and int(model.body_parentid[b]) in found:
                found.add(b)
                changed = True
    return frozenset(found)


def floor_contacts(model: mujoco.MjModel, data: mujoco.MjData, floor_geom: int, robot: frozenset[int] | None = None) -> tuple[str, ...]:
    """The robot's bodies touching the ground (any static geom) now."""

    bodies = set()
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if _ground(model, g1) != _ground(model, g2):
            other = g2 if _ground(model, g1) else g1
            body = int(model.geom_bodyid[other])
            if robot is not None and body not in robot:
                continue
            bodies.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body_{body}")
    return tuple(sorted(bodies))


def run_held(body: MobileBody, joints: dict[str, float], *, duration_s: float = SETTLE_S, roll_deg: float = 0.0, pitch_deg: float = 0.0, drop_m: float = 0.0,
             control_schedule=None) -> Run:
    """Simulate the body on the floor with its servos holding ``joints`` (or a
    schedule of targets by time), recording every step."""

    model = body.floor_model
    data = mujoco.MjData(model)
    place(model, data, body, joints, roll_deg=roll_deg, pitch_deg=pitch_deg, drop_m=drop_m)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.declaration["base_body"])
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    recorder = PhysicsRecorder(model)
    steps = int(round(duration_s / model.opt.timestep))
    heights: list[float] = []
    speeds: list[float] = []
    for step in range(steps + 1):
        targets = joints if control_schedule is None else control_schedule(float(data.time))
        control = stance_controls(model, targets)
        recorder.capture(data, control, control_time_s=float(data.time))
        if step == steps:
            break
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        if step % 25 == 0:
            heights.append(float(data.xpos[base][2]))
            speeds.append(float(np.linalg.norm(data.cvel[base][3:])))
    last_second = max(1, int(1.0 / (25 * model.opt.timestep)))
    settled = bool(max(speeds[-last_second:]) < STILL_SPEED_MPS and (max(heights[-last_second:]) - min(heights[-last_second:])) < STILL_HEIGHT_M)
    tilt = _tilt_deg(model, data, base)
    return Run(record=recorder.finish(), floor_geom=floor_geom, contacts_at_end=floor_contacts(model, data, floor_geom, robot_bodies(model, body.declaration["base_body"])), base_height_m=float(data.xpos[base][2]), tilt_deg=tilt,
               settled=settled, upright=tilt < UPRIGHT_TILT_DEG, heights=heights)


def support_polygon_area(model: mujoco.MjModel, data: mujoco.MjData, floor_geom: int, robot: frozenset[int] | None = None) -> float:
    points = []
    for i in range(data.ncon):
        c = data.contact[i]
        if _ground(model, int(c.geom1)) != _ground(model, int(c.geom2)):
            other = int(c.geom2) if _ground(model, int(c.geom1)) else int(c.geom1)
            if robot is not None and int(model.geom_bodyid[other]) not in robot:
                continue
            points.append((float(c.pos[0]), float(c.pos[1])))
    if len(points) < 3:
        return 0.0
    pts = sorted(set(points))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = 0.0
    for k in range(len(hull)):
        x1, y1 = hull[k]
        x2, y2 = hull[(k + 1) % len(hull)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def measure_stance(body: MobileBody, stance: str, *, duration_s: float = SETTLE_S) -> tuple[StanceMeasurementV1, Run]:
    declared = body.declaration["stances"][stance]
    run = run_held(body, declared["joints"], duration_s=duration_s)
    model = body.floor_model
    data = mujoco.MjData(model)
    from rigby_core.simulation.recording import STATE_SPEC

    mujoco.mj_setState(model, data, run.record.arrays["state"][-1], STATE_SPEC)
    mujoco.mj_forward(model, data)
    area = support_polygon_area(model, data, run.floor_geom, robot_bodies(model, body.declaration["base_body"]))
    declared_support = set(body.declaration["support_members"])
    undeclared = tuple(sorted(set(run.contacts_at_end) - declared_support))
    broad = any(m in run.contacts_at_end for m in declared_support) and area > 0.01
    stable = bool(run.settled and run.upright and (len(run.contacts_at_end) >= 3 or broad) and not undeclared)
    measurement = StanceMeasurementV1(stance=stance, declared_statically_stable=bool(declared["statically_stable"]), settled=run.settled, upright=run.upright, statically_stable=stable,
                                      base_height_m=run.base_height_m, tilt_deg=run.tilt_deg, support_contacts=run.contacts_at_end, undeclared_contacts=undeclared, support_polygon_area_m2=area, duration_s=duration_s)
    return measurement, run


def recovery_trial(body: MobileBody, stance: str, trial_id: str, *, perturbation: str, roll_deg: float, pitch_deg: float, drop_m: float, supported: bool, duration_s: float = SETTLE_S,
                   expected_height_m: float | None = None, joints: dict[str, float] | None = None) -> tuple[RecoveryTrialV1, Run]:
    """Recovered means: settled, upright, and back within a quarter of the
    stance's own height (measured by its settling test; the declared
    standing height when none is given). ``joints`` overrides the stance's
    targets for a manoeuvre the body does not declare (the dog rearing)."""

    joints = dict(joints if joints is not None else body.declaration["stances"][stance]["joints"])
    run = run_held(body, joints, duration_s=duration_s, roll_deg=roll_deg, pitch_deg=pitch_deg, drop_m=drop_m)
    expected_height = float(expected_height_m if expected_height_m is not None else body.declaration["expected_standing_height_m"])
    recovered = bool(run.settled and run.upright and abs(run.base_height_m - expected_height) <= 0.25 * expected_height)
    return RecoveryTrialV1(trial_id=trial_id, perturbation=perturbation, roll_deg=roll_deg, pitch_deg=pitch_deg, drop_m=drop_m, recovered=recovered, final_tilt_deg=run.tilt_deg, final_height_m=run.base_height_m, supported=supported), run


def inspection_sweep(body: MobileBody, stance: str, *, per_joint_s: float = 1.2, fraction: float = 0.5, limbs: tuple[str, ...] | None = None,
                     fraction_by_role: dict[str, float] | None = None) -> tuple[Run, list[dict]]:
    """Every joint of the chosen limbs swept in turn from its rest towards each
    end of its range and back, the rest of the body holding its stance on
    the floor: the physical inspection. The excursion is a fraction of the
    distance from the rest to each end, by joint role when given (a leg
    that bears the body is swept less far than an arm that does not);
    wheels, having no range, are left to spin free."""

    declaration = body.declaration
    joints = dict(declaration["stances"][stance]["joints"])
    order = [j for j in declaration["joints"] if limbs is None or j["limb"] in limbs]
    fractions = fraction_by_role or {}
    schedule = []
    t = 1.0
    for j in order:
        schedule.append({"joint": j["name"], "start_s": t, "end_s": t + per_joint_s, "limb": j["limb"], "role": j["role"], "fraction": float(fractions.get(j["role"], fraction))})
        t += per_joint_s
    total = t + 1.0
    by_name = {j["name"]: j for j in order}

    def targets(now: float) -> dict[str, float]:
        current = dict(joints)
        for entry in schedule:
            if entry["start_s"] <= now < entry["end_s"]:
                j = by_name[entry["joint"]]
                if j["range"] is None:
                    continue
                phase = (now - entry["start_s"]) / per_joint_s
                rest = float(joints.get(j["name"], j["rest"]))
                low, high = j["range"]
                excursion = entry["fraction"]
                # a triangle wave: rest -> towards high -> rest -> towards low -> rest
                if phase < 0.25:
                    value = rest + (high - rest) * excursion * (phase / 0.25)
                elif phase < 0.5:
                    value = rest + (high - rest) * excursion * (1 - (phase - 0.25) / 0.25)
                elif phase < 0.75:
                    value = rest + (low - rest) * excursion * ((phase - 0.5) / 0.25)
                else:
                    value = rest + (low - rest) * excursion * (1 - (phase - 0.75) / 0.25)
                current[j["name"]] = float(np.clip(value, low, high))
        return current

    run = run_held(body, joints, duration_s=total, control_schedule=targets)
    return run, schedule


__all__ = ["Run", "floor_contacts", "inspection_sweep", "measure_stance", "place", "recovery_trial", "robot_bodies", "run_held", "stance_controls", "support_polygon_area"]
