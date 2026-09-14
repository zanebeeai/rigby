"""Locomotion trials on the frozen course: travel, stop, turn and return, recorded on physics, with the perturbations a body has to recover from.

A trial compiles the body into the course (its floor, the pads, walls,
station, tray and ramp), starts it on the start pad with the seed's
jitter, and lets the navigator drive it through the travel goal's
waypoints -- out to the station, hold, back to the start, hold -- under
a time cap. Success is the goal's own rule: every waypoint reached in
turn within its radius, stable for two seconds at the end, no fall. A
perturbation is a push on the base (an external force the record keeps
and the replay reapplies), a low-friction patch across the route (a
static geom of the world), or a support disturbance appropriate to the
mechanics: a leg's servos fought by an external joint torque, a wheel's
drive cut by the controller itself, a tentacle's servos fought. Nothing
teleports, props or steadies the body; the controller's own actuators
are all it has.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
from rigby_core.simulation.recording import PhysicsRecord, PhysicsRecorder

from .ingest import MobileBody
from .locomotion import Locomotor, Navigator, make_locomotor
from .validate import Run, floor_contacts, robot_bodies
from .world import Course


FALL_TILT_DEG = 55.0
STABLE_SPEED_MPS = 0.05
STABLE_WINDOW_S = 2.0


@dataclass(frozen=True)
class Perturbation:
    kind: str
    """push | patch | support"""
    detail: str
    at_s: float = 0.0
    duration_s: float = 0.0
    magnitude: float = 0.0
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    target: str = ""
    """a limb or joint for a support disturbance; a route fraction for a patch"""
    patch_friction: float = 0.15
    patch_centre_m: tuple[float, float] = (1.5, 0.0)
    patch_half_m: tuple[float, float] = (0.4, 0.8)

    def as_json(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "at_s": self.at_s, "duration_s": self.duration_s, "magnitude": self.magnitude, "direction": list(self.direction), "target": self.target,
                "patch_friction": self.patch_friction, "patch_centre_m": list(self.patch_centre_m), "patch_half_m": list(self.patch_half_m)}


def course_world_xml(body: MobileBody, course: Course, perturbation: Perturbation | None = None) -> str:
    features = course.mjcf_features()
    if perturbation is not None and perturbation.kind == "patch":
        cx, cy = perturbation.patch_centre_m
        hx, hy = perturbation.patch_half_m
        features += (f'\n    <geom name="patch_slick" type="box" pos="{cx:.3f} {cy:.3f} 0.0005" size="{hx:.3f} {hy:.3f} 0.0005" '
                     f'friction="{perturbation.patch_friction:.3f} 0.001 0.00001" rgba="0.55 0.75 0.95 1" priority="2"/>')
    return body.floor_xml.replace('<light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>', '<light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n    ' + features, 1)


@dataclass
class TrialResult:
    success: bool
    reason: str
    reached: list[dict]
    fell: bool
    fall_time_s: float | None
    collisions: int
    """steps on which the base or an arm link touched a wall, the station or the tray"""
    energy_j: float
    distance_m: float
    mean_speed_mps: float
    peak_speed_mps: float
    duration_s: float
    cap_s: float
    control_latency_s: dict
    stable_at_end: bool
    final_distance_m: float
    contacts_at_end: tuple[str, ...]
    navigator_log: list[dict]
    perturbation_log: list[dict]
    run: Run
    samples: int
    actuation: dict = field(default_factory=dict)
    """per actuator: the fraction of steps its force sat at its limit, and its peak |force|; the controller's provenance"""
    support: dict = field(default_factory=dict)
    """per declared support member: the fraction of sampled steps it touched the ground; the undeclared members that did, with their fractions"""


def _tilt(model: mujoco.MjModel, data: mujoco.MjData, base: int) -> float:
    up = np.zeros(3)
    mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), data.xquat[base])
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def run_travel(body: MobileBody, course: Course, *, seed: int, waypoints: list[tuple[float, float]], cap_s: float, jitter_xy_m: float = 0.10, jitter_yaw_deg: float = 10.0,
               perturbation: Perturbation | None = None, settle_s: float = 1.5, start_stance: str | None = None, navigator_options: dict | None = None) -> TrialResult:
    """One travel trial, recorded every step. `navigator_options` are the Navigator's fields (the before/after pairs turn its fixes off)."""

    rng = np.random.default_rng(seed)
    world_xml = course_world_xml(body, course, perturbation)
    model = mujoco.MjSpec.from_string(world_xml).compile()
    data = mujoco.MjData(model)
    locomotor = make_locomotor(body, model)
    stance = body.declaration["stances"][start_stance or locomotor.working_stance()]["joints"]
    from .validate import place

    place(model, data, body, stance, yaw_deg=float(rng.uniform(-jitter_yaw_deg, jitter_yaw_deg)))
    root = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, body.declaration["root_joint"])])
    data.qpos[root] += float(rng.uniform(-jitter_xy_m, jitter_xy_m))
    data.qpos[root + 1] += float(rng.uniform(-jitter_xy_m, jitter_xy_m))
    mujoco.mj_forward(model, data)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.declaration["base_body"])
    robot = robot_bodies(model, body.declaration["base_body"])
    static_hazards = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("course_corridor_left", "course_corridor_right", "course_station", "course_tray_plinth", "course_tray", "course_doorway_left", "course_doorway_right")}
    static_hazards.discard(-1)
    arm_bodies = {b for b in robot if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("arm_")} | {base}
    navigator = Navigator(waypoints=list(waypoints), **(navigator_options or {}))
    recorder = PhysicsRecorder(model)
    locomotor.reset(data)
    steps = int(round(cap_s / model.opt.timestep))
    energy = 0.0
    collisions = 0
    peak = 0.0
    fell = False
    fall_time = None
    latencies = []
    perturbation_log: list[dict] = []
    positions = []
    speeds_window = []
    window_steps = int(STABLE_WINDOW_S / model.opt.timestep)
    dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, body.declaration["root_joint"])])
    support_joint_dofs = []
    if perturbation is not None and perturbation.kind == "support" and perturbation.target in body.declaration["limbs"]:
        support_joint_dofs = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in body.declaration["limbs"][perturbation.target]]
    limit = np.where(model.actuator_forcelimited.astype(bool), np.abs(model.actuator_forcerange).max(axis=1), np.inf)
    saturated = np.zeros(model.nu, dtype=np.int64)
    peak_force = np.zeros(model.nu)
    declared_support = list(body.declaration["support_members"])
    contact_counts: dict[str, int] = {}
    contact_samples = 0
    for step in range(steps + 1):
        now = float(data.time)
        state = locomotor.base_state(data)
        # the perturbation: a push on the base or a torque fighting a limb's servos, as recorded user input
        data.xfrc_applied[:] = 0.0
        data.qfrc_applied[:] = 0.0
        if perturbation is not None and perturbation.at_s <= now < perturbation.at_s + perturbation.duration_s:
            if perturbation.kind == "push":
                data.xfrc_applied[base, :3] = np.array(perturbation.direction) * perturbation.magnitude
            elif perturbation.kind == "support" and support_joint_dofs:
                for d in support_joint_dofs:
                    data.qfrc_applied[d] = perturbation.magnitude * math.sin(2 * math.pi * 3.0 * (now - perturbation.at_s))
            if not perturbation_log or perturbation_log[-1].get("event") != "on":
                perturbation_log.append({"event": "on", "time_s": now, "kind": perturbation.kind, "detail": perturbation.detail})
        elif perturbation_log and perturbation_log[-1].get("event") == "on":
            perturbation_log.append({"event": "off", "time_s": now})
        v, omega = (0.0, 0.0) if now < settle_s else navigator.command(state, now, locomotor)
        started = time.perf_counter()
        control = locomotor.control(data, v, omega)
        if perturbation is not None and perturbation.kind == "support" and perturbation.target.startswith("wheel:") and perturbation.at_s <= now < perturbation.at_s + perturbation.duration_s:
            # the controller itself cuts one wheel's drive: a support disturbance a wheeled body meets when a motor drops out
            control[locomotor.actuator_of[perturbation.target.split(":", 1)[1]]] = 0.0
        latencies.append(time.perf_counter() - started)
        recorder.capture(data, control, control_time_s=now)
        if step == steps:
            break
        data.ctrl[:] = control
        mujoco.mj_step(model, data)
        speed = float(np.linalg.norm(data.qvel[dof: dof + 2]))
        peak = max(peak, speed)
        speeds_window.append(speed)
        if len(speeds_window) > window_steps:
            speeds_window.pop(0)
        energy += float(np.sum(np.abs(data.qfrc_actuator * data.qvel))) * model.opt.timestep
        force = np.abs(data.actuator_force)
        saturated += (force >= 0.98 * limit).astype(np.int64)
        peak_force = np.maximum(peak_force, force)
        if step % 10 == 0:
            positions.append(np.array(data.xpos[base][:2]))
            contact_samples += 1
            for name in floor_contacts(model, data, 0, robot):
                contact_counts[name] = contact_counts.get(name, 0) + 1
            for i in range(data.ncon):
                c = data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in static_hazards and int(model.geom_bodyid[g2]) in arm_bodies) or (g2 in static_hazards and int(model.geom_bodyid[g1]) in arm_bodies):
                    collisions += 1
                    break
        tilt = _tilt(model, data, base)
        if tilt > FALL_TILT_DEG and not fell:
            fell = True
            fall_time = float(data.time)
        if fell and data.time > fall_time + 1.0:
            break
        if navigator.done and len(speeds_window) == window_steps and max(speeds_window) < STABLE_SPEED_MPS and not fell:
            break
    record = recorder.finish()
    final_state = locomotor.base_state(data)
    final_distance = float(np.linalg.norm(final_state.position[:2] - np.array(waypoints[-1])))
    stable = bool(len(speeds_window) == window_steps and max(speeds_window) < STABLE_SPEED_MPS and _tilt(model, data, base) < 15.0)
    reached = [e for e in navigator.log if e["event"] == "released"]
    success = bool(navigator.done and stable and not fell and final_distance <= navigator.release_m)
    if fell:
        reason = f"fell at {fall_time:.1f} s"
    elif not navigator.done:
        reason = f"time cap: waypoint {navigator.index} of {len(waypoints)} not reached"
    elif not stable:
        reason = "not stable for two seconds at the end"
    elif final_distance > navigator.release_m:
        reason = "drifted off the last waypoint"
    else:
        reason = ""
    distance = float(sum(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1))) if len(positions) > 1 else 0.0
    duration = float(data.time)
    latencies_arr = np.asarray(latencies)
    run = Run(record=record, floor_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"), contacts_at_end=floor_contacts(model, data, 0, robot), base_height_m=float(data.xpos[base][2]),
              tilt_deg=_tilt(model, data, base), settled=stable, upright=_tilt(model, data, base) < 15.0, heights=[])
    steps_run = max(1, step)
    actuation = {"provenance": locomotor.provenance, "controller": type(locomotor).__name__, "control_hz": 1.0 / float(model.opt.timestep),
                 "saturation_fraction": {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a): round(float(saturated[a]) / steps_run, 4) for a in range(model.nu)},
                 "peak_force": {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a): round(float(peak_force[a]), 3) for a in range(model.nu)},
                 "external_inputs": "none" if perturbation is None else f"{perturbation.kind}: {perturbation.detail} (recorded as user input, replayed exactly)",
                 "root_writes": "none after placement", "artificial_support": "none"}
    support = {"declared_contact_fraction": {m: round(contact_counts.get(m, 0) / max(1, contact_samples), 4) for m in declared_support},
               "undeclared_contact_fraction": {m: round(c / max(1, contact_samples), 4) for m, c in sorted(contact_counts.items()) if m not in declared_support}, "samples": contact_samples}
    return TrialResult(success=success, reason=reason, reached=reached, fell=fell, fall_time_s=fall_time, collisions=collisions, energy_j=energy, distance_m=distance,
                       mean_speed_mps=distance / max(duration - settle_s, 1e-6), peak_speed_mps=peak, duration_s=duration, cap_s=cap_s,
                       control_latency_s={"mean": float(latencies_arr.mean()), "p99": float(np.percentile(latencies_arr, 99)), "max": float(latencies_arr.max())}, stable_at_end=stable, final_distance_m=final_distance,
                       contacts_at_end=run.contacts_at_end, navigator_log=navigator.log, perturbation_log=perturbation_log, run=run, samples=len(record.arrays["time_s"]), actuation=actuation, support=support)


__all__ = ["FALL_TILT_DEG", "Perturbation", "TrialResult", "course_world_xml", "run_travel"]
