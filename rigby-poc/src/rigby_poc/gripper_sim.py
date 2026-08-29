"""The gripper in physics, run by the same six-phase controller as the hand.

The phases are copied in name and order from scripted_grasp because that is the
claim being tested: approach, open, engulf, close, squeeze, lift, with the same
thresholds keyed to the same metrics. If a two-finger gripper with four joints
picks the block up under that sequence, the sequence was about grasping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .gripper import (
    GripperState,
    _CARRY_FORCE_N,
    chosen_face,
    forward,
    holding,
    object_in_grasp_m,
    palm_facing,
    palm_to_object_m,
    solve_close_grip,
    solve_lift,
    solve_move_to,
    solve_open_grip,
    spec,
)

#: The same order the hand uses, with the same amplitudes.
PHASES: tuple[tuple[str, float], ...] = (
    ("move_to", 1.0),
    ("open_grip", 1.0),
    ("move_to", 1.0),
    ("close_grip", 0.4),
    ("close_grip", 1.0),
    ("lift", 0.05),
)

_ARRIVED_M = 0.10
_ENGULFED_FRACTION = 0.8
_OPEN_DWELL_S = 0.8


def _model_xml(block_half: np.ndarray, block_at: np.ndarray,
               table_top: float) -> str:
    """Bodies welded to mocap targets, exactly as the hand's model is built.

    The same trick and for the same reason: MuJoCo reports zero velocity for a
    mocap body however fast it is teleported, so friction can never carry an
    object. A dynamic body welded to a mocap target has real velocity, and the
    weld is soft on the fingers so the block can push back.
    """
    document = spec()
    finger = document["kinematics"]["finger"]
    reach = float(finger["length_m"]) / 2.0
    thick = float(finger["thickness_m"])
    pad = float(finger["pad_width_m"]) / 2.0
    return f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <light name="key" pos="0.6 -1.0 2.6" dir="-0.2 0.4 -1" diffuse="1 1 1"/>
    <geom name="table" type="plane" pos="0 0 {table_top}" size="2 2 0.1"
          rgba="0.4 0.42 0.48 1" friction="0.9 0.02 0.001"/>
    <body name="block" pos="{block_at[0]} {block_at[1]} {block_at[2]}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box"
            size="{block_half[0]} {block_half[1]} {block_half[2]}"
            mass="0.25" rgba="0.85 0.55 0.3 1" friction="0.9 0.02 0.001"
            solref="0.004 1" solimp="0.98 0.999 0.0005"/>
    </body>
    <body name="left_target" mocap="true"/>
    <body name="right_target" mocap="true"/>
    <body name="plate_target" mocap="true"/>
    <body name="finger_left" pos="0 0 0">
      <freejoint name="left_free"/>
      <geom name="left_geom" type="box" size="{thick} {pad} {reach}" mass="0.05"
            rgba="0.3 0.7 0.9 1" friction="0.9 0.02 0.001"
            solref="0.004 1" solimp="0.98 0.999 0.0005"/>
    </body>
    <body name="finger_right" pos="0 0 0">
      <freejoint name="right_free"/>
      <geom name="right_geom" type="box" size="{thick} {pad} {reach}" mass="0.05"
            rgba="0.3 0.7 0.9 1" friction="0.9 0.02 0.001"
            solref="0.004 1" solimp="0.98 0.999 0.0005"/>
    </body>
    <body name="plate" pos="0 0 0">
      <freejoint name="plate_free"/>
      <geom name="plate_geom" type="box" size="0.05 0.04 0.015" mass="0.4"
            rgba="0.25 0.5 0.7 1" friction="0.9 0.02 0.001"/>
    </body>
  </worldbody>
  <equality>
    <weld name="left_track" body1="finger_left" body2="left_target"
          relpose="0 0 0 1 0 0 0" anchor="0 0 0"
          solref="0.012 1" solimp="0.88 0.97 0.004"/>
    <weld name="right_track" body1="finger_right" body2="right_target"
          relpose="0 0 0 1 0 0 0" anchor="0 0 0"
          solref="0.012 1" solimp="0.88 0.97 0.004"/>
    <weld name="plate_track" body1="plate" body2="plate_target"
          relpose="0 0 0 1 0 0 0" anchor="0 0 0"
          solref="0.002 1" solimp="0.99 0.999 0.001"/>
  </equality>
</mujoco>
"""


def _mj(point: np.ndarray) -> np.ndarray:
    """App coordinates are Y-up; MuJoCo here is Z-up."""
    return np.asarray([point[0], point[2], point[1]], dtype=float)


def _app(point: np.ndarray) -> np.ndarray:
    return np.asarray([point[0], point[2], point[1]], dtype=float)


def _quaternion(across: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """A MuJoCo wxyz quaternion for a frame with these axes, in MuJoCo space."""
    x = _mj(across)
    z = _mj(approach)
    x = x / max(np.linalg.norm(x), 1e-9)
    z = z / max(np.linalg.norm(z), 1e-9)
    y = np.cross(z, x)
    y = y / max(np.linalg.norm(y), 1e-9)
    x = np.cross(y, z)
    matrix = np.column_stack([x, y, z])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, matrix.flatten())
    return quaternion


@dataclass
class GripperRun:
    """What happened, in the same shape the hand's runs report."""

    states: list[GripperState] = field(default_factory=list)
    block: list[np.ndarray] = field(default_factory=list)
    forces: list[dict[str, float]] = field(default_factory=list)
    phases: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def lift_achieved(self) -> dict[str, float]:
        heights = [float(p[1]) for p in self.block]
        return {
            "peak_lift_m": max(heights) - heights[0],
            "final_lift_m": heights[-1] - heights[0],
            "displaced_m": float(np.linalg.norm(self.block[-1] - self.block[0])),
        }


def run(block_half: np.ndarray, block_at: np.ndarray, table_top: float = 0.72,
        seconds: float = 9.0, fps: int = 30, verbose: bool = True) -> GripperRun:
    xml = _model_xml(block_half, _mj(block_at), table_top)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    block_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    targets = {
        name: int(model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_target")])
        for name in ("left", "right", "plate")
    }

    state = GripperState()
    out = GripperRun()
    phase = 0
    dt = float(model.opt.timestep)
    per_frame = max(1, int(round((1.0 / fps) / dt)))
    previous = None

    def place(current: GripperState) -> None:
        spot = forward(current)
        quaternion = _quaternion(spot["across"], spot["approach"])
        for name, key in (("left", "left_pad"), ("right", "right_pad"),
                          ("plate", "plate")):
            data.mocap_pos[targets[name]] = _mj(spot[key])
            data.mocap_quat[targets[name]] = quaternion

    place(state)
    for name in ("finger_left", "finger_right", "plate"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        key = {"finger_left": "left_pad", "finger_right": "right_pad",
               "plate": "plate"}[name]
        data.qpos[model.jnt_qposadr[model.body_jntadr[body]]:
                  model.jnt_qposadr[model.body_jntadr[body]] + 3] = _mj(forward(state)[key])
    mujoco.mj_forward(model, data)

    for index in range(int(seconds * fps)):
        now = index / fps
        forces: dict[str, float] = {}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1),
                     mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)}
            if "block_geom" not in names:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            for geom, digit in (("left_geom", "finger_left"),
                                ("right_geom", "finger_right"),
                                # The PLATE is not a finger, and counting its
                                # contact as grip is how a gripper reports 32 N
                                # while its opening is still wider than the
                                # block: the wrist was leaning on the object,
                                # not holding it.
                                ("plate_geom", "plate")):
                if geom in names:
                    forces[digit] = max(forces.get(digit, 0.0), float(abs(force[0])))

        obj = _app(data.xpos[block_body].copy())
        velocity = (obj - previous) * fps if previous is not None else np.zeros(3)
        previous = obj

        gap = palm_to_object_m(state, obj, block_half)
        inside = object_in_grasp_m(state, obj)
        if phase == 0 and gap <= _ARRIVED_M:
            phase = 1
        elif phase == 1 and now > _OPEN_DWELL_S:
            phase = 2
        elif phase == 2 and inside <= float(np.min(block_half)) * _ENGULFED_FRACTION:
            phase = 3
        elif phase == 3 and holding(forces):
            phase = 4
        elif phase == 4 and holding(forces):
            phase = 5

        name, amount = PHASES[phase]
        if name == "move_to":
            nxt = solve_move_to(state, obj, block_half, forces, velocity, amount)
        elif name == "open_grip":
            nxt = solve_open_grip(state, obj, block_half, forces, amount)
        elif name == "close_grip":
            nxt = solve_close_grip(state, obj, block_half, forces, amount)
        else:
            nxt = solve_lift(state, forces, amount)
            if nxt is None:  # too weak to carry: squeeze instead
                nxt = solve_close_grip(state, obj, block_half, forces, 1.0)
        if nxt is not None:
            state = nxt

        place(state)
        for _ in range(per_frame):
            mujoco.mj_step(model, data)

        out.states.append(state.copy())
        out.block.append(obj)
        out.forces.append(dict(forces))
        out.phases.append(phase)
        out.times.append(now)
        if verbose and index % 15 == 0:
            print(f"  t={now:5.2f} phase {phase} {name:<11} "
                  f"palm={gap * 100:6.2f}cm in={inside * 100:5.2f}cm "
                  f"open={forward(state)['opening'] * 100:5.2f}cm "
                  f"face={palm_facing(state, obj, block_half):+.2f} "
                  f"f={ {k: round(v, 1) for k, v in forces.items() if v > 0.3} }")
    return out
