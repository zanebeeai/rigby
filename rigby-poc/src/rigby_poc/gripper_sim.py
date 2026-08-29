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
    rate_limited,
    bin_spec,
    object_above_rim_m,
    object_in_target,
    object_over_target_m,
    solve_carry_over,
    solve_release,
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
    # Pick was six phases; place adds two. Carrying is its own problem -- the
    # object has to clear the rim before it crosses, not after -- and letting go
    # is a decision rather than the absence of one.
    ("carry_over", 0.35),
    ("release", 0.5),
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
    bin_xml = _bin_xml()
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
    {bin_xml}
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


def _bin_xml() -> str:
    """An open-topped bin on a riser: four walls, a floor and a post.

    Static geometry, so it is scenery the solver must respect rather than
    another thing to hold. The riser is the point of the task: carrying to a
    target at a different height is what a lift alone does not test.
    """
    document = spec()["scene"]["bin"]
    centre = np.asarray(document["centre"], dtype=float)
    inner = np.asarray(document["inner_half_m"], dtype=float)
    wall = float(document["wall_m"])
    riser = np.asarray(document["riser_from"], dtype=float)
    mj = _mj(centre)
    pieces = [
        f'<geom name="bin_floor" type="box" pos="{mj[0]} {mj[1]} {mj[2] - inner[1] - wall}" '
        f'size="{inner[0] + wall} {inner[2] + wall} {wall}" rgba="0.45 0.5 0.58 1" '
        'friction="0.9 0.02 0.001"/>',
    ]
    for index, (dx, dz) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        along = inner[0] if dx else inner[2]
        pieces.append(
            f'<geom name="bin_wall{index}" type="box" '
            f'pos="{mj[0] + dx * (inner[0] + wall)} '
            f'{mj[1] + dz * (inner[2] + wall)} {mj[2]}" '
            f'size="{wall if dx else inner[0] + wall * 2} '
            f'{inner[2] + wall * 2 if dx else wall} {inner[1]}" '
            'rgba="0.5 0.55 0.63 1" friction="0.9 0.02 0.001"/>')
    post = _mj((centre + riser) / 2.0)
    height = float(centre[1] - riser[1]) / 2.0
    pieces.append(
        f'<geom name="bin_riser" type="cylinder" pos="{post[0]} {post[1]} {post[2]}" '
        f'size="0.045 {max(height, 0.01)}" rgba="0.4 0.44 0.52 1"/>')
    return "".join(pieces)


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
    # Seed ORIENTATION as well as position. Setting only the position left every
    # body at identity rotation while its mocap target was already turned, and
    # the welds -- the plate's especially, at solref 0.002 -- resolved that gap
    # as an impulse: NaN in QACC at t=0.008, on the third step. A run that
    # begins unstable is not a measurement of anything.
    spot = forward(state)
    orientation = _quaternion(spot["across"], spot["approach"])
    for name, key in (("finger_left", "left_pad"), ("finger_right", "right_pad"),
                      ("plate", "plate")):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        address = model.jnt_qposadr[model.body_jntadr[body]]
        data.qpos[address:address + 3] = _mj(spot[key])
        data.qpos[address + 3:address + 7] = orientation
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
        elif phase == 5 and object_above_rim_m(obj, block_half) >= 0.02:
            phase = 6
        elif phase == 6 and object_over_target_m(obj) <= 0.03 and holding(forces):
            phase = 7

        name, amount = PHASES[phase]
        if name == "move_to":
            nxt = solve_move_to(state, obj, block_half, forces, velocity, amount)
        elif name == "open_grip":
            nxt = solve_open_grip(state, obj, block_half, forces, amount)
        elif name == "close_grip":
            nxt = solve_close_grip(state, obj, block_half, forces, amount)
        elif name == "carry_over":
            nxt = solve_carry_over(state, obj, block_half, forces, amount)
        elif name == "release":
            nxt = solve_release(state, obj, block_half, amount)
        else:
            nxt = solve_lift(state, forces, amount)
            if nxt is None:  # too weak to carry: squeeze instead
                nxt = solve_close_grip(state, obj, block_half, forces, 1.0)
        if nxt is not None:
            # No faster than the machine is declared to move. Without this the
            # solvers close a fraction of the REMAINING distance each frame, so
            # a big correction snaps: 472 deg/s at the elbow on the handover
            # from lift to carry, which is what it looked like.
            state = rate_limited(state, nxt, 1.0 / fps)

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
                  f"over={object_over_target_m(obj) * 100:5.1f}cm "
                  f"rim={object_above_rim_m(obj, block_half) * 100:+5.1f}cm "
                  f"f={ {k: round(v, 1) for k, v in forces.items() if v > 0.3} }")
    return out
