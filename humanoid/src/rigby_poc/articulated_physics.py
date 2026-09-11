"""Grasp simulation where the fingers are joints, not teleported bodies.

The old path welded sixteen rigid bodies to mocap targets and moved them onto
whatever pose the renderer had authored. Contact could not influence that hand:
told to close, a finger closed, and if the authored shape put it inside the
block then inside the block is where it went. MuJoCo answers a commanded
interpenetration by ejecting the object, which it did, at 0.65 m/s, before any
grip existed.

Here the palm is still carried by the arm -- placement is the renderer's job and
it already solves it -- but the fifteen finger joints are real hinges driven by
force-bounded position servos. The renderer's own joint angles become the servo
targets, so the simulated hand is trying to reach the drawn pose rather than
being placed in it. When the block stops a finger short, the servo keeps pushing
and that push is the grip. Nothing has to plan a pose that happens to land on the
object without passing through it, which is what the aperture search, the seating
term and the closure controller were all separately attempting.

What this cannot do is hold the wrist against the weight of a carried object,
because the wrist is still kinematic. That shows up as an object that is gripped
correctly and then carried by a hand no torque is being asked of. Making the arm
dynamic is a larger change and a different question from whether a grasp forms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .analysis.anatomy.frame import all_frames, decompose
from .articulated_hand import (
    CHAINS,
    FLEXION_DEG,
    actuator_xml,
    hand_xml,
    joint_names,
)
from .models import ClipFrame, Hand, SceneObject
from .physics import (
    _app_rotation_to_mj,
    _body_id,
    _contact_snapshot,
    _frame_hand_landmarks,
    _mj_quaternion,
    _opposing_contact,
    _palm_transform,
    _APP_TO_MJ,
    app_to_mj_position,
    mujoco,
)


@dataclass
class ArticulatedOutcome:
    frames_objects: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _model_xml(block: SceneObject, hand: Hand, support_height_m: float) -> str:
    dimensions = block.dimensions_m
    half = (dimensions.x / 2.0, dimensions.y / 2.0, dimensions.z / 2.0)
    position = app_to_mj_position(block.transform.translation.as_list())
    quaternion = _app_rotation_to_mj(block.transform.rotation.as_list())
    side = hand.value
    return f"""
<mujoco model="rigby_articulated_hand">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" iterations="100" tolerance="1e-10"/>
  <size memory="64M"/>
  <!-- Skin compliance, as in the embodied model. A near-rigid contact resolves
       the approach inside one 2 ms step and launches the object. -->
  <default>
    <geom condim="6" solref="0.010 1" solimp="0.90 0.96 0.004"
          friction="{block.friction} 0.02 0.002"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0" friction="1 0.01 0.001"/>
    <geom name="table" type="box" size="0.5 0.5 0.025"
          pos="0 0 {support_height_m - 0.025}" rgba="0.35 0.28 0.2 1"/>
    <body name="{side}_palm_target" mocap="true"/>
    {hand_xml(hand)}
    <body name="block" pos="{position[0]} {position[1]} {position[2]}"
          quat="{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box" size="{half[0]} {half[2]} {half[1]}"
            mass="{block.mass_kg}" friction="{block.friction} 0.03 0.003"
            rgba="0.25 0.55 0.9 1"/>
    </body>
  </worldbody>
  <equality>
    <!-- The palm alone is welded. Everything distal to it is jointed, which is
         the whole point: the weld carries placement, the joints carry grasp. -->
    <weld name="{side}_palm_track" body1="{side}_palm" body2="{side}_palm_target"
          relpose="0 0 0 1 0 0 0" anchor="0 0 0"
          solref="0.002 1" solimp="0.99 0.999 0.001"/>
  </equality>
  <actuator>
    {actuator_xml(hand)}
  </actuator>
</mujoco>
"""


def rendered_joint_targets(frame: ClipFrame, hand: Hand) -> dict[str, float]:
    """The drawn hand's own flexion angles, as servo targets.

    Read off the rendered rotations rather than from a curl parameter so the
    simulated hand chases the pose actually on screen. Magnitudes are used and
    clamped into each hinge's range, because a target outside the range is
    silently saturated by MuJoCo and the disagreement would not surface.

    The thumb is read on a different DOF from the fingers, and that is not a
    fudge. ``rom.v1.json`` records that this rig's own crate_grip preset drives
    all three thumb segments about an axis that decomposes to pure abduction
    with a zero flexion component -- the thumb simply has no flexion to read.
    Reading it anyway produced a thumb commanded to 11 degrees at its base and
    0 at both other joints for every frame of the clip, closing on nothing while
    the fingers reached 80, which is exactly the symptom this hand was rebuilt
    to remove.
    """
    frames = all_frames()
    targets: dict[str, float] = {}
    side = hand.value
    for finger, bones in CHAINS.items():
        limits = FLEXION_DEG[finger]
        for index, bone in enumerate(bones):
            name = f"{side}{bone}"
            pose = frame.bones.get(name)
            joint = f"{side}_{finger}_j{index + 1}"
            if pose is None or name not in frames:
                targets[joint] = 0.0
                continue
            angles = decompose(
                np.asarray(
                    [pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w],
                    dtype=float,
                ),
                frames[name],
            )
            travel = (
                angles.abduction_rad if finger == "thumb" else angles.flexion_rad
            )
            targets[joint] = float(
                np.clip(abs(travel), 0.0, math.radians(limits[index]))
            )
    return targets


def simulate_articulated_grasp(
    block: SceneObject,
    hand: Hand,
    frames: list[ClipFrame],
    *,
    support_height_m: float,
) -> ArticulatedOutcome:
    """Drive the jointed hand from the rendered clip and report what it grasped."""
    if not frames:
        raise ValueError("an articulated grasp needs authored frames to ride on")

    model = mujoco.MjModel.from_xml_string(_model_xml(block, hand, support_height_m))
    data = mujoco.MjData(model)
    side = hand.value
    block_body = _body_id(model, "block")
    palm_body = _body_id(model, f"{side}_palm")
    target_body = _body_id(model, f"{side}_palm_target")
    mocap = int(model.body_mocapid[target_body])
    joints = joint_names(hand)
    actuator = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act")
        for name in joints
    }

    def place_palm(frame: ClipFrame) -> None:
        landmarks = _frame_hand_landmarks(frame, hand)
        centre, rotation = _palm_transform(landmarks, hand)
        data.mocap_pos[mocap] = _APP_TO_MJ @ centre
        data.mocap_quat[mocap] = _mj_quaternion(_APP_TO_MJ @ rotation)

    # Seat the palm on its target before the first step, or the weld hauls the
    # whole hand across the scene and through the table on frame one.
    place_palm(frames[0])
    mujoco.mj_forward(model, data)
    address = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_palm_free")
    ]
    data.qpos[address : address + 3] = data.mocap_pos[mocap]
    data.qpos[address + 3 : address + 7] = data.mocap_quat[mocap]
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    duration = frames[-1].time_s
    times = np.asarray([f.time_s for f in frames], dtype=float)
    start_z = float(data.xpos[block_body, 2])
    peak_z = start_z
    max_penetration = 0.0
    opposed_steps = 0
    contact_steps = 0
    peak_force: dict[str, float] = {}
    tracking = []
    trajectory: list[dict[str, Any]] = []
    next_frame = 0

    for step in range(int(math.ceil(duration / dt)) + 1):
        now = min(step * dt, duration)
        index = int(np.clip(np.searchsorted(times, now), 0, len(frames) - 1))
        frame = frames[index]

        place_palm(frame)
        for name, value in rendered_joint_targets(frame, hand).items():
            slot = actuator.get(name, -1)
            if slot >= 0:
                data.ctrl[slot] = value
        mujoco.mj_step(model, data)

        tracking.append(
            float(np.linalg.norm(data.xpos[palm_body] - data.xpos[target_body]))
        )
        names, penetration, detail = _contact_snapshot(model, data)
        max_penetration = max(max_penetration, penetration)
        peak_z = max(peak_z, float(data.xpos[block_body, 2]))
        digits = [d for d in detail if d[0] not in {"table", "floor"}]
        if digits:
            contact_steps += 1
        for geom, _position, force, _normal in digits:
            peak_force[geom] = max(peak_force.get(geom, 0.0), force)
        if _opposing_contact(detail) is not None:
            opposed_steps += 1

        while next_frame < len(frames) and frames[next_frame].time_s <= now + dt * 0.5:
            trajectory.append(
                {
                    "time_s": frames[next_frame].time_s,
                    "position": [float(v) for v in data.xpos[block_body]],
                    "quaternion": [float(v) for v in data.xquat[block_body]],
                }
            )
            next_frame += 1

    total = max(step + 1, 1)
    return ArticulatedOutcome(
        frames_objects=trajectory,
        metrics={
            "protocol": "articulated_hand_v1",
            "lift_height_m": peak_z - start_z,
            "opposing_contacts": opposed_steps > 0,
            "opposing_contact_ratio": opposed_steps / total,
            "digit_contact_ratio": contact_steps / total,
            "max_penetration_m": max_penetration,
            "palm_tracking_error_m": float(np.mean(tracking)) if tracking else 0.0,
            "peak_digit_force_n": {k: round(v, 2) for k, v in sorted(peak_force.items())},
            # Stated because it is the honest limit of this model, not a detail:
            # the fingers are dynamic and the wrist is not.
            "wrist_is_kinematic": True,
            "object_weld_used": False,
        },
    )
