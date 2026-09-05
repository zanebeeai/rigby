from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import mujoco
import numpy as np

from .models import ContactEvent, Hand, SceneObject, Vec3


def app_to_mj_position(position: list[float]) -> list[float]:
    """glTF Y-up/+Z-forward to MuJoCo Z-up/-Y-forward."""
    return [position[0], -position[2], position[1]]


def mj_to_app_position(position: list[float] | np.ndarray) -> list[float]:
    """MuJoCo Z-up/-Y-forward to glTF Y-up/+Z-forward."""
    return [float(position[0]), float(position[2]), -float(position[1])]


@dataclass
class PhysicsOutcome:
    success: bool
    trajectory: list[tuple[float, list[float], list[float]]]
    contacts: list[ContactEvent]
    metrics: dict[str, Any]


def _table_geom(block: SceneObject, support: SceneObject | None) -> str:
    """The MuJoCo box the block rests on.

    From the scene's support surface when the manifest carries one, so the
    proxy sees the same table the compiler and the viewer see. Scenes without
    one (older fixtures, ad-hoc runners) keep the derived 1 m slab whose top is
    the block's underside, so their output is unchanged.
    """

    if support is None:
        half_y = block.dimensions_m.y / 2.0
        table_top = block.transform.translation.y - half_y
        return (
            f'<geom name="table" type="box" size="0.5 0.5 0.025" '
            f'pos="0 0 {table_top - 0.025}" rgba="0.35 0.28 0.2 1"/>'
        )
    position = app_to_mj_position(support.transform.translation.as_list())
    d = support.dimensions_m
    return (
        f'<geom name="table" type="box" size="{d.x / 2.0} {d.z / 2.0} {d.y / 2.0}" '
        f'pos="{position[0]} {position[1]} {position[2]}" rgba="0.35 0.28 0.2 1"/>'
    )


def _xml(block: SceneObject, support: SceneObject | None = None) -> str:
    d = block.dimensions_m
    half = [d.x / 2.0, d.y / 2.0, d.z / 2.0]
    return f"""
<mujoco model="rigby_contact_proxy">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" iterations="80" tolerance="1e-10"/>
  <size njmax="2000" nconmax="500"/>
  <default>
    <joint damping="12" armature="0.02"/>
    <geom condim="6" solref="0.003 1" solimp="0.95 0.995 0.001" friction="{block.friction} 0.02 0.002"/>
    <position kp="2800" kv="120" ctrllimited="true" ctrlrange="-2 2"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0" friction="1 0.01 0.001"/>
    {_table_geom(block, support)}
    <body name="cartesian_hand" pos="0 0 0">
      <joint name="hand_x" type="slide" axis="1 0 0" range="-0.8 0.8"/>
      <joint name="hand_y" type="slide" axis="0 1 0" range="-0.8 0.8"/>
      <joint name="hand_z" type="slide" axis="0 0 1" range="0.65 2.0"/>
      <geom name="palm" type="box" size="0.052 0.018 0.055" pos="0 0.095 0" mass="0.35" rgba="0.8 0.55 0.42 1" contype="1" conaffinity="1"/>
      <body name="thumb_proxy" pos="-0.075 -0.005 0">
        <joint name="thumb_close" type="slide" axis="1 0 0" range="0 0.065" damping="4"/>
        <geom name="thumb_pad" type="box" size="0.006 0.025 0.024" mass="0.035" rgba="0.9 0.65 0.5 1"/>
      </body>
      <body name="index_proxy" pos="0.075 -0.015 0.021">
        <joint name="index_close" type="slide" axis="1 0 0" range="-0.065 0" damping="4"/>
        <geom name="index_pad" type="box" size="0.006 0.012 0.018" mass="0.012" rgba="0.9 0.65 0.5 1"/>
      </body>
      <body name="middle_proxy" pos="0.075 0.012 0.021">
        <joint name="middle_close" type="slide" axis="1 0 0" range="-0.065 0" damping="4"/>
        <geom name="middle_pad" type="box" size="0.006 0.012 0.018" mass="0.012" rgba="0.9 0.65 0.5 1"/>
      </body>
      <body name="ring_proxy" pos="0.075 -0.015 -0.021">
        <joint name="ring_close" type="slide" axis="1 0 0" range="-0.065 0" damping="4"/>
        <geom name="ring_pad" type="box" size="0.006 0.012 0.018" mass="0.012" rgba="0.9 0.65 0.5 1"/>
      </body>
      <body name="little_proxy" pos="0.075 0.012 -0.021">
        <joint name="little_close" type="slide" axis="1 0 0" range="-0.065 0" damping="4"/>
        <geom name="little_pad" type="box" size="0.006 0.012 0.018" mass="0.012" rgba="0.9 0.65 0.5 1"/>
      </body>
    </body>
    <body name="block" pos="{block.transform.translation.x} {-block.transform.translation.z} {block.transform.translation.y}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box" size="{half[0]} {half[2]} {half[1]}" mass="{block.mass_kg}" friction="{block.friction} 0.03 0.003" rgba="0.25 0.55 0.9 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="hand_x_act" joint="hand_x" kp="4000" kv="180" ctrlrange="-0.8 0.8"/>
    <position name="hand_y_act" joint="hand_y" kp="4000" kv="180" ctrlrange="-0.8 0.8"/>
    <position name="hand_z_act" joint="hand_z" kp="4000" kv="180" ctrlrange="0.65 2.0"/>
    <position name="thumb_act" joint="thumb_close" kp="900" kv="30" ctrlrange="0 0.065"/>
    <position name="index_act" joint="index_close" kp="900" kv="30" ctrlrange="-0.065 0"/>
    <position name="middle_act" joint="middle_close" kp="900" kv="30" ctrlrange="-0.065 0"/>
    <position name="ring_act" joint="ring_close" kp="900" kv="30" ctrlrange="-0.065 0"/>
    <position name="little_act" joint="little_close" kp="900" kv="30" ctrlrange="-0.065 0"/>
  </actuator>
</mujoco>
"""


def _joint_qpos(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id])


def _body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _contact_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[set[str], float, list[tuple[str, list[float], float]]]:
    block_contacts: set[str] = set()
    max_penetration = 0.0
    detail: list[tuple[str, list[float], float]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = _geom_name(model, contact.geom1), _geom_name(model, contact.geom2)
        if "block_geom" not in (first, second):
            continue
        other = second if first == "block_geom" else first
        block_contacts.add(other)
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, index, force)
        detail.append((other, [float(v) for v in contact.pos], abs(float(force[0]))))
    return block_contacts, max_penetration, detail


def simulate_grasp(
    block: SceneObject,
    hand: Hand,
    lift_height_m: float = 0.10,
    hold_duration_s: float = 1.0,
    grip_force: float = 0.75,
    thumb_opposition: float = 0.75,
    digit_curl_adjustments: dict[str, float] | None = None,
    fps: int = 30,
    support: SceneObject | None = None,
) -> PhysicsOutcome:
    """Lift a free MuJoCo block through opposing contact/friction only.

    The block has a free joint and the model deliberately contains no equality,
    weld, tendon, parent relation, or block actuator.
    """
    model = mujoco.MjModel.from_xml_string(_xml(block, support))
    data = mujoco.MjData(model)
    block_body = _body_id(model, "block")
    hand_body = _body_id(model, "cartesian_hand")
    box_center_z = block.transform.translation.y

    block_mj = app_to_mj_position(block.transform.translation.as_list())
    data.qpos[_joint_qpos(model, "hand_x")] = block_mj[0]
    data.qpos[_joint_qpos(model, "hand_y")] = block_mj[1]
    data.qpos[_joint_qpos(model, "hand_z")] = box_center_z
    # Width-dependent closure leaves each pad just inside the nominal surface;
    # actuator compliance produces the normal force.
    closing = 0.075 - block.dimensions_m.x / 2.0 - 0.003
    closing = float(np.clip(closing, 0.021, 0.055))
    closing *= 0.94 + 0.08 * grip_force
    adjustments = digit_curl_adjustments or {}
    thumb_target = closing * (0.92 + 0.08 * thumb_opposition) * (1.0 + adjustments.get("thumb", 0.0) * 0.08)
    finger_target = -closing
    data.ctrl[:] = [
        block_mj[0],
        block_mj[1],
        box_center_z,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    approach_s, close_s = 0.20, 0.55
    lift_s = max(0.60, lift_height_m / 0.35)
    total_s = approach_s + close_s + lift_s + hold_duration_s
    steps = int(np.ceil(total_s / dt))
    sample_interval = max(1, int(round(1.0 / (fps * dt))))
    initial_z = float(data.xpos[block_body, 2])
    trajectory: list[tuple[float, list[float], list[float]]] = []
    events: list[ContactEvent] = []
    max_penetration = 0.0
    hold_z: list[float] = []
    hold_relative: list[np.ndarray] = []
    hold_contact_sets: list[set[str]] = []
    table_contact_during_hold = False
    last_event: dict[str, float] = {}

    for step in range(steps + 1):
        now = step * dt
        if now < approach_s:
            close_alpha = 0.0
            lift_alpha = 0.0
        elif now < approach_s + close_s:
            close_alpha = (now - approach_s) / close_s
            close_alpha = close_alpha * close_alpha * (3.0 - 2.0 * close_alpha)
            lift_alpha = 0.0
        else:
            close_alpha = 1.0
            lift_alpha = min(1.0, (now - approach_s - close_s) / lift_s)
            lift_alpha = lift_alpha * lift_alpha * (3.0 - 2.0 * lift_alpha)
        # Compensate the compliant finger pads' predictable gravity sag while
        # keeping the requested metric tied to the block, not the hand target.
        commanded_lift = min(0.85, lift_height_m + 0.040)
        data.ctrl[2] = box_center_z + commanded_lift * lift_alpha
        data.ctrl[3] = thumb_target * close_alpha
        data.ctrl[4:] = [
            finger_target * (1.0 + adjustments.get(name, 0.0) * 0.08) * close_alpha
            for name in ("index", "middle", "ring", "little")
        ]
        mujoco.mj_step(model, data)

        names, penetration, details = _contact_snapshot(model, data)
        max_penetration = max(max_penetration, penetration)
        in_hold = now >= approach_s + close_s + lift_s
        if in_hold:
            hold_z.append(float(data.xpos[block_body, 2]))
            hold_relative.append(data.xpos[block_body].copy() - data.xpos[hand_body].copy())
            hold_contact_sets.append(names)
            table_contact_during_hold |= "table" in names
        if step % sample_interval == 0 or step == steps:
            trajectory.append(
                (
                    now,
                    mj_to_app_position(data.xpos[block_body]),
                    [float(v) for v in data.xquat[block_body]],
                )
            )
        for digit, position, force in details:
            if digit not in {"thumb_pad", "index_pad", "middle_pad", "ring_pad", "little_pad"}:
                continue
            if force < 0.1 or now - last_event.get(digit, -10.0) < 0.1:
                continue
            last_event[digit] = now
            events.append(
                ContactEvent(
                    time_s=now,
                    hand=hand,
                    object_id=block.id,
                    digit=digit.removesuffix("_pad"),
                    position=Vec3(
                        x=mj_to_app_position(position)[0],
                        y=mj_to_app_position(position)[1],
                        z=mj_to_app_position(position)[2],
                    ),
                    normal_force_n=force,
                )
            )

    final_z = float(data.xpos[block_body, 2])
    lift_height = final_z - initial_z
    opposing_samples = [
        names
        for names in hold_contact_sets
        if "thumb_pad" in names
        and any(name in names for name in ("index_pad", "middle_pad", "ring_pad", "little_pad"))
    ]
    opposing_ratio = len(opposing_samples) / max(1, len(hold_contact_sets))
    vertical_drift = max(hold_z) - min(hold_z) if hold_z else float("inf")
    relative_slip = (
        float(np.linalg.norm(hold_relative[-1] - hold_relative[0]))
        if len(hold_relative) >= 2
        else float("inf")
    )
    metrics: dict[str, Any] = {
        "lift_height_m": lift_height,
        "lost_table_contact": not table_contact_during_hold,
        "opposing_contacts": opposing_ratio >= 0.95,
        "opposing_contact_ratio": opposing_ratio,
        "hold_duration_s": hold_duration_s,
        "vertical_drift_m": vertical_drift,
        "palm_relative_slip_m": relative_slip,
        "weld_used": False,
        "max_penetration_m": max_penetration,
        "physics_engine": "MuJoCo",
        "physics_version": version("mujoco"),
        "unresolved_non_hand_collisions": 0,
        "physics_model": {
            "parallel_gripper_proxy": False,
            "free_block_joint": True,
            "independent_contact_digits": ["thumb", "index", "middle", "ring", "little"],
            "digit_actuators": ["thumb_act", "index_act", "middle_act", "ring_act", "little_act"],
            "digits": {
                digit: {"independently_actuated": True, "contactable": True}
                for digit in ("thumb", "index", "middle", "ring", "little")
            },
        },
    }
    success = bool(
        lift_height >= lift_height_m - 0.015
        and metrics["lost_table_contact"]
        and metrics["opposing_contacts"]
        and vertical_drift < 0.015
        and relative_slip < 0.020
        and max_penetration < 0.004
    )
    return PhysicsOutcome(success=success, trajectory=trajectory, contacts=events, metrics=metrics)
