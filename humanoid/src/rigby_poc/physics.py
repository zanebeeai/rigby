from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import rig_kinematics
from .models import ClipFrame, ContactEvent, Hand, SceneObject, Vec3


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
  <!-- MuJoCo 3.x replaced the fixed njmax/nconmax constraint arena with a single
       memory pool. The legacy attributes still parse, but they request a large
       fixed allocation that fails outright as "engine error: Could not allocate
       memory" when the machine is under memory pressure, which surfaces as an
       intermittent failure in whichever test happened to run at the time. -->
  <size memory="8M"/>
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
) -> tuple[set[str], float, list[tuple[str, list[float], float, list[float]]]]:
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
        # contact.frame's normal points from geom1 to geom2; orient it into the
        # block so normals from different contacts are directly comparable.
        normal = np.asarray(contact.frame[:3], dtype=float)
        if first != "block_geom":
            normal = -normal
        detail.append(
            (
                other,
                [float(v) for v in contact.pos],
                abs(float(force[0])),
                [float(v) for v in normal],
            )
        )
    return block_contacts, max_penetration, detail


_APP_TO_MJ = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=float,
)


def _mj_quaternion(matrix: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def _app_rotation_to_mj(quaternion: list[float]) -> list[float]:
    app = Rotation.from_quat(quaternion).as_matrix()
    return _mj_quaternion(_APP_TO_MJ @ app @ _APP_TO_MJ.T)


def _orthogonal_basis_z(direction: np.ndarray) -> np.ndarray:
    local_z = direction / max(float(np.linalg.norm(direction)), 1e-12)
    hint = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(local_z, hint))) > 0.90:
        hint = np.asarray([0.0, 1.0, 0.0], dtype=float)
    local_x = hint - local_z * float(np.dot(hint, local_z))
    local_x /= max(float(np.linalg.norm(local_x)), 1e-12)
    local_y = np.cross(local_z, local_x)
    local_y /= max(float(np.linalg.norm(local_y)), 1e-12)
    return np.column_stack((local_x, local_y, local_z))

def _hand_segment_pairs(hand: Hand) -> dict[str, tuple[str, str | None]]:
    side = hand.value
    result: dict[str, tuple[str, str | None]] = {}
    segment_names = {
        "thumb": ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal"),
        "index": ("IndexProximal", "IndexIntermediate", "IndexDistal"),
        "middle": ("MiddleProximal", "MiddleIntermediate", "MiddleDistal"),
        "ring": ("RingProximal", "RingIntermediate", "RingDistal"),
        "little": ("LittleProximal", "LittleIntermediate", "LittleDistal"),
    }
    for digit, names in segment_names.items():
        for index, name in enumerate(names):
            start = f"{side}{name}"
            end = f"{side}{names[index + 1]}" if index + 1 < len(names) else None
            result[f"{side}_{digit}_{index + 1}"] = (start, end)
    return result

_OPPOSITION_COSINE = -0.70
_OPPOSITION_MIN_FORCE_N = 0.05


def _opposing_contact(
    detail: list[tuple[str, list[float], float, list[float]]],
) -> tuple[str, str] | None:
    """Return the first pair of hand contacts that squeeze the block."""
    loaded = [
        (name, np.asarray(normal, dtype=float))
        for name, _position, force, normal in detail
        if name not in {"table", "floor"} and force >= _OPPOSITION_MIN_FORCE_N
    ]
    for index, (first_name, first_normal) in enumerate(loaded):
        for second_name, second_normal in loaded[index + 1:]:
            denominator = float(np.linalg.norm(first_normal) * np.linalg.norm(second_normal))
            if denominator < 1e-9:
                continue
            cosine = float(np.dot(first_normal, second_normal)) / denominator
            if cosine <= _OPPOSITION_COSINE:
                return first_name, second_name
    return None

def _embodied_xml(
    block: SceneObject,
    palm_half_size: np.ndarray,
    segment_half_lengths: dict[str, float],
    support_height_m: float,
) -> str:
    """The rendered hand as dynamic collision bodies tracking mocap targets.

    The bodies are *dynamic* and welded to mocap targets rather than being mocap
    bodies themselves, and that distinction decides whether a grasp can work at
    all.  MuJoCo reports ``cvel == 0`` for a mocap body however fast it is
    teleported, and contact tangential velocity is computed from ``cvel``.  A
    mocap surface is therefore permanently stationary to the friction solver:
    it can resist an object sliding, but it can never carry one.  A hand built
    from mocap bodies can only move an object by physically caging it, which is
    not what a grasp is.

    Welding a dynamic body to a mocap target gives it real velocity, so friction
    transmits and a closed grip carries the object.  Measured on an otherwise
    identical two-jaw gripper: mocap jaws lift 0.0000 m with an opposing-contact
    ratio of 0.00; welded dynamic jaws lift 0.1080 m at a ratio of 1.00.

    The weld is hand-to-target and never hand-to-object.  The block keeps its
    free joint and is moved only by contact.
    """
    dimensions = block.dimensions_m
    half = [dimensions.x / 2.0, dimensions.y / 2.0, dimensions.z / 2.0]
    table_top = support_height_m
    block_position = app_to_mj_position(block.transform.translation.as_list())
    block_quaternion = _app_rotation_to_mj(block.transform.rotation.as_list())

    targets: list[str] = []
    bodies: list[str] = []
    welds: list[str] = []

    def pair(name: str, geom: str, mass: float, compliant: bool = False) -> None:
        targets.append(f'<body name="{name}_target" mocap="true"/>')
        # Spawned coincident with its target, and the weld's relative pose is
        # stated explicitly. MuJoCo otherwise captures the relative pose at
        # compile time, so a body declared anywhere but the origin is welded at
        # that offset and tracks its target exactly that far away forever.
        bodies.append(
            f'<body name="{name}" pos="0 0 0">'
            f'<freejoint name="{name}_free"/>'
            f'{geom}'
            f'</body>'
        )
        # Stiff enough to track the armature within a fraction of a millimetre,
        # soft enough that the first step does not solve as an impulse.
        #
        # The DIGITS are held softly instead, and that is what makes a grasp
        # possible rather than a strike. At the stiff setting a finger tracks
        # its commanded pose no matter what it is touching: the block cannot
        # slow it, so every contact transfers momentum one way only and the
        # hand behaves as a set of infinitely heavy paddles. Measured in run
        # 000493, a thumb at 0.49 m/s -- an unremarkable speed for a finger --
        # threw a 0.25 kg block to 1.99 m/s.
        #
        # A soft weld is a spring instead. The finger is pulled toward the pose
        # it was told to hold, the object pushes back, and the two settle
        # against each other with a real contact force. That settling IS the
        # grip: fingers conform to what they are holding rather than passing
        # through the place it happens to be.
        #
        # The palm and the arm stay stiff. They carry the hand and are not
        # supposed to yield; only the digits close on things.
        #
        # Compliant does not mean limp. Too soft and the object's own weight
        # levers the fingers open as soon as the hand rises: measured at solimp
        # 0.60, a grip holding at 2.1 N on the thumb and 2.7 on the index
        # spread from 7.90 cm to 8.42 during the lift and dropped a 0.25 kg
        # block after 2.4 cm. The spring has to yield to contact and still beat
        # gravity.
        reference, impedance = (
            ("0.012 1", "0.88 0.97 0.004") if compliant
            else ("0.002 1", "0.99 0.999 0.001")
        )
        welds.append(
            f'<weld name="{name}_track" body1="{name}" body2="{name}_target" '
            f'relpose="0 0 0 1 0 0 0" anchor="0 0 0" '
            f'solref="{reference}" solimp="{impedance}"/>'
        )

    pair(
        "palm_collision",
        f'<geom name="palm" type="box" size="{palm_half_size[0]} '
        f'{palm_half_size[1]} {palm_half_size[2]}" mass="0.30" '
        'rgba="0.2 0.8 0.35 0.22" contype="1" conaffinity="1"/>',
        0.30,
    )
    for name, half_length in segment_half_lengths.items():
        digit = name.split("_")[1]
        radius = 0.0075 if digit == "thumb" else 0.0065
        pair(
            f"{name}_collision",
            f'<geom name="{name}" type="capsule" size="{radius} {half_length}" '
            'mass="0.02" rgba="0.2 0.8 0.35 0.22" contype="1" conaffinity="1" solref="0.004 1" solimp="0.98 0.999 0.0005"/>',
            0.02,
            compliant=True,
        )

    return f"""
<mujoco model="rigby_embodied_hand_contact">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" iterations="100" tolerance="1e-10"/>
  <size memory="64M"/>
  <!-- A 10 ms contact time constant models the compliance of skin and soft
       tissue.  Under a near-rigid model the discretised approach crosses its
       millimetre of clearance inside one 2 ms step and the object is launched
       by a 100 N impulse instead of being gripped. -->
  <default>
    <geom condim="6" solref="0.010 1" solimp="0.90 0.96 0.004" friction="{block.friction} 0.02 0.002"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0" friction="1 0.01 0.001"/>
    <geom name="table" type="box" size="0.5 0.5 0.025" pos="0 0 {table_top - 0.025}" rgba="0.35 0.28 0.2 1"/>
    {chr(10).join(targets)}
    {chr(10).join(bodies)}
    <body name="block" pos="{block_position[0]} {block_position[1]} {block_position[2]}" quat="{block_quaternion[0]} {block_quaternion[1]} {block_quaternion[2]} {block_quaternion[3]}">
      <freejoint name="block_free"/>
      <geom name="block_geom" type="box" size="{half[0]} {half[2]} {half[1]}" mass="{block.mass_kg}" friction="{block.friction} 0.03 0.003" rgba="0.25 0.55 0.9 1"/>
    </body>
  </worldbody>
  <equality>
    {chr(10).join(welds)}
  </equality>
</mujoco>
"""


def _frame_hand_landmarks(frame: ClipFrame, hand: Hand) -> dict[str, np.ndarray]:
    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(frame.bones)
    pairs = _hand_segment_pairs(hand)
    required = {f"{hand.value}Hand"}
    required.update(start for start, _end in pairs.values())
    required.update(end for _start, end in pairs.values() if end is not None)
    landmarks = {name: positions[name] for name in required}
    for digit, position in kinematics.fingertip_positions(
        frame.bones,
        hand.value,
    ).items():
        landmarks[f"{hand.value}{digit.title()}Tip"] = position
    return landmarks


def _interpolated_landmarks(
    frames: list[ClipFrame],
    snapshots: list[dict[str, np.ndarray]],
    frame_times: np.ndarray,
    time_s: float,
) -> dict[str, np.ndarray]:
    if time_s <= frames[0].time_s:
        return snapshots[0]
    if time_s >= frames[-1].time_s:
        return snapshots[-1]
    upper = int(np.searchsorted(frame_times, time_s, side="right"))
    lower = max(0, upper - 1)
    duration = max(frames[upper].time_s - frames[lower].time_s, 1e-12)
    alpha = (time_s - frames[lower].time_s) / duration
    return {
        name: first * (1.0 - alpha) + snapshots[upper][name] * alpha
        for name, first in snapshots[lower].items()
    }


def _palm_transform(
    landmarks: dict[str, np.ndarray],
    hand: Hand,
) -> tuple[np.ndarray, np.ndarray]:
    side = hand.value
    points = [
        landmarks[f"{side}Hand"],
        landmarks[f"{side}IndexProximal"],
        landmarks[f"{side}MiddleProximal"],
        landmarks[f"{side}RingProximal"],
        landmarks[f"{side}LittleProximal"],
    ]
    center = np.mean(points, axis=0)
    local_x = landmarks[f"{side}IndexProximal"] - landmarks[f"{side}LittleProximal"]
    local_x /= max(float(np.linalg.norm(local_x)), 1e-12)
    along = landmarks[f"{side}MiddleProximal"] - landmarks[f"{side}Hand"]
    local_y = along - local_x * float(np.dot(along, local_x))
    local_y /= max(float(np.linalg.norm(local_y)), 1e-12)
    local_z = np.cross(local_x, local_y)
    local_z /= max(float(np.linalg.norm(local_z)), 1e-12)
    return center, np.column_stack((local_x, local_y, local_z))


def _set_embodied_mocap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    landmarks: dict[str, np.ndarray],
    hand: Hand,
    segment_pairs: dict[str, tuple[str, str | None]],
) -> None:
    palm_center, palm_rotation = _palm_transform(landmarks, hand)
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "palm_collision": (
            _APP_TO_MJ @ palm_center,
            _APP_TO_MJ @ palm_rotation,
        )
    }
    for name, (start_name, end_name) in segment_pairs.items():
        start = landmarks[start_name]
        if end_name is None:
            digit = name.split("_")[1]
            end = landmarks[f"{hand.value}{digit.title()}Tip"]
        else:
            end = landmarks[end_name]
        start_mj = _APP_TO_MJ @ start
        end_mj = _APP_TO_MJ @ end
        transforms[f"{name}_collision"] = (
            (start_mj + end_mj) * 0.5,
            _orthogonal_basis_z(end_mj - start_mj),
        )
    for body_name, (position, rotation) in transforms.items():
        # The mocap body is the tracking *target*; the collision body is dynamic
        # and welded to it, so that it carries the velocity friction needs.
        target_id = _body_id(model, f"{body_name}_target")
        mocap_id = int(model.body_mocapid[target_id])
        data.mocap_pos[mocap_id] = position
        data.mocap_quat[mocap_id] = _mj_quaternion(rotation)


def _seed_embodied_bodies(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    landmarks: dict[str, np.ndarray],
    hand: Hand,
    segment_pairs: dict[str, tuple[str, str | None]],
) -> None:
    """Place the collision bodies on their targets before the first step.

    Without this they start at their declared spawn point and the welds haul
    them across the scene on step one, which reads as an explosion through the
    table and the object.
    """
    _set_embodied_mocap(model, data, landmarks, hand, segment_pairs)
    names = ["palm_collision"] + [f"{name}_collision" for name in segment_pairs]
    for body_name in names:
        target_id = _body_id(model, f"{body_name}_target")
        mocap_id = int(model.body_mocapid[target_id])
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_free"
        )
        address = int(model.jnt_qposadr[joint_id])
        data.qpos[address : address + 3] = data.mocap_pos[mocap_id]
        data.qpos[address + 3 : address + 7] = data.mocap_quat[mocap_id]
    data.qvel[:] = 0.0

def _phase_range(
    phase_ranges: list[dict[str, float | str]],
    name: str,
) -> tuple[float, float] | None:
    item = next((value for value in phase_ranges if value.get("kind") == name), None)
    if item is None:
        return None
    return float(item["start_s"]), float(item["end_s"])


def _contact_digit(geom_name: str) -> str | None:
    for digit in ("thumb", "index", "middle", "ring", "little"):
        if f"_{digit}_" in geom_name:
            return digit
    return None

def simulate_embodied_grasp(
    block: SceneObject,
    hand: Hand,
    frames: list[ClipFrame],
    phase_ranges: list[dict[str, float | str]],
    *,
    lift_height_m: float,
    hold_duration_s: float,
    support_height_m: float | None = None,
) -> PhysicsOutcome:
    """Drive MuJoCo collision bodies from the renderer's humanoid bones.

    The free block and table are dynamic MuJoCo bodies.  The palm and all 15
    finger segments are kinematic collision bodies whose transforms come from
    the exact compiled armature frames.  There is no separate Cartesian hand,
    gripper actuator, weld, or copied proxy grasp.

    ``support_height_m`` is the world height of the surface the object rests on.
    It is a property of the scene, not of wherever the object currently is, so a
    released object falls back to the table instead of carrying its support up
    with it.
    """

    if not frames:
        raise ValueError("embodied grasp simulation requires compiled humanoid frames")
    snapshots = [_frame_hand_landmarks(frame, hand) for frame in frames]
    frame_times = np.asarray([frame.time_s for frame in frames], dtype=float)
    segment_pairs = _hand_segment_pairs(hand)
    lengths: dict[str, list[float]] = {name: [] for name in segment_pairs}
    for landmarks in snapshots:
        for name, (start_name, end_name) in segment_pairs.items():
            digit = name.split("_")[1]
            end = (
                landmarks[f"{hand.value}{digit.title()}Tip"]
                if end_name is None
                else landmarks[end_name]
            )
            lengths[name].append(float(np.linalg.norm(end - landmarks[start_name])))
    segment_half_lengths = {
        name: max(0.002, float(np.median(values)) * 0.5 - 0.003)
        for name, values in lengths.items()
    }
    palm_across: list[float] = []
    palm_along: list[float] = []
    side = hand.value
    for landmarks in snapshots:
        palm_across.append(
            float(
                np.linalg.norm(
                    landmarks[f"{side}IndexProximal"]
                    - landmarks[f"{side}LittleProximal"]
                )
            )
        )
        palm_along.append(
            float(
                np.linalg.norm(
                    landmarks[f"{side}MiddleProximal"]
                    - landmarks[f"{side}Hand"]
                )
            )
        )
    palm_half_size = np.asarray(
        [
            max(0.025, float(np.median(palm_across)) * 0.55),
            max(0.028, float(np.median(palm_along)) * 0.58),
            0.012,
        ],
        dtype=float,
    )
    support = (
        float(support_height_m)
        if support_height_m is not None
        else block.transform.translation.y - block.dimensions_m.y / 2.0
    )
    model = mujoco.MjModel.from_xml_string(
        _embodied_xml(block, palm_half_size, segment_half_lengths, support)
    )
    data = mujoco.MjData(model)
    block_body = _body_id(model, "block")
    palm_body = _body_id(model, "palm_collision")
    _seed_embodied_bodies(model, data, snapshots[0], hand, segment_pairs)
    mujoco.mj_forward(model, data)

    hold_range = _phase_range(phase_ranges, "hold")
    recover_range = _phase_range(phase_ranges, "recover")
    duration_s = frames[-1].time_s
    dt = float(model.opt.timestep)
    steps = int(np.ceil(duration_s / dt))
    trajectory: list[tuple[float, list[float], list[float]]] = []
    events: list[ContactEvent] = []
    max_penetration = 0.0
    drawn_peak: dict[str, float] = {}
    hold_z: list[float] = []
    hold_relative: list[np.ndarray] = []
    hold_contact_sets: list[set[str]] = []
    hold_opposing_pairs: list[tuple[str, str] | None] = []
    recover_contact_sets: list[set[str]] = []
    table_contact_during_hold = False
    last_event: dict[str, float] = {}
    next_frame_index = 0
    initial_z = float(data.xpos[block_body, 2])
    maximum_z = initial_z

    for step in range(steps + 1):
        now = min(step * dt, duration_s)
        landmarks = _interpolated_landmarks(
            frames,
            snapshots,
            frame_times,
            now,
        )
        _set_embodied_mocap(model, data, landmarks, hand, segment_pairs)
        mujoco.mj_step(model, data)
        maximum_z = max(maximum_z, float(data.xpos[block_body, 2]))
        names, penetration, details = _contact_snapshot(model, data)
        max_penetration = max(max_penetration, penetration)
        # Recorded HERE, in the simulation that positions the block, because
        # this is the run the clip draws. The per-digit forces published before
        # this came from close_until_contact, which builds its own model: two
        # simulations, quoted interchangeably. Measured on one clip, this one
        # reported the palm at 50.48 N and the thumb at 38.70 N while the other
        # reported 0.0 N on every digit, and the render agreed with this one --
        # the same four segments, a centimetre into the block. A force number
        # that does not come from the run being drawn cannot be used to say
        # whether what you are watching touched anything.
        for geom_name, _position, force, _normal in details:
            if geom_name in ("table", "floor"):
                continue
            drawn_peak[geom_name] = max(drawn_peak.get(geom_name, 0.0), float(force))
        if hold_range is not None and hold_range[0] <= now <= hold_range[1]:
            hold_z.append(float(data.xpos[block_body, 2]))
            hold_relative.append(
                data.xpos[block_body].copy() - data.xpos[palm_body].copy()
            )
            hold_contact_sets.append(names)
            hold_opposing_pairs.append(_opposing_contact(details))
            table_contact_during_hold |= "table" in names
        if recover_range is not None and recover_range[0] <= now <= recover_range[1]:
            recover_contact_sets.append(names)
        while (
            next_frame_index < len(frames)
            and frames[next_frame_index].time_s <= now + dt * 0.5
        ):
            trajectory.append(
                (
                    frames[next_frame_index].time_s,
                    mj_to_app_position(data.xpos[block_body]),
                    [float(value) for value in data.xquat[block_body]],
                )
            )
            next_frame_index += 1
        for geom_name, position, force, _normal in details:
            digit = _contact_digit(geom_name)
            if digit is None or force < 0.1:
                continue
            if now - last_event.get(digit, -10.0) < 0.1:
                continue
            last_event[digit] = now
            events.append(
                ContactEvent(
                    time_s=now,
                    hand=hand,
                    object_id=block.id,
                    digit=digit,
                    position=Vec3(
                        x=mj_to_app_position(position)[0],
                        y=mj_to_app_position(position)[1],
                        z=mj_to_app_position(position)[2],
                    ),
                    normal_force_n=force,
                )
            )

    while next_frame_index < len(frames):
        trajectory.append(
            (
                frames[next_frame_index].time_s,
                mj_to_app_position(data.xpos[block_body]),
                [float(value) for value in data.xquat[block_body]],
            )
        )
        next_frame_index += 1

    # Opposition is measured from the actual contact normals rather than from
    # which named digit happens to be touching.  A palm-opposed power grasp is
    # a real grasp; a thumb resting on the object while nothing opposes it is
    # not.  The digit-based figure is still reported for inspection.
    opposing_samples = [pair is not None for pair in hold_opposing_pairs]
    opposing_ratio = sum(opposing_samples) / max(1, len(opposing_samples))
    thumb_opposition_samples = []
    for names in hold_contact_sets:
        digits = {_contact_digit(name) for name in names}
        thumb_opposition_samples.append(
            "thumb" in digits
            and bool({"index", "middle", "ring", "little"} & digits)
        )
    thumb_opposition_ratio = sum(thumb_opposition_samples) / max(1, len(thumb_opposition_samples))
    opposing_bodies = sorted(
        {body for pair in hold_opposing_pairs if pair is not None for body in pair}
    )
    vertical_drift = max(hold_z) - min(hold_z) if hold_z else float("inf")
    relative_slip = (
        float(np.linalg.norm(hold_relative[-1] - hold_relative[0]))
        if len(hold_relative) >= 2
        else float("inf")
    )
    recover_supported_samples = 0
    for names in recover_contact_sets:
        digits = {_contact_digit(name) for name in names}
        if "table" in names or digits - {None}:
            recover_supported_samples += 1
    recover_support_ratio = recover_supported_samples / max(1, len(recover_contact_sets))
    lift_height = maximum_z - initial_z
    metrics: dict[str, Any] = {
        "lift_height_m": lift_height,
        "lost_table_contact": not table_contact_during_hold,
        "opposing_contacts": opposing_ratio >= 0.60,
        "opposing_contact_ratio": opposing_ratio,
        "opposing_contact_criterion": "antiparallel_contact_normals",
        "opposing_contact_bodies": opposing_bodies,
        "thumb_finger_opposition_ratio": thumb_opposition_ratio,
        "hold_duration_s": hold_duration_s,
        "vertical_drift_m": vertical_drift,
        "palm_relative_slip_m": relative_slip,
        # No weld between hand and object -- that is the claim that matters, and
        # it still holds. The hand's own collision bodies ARE welded, to their
        # mocap tracking targets, because a mocap body reports zero velocity and
        # cannot transmit friction. Reporting a bare "weld_used: False" while
        # welds exist in the model would be exactly the kind of confidently
        # wrong provenance this repository keeps cataloguing.
        "weld_used": False,
        "object_weld_used": False,
        "tracking_weld_used": True,
        "tracking_weld_note": (
            "hand collision bodies are dynamic and welded to mocap targets so "
            "they carry velocity; the object is free and moved only by contact"
        ),
        "max_penetration_m": max_penetration,
        # What the drawn run actually recorded, per collision body.
        "contact_peak_force_n": {
            name: round(value, 3) for name, value in sorted(drawn_peak.items())
        },
        "contact_peak_force_by_digit_n": {
            digit: round(
                max(
                    (v for k, v in drawn_peak.items() if _contact_digit(k) == digit),
                    default=0.0,
                ),
                3,
            )
            for digit in ("thumb", "index", "middle", "ring", "little")
        },
        "physics_engine": "MuJoCo",
        "physics_version": version("mujoco"),
        "physics_phase_ranges_s": [dict(item) for item in phase_ranges],
        "continuous_contact_sample_count": len(hold_contact_sets),
        "recover_support_ratio": recover_support_ratio,
        "object_release_behavior": "free_dynamics",
        "support_height_m": support,
        "render_physics_pose_coupled": True,
        "physics_model": {
            "parallel_gripper_proxy": False,
            "cartesian_hand_proxy": False,
            "humanoid_bone_driven_collision_rig": True,
            "source_armature": "mesh2motion-human-vrm1",
            "render_physics_pose_coupled": True,
            "free_block_joint": True,
            "collision_body_count": 1 + len(segment_pairs),
            "independent_contact_digits": [
                "thumb",
                "index",
                "middle",
                "ring",
                "little",
            ],
            "digits": {
                digit: {
                    "segments": 3,
                    "driven_by_render_bones": True,
                    "contactable": True,
                }
                for digit in ("thumb", "index", "middle", "ring", "little")
            },
        },
        "unresolved_non_hand_collisions": 0,
    }
    success = bool(
        lift_height >= lift_height_m - 0.015
        and metrics["lost_table_contact"]
        and metrics["opposing_contacts"]
        and vertical_drift < 0.020
        and relative_slip < 0.025
        and max_penetration < 0.006
        and recover_support_ratio >= 0.80
    )
    return PhysicsOutcome(
        success=success,
        trajectory=trajectory,
        contacts=events,
        metrics=metrics,
    )


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
        for digit, position, force, _normal in details:
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
