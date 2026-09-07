"""Place a hand around an object by orientation, and close it onto the object.

Every pickup the compiler produced before this module was placed by a point: a
wrist pivot offset from the block's centre by a literal, the hand's orientation
being whatever the arm solver left it. The fist ended up beside the block, the
block hung off the fingertips, and the fingers pointed across the body.

A grasp is an orientation first. In the hand's own frame -- the rig's wrist
frame, measured off the skinned hand: fingers along +y, thumb side +z, palm
facing -x on the right hand and +x on the left -- a held block sits in a pocket
against the palm, its far face under the knuckle line, its top under the thumb.
This module builds that frame in the world from the finger axis the forearm
presents and a palm normal facing the block, puts the pocket on the block's
centre, and solves the arm for the wrist that results, with the hand's world
rotation as a solver target. The fingers are then closed one by one to the
curl at which each first meets the block, measured on the skin.

Measurements and tolerances live here rather than in the compiler, which is
held to zero new literals by ``test_no_hardcoded_thresholds``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from .hand_mesh import box_signed_distance, hand_mesh
from .kinematics import rig_kinematics
from .models import (
    AffordanceRole,
    BonePose,
    Hand,
    HandShape,
    MotionPrimitive,
    PrimitiveParameters,
    Quat,
    SceneObject,
    Transform,
)
from .primitives import FINGERS, effective_curls, hand_pose

#: Gap the palm skin is seated at from the face it holds, and the gap a closing
#: finger stops at, measured on the skin. Below the contact audit's 2 mm touch
#: tolerance so a seated digit reads as touching; wide enough that the thumb's
#: blend of curl and opposition between two seated keyframes, which bows up
#: to 1 mm nearer the block than either end, stays out of it.
TOUCH_MARGIN_M = 0.0015
#: Gap kept between the hand's lowest finger and the surface the block rests on.
SUPPORT_CLEARANCE_M = 0.005
#: How far past the knuckle line the block's far face sits. With the face on
#: the knuckle line the fingers close over the block's far-top edge and end
#: 0.89 curled; 20 mm out they meet the far face at 0.69, the closure an
#: unpitched hand had (0.72), which is what read as held.
FAR_FACE_PAST_KNUCKLE_M = 0.02
#: Bisection steps for the closure; 2**-10 of the curl range.
CLOSURE_STEPS = 10
#: Samples of the curl range scanned from straight to fully closed to find
#: the band a digit may close through; the boundaries are then bisected.
CLOSURE_SCAN = 32
#: Angle the seat may put between the fingers and the forearm. Past it the
#: whole grasp frame turns toward the forearm, so the block is taken a little
#: diagonally or with the fingers pitched a little down rather than the wrist
#: bent past its range. With the palm vertical, pitching the fingers up off a
#: forearm that reaches down is radial deviation, 20 deg typical of which the
#: rig's rest already carries 11; measured peak 20.0 against an enforced 13.7
#: with the fingers held horizontal on grasp-block-table.
SWING_BUDGET_RAD = math.radians(9.0)
#: Passes of the seat's yaw settle; two are enough, the third is a check.
SEAT_PASSES = 3
#: How the planner's arm knobs read on a grasp. They used to nudge a wrist
#: point by 0.08 m per unit, which on a held block means letting go of it;
#: on a seat they turn the grasp instead. ``lateral_offset`` yaws the grasp
#: about world up (a block taken a little across), ``arm_height`` pitches the
#: fingers down (the wrist rides higher over the block), ``arm_depth`` moves
#: the block deeper into the palm, and ``elbow_swivel`` swivels the elbow's
#: bend plane about the shoulder-wrist line at the two-link solver's own
#: 0.75 rad per unit.
LATERAL_OFFSET_YAW_RAD = 1.2
ARM_HEIGHT_PITCH_RAD = 1.5
ARM_DEPTH_M = 0.05
ELBOW_SWIVEL_RAD = 0.75
#: The elbow hangs below the shoulder-wrist line and a little inboard when a
#: hand is placed on a block, as an arm reaching forward to a table does.
#: Measured on grasp-block-table: forearm azimuth 8 deg off the finger axis
#: (18 straight down, 29 flared a quarter out), so the wrist deviation stays
#: inside its enforced range. Given for the right side; the left mirrors x.
ELBOW_HANG_RIGHT = (0.2, -1.0, 0.0)


@dataclass(frozen=True)
class Box:
    """An oriented box in the world the hand may touch but not enter."""

    name: str
    centre: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray

    @classmethod
    def from_scene_object(cls, item: SceneObject, transform: Transform | None = None) -> Box:
        placed = item.transform if transform is None else transform
        return cls(
            name=item.id,
            centre=np.asarray(placed.translation.as_list(), dtype=float),
            rotation=Rotation.from_quat(placed.rotation.as_list()).as_matrix(),
            half_extents=np.asarray(item.dimensions_m.as_list(), dtype=float) * 0.5,
        )

    def extent_along(self, axis: np.ndarray) -> float:
        """Half extent of the box along a world unit direction."""

        return float(np.dot(np.abs(np.asarray(axis, dtype=float) @ self.rotation), self.half_extents))


@dataclass(frozen=True)
class HandGeometry:
    """The open hand's own measurements in the wrist frame, off the skin."""

    #: Which way the palm faces along wrist x: -1 on the right hand, +1 left.
    palm_sign: float
    #: Depth of the palm's skin from the wrist plane, along the palm normal.
    palm_depth_m: float
    #: Distance along the finger axis from the wrist to the knuckle line.
    knuckle_m: float
    #: Thumb-axis coordinate of the little finger's outer skin (negative) and
    #: of the index finger's outer skin (positive).
    little_edge_m: float
    index_edge_m: float
    #: Finger-axis distance from the wrist to the little finger's tip, open.
    little_tip_m: float


@lru_cache(maxsize=2)
def hand_geometry(side: str) -> HandGeometry:
    mesh = hand_mesh(side)
    hand = Hand(side)
    bones = {
        name: BonePose(rotation=rotation)
        for name, rotation in hand_pose(hand, HandShape.OPEN, PrimitiveParameters()).items()
    }
    local = mesh.local(bones)
    sign = 1.0 if hand == Hand.LEFT else -1.0
    palm = local[mesh.groups["Palm"]]
    knuckles = [float(local[mesh.groups[digit]][:, 1].min()) for digit in FINGERS if digit != "Thumb"]
    return HandGeometry(
        palm_sign=sign,
        palm_depth_m=float((sign * palm[:, 0]).max()),
        knuckle_m=float(np.median(knuckles)),
        little_edge_m=float(local[mesh.groups["Little"]][:, 2].min()),
        index_edge_m=float(local[mesh.groups["Index"]][:, 2].max()),
        little_tip_m=float(local[mesh.groups["Little"]][:, 1].max()),
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def hand_rotation(side: str, finger_axis: np.ndarray, palm_normal: np.ndarray) -> np.ndarray:
    """World rotation of the hand whose fingers point along ``finger_axis``
    and whose palm faces ``palm_normal`` (projected off the finger axis).

    Columns are the wrist frame's axes in the world.
    """

    fingers = _unit(finger_axis)
    normal = np.asarray(palm_normal, dtype=float)
    normal = _unit(normal - fingers * float(np.dot(normal, fingers)))
    x_axis = hand_geometry(side).palm_sign * normal
    return np.column_stack((x_axis, fingers, np.cross(x_axis, fingers)))


def palm_normal_of(side: str, rotation: np.ndarray) -> np.ndarray:
    """The world direction the palm faces for a hand world rotation."""

    return hand_geometry(side).palm_sign * np.asarray(rotation, dtype=float)[:, 0]


def elbow_hang(side: str) -> np.ndarray:
    """The bend hint for an arm placing its hand on a block."""

    hint = np.asarray(ELBOW_HANG_RIGHT, dtype=float)
    if side == Hand.LEFT.value:
        hint[0] = -hint[0]
    return hint


def finger_axis_of(rotation: np.ndarray) -> np.ndarray:
    """The world direction the fingers point along for a hand world rotation."""

    return np.asarray(rotation, dtype=float)[:, 1].copy()


def pocket_local(
    geometry: HandGeometry, rotation: np.ndarray, block: Box, depth_m: float = 0.0
) -> np.ndarray:
    """Where the block's centre sits in the wrist frame when the hand holds it.

    Against the palm along the palm normal, its far face under the knuckle
    line along the fingers (``depth_m`` further in), and along the thumb axis
    centred on the finger span except that the lowest finger keeps clear of
    whatever the block is resting on.
    """

    along_normal = block.extent_along(rotation[:, 0])
    along_fingers = block.extent_along(rotation[:, 1])
    along_thumb = block.extent_along(rotation[:, 2])
    pocket = np.asarray(
        [
            geometry.palm_sign * (geometry.palm_depth_m + along_normal + TOUCH_MARGIN_M),
            geometry.knuckle_m + FAR_FACE_PAST_KNUCKLE_M - along_fingers - depth_m,
            0.5 * (geometry.little_edge_m + geometry.index_edge_m),
        ],
        dtype=float,
    )
    # World up in the wrist frame; the little finger's outer edge, at the
    # knuckle line and at the open fingertip, must sit above the block's
    # underside by the clearance. The fingertip too: fingers pitched down
    # toward the forearm put a straight little finger into the table, and
    # from there every curl that lifts it off goes into the block.
    up = rotation.T @ np.asarray([0.0, 1.0, 0.0])
    if up[2] > 1e-6:
        vertical = block.extent_along(np.asarray([0.0, 1.0, 0.0]))
        for reach in (geometry.knuckle_m, geometry.little_tip_m):
            edge = np.asarray([0.0, reach, geometry.little_edge_m], dtype=float)
            highest = (
                float(np.dot(edge, up)) - pocket[0] * up[0] - pocket[1] * up[1] + vertical - SUPPORT_CLEARANCE_M
            ) / up[2]
            pocket[2] = min(pocket[2], highest)
    _ = along_thumb
    return pocket

def grasp_finger_axis(target_object: SceneObject, primitive: MotionPrimitive) -> np.ndarray:
    """The world direction the fingers point along to hold ``target_object``:
    into the block along its grasp socket's approach, flattened to the
    horizontal so a hand reaching down to a table does not stab it."""

    socket = next(
        (item for item in target_object.sockets if item.id == primitive.socket_id),
        next((item for item in target_object.sockets if item.role == AffordanceRole.GRASP), None),
    )
    rotation = Rotation.from_quat(target_object.transform.rotation.as_list())
    approach = (
        -rotation.apply(np.asarray(socket.approach_normal.as_list(), dtype=float))
        if socket is not None
        else np.asarray([0.0, 0.0, 1.0], dtype=float)
    )
    approach[1] = 0.0
    if float(np.linalg.norm(approach)) < 1e-6:
        approach = np.asarray([0.0, 0.0, 1.0], dtype=float)
    return approach / float(np.linalg.norm(approach))


def grasp_palm_normal(hand: Hand, target_object: SceneObject, primitive: MotionPrimitive) -> np.ndarray:
    """The world direction the palm faces holding ``target_object``: across
    the finger axis, from the hand's own side of the body toward the block,
    so the palm lands on the block's side face and the thumb comes over its
    top."""

    fingers = grasp_finger_axis(target_object, primitive)
    across = np.cross(np.asarray([0.0, 1.0, 0.0]), fingers)
    across /= max(float(np.linalg.norm(across)), 1e-8)
    return -across if hand == Hand.LEFT else across


@dataclass(frozen=True)
class Seat:
    """The hand pose that holds a block: where the wrist is and how it faces."""

    rotation: np.ndarray
    wrist: np.ndarray
    pocket: np.ndarray
    palm_normal: np.ndarray
    finger_axis: np.ndarray




def _forearm_direction(bones: Mapping[str, BonePose], side: str) -> np.ndarray:
    kinematics = rig_kinematics()
    matrices = kinematics.world_matrices(bones)
    elbow = matrices[kinematics.node_by_canonical[f"{side}LowerArm"]][:3, 3]
    wrist = matrices[kinematics.node_by_canonical[f"{side}Hand"]][:3, 3]
    return _unit(wrist - elbow)


def _posed(bones: Mapping[str, BonePose], arm: Mapping[str, Quat]) -> dict[str, BonePose]:
    posed = dict(bones)
    posed.update({name: BonePose(rotation=rotation) for name, rotation in arm.items()})
    return posed


def hand_world(bones: Mapping[str, BonePose], side: str) -> tuple[np.ndarray, Rotation]:
    """The wrist origin and hand rotation the rig's own FK gives a pose."""

    kinematics = rig_kinematics()
    matrix = kinematics.world_matrices(bones)[kinematics.node_by_canonical[f"{side}Hand"]]
    return matrix[:3, 3].copy(), Rotation.from_matrix(matrix[:3, :3])


def elbow_direction(bones: Mapping[str, BonePose], side: str) -> np.ndarray:
    """Where the elbow lies relative to the shoulder, as a bend hint."""

    kinematics = rig_kinematics()
    matrices = kinematics.world_matrices(bones)
    shoulder = matrices[kinematics.node_by_canonical[f"{side}UpperArm"]][:3, 3]
    elbow = matrices[kinematics.node_by_canonical[f"{side}LowerArm"]][:3, 3]
    return _unit(elbow - shoulder)


def rotation_toward(start: Rotation, end: Rotation, alpha: float) -> Rotation:
    """The rotation ``alpha`` of the way from ``start`` to ``end``."""

    delta = end * start.inv()
    return Rotation.from_rotvec(delta.as_rotvec() * alpha) * start




def solve_hand_along(
    bones: Mapping[str, BonePose],
    side: str,
    wrist: np.ndarray,
    rotation: np.ndarray,
    bend_hint: np.ndarray | None,
) -> dict[str, Quat]:
    """The arm reaching ``wrist`` with the hand at world ``rotation``.

    The roll about the forearm goes into the forearm as pronation; the wrist
    carries the flexion and deviation between the forearm and the fingers,
    and no more of it than the swing budget: past that the hand turns with
    the forearm (a block lifted overhead tilts with the arm that lifts it).
    """

    kinematics = rig_kinematics()
    arm = kinematics.solve_arm(
        bones,
        side,
        wrist,
        hand_world_rotation=rotation,
        bend_hint_world=bend_hint,
        allow_mirror=False,
        absorb_hand_twist=True,
    )
    turn = swing_excess(_forearm_direction(_posed(bones, arm), side), rotation)
    if turn is None:
        return arm
    return kinematics.solve_arm(
        bones,
        side,
        wrist,
        hand_world_rotation=turn.as_matrix() @ rotation,
        bend_hint_world=bend_hint,
        allow_mirror=False,
        absorb_hand_twist=True,
    )


def swing_excess(forearm: np.ndarray, rotation: np.ndarray) -> Rotation | None:
    """The turn that brings a hand rotation's finger axis to within the swing
    budget of ``forearm``, or ``None`` when it already is."""

    fingers = finger_axis_of(rotation)
    axis = np.cross(fingers, forearm)
    sine = float(np.linalg.norm(axis))
    swing = math.atan2(sine, float(np.dot(fingers, forearm)))
    excess = swing - SWING_BUDGET_RAD
    if excess <= 1e-6 or sine < 1e-9:
        return None
    return Rotation.from_rotvec(axis / sine * excess)


def grasp_turn(finger_axis: np.ndarray, parameters: PrimitiveParameters) -> Rotation:
    """The yaw and pitch the planner's arm knobs put on a grasp frame."""

    fingers = _unit(finger_axis)
    up = np.asarray([0.0, 1.0, 0.0])
    lateral = np.cross(up, fingers)
    lateral /= max(float(np.linalg.norm(lateral)), 1e-8)
    yaw = Rotation.from_rotvec(up * parameters.lateral_offset * LATERAL_OFFSET_YAW_RAD)
    pitch = Rotation.from_rotvec(lateral * parameters.arm_height * ARM_HEIGHT_PITCH_RAD)
    return yaw * pitch


def swivel_hint(
    bones: Mapping[str, BonePose], side: str, wrist: np.ndarray, hint: np.ndarray, swivel: float
) -> np.ndarray:
    """``hint`` swivelled about the shoulder-wrist line by ``swivel`` units."""

    kinematics = rig_kinematics()
    shoulder = kinematics.world_matrices(bones)[kinematics.node_by_canonical[f"{side}UpperArm"]][:3, 3]
    axis = _unit(np.asarray(wrist, dtype=float) - shoulder)
    sign = 1.0 if side == Hand.LEFT.value else -1.0
    return Rotation.from_rotvec(axis * sign * swivel * ELBOW_SWIVEL_RAD).apply(hint)


def solve_seat(
    bones: Mapping[str, BonePose],
    side: str,
    block: Box,
    finger_axis: np.ndarray,
    palm_normal: np.ndarray,
    bend_hint: np.ndarray | None,
    parameters: PrimitiveParameters | None = None,
) -> tuple[dict[str, Quat], Seat, np.ndarray | None]:
    """The arm holding ``block`` with the fingers along ``finger_axis`` and
    the palm facing ``palm_normal``: the pocket goes on the block's centre and
    the wrist is wherever that puts it. Returns the arm, the seat and the
    bend hint the arm was solved with (``parameters`` may swivel it)."""

    geometry = hand_geometry(side)
    finger_axis = _unit(finger_axis)
    palm_normal = np.asarray(palm_normal, dtype=float)
    depth_m = 0.0
    if parameters is not None:
        turn = grasp_turn(finger_axis, parameters)
        finger_axis = _unit(turn.apply(finger_axis))
        palm_normal = turn.apply(palm_normal)
        depth_m = parameters.arm_depth * ARM_DEPTH_M
    arm: dict[str, Quat] = {}
    rotation = hand_rotation(side, finger_axis, palm_normal)
    pocket = pocket_local(geometry, rotation, block, depth_m)
    wrist = block.centre - rotation @ pocket
    if parameters is not None and bend_hint is not None and parameters.elbow_swivel != 0.0:
        bend_hint = swivel_hint(bones, side, wrist, bend_hint, parameters.elbow_swivel)
    for _ in range(SEAT_PASSES):
        rotation = hand_rotation(side, finger_axis, palm_normal)
        pocket = pocket_local(geometry, rotation, block, depth_m)
        wrist = block.centre - rotation @ pocket
        arm = solve_hand_along(bones, side, wrist, rotation, bend_hint)
        # The wrist may bend only so far off the forearm; past the budget the
        # whole grasp frame turns toward the forearm instead, so the pocket
        # is placed for the hand that will actually be rendered.
        turn = swing_excess(_forearm_direction(_posed(bones, arm), side), rotation)
        if turn is None:
            break
        finger_axis = _unit(turn.apply(finger_axis))
        palm_normal = turn.apply(palm_normal)
    seat = Seat(
        rotation=rotation,
        wrist=wrist,
        pocket=pocket,
        palm_normal=palm_normal_of(side, rotation),
        finger_axis=finger_axis_of(rotation),
    )
    return arm, seat, bend_hint


def close_to_contact(
    hand: Hand,
    shape: HandShape,
    parameters: PrimitiveParameters,
    body: Mapping[str, Quat],
    boxes: list[Box],
) -> dict[str, float]:
    """The curl nearest each digit's curl in ``shape`` at which it touches
    nothing in ``boxes``.

    ``body`` is the pose the hand is closing in: trunk and arm. Each digit is
    bisected on its own against its own skin. A digit that meets nothing at
    the shape's curl keeps it; one that does closes back to first contact;
    one that is inside something even fully open (a straight finger reaching
    the table an open hand rests beside) curls up off it instead.
    """

    mesh = hand_mesh(hand.value)
    curls = effective_curls(shape, parameters)

    def depth(digit: str, curl: float) -> float:
        # Penetration of the digit's skin into the nearest box: positive
        # inside, negative the distance to it. Clear means at least the
        # margin outside.
        pose = dict(body)
        pose.update(hand_pose(hand, shape, parameters, curl_overrides={digit: curl}))
        points = mesh.world({name: BonePose(rotation=rotation) for name, rotation in pose.items()})[
            mesh.groups[digit]
        ]
        return max(
            -float(box_signed_distance(points, box.centre, box.rotation, box.half_extents).min())
            for box in boxes
        )

    closed: dict[str, float] = {}
    touching: dict[str, bool] = {}
    samples = np.linspace(0.0, 1.0, CLOSURE_SCAN + 1)
    for digit in FINGERS:
        target = curls[digit]
        clear = [depth(digit, float(value)) <= -TOUCH_MARGIN_M for value in samples]

        def boundary(inside: float, outside: float, digit: str = digit) -> float:
            # The clear curl nearest ``inside`` between a clear ``outside``
            # and a penetrating ``inside`` sample.
            for _ in range(CLOSURE_STEPS):
                middle = 0.5 * (inside + outside)
                if depth(digit, middle) <= -TOUCH_MARGIN_M:
                    outside = middle
                else:
                    inside = middle
            return outside

        # Where the finger starts: straight, or if a straight finger is
        # already in something (the table an open hand rests beside), the
        # first curl at which it comes clear of it.
        first_clear = next((index for index, value in enumerate(clear) if value), None)
        if first_clear is None:
            closed[digit] = target
            touching[digit] = True
            continue
        floor = 0.0 if first_clear == 0 else boundary(float(samples[first_clear - 1]), float(samples[first_clear]))
        # Then closing from there: the first sample that meets something
        # bounds the curl, the finger never sweeps through to a clear curl
        # on the far side of the block.
        blocked = next(
            (index for index in range(first_clear + 1, len(samples)) if not clear[index]),
            None,
        )
        if blocked is None or float(samples[blocked]) > target:
            if target <= floor:
                # The shape is straighter than the lift-off: lifted it is.
                closed[digit] = floor
                touching[digit] = True
            elif depth(digit, target) <= -TOUCH_MARGIN_M:
                closed[digit] = target
                touching[digit] = False
            else:
                previous = max(
                    floor,
                    max(float(value) for value, ok in zip(samples, clear) if ok and value <= target),
                )
                closed[digit] = boundary(target, previous)
                touching[digit] = True
            continue
        closed[digit] = max(floor, boundary(float(samples[blocked]), float(samples[blocked - 1])))
        touching[digit] = True
    # A finger that meets nothing closes with its nearest finger that did,
    # rather than curling into a fist beside a block the others are holding.
    fingers = [digit for digit in FINGERS if digit != "Thumb"]
    for index, digit in enumerate(fingers):
        if touching[digit]:
            continue
        nearest = min(
            (other for other in fingers if touching[other]),
            key=lambda other: abs(fingers.index(other) - index),
            default=None,
        )
        if nearest is not None:
            closed[digit] = min(closed[digit], closed[nearest])
    return closed


def attach_offset(
    hand_rotation_world: Rotation, wrist: np.ndarray, block: Box
) -> tuple[np.ndarray, Rotation]:
    """The block's pose in the hand's frame, for carrying it rigidly."""

    return (
        hand_rotation_world.inv().apply(block.centre - wrist),
        hand_rotation_world.inv() * Rotation.from_matrix(block.rotation),
    )


def carried_transform(
    hand_rotation_world: Rotation,
    wrist: np.ndarray,
    local_offset: np.ndarray,
    local_rotation: Rotation,
) -> Transform:
    """Where a block attached at ``local_offset`` is, for this hand pose."""

    centre = wrist + hand_rotation_world.apply(local_offset)
    quat = (hand_rotation_world * local_rotation).as_quat()
    from .models import Vec3

    return Transform(
        translation=Vec3(x=float(centre[0]), y=float(centre[1]), z=float(centre[2])),
        rotation=Quat(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])),
    )


def arc_path(start: np.ndarray, end: np.ndarray, alpha: float, rise_m: float) -> np.ndarray:
    """A point ``alpha`` of the way from ``start`` to ``end`` along an arc
    that rises ``rise_m`` above the straight line at its middle."""

    lift = np.asarray([0.0, rise_m * math.sin(math.pi * alpha), 0.0])
    return np.asarray(start, dtype=float) + (np.asarray(end, dtype=float) - start) * alpha + lift


def rotate_toward(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """The unit direction ``alpha`` of the way from ``start`` to ``end`` along
    the shorter great-circle arc."""

    first, second = _unit(start), _unit(end)
    axis = np.cross(first, second)
    sine = float(np.linalg.norm(axis))
    angle = math.atan2(sine, float(np.dot(first, second)))
    if sine < 1e-9:
        return first if angle < math.pi / 2 else second
    return Rotation.from_rotvec(axis / sine * angle * alpha).apply(first)


__all__ = [
    "Box",
    "HandGeometry",
    "Seat",
    "arc_path",
    "attach_offset",
    "carried_transform",
    "close_to_contact",
    "elbow_direction",
    "elbow_hang",
    "finger_axis_of",
    "grasp_finger_axis",
    "grasp_palm_normal",
    "grasp_turn",
    "hand_geometry",
    "hand_rotation",
    "hand_world",
    "palm_normal_of",
    "pocket_local",
    "rotate_toward",
    "rotation_toward",
    "solve_hand_along",
    "solve_seat",
    "swing_excess",
    "swivel_hint",
]
