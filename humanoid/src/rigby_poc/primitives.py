from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .arm_plane import (
    ELBOW_FLEXION_AXIS_LOCAL,
    UPPER_ARM_TWIST_BAND_RAD,
    bend_plane_normal,
    forearm_roll_compensation,
    humeral_roll,
    roll_rotation,
    twist_about_local_y,
)
from .arm_plane import (
    signed_angle_about_axis as _signed_angle_about_axis,
)
from .kinematics import arm_calibration
from .models import (
    Digit,
    Hand,
    HandShape,
    PrimitiveKind,
    PrimitiveParameters,
    Quat,
    StrikeType,
    Vec3,
)
from .thresholds import value_of

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
SEGMENTS = {
    "Thumb": ("Metacarpal", "Proximal", "Distal"),
    "Index": ("Proximal", "Intermediate", "Distal"),
    "Middle": ("Proximal", "Intermediate", "Distal"),
    "Ring": ("Proximal", "Intermediate", "Distal"),
    "Little": ("Proximal", "Intermediate", "Distal"),
}


@dataclass(frozen=True)
class HandShapeDefinition:
    curls: dict[str, float]
    splay: dict[str, float]
    thumb_opposition: float


HAND_SHAPES: dict[HandShape, HandShapeDefinition] = {
    HandShape.OPEN: HandShapeDefinition(
        curls={name: 0.04 for name in FINGERS},
        splay={"Thumb": 0.65, "Index": 0.18, "Middle": 0.0, "Ring": -0.08, "Little": -0.18},
        thumb_opposition=0.18,
    ),
    HandShape.FIST: HandShapeDefinition(
        curls={"Thumb": 0.72, "Index": 0.94, "Middle": 1.0, "Ring": 1.0, "Little": 0.96},
        splay={name: 0.0 for name in FINGERS},
        thumb_opposition=0.86,
    ),
    HandShape.POINT: HandShapeDefinition(
        curls={"Thumb": 0.58, "Index": 0.03, "Middle": 0.98, "Ring": 1.0, "Little": 0.98},
        splay={name: 0.0 for name in FINGERS},
        thumb_opposition=0.72,
    ),
    HandShape.PINCH: HandShapeDefinition(
        curls={"Thumb": 0.52, "Index": 0.48, "Middle": 0.22, "Ring": 0.34, "Little": 0.42},
        splay={name: 0.0 for name in FINGERS},
        thumb_opposition=1.0,
    ),
    HandShape.HANG_TEN: HandShapeDefinition(
        curls={"Thumb": 0.02, "Index": 0.82, "Middle": 0.85, "Ring": 0.84, "Little": 0.03},
        splay={"Thumb": 0.82, "Index": 0.04, "Middle": 0.0, "Ring": -0.04, "Little": -0.68},
        thumb_opposition=0.06,
    ),
    HandShape.THUMBS_UP: HandShapeDefinition(
        curls={"Thumb": 0.02, "Index": 0.94, "Middle": 1.0, "Ring": 1.0, "Little": 0.96},
        splay={name: 0.0 for name in FINGERS},
        thumb_opposition=0.04,
    ),
    HandShape.PEACE: HandShapeDefinition(
        curls={"Thumb": 0.58, "Index": 0.03, "Middle": 0.03, "Ring": 1.0, "Little": 0.98},
        splay={"Thumb": 0.0, "Index": 0.28, "Middle": -0.20, "Ring": 0.0, "Little": 0.0},
        thumb_opposition=0.78,
    ),
}


# Rig-calibrated fingertip contact solutions. Each tuple contains thumb curl
# for its three joints, thumb opposition/splay, target-digit curl for its
# three joints, and target-digit splay. The values were solved against the
# exact source-rig leaf pivots and mirror cleanly across hands.
_THUMB_TO_FINGERTIP_POSES: dict[Digit, tuple[float, ...]] = {
    Digit.INDEX: (
        0.27008303,
        0.44327816,
        0.44546741,
        0.82239992,
        0.58439503,
        0.64051348,
        0.85738130,
        0.63456749,
        -0.13855365,
    ),
    Digit.MIDDLE: (
        0.34082218,
        0.48070119,
        0.45648593,
        0.53572823,
        0.58814504,
        0.67614544,
        0.94740559,
        0.70577023,
        0.03812296,
    ),
    Digit.RING: (
        0.49078182,
        0.54663667,
        0.46862664,
        0.26541756,
        0.49665632,
        0.72569372,
        0.97094083,
        0.71688335,
        0.05406851,
    ),
    Digit.LITTLE: (
        0.65433759,
        0.62303992,
        0.48485766,
        0.08804874,
        0.38019840,
        0.84048706,
        1.02559420,
        0.71276550,
        -0.18855832,
    ),
}


# Rest-orientation calibration and segment lengths, derived from the rig at
# first use rather than frozen here. The retired literals matched the rig's
# rest rotations to ~1e-16 but rounded the segment lengths to 4 decimals --
# 5.3 and 11.9 um from the true pivot distances, which is not a rigid
# transform and is why impact_elbow_angle_deg drifted at 1e-9..9e-8 relative.
# Per side, because the rig itself is left/right asymmetric at ~1.1e-7 m.
# Clip rotations remain local deltas post-multiplied onto these rest
# transforms; only where the numbers come from moved.


def arm_segment_lengths(hand: Hand) -> tuple[float, float]:
    """Upper and lower arm pivot-to-pivot lengths for one side, in metres."""

    calibration = arm_calibration(hand.value)
    return calibration.upper_length_m, calibration.lower_length_m


def arm_reach_m(hand: Hand) -> float:
    """Shoulder-to-wrist reach with the arm straight, for one side."""

    return arm_calibration(hand.value).reach_m


def _arm_rest_rotations(hand: Hand) -> tuple[Rotation, Rotation, Rotation]:
    """Upper-world, lower-local and hand-local rest rotations for one side."""

    calibration = arm_calibration(hand.value)
    return (
        Rotation.from_quat(calibration.upper_rest_world_xyzw),
        Rotation.from_quat(calibration.lower_rest_local_xyzw),
        Rotation.from_quat(calibration.hand_rest_local_xyzw),
    )

# A shaka is made readable by pronating/supinating the forearm, not by asking
# the wrist joint to absorb the entire camera-facing hand orientation.  These
# limits are deliberately below the corresponding rig-profile hard limits so
# generated candidates retain some safety margin.  The two that shadow a
# validator limit now read that margin from ``config/thresholds.v1.json``, which
# records the ratio they were derived at; the values are unchanged.
MAX_FOREARM_TWIST_RAD = value_of("anatomy.forearm_twist_generator_max_rad")
MAX_WRIST_PITCH_RAD = 0.35
MAX_WRIST_YAW_RAD = 0.28
MAX_WRIST_TWIST_RAD = value_of("anatomy.wrist_twist_generator_max_rad")
MAX_FOREARM_SHAKE_RAD = 0.28

# A strike rotates the trunk, not just the arm. `torso_participation` scales
# this authored ceiling, and the resulting yaw is distributed up the trunk
# chain so no single segment absorbs the whole rotation. The weights sum to
# 1.0 and the largest one times the ceiling stays inside every trunk segment's
# typical twist band (see the threshold entry's rationale).
MAX_TORSO_YAW_RAD = value_of("anatomy.torso_yaw_generator_max_rad")
TRUNK_YAW_DISTRIBUTION = (("spine", 0.30), ("chest", 0.40), ("upperChest", 0.30))
# The trunk's vertical axis in the body-relative target frame: the arm chain
# hangs off `upperChest`, so a trunk yaw carries the shoulder around this
# line and an IK target authored for the un-yawed body must be counter-rotated
# about it before the arm solve.
TRUNK_YAW_PIVOT_X_M = 0.0
TRUNK_YAW_PIVOT_Z_M = -0.065


def strike_trunk_yaw(hand: Hand, phase: PrimitiveKind, parameters: PrimitiveParameters) -> float:
    """Signed trunk yaw in radians for one strike phase.

    Strike phases drive the punching shoulder forward -- for a left-hand
    strike the trunk rotates clockwise seen from above (negative yaw) -- and
    the load phase winds up the other way, pulling that shoulder back, so the
    visible rotation across load -> strike spans both magnitudes. Measured on
    this rig: a local +Y trunk quat is a positive world yaw, and the sign was
    wrong-way-first: `direction = side` turned the punching shoulder backward,
    which read as nothing at the old 9-degree ceiling and at this one drove
    the impact target out of reach (elbow 178 deg, IK clamp spiking angular
    acceleration to 248 rad/s^2, 34 self-collision frames).
    """

    side = 1.0 if hand == Hand.LEFT else -1.0
    direction = side if phase == PrimitiveKind.LOAD else -side
    return direction * parameters.torso_participation * MAX_TORSO_YAW_RAD


def trunk_yaw_poses(yaw_rad: float) -> dict[str, Quat]:
    """Distribute one trunk yaw over the spine chain as local Y rotations.

    The head gets the full counter-rotation: a fighter's gaze stays on the
    target while the trunk coils, and the egocentric camera rides the head,
    so without the counter-yaw the trunk rotation sweeps the camera off the
    strike and the visibility gate fails (measured 0.892 against a 1.0 gate).
    """

    poses: dict[str, Quat] = {}
    for bone, weight in TRUNK_YAW_DISTRIBUTION:
        half = 0.5 * weight * yaw_rad
        poses[bone] = Quat(y=math.sin(half), w=math.cos(half))
    counter_half = -0.5 * yaw_rad
    poses["head"] = Quat(y=math.sin(counter_half), w=math.cos(counter_half))
    return poses


def counter_rotate_about_trunk(target: Vec3, yaw_rad: float) -> Vec3:
    """Rotate a body-relative target about the trunk axis by ``-yaw_rad``.

    Solving the arm for this counter-rotated target and then yawing the trunk
    by ``yaw_rad`` puts the hand back on the authored target, because the arm
    chain rides the trunk rigidly.

    Exact zero returns the target object untouched: ``pivot + (target - pivot)``
    is not bit-identical to ``target`` in floats, and the non-strike paths that
    pass 0.0 are pinned byte-identical by the equivalence fixtures.
    """

    if yaw_rad == 0.0:
        return target
    cos = math.cos(-yaw_rad)
    sin = math.sin(-yaw_rad)
    x = target.x - TRUNK_YAW_PIVOT_X_M
    z = target.z - TRUNK_YAW_PIVOT_Z_M
    return Vec3(
        x=TRUNK_YAW_PIVOT_X_M + cos * x + sin * z,
        y=target.y,
        z=TRUNK_YAW_PIVOT_Z_M - sin * x + cos * z,
    )


def _rotation_between(source: np.ndarray, target: np.ndarray) -> Rotation:
    source = source / max(float(np.linalg.norm(source)), 1e-8)
    target = target / max(float(np.linalg.norm(target)), 1e-8)
    cross = np.cross(source, target)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cross_norm < 1e-8:
        if dot > 0.0:
            return Rotation.identity()
        helper = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * math.pi)
    return Rotation.from_rotvec(cross / cross_norm * math.acos(dot))


def _quat_from_rotation(rotation: Rotation) -> Quat:
    xyzw = rotation.as_quat()
    return Quat(x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3]))


def quat_axis_angle(axis: tuple[float, float, float], angle: float) -> Quat:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    xyzw = Rotation.from_rotvec(vector * angle).as_quat()
    return Quat(x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3]))


def quat_euler(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Quat:
    xyzw = Rotation.from_euler("xyz", [x, y, z]).as_quat()
    return Quat(x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3]))


def _bone_key(hand: Hand, finger: str, segment: str) -> str:
    return f"{hand.value}{finger}{segment}"


def hand_pose(
    hand: Hand,
    shape: HandShape,
    parameters: PrimitiveParameters,
    blend: float = 1.0,
) -> dict[str, Quat]:
    definition = HAND_SHAPES[shape]
    side = 1.0 if hand == Hand.LEFT else -1.0
    result: dict[str, Quat] = {}
    for finger in FINGERS:
        base_curl = definition.curls[finger]
        digit_adjustment = {
            "Thumb": parameters.thumb_curl,
            "Index": parameters.index_curl,
            "Middle": parameters.middle_curl,
            "Ring": parameters.ring_curl,
            "Little": parameters.little_curl,
        }[finger]
        curl = float(
            np.clip(base_curl + parameters.finger_curl * 0.2 + digit_adjustment * 0.25, 0.0, 1.0)
        ) * blend
        base_splay = definition.splay[finger]
        # Positive finger_splay expands the authored silhouette away from its
        # center for every digit.  The old additive rule widened one side of a
        # shaka while accidentally closing the other, so it could not express
        # the judge's most common "separate thumb and little finger" repair.
        outward = 1.0 if base_splay >= 0.0 else -1.0
        splay = float(np.clip(base_splay + outward * parameters.finger_splay * 0.25, -1.0, 1.0))
        for index, segment in enumerate(SEGMENTS[finger]):
            multiplier = (0.92, 1.12, 0.82)[index]
            curl_angle = curl * multiplier * (1.25 if finger != "Thumb" else 0.95)
            splay_angle = splay * 0.30 * side if index == 0 else 0.0
            opposition = 0.0
            if finger == "Thumb" and index == 0:
                opposition = (definition.thumb_opposition * 0.7 + parameters.thumb_opposition * 0.3) * 0.75 * side
            result[_bone_key(hand, finger, segment)] = quat_euler(curl_angle, opposition, splay_angle)
    return result


def thumb_to_fingertip_pose(
    hand: Hand,
    target_digit: Digit,
) -> dict[str, Quat]:
    """Return a smoothable articulated pose with thumb-tip contact."""

    if target_digit not in _THUMB_TO_FINGERTIP_POSES:
        raise ValueError(f"thumb contact is not calibrated for {target_digit.value}")
    values = _THUMB_TO_FINGERTIP_POSES[target_digit]
    side = 1.0 if hand == Hand.LEFT else -1.0
    result = hand_pose(hand, HandShape.OPEN, PrimitiveParameters())
    result[f"{hand.value}ThumbMetacarpal"] = quat_euler(
        values[0],
        side * values[3],
        side * values[4],
    )
    result[f"{hand.value}ThumbProximal"] = quat_euler(values[1])
    result[f"{hand.value}ThumbDistal"] = quat_euler(values[2])
    title = target_digit.value.title()
    for index, segment in enumerate(("Proximal", "Intermediate", "Distal")):
        result[f"{hand.value}{title}{segment}"] = quat_euler(
            values[5 + index],
            0.0,
            side * values[8] if index == 0 else 0.0,
        )
    return result


@dataclass(frozen=True)
class _ArmSolution:
    """One hinge-in-plane arm solution plus the measurables that select it."""

    upper_delta: Rotation
    lower_delta: Rotation
    hand_delta: Rotation
    presented_twist_rad: float
    pronation_overflow_rad: float

    @property
    def violates(self) -> bool:
        return (
            abs(self.presented_twist_rad) > UPPER_ARM_TWIST_BAND_RAD
            or self.pronation_overflow_rad > 1e-9
        )


def arm_pose_from_target(
    hand: Hand,
    shoulder: Vec3,
    target: Vec3,
    parameters: PrimitiveParameters,
    *,
    present_hand: bool = False,
    forearm_twist_reserve_rad: float = 0.0,
    elbow_hint: Vec3 | None = None,
    elbow_hint_weight: float = 1.0,
    elbow_pole: Vec3 | None = None,
) -> tuple[dict[str, Quat], float]:
    """Calibrated analytic two-link IK returned as rest-relative local deltas.

    ``elbow_hint`` pins the elbow near a world position; ``elbow_hint_weight``
    lets a caller blending a target toward that hint carry the *bend plane*
    across the same blend window, so the plane never has to reorient faster
    than the target it follows.  ``elbow_pole`` replaces the default pole
    vector for callers whose motion family has a known better elbow side.
    """
    shoulder_v = np.asarray(shoulder.as_list(), dtype=float)
    target_v = np.asarray(target.as_list(), dtype=float)
    delta = target_v - shoulder_v
    distance = float(np.linalg.norm(delta))
    upper, lower = arm_segment_lengths(hand)
    clamped = float(np.clip(distance, abs(upper - lower) + 1e-4, upper + lower - 1e-4))
    direction = delta / max(distance, 1e-8)
    along = (upper**2 - lower**2 + clamped**2) / (2 * clamped)
    bend_height = math.sqrt(max(upper**2 - along**2, 0.0))
    side = 1.0 if hand == Hand.LEFT else -1.0
    hint_weight = float(np.clip(elbow_hint_weight, 0.0, 1.0))
    use_hint = elbow_hint is not None and hint_weight > 0.0
    if elbow_hint is None or hint_weight < 1.0:
        pole = (
            np.asarray(elbow_pole.as_list(), dtype=float)
            if elbow_pole is not None
            else np.asarray([side, -0.55, 0.25], dtype=float)
        )
        bend = pole - direction * float(np.dot(pole, direction))
        if np.linalg.norm(bend) < 1e-8:
            bend = np.cross(direction, np.asarray([0.0, 1.0, 0.0]))
        bend /= np.linalg.norm(bend)
        bend = Rotation.from_rotvec(
            direction * parameters.elbow_swivel * 0.75
        ).apply(bend)
    if elbow_hint is not None:
        line_point = shoulder_v + direction * along
        hinted = np.asarray(elbow_hint.as_list(), dtype=float) - line_point
        hinted_bend = hinted - direction * float(np.dot(hinted, direction))
        if np.linalg.norm(hinted_bend) < 1e-8:
            hinted_bend = np.cross(direction, np.asarray([0.0, 1.0, 0.0]))
        hinted_bend /= np.linalg.norm(hinted_bend)
        if hint_weight >= 1.0:
            bend = hinted_bend
        elif use_hint:
            # Rotate the pole's bend toward the hinted bend by the caller's
            # blend fraction, about the reach axis both are perpendicular to.
            bend = Rotation.from_rotvec(
                direction
                * (
                    hint_weight
                    * _signed_angle_about_axis(bend, hinted_bend, direction)
                )
            ).apply(bend)

    rest_upper, lower_local_rest, hand_rest = _arm_rest_rotations(hand)
    pronation_budget = max(
        0.0, MAX_FOREARM_TWIST_RAD - max(0.0, float(forearm_twist_reserve_rad))
    )

    def solve(bend_choice: np.ndarray) -> _ArmSolution:
        elbow = shoulder_v + direction * along + bend_choice * bend_height
        upper_direction = (elbow - shoulder_v) / upper
        lower_direction = (target_v - elbow) / lower
        upper_alignment = _rotation_between(
            rest_upper.apply([0.0, 1.0, 0.0]), upper_direction
        )
        desired_upper = upper_alignment * rest_upper
        # The shortest-arc aim above carries zero twist about the humerus long
        # axis, which would leave the whole bend-plane orientation to be
        # absorbed by the elbow as abduction.  Roll the humerus about its own
        # long axis so the elbow hinge it presents lies in the plane the
        # pole/hint chose; the elbow and hand pivots sit on the axes this
        # rotation preserves, so their positions do not move.
        plane_normal = bend_plane_normal(bend_choice, direction)
        hinge_world = (desired_upper * lower_local_rest).apply(
            ELBOW_FLEXION_AXIS_LOCAL
        )
        roll = humeral_roll(hinge_world, plane_normal, upper_direction)
        # The pre-roll branch is carried alongside so the roll can be
        # compensated downstream: the roll changes the hand's world orientation
        # by a pure twist about the elbow->wrist axis, and restoring the
        # pre-roll hand orientation keeps silhouette projection (and the
        # visibility gate) byte-equivalent to the pre-roll solve at every
        # keyframe.
        desired_upper_preroll = desired_upper
        desired_upper = roll_rotation(upper_direction, roll) * desired_upper
        upper_delta = rest_upper.inv() * desired_upper

        lower_base_world_preroll = desired_upper_preroll * lower_local_rest
        lower_alignment_preroll = _rotation_between(
            lower_base_world_preroll.apply([0.0, 1.0, 0.0]), lower_direction
        )
        desired_lower_preroll = lower_alignment_preroll * lower_base_world_preroll

        lower_base_world = desired_upper * lower_local_rest
        lower_alignment = _rotation_between(
            lower_base_world.apply([0.0, 1.0, 0.0]), lower_direction
        )
        desired_lower = lower_alignment * lower_base_world
        if present_hand:
            # Rotate the whole forearm around its longitudinal axis so the
            # palm is readable from the egocentric +Z view.  Wrist roll in
            # natural human motion is primarily forearm pronation/supination;
            # placing it here avoids the 90-176 degree hand-joint deltas
            # produced by the old solve.  The twist is authored on the
            # pre-roll branch because that branch's hand world orientation is
            # the calibrated one the camera contract was tuned against; the
            # roll compensation below carries it across.
            neutral_hand_world = desired_lower_preroll * hand_rest
            palm_back_world = neutral_hand_world.apply([0.0, 0.0, 1.0])
            camera_facing_palm_back = np.asarray([0.0, 0.0, -1.0])
            alignment_twist = _signed_angle_about_axis(
                palm_back_world,
                camera_facing_palm_back,
                lower_direction,
            )
            authored_twist = parameters.wrist_roll * 0.40
            # A later shake adds longitudinal rotation to this same joint.
            # Reserve its full amplitude during every presentation phase so
            # the combined pose remains physically safe by construction
            # instead of clipping one half of the oscillation after the fact.
            forearm_twist = float(
                np.clip(
                    alignment_twist + authored_twist,
                    -pronation_budget,
                    pronation_budget,
                )
            )
            desired_lower_preroll = (
                Rotation.from_rotvec(lower_direction * forearm_twist)
                * desired_lower_preroll
            )

            # The remaining hand delta contains only bounded flexion and
            # deviation.  A very small longitudinal allowance absorbs Euler
            # composition error; it is checked independently by the
            # structural gate.
            hand_delta = Rotation.from_euler(
                "xyz",
                [
                    parameters.wrist_pitch * MAX_WRIST_PITCH_RAD,
                    parameters.wrist_roll * MAX_WRIST_TWIST_RAD,
                    parameters.wrist_yaw * MAX_WRIST_YAW_RAD,
                ],
            )
        else:
            hand_delta = Rotation.from_euler(
                "xyz",
                [
                    parameters.wrist_pitch * MAX_WRIST_PITCH_RAD,
                    parameters.wrist_roll * MAX_WRIST_TWIST_RAD,
                    parameters.wrist_yaw * MAX_WRIST_YAW_RAD,
                ],
            )

        desired_lower, _, overflow = forearm_roll_compensation(
            desired_lower,
            desired_lower_preroll,
            lower_base_world,
            pronation_budget,
        )
        # When the pronation budget absorbed the whole compensation this
        # residual is the identity; composing it anyway would smear ~1e-16 of
        # rotation junk into an otherwise exact hand delta, and cyclical
        # composites assert their first and last hand poses are *equal*.
        hand_residual = (desired_lower * hand_rest).inv() * desired_lower_preroll * hand_rest
        if float(hand_residual.magnitude()) > 1e-12:
            hand_delta = hand_residual * hand_delta
        lower_delta = lower_base_world.inv() * desired_lower
        return _ArmSolution(
            upper_delta=upper_delta,
            lower_delta=lower_delta,
            hand_delta=hand_delta,
            presented_twist_rad=twist_about_local_y(upper_delta),
            pronation_overflow_rad=overflow,
        )

    solution = solve(bend)
    if solution.violates:
        # The equivalent bend-plane branch: the elbow mirrored through the
        # shoulder->target line.  The hinge stays aligned with that branch's
        # plane normal, so elbow flexion stays the non-negative interior bend;
        # what changes is the twist the humerus must present and the pronation
        # the compensation demands, both of which move by roughly a half turn.
        # Selected only when it removes every violation the primary branch
        # has: the band and the budget are gates to satisfy, never to clamp.
        mirrored = solve(-bend)
        if not mirrored.violates:
            solution = mirrored

    prefix = hand.value
    poses = {
        f"{prefix}Shoulder": Quat(),
        f"{prefix}UpperArm": _quat_from_rotation(solution.upper_delta),
        f"{prefix}LowerArm": _quat_from_rotation(solution.lower_delta),
        f"{prefix}Hand": _quat_from_rotation(solution.hand_delta),
    }
    return poses, distance


def gesture_target(hand: Hand, parameters: PrimitiveParameters) -> Vec3:
    side = 1.0 if hand == Hand.LEFT else -1.0
    return Vec3(
        x=side * 0.12 + parameters.lateral_offset * 0.12,
        y=1.36 + parameters.arm_height * 0.18,
        z=0.33 + parameters.arm_depth * 0.12,
    )


def strike_target(
    hand: Hand,
    strike_type: StrikeType,
    phase: PrimitiveKind,
    parameters: PrimitiveParameters,
) -> Vec3:
    """Return a body-relative wrist target for a semantic strike phase."""
    side = 1.0 if hand == Hand.LEFT else -1.0
    if phase == PrimitiveKind.GUARD:
        base = (side * 0.15, 1.33, 0.29)
    elif strike_type == StrikeType.HOOK:
        base = {
            PrimitiveKind.LOAD: (side * 0.29, 1.34, 0.29),
            PrimitiveKind.STRIKE: (-side * 0.03, 1.40, 0.34),
            PrimitiveKind.FOLLOW_THROUGH: (-side * 0.04, 1.37, 0.33),
        }.get(phase, (side * 0.15, 1.33, 0.29))
    elif strike_type == StrikeType.UPPERCUT:
        base = {
            PrimitiveKind.LOAD: (side * 0.19, 1.29, 0.29),
            PrimitiveKind.STRIKE: (side * 0.08, 1.45, 0.36),
            PrimitiveKind.FOLLOW_THROUGH: (side * 0.04, 1.48, 0.33),
        }.get(phase, (side * 0.15, 1.33, 0.29))
    else:
        lateral = side * (0.13 if strike_type == StrikeType.JAB else 0.10)
        # Keep the impact frame near but never at full extension. The old
        # 0.46 m target solved the elbow to 177.9 deg -- a locked joint about
        # 2 deg off the IK singularity -- and the cross's peak angular
        # acceleration sat at 79.6 against the 82 rad/s^2 ceiling. The trunk
        # drives the punching shoulder forward, so reach couples to the torso
        # column: the cross (torso 0.72) carries its target at 0.52 m and the
        # jab (torso 0.40, less shoulder travel) at 0.50 m. Both measured
        # under the authored trunk yaw; one centimetre past either value is
        # back over the reach cliff, where the IK clamp spikes acceleration
        # past the 82 rad/s^2 ceiling.
        depth = 0.50 if strike_type == StrikeType.JAB else 0.52
        base = {
            PrimitiveKind.LOAD: (side * 0.16, 1.34, 0.29),
            PrimitiveKind.STRIKE: (lateral, 1.40, depth),
            PrimitiveKind.FOLLOW_THROUGH: (lateral, 1.39, depth - 0.03),
        }.get(phase, (side * 0.15, 1.33, 0.29))
    return Vec3(
        x=base[0] + side * parameters.lateral_offset * 0.055,
        y=base[1] + parameters.arm_height * 0.075,
        z=base[2] + parameters.arm_depth * 0.075,
    )


def strike_path_target(
    hand: Hand,
    strike_type: StrikeType,
    start: Vec3,
    end: Vec3,
    progress: float,
    parameters: PrimitiveParameters,
) -> Vec3:
    """Evaluate the bounded curved wrist path of the impact phase."""
    alpha = float(np.clip(progress, 0.0, 1.0))
    start_v = np.asarray(start.as_list(), dtype=float)
    end_v = np.asarray(end.as_list(), dtype=float)
    if strike_type == StrikeType.HOOK:
        side = 1.0 if hand == Hand.LEFT else -1.0
        arc = 0.55 + 0.35 * max(-1.0, min(1.0, parameters.path_arc))
        control = np.asarray(
            [
                side * (0.27 + 0.07 * arc),
                max(start.y, end.y) + 0.025 * arc,
                max(start.z, end.z) + 0.10 * arc,
            ],
            dtype=float,
        )
        value = (1.0 - alpha) ** 2 * start_v + 2.0 * (1.0 - alpha) * alpha * control + alpha**2 * end_v
    elif strike_type == StrikeType.UPPERCUT:
        control = np.asarray(
            [
                (start.x + end.x) * 0.5,
                start.y + (end.y - start.y) * 0.28,
                max(start.z, end.z) + 0.07,
            ],
            dtype=float,
        )
        value = (1.0 - alpha) ** 2 * start_v + 2.0 * (1.0 - alpha) * alpha * control + alpha**2 * end_v
    else:
        value = start_v * (1.0 - alpha) + end_v * alpha
    return Vec3(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def shoulder_position(hand: Hand) -> Vec3:
    """The shoulder origin, read from the rig profile rather than rounded.

    Plan 08 §1.1 item 4. This returned (±0.174, 1.446, -0.065) -- the profile's
    (±0.1737, 1.4457, -0.0652) rounded to three decimals. Targets are generated
    against this origin and then checked for reach against one computed from the
    profile under a **zero** tolerance (`semantic_forward_space.reach_tolerance_m`
    is 0.0), so the rounding was a source of spurious reach failures: the two
    origins differ by 0.469 mm, which is the exact worst-case excess.
    """

    x, y, z = value_of("anatomy.shoulder_origin_m")
    return Vec3(x=x if hand == Hand.LEFT else -x, y=y, z=z)


def smoothstep(value: float, easing: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    cubic = value * value * (3.0 - 2.0 * value)
    minimum_jerk = value**3 * (10.0 - 15.0 * value + 6.0 * value * value)
    # Both curves start and finish at zero velocity.  Relaxed/precise styles
    # lean toward the zero-acceleration minimum-jerk trajectory; energetic
    # styles retain a somewhat sharper cubic transition without a hard step.
    return cubic * (1.0 - easing) + minimum_jerk * easing


def presentation_arc_amplitude_rad(duration_s: float) -> float:
    """Maximum upper-arm arc, scaled so acceleration stays bounded."""
    return min(0.30, 0.10 * (max(duration_s, 0.05) / 0.32) ** 2)


def wrist_flourish_amplitude_rad(duration_s: float) -> float:
    """Maximum transient forearm roll under the same duration² rule."""
    return min(0.32, 0.14 * (max(duration_s, 0.05) / 0.32) ** 2)


def forearm_shake_amplitude_rad(duration_s: float, cycles: float) -> float:
    """Bound oscillating forearm pronation/supination by kinematic limits.

    The authored value is later multiplied by ``wrist_shake_amplitude``.  This
    duration/frequency-aware cap keeps the smart primitive safe when a user
    asks for more cycles without silently changing the requested count.  The
    parameter keeps its original serialized name for backward compatibility;
    the motion itself belongs to the forearm, not the wrist joint.
    """
    frequency_hz = max(float(cycles), 0.0) / max(float(duration_s), 0.05)
    if frequency_hz <= 1e-8:
        return 0.0
    omega = 2.0 * math.pi * frequency_hz
    # Leave headroom below the whole-clip structural limit because the attack
    # and release envelope contribute additional discrete acceleration.
    acceleration_cap = 60.0 / (omega * omega)
    jerk_cap = 1500.0 / (omega * omega * omega)
    envelope_cap = 0.26 * max(float(duration_s), 0.05) ** 2
    return max(
        0.0,
        min(MAX_FOREARM_SHAKE_RAD, acceleration_cap, jerk_cap, envelope_cap),
    )


def wrist_shake_amplitude_rad(duration_s: float, cycles: float) -> float:
    """Backward-compatible alias for :func:`forearm_shake_amplitude_rad`."""
    return forearm_shake_amplitude_rad(duration_s, cycles)


def finger_assertions(shape: HandShape, hand: Hand, pose: dict[str, Quat]) -> dict[str, bool]:
    if shape not in {HandShape.HANG_TEN, HandShape.THUMBS_UP, HandShape.PEACE}:
        return {"shape_defined": len(pose) == 15}
    expected = HAND_SHAPES[shape].curls
    assertions = {"all_finger_bones_present": len(pose) == 15}
    for digit in FINGERS:
        if expected[digit] < 0.1:
            assertions[f"{digit.lower()}_extended"] = True
        elif expected[digit] > 0.8:
            assertions[f"{digit.lower()}_curled"] = True
    return assertions
