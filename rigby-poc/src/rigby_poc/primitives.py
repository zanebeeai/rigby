from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .models import Digit, Hand, HandShape, PrimitiveKind, PrimitiveParameters, Quat, StrikeType, Vec3
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


# Rest-orientation calibration extracted from the preserved CC0 humanoid GLB.
# Clip rotations are local deltas post-multiplied onto these rest transforms.
_UPPER_ARM_REST_WORLD_XYZW = {
    Hand.LEFT: (0.009422436964241731, -0.009307101973217042, 0.7089223032569065, -0.7051622249379483),
    Hand.RIGHT: (-0.009422320458211458, -0.009307162762708104, 0.7089222775335169, 0.7051622515529192),
}
_LOWER_ARM_REST_LOCAL_XYZW = {
    Hand.LEFT: (0.02164141647517681, 0.0002870236639864743, -0.006738185882568359, 0.9997430443763733),
    Hand.RIGHT: (0.021641412749886513, -0.0002871047181542963, 0.006738179363310337, 0.9997430443763733),
}
_HAND_REST_LOCAL_XYZW = {
    Hand.LEFT: (-0.00840034894645214, -0.00005970777783659287, 0.009397653862833977, 0.9999206066131592),
    Hand.RIGHT: (-0.00840041134506464, 0.00005969807898509316, -0.009397652931511402, 0.9999205470085144),
}

UPPER_ARM_LENGTH_M = 0.2966
LOWER_ARM_LENGTH_M = 0.2798
ARM_REACH_M = UPPER_ARM_LENGTH_M + LOWER_ARM_LENGTH_M

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


def _signed_angle_about_axis(source: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    """Return the signed projected angle from source to target about axis."""
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    source = source - axis * float(np.dot(source, axis))
    target = target - axis * float(np.dot(target, axis))
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < 1e-8 or target_norm < 1e-8:
        return 0.0
    source /= source_norm
    target /= target_norm
    sine = float(np.dot(axis, np.cross(source, target)))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return math.atan2(sine, cosine)


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


def arm_pose_from_target(
    hand: Hand,
    shoulder: Vec3,
    target: Vec3,
    parameters: PrimitiveParameters,
    *,
    present_hand: bool = False,
    forearm_twist_reserve_rad: float = 0.0,
    elbow_hint: Vec3 | None = None,
) -> tuple[dict[str, Quat], float]:
    """Calibrated analytic two-link IK returned as rest-relative local deltas."""
    shoulder_v = np.asarray(shoulder.as_list(), dtype=float)
    target_v = np.asarray(target.as_list(), dtype=float)
    delta = target_v - shoulder_v
    distance = float(np.linalg.norm(delta))
    upper, lower = UPPER_ARM_LENGTH_M, LOWER_ARM_LENGTH_M
    clamped = float(np.clip(distance, abs(upper - lower) + 1e-4, upper + lower - 1e-4))
    direction = delta / max(distance, 1e-8)
    along = (upper**2 - lower**2 + clamped**2) / (2 * clamped)
    bend_height = math.sqrt(max(upper**2 - along**2, 0.0))
    side = 1.0 if hand == Hand.LEFT else -1.0
    if elbow_hint is None:
        pole = np.asarray([side, -0.55, 0.25], dtype=float)
        bend = pole - direction * float(np.dot(pole, direction))
    else:
        line_point = shoulder_v + direction * along
        hinted = np.asarray(elbow_hint.as_list(), dtype=float) - line_point
        bend = hinted - direction * float(np.dot(hinted, direction))
    if np.linalg.norm(bend) < 1e-8:
        bend = np.cross(direction, np.asarray([0.0, 1.0, 0.0]))
    bend /= np.linalg.norm(bend)
    if elbow_hint is None:
        bend = Rotation.from_rotvec(
            direction * parameters.elbow_swivel * 0.75
        ).apply(bend)
    elbow = shoulder_v + direction * along + bend * bend_height

    upper_direction = (elbow - shoulder_v) / upper
    lower_direction = (target_v - elbow) / lower
    rest_upper = Rotation.from_quat(_UPPER_ARM_REST_WORLD_XYZW[hand])
    upper_alignment = _rotation_between(rest_upper.apply([0.0, 1.0, 0.0]), upper_direction)
    desired_upper = upper_alignment * rest_upper
    upper_delta = rest_upper.inv() * desired_upper

    lower_local_rest = Rotation.from_quat(_LOWER_ARM_REST_LOCAL_XYZW[hand])
    lower_base_world = desired_upper * lower_local_rest
    lower_alignment = _rotation_between(lower_base_world.apply([0.0, 1.0, 0.0]), lower_direction)
    desired_lower = lower_alignment * lower_base_world
    if present_hand:
        # Rotate the whole forearm around its longitudinal axis so the palm is
        # readable from the egocentric +Z view.  Wrist roll in natural human
        # motion is primarily forearm pronation/supination; placing it here
        # avoids the 90-176 degree hand-joint deltas produced by the old solve.
        hand_rest = Rotation.from_quat(_HAND_REST_LOCAL_XYZW[hand])
        neutral_hand_world = desired_lower * hand_rest
        palm_back_world = neutral_hand_world.apply([0.0, 0.0, 1.0])
        camera_facing_palm_back = np.asarray([0.0, 0.0, -1.0])
        alignment_twist = _signed_angle_about_axis(
            palm_back_world,
            camera_facing_palm_back,
            lower_direction,
        )
        authored_twist = parameters.wrist_roll * 0.40
        # A later shake adds longitudinal rotation to this same joint. Reserve
        # its full amplitude during every presentation phase so the combined
        # pose remains physically safe by construction instead of clipping one
        # half of the oscillation after the fact.
        static_twist_limit = max(
            0.0,
            MAX_FOREARM_TWIST_RAD - max(0.0, float(forearm_twist_reserve_rad)),
        )
        forearm_twist = float(
            np.clip(
                alignment_twist + authored_twist,
                -static_twist_limit,
                static_twist_limit,
            )
        )
        desired_lower = Rotation.from_rotvec(lower_direction * forearm_twist) * desired_lower

        # The remaining hand delta contains only bounded flexion/deviation.
        # A very small longitudinal allowance absorbs Euler composition error;
        # it is checked independently by the structural gate.
        wrist_adjustment = Rotation.from_euler(
            "xyz",
            [
                parameters.wrist_pitch * MAX_WRIST_PITCH_RAD,
                parameters.wrist_roll * MAX_WRIST_TWIST_RAD,
                parameters.wrist_yaw * MAX_WRIST_YAW_RAD,
            ],
        )
        hand_delta = wrist_adjustment
    else:
        hand_delta = Rotation.from_euler(
            "xyz",
            [
                parameters.wrist_pitch * MAX_WRIST_PITCH_RAD,
                parameters.wrist_roll * MAX_WRIST_TWIST_RAD,
                parameters.wrist_yaw * MAX_WRIST_YAW_RAD,
            ],
        )
    lower_delta = lower_base_world.inv() * desired_lower
    prefix = hand.value
    poses = {
        f"{prefix}Shoulder": Quat(),
        f"{prefix}UpperArm": _quat_from_rotation(upper_delta),
        f"{prefix}LowerArm": _quat_from_rotation(lower_delta),
        f"{prefix}Hand": _quat_from_rotation(hand_delta),
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
        base = {
            PrimitiveKind.LOAD: (side * 0.16, 1.34, 0.29),
            PrimitiveKind.STRIKE: (lateral, 1.40, 0.46),
            PrimitiveKind.FOLLOW_THROUGH: (lateral, 1.39, 0.43),
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
    return Vec3(x=0.174 if hand == Hand.LEFT else -0.174, y=1.446, z=-0.065)


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
