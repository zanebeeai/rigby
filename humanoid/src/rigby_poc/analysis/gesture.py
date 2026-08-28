"""Gesture-layer structural quality metrics.

Moved verbatim from ``rigby_poc.quality``, which now re-exports this module so
existing call sites keep working. Nothing here touches generation state: every
function reads a finished clip.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from ..models import ClipFrame, Hand, HandShape
from ..primitives import (
    LOWER_ARM_LENGTH_M,
    UPPER_ARM_LENGTH_M,
    _HAND_REST_LOCAL_XYZW,
    _LOWER_ARM_REST_LOCAL_XYZW,
    _UPPER_ARM_REST_WORLD_XYZW,
    shoulder_position,
)
from .context import AnalysisContext
from .contract import ANATOMY, SIGNAL, CheckResult, count_check, lower_bound_check, upper_bound_check
from .rig import EGO_NEUTRAL_GAZE, PROJECT_ROOT


QUALITY_REFERENCE = PROJECT_ROOT / "config" / "motion_quality_reference.json"

#: The ego camera's world position at the calibrated rest head pose.
#:
#: ``frontend/src/camera.ts::computeEgoCameraPose`` builds this as the head's
#: rest world position plus a head-local eye offset of ``(0, 0.04, 0.11)``. The
#: literal here is that sum with the head position *rounded*, and it stays the
#: anchor rather than being recomputed from the frontend's clean offset: the two
#: disagree by 2.4e-5 m in z, and the rest pose sits 1.5e-5 m inside the vertical
#: frustum, so adopting the frontend constant flips frame 0 of every gesture and
#: strike clip out of view. Reconciling the two literals is separate, measured
#: work; anchoring on the frozen one keeps this change to what it claims to be.
_REST_EGO_CAMERA_POSITION = np.asarray([0.0, 1.5685 + 0.04, 0.0114 + 0.11], dtype=float)


def quality_reference() -> dict[str, Any]:
    return json.loads(QUALITY_REFERENCE.read_text(encoding="utf-8"))


def _angle(rotation: Rotation) -> float:
    return float(rotation.magnitude())


def swing_twist_angles(quaternion_xyzw: Iterable[float], axis: Iterable[float]) -> tuple[float, float]:
    """Decompose a local rotation into swing and signed twist magnitudes."""
    value = np.asarray(list(quaternion_xyzw), dtype=float)
    value /= max(float(np.linalg.norm(value)), 1e-12)
    axis_value = np.asarray(list(axis), dtype=float)
    axis_value /= max(float(np.linalg.norm(axis_value)), 1e-12)
    projected = axis_value * float(np.dot(value[:3], axis_value))
    twist_value = np.asarray([projected[0], projected[1], projected[2], value[3]], dtype=float)
    twist_norm = float(np.linalg.norm(twist_value))
    if twist_norm < 1e-12:
        twist = Rotation.identity()
    else:
        twist = Rotation.from_quat(twist_value / twist_norm)
    full = Rotation.from_quat(value)
    swing = full * twist.inv()
    signed_twist = _angle(twist)
    if float(np.dot(twist.as_rotvec(), axis_value)) < 0.0:
        signed_twist *= -1.0
    return _angle(swing), signed_twist


def arm_landmarks(frame: ClipFrame, hand: Hand) -> tuple[np.ndarray, np.ndarray, np.ndarray, Rotation]:
    prefix = hand.value
    shoulder = np.asarray(shoulder_position(hand).as_list(), dtype=float)
    upper_delta = Rotation.from_quat(frame.bones[f"{prefix}UpperArm"].rotation.as_list())
    lower_delta = Rotation.from_quat(frame.bones[f"{prefix}LowerArm"].rotation.as_list())
    upper_world = Rotation.from_quat(_UPPER_ARM_REST_WORLD_XYZW[hand]) * upper_delta
    lower_base = upper_world * Rotation.from_quat(_LOWER_ARM_REST_LOCAL_XYZW[hand])
    lower_world = lower_base * lower_delta
    elbow = shoulder + upper_world.apply([0.0, UPPER_ARM_LENGTH_M, 0.0])
    wrist = elbow + lower_world.apply([0.0, LOWER_ARM_LENGTH_M, 0.0])
    hand_world = (
        lower_world
        * Rotation.from_quat(_HAND_REST_LOCAL_XYZW[hand])
        * Rotation.from_quat(frame.bones[f"{prefix}Hand"].rotation.as_list())
    )
    return shoulder, elbow, wrist, hand_world


def world_arm_landmarks(
    ctx: AnalysisContext, index: int, hand: Hand
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Rotation]:
    """Shoulder, elbow, wrist and hand orientation in world, for one frame.

    ``arm_landmarks`` rebuilds the arm from a fixed rest shoulder and five rest
    constants, so it cannot see the trunk: what it returns is the arm measured
    relative to the chest. Since strikes distribute their yaw over the spine,
    chest and upperChest, that is the wrong frame for anything projected onto a
    world axis -- a world lateral excursion read there is the arm's excursion
    across a torso that is itself turning.

    The rig already carries the answer: the ``*UpperArm``, ``*LowerArm`` and
    ``*Hand`` pivots are the shoulder, elbow and wrist, so this reads them out
    of the frame's world matrices instead of reconstructing them. Taking the
    context and a frame index rather than a :class:`ClipFrame` is deliberate --
    it makes the function unable to evaluate forward kinematics of its own, so
    it costs nothing beyond the one pass per frame the context already memoises.

    ``arm_landmarks`` stays as-is and stays the answer for generation (the
    compiler places carried objects with it) and for hand-local attachment
    offsets, neither of which wants a world frame.
    """

    prefix = hand.value
    matrices = ctx.world_matrices(index)
    node = ctx.kinematics.node_by_canonical
    hand_matrix = matrices[node[f"{prefix}Hand"]]
    return (
        matrices[node[f"{prefix}UpperArm"]][:3, 3].copy(),
        matrices[node[f"{prefix}LowerArm"]][:3, 3].copy(),
        hand_matrix[:3, 3].copy(),
        Rotation.from_matrix(hand_matrix[:3, :3]),
    )


def _inside_torso(point: np.ndarray) -> bool:
    if not 1.01 <= float(point[1]) <= 1.48:
        return False
    # Conservative torso capsule in the calibrated rig's X/Z cross-section.
    # The arms are permitted to pass in front of the chest, but not through it.
    normalized = (float(point[0]) / 0.19) ** 2 + ((float(point[2]) + 0.015) / 0.145) ** 2
    return normalized < 1.0


def _arm_self_collision(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> bool:
    samples: list[np.ndarray] = []
    # Ignore the proximal upper-arm section that begins on the torso surface.
    # The shoulder joint starts inside the torso mesh. Samples before the
    # midpoint falsely classify a naturally adducted upper arm (as used by
    # the travel signal) as penetrating the chest even when its elbow and
    # complete forearm remain in front of the torso surface.
    for alpha in np.linspace(0.55, 1.0, 5):
        samples.append(shoulder * (1.0 - alpha) + elbow * alpha)
    for alpha in np.linspace(0.0, 1.0, 7):
        samples.append(elbow * (1.0 - alpha) + wrist * alpha)
    return any(_inside_torso(point) for point in samples)


def _full_hand_visible(
    wrist: np.ndarray,
    hand_world: Rotation,
    contract: dict[str, Any],
    hand_shape: HandShape | None = None,
) -> bool:
    # The rest-pose ego camera. ``_REST_EGO_CAMERA_POSITION`` and
    # ``EGO_NEUTRAL_GAZE`` are the same literals this function used to build
    # inline, named rather than retyped, so this path is unchanged bit for bit.
    # Carrying the camera on the head is a separate change -- see G6b.
    position = _REST_EGO_CAMERA_POSITION
    forward = EGO_NEUTRAL_GAZE
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    relative = wrist - position
    depth = float(np.dot(relative, forward))
    if depth <= 0.015:
        return False
    horizontal = float(np.dot(relative, right))
    vertical = float(np.dot(relative, up))
    half_vertical = math.tan(math.radians(float(contract["vertical_fov_deg"])) / 2.0) * depth
    half_horizontal = half_vertical * float(contract["aspect_ratio"])

    # A shaka is wider along the local finger axis than a closed hand.  Project
    # that span into camera X/Y so the complete silhouette, not only the wrist,
    # must remain inside the frustum.
    local_finger = hand_world.apply([0.0, 1.0, 0.0])
    local_palm = hand_world.apply([1.0, 0.0, 0.0])
    finger_half_span = 0.060 if hand_shape == HandShape.FIST else 0.095
    palm_half_span = 0.050 if hand_shape == HandShape.FIST else 0.055
    radius_x = abs(float(np.dot(local_finger, right))) * finger_half_span + abs(
        float(np.dot(local_palm, right))
    ) * palm_half_span
    radius_y = abs(float(np.dot(local_finger, up))) * finger_half_span + abs(
        float(np.dot(local_palm, up))
    ) * palm_half_span
    radius_x = max(radius_x, float(contract["hand_visibility_radius_m"]) * 0.55)
    radius_y = max(radius_y, float(contract["hand_visibility_radius_m"]) * 0.55)
    return abs(horizontal) + radius_x <= half_horizontal and abs(vertical) + radius_y <= half_vertical


def _angular_kinematics(frames: list[ClipFrame], bone_names: list[str]) -> tuple[float, float, float]:
    if len(frames) < 4:
        return 0.0, 0.0, 0.0
    times = np.asarray([frame.time_s for frame in frames], dtype=float)
    velocities: list[float] = []
    velocity_times: list[float] = []
    for index, (first, second) in enumerate(zip(frames, frames[1:])):
        delta_t = max(float(times[index + 1] - times[index]), 1e-8)
        maximum = 0.0
        for name in bone_names:
            a = Rotation.from_quat(first.bones[name].rotation.as_list())
            b = Rotation.from_quat(second.bones[name].rotation.as_list())
            maximum = max(maximum, _angle(a.inv() * b) / delta_t)
        velocities.append(maximum)
        velocity_times.append((times[index + 1] + times[index]) / 2.0)
    accelerations: list[float] = []
    acceleration_times: list[float] = []
    for index, (first, second) in enumerate(zip(velocities, velocities[1:])):
        delta_t = max(velocity_times[index + 1] - velocity_times[index], 1e-8)
        accelerations.append(abs(second - first) / delta_t)
        acceleration_times.append((velocity_times[index + 1] + velocity_times[index]) / 2.0)
    jerks: list[float] = []
    for index, (first, second) in enumerate(zip(accelerations, accelerations[1:])):
        delta_t = max(acceleration_times[index + 1] - acceleration_times[index], 1e-8)
        jerks.append(abs(second - first) / delta_t)
    return (
        max(velocities, default=0.0),
        max(accelerations, default=0.0),
        max(jerks, default=0.0),
    )


def _oscillation_summary(values: list[float]) -> tuple[float, float]:
    """Return measured cycles and peak amplitude for a signed joint channel."""
    amplitude = max((abs(value) for value in values), default=0.0)
    if amplitude < 1e-4:
        return 0.0, amplitude
    threshold = max(1e-4, amplitude * 0.15)
    signs: list[int] = []
    for value in values:
        if abs(value) < threshold:
            continue
        sign = 1 if value > 0.0 else -1
        if not signs or signs[-1] != sign:
            signs.append(sign)
    return len(signs) / 2.0, amplitude


def shake_joint_oscillation_metrics(
    frames: list[ClipFrame],
    hand: Hand,
    shake_ranges: list[tuple[float, float]],
) -> dict[str, float]:
    """Measure which physical joint actually carries the authored shake.

    Values come from frame rotations, not planner parameters. This matters for
    adversarial clips that move an oscillation from the lower arm to the hand.
    """
    selected = [
        frame
        for frame in frames
        if any(start - 1e-8 <= frame.time_s <= end + 1e-8 for start, end in shake_ranges)
    ]
    if len(selected) < 3:
        return {
            "forearm_rotation_cycles": 0.0,
            "forearm_rotation_amplitude_rad": 0.0,
            "wrist_flexion_cycles": 0.0,
            "wrist_flexion_amplitude_rad": 0.0,
            "wrist_deviation_cycles": 0.0,
            "wrist_deviation_amplitude_rad": 0.0,
        }

    prefix = hand.value
    lower_name = f"{prefix}LowerArm"
    hand_name = f"{prefix}Hand"
    lower_baseline = Rotation.from_quat(selected[0].bones[lower_name].rotation.as_list())
    hand_baseline = Rotation.from_quat(selected[0].bones[hand_name].rotation.as_list())
    forearm: list[float] = []
    wrist_flexion: list[float] = []
    wrist_deviation: list[float] = []
    for frame in selected:
        lower = Rotation.from_quat(frame.bones[lower_name].rotation.as_list())
        lower_relative = lower_baseline.inv() * lower
        _, lower_twist = swing_twist_angles(lower_relative.as_quat(), [0.0, 1.0, 0.0])
        forearm.append(lower_twist)

        current_hand = Rotation.from_quat(frame.bones[hand_name].rotation.as_list())
        hand_rotvec = (hand_baseline.inv() * current_hand).as_rotvec()
        wrist_flexion.append(float(hand_rotvec[0]))
        wrist_deviation.append(float(hand_rotvec[2]))

    forearm_cycles, forearm_amplitude = _oscillation_summary(forearm)
    flexion_cycles, flexion_amplitude = _oscillation_summary(wrist_flexion)
    deviation_cycles, deviation_amplitude = _oscillation_summary(wrist_deviation)
    return {
        "forearm_rotation_cycles": forearm_cycles,
        "forearm_rotation_amplitude_rad": forearm_amplitude,
        "wrist_flexion_cycles": flexion_cycles,
        "wrist_flexion_amplitude_rad": flexion_amplitude,
        "wrist_deviation_cycles": deviation_cycles,
        "wrist_deviation_amplitude_rad": deviation_amplitude,
    }


def evaluate_gesture_structure(
    frames: list[ClipFrame],
    hand: Hand,
    presentation_ranges: list[tuple[float, float]],
    hand_shape: HandShape | None = None,
) -> dict[str, Any]:
    """Structural quality for one hand over a finished clip.

    Visibility is still judged against the reconstructed, chest-relative arm and
    the rest-pose camera. Moving it onto the world-true wrist and a head-carried
    camera is G6b, held separately: the two halves are only correct together,
    and the camera's eye offset is 2.4e-5 m from the renderer's own constant,
    which is the same size as the frustum margin that decides the verdict.
    """

    reference = quality_reference()
    limits = reference["hard_limits"]
    contract = reference["camera_contract"]
    prefix = hand.value
    swing_values: list[float] = []
    twist_values: list[float] = []
    forearm_twist_values: list[float] = []
    collisions = 0
    visible: list[bool] = []
    for frame in frames:
        swing, twist = swing_twist_angles(frame.bones[f"{prefix}Hand"].rotation.as_list(), [0.0, 1.0, 0.0])
        swing_values.append(swing)
        twist_values.append(abs(twist))
        _, forearm_twist = swing_twist_angles(
            frame.bones[f"{prefix}LowerArm"].rotation.as_list(), [0.0, 1.0, 0.0]
        )
        forearm_twist_values.append(abs(forearm_twist))
        # Self-collision stays on the reconstructed arm deliberately.
        # ``_inside_torso`` pins a *fixed* world capsule over y in [1.01, 1.48]
        # while the trunk yaw is graded across spine/chest/upperChest, so world
        # points would be tested against a chest-frame torso -- half a fix, and
        # the brief forbids moving self-collision silently. Measured either way
        # the check stays latent: closest approach is 1.08-1.55 normalised radii
        # against a gate at 1.0, and 0 of 407 strike frames fire. Carrying the
        # capsule on the chest is the real fix and is its own change.
        shoulder, elbow, wrist, hand_world = arm_landmarks(frame, hand)
        collisions += int(_arm_self_collision(shoulder, elbow, wrist))
        if any(start - 1e-8 <= frame.time_s <= end + 1e-8 for start, end in presentation_ranges):
            visible.append(_full_hand_visible(wrist, hand_world, contract, hand_shape))

    velocity, acceleration, jerk = _angular_kinematics(
        frames, [f"{prefix}UpperArm", f"{prefix}LowerArm", f"{prefix}Hand"]
    )
    max_swing = max(swing_values, default=0.0)
    max_twist = max(twist_values, default=0.0)
    max_forearm_twist = max(forearm_twist_values, default=0.0)
    visibility_fraction = sum(visible) / len(visible) if visible else 0.0
    wrist_violations = sum(
        swing > float(limits["wrist_swing_rad"]) + 1e-8
        or twist > float(limits["wrist_twist_rad"]) + 1e-8
        for swing, twist in zip(swing_values, twist_values)
    )
    failures: list[str] = []
    if wrist_violations:
        failures.append(f"{wrist_violations} wrist swing/twist limit violations")
    if max_forearm_twist > float(limits["forearm_twist_rad"]) + 1e-8:
        failures.append(f"forearm twist {max_forearm_twist:.3f} rad exceeds calibrated limit")
    if collisions:
        failures.append(f"self collision detected in {collisions} frames")
    if visibility_fraction < float(limits["minimum_active_hand_visibility_fraction"]) - 1e-8:
        failures.append(f"active hand visibility is {visibility_fraction:.3f}")
    for label, value, key in (
        ("angular velocity", velocity, "angular_velocity_rad_s"),
        ("angular acceleration", acceleration, "angular_acceleration_rad_s2"),
        ("angular jerk", jerk, "angular_jerk_rad_s3"),
    ):
        if value > float(limits[key]) + 1e-8:
            failures.append(f"{label} {value:.3f} exceeds calibrated limit {limits[key]:.3f}")
    return {
        "structural_valid": not failures,
        "structural_failures": failures,
        "wrist_swing_twist_limit_violations": wrist_violations,
        "max_wrist_swing_rad": max_swing,
        "max_wrist_twist_rad": max_twist,
        "max_forearm_twist_rad": max_forearm_twist,
        "self_collision_frames": collisions,
        "active_hand_visibility_fraction": visibility_fraction,
        "active_hand_visibility_samples": len(visible),
        "max_angular_velocity_rad_s": velocity,
        "max_angular_acceleration_rad_s2": acceleration,
        "max_angular_jerk_rad_s3": jerk,
        "quality_reference_schema": reference["schema_version"],
        "quality_reference_source": reference["source"],
        "quality_limits": limits,
        "full_fov_camera_contract": contract,
    }


def gesture_structure_checks(structure: dict[str, Any]) -> list[CheckResult]:
    """Report one ``evaluate_gesture_structure`` result as addressable checks.

    Same thresholds, same tolerances and same order as the ``structural_failures``
    list the function itself builds; this is a typed view of it, not a second
    opinion.
    """

    limits = structure["quality_limits"]
    return [
        count_check(
            "anatomy.wrist.swing_twist_limit",
            ANATOMY,
            int(structure["wrist_swing_twist_limit_violations"]),
            detail="wrist swing/twist limit violations",
        ),
        upper_bound_check(
            "anatomy.forearm.twist",
            ANATOMY,
            float(structure["max_forearm_twist_rad"]),
            float(limits["forearm_twist_rad"]),
            tolerance=1e-8,
            detail="forearm twist exceeds the calibrated limit",
        ),
        count_check(
            "anatomy.arm.self_collision",
            ANATOMY,
            int(structure["self_collision_frames"]),
            detail="self collision frames",
        ),
        lower_bound_check(
            "contract.camera.active_hand_visibility",
            SIGNAL,
            float(structure["active_hand_visibility_fraction"]),
            float(limits["minimum_active_hand_visibility_fraction"]),
            tolerance=1e-8,
            scale=1.0,
            detail="active hand leaves the egocentric frustum",
        ),
        upper_bound_check(
            "signal.angular.velocity",
            SIGNAL,
            float(structure["max_angular_velocity_rad_s"]),
            float(limits["angular_velocity_rad_s"]),
            tolerance=1e-8,
            detail="angular velocity exceeds the calibrated limit",
        ),
        upper_bound_check(
            "signal.angular.acceleration",
            SIGNAL,
            float(structure["max_angular_acceleration_rad_s2"]),
            float(limits["angular_acceleration_rad_s2"]),
            tolerance=1e-8,
            detail="angular acceleration exceeds the calibrated limit",
        ),
        upper_bound_check(
            "signal.angular.jerk",
            SIGNAL,
            float(structure["max_angular_jerk_rad_s3"]),
            float(limits["angular_jerk_rad_s3"]),
            tolerance=1e-8,
            detail="angular jerk exceeds the calibrated limit",
        ),
    ]
