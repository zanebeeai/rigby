"""Which metric keys the analysis layer owns, and which it does not yet.

This is the ledger the equivalence harness compares against. ``owned`` is the
superset a program's intent may produce; ``required`` is the subset that must
always be present. Together they catch both drift directions: a key that
appears from nowhere, and a key that silently stops being computed.

As 02b–02e land, entries move out of ``DEFERRED_TO_COMPILER`` and into the
per-intent sets, and the harness gets stricter for free.
"""

from __future__ import annotations

from ..models import Intent, MotionProgram, ObjectAction


SAFETY_KEYS = frozenset(
    {
        "joint_limit_violations",
        "root_drift_m",
        "foot_drift_m",
        "fixed_root_foot_max_rotation_delta_rad",
        "nan_count",
        "discontinuities",
        "max_frame_rotation_delta_rad",
        "quaternion_norm_max_error",
        "safety_derivation",
    }
)

ANGULAR_KEYS = frozenset(
    {
        "max_angular_velocity_rad_s",
        "max_angular_acceleration_rad_s2",
        "max_angular_jerk_rad_s3",
    }
)

GESTURE_STRUCTURE_KEYS = frozenset(
    {
        "structural_valid",
        "structural_failures",
        "wrist_swing_twist_limit_violations",
        "max_wrist_swing_rad",
        "max_wrist_twist_rad",
        "max_forearm_twist_rad",
        "self_collision_frames",
        "active_hand_visibility_fraction",
        "active_hand_visibility_samples",
        "max_angular_velocity_rad_s",
        "max_angular_acceleration_rad_s2",
        "max_angular_jerk_rad_s3",
        "quality_reference_schema",
        "quality_reference_source",
        "quality_limits",
        "full_fov_camera_contract",
        "unresolved_non_hand_collisions",
    }
)

SHAKE_KEYS = frozenset(
    {
        "forearm_rotation_cycles",
        "forearm_rotation_amplitude_rad",
        "wrist_flexion_cycles",
        "wrist_flexion_amplitude_rad",
        "wrist_deviation_cycles",
        "wrist_deviation_amplitude_rad",
    }
)

CONTACT_KEYS = frozenset(
    {
        "intra_hand_contact_expected_order",
        "intra_hand_contact_observed_order",
        "intra_hand_contact_count",
        "intra_hand_contact_records",
        "intra_hand_minimum_release_separation_m",
        "gaze_target_records",
        "gaze_max_endpoint_angle_deg",
    }
)

SEMANTIC_KEYS = frozenset(
    {
        "semantic_cycle_action",
        "semantic_cycle_requested_cycles",
        "semantic_cycle_requested_amplitude_m",
        "semantic_cycle_min_reversal_count",
        "semantic_cycle_min_excursion_m",
        "semantic_cycle_hands",
    }
)

PARALLEL_FOREARM_KEYS = frozenset(
    {
        "parallel_forearm_max_axis_error_deg",
        "parallel_forearm_p95_axis_error_deg",
        "parallel_forearm_max_frontal_axis_error_deg",
        "parallel_forearm_p95_frontal_axis_error_deg",
        "parallel_forearm_minimum_separation_m",
        "parallel_forearm_minimum_hand_separation_m",
        "travel_wheel_maximum_opposite_elbow_distance_m",
        "travel_wheel_cross_body_fraction",
        "travel_wheel_minimum_vertical_order_m",
        "travel_wheel_maximum_vertical_order_m",
        "travel_wheel_vertical_order_range_m",
        "travel_wheel_minimum_depth_order_m",
        "travel_wheel_maximum_depth_order_m",
        "travel_wheel_depth_order_range_m",
    }
)


def _handoff(program: MotionProgram) -> bool:
    return (
        program.intent == Intent.OBJECT_INTERACTION
        and program.object_action == ObjectAction.HANDOFF
    )


def owned_metric_keys(program: MotionProgram) -> frozenset[str]:
    """Every key ``analyze`` may return for this program."""

    if program.intent == Intent.UNSUPPORTED:
        return frozenset()
    owned = set(SAFETY_KEYS)
    if program.intent in {Intent.GESTURE, Intent.STRIKE}:
        owned |= GESTURE_STRUCTURE_KEYS
    if program.intent == Intent.GESTURE:
        owned |= SHAKE_KEYS
    if program.intent == Intent.COMPOSITE:
        owned |= CONTACT_KEYS | SEMANTIC_KEYS | PARALLEL_FOREARM_KEYS
    if program.intent == Intent.FULL_BODY:
        owned |= SEMANTIC_KEYS | ANGULAR_KEYS
    if program.intent == Intent.SEQUENCE:
        owned |= ANGULAR_KEYS
    if program.intent == Intent.OBJECT_INTERACTION and not _handoff(program):
        owned |= ANGULAR_KEYS
    return frozenset(owned)


def required_metric_keys(program: MotionProgram) -> frozenset[str]:
    """Keys ``analyze`` must return for this program, whatever the clip shows.

    Conditional families — contact, semantic cycle, parallel forearm — are
    excluded because they legitimately emit nothing when their assertion or
    primitive is absent.
    """

    if program.intent == Intent.UNSUPPORTED:
        return frozenset()
    required = set(SAFETY_KEYS)
    if program.intent in {Intent.GESTURE, Intent.STRIKE}:
        required |= GESTURE_STRUCTURE_KEYS
    if program.intent == Intent.GESTURE:
        required |= SHAKE_KEYS
    if program.intent in {Intent.FULL_BODY, Intent.SEQUENCE}:
        required |= ANGULAR_KEYS
    if program.intent == Intent.OBJECT_INTERACTION and not _handoff(program):
        required |= ANGULAR_KEYS
    return frozenset(required)


# Metric families still computed inside compiler.py, with the PR that moves
# them. Kept here so the gap between "what the compiler emits" and "what the
# analysis layer owns" is a readable number rather than folklore.
DEFERRED_TO_COMPILER: dict[str, str] = {
    "full-body per-action blocks (13 actions)": "02b",
    "IK support targets and commanded root yaw": "02b",
    "composite per-hand gesture-structure fold": "02c",
    "gesture/strike arm-landmark strike metrics": "02c",
    "shake parameter echo (wrist_shake_* aliases)": "02c",
    "object interaction and handoff lifecycle": "02d",
    "MuJoCo grasp metrics": "02d",
    "sequence step re-splitting": "02e",
}
