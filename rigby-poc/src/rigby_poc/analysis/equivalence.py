"""Which metric keys the analysis layer owns, and which it does not yet.

This is the ledger the equivalence harness compares against. ``owned`` is the
superset a program's intent may produce; ``required`` is the subset that must
always be present. Together they catch both drift directions: a key that
appears from nowhere, and a key that silently stops being computed.

As 02b–02e land, entries move out of ``DEFERRED_TO_COMPILER`` and into the
per-intent sets, and the harness gets stricter for free.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

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


# --- whole body (02b) --------------------------------------------------------
#
# Every whole-body clip emits these, whatever it depicts: root travel, the
# ground plane and its clearances, the terminal balance test, the support-target
# comparison, and the structural verdict.
FULL_BODY_SHARED_KEYS = frozenset(
    {
        "airborne_frame_count",
        "final_balanced_leg_error_rad",
        "final_foot_ground_error_m",
        "final_root_vertical_speed_m_s",
        "final_root_yaw_deg",
        "final_support_center_offset_m",
        "ground_height_m",
        "ground_penetration_m",
        "horizontal_pose_requested",
        "max_support_foot_slide_per_frame_m",
        "max_support_foot_target_error_m",
        "maximum_foot_clearance_m",
        "minimum_nonfoot_body_clearance_m",
        "requested_body_cycles",
        "root_displacement_m",
        "root_path_length_m",
        "root_vertical_max_m",
        "root_vertical_min_m",
        "structural_failures",
        "structural_valid",
        "support_contact_fraction",
    }
)

# The rest are gated on a primitive selector, so they are owned but never
# required. One frozenset per block in ``analysis.full_body``, which is also how
# ``test_full_body_analysis.py`` asserts the fixture reaches every one of them.
JUMPING_JACK_KEYS = frozenset(
    {
        "requested_jumping_jack_cycles",
        "measured_jumping_jack_cycles",
        "jumping_jack_foot_spread_excursion_m",
        "jumping_jack_max_wrist_height_m",
    }
)

BURPEE_KEYS = frozenset(
    {
        "requested_burpee_cycles",
        "measured_burpee_jump_cycles",
        "measured_burpee_push_up_cycles",
        "burpee_floor_support_phase_count",
        "burpee_airborne_phase_count",
        "burpee_root_vertical_excursion_m",
        "burpee_max_wrist_height_m",
    }
)

SQUAT_KEYS = frozenset(
    {
        "requested_squat_cycles",
        "measured_squat_cycles",
        "squat_minimum_depth_m",
        "squat_max_stance_return_error_m",
    }
)

LUNGE_KEYS = frozenset(
    {
        "requested_lunge_cycles",
        "measured_lunge_cycles",
        "lunge_minimum_root_drop_m",
        "lunge_minimum_foot_stagger_m",
        "lunge_max_stance_return_error_m",
    }
)

SINGLE_LEG_KEYS = frozenset(
    {
        "single_leg_balance_phase_count",
        "single_leg_min_raised_foot_clearance_m",
        "single_leg_max_support_foot_slide_m",
        "single_leg_support_contact_fraction",
    }
)

SIT_UP_KEYS = frozenset(
    {
        "requested_sit_up_cycles",
        "measured_sit_up_cycles",
        "sit_up_minimum_head_lift_m",
        "sit_up_max_supine_return_error_m",
    }
)

CRAWL_KEYS = frozenset(
    {
        "requested_crawl_cycles",
        "requested_crawl_distance_m",
        "crawl_root_displacement_m",
        "crawl_hand_alternation_range_m",
    }
)

PUSH_UP_KEYS = frozenset(
    {
        "requested_push_up_cycles",
        "measured_push_up_cycles",
        "push_up_vertical_excursion_m",
        "push_up_palm_height_range_m",
        "push_up_toe_height_range_m",
        "push_up_toe_position_range_m",
        "push_up_toe_support_clearance_m",
        "push_up_min_knee_extension_deg",
        "plank_hold_vertical_range_m",
    }
)

CLIMB_KEYS = frozenset(
    {
        "requested_climb_height_m",
        "measured_climb_height_m",
        "climb_vertical_completion_fraction",
        "requested_climb_cycles",
        "climb_support_target_max_error_m",
        "climb_three_point_support_fraction",
        "climb_final_supported_limb_count",
        "climb_missing_support_object_count",
        "climb_phase_metrics",
    }
)

DANCE_KEYS = frozenset(
    {
        "requested_dance_beats",
        "measured_dance_beats",
        "dance_alternating_lift_count",
        "dance_lateral_root_range_m",
        "dance_left_foot_peak_clearance_m",
        "dance_right_foot_peak_clearance_m",
        "dance_phase_metrics",
    }
)

ROTATION_KEYS = frozenset(
    {
        "body_rotation_phase_metrics",
        "requested_body_rotation_degrees",
        "measured_body_rotation_degrees",
        "minimum_body_rotation_completion_fraction",
        "minimum_rotation_travel_completion_fraction",
        "floor_roll_nonfoot_contact_frame_count",
        "floor_roll_nonfoot_contact_fraction",
        "cartwheel_hand_contact_frame_count",
        "cartwheel_hand_contact_fraction",
        "cartwheel_inverted_frame_count",
        "cartwheel_max_foot_clearance_m",
        "cartwheel_minimum_head_clearance_m",
        "airborne_rotation_airborne_frame_count",
        "airborne_rotation_airborne_fraction",
        "airborne_rotation_peak_root_height_m",
    }
)

OBSTACLE_KEYS = frozenset(
    {
        "obstacle_traversal_phase_metrics",
        "obstacle_missing_target_count",
        "minimum_obstacle_step_foot_clearance_m",
        "maximum_obstacle_step_crossing_error_m",
        "minimum_obstacle_avoidance_root_clearance_m",
    }
)

HORIZONTAL_POSE_KEYS = frozenset(
    {
        "horizontal_pose_minimum_clearance_m",
        "horizontal_pose_contact_point_count",
        "horizontal_body_axis_vertical_fraction",
        "horizontal_pose_pelvis_pitch_deg",
        "horizontal_pose_variant",
    }
)

FULL_BODY_GATED_KEYS: dict[str, frozenset[str]] = {
    "jumping_jack": JUMPING_JACK_KEYS,
    "burpee": BURPEE_KEYS,
    "squat": SQUAT_KEYS,
    "lunge": LUNGE_KEYS,
    "single_leg": SINGLE_LEG_KEYS,
    "sit_up": SIT_UP_KEYS,
    "crawl": CRAWL_KEYS,
    "push_up": PUSH_UP_KEYS,
    "climb": CLIMB_KEYS,
    "dance": DANCE_KEYS,
    "rotation": ROTATION_KEYS,
    "obstacle": OBSTACLE_KEYS,
    "horizontal_pose": HORIZONTAL_POSE_KEYS,
}

FULL_BODY_KEYS = FULL_BODY_SHARED_KEYS.union(*FULL_BODY_GATED_KEYS.values())

# Written by the compiler *before* the measurement pass and read by it, not
# produced by it. ``analyze`` never emits them, so they belong in neither set.
COMPOSITE_CARRIED_KEYS = frozenset({"phase_ranges_s", "presentation_ranges_s"})

FULL_BODY_CARRIED_KEYS = frozenset(
    {
        "phase_ranges_s",
        "body_actions",
        "root_motion_enabled",
        "support_constraints",
        "climb_support_constraints",
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
        owned |= SEMANTIC_KEYS | ANGULAR_KEYS | FULL_BODY_KEYS
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
    if program.intent == Intent.FULL_BODY:
        required |= FULL_BODY_SHARED_KEYS
    if program.intent == Intent.OBJECT_INTERACTION and not _handoff(program):
        required |= ANGULAR_KEYS
    return frozenset(required)


# Metric families still computed inside compiler.py, with the PR that moves
# them. Kept here so the gap between "what the compiler emits" and "what the
# analysis layer owns" is a readable number rather than folklore.
DEFERRED_TO_COMPILER: dict[str, str] = {
    "composite per-hand gesture-structure fold": "02c",
    "gesture/strike arm-landmark strike metrics": "02c",
    "shake parameter echo (wrist_shake_* aliases)": "02c",
    "object interaction and handoff lifecycle": "02d",
    "MuJoCo grasp metrics": "02d",
    "sequence step re-splitting": "02e",
}


# --- the carry-over contract for mutated clips (02b.1) -----------------------
#
# A mutation perturbs a compiled ``ClipResult`` and recompiles nothing (plan 06
# §3.5). So after a mutation ``clip.metrics`` holds three different kinds of
# thing, and only two of them are safe to read:
#
# 1. keys the analysis layer recomputes from the mutated frames — correct by
#    construction, because they are computed from what is actually there;
# 2. keys that are **authoring intent** — correct *because* they did not change.
#    ``support_constraints`` records where the IK solver was told to put the
#    ankle; moving the frames does not change what was commanded, and holding it
#    fixed is precisely what makes "the foot missed where it was told to go"
#    measurable;
# 3. keys that are **stale observations** — a measurement of a clip that no
#    longer exists.
#
# The third kind is not merely wrong, it is wrong in the direction that
# manufactures a false negative. ``structural_valid`` is deferred to the
# compiler for object interaction and sequence, so a mutated object clip
# inherits ``structural_valid: True`` from the clean compile. A detection matrix
# scores that as "the targeted check did not fire", which is indistinguishable
# from a genuine hole in check coverage — the harness hides exactly the finding
# it exists to surface.
#
# Hence: reading a stale key off a mutated clip raises. It does not return
# ``None`` and it is not silently absent, because a missing key becomes
# ``metrics.get(key)`` returning ``None`` somewhere downstream and resurfaces as
# an unrelated symptom three layers away.


#: Keys ``compiler._base_metrics`` seeds on every clip and that several compile
#: paths never write. They are not stale — they were never measured. A whole-body
#: clip publishes ``max_penetration_m: 0.0`` having never looked for penetration,
#: and ``lost_table_contact: True`` with no table in the scene.
#:
#: This is the same defect lane `infra` found in ``foot_drift_m``, which is a
#: member of the same dict: a published constant compared against an acceptance
#: criterion it can never breach. It is one constructor rather than one key, and
#: which members are real depends on the compile path. Recorded here because the
#: mutation ledger has to withhold them either way; measuring or deleting them is
#: plan 08's work.
UNWRITTEN_BASE_DEFAULTS = frozenset(
    {
        "finger_assertions",
        "hold_duration_s",
        "lift_height_m",
        "lost_table_contact",
        "max_penetration_m",
        "opposing_contacts",
        "palm_relative_slip_m",
        "unresolved_non_hand_collisions",
        "vertical_drift_m",
        "weld_used",
    }
)


class StaleMetricError(KeyError):
    """A metric that describes the clip as it was before it was mutated."""


#: Keys that survive a mutation because they were never observations of motion.
#: Deliberately conservative: a key earns a place here only when its value is a
#: function of the program, the scene or the compiler configuration, never of
#: the frames. Everything else is treated as stale, because the failure modes are
#: not symmetric — a key wrongly called stale raises an error a developer fixes
#: in one line, while a key wrongly called intent returns a plausible number
#: nobody ever questions.
#:
#: Incomplete on purpose for object interaction and sequence. Those paths keep
#: ~79 metric keys inside ``compiler.py`` (plan 02 §6.4), and classifying them
#: needs the same reading of the compiler that porting them would need. Until
#: 02d/02e land, their keys stay stale unless listed here.
CARRIED_AUTHORING_INTENT = frozenset(
    {
        # authored timing and structure, read back rather than re-derived
        "phase_ranges_s",
        "body_actions",
        "root_motion_enabled",
        "sequence_step_ranges_s",
        "presentation_ranges_s",
        "sequence_step_count",
        "sequence_intents",
        "object_action",
        # commanded IK targets: what the solver was asked for, not what it hit
        "support_constraints",
        "climb_support_constraints",
        # compiler and physics provenance
        "physics_engine",
        "physics_model",
        "physics_version",
        "safety_derivation",
        "quality_reference_schema",
        "quality_reference_source",
        "quality_limits",
        "full_fov_camera_contract",
    }
)


def stale_after_mutation(
    compiled_metrics: "Mapping[str, Any]", program: MotionProgram
) -> frozenset[str]:
    """Keys of a compiled clip that a mutation invalidates.

    Derived, not listed: whatever the compiler emitted, minus what the analysis
    layer recomputes for this program, minus verified authoring intent. That
    makes the set shrink on its own as 02c–02e port more paths, and it cannot
    drift out of step with the registry the way a hand-maintained list would —
    the drift would be silent and in the direction of re-admitting stale keys.
    """

    return frozenset(
        set(compiled_metrics) - owned_metric_keys(program) - CARRIED_AUTHORING_INTENT
    )


class MutationSafeMetrics(Mapping[str, Any]):
    """``clip.metrics`` for a mutated clip, with the stale keys made loud.

    Recomputed and carried keys read normally. A stale key raises
    :class:`StaleMetricError` naming the key and the PR that would fix it, at
    the line that asked for it.

    Iteration and ``len`` cover only the readable keys, so a consumer that
    enumerates metrics sees the honest set rather than one that explodes
    halfway through. ``stale`` exposes what was withheld, which is what lets a
    detection matrix render "no detector exists" instead of "zero detections" —
    different claims, and only one of them is true.
    """

    def __init__(
        self, compiled_metrics: "Mapping[str, Any]", program: MotionProgram
    ) -> None:
        self._metrics = dict(compiled_metrics)
        self.stale = stale_after_mutation(compiled_metrics, program)

    def __getitem__(self, key: str) -> Any:
        if key in self.stale:
            raise StaleMetricError(
                f"{key!r} was measured before this clip was mutated and no "
                "longer describes it. The analysis layer does not own it yet "
                f"(see DEFERRED_TO_COMPILER); recompute it or drop the claim "
                "that depends on it."
            )
        return self._metrics[key]

    def __iter__(self) -> "Iterator[str]":
        return iter(key for key in self._metrics if key not in self.stale)

    def __len__(self) -> int:
        return len(self._metrics) - len(self.stale)
