from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import struct
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Protocol
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from rigby_poc.models import ClipFrame, Hand
from rigby_poc.quality import quality_reference, shake_joint_oscillation_metrics
from scipy.spatial.transform import Rotation


CAPTURE_WIDTH = 1600
CAPTURE_HEIGHT = 900
CAPTURE_FOV_DEG = 94.0
SAFE_RESULT_ID = re.compile(r"^[a-zA-Z0-9-]+$")


def motion_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    clip = payload.get("clip") if isinstance(payload.get("clip"), dict) else payload
    metrics = clip.get("metrics") if isinstance(clip.get("metrics"), dict) else {}
    measured_joint_motion: dict[str, float] = {}
    raw_frames = clip.get("frames") if isinstance(clip.get("frames"), list) else []
    program = payload.get("program") if isinstance(payload.get("program"), dict) else {}
    intent = str(program.get("intent", ""))
    oscillation_ranges = [
        (float(item["start_s"]), float(item["end_s"]))
        for item in metrics.get("phase_ranges_s", [])
        if isinstance(item, dict) and item.get("kind") in {"shake", "cycle"}
    ]
    if raw_frames and oscillation_ranges:
        try:
            frames = [ClipFrame.model_validate(frame) for frame in raw_frames]
            raw_hands = program.get("hands")
            hands = (
                [Hand(str(item).lower()) for item in raw_hands]
                if isinstance(raw_hands, list) and raw_hands
                else [Hand(str(program.get("hand", "right")).lower())]
            )
            per_hand_motion = [
                shake_joint_oscillation_metrics(frames, hand, oscillation_ranges)
                for hand in hands
            ]
            measured_joint_motion = {
                "forearm_rotation_cycles": min(
                    item["forearm_rotation_cycles"] for item in per_hand_motion
                ),
                "forearm_rotation_amplitude_rad": min(
                    item["forearm_rotation_amplitude_rad"] for item in per_hand_motion
                ),
                "wrist_flexion_cycles": max(
                    item["wrist_flexion_cycles"] for item in per_hand_motion
                ),
                "wrist_flexion_amplitude_rad": max(
                    item["wrist_flexion_amplitude_rad"] for item in per_hand_motion
                ),
                "wrist_deviation_cycles": max(
                    item["wrist_deviation_cycles"] for item in per_hand_motion
                ),
                "wrist_deviation_amplitude_rad": max(
                    item["wrist_deviation_amplitude_rad"] for item in per_hand_motion
                ),
            }
        except (KeyError, TypeError, ValueError):
            measured_joint_motion = {}
    limits = quality_reference()["hard_limits"]
    angular_measurements = {
        "max_angular_velocity_rad_s": {
            "value": metrics.get("max_angular_velocity_rad_s"),
        },
        "max_angular_acceleration_rad_s2": {
            "value": metrics.get("max_angular_acceleration_rad_s2"),
        },
        "max_angular_jerk_rad_s3": {
            "value": metrics.get("max_angular_jerk_rad_s3"),
        },
    }
    if intent != "full_body":
        angular_measurements["max_angular_velocity_rad_s"]["maximum_reference"] = limits[
            "angular_velocity_rad_s"
        ]
        angular_measurements["max_angular_acceleration_rad_s2"]["maximum_reference"] = limits[
            "angular_acceleration_rad_s2"
        ]
        angular_measurements["max_angular_jerk_rad_s3"]["maximum_reference"] = limits[
            "angular_jerk_rad_s3"
        ]
    measurements = {
        **angular_measurements,
        "max_frame_rotation_delta_rad": {
            "value": metrics.get("max_frame_rotation_delta_rad"),
            "maximum_reference": 0.35,
        },
        "rotational_discontinuities": {
            "value": metrics.get("discontinuities"),
            "maximum_reference": 0,
        },
        "max_wrist_swing_rad": {
            "value": metrics.get("max_wrist_swing_rad"),
            "maximum_reference": limits["wrist_swing_rad"],
        },
        "max_wrist_twist_rad": {
            "value": metrics.get("max_wrist_twist_rad"),
            "maximum_reference": limits["wrist_twist_rad"],
        },
        "max_forearm_twist_rad": {
            "value": metrics.get("max_forearm_twist_rad"),
            "maximum_reference": limits["forearm_twist_rad"],
        },
        "forearm_rotation_cycles": {
            "value": measured_joint_motion.get(
                "forearm_rotation_cycles", metrics.get("forearm_rotation_cycles")
            ),
        },
        "forearm_rotation_amplitude_rad": {
            "value": measured_joint_motion.get(
                "forearm_rotation_amplitude_rad",
                metrics.get("forearm_rotation_amplitude_rad"),
            ),
        },
        "wrist_flexion_cycles": {
            "value": measured_joint_motion.get(
                "wrist_flexion_cycles", metrics.get("wrist_flexion_cycles")
            ),
        },
        "wrist_flexion_amplitude_rad": {
            "value": measured_joint_motion.get(
                "wrist_flexion_amplitude_rad", metrics.get("wrist_flexion_amplitude_rad")
            ),
        },
        "wrist_deviation_cycles": {
            "value": measured_joint_motion.get(
                "wrist_deviation_cycles", metrics.get("wrist_deviation_cycles")
            ),
        },
        "wrist_deviation_amplitude_rad": {
            "value": measured_joint_motion.get(
                "wrist_deviation_amplitude_rad", metrics.get("wrist_deviation_amplitude_rad")
            ),
        },
        "self_collision_frames": {
            "value": metrics.get("self_collision_frames"),
            "maximum_reference": 0,
        },
        "active_hand_visibility_fraction": {
            "value": metrics.get("active_hand_visibility_fraction"),
            "minimum_reference": limits["minimum_active_hand_visibility_fraction"],
        },
        "parallel_forearm_max_axis_error_deg": {
            "value": metrics.get("parallel_forearm_max_axis_error_deg"),
            "maximum_reference": 20.0,
        },
        "parallel_forearm_max_frontal_axis_error_deg": {
            "value": metrics.get(
                "parallel_forearm_max_frontal_axis_error_deg"
            ),
            "maximum_reference": 8.0,
        },
        "parallel_forearm_minimum_separation_m": {
            "value": metrics.get("parallel_forearm_minimum_separation_m"),
            "minimum_reference": 0.025,
        },
        "parallel_forearm_minimum_hand_separation_m": {
            "value": metrics.get(
                "parallel_forearm_minimum_hand_separation_m"
            ),
            "minimum_reference": 0.28,
        },
        "travel_wheel_cross_body_fraction": {
            "value": metrics.get("travel_wheel_cross_body_fraction"),
            "minimum_reference": 0.95,
        },
        "travel_wheel_maximum_opposite_elbow_distance_m": {
            "value": metrics.get(
                "travel_wheel_maximum_opposite_elbow_distance_m"
            ),
            "maximum_reference": 0.20,
        },
        "travel_wheel_vertical_order_range_m": {
            "value": metrics.get("travel_wheel_vertical_order_range_m"),
            "minimum_reference": 0.30,
        },
        "travel_wheel_depth_order_range_m": {
            "value": metrics.get("travel_wheel_depth_order_range_m"),
            "minimum_reference": 0.16,
        },
        "object_lift_height_m": {
            "value": metrics.get("lift_height_m"),
        },
        "opposing_finger_contacts": {
            "value": metrics.get("opposing_contacts"),
            "required": True,
        },
        "lost_table_contact": {
            "value": metrics.get("lost_table_contact"),
            "required": True,
        },
        "object_vertical_drift_m": {
            "value": metrics.get("vertical_drift_m"),
            "maximum_reference": 0.015,
        },
        "palm_relative_object_slip_m": {
            "value": metrics.get("palm_relative_slip_m"),
            "maximum_reference": 0.020,
        },
        "maximum_object_penetration_m": {
            "value": metrics.get("max_penetration_m"),
            "maximum_reference": 0.004,
        },
        "object_attachment_slip_m": {
            "value": metrics.get("palm_relative_object_slip_m"),
            "maximum_reference": 0.005,
        },
        "object_max_step_m": {
            "value": metrics.get("object_max_step_m"),
            "maximum_reference": metrics.get("object_max_step_reference_m"),
        },
        "handoff_dual_contact_duration_s": {
            "value": metrics.get("handoff_dual_contact_duration_s"),
            "minimum_reference": 0.08,
        },
        "handoff_receiver_retained": {
            "value": metrics.get("handoff_receiver_retained"),
            "required": True,
        },
        "handoff_attachment_slip_m": {
            "value": metrics.get("handoff_attachment_slip_m"),
            "maximum_reference": 0.005,
        },
        "catch_intercept_error_m": {
            "value": metrics.get("catch_intercept_error_m"),
            "maximum_reference": 0.015,
        },
        "landing_height_error_m": {
            "value": metrics.get("landing_height_error_m"),
            "maximum_reference": 0.010,
        },
        "object_flight_distance_m": {
            "value": metrics.get("object_flight_distance_m"),
        },
        "object_flight_duration_s": {
            "value": metrics.get("object_flight_duration_s"),
        },
        "object_apex_height_m": {
            "value": metrics.get("object_apex_height_m"),
        },
        "object_guided_projected_distance_m": {
            "value": metrics.get("object_guided_projected_distance_m"),
            "minimum_reference": (
                float(metrics.get("object_guided_distance_m", 0.0)) * 0.75
                if metrics.get("object_guided_distance_m") is not None
                else None
            ),
        },
        "object_guided_lateral_error_m": {
            "value": metrics.get("object_guided_lateral_error_m"),
            "maximum_reference": 0.06,
        },
        "object_guided_support_height_error_m": {
            "value": metrics.get("object_guided_support_height_error_m"),
            "maximum_reference": 0.01,
        },
        "object_expected_roll_turns": {
            "value": metrics.get("object_expected_roll_turns"),
        },
        "object_measured_roll_turns": {
            "value": metrics.get("object_measured_roll_turns"),
            "minimum_reference": (
                float(metrics.get("object_expected_roll_turns", 0.0)) * 0.75
                if metrics.get("object_expected_roll_turns") is not None
                else None
            ),
        },
        "object_measured_support_spin_turns": {
            "value": metrics.get("object_measured_support_spin_turns"),
            "minimum_reference": (
                abs(float(metrics.get("object_spin_turns", 0.0))) * 0.90
                if metrics.get("object_spin_turns") is not None
                else None
            ),
        },
        "object_placement_horizontal_distance_m": {
            "value": metrics.get("object_placement_horizontal_distance_m"),
        },
        "carried_object_ids": {
            "value": metrics.get("carried_object_ids"),
        },
        "carried_object_max_step_m": {
            "value": metrics.get("carried_object_max_step_m"),
            "maximum_reference": metrics.get("carried_object_max_step_reference_m"),
        },
        "stateful_object_transition_count": {
            "value": metrics.get("stateful_object_transition_count"),
        },
        "stateful_object_release_displacement_m": {
            "value": metrics.get("stateful_object_release_displacement_m"),
        },
        "stateful_object_landing_height_error_m": {
            "value": metrics.get("stateful_object_landing_height_error_m"),
            "maximum_reference": 0.01,
        },
        "strike_wrist_path_length_m": {
            "value": metrics.get("strike_wrist_path_length_m"),
        },
        "strike_lateral_excursion_m": {
            "value": metrics.get("strike_lateral_excursion_m"),
        },
        "strike_forward_excursion_m": {
            "value": metrics.get("strike_forward_excursion_m"),
        },
        "impact_elbow_angle_deg": {
            "value": metrics.get("impact_elbow_angle_deg"),
            "hook_reference": "approximately 70-120 degrees at impact",
        },
        "active_hands": {
            "value": metrics.get("active_hands"),
        },
        "ordered_intra_hand_contact_count": {
            "value": metrics.get("intra_hand_contact_count"),
            "minimum_reference": len(
                metrics.get("intra_hand_contact_expected_order", []) or []
            ),
        },
        "ordered_intra_hand_contact_sequence": {
            "value": metrics.get("intra_hand_contact_observed_order"),
            "required_sequence": metrics.get("intra_hand_contact_expected_order"),
        },
        "intra_hand_contact_records": {
            "value": metrics.get("intra_hand_contact_records"),
        },
        "intra_hand_release_separation_m": {
            "value": metrics.get("intra_hand_minimum_release_separation_m"),
            "minimum_reference": 0.025,
        },
        "gaze_max_endpoint_angle_deg": {
            "value": metrics.get("gaze_max_endpoint_angle_deg"),
            "maximum_reference": 12.0,
        },
        "trajectory_cycles": {
            "value": metrics.get("trajectory_cycles"),
        },
        "trajectory_amplitude_m": {
            "value": metrics.get("trajectory_amplitude_m"),
        },
        "root_path_length_m": {
            "value": metrics.get("root_path_length_m"),
        },
        "root_displacement_m": {
            "value": metrics.get("root_displacement_m"),
        },
        "root_vertical_min_m": {
            "value": metrics.get("root_vertical_min_m"),
        },
        "root_vertical_max_m": {
            "value": metrics.get("root_vertical_max_m"),
        },
        "final_root_yaw_deg": {
            "value": metrics.get("final_root_yaw_deg"),
        },
        "requested_body_cycles": {
            "value": metrics.get("requested_body_cycles"),
        },
        "requested_dance_beats": {
            "value": metrics.get("requested_dance_beats"),
        },
        "measured_dance_beats": {
            "value": metrics.get("measured_dance_beats"),
        },
        "dance_alternating_lift_count": {
            "value": metrics.get("dance_alternating_lift_count"),
        },
        "dance_lateral_root_range_m": {
            "value": metrics.get("dance_lateral_root_range_m"),
            "minimum_reference": (
                0.035
                if int(metrics.get("requested_dance_beats", 0) or 0) == 1
                else 0.070
            ),
        },
        "dance_left_foot_peak_clearance_m": {
            "value": metrics.get("dance_left_foot_peak_clearance_m"),
            "minimum_reference": 0.050,
        },
        "dance_right_foot_peak_clearance_m": {
            "value": metrics.get("dance_right_foot_peak_clearance_m"),
            "minimum_reference": (
                0.050
                if int(metrics.get("requested_dance_beats", 0) or 0) >= 2
                else None
            ),
        },
        "dance_phase_metrics": {
            "value": metrics.get("dance_phase_metrics"),
        },
        "requested_climb_height_m": {
            "value": metrics.get("requested_climb_height_m"),
        },
        "measured_climb_height_m": {
            "value": metrics.get("measured_climb_height_m"),
        },
        "climb_vertical_completion_fraction": {
            "value": metrics.get("climb_vertical_completion_fraction"),
            "minimum_reference": 0.90,
            "maximum_reference": 1.08,
        },
        "requested_climb_cycles": {
            "value": metrics.get("requested_climb_cycles"),
        },
        "climb_support_target_max_error_m": {
            "value": metrics.get("climb_support_target_max_error_m"),
            "maximum_reference": 0.16,
        },
        "climb_three_point_support_fraction": {
            "value": metrics.get("climb_three_point_support_fraction"),
            "minimum_reference": 0.70,
        },
        "climb_final_supported_limb_count": {
            "value": metrics.get("climb_final_supported_limb_count"),
            "minimum_reference": 3,
        },
        "climb_missing_support_object_count": {
            "value": metrics.get("climb_missing_support_object_count"),
            "maximum_reference": 0,
        },
        "climb_phase_metrics": {
            "value": metrics.get("climb_phase_metrics"),
        },
        "final_balanced_leg_error_rad": {
            "value": metrics.get("final_balanced_leg_error_rad"),
            "note": "legacy local-pose distance; not a balance acceptance bound",
        },
        "final_support_center_offset_m": {
            "value": metrics.get("final_support_center_offset_m"),
            "maximum_reference": 0.14,
        },
        "final_foot_ground_error_m": {
            "value": metrics.get("final_foot_ground_error_m"),
            "maximum_reference": 0.025,
        },
        "final_root_vertical_speed_m_s": {
            "value": metrics.get("final_root_vertical_speed_m_s"),
            "maximum_reference": 0.04,
        },
        "ground_penetration_m": {
            "value": metrics.get("ground_penetration_m"),
            "maximum_reference": 0.012,
        },
        "maximum_foot_clearance_m": {
            "value": metrics.get("maximum_foot_clearance_m"),
        },
        "airborne_frame_count": {
            "value": metrics.get("airborne_frame_count"),
        },
        "max_support_foot_target_error_m": {
            "value": metrics.get("max_support_foot_target_error_m"),
            "maximum_reference": 0.012,
        },
        "max_support_foot_slide_per_frame_m": {
            "value": metrics.get("max_support_foot_slide_per_frame_m"),
            "maximum_reference": 0.006,
        },
        "support_contact_fraction": {
            "value": metrics.get("support_contact_fraction"),
            "minimum_reference": 0.98,
        },
        "horizontal_pose_variant": {
            "value": metrics.get("horizontal_pose_variant")
            or metrics.get("horizontal_pose_variants"),
        },
        "horizontal_pose_contact_points": {
            "value": metrics.get("horizontal_pose_contact_point_count"),
            "minimum_reference": (
                4
                if metrics.get("horizontal_pose_variant") == "quadruped"
                else 2
            ),
        },
        "horizontal_pose_ground_clearance_m": {
            "value": metrics.get("horizontal_pose_minimum_clearance_m"),
            "minimum_reference": -0.012,
        },
        "horizontal_body_axis_vertical_fraction": {
            "value": metrics.get("horizontal_body_axis_vertical_fraction"),
            "maximum_reference": 0.35,
        },
        "requested_push_up_cycles": {
            "value": metrics.get("requested_push_up_cycles"),
        },
        "push_up_vertical_excursion_m": {
            "value": metrics.get("push_up_vertical_excursion_m"),
            "minimum_reference": (
                0.06
                if float(metrics.get("requested_push_up_cycles", 0.0) or 0.0) > 0.0
                else None
            ),
        },
        "plank_hold_vertical_range_m": {
            "value": metrics.get("plank_hold_vertical_range_m"),
            "maximum_reference": 0.025,
        },
        "measured_push_up_cycles": {
            "value": metrics.get("measured_push_up_cycles"),
        },
        "push_up_palm_height_range_m": {
            "value": metrics.get("push_up_palm_height_range_m"),
            "maximum_reference": 0.045,
        },
        "push_up_toe_support_clearance_m": {
            "value": metrics.get("push_up_toe_support_clearance_m"),
            "maximum_reference": 0.035,
        },
        "push_up_toe_height_range_m": {
            "value": metrics.get("push_up_toe_height_range_m"),
            "maximum_reference": 0.015,
        },
        "push_up_toe_position_range_m": {
            "value": metrics.get("push_up_toe_position_range_m"),
            "maximum_reference": 0.02,
        },
        "push_up_min_knee_extension_deg": {
            "value": metrics.get("push_up_min_knee_extension_deg"),
            "minimum_reference": 150.0,
        },
        "requested_crawl_cycles": {
            "value": metrics.get("requested_crawl_cycles"),
        },
        "requested_crawl_distance_m": {
            "value": metrics.get("requested_crawl_distance_m"),
        },
        "crawl_root_displacement_m": {
            "value": metrics.get("crawl_root_displacement_m"),
        },
        "crawl_hand_alternation_range_m": {
            "value": metrics.get("crawl_hand_alternation_range_m"),
            "minimum_reference": 0.10,
        },
        "requested_jumping_jack_cycles": {
            "value": metrics.get("requested_jumping_jack_cycles"),
        },
        "measured_jumping_jack_cycles": {
            "value": metrics.get("measured_jumping_jack_cycles"),
        },
        "jumping_jack_foot_spread_excursion_m": {
            "value": metrics.get("jumping_jack_foot_spread_excursion_m"),
            "minimum_reference": 0.30,
        },
        "jumping_jack_max_wrist_height_m": {
            "value": metrics.get("jumping_jack_max_wrist_height_m"),
            "minimum_reference": 1.65,
        },
        "requested_burpee_cycles": {
            "value": metrics.get("requested_burpee_cycles"),
        },
        "measured_burpee_jump_cycles": {
            "value": metrics.get("measured_burpee_jump_cycles"),
        },
        "measured_burpee_push_up_cycles": {
            "value": metrics.get("measured_burpee_push_up_cycles"),
        },
        "burpee_floor_support_phase_count": {
            "value": metrics.get("burpee_floor_support_phase_count"),
        },
        "burpee_airborne_phase_count": {
            "value": metrics.get("burpee_airborne_phase_count"),
        },
        "burpee_root_vertical_excursion_m": {
            "value": metrics.get("burpee_root_vertical_excursion_m"),
            "minimum_reference": 0.55,
        },
        "burpee_max_wrist_height_m": {
            "value": metrics.get("burpee_max_wrist_height_m"),
            "minimum_reference": 1.65,
        },
        "requested_squat_cycles": {
            "value": metrics.get("requested_squat_cycles"),
        },
        "measured_squat_cycles": {
            "value": metrics.get("measured_squat_cycles"),
        },
        "squat_minimum_depth_m": {
            "value": metrics.get("squat_minimum_depth_m"),
            "minimum_reference": 0.10,
        },
        "squat_max_stance_return_error_m": {
            "value": metrics.get("squat_max_stance_return_error_m"),
            "maximum_reference": 0.035,
        },
        "requested_lunge_cycles": {
            "value": metrics.get("requested_lunge_cycles"),
        },
        "measured_lunge_cycles": {
            "value": metrics.get("measured_lunge_cycles"),
        },
        "lunge_minimum_root_drop_m": {
            "value": metrics.get("lunge_minimum_root_drop_m"),
            "minimum_reference": 0.10,
        },
        "lunge_minimum_foot_stagger_m": {
            "value": metrics.get("lunge_minimum_foot_stagger_m"),
            "minimum_reference": 0.12,
        },
        "lunge_max_stance_return_error_m": {
            "value": metrics.get("lunge_max_stance_return_error_m"),
            "maximum_reference": 0.035,
        },
        "single_leg_balance_phase_count": {
            "value": metrics.get("single_leg_balance_phase_count"),
        },
        "single_leg_min_raised_foot_clearance_m": {
            "value": metrics.get("single_leg_min_raised_foot_clearance_m"),
            "minimum_reference": 0.18,
        },
        "single_leg_max_support_foot_slide_m": {
            "value": metrics.get("single_leg_max_support_foot_slide_m"),
            "maximum_reference": 0.01,
        },
        "single_leg_support_contact_fraction": {
            "value": metrics.get("single_leg_support_contact_fraction"),
            "minimum_reference": 0.95,
        },
        "requested_sit_up_cycles": {
            "value": metrics.get("requested_sit_up_cycles"),
        },
        "measured_sit_up_cycles": {
            "value": metrics.get("measured_sit_up_cycles"),
        },
        "sit_up_minimum_head_lift_m": {
            "value": metrics.get("sit_up_minimum_head_lift_m"),
            "minimum_reference": 0.15,
        },
        "sit_up_max_supine_return_error_m": {
            "value": metrics.get("sit_up_max_supine_return_error_m"),
            "maximum_reference": 0.05,
        },
        "requested_body_rotation_degrees": {
            "value": metrics.get("requested_body_rotation_degrees"),
        },
        "measured_body_rotation_degrees": {
            "value": metrics.get("measured_body_rotation_degrees"),
        },
        "minimum_body_rotation_completion_fraction": {
            "value": metrics.get("minimum_body_rotation_completion_fraction"),
            "minimum_reference": 0.94,
            "maximum_reference": 1.08,
        },
        "minimum_rotation_travel_completion_fraction": {
            "value": metrics.get("minimum_rotation_travel_completion_fraction"),
            "minimum_reference": 0.90,
        },
        "floor_roll_nonfoot_contact_fraction": {
            "value": metrics.get("floor_roll_nonfoot_contact_fraction"),
            "minimum_reference": 0.20,
        },
        "cartwheel_hand_contact_frame_count": {
            "value": metrics.get("cartwheel_hand_contact_frame_count"),
            "minimum_reference": 1,
        },
        "cartwheel_inverted_frame_count": {
            "value": metrics.get("cartwheel_inverted_frame_count"),
            "minimum_reference": 1,
        },
        "cartwheel_max_foot_clearance_m": {
            "value": metrics.get("cartwheel_max_foot_clearance_m"),
            "minimum_reference": 0.60,
        },
        "cartwheel_minimum_head_clearance_m": {
            "value": metrics.get("cartwheel_minimum_head_clearance_m"),
            "minimum_reference": 0.08,
        },
        "airborne_rotation_airborne_frame_count": {
            "value": metrics.get("airborne_rotation_airborne_frame_count"),
            "minimum_reference": 1,
        },
        "body_rotation_phase_metrics": {
            "value": metrics.get("body_rotation_phase_metrics"),
        },
        "obstacle_missing_target_count": {
            "value": metrics.get("obstacle_missing_target_count"),
            "maximum_reference": 0,
        },
        "minimum_obstacle_step_foot_clearance_m": {
            "value": metrics.get("minimum_obstacle_step_foot_clearance_m"),
            "minimum_reference": 0.02,
        },
        "maximum_obstacle_step_crossing_error_m": {
            "value": metrics.get("maximum_obstacle_step_crossing_error_m"),
        },
        "minimum_obstacle_avoidance_root_clearance_m": {
            "value": metrics.get("minimum_obstacle_avoidance_root_clearance_m"),
        },
        "obstacle_traversal_phase_metrics": {
            "value": metrics.get("obstacle_traversal_phase_metrics"),
        },
    }
    if intent and intent != "grab":
        for key in (
            "object_lift_height_m",
            "opposing_finger_contacts",
            "lost_table_contact",
            "object_vertical_drift_m",
            "palm_relative_object_slip_m",
            "maximum_object_penetration_m",
        ):
            measurements.pop(key, None)
    if intent != "object_interaction":
        for key in (
            "object_attachment_slip_m",
            "object_max_step_m",
            "catch_intercept_error_m",
            "landing_height_error_m",
            "object_flight_distance_m",
            "object_flight_duration_s",
            "object_apex_height_m",
            "object_guided_projected_distance_m",
            "object_guided_lateral_error_m",
            "object_guided_support_height_error_m",
        ):
            measurements.pop(key, None)
    return {
        "schema_version": "1.0",
        "role": "objective measurements from the rendered animation; no acceptance label",
        "measurements": measurements,
        "finger_shape": finger_shape_diagnostics(payload),
    }


def finger_shape_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    clip = payload.get("clip") if isinstance(payload.get("clip"), dict) else payload
    program = payload.get("program") if isinstance(payload.get("program"), dict) else {}
    frames = clip.get("frames") if isinstance(clip.get("frames"), list) else []
    metrics = clip.get("metrics") if isinstance(clip.get("metrics"), dict) else {}
    hold_ranges = [
        (float(item["start_s"]), float(item["end_s"]))
        for item in metrics.get("phase_ranges_s", [])
        if isinstance(item, dict) and item.get("kind") == "hold"
    ]
    if not frames or not hold_ranges:
        return {}
    hold_start, hold_end = hold_ranges[0]
    hold_midpoint = (hold_start + hold_end) / 2.0
    hold_frames = [
        frame
        for frame in frames
        if isinstance(frame, dict)
        and hold_start - 1e-8 <= float(frame.get("time_s", -1.0)) <= hold_end + 1e-8
    ]
    if not hold_frames:
        return {}
    frame = min(hold_frames, key=lambda item: abs(float(item["time_s"]) - hold_midpoint))
    hand = str(program.get("hand", "right")).lower()
    segments = {
        "thumb": "ThumbMetacarpal",
        "index": "IndexProximal",
        "middle": "MiddleProximal",
        "ring": "RingProximal",
        "little": "LittleProximal",
    }
    normalized_curls: dict[str, float] = {}
    bones = frame.get("bones") if isinstance(frame.get("bones"), dict) else {}
    for digit, suffix in segments.items():
        bone = bones.get(f"{hand}{suffix}")
        rotation = bone.get("rotation") if isinstance(bone, dict) else None
        if not isinstance(rotation, dict):
            continue
        quaternion = [float(rotation[key]) for key in ("x", "y", "z", "w")]
        curl_angle = abs(float(Rotation.from_quat(quaternion).as_euler("xyz")[0]))
        normalized_curls[digit] = min(1.0, curl_angle / (0.95 if digit == "thumb" else 1.15))
    requested_shape = next(
        (
            primitive.get("hand_shape")
            for primitive in program.get("primitives", [])
            if isinstance(primitive, dict) and primitive.get("kind") == "hold"
        ),
        None,
    )
    references = {
        "hang_ten": {
            "thumb": {"maximum_normalized_curl": 0.35},
            "index": {"minimum_normalized_curl": 0.65},
            "middle": {"minimum_normalized_curl": 0.65},
            "ring": {"minimum_normalized_curl": 0.65},
            "little": {"maximum_normalized_curl": 0.35},
        },
        "thumbs_up": {
            "thumb": {"maximum_normalized_curl": 0.30},
            "index": {"minimum_normalized_curl": 0.70},
            "middle": {"minimum_normalized_curl": 0.70},
            "ring": {"minimum_normalized_curl": 0.70},
            "little": {"minimum_normalized_curl": 0.70},
        },
        "peace": {
            "index": {"maximum_normalized_curl": 0.30},
            "middle": {"maximum_normalized_curl": 0.30},
            "ring": {"minimum_normalized_curl": 0.70},
            "little": {"minimum_normalized_curl": 0.70},
        },
    }
    reference = references.get(str(requested_shape), {})
    return {
        "sample_time_s": float(frame["time_s"]),
        "requested_hand_shape": requested_shape,
        "normalized_curls": normalized_curls,
        "human_calibrated_reference": reference,
    }


def refresh_motion_diagnostics(
    manifest_path: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_id = str(manifest.get("result_id", ""))
    if not SAFE_RESULT_ID.fullmatch(result_id):
        raise ValueError("manifest result id contains unsupported characters")
    manifest["motion_diagnostics"] = motion_diagnostics(_result_payload(base_url, result_id))
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def _result_payload(base_url: str, result_id: str) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/api/v1/results/{quote(result_id)}",
        headers={"Accept": "application/json"},
    )
    with urlopen(request, timeout=20) as response:  # noqa: S310 - local API is caller-controlled
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("result endpoint did not return an object")
    return value


def phase_sampling_points(payload: dict[str, Any]) -> list[dict[str, float | str]]:
    clip = payload.get("clip") if isinstance(payload.get("clip"), dict) else payload
    program = payload.get("program") if isinstance(payload.get("program"), dict) else {}
    sequence_steps = (
        program.get("steps") if isinstance(program.get("steps"), list) else []
    )
    full_body_program = program.get("intent") == "full_body" or any(
        isinstance(step, dict) and step.get("intent") == "full_body"
        for step in sequence_steps
    )
    primitives = program.get("primitives") if isinstance(program.get("primitives"), list) else []
    if sequence_steps:
        primitives = [
            primitive
            for step in sequence_steps
            if isinstance(step, dict) and isinstance(step.get("primitives"), list)
            for primitive in step["primitives"]
            if isinstance(primitive, dict)
        ]
    body_primitives = [
        primitive
        for primitive in primitives
        if isinstance(primitive, dict) and primitive.get("kind") == "body"
    ]
    primitive_by_label = {
        str(primitive.get("label")): primitive
        for primitive in primitives
        if isinstance(primitive, dict) and primitive.get("label")
    }
    dexterous_program = any(
        isinstance(primitive, dict)
        and isinstance(primitive.get("intra_hand_contact"), dict)
        for primitive in primitives
    )
    body_primitive_index = 0
    metrics = clip.get("metrics") if isinstance(clip.get("metrics"), dict) else {}
    obstacle_records = (
        metrics.get("obstacle_traversal_phase_metrics")
        if isinstance(metrics.get("obstacle_traversal_phase_metrics"), list)
        else []
    )
    raw_ranges = metrics.get("phase_ranges_s")
    ranges = raw_ranges if isinstance(raw_ranges, list) else []
    points: list[dict[str, float | str]] = []
    occurrence: dict[str, int] = {}
    shake_cycles = float(metrics.get("wrist_shake_cycles", 0.0) or 0.0)
    shake_extrema = (
        tuple(
            (
                (2 * index + 1) / (4.0 * shake_cycles),
                f"shake_extreme_{index + 1}",
            )
            for index in range(max(1, int(round(shake_cycles * 2.0))))
        )
        if shake_cycles > 0.0
        else ((0.50, "shake_midpoint"),)
    )
    trajectory_cycles = float(metrics.get("trajectory_cycles", 0.0) or 0.0)
    cycle_extrema = (
        tuple(
            (
                (2 * index + 1) / (4.0 * trajectory_cycles),
                f"cycle_extreme_{index + 1}",
            )
            for index in range(max(1, min(16, int(round(trajectory_cycles * 2.0)))))
        )
        if trajectory_cycles > 0.0
        else ((0.50, "cycle_midpoint"),)
    )
    fractions = {
        "guard": (
            (0.20, "guard_rising"),
            (0.80, "guard_ready"),
        ),
        "load": (
            (0.20, "load_start"),
            (0.80, "load_ready"),
        ),
        "strike": (
            (0.10, "strike_start"),
            (0.35, "strike_early"),
            (0.60, "strike_midpoint"),
            (0.90, "impact_pose"),
        ),
        "follow_through": (
            (0.25, "follow_through_early"),
            (0.75, "follow_through_late"),
        ),
        "reach": (
            (0.10, "reach_start"),
            (0.50, "reach_midpoint"),
            (0.90, "reach_end"),
        ),
        "preshape": (
            (0.20, "preshape_start"),
            (0.80, "preshape_ready"),
        ),
        "contact": ((0.50, "contact_established"),),
        "close": (
            (0.20, "close_start"),
            (0.80, "grasp_closed"),
        ),
        "lift": (
            (0.15, "lift_start"),
            (0.55, "lift_midpoint"),
            (0.90, "lift_clear"),
        ),
        "windup": (
            (0.20, "windup_start"),
            (0.80, "windup_ready"),
        ),
        "release": (
            (0.20, "release_start"),
            (0.75, "release_opening"),
            (0.95, "release_pose"),
        ),
        "flight": (
            (0.08, "flight_start"),
            (0.35, "flight_rising"),
            (0.55, "flight_apex"),
            (0.82, "flight_descending"),
            (0.97, "flight_end"),
        ),
        "receive": (
            (0.20, "receive_start"),
            (0.80, "receive_ready"),
        ),
        "absorb": (
            (0.20, "absorb_start"),
            (0.80, "absorb_end"),
        ),
        "present": (
            (0.10, "present_start"),
            (0.30, "present_early"),
            (0.50, "present_motion"),
            (0.70, "present_late"),
            (0.90, "presented_pose"),
        ),
        "hold": (
            (0.10, "hold_start"),
            (0.50, "hold_midpoint"),
            (0.90, "hold_end"),
        ),
        "shake": ((0.03, "shake_start"), *shake_extrema, (0.97, "shake_end")),
        "move": (
            (0.10, "move_start"),
            (0.50, "move_midpoint"),
            (0.90, "move_end"),
        ),
        "cycle": ((0.03, "cycle_start"), *cycle_extrema, (0.97, "cycle_end")),
        "body": (
            (0.05, "body_start"),
            (0.25, "body_early"),
            (0.50, "body_midpoint"),
            (0.75, "body_late"),
            (0.95, "body_end"),
        ),
        "recover": (
            (0.10, "recover_start"),
            (0.30, "recover_early"),
            (0.50, "recovery_motion"),
            (0.70, "recover_late"),
            (0.90, "recover_end"),
        ),
    }
    object_actions = {
        str(value)
        for value in (
            [program.get("object_action")]
            + [
                step.get("object_action")
                for step in sequence_steps
                if isinstance(step, dict)
            ]
        )
        if value
    }
    if "drop" in object_actions:
        fractions["flight"] = (
            (0.08, "fall_start"),
            (0.35, "fall_early"),
            (0.60, "fall_midpoint"),
            (0.85, "fall_late"),
            (0.97, "landed"),
        )
    if "spin" in object_actions:
        fractions["move"] = (
            (0.03, "spin_start"),
            (0.25, "spin_quarter_turn"),
            (0.50, "spin_half_turn"),
            (0.75, "spin_three_quarter_turn"),
            (0.97, "spin_complete"),
        )
    for item in ranges:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        if kind not in fractions:
            continue
        start = float(item.get("start_s", 0.0))
        end = float(item.get("end_s", start))
        if end <= start:
            continue
        occurrence[kind] = occurrence.get(kind, 0) + 1
        suffix = "" if occurrence[kind] == 1 else f"_{occurrence[kind]}"
        phase_fractions = fractions[kind]
        phase_label = str(item.get("label", ""))
        phase_primitive = primitive_by_label.get(phase_label, {})
        if kind == "move" and dexterous_program:
            contact = phase_primitive.get("intra_hand_contact")
            if isinstance(contact, dict):
                suffix = ""
                phase_fractions = (
                    (
                        0.98,
                        "thumb_to_"
                        + str(contact.get("target_digit", "fingertip")),
                    ),
                )
            elif occurrence[kind] == 1:
                phase_fractions = ((0.90, "dexterous_hand_presented"),)
            else:
                # Open release phases are represented by the spacing between
                # consecutive contact tiles. Capturing three generic samples
                # from every release would spend most of the VLM payload on
                # near-duplicate poses and obscure the ordered contacts.
                phase_fractions = ()
        elif kind == "body" and body_primitive_index < len(body_primitives):
            primitive = body_primitives[body_primitive_index]
            body_primitive_index += 1
            body = primitive.get("body") if isinstance(primitive.get("body"), dict) else {}
            parameters = (
                primitive.get("parameters")
                if isinstance(primitive.get("parameters"), dict)
                else {}
            )
            obstacle_mode = str(body.get("obstacle_mode", "none"))
            if obstacle_mode != "none":
                label = str(primitive.get("label", ""))
                record = next(
                    (
                        value
                        for value in obstacle_records
                        if isinstance(value, dict)
                        and str(value.get("label", "")) == label
                    ),
                    None,
                )
                closest_time = (
                    float(record["closest_approach_time_s"])
                    if isinstance(record, dict)
                    and isinstance(record.get("closest_approach_time_s"), (int, float))
                    else start + 0.5 * (end - start)
                )
                closest_fraction = min(
                    0.90,
                    max(0.10, (closest_time - start) / (end - start)),
                )
                phase_fractions = tuple(
                    sorted(
                        {
                            (0.08, "obstacle_approach"),
                            (
                                max(0.10, closest_fraction - 0.13),
                                "obstacle_pre_clearance",
                            ),
                            (closest_fraction, "obstacle_closest_approach"),
                            (
                                min(0.90, closest_fraction + 0.13),
                                "obstacle_post_clearance",
                            ),
                            (0.92, "obstacle_departure"),
                        }
                    )
                )
            concurrent_cycles = float(parameters.get("trajectory_cycles", 0.0) or 0.0)
            concurrent_effectors = (
                primitive.get("effectors")
                if isinstance(primitive.get("effectors"), list)
                else []
            )
            if obstacle_mode != "none":
                pass
            elif body.get("action") == "climb":
                cycles = max(
                    1,
                    min(8, int(round(float(body.get("cycles", 1.0) or 1.0)))),
                )
                phase_fractions = (
                    (0.05, "climb_entry"),
                    (0.28, "climb_contacts_acquired"),
                    *tuple(
                        (
                            0.30 + 0.66 * (index + 0.5) / cycles,
                            f"climb_cycle_{index + 1}",
                        )
                        for index in range(cycles)
                    ),
                    (0.98, "climb_finish"),
                )
            elif body.get("action") == "dance":
                beats = max(
                    1,
                    min(8, int(round(float(body.get("cycles", 1.0) or 1.0)))),
                )
                phase_fractions = (
                    (0.03, "dance_entry"),
                    *tuple(
                        (
                            (index + 0.5) / beats,
                            f"dance_beat_{index + 1}",
                        )
                        for index in range(beats)
                    ),
                    (0.97, "dance_exit"),
                )
            elif (
                concurrent_effectors
                and primitive.get("trajectory") in {"circle", "oscillate"}
                and concurrent_cycles > 0.0
            ):
                pose = body.get("pose") if isinstance(body.get("pose"), dict) else {}
                moving_quadruped = bool(
                    pose.get("support_mode") == "quadruped"
                    and (
                        abs(float(pose.get("root_shift_x_m", 0.0) or 0.0))
                        + abs(float(pose.get("root_shift_z_m", 0.0) or 0.0))
                        > 1e-8
                    )
                )
                cycle_start = 0.28 if moving_quadruped else 0.0
                cycle_span = 0.62 if moving_quadruped else 1.0
                extrema = tuple(
                    (
                        cycle_start
                        + cycle_span
                        * (2 * index + 1)
                        / (4.0 * concurrent_cycles),
                        f"body_arm_cycle_extreme_{index + 1}",
                    )
                    for index in range(
                        max(1, min(16, int(round(concurrent_cycles * 2.0))))
                    )
                )
                phase_fractions = (
                    (
                        cycle_start if moving_quadruped else 0.03,
                        "body_arm_cycle_start",
                    ),
                    *extrema,
                    (
                        cycle_start + cycle_span
                        if moving_quadruped
                        else 0.97,
                        "body_arm_cycle_end",
                    ),
                )
            elif body.get("action") in {"walk", "run"}:
                steps = max(1, int(round(float(body.get("cycles", 1.0) or 1.0))))
                # Sampling every step at local phase 0.5 catches the passing
                # pose: after the first footfall both feet overlap in depth,
                # which makes a real run look like a two-footed hop.  Runs are
                # sampled later in swing, while both feet are airborne and the
                # stride is split; walks use alternating pre/post-passing
                # phases so stance and swing remain legible.
                local_phases = (
                    (0.15, 0.70, 0.70, 0.70, 0.85)
                    if body.get("action") == "run"
                    else (0.15, 0.38, 0.62, 0.38, 0.85)
                )
                base_fractions = (0.05, 0.25, 0.50, 0.75, 0.95)
                phase_fractions = tuple(
                    (
                        (
                            min(steps - 1, int(math.floor(base_fraction * steps)))
                            + local_phase
                        )
                        / steps,
                        label,
                    )
                    for base_fraction, local_phase, (_, label) in zip(
                        base_fractions,
                        local_phases,
                        fractions[kind],
                        strict=True,
                    )
                )
            elif body.get("action") not in {"jump"}:
                # Static or single-transfer skills are fully characterized by
                # entry, peak, and exit.  Five near-duplicate poses dilute a
                # compound-action contact sheet and spend vision tokens without
                # adding evidence.  Multi-cycle jumps retain all five samples
                # so every takeoff/apex/landing remains visible.
                phase_fractions = (
                    (0.10, "body_start"),
                    (0.50, "body_midpoint"),
                    (0.90, "body_end"),
                )
        elif kind == "recover" and dexterous_program:
            phase_fractions = ((0.90, "dexterous_recovered"),)
        elif kind == "recover" and full_body_program:
            phase_fractions = (
                (0.10, "recover_start"),
                (0.50, "recovery_motion"),
                (0.90, "recover_end"),
            )
        for fraction, label in phase_fractions:
            points.append(
                {
                    "phase": kind,
                    "label": f"{label}{suffix}",
                    "time_s": start + (end - start) * fraction,
                }
            )
    if points:
        return points
    duration = float(clip.get("duration_s", clip.get("duration", 0.0)))
    if duration <= 0:
        raise ValueError("clip has no positive duration or phase ranges")
    return [
        {"phase": "unknown", "label": label, "time_s": duration * fraction}
        for fraction, label in ((0.25, "early"), (0.55, "middle"), (0.82, "late"))
    ]


def _png_size(value: bytes) -> tuple[int, int]:
    if len(value) < 24 or value[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("capture is not a PNG")
    return struct.unpack(">II", value[16:24])


ASSET_MANIFEST = Path(__file__).resolve().parents[1] / "assets" / "manifest.json"
HUMANOID_ASSET_ID = "mesh2motion-human-male"

# A seek is sub-second once the page is live, so the generous per-attempt budgets a full
# page reload needed are now the reason a wedged capture can outlive its own subprocess.
NAVIGATION_TIMEOUT_MS = 15_000
READY_TIMEOUT_MS = 15_000
SEEK_TIMEOUT_MS = 15_000
SCREENSHOT_TIMEOUT_MS = 15_000
CAPTURE_ATTEMPTS = 3

# Phase sampling is per intent and grows with the phase count. The bound exists so a
# timeout budget can be computed before the sampling table has been consulted, and it is
# checked rather than assumed, so an intent that outgrows it fails loudly here instead of
# timing out two layers up. Raising it means raising the budget with it -- which is the
# point of deriving one from the other.
#
# Measured across all 47 corpus cases (41 compile; the rest are known-bad by design):
#
#     60 pts   589 frames   full_body            fullbody-burpee-cycle
#     36 pts   163 frames   sequence             sequence-push-then-pull
#     30 pts   221 frames   sequence             sequence-grab-then-wave
#     29 pts   141 frames   object_interaction   object-throw-far
#
# The previous value of 32 was set from a 7-intent sample whose largest was 24, and the
# 47-case corpus that arrived with 03b breaches it on two cases. 96 is 60% above the
# measured maximum. This is the second time this constant has been set from too small a
# sample, so: the number below is a *measurement of the corpus*, and it must be re-derived
# from the corpus rather than adjusted upward until the error stops -- which is what
# `test_the_snapshot_bound_covers_the_corpus` exists to force.
MAX_SNAPSHOTS_PER_VIEW = 96

# Capture enforces its own wall-clock deadline and `pipeline.py` derives the subprocess
# budget from it, so `TimeoutExpired` at the subprocess layer means "capture is wedged",
# never "capture was still working".
#
# Per-snapshot cost measured on real captures, seek path, one browser:
#
#     122 ms   15 snapshots   jab, orbit only
#     116 ms   32 snapshots   composite-beckon-right
#      98 ms   30 snapshots   jab, both views
#      89 ms   24 snapshots   two-step sequence, orbit only
#      65 ms   48 snapshots   two-step sequence, both views
#
# Longer captures are cheaper per snapshot because the fixed per-result cost -- page
# open, result fetch, GLB fetch and parse, roughly 0.75 s -- spreads over more of them.
# 1.0 s is 8x the worst measured. It was 3 s, chosen when the snapshot bound was 32; at
# a bound of 96 that made the five-candidate backstop 52 minutes, which is too loose to
# reap a wedged browser in any useful time.
PER_SNAPSHOT_BUDGET_S = 1.0
PER_RESULT_OVERHEAD_S = 30.0
SESSION_OVERHEAD_S = 30.0
SUBPROCESS_GRACE_S = 60.0


class CaptureAssetError(RuntimeError):
    """The page did not render the humanoid the clip was compiled against.

    Separate from a generic capture failure because it is never transient: retrying a
    missing or mismatched asset just produces the same wrong body again.
    """


def humanoid_asset_sha256(manifest_path: Path | None = None) -> str:
    """The sha256 the humanoid GLB is expected to have, from `assets/manifest.json`."""
    path = manifest_path or ASSET_MANIFEST
    document = json.loads(path.read_text(encoding="utf-8"))
    for asset in document.get("assets", []):
        if isinstance(asset, dict) and asset.get("id") == HUMANOID_ASSET_ID:
            digest = str(asset.get("sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{HUMANOID_ASSET_ID} has no usable sha256 in {path}")
            return digest
    raise ValueError(f"{path} does not describe asset {HUMANOID_ASSET_ID}")


class CaptureDeadlineExceeded(RuntimeError):
    """Capture ran past its own wall-clock budget.

    Raised by capture itself so the failure names the result and the snapshot it was on.
    The subprocess budget in `pipeline.py` sits strictly above this, so an operator
    seeing `TimeoutExpired` instead knows the capture process stopped responding rather
    than merely taking too long.
    """


def capture_deadline_s(
    *,
    result_count: int = 1,
    snapshots_per_result: int = MAX_SNAPSHOTS_PER_VIEW * 2,
) -> float:
    """The wall clock capture allows itself for `result_count` results."""
    per_result = PER_RESULT_OVERHEAD_S + snapshots_per_result * PER_SNAPSHOT_BUDGET_S
    return round(SESSION_OVERHEAD_S + result_count * per_result, 3)


def subprocess_timeout_s(
    *,
    result_count: int = 1,
    snapshots_per_result: int = MAX_SNAPSHOTS_PER_VIEW * 2,
) -> float:
    """The subprocess budget, derived from capture's own deadline plus process grace.

    Plan 05 section 1.4: the hardcoded 300 s in `pipeline.py` was *below* capture's own
    worst case, so the normal failure mode was `TimeoutExpired` escaping a layer that
    did not catch it, leaving Chrome orphaned. Deriving one number from the other makes
    that ordering true by construction rather than by coincidence.
    """
    return round(
        capture_deadline_s(
            result_count=result_count, snapshots_per_result=snapshots_per_result
        )
        + SUBPROCESS_GRACE_S,
        3,
    )


def _clip_document(payload: dict[str, Any]) -> dict[str, Any]:
    clip = payload.get("clip")
    return clip if isinstance(clip, dict) else payload


def _clip_duration_s(clip: dict[str, Any]) -> float:
    duration = clip.get("duration_s", clip.get("duration"))
    if isinstance(duration, (int, float)):
        return float(duration)
    frames = clip.get("frames") if isinstance(clip.get("frames"), list) else []
    return float(frames[-1].get("time_s", 0.0)) if frames else 0.0


def frame_at(clip: dict[str, Any], time_s: float) -> dict[str, Any] | None:
    """The frame the page will render for `time_s`.

    This mirrors `frontend/src/motion.ts` `frameAt` exactly -- the last frame at or
    before the requested time, with no interpolation, clamped to the clip duration.
    Pose hashing is only meaningful if Python selects the same frame the renderer did,
    and `_verify_rendered_frame` asserts that agreement on every snapshot rather than
    trusting this comment.
    """
    frames = clip.get("frames") if isinstance(clip.get("frames"), list) else []
    if not frames:
        return None
    requested = min(float(time_s), _clip_duration_s(clip))
    best = frames[0]
    for frame in frames:
        if float(frame.get("time_s", frame.get("time", 0.0))) > requested:
            break
        best = frame
    return best if isinstance(best, dict) else None


def seek_time_s(value: float) -> float:
    """The requested time both the reload and the seek path use.

    The reload path passes the time through a `%.9f` query parameter, so the page sees a
    rounded value; the seek path passes a double straight through. Left alone the two
    strategies can land on different frames either side of a boundary, and Python's own
    pose hashing would disagree with whichever one ran. Normalising here makes all three
    agree by construction rather than by luck.
    """
    return float(f"{float(value):.9f}")


def _canonical_vector(value: Any, length: int) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= length:
        return [float(component) for component in value[:length]]
    if isinstance(value, dict):
        keys = ("x", "y", "z", "w")[:length]
        if all(isinstance(value.get(key), (int, float)) for key in keys):
            return [float(value[key]) for key in keys]
    return None


def pose_sha256(frame: dict[str, Any] | None) -> str:
    """Hash the pose applied at a frame: bone rotations, offsets, and object placement.

    Machine independent by construction -- it reads the compiled clip, never the render
    -- so it is the hash a determinism test should assert across machines. The pixel
    hash cannot be: MSAA resolve and shadow filtering are GPU and driver dependent.
    """
    if frame is None:
        return hashlib.sha256(b"no-frame").hexdigest()
    bones: dict[str, dict[str, list[float]]] = {}
    for name, pose in sorted((frame.get("bones") or {}).items()):
        entry: dict[str, list[float]] = {}
        source = pose if isinstance(pose, dict) else {"rotation": pose}
        rotation = _canonical_vector(source.get("rotation"), 4)
        position = _canonical_vector(source.get("position", source.get("translation")), 3)
        if rotation is not None:
            entry["rotation"] = rotation
        if position is not None:
            entry["position"] = position
        if entry:
            bones[str(name)] = entry
    objects: dict[str, dict[str, list[float]]] = {}
    for name, pose in sorted((frame.get("objects") or {}).items()):
        if not isinstance(pose, dict):
            continue
        entry = {}
        for key, length in (("position", 3), ("translation", 3), ("rotation", 4), ("scale", 3)):
            vector = _canonical_vector(pose.get(key), length)
            if vector is not None:
                entry["position" if key == "translation" else key] = vector
        if entry:
            objects[str(name)] = entry
    document = {
        "schema": "rigby.pose.v1",
        "time_s": float(frame.get("time_s", frame.get("time", 0.0))),
        "bones": bones,
        "objects": objects,
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CapturePage(Protocol):
    """The browser operations capture needs, so the manifest path is testable without one."""

    def open(self, url: str) -> dict[str, Any]:
        """Navigate and return the page's capture state once it settles."""

    def seek(self, time_s: float, view: str) -> dict[str, Any]:
        """Re-pose and re-aim the live page; return its new capture state."""

    def screenshot(self) -> bytes:
        """PNG bytes of the capture canvas alone."""

    def recycle(self) -> None:
        """Discard and rebuild the underlying page after a transient failure."""

    def supports_seek(self) -> bool:
        """Whether the loaded page exposes the seek hook."""


_CANVAS_PNG_SCRIPT = """
() => {
  const canvas = document.querySelector("#capture-app canvas");
  if (!canvas) throw new Error("capture canvas is not present");
  return canvas.toDataURL("image/png");
}
"""

_SEEK_SCRIPT = """
([timeS, view, timeoutMs]) => {
  if (typeof window.__RIGBY_SEEK__ !== "function") {
    throw new Error("capture page does not expose __RIGBY_SEEK__");
  }
  return Promise.race([
    window.__RIGBY_SEEK__(timeS, view),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("seek timed out")), timeoutMs)
    ),
  ]);
}
"""


class _PlaywrightPage:
    """`CapturePage` over a real browser tab."""

    def __init__(self, context: Any, *, screenshot_source: str = "canvas") -> None:
        if screenshot_source not in {"canvas", "compositor"}:
            raise ValueError("screenshot source must be canvas or compositor")
        self._context = context
        self._screenshot_source = screenshot_source
        self._page: Any = None

    def _live(self) -> Any:
        if self._page is None:
            self._page = self._context.new_page()
            self._page.set_default_timeout(NAVIGATION_TIMEOUT_MS)
        return self._page

    def open(self, url: str) -> dict[str, Any]:
        page = self._live()
        # The explicit status marker is the authoritative render barrier; network-idle
        # is both slower and vulnerable to unrelated browser background work. Waiting
        # for "error" as well as "ready" is what turns a failed asset load into an
        # immediate, described failure instead of three silent selector timeouts.
        page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
        page.wait_for_selector(
            'body[data-capture-status="ready"], body[data-capture-status="error"]',
            timeout=READY_TIMEOUT_MS,
        )
        state = page.evaluate("window.__RIGBY_CAPTURE__")
        return state if isinstance(state, dict) else {"status": "error", "error": repr(state)}

    def seek(self, time_s: float, view: str) -> dict[str, Any]:
        state = self._live().evaluate(_SEEK_SCRIPT, [float(time_s), view, SEEK_TIMEOUT_MS])
        return state if isinstance(state, dict) else {"status": "error", "error": repr(state)}

    def screenshot(self) -> bytes:
        """Read the rendered canvas.

        Reading `canvas.toDataURL` is 5.3x faster than a Playwright element screenshot
        (20.0 ms against 105.7 ms, measured over 12 snapshots) and the two are
        pixel-identical -- 0 differing pixels of 1,440,000, pinned by
        `test_capture_screenshot_sources_agree`. It is also the more literal reading of
        the `raw_canvas_only` capture contract: the bytes come from the WebGL drawing
        buffer rather than through the browser compositor.
        """
        if self._screenshot_source == "compositor":
            return self._live().locator("#capture-app canvas").screenshot(
                type="png",
                animations="disabled",
                scale="css",
                timeout=SCREENSHOT_TIMEOUT_MS,
            )
        data_url = self._live().evaluate(_CANVAS_PNG_SCRIPT)
        if not isinstance(data_url, str) or not data_url.startswith("data:image/png;base64,"):
            raise RuntimeError("capture canvas did not yield a PNG data URL")
        return base64.b64decode(data_url.split(",", 1)[1])

    def recycle(self) -> None:
        page, self._page = self._page, None
        if page is not None:
            # A page that will not close is already gone; the browser teardown reaps it.
            with suppress(Exception):
                page.close()

    def supports_seek(self) -> bool:
        return bool(self._live().evaluate("typeof window.__RIGBY_SEEK__ === 'function'"))


class CaptureSession:
    """Capture evidence for one or more results through a single browser.

    A session opens the capture page once per result and seeks within it. The previous
    implementation issued a fresh `page.goto` for every sample point and view, which
    re-fetched the result JSON and re-parsed the humanoid GLB 30 times per candidate.
    """

    def __init__(
        self,
        page: CapturePage,
        *,
        base_url: str,
        browser_version: str = "",
        browser_channel: str = "",
        deterministic_render: bool = False,
        strategy: str = "seek",
        screenshot_source: str = "canvas",
        expected_asset_sha256: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if strategy not in {"seek", "reload"}:
            raise ValueError("capture strategy must be seek or reload")
        self._screenshot_source = screenshot_source
        self._page = page
        self._clock = clock
        self._base_url = base_url.rstrip("/")
        self._browser_version = browser_version
        self._browser_channel = browser_channel
        self._deterministic_render = deterministic_render
        self._strategy = strategy
        self._expected_asset_sha256 = (
            humanoid_asset_sha256() if expected_asset_sha256 is None else expected_asset_sha256
        )

    def _url(self, result_id: str, time_s: float, view: str) -> str:
        query = {"result": result_id, "time": f"{seek_time_s(time_s):.9f}", "view": view}
        if self._deterministic_render:
            query["render"] = "deterministic"
        return f"{self._base_url}/capture.html?{urlencode(query)}"

    def _check_state(self, state: dict[str, Any], *, result_id: str) -> dict[str, Any]:
        """Reject any state that is not a verified render of the compiled humanoid."""
        if state.get("status") != "ready":
            raise CaptureAssetError(
                f"capture page for {result_id} reported "
                f"{state.get('status')!r}: {state.get('error')!r}"
            )
        provenance = state.get("render_provenance")
        if not isinstance(provenance, dict):
            raise CaptureAssetError(
                f"capture page for {result_id} reported no render provenance; "
                "the frontend bundle predates plan 05 and cannot attest what it rendered"
            )
        if provenance.get("asset_status") != "loaded":
            raise CaptureAssetError(
                f"capture page for {result_id} rendered asset_status="
                f"{provenance.get('asset_status')!r}: {provenance.get('asset_error')!r}"
            )
        rendered = provenance.get("asset_sha256")
        if rendered != self._expected_asset_sha256:
            raise CaptureAssetError(
                f"capture page for {result_id} rendered humanoid sha256 {rendered!r}, "
                f"but assets/manifest.json declares {self._expected_asset_sha256!r}; "
                "the evidence would depict a different body than the clip was compiled against"
            )
        return provenance

    def capture(
        self,
        result_id: str,
        output_dir: Path,
        *,
        views: Iterable[str] = ("ego", "orbit"),
    ) -> Path:
        if not SAFE_RESULT_ID.fullmatch(result_id):
            raise ValueError("result id contains unsupported characters")
        selected_views = tuple(views)
        if not selected_views or any(view not in {"ego", "orbit"} for view in selected_views):
            raise ValueError("views must contain ego and/or orbit")

        payload = _result_payload(self._base_url, result_id)
        program = payload.get("program") if isinstance(payload.get("program"), dict) else {}
        prompt = str(program.get("source_text", ""))
        clip = _clip_document(payload)
        duration_s = _clip_duration_s(clip)
        points = phase_sampling_points(payload)
        if len(points) > MAX_SNAPSHOTS_PER_VIEW:
            raise RuntimeError(
                f"{result_id} samples {len(points)} phase points per view, above the "
                f"{MAX_SNAPSHOTS_PER_VIEW} the capture timeout budget is derived from"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        snapshots: list[dict[str, Any]] = []
        provenance: dict[str, Any] = {}
        opened = False
        deadline = self._clock() + capture_deadline_s(
            snapshots_per_result=len(points) * len(selected_views)
        )
        for point_index, point in enumerate(points, start=1):
            for view in selected_views:
                if self._clock() > deadline:
                    raise CaptureDeadlineExceeded(
                        f"{result_id}: capture exceeded its budget at snapshot "
                        f"{point_index}-{view} of {len(points)}x{len(selected_views)}"
                    )
                time_s = seek_time_s(point["time_s"])
                state, snapshot_provenance, image = self._snapshot(
                    result_id, time_s, view, opened=opened and self._strategy == "seek"
                )
                opened = True
                provenance = provenance or snapshot_provenance
                width, height = _png_size(image)
                if (width, height) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
                    raise RuntimeError(
                        f"raw capture was {width}x{height}; expected {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}"
                    )
                rendered_time_s = float(state.get("rendered_time_s", time_s))
                frame = self._verified_frame(clip, time_s, rendered_time_s, result_id)
                # `requested > rendered` is true for almost every ordinary sample, because
                # frame selection snaps backward to the last frame at or before the
                # request. So a consumer cannot infer a clamp from the two times alone --
                # the obvious heuristic is wrong in the common case and right in the rare
                # one. Recording it as a fact is the only way to tell a genuine last frame
                # from a clamped one.
                clamped = time_s > duration_s + 1e-9
                # A bare filename, never a path: the manifest is read on both macOS and
                # Windows and `judge.py` resolves it against the manifest's directory.
                filename = f"{point_index:02d}-{point['label']}-{view}.png"
                (output_dir / filename).write_bytes(image)
                pixel_digest = hashlib.sha256(image).hexdigest()
                camera = state.get("camera") if isinstance(state.get("camera"), dict) else {}
                snapshots.append(
                    {
                        "id": f"{point_index:02d}-{view}",
                        "phase": point["phase"],
                        "label": point["label"],
                        "view": view,
                        "requested_time_s": time_s,
                        "rendered_time_s": rendered_time_s,
                        "clamped": clamped,
                        "path": filename,
                        # `sha256` is retained under its original name because
                        # `judge.py` verifies snapshots against it. `pixel_sha256` is
                        # the same value under the name that says what it actually is,
                        # and what it is not: reproducible across machines.
                        "sha256": pixel_digest,
                        "pixel_sha256": pixel_digest,
                        "pose_sha256": pose_sha256(frame),
                        "width_px": width,
                        "height_px": height,
                        "camera": camera,
                    }
                )

        manifest = {
            "schema_version": "1.1",
            "result_id": result_id,
            "prompt": prompt,
            "intent": program.get("intent"),
            # The clip's own span, so a consumer can interpret `clamped` and the sample
            # times without refetching the source payload. Deliberately NOT `fps`: frame
            # spacing is piecewise-uniform, not uniform -- constant within a phase, at a
            # per-phase rate at or above 1/fps, with a 2.0x-2.5x gap at every handover
            # (measured by lane `analysis` across 33 cases, reproduced here on 7 of 8).
            # A single clip-level rate would be right often enough to look trustworthy and
            # wrong by up to 8% within a phase and 150% across a handover, with nothing
            # able to detect the error.
            "clip_duration_s": duration_s,
            "source_url": f"{self._base_url}/api/v1/results/{quote(result_id)}",
            "capture_contract": {
                "raw_canvas_only": True,
                "width_px": CAPTURE_WIDTH,
                "height_px": CAPTURE_HEIGHT,
                "aspect_ratio": CAPTURE_WIDTH / CAPTURE_HEIGHT,
                "egocentric_vertical_fov_deg": CAPTURE_FOV_DEG,
                "device_pixel_ratio": 1,
                "phase_aligned": True,
                "ui_overlay_included": False,
            },
            "render_provenance": self._manifest_provenance(provenance),
            "motion_diagnostics": motion_diagnostics(payload),
            "snapshots": snapshots,
        }
        manifest_path = output_dir / "evidence-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest_path

    def _snapshot(
        self, result_id: str, time_s: float, view: str, *, opened: bool
    ) -> tuple[dict[str, Any], dict[str, Any], bytes]:
        live = opened
        for attempt in range(1, CAPTURE_ATTEMPTS + 1):
            try:
                if live:
                    state = self._page.seek(seek_time_s(time_s), view)
                else:
                    state = self._page.open(self._url(result_id, time_s, view))
                    if self._strategy == "seek" and not self._page.supports_seek():
                        raise RuntimeError(
                            "capture page does not expose __RIGBY_SEEK__; rebuild frontend/dist"
                        )
                provenance = self._check_state(state, result_id=result_id)
                return state, provenance, self._page.screenshot()
            except CaptureAssetError:
                # Never retried. A missing or mismatched humanoid is not transient, and
                # retrying it only burns the timeout budget producing the same wrong body.
                raise
            except (PlaywrightTimeoutError, RuntimeError):
                if attempt == CAPTURE_ATTEMPTS:
                    raise
                self._page.recycle()
                live = False
        raise RuntimeError("capture retry loop produced no image")

    def _verified_frame(
        self, clip: dict[str, Any], time_s: float, rendered_time_s: float, result_id: str
    ) -> dict[str, Any] | None:
        frame = frame_at(clip, time_s)
        selected = float(frame.get("time_s", frame.get("time", 0.0))) if frame else None
        if selected is None or not math.isclose(selected, rendered_time_s, abs_tol=1e-9):
            raise RuntimeError(
                f"{result_id}: the page rendered t={rendered_time_s} but pose hashing "
                f"selected t={selected} for requested t={time_s}; the pose hash would "
                "describe a different frame than the pixels"
            )
        return frame

    def _manifest_provenance(self, provenance: dict[str, Any]) -> dict[str, Any]:
        return {
            **provenance,
            "browser_version": self._browser_version,
            "browser_channel": self._browser_channel,
            "expected_asset_sha256": self._expected_asset_sha256,
            "capture_strategy": self._strategy,
            "screenshot_source": self._screenshot_source,
            "platform": sys.platform,
        }


@contextmanager
def capture_session(
    *,
    base_url: str = "http://127.0.0.1:8000",
    deterministic_render: bool = False,
    strategy: str = "seek",
    screenshot_source: str = "canvas",
) -> Iterator[CaptureSession]:
    """One browser for many results.

    The subprocess this runs in is not for isolation: the flywheel runs on a daemon
    thread inside the FastAPI process, and Playwright's sync API refuses to run on a
    thread with a live asyncio event loop. Batching results into one session amortises
    the browser launch that workaround costs.
    """
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(
            viewport={"width": CAPTURE_WIDTH, "height": CAPTURE_HEIGHT},
            device_scale_factor=1,
        )
        page = _PlaywrightPage(context, screenshot_source=screenshot_source)
        try:
            yield CaptureSession(
                page,
                base_url=base_url,
                browser_version=str(getattr(browser, "version", "")),
                browser_channel="chrome",
                deterministic_render=deterministic_render,
                strategy=strategy,
                screenshot_source=screenshot_source,
            )
        finally:
            page.recycle()
            context.close()
            browser.close()


def capture_result_frames(
    result_id: str,
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    views: Iterable[str] = ("ego", "orbit"),
    deterministic_render: bool = False,
    strategy: str = "seek",
    screenshot_source: str = "canvas",
) -> Path:
    """Capture one result. Retained for every existing call site."""
    with capture_session(
        base_url=base_url,
        deterministic_render=deterministic_render,
        strategy=strategy,
        screenshot_source=screenshot_source,
    ) as session:
        return session.capture(result_id, output_dir, views=views)


def capture_results(
    requests: Sequence[tuple[str, Path]],
    *,
    base_url: str = "http://127.0.0.1:8000",
    views: Iterable[str] = ("ego", "orbit"),
    deterministic_render: bool = False,
    strategy: str = "seek",
    screenshot_source: str = "canvas",
) -> list[Path]:
    """Capture many results through one browser launch."""
    selected = tuple(views)
    with capture_session(
        base_url=base_url,
        deterministic_render=deterministic_render,
        strategy=strategy,
        screenshot_source=screenshot_source,
    ) as session:
        return [
            session.capture(result_id, output_dir, views=selected)
            for result_id, output_dir in requests
        ]


def _batch_from_file(path: Path) -> tuple[str, list[tuple[str, Path]], tuple[str, ...]]:
    """Read a batch descriptor.

    A file rather than repeated argv pairs: result ids and output directories are
    user-derived and Windows command lines have both length and quoting limits that a
    JSON file sidesteps entirely.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("capture batch descriptor must be an object")
    requests = document.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("capture batch descriptor lists no requests")
    views = document.get("views")
    selected = tuple(views) if isinstance(views, list) and views else ("ego", "orbit")
    return (
        str(document.get("base_url", "http://127.0.0.1:8000")),
        [(str(item["result_id"]), Path(item["output_dir"])) for item in requests],
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture raw full-FOV Rigby evidence frames")
    parser.add_argument("result_id", nargs="?")
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("--batch", type=Path, help="JSON descriptor for a multi-result capture")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--view", action="append", choices=("ego", "orbit"), dest="views")
    parser.add_argument(
        "--deterministic-render",
        action="store_true",
        help="antialiasing off and hard shadows, for renders compared numerically",
    )
    parser.add_argument(
        "--strategy",
        choices=("seek", "reload"),
        default="seek",
        help="seek within one loaded page, or reload the page per snapshot",
    )
    parser.add_argument(
        "--screenshot-source",
        choices=("canvas", "compositor"),
        default="canvas",
        help="read the WebGL drawing buffer, or go through the browser compositor",
    )
    arguments = parser.parse_args()

    if arguments.batch is not None:
        base_url, requests, batch_views = _batch_from_file(arguments.batch)
        paths = capture_results(
            requests,
            base_url=base_url,
            views=arguments.views or batch_views,
            deterministic_render=arguments.deterministic_render,
            strategy=arguments.strategy,
            screenshot_source=arguments.screenshot_source,
        )
        for path in paths:
            print(path)
        return

    if not arguments.result_id or arguments.output_dir is None:
        parser.error("result_id and output_dir are required without --batch")
    path = capture_result_frames(
        arguments.result_id,
        arguments.output_dir,
        base_url=arguments.base_url,
        views=arguments.views or ("ego", "orbit"),
        deterministic_render=arguments.deterministic_render,
        strategy=arguments.strategy,
        screenshot_source=arguments.screenshot_source,
    )
    print(path)


if __name__ == "__main__":
    main()
