"""The full-body structural checks.

A flat sequence of ``if metric > threshold: failures.append(...)`` reading keys
set earlier in the same pass — never a loop variable, never a frame. That is
what makes structural validation a pure ``(metrics, program) -> list[str]`` once
the metrics have moved.
"""

from __future__ import annotations

from typing import Any

from ...models import BodyAction, BodyObstacleMode, BodyRotationMode
from ..semantic import semantic_cycle_failures as _semantic_cycle_failures
from .selectors import FullBodyPass
from ..safety import clip_contract_violations


def full_body_failures(fb: FullBodyPass, metrics: dict[str, Any], horizontal_contact_count: int) -> list[str]:
    program = fb.program
    terminal_climb = fb.terminal_climb
    horizontal_pose_requested = fb.horizontal_pose_requested
    burpee_jump_primitives = fb.burpee_jump_primitives
    climb_primitives = fb.climb_primitives
    crawl_primitives = fb.crawl_primitives
    dance_primitives = fb.dance_primitives
    jumping_jack_primitives = fb.jumping_jack_primitives
    lunge_primitives = fb.lunge_primitives
    obstacle_primitives = fb.obstacle_primitives
    rotation_primitives = fb.rotation_primitives
    single_leg_primitives = fb.single_leg_primitives
    sit_up_primitives = fb.sit_up_primitives
    squat_primitives = fb.squat_primitives

    structural_failures: list[str] = []
    if metrics["nan_count"]:
        structural_failures.append("clip contains non-finite transforms")
    if clip_contract_violations(metrics, allow_root_motion=True):
        structural_failures.append("clip exceeds a joint limit")
    if metrics["discontinuities"]:
        structural_failures.append(
            f"clip contains {metrics['discontinuities']} rotational discontinuities"
        )
    if float(metrics["final_root_vertical_speed_m_s"]) > 0.04:
        structural_failures.append("root did not settle vertically after recovery")
    if not terminal_climb and float(metrics["final_support_center_offset_m"]) > 0.14:
        structural_failures.append("root finished outside the balanced support region")
    if not terminal_climb and float(metrics["final_foot_ground_error_m"]) > 0.025:
        structural_failures.append("a recovery foot did not finish on the ground")
    if float(metrics["ground_penetration_m"]) > 0.012:
        structural_failures.append(
            f"foot penetrates ground by {metrics['ground_penetration_m']:.3f} m"
        )
    if float(metrics["max_support_foot_target_error_m"]) > 0.012:
        structural_failures.append(
            "support foot failed its planted world-space target"
        )
    if float(metrics["max_support_foot_slide_per_frame_m"]) > 0.006:
        structural_failures.append(
            "support foot slides during a planted stance interval"
        )
    structural_failures.extend(_semantic_cycle_failures(program, metrics))
    allows_airborne = horizontal_pose_requested or any(
        primitive.body is not None
        and primitive.body.action in {
            BodyAction.JUMP,
            BodyAction.RUN,
            BodyAction.CLIMB,
            BodyAction.ROTATE,
        }
        for primitive in program.primitives
    )
    if not allows_airborne and int(metrics["airborne_frame_count"]) > 0:
        structural_failures.append("non-jumping motion loses both ground contacts")
    if dance_primitives:
        requested_dance_beats = int(metrics.get("requested_dance_beats", 0))
        if int(metrics.get("measured_dance_beats", 0)) != requested_dance_beats:
            structural_failures.append(
                "dance foot-lift count does not match the requested beats"
            )
        if int(metrics.get("dance_alternating_lift_count", 0)) != requested_dance_beats:
            structural_failures.append(
                "dance foot lifts do not alternate between left and right"
            )
        required_lateral_range = 0.035 if requested_dance_beats == 1 else 0.070
        if float(metrics.get("dance_lateral_root_range_m", 0.0)) < required_lateral_range:
            structural_failures.append(
                "dance does not visibly transfer weight between beats"
            )
        if float(metrics.get("dance_left_foot_peak_clearance_m", 0.0)) < 0.050:
            structural_failures.append("dance does not visibly lift the left foot")
        if requested_dance_beats >= 2 and float(
            metrics.get("dance_right_foot_peak_clearance_m", 0.0)
        ) < 0.050:
            structural_failures.append("dance does not visibly lift the right foot")
    if climb_primitives:
        if int(metrics.get("climb_missing_support_object_count", 0)) > 0:
            structural_failures.append(
                "climb references a missing scene support object"
            )
        if float(metrics.get("climb_vertical_completion_fraction", 0.0)) < 0.90:
            structural_failures.append(
                "climb does not complete the requested vertical travel"
            )
        if float(metrics.get("climb_vertical_completion_fraction", 0.0)) > 1.08:
            structural_failures.append(
                "climb overshoots the requested vertical travel"
            )
        if float(metrics.get("climb_support_target_max_error_m", 999.0)) > 0.16:
            structural_failures.append(
                "climbing limbs do not reach the support affordance"
            )
        if float(metrics.get("climb_three_point_support_fraction", 0.0)) < 0.70:
            structural_failures.append(
                "climb does not preserve three-point support"
            )
        if int(metrics.get("climb_final_supported_limb_count", 0)) < 3:
            structural_failures.append(
                "climb does not finish with stable ladder contacts"
            )
    if rotation_primitives:
        if len(metrics.get("body_rotation_phase_metrics", [])) != len(
            rotation_primitives
        ):
            structural_failures.append(
                "rotation phases could not be matched to their authored targets"
            )
        completion_fraction = float(
            metrics.get("minimum_body_rotation_completion_fraction", 0.0)
        )
        if completion_fraction < 0.94:
            structural_failures.append(
                "whole-body rotation does not complete the requested angle"
            )
        if completion_fraction > 1.08:
            structural_failures.append(
                "whole-body rotation overshoots the requested angle"
            )
        if float(
            metrics.get("minimum_rotation_travel_completion_fraction", 0.0)
        ) < 0.90:
            structural_failures.append(
                "whole-body rotation does not complete its requested travel"
            )
        floor_rotations = [
            primitive
            for primitive in rotation_primitives
            if primitive.body is not None
            and primitive.body.rotation_mode == BodyRotationMode.FLOOR
        ]
        if floor_rotations and float(
            metrics.get("floor_roll_nonfoot_contact_fraction", 0.0)
        ) < 0.20:
            structural_failures.append(
                "floor roll never transfers support from the feet to the body"
            )
        cartwheel_rotations = [
            primitive
            for primitive in rotation_primitives
            if primitive.body is not None
            and primitive.body.rotation_mode == BodyRotationMode.CARTWHEEL
        ]
        if cartwheel_rotations:
            if int(metrics.get("cartwheel_hand_contact_frame_count", 0)) < len(
                cartwheel_rotations
            ):
                structural_failures.append(
                    "cartwheel never establishes hand support"
                )
            if int(metrics.get("cartwheel_inverted_frame_count", 0)) < len(
                cartwheel_rotations
            ):
                structural_failures.append(
                    "cartwheel never carries the feet above the hands"
                )
            if float(metrics.get("cartwheel_max_foot_clearance_m", 0.0)) < 0.60:
                structural_failures.append(
                    "cartwheel does not visibly lift the feet over the body"
                )
            if float(
                metrics.get("cartwheel_minimum_head_clearance_m", 0.0)
            ) < 0.08:
                structural_failures.append(
                    "cartwheel places the head on the floor instead of the hands"
                )
        airborne_rotations = [
            primitive
            for primitive in rotation_primitives
            if primitive.body is not None
            and primitive.body.rotation_mode == BodyRotationMode.AIRBORNE
        ]
        if airborne_rotations and int(
            metrics.get("airborne_rotation_airborne_frame_count", 0)
        ) < len(airborne_rotations):
            structural_failures.append(
                "airborne rotation never leaves ground support"
            )
    if obstacle_primitives:
        if int(metrics.get("obstacle_missing_target_count", 0)) > 0:
            structural_failures.append(
                "obstacle traversal references a missing scene object"
            )
        for record in metrics.get("obstacle_traversal_phase_metrics", []):
            if not record.get("target_found"):
                continue
            if record.get("mode") == BodyObstacleMode.OVER.value:
                requested_clearance = float(record["requested_clearance_m"])
                if float(record["measured_foot_clearance_m"]) < (
                    requested_clearance - 0.012
                ):
                    structural_failures.append(
                        "swing foot does not clear the named obstacle"
                    )
                if float(record["closest_horizontal_distance_m"]) > (
                    float(record["object_horizontal_radius_m"]) + 0.14
                ):
                    structural_failures.append(
                        "step path does not pass over the named obstacle"
                    )
            elif float(record["minimum_root_clearance_m"]) < float(
                record["required_root_clearance_m"]
            ):
                structural_failures.append(
                    "detour path does not clear the named obstacle"
                )
    if jumping_jack_primitives:
        if int(metrics.get("measured_jumping_jack_cycles", 0)) != int(
            round(float(metrics.get("requested_jumping_jack_cycles", 0.0)))
        ):
            structural_failures.append(
                "jumping-jack repetition count does not match the request"
            )
        if float(metrics.get("jumping_jack_foot_spread_excursion_m", 0.0)) < 0.30:
            structural_failures.append(
                "jumping jack does not visibly spread both feet"
            )
        if float(metrics.get("jumping_jack_max_wrist_height_m", 0.0)) < 1.65:
            structural_failures.append(
                "jumping jack does not raise both hands overhead"
            )
    if burpee_jump_primitives:
        requested_burpees = int(metrics.get("requested_burpee_cycles", 0))
        if int(metrics.get("measured_burpee_jump_cycles", 0)) != requested_burpees:
            structural_failures.append(
                "burpee jump count does not match the request"
            )
        if int(metrics.get("measured_burpee_push_up_cycles", 0)) != requested_burpees:
            structural_failures.append(
                "burpee push-up count does not match the request"
            )
        if int(metrics.get("burpee_floor_support_phase_count", 0)) != requested_burpees:
            structural_failures.append(
                "burpee does not enter plank support once per repetition"
            )
        if int(metrics.get("burpee_airborne_phase_count", 0)) != requested_burpees:
            structural_failures.append(
                "burpee does not become airborne once per repetition"
            )
        if float(metrics.get("burpee_root_vertical_excursion_m", 0.0)) < 0.55:
            structural_failures.append(
                "burpee does not traverse a full floor-to-jump vertical range"
            )
        if float(metrics.get("burpee_max_wrist_height_m", 0.0)) < 1.65:
            structural_failures.append(
                "burpee does not finish its jump with raised arms"
            )
    if squat_primitives:
        requested_squats = int(metrics.get("requested_squat_cycles", 0))
        if int(metrics.get("measured_squat_cycles", 0)) != requested_squats:
            structural_failures.append(
                "squat repetition count does not match complete down-and-up cycles"
            )
        if float(metrics.get("squat_minimum_depth_m", 0.0)) < 0.10:
            structural_failures.append("squat does not reach a visible depth")
        if float(metrics.get("squat_max_stance_return_error_m", 1.0)) > 0.035:
            structural_failures.append(
                "squat repetition does not return to standing height"
            )
    if lunge_primitives:
        requested_lunges = int(metrics.get("requested_lunge_cycles", 0))
        if int(metrics.get("measured_lunge_cycles", 0)) != requested_lunges:
            structural_failures.append(
                "lunge repetition count does not match complete stagger-and-return cycles"
            )
        if float(metrics.get("lunge_minimum_root_drop_m", 0.0)) < 0.10:
            structural_failures.append("lunge does not reach a visible depth")
        if float(metrics.get("lunge_minimum_foot_stagger_m", 0.0)) < 0.12:
            structural_failures.append("lunge does not visibly stagger the feet")
        if float(metrics.get("lunge_max_stance_return_error_m", 1.0)) > 0.035:
            structural_failures.append(
                "lunge repetition does not return to standing height"
            )
    if single_leg_primitives:
        if float(metrics.get("single_leg_min_raised_foot_clearance_m", 0.0)) < 0.18:
            structural_failures.append(
                "single-leg balance does not visibly lift the free foot"
            )
        if float(metrics.get("single_leg_max_support_foot_slide_m", 1.0)) > 0.01:
            structural_failures.append(
                "single-leg balance slides its planted support foot"
            )
        if float(metrics.get("single_leg_support_contact_fraction", 0.0)) < 0.95:
            structural_failures.append(
                "single-leg balance loses planted-foot support"
            )
    if sit_up_primitives:
        requested_sit_ups = int(metrics.get("requested_sit_up_cycles", 0))
        if int(metrics.get("measured_sit_up_cycles", 0)) != requested_sit_ups:
            structural_failures.append(
                "sit-up repetition count does not match complete curl-and-return cycles"
            )
        if float(metrics.get("sit_up_minimum_head_lift_m", 0.0)) < 0.15:
            structural_failures.append(
                "sit-up does not visibly lift the head and shoulders"
            )
        if float(metrics.get("sit_up_max_supine_return_error_m", 1.0)) > 0.05:
            structural_failures.append(
                "sit-up does not return to the supine floor pose"
            )
    if horizontal_pose_requested:
        if float(metrics.get("horizontal_pose_minimum_clearance_m", 0.0)) < -0.012:
            structural_failures.append("horizontal pose penetrates the ground plane")
        required_horizontal_contacts = (
            4
            if metrics.get("horizontal_pose_variant") == "quadruped"
            else 2
        )
        if horizontal_contact_count < required_horizontal_contacts:
            structural_failures.append(
                "grounded horizontal pose does not establish its requested support set"
            )
        if float(metrics.get("horizontal_body_axis_vertical_fraction", 1.0)) > 0.35:
            structural_failures.append("requested grounded horizontal pose remains too upright")
        if metrics.get("horizontal_pose_variant") == "plank":
            requested_plank_cycles = int(
                round(float(metrics.get("requested_push_up_cycles", 0.0)))
            )
            if requested_plank_cycles > 0:
                if float(metrics.get("push_up_vertical_excursion_m", 0.0)) < 0.06:
                    structural_failures.append(
                        "push-up cycles do not create enough measured torso travel"
                    )
                if int(metrics.get("measured_push_up_cycles", 0)) != requested_plank_cycles:
                    structural_failures.append(
                        "measured push-up repetition count does not match the request"
                    )
            elif float(metrics.get("plank_hold_vertical_range_m", 1.0)) > 0.025:
                structural_failures.append(
                    "static plank does not maintain a stable torso height"
                )
            if float(metrics.get("push_up_palm_height_range_m", 0.0)) > 0.045:
                structural_failures.append(
                    "push-up palms do not remain on a stable support plane"
                )
            if float(metrics.get("push_up_toe_support_clearance_m", 1.0)) > 0.035:
                structural_failures.append(
                    "push-up toes do not remain on the support plane"
                )
            if float(metrics.get("push_up_toe_height_range_m", 1.0)) > 0.015:
                structural_failures.append(
                    "push-up toe support changes height during repetitions"
                )
            if float(metrics.get("push_up_toe_position_range_m", 1.0)) > 0.02:
                structural_failures.append(
                    "push-up toes slide during repetitions"
                )
            if float(metrics.get("push_up_min_knee_extension_deg", 0.0)) < 150.0:
                structural_failures.append(
                    "push-up legs do not retain a straight plank alignment"
                )
        if crawl_primitives:
            if float(metrics.get("crawl_root_displacement_m", 0.0)) < 0.80 * float(
                metrics.get("requested_crawl_distance_m", 0.0)
            ):
                structural_failures.append(
                    "crawl does not retain its requested root travel through recovery"
                )
            if float(metrics.get("crawl_hand_alternation_range_m", 0.0)) < 0.10:
                structural_failures.append(
                    "crawl does not show opposed alternating hand advances"
                )
    metrics["structural_failures"] = structural_failures
    return structural_failures


__all__ = ["full_body_failures"]
