"""Horizontal support poses: planks, supine and prone lies, quadruped stances.

Returns the contact-point count as well as writing metrics, because the
structural-failure pass needs it and the compiler kept it as a local.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...models import BodyAction, BodySupportMode, Hand
from .selectors import FullBodyPass


def posture_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> int:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    program = fb.program
    ground_height = fb.ground_height
    horizontal_pose_requested = fb.horizontal_pose_requested
    horizontal_pose_labels = fb.horizontal_pose_labels

    metrics["horizontal_pose_requested"] = horizontal_pose_requested
    horizontal_support_bones = (
        "hips",
        "chest",
        "upperChest",
        "head",
        "leftUpperArm",
        "leftLowerArm",
        "leftHand",
        "rightUpperArm",
        "rightLowerArm",
        "rightHand",
        "leftUpperLeg",
        "leftLowerLeg",
        "rightUpperLeg",
        "rightLowerLeg",
    )
    horizontal_body_clearances = np.asarray(
        [
            min(
                float(position[name][1]) - ground_height
                for name in horizontal_support_bones
            )
            for position in world_positions
        ],
        dtype=float,
    )
    metrics["minimum_nonfoot_body_clearance_m"] = float(
        np.min(horizontal_body_clearances)
    )
    horizontal_contact_count = 0
    if horizontal_pose_requested:
        horizontal_ranges = [
            item
            for item in phase_ranges
            if item.get("action") == BodyAction.POSE.value
            and item.get("label") in horizontal_pose_labels
        ]
        decisive_time = (
            float(horizontal_ranges[-1]["start_s"])
            + 0.88
            * (
                float(horizontal_ranges[-1]["end_s"])
                - float(horizontal_ranges[-1]["start_s"])
            )
            if horizontal_ranges
            else frames[len(frames) // 2].time_s
        )
        decisive_index = min(
            range(len(frames)),
            key=lambda index: abs(frames[index].time_s - decisive_time),
        )
        decisive_positions = world_positions[decisive_index]
        horizontal_contact_count = sum(
            float(decisive_positions[name][1]) - ground_height <= 0.14
            for name in horizontal_support_bones
        )
        horizontal_minimum = min(
            float(decisive_positions[name][1]) - ground_height
            for name in horizontal_support_bones
        )
        metrics["horizontal_pose_minimum_clearance_m"] = horizontal_minimum
        metrics["horizontal_pose_contact_point_count"] = horizontal_contact_count
        horizontal_target = next(
            primitive.body.pose
            for primitive in program.primitives
            if primitive.body is not None
            and primitive.body.action == BodyAction.POSE
            and (
                primitive.body.pose.support_mode != BodySupportMode.FEET
                or (
                    not primitive.body.pose.lock_feet
                    and abs(primitive.body.pose.pelvis_pitch_deg) >= 80.0
                )
            )
        )
        body_axis = decisive_positions["head"] - decisive_positions["hips"]
        metrics["horizontal_body_axis_vertical_fraction"] = abs(
            float(body_axis[1])
        ) / max(float(np.linalg.norm(body_axis)), 1e-8)
        metrics["horizontal_pose_pelvis_pitch_deg"] = float(
            horizontal_target.pelvis_pitch_deg
        )
        metrics["horizontal_pose_variant"] = (
            "quadruped"
            if horizontal_target.support_mode == BodySupportMode.QUADRUPED
            else "plank"
            if horizontal_target.support_mode == BodySupportMode.PLANK
            else "side_left"
            if abs(horizontal_target.pelvis_pitch_deg) < 45.0
            and horizontal_target.pelvis_roll_deg > 0.0
            else "side_right"
            if abs(horizontal_target.pelvis_pitch_deg) < 45.0
            and horizontal_target.pelvis_roll_deg < 0.0
            else "prone"
            if horizontal_target.pelvis_pitch_deg > 0.0
            else "supine"
        )
        if horizontal_target.support_mode == BodySupportMode.PLANK:
            metrics["push_up_toe_support_clearance_m"] = max(
                float(decisive_positions["leftToes"][1]) - ground_height,
                float(decisive_positions["rightToes"][1]) - ground_height,
            )
            knee_angles: list[float] = []
            for side in (Hand.LEFT.value, Hand.RIGHT.value):
                hip = decisive_positions[f"{side}UpperLeg"]
                knee = decisive_positions[f"{side}LowerLeg"]
                ankle = decisive_positions[f"{side}Foot"]
                thigh = hip - knee
                shin = ankle - knee
                knee_angles.append(
                    math.degrees(
                        math.acos(
                            float(
                                np.clip(
                                    np.dot(thigh, shin)
                                    / max(
                                        np.linalg.norm(thigh)
                                        * np.linalg.norm(shin),
                                        1e-8,
                                    ),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                )
            metrics["push_up_min_knee_extension_deg"] = min(knee_angles)
    return horizontal_contact_count


__all__ = ["posture_metrics"]
