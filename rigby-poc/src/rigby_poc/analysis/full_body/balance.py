"""Support-constraint tracking, angular kinematics and the terminal balance test.

``max_support_foot_target_error_m``, ``max_support_foot_slide_per_frame_m`` and
``support_contact_fraction`` all compare the achieved ankle against the ankle
the IK solver was *commanded* to hit, so they depend on the support constraints
the compiler now persists.

The bone list fed to the angular pass is a property of the compile path, not of
the action: whole-body uses these ten including both feet, sequence swaps the
feet for the forearms, object interaction uses three bones of the active arm.
An analyzer that assumes one list disagrees on three paths.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...models import Hand
from ..gesture import _angular_kinematics
from .selectors import FullBodyPass


def support_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    world_positions = fb.world_positions
    toe_clearances = fb.toe_clearances
    support_constraints = fb.support_constraints

    support_errors: list[float] = []
    support_slides: list[float] = []
    support_contacts = 0
    previous_constraint: dict[str, Any] | None = None
    previous_actual: np.ndarray | None = None
    for constraint in support_constraints:
        frame_index = int(constraint["frame_index"])
        side = str(constraint["side"])
        actual = world_positions[frame_index][f"{side}Foot"]
        target_position = np.asarray(constraint["ankle_target"], dtype=float)
        support_errors.append(float(np.linalg.norm(actual - target_position)))
        support_contacts += int(toe_clearances[side][frame_index] <= 0.025)
        if (
            previous_constraint is not None
            and previous_actual is not None
            and previous_constraint["side"] == side
            and np.linalg.norm(
                np.asarray(previous_constraint["ankle_target"], dtype=float)
                - target_position
            )
            <= 1e-8
        ):
            support_slides.append(
                float(np.linalg.norm((actual - previous_actual)[[0, 2]]))
            )
        previous_constraint = constraint
        previous_actual = actual
    metrics["max_support_foot_target_error_m"] = max(support_errors, default=0.0)
    metrics["max_support_foot_slide_per_frame_m"] = max(support_slides, default=0.0)
    metrics["support_contact_fraction"] = (
        support_contacts / len(support_constraints) if support_constraints else 1.0
    )


def angular_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    program = fb.program

    body_bones = [
        "hips",
        "chest",
        "leftUpperLeg",
        "leftLowerLeg",
        "leftFoot",
        "rightUpperLeg",
        "rightLowerLeg",
        "rightFoot",
        "leftUpperArm",
        "rightUpperArm",
    ]
    angular_velocity, angular_acceleration, angular_jerk = _angular_kinematics(
        frames,
        body_bones,
    )
    metrics["max_angular_velocity_rad_s"] = angular_velocity
    metrics["max_angular_acceleration_rad_s2"] = angular_acceleration
    metrics["max_angular_jerk_rad_s3"] = angular_jerk
    metrics["requested_body_cycles"] = max(
        (
            primitive.body.cycles
            for primitive in program.primitives
            if primitive.body is not None
        ),
        default=0.0,
    )


def final_balance_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    world_positions = fb.world_positions
    hips_positions = fb.hips_positions
    toe_clearances = fb.toe_clearances

    final_leg_error = max(
        2.0
        * math.acos(
            float(
                np.clip(
                    abs(frames[-1].bones[name].rotation.w),
                    0.0,
                    1.0,
                )
            )
        )
        for name in (
            "leftUpperLeg",
            "leftLowerLeg",
            "leftFoot",
            "rightUpperLeg",
            "rightLowerLeg",
            "rightFoot",
        )
    )
    metrics["final_balanced_leg_error_rad"] = final_leg_error
    final_root_horizontal = hips_positions[-1, [0, 2]]
    final_left_foot = world_positions[-1]["leftFoot"][[0, 2]]
    final_right_foot = world_positions[-1]["rightFoot"][[0, 2]]
    support_segment = final_right_foot - final_left_foot
    support_length_squared = float(np.dot(support_segment, support_segment))
    if support_length_squared <= 1e-10:
        closest_support = final_left_foot
    else:
        support_fraction = float(
            np.clip(
                np.dot(final_root_horizontal - final_left_foot, support_segment)
                / support_length_squared,
                0.0,
                1.0,
            )
        )
        closest_support = final_left_foot + support_segment * support_fraction
    metrics["final_support_center_offset_m"] = float(
        np.linalg.norm(final_root_horizontal - closest_support)
    )
    metrics["final_foot_ground_error_m"] = max(
        abs(float(toe_clearances[Hand.LEFT.value][-1])),
        abs(float(toe_clearances[Hand.RIGHT.value][-1])),
    )
    if len(frames) >= 2:
        final_delta_t = max(frames[-1].time_s - frames[-2].time_s, 1e-8)
        metrics["final_root_vertical_speed_m_s"] = abs(
            float(hips_positions[-1, 1] - hips_positions[-2, 1]) / final_delta_t
        )
    else:
        metrics["final_root_vertical_speed_m_s"] = 0.0


__all__ = ["angular_metrics", "final_balance_metrics", "support_metrics"]
