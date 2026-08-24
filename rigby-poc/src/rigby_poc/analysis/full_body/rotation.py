"""Whole-body rotation: cartwheels, floor rolls and airborne spins.

All three are ``BodyAction.ROTATE``; ``BodyRotationMode`` picks which family of
keys the block emits, so a single case cannot cover this module — the fixture
carries one cartwheel and one floor roll.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...models import BodyAction, BodyRotationMode, Hand
from .selectors import FullBodyPass


def rotation_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    hips_positions = fb.hips_positions
    ground_height = fb.ground_height
    toe_clearances = fb.toe_clearances
    rotation_primitives = fb.rotation_primitives

    if rotation_primitives:
        rotation_ranges = [
            item
            for item in phase_ranges
            if item.get("action") == BodyAction.ROTATE.value
        ]
        phase_metrics: list[dict[str, Any]] = []
        floor_contact_frames = 0
        floor_sample_frames = 0
        cartwheel_hand_contact_frames = 0
        cartwheel_sample_frames = 0
        cartwheel_inverted_frames = 0
        cartwheel_max_foot_clearance = 0.0
        cartwheel_min_head_clearance = float("inf")
        airborne_rotation_frames = 0
        airborne_rotation_samples = 0
        airborne_rotation_peak_root_height = ground_height
        nonfoot_support_bones = (
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
        )
        for primitive, phase in zip(
            rotation_primitives,
            rotation_ranges,
            strict=False,
        ):
            assert primitive.body is not None
            target = primitive.body
            indices = [
                index
                for index, frame in enumerate(frames)
                if float(phase["start_s"])
                <= frame.time_s
                <= float(phase["end_s"])
            ]
            measured_degrees = 0.0
            for previous_index, current_index in zip(indices, indices[1:]):
                previous_rotation = np.asarray(
                    frames[previous_index].bones["hips"].rotation.as_list(),
                    dtype=float,
                )
                current_rotation = np.asarray(
                    frames[current_index].bones["hips"].rotation.as_list(),
                    dtype=float,
                )
                measured_degrees += math.degrees(
                    2.0
                    * math.acos(
                        float(
                            np.clip(
                                abs(np.dot(previous_rotation, current_rotation)),
                                0.0,
                                1.0,
                            )
                        )
                    )
                )
            direction = np.asarray(
                [target.direction_x, target.direction_z],
                dtype=float,
            )
            direction_norm = float(np.linalg.norm(direction))
            projected_travel = 0.0
            if indices and direction_norm > 1e-8:
                direction /= direction_norm
                root_delta = (
                    hips_positions[indices[-1], [0, 2]]
                    - hips_positions[indices[0], [0, 2]]
                )
                projected_travel = float(np.dot(root_delta, direction))
            record: dict[str, Any] = {
                "label": primitive.label or BodyAction.ROTATE.value,
                "mode": target.rotation_mode.value,
                "axis": target.rotation_axis.value,
                "requested_degrees": abs(float(target.rotation_degrees)),
                "measured_degrees": measured_degrees,
                "requested_travel_m": float(target.distance_m),
                "projected_travel_m": projected_travel,
                "sample_frame_count": len(indices),
            }
            if target.rotation_mode == BodyRotationMode.FLOOR:
                contact_count = sum(
                    any(
                        float(world_positions[index][name][1])
                        <= ground_height + 0.045
                        for name in nonfoot_support_bones
                    )
                    for index in indices
                )
                floor_contact_frames += contact_count
                floor_sample_frames += len(indices)
                record["nonfoot_contact_frame_count"] = contact_count
            elif target.rotation_mode == BodyRotationMode.CARTWHEEL:
                hand_contact_count = sum(
                    min(
                        float(world_positions[index]["leftHand"][1]),
                        float(world_positions[index]["rightHand"][1]),
                    )
                    <= ground_height + 0.055
                    for index in indices
                )
                inverted_count = sum(
                    min(
                        float(world_positions[index]["leftFoot"][1]),
                        float(world_positions[index]["rightFoot"][1]),
                    )
                    >= max(
                        float(world_positions[index]["leftHand"][1]),
                        float(world_positions[index]["rightHand"][1]),
                    )
                    + 0.30
                    for index in indices
                )
                max_foot_clearance = max(
                    (
                        max(
                            float(world_positions[index]["leftFoot"][1]),
                            float(world_positions[index]["rightFoot"][1]),
                            float(world_positions[index]["leftToes"][1]),
                            float(world_positions[index]["rightToes"][1]),
                        )
                        - ground_height
                        for index in indices
                    ),
                    default=0.0,
                )
                min_head_clearance = min(
                    (
                        float(world_positions[index]["head"][1])
                        - ground_height
                        for index in indices
                    ),
                    default=float("inf"),
                )
                cartwheel_hand_contact_frames += hand_contact_count
                cartwheel_sample_frames += len(indices)
                cartwheel_inverted_frames += inverted_count
                cartwheel_max_foot_clearance = max(
                    cartwheel_max_foot_clearance,
                    max_foot_clearance,
                )
                cartwheel_min_head_clearance = min(
                    cartwheel_min_head_clearance,
                    min_head_clearance,
                )
                record["hand_contact_frame_count"] = hand_contact_count
                record["inverted_frame_count"] = inverted_count
                record["max_foot_clearance_m"] = max_foot_clearance
                record["minimum_head_clearance_m"] = min_head_clearance
            elif target.rotation_mode == BodyRotationMode.AIRBORNE:
                airborne_count = int(sum(
                    toe_clearances[Hand.LEFT.value][index] > 0.028
                    and toe_clearances[Hand.RIGHT.value][index] > 0.028
                    for index in indices
                ))
                peak_height = max(
                    (
                        float(world_positions[index]["hips"][1])
                        for index in indices
                    ),
                    default=ground_height,
                )
                airborne_rotation_frames += airborne_count
                airborne_rotation_samples += len(indices)
                airborne_rotation_peak_root_height = max(
                    airborne_rotation_peak_root_height,
                    peak_height,
                )
                record["airborne_frame_count"] = airborne_count
                record["peak_root_height_m"] = peak_height
            phase_metrics.append(record)
        metrics["body_rotation_phase_metrics"] = phase_metrics
        metrics["requested_body_rotation_degrees"] = sum(
            float(item["requested_degrees"]) for item in phase_metrics
        )
        metrics["measured_body_rotation_degrees"] = sum(
            float(item["measured_degrees"]) for item in phase_metrics
        )
        metrics["minimum_body_rotation_completion_fraction"] = min(
            (
                float(item["measured_degrees"])
                / max(float(item["requested_degrees"]), 1e-8)
                for item in phase_metrics
            ),
            default=0.0,
        )
        metrics["minimum_rotation_travel_completion_fraction"] = min(
            (
                float(item["projected_travel_m"])
                / max(float(item["requested_travel_m"]), 1e-8)
                for item in phase_metrics
                if float(item["requested_travel_m"]) > 1e-8
            ),
            default=1.0,
        )
        metrics["floor_roll_nonfoot_contact_frame_count"] = floor_contact_frames
        metrics["floor_roll_nonfoot_contact_fraction"] = (
            float(floor_contact_frames / floor_sample_frames)
            if floor_sample_frames
            else 0.0
        )
        metrics["cartwheel_hand_contact_frame_count"] = (
            cartwheel_hand_contact_frames
        )
        metrics["cartwheel_hand_contact_fraction"] = (
            float(cartwheel_hand_contact_frames / cartwheel_sample_frames)
            if cartwheel_sample_frames
            else 0.0
        )
        metrics["cartwheel_inverted_frame_count"] = cartwheel_inverted_frames
        metrics["cartwheel_max_foot_clearance_m"] = cartwheel_max_foot_clearance
        metrics["cartwheel_minimum_head_clearance_m"] = (
            cartwheel_min_head_clearance
            if math.isfinite(cartwheel_min_head_clearance)
            else 0.0
        )
        metrics["airborne_rotation_airborne_frame_count"] = (
            airborne_rotation_frames
        )
        metrics["airborne_rotation_airborne_fraction"] = (
            float(airborne_rotation_frames / airborne_rotation_samples)
            if airborne_rotation_samples
            else 0.0
        )
        metrics["airborne_rotation_peak_root_height_m"] = (
            airborne_rotation_peak_root_height
        )


__all__ = ["rotation_metrics"]
