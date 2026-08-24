"""The label-keyed exercise families of the full-body path.

Eight blocks, and none of them is selected by ``BodyAction`` alone. Jumping
jacks are JUMP primitives with a lateral foot shift; burpees are JUMP and POSE
primitives sharing a ``burpee_`` label prefix; squats are CROUCH with a
``squat_`` prefix; lunges, single-leg balances, sit-ups, crawls and push-ups are
all POSE, separated by label or by ``pose.support_mode``. :mod:`.selectors`
holds every one of those predicates.

Each function is the compiler's block verbatim, reading the same local names.
The one edit is that the primitive selection moved to ``FullBodyPass`` so the
structural-failure pass can reuse it, which is exactly what the compiler does
with the same lists further down its own function.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from ...models import Hand
from .selectors import FullBodyPass


def jumping_jack_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    jumping_jack_primitives = fb.jumping_jack_primitives

    if jumping_jack_primitives:
        jack_labels = {primitive.label for primitive in jumping_jack_primitives}
        jack_ranges = [
            item for item in phase_ranges if item.get("label") in jack_labels
        ]
        jack_indices = [
            index
            for index, frame in enumerate(frames)
            if any(
                float(item["start_s"]) <= frame.time_s <= float(item["end_s"])
                for item in jack_ranges
            )
        ]
        jack_root_heights = np.asarray(
            [world_positions[index]["hips"][1] for index in jack_indices],
            dtype=float,
        )
        jack_foot_separations = np.asarray(
            [
                np.linalg.norm(
                    world_positions[index]["leftFoot"][[0, 2]]
                    - world_positions[index]["rightFoot"][[0, 2]]
                )
                for index in jack_indices
            ],
            dtype=float,
        )
        metrics["requested_jumping_jack_cycles"] = sum(
            float(primitive.body.cycles)
            for primitive in jumping_jack_primitives
            if primitive.body is not None
        )
        metrics["measured_jumping_jack_cycles"] = int(
            len(find_peaks(jack_root_heights, prominence=0.08)[0])
            if jack_root_heights.size
            else 0
        )
        metrics["jumping_jack_foot_spread_excursion_m"] = (
            float(np.ptp(jack_foot_separations))
            if jack_foot_separations.size
            else 0.0
        )
        metrics["jumping_jack_max_wrist_height_m"] = max(
            (
                float(world_positions[index][f"{side}Hand"][1])
                for index in jack_indices
                for side in (Hand.LEFT.value, Hand.RIGHT.value)
            ),
            default=0.0,
        )


def burpee_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    neutral_ground_height = fb.neutral_ground_height
    burpee_jump_primitives = fb.burpee_jump_primitives
    burpee_floor_primitives = fb.burpee_floor_primitives

    if burpee_jump_primitives:
        burpee_jump_labels = {
            primitive.label for primitive in burpee_jump_primitives
        }
        burpee_jump_ranges = [
            item
            for item in phase_ranges
            if item.get("label") in burpee_jump_labels
        ]
        measured_jumps = 0
        airborne_phases = 0
        for item in burpee_jump_ranges:
            indices = [
                index
                for index, frame in enumerate(frames)
                if float(item["start_s"]) <= frame.time_s <= float(item["end_s"])
            ]
            heights = np.asarray(
                [world_positions[index]["hips"][1] for index in indices],
                dtype=float,
            )
            measured_jumps += len(find_peaks(heights, prominence=0.08)[0])
            airborne_phases += int(
                any(
                    float(world_positions[index]["leftToes"][1])
                    > neutral_ground_height + 0.028
                    and float(world_positions[index]["rightToes"][1])
                    > neutral_ground_height + 0.028
                    for index in indices
                )
            )
        metrics["requested_burpee_cycles"] = len(burpee_jump_primitives)
        metrics["measured_burpee_jump_cycles"] = measured_jumps
        metrics["measured_burpee_push_up_cycles"] = 0
        metrics["burpee_floor_support_phase_count"] = len(
            burpee_floor_primitives
        )
        metrics["burpee_airborne_phase_count"] = airborne_phases
        metrics["burpee_root_vertical_excursion_m"] = float(
            metrics["root_vertical_max_m"] - metrics["root_vertical_min_m"]
        )
        burpee_jump_indices = [
            index
            for item in burpee_jump_ranges
            for index, frame in enumerate(frames)
            if float(item["start_s"]) <= frame.time_s <= float(item["end_s"])
        ]
        metrics["burpee_max_wrist_height_m"] = max(
            (
                float(world_positions[index][f"{side}Hand"][1])
                for index in burpee_jump_indices
                for side in (Hand.LEFT.value, Hand.RIGHT.value)
            ),
            default=0.0,
        )


def squat_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    squat_primitives = fb.squat_primitives

    if squat_primitives:
        squat_labels = {primitive.label for primitive in squat_primitives}
        squat_ranges = [
            item for item in phase_ranges if item.get("label") in squat_labels
        ]
        measured_squats = 0
        squat_depths: list[float] = []
        stance_return_errors: list[float] = []
        for range_index, item in enumerate(squat_ranges):
            phase_indices = [
                index
                for index, frame in enumerate(frames)
                if float(item["start_s"]) <= frame.time_s <= float(item["end_s"])
            ]
            if not phase_indices:
                continue
            start_height = float(world_positions[phase_indices[0]]["hips"][1])
            minimum_height = min(
                float(world_positions[index]["hips"][1])
                for index in phase_indices
            )
            depth = start_height - minimum_height
            squat_depths.append(depth)
            next_descent_start = (
                float(squat_ranges[range_index + 1]["start_s"])
                if range_index + 1 < len(squat_ranges)
                else float(frames[-1].time_s)
            )
            recovery_indices = [
                index
                for index, frame in enumerate(frames)
                if float(item["end_s"]) <= frame.time_s <= next_descent_start
            ]
            recovered_height = max(
                (
                    float(world_positions[index]["hips"][1])
                    for index in recovery_indices
                ),
                default=minimum_height,
            )
            return_error = abs(start_height - recovered_height)
            stance_return_errors.append(return_error)
            measured_squats += int(depth >= 0.10 and return_error <= 0.035)
        metrics["requested_squat_cycles"] = len(squat_primitives)
        metrics["measured_squat_cycles"] = measured_squats
        metrics["squat_minimum_depth_m"] = min(squat_depths, default=0.0)
        metrics["squat_max_stance_return_error_m"] = max(
            stance_return_errors,
            default=0.0,
        )


def lunge_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    lunge_primitives = fb.lunge_primitives

    if lunge_primitives:
        lunge_labels = {primitive.label for primitive in lunge_primitives}
        lunge_ranges = [
            item for item in phase_ranges if item.get("label") in lunge_labels
        ]
        measured_lunges = 0
        lunge_depths: list[float] = []
        lunge_staggers: list[float] = []
        lunge_return_errors: list[float] = []
        for range_index, item in enumerate(lunge_ranges):
            phase_indices = [
                index
                for index, frame in enumerate(frames)
                if float(item["start_s"]) <= frame.time_s <= float(item["end_s"])
            ]
            if not phase_indices:
                continue
            start_height = float(world_positions[phase_indices[0]]["hips"][1])
            minimum_height = min(
                float(world_positions[index]["hips"][1])
                for index in phase_indices
            )
            depth = start_height - minimum_height
            lunge_depths.append(depth)
            start_separation = (
                world_positions[phase_indices[0]]["leftFoot"][[0, 2]]
                - world_positions[phase_indices[0]]["rightFoot"][[0, 2]]
            )
            stagger = max(
                float(
                    np.linalg.norm(
                        (
                            world_positions[index]["leftFoot"][[0, 2]]
                            - world_positions[index]["rightFoot"][[0, 2]]
                        )
                        - start_separation
                    )
                )
                for index in phase_indices
            )
            lunge_staggers.append(stagger)
            next_lunge_start = (
                float(lunge_ranges[range_index + 1]["start_s"])
                if range_index + 1 < len(lunge_ranges)
                else float(frames[-1].time_s)
            )
            recovery_indices = [
                index
                for index, frame in enumerate(frames)
                if float(item["end_s"]) <= frame.time_s <= next_lunge_start
            ]
            recovered_height = max(
                (
                    float(world_positions[index]["hips"][1])
                    for index in recovery_indices
                ),
                default=minimum_height,
            )
            return_error = abs(start_height - recovered_height)
            lunge_return_errors.append(return_error)
            measured_lunges += int(
                depth >= 0.10 and stagger >= 0.12 and return_error <= 0.035
            )
        metrics["requested_lunge_cycles"] = len(lunge_primitives)
        metrics["measured_lunge_cycles"] = measured_lunges
        metrics["lunge_minimum_root_drop_m"] = min(lunge_depths, default=0.0)
        metrics["lunge_minimum_foot_stagger_m"] = min(lunge_staggers, default=0.0)
        metrics["lunge_max_stance_return_error_m"] = max(
            lunge_return_errors,
            default=0.0,
        )


def single_leg_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    neutral_ground_height = fb.neutral_ground_height
    single_leg_primitives = fb.single_leg_primitives

    if single_leg_primitives:
        balance_labels = {primitive.label for primitive in single_leg_primitives}
        balance_ranges = [
            item for item in phase_ranges if item.get("label") in balance_labels
        ]
        raised_clearances: list[float] = []
        support_slides: list[float] = []
        support_contact_samples = 0
        support_sample_count = 0
        for primitive, item in zip(
            single_leg_primitives,
            balance_ranges,
            strict=True,
        ):
            assert primitive.body is not None
            raised_side = (
                Hand.LEFT.value
                if primitive.body.pose.left_foot_lift_m > 1e-8
                else Hand.RIGHT.value
            )
            support_side = (
                Hand.RIGHT.value
                if raised_side == Hand.LEFT.value
                else Hand.LEFT.value
            )
            indices = [
                index
                for index, frame in enumerate(frames)
                if float(item["start_s"]) <= frame.time_s <= float(item["end_s"])
            ]
            if not indices:
                continue
            raised_clearances.append(
                max(
                    float(world_positions[index][f"{raised_side}Toes"][1])
                    - neutral_ground_height
                    for index in indices
                )
            )
            support_origin = world_positions[indices[0]][f"{support_side}Foot"]
            support_slides.append(
                max(
                    float(
                        np.linalg.norm(
                            world_positions[index][f"{support_side}Foot"]
                            - support_origin
                        )
                    )
                    for index in indices
                )
            )
            support_contact_samples += sum(
                float(world_positions[index][f"{support_side}Toes"][1])
                <= neutral_ground_height + 0.028
                for index in indices
            )
            support_sample_count += len(indices)
        metrics["single_leg_balance_phase_count"] = len(single_leg_primitives)
        metrics["single_leg_min_raised_foot_clearance_m"] = min(
            raised_clearances,
            default=0.0,
        )
        metrics["single_leg_max_support_foot_slide_m"] = max(
            support_slides,
            default=0.0,
        )
        metrics["single_leg_support_contact_fraction"] = (
            support_contact_samples / support_sample_count
            if support_sample_count
            else 0.0
        )


def sit_up_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    sit_up_primitives = fb.sit_up_primitives

    if sit_up_primitives:
        sit_up_labels = {primitive.label for primitive in sit_up_primitives}
        sit_up_ranges = [
            item for item in phase_ranges if item.get("label") in sit_up_labels
        ]
        measured_sit_ups = 0
        head_lifts: list[float] = []
        return_errors: list[float] = []
        for primitive, item in zip(
            sit_up_primitives,
            sit_up_ranges,
            strict=True,
        ):
            phase_indices = [
                index
                for index, frame in enumerate(frames)
                if float(item["start_s"]) <= frame.time_s <= float(item["end_s"])
            ]
            if not phase_indices:
                continue
            start_height = float(world_positions[phase_indices[0]]["head"][1])
            lift = max(
                float(world_positions[index]["head"][1])
                for index in phase_indices
            ) - start_height
            head_lifts.append(lift)
            return_label = str(primitive.label).replace("_curl", "_return")
            return_range = next(
                (
                    phase
                    for phase in phase_ranges
                    if phase.get("label") == return_label
                ),
                None,
            )
            return_indices = (
                [
                    index
                    for index, frame in enumerate(frames)
                    if float(return_range["start_s"])
                    <= frame.time_s
                    <= float(return_range["end_s"])
                ]
                if return_range is not None
                else []
            )
            returned_height = (
                float(world_positions[return_indices[-1]]["head"][1])
                if return_indices
                else float("inf")
            )
            return_error = abs(returned_height - start_height)
            return_errors.append(return_error)
            measured_sit_ups += int(lift >= 0.15 and return_error <= 0.05)
        metrics["requested_sit_up_cycles"] = len(sit_up_primitives)
        metrics["measured_sit_up_cycles"] = measured_sit_ups
        metrics["sit_up_minimum_head_lift_m"] = min(head_lifts, default=0.0)
        metrics["sit_up_max_supine_return_error_m"] = max(
            return_errors,
            default=0.0,
        )


def crawl_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    crawl_primitives = fb.crawl_primitives

    if crawl_primitives:
        crawl_labels = {primitive.label for primitive in crawl_primitives}
        crawl_ranges = [
            item for item in phase_ranges if item.get("label") in crawl_labels
        ]
        crawl_indices = [
            index
            for index, frame in enumerate(frames)
            if any(
                float(item["start_s"]) <= frame.time_s <= float(item["end_s"])
                for item in crawl_ranges
            )
        ]
        hand_opposition = np.asarray(
            [
                float(world_positions[index]["leftHand"][2])
                - float(world_positions[index]["rightHand"][2])
                for index in crawl_indices
            ],
            dtype=float,
        )
        metrics["requested_crawl_cycles"] = max(
            float(primitive.body.cycles)
            for primitive in crawl_primitives
            if primitive.body is not None
        )
        metrics["requested_crawl_distance_m"] = max(
            math.hypot(
                primitive.body.pose.root_shift_x_m,
                primitive.body.pose.root_shift_z_m,
            )
            for primitive in crawl_primitives
            if primitive.body is not None
        )
        metrics["crawl_root_displacement_m"] = float(
            metrics["root_displacement_m"]
        )
        metrics["crawl_hand_alternation_range_m"] = (
            float(np.ptp(hand_opposition)) if hand_opposition.size else 0.0
        )


def push_up_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    push_up_primitives = fb.push_up_primitives
    burpee_jump_primitives = fb.burpee_jump_primitives

    if push_up_primitives:
        push_up_labels = {primitive.label for primitive in push_up_primitives}
        push_up_ranges = [
            item
            for item in phase_ranges
            if item.get("label") in push_up_labels
        ]
        sampled_range_indices: list[list[int]] = []
        for item in push_up_ranges:
            start_s = float(item["start_s"])
            duration_s = float(item["end_s"]) - start_s
            sampled_range_indices.append(
                [
                index
                for index, frame in enumerate(frames)
                if start_s + 0.40 * duration_s
                <= frame.time_s
                <= start_s + 0.90 * duration_s
                ]
            )
        sampled_indices = [
            index
            for range_indices in sampled_range_indices
            for index in range_indices
        ]
        relative_chest_heights = np.asarray(
            [
                float(world_positions[index]["chest"][1])
                - 0.5
                * (
                    float(world_positions[index]["leftHand"][1])
                    + float(world_positions[index]["rightHand"][1])
                )
                for index in sampled_indices
            ],
            dtype=float,
        )
        palm_heights = np.asarray(
            [
                0.5
                * (
                    float(world_positions[index]["leftHand"][1])
                    + float(world_positions[index]["rightHand"][1])
                )
                for index in sampled_indices
            ],
            dtype=float,
        )
        metrics["requested_push_up_cycles"] = sum(
            float(primitive.body.cycles)
            for primitive in push_up_primitives
            if primitive.body is not None
        )
        per_phase_heights = [
            np.asarray(
                [
                    float(world_positions[index]["chest"][1])
                    - 0.5
                    * (
                        float(world_positions[index]["leftHand"][1])
                        + float(world_positions[index]["rightHand"][1])
                    )
                    for index in range_indices
                ],
                dtype=float,
            )
            for range_indices in sampled_range_indices
            if range_indices
        ]
        metrics["push_up_vertical_excursion_m"] = min(
            (float(np.ptp(values)) for values in per_phase_heights),
            default=0.0,
        )
        metrics["plank_hold_vertical_range_m"] = metrics[
            "push_up_vertical_excursion_m"
        ]
        metrics["measured_push_up_cycles"] = sum(
            len(find_peaks(-values, prominence=0.02)[0])
            for values in per_phase_heights
        )
        if burpee_jump_primitives:
            metrics["measured_burpee_push_up_cycles"] = int(
                metrics["measured_push_up_cycles"]
            )
        metrics["push_up_palm_height_range_m"] = (
            float(np.ptp(palm_heights)) if palm_heights.size else 0.0
        )
        toe_tracks = {
            side: np.asarray(
                [world_positions[index][f"{side}Toes"] for index in sampled_indices],
                dtype=float,
            )
            for side in (Hand.LEFT.value, Hand.RIGHT.value)
        }
        metrics["push_up_toe_height_range_m"] = max(
            (float(np.ptp(track[:, 1])) for track in toe_tracks.values() if track.size),
            default=0.0,
        )
        metrics["push_up_toe_position_range_m"] = max(
            (
                float(np.max(np.linalg.norm(track - track[0], axis=1)))
                for track in toe_tracks.values()
                if track.size
            ),
            default=0.0,
        )


__all__ = [
    "burpee_metrics",
    "crawl_metrics",
    "jumping_jack_metrics",
    "lunge_metrics",
    "push_up_metrics",
    "single_leg_metrics",
    "sit_up_metrics",
    "squat_metrics",
]
