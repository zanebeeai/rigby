"""Stepping over and detouring around a scene obstacle.

Selected by ``body.obstacle_mode`` rather than by action, so it composes with
whatever locomotion the primitive already carries.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...models import BodyAction, BodyObstacleMode
from .selectors import FullBodyPass


def obstacle_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    hips_positions = fb.hips_positions
    scene = fb.scene
    obstacle_primitives = fb.obstacle_primitives

    if obstacle_primitives:
        obstacle_ranges = [
            item
            for item in phase_ranges
            if item.get("action")
            in {
                BodyAction.STEP.value,
                BodyAction.WALK.value,
                BodyAction.RUN.value,
            }
        ]
        obstacle_phase_metrics: list[dict[str, Any]] = []
        missing_target_count = 0
        for primitive, phase in zip(
            obstacle_primitives,
            (
                item
                for item in obstacle_ranges
                if item.get("label")
                in {value.label for value in obstacle_primitives}
            ),
            strict=False,
        ):
            assert primitive.body is not None
            target = primitive.body
            obstacle = (
                scene.object_by_id(target.obstacle_object_id)
                if target.obstacle_object_id
                else None
            )
            indices = [
                index
                for index, frame in enumerate(frames)
                if float(phase["start_s"])
                <= frame.time_s
                <= float(phase["end_s"])
            ]
            record: dict[str, Any] = {
                "label": primitive.label or target.action.value,
                "mode": target.obstacle_mode.value,
                "object_id": target.obstacle_object_id,
                "sample_frame_count": len(indices),
            }
            if obstacle is None:
                missing_target_count += 1
                record["target_found"] = False
                obstacle_phase_metrics.append(record)
                continue
            record["target_found"] = True
            object_center = np.asarray(
                [
                    obstacle.transform.translation.x,
                    obstacle.transform.translation.z,
                ],
                dtype=float,
            )
            object_radius = 0.5 * math.hypot(
                obstacle.dimensions_m.x,
                obstacle.dimensions_m.z,
            )
            if target.obstacle_mode == BodyObstacleMode.OVER:
                side = target.lead_side.value
                closest_index = min(
                    indices,
                    key=lambda index: float(
                        np.linalg.norm(
                            world_positions[index][f"{side}Foot"][[0, 2]]
                            - object_center
                        )
                    ),
                    default=None,
                )
                if closest_index is None:
                    crossing_error = float("inf")
                    foot_clearance = float("-inf")
                else:
                    crossing_error = float(
                        np.linalg.norm(
                            world_positions[closest_index][f"{side}Foot"][[0, 2]]
                            - object_center
                        )
                    )
                    object_top = (
                        obstacle.transform.translation.y
                        + 0.5 * obstacle.dimensions_m.y
                    )
                    foot_clearance = min(
                        float(world_positions[closest_index][f"{side}Foot"][1]),
                        float(world_positions[closest_index][f"{side}Toes"][1]),
                    ) - float(object_top)
                record.update(
                    {
                        "closest_approach_time_s": (
                            float(frames[closest_index].time_s)
                            if closest_index is not None
                            else None
                        ),
                        "closest_horizontal_distance_m": crossing_error,
                        "object_horizontal_radius_m": object_radius,
                        "measured_foot_clearance_m": foot_clearance,
                        "requested_clearance_m": target.obstacle_clearance_m,
                    }
                )
            else:
                closest_index = min(
                    indices,
                    key=lambda index: float(
                        np.linalg.norm(
                            hips_positions[index, [0, 2]] - object_center
                        )
                    ),
                    default=None,
                )
                minimum_root_clearance = (
                    float(
                        np.linalg.norm(
                            hips_positions[closest_index, [0, 2]] - object_center
                        )
                    )
                    if closest_index is not None
                    else 0.0
                )
                required_root_clearance = object_radius + 0.22
                record.update(
                    {
                        "closest_approach_time_s": (
                            float(frames[closest_index].time_s)
                            if closest_index is not None
                            else None
                        ),
                        "minimum_root_clearance_m": minimum_root_clearance,
                        "required_root_clearance_m": required_root_clearance,
                        "lateral_detour_m": target.path_lateral_offset_m,
                    }
                )
            obstacle_phase_metrics.append(record)
        metrics["obstacle_traversal_phase_metrics"] = obstacle_phase_metrics
        metrics["obstacle_missing_target_count"] = missing_target_count
        over_records = [
            item
            for item in obstacle_phase_metrics
            if item.get("mode") == BodyObstacleMode.OVER.value
            and item.get("target_found")
        ]
        around_records = [
            item
            for item in obstacle_phase_metrics
            if item.get("mode") == BodyObstacleMode.AROUND.value
            and item.get("target_found")
        ]
        metrics["minimum_obstacle_step_foot_clearance_m"] = min(
            (float(item["measured_foot_clearance_m"]) for item in over_records),
            default=0.0,
        )
        metrics["maximum_obstacle_step_crossing_error_m"] = max(
            (float(item["closest_horizontal_distance_m"]) for item in over_records),
            default=0.0,
        )
        metrics["minimum_obstacle_avoidance_root_clearance_m"] = min(
            (float(item["minimum_root_clearance_m"]) for item in around_records),
            default=0.0,
        )


__all__ = ["obstacle_metrics"]
