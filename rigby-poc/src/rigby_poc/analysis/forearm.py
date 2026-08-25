"""Parallel-forearm ("travel signal") metrics and their structural checks.

Moved verbatim from ``compiler._parallel_forearm_metrics`` and the inline
threshold block in ``compiler._compile_composite``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..kinematics import rig_kinematics
from ..models import ClipFrame
from .contract import ANATOMY, SIGNAL, CheckResult, lower_bound_check, upper_bound_check
from .geometry import line_segment_distance


TRAVEL_CYCLE_LABEL = "parallel_forearm_travel_cycle"


def parallel_forearm_metrics(
    frames: list[ClipFrame],
    phase_ranges: list[dict[str, float | str]],
    *,
    world_positions: list[dict[str, np.ndarray]] | None = None,
) -> dict[str, float]:
    intervals = [
        (float(item["start_s"]), float(item["end_s"]))
        for item in phase_ranges
        if item.get("label") == TRAVEL_CYCLE_LABEL
    ]
    if not intervals:
        return {}
    kinematics = rig_kinematics() if world_positions is None else None
    axis_errors: list[float] = []
    frontal_axis_errors: list[float] = []
    separations: list[float] = []
    hand_separations: list[float] = []
    opposite_elbow_distances: list[float] = []
    cross_body_samples: list[bool] = []
    wrist_vertical_orders: list[float] = []
    wrist_depth_orders: list[float] = []
    for index, frame in enumerate(frames):
        if not any(start <= frame.time_s <= end for start, end in intervals):
            continue
        positions = (
            world_positions[index]
            if world_positions is not None
            else kinematics.canonical_positions(frame.bones)  # type: ignore[union-attr]
        )
        left_elbow = positions["leftLowerArm"]
        left_wrist = positions["leftHand"]
        right_elbow = positions["rightLowerArm"]
        right_wrist = positions["rightHand"]
        left_axis = left_wrist - left_elbow
        right_axis = right_wrist - right_elbow
        alignment = abs(
            float(np.dot(left_axis, right_axis))
            / max(
                float(np.linalg.norm(left_axis) * np.linalg.norm(right_axis)),
                1e-10,
            )
        )
        axis_errors.append(
            math.degrees(math.acos(float(np.clip(alignment, -1.0, 1.0))))
        )
        left_frontal = left_axis[:2]
        right_frontal = right_axis[:2]
        frontal_alignment = abs(
            float(np.dot(left_frontal, right_frontal))
            / max(
                float(
                    np.linalg.norm(left_frontal)
                    * np.linalg.norm(right_frontal)
                ),
                1e-10,
            )
        )
        frontal_axis_errors.append(
            math.degrees(
                math.acos(float(np.clip(frontal_alignment, -1.0, 1.0)))
            )
        )
        separations.append(
            line_segment_distance(
                left_elbow,
                left_wrist,
                right_elbow,
                right_wrist,
            )
        )
        hand_separations.append(float(np.linalg.norm(left_wrist - right_wrist)))
        opposite_elbow_distances.extend(
            (
                float(np.linalg.norm(left_wrist - right_elbow)),
                float(np.linalg.norm(right_wrist - left_elbow)),
            )
        )
        cross_body_samples.append(
            float(left_wrist[0]) < -0.08 and float(right_wrist[0]) > 0.08
        )
        wrist_vertical_orders.append(float(left_wrist[1] - right_wrist[1]))
        wrist_depth_orders.append(float(left_wrist[2] - right_wrist[2]))
    if not axis_errors:
        return {}
    return {
        "parallel_forearm_max_axis_error_deg": max(axis_errors),
        "parallel_forearm_p95_axis_error_deg": float(
            np.percentile(axis_errors, 95.0)
        ),
        "parallel_forearm_max_frontal_axis_error_deg": max(
            frontal_axis_errors
        ),
        "parallel_forearm_p95_frontal_axis_error_deg": float(
            np.percentile(frontal_axis_errors, 95.0)
        ),
        "parallel_forearm_minimum_separation_m": min(separations),
        "parallel_forearm_minimum_hand_separation_m": min(hand_separations),
        "travel_wheel_maximum_opposite_elbow_distance_m": max(
            opposite_elbow_distances
        ),
        "travel_wheel_cross_body_fraction": sum(cross_body_samples)
        / len(cross_body_samples),
        "travel_wheel_minimum_vertical_order_m": min(wrist_vertical_orders),
        "travel_wheel_maximum_vertical_order_m": max(wrist_vertical_orders),
        "travel_wheel_vertical_order_range_m": max(wrist_vertical_orders)
        - min(wrist_vertical_orders),
        "travel_wheel_minimum_depth_order_m": min(wrist_depth_orders),
        "travel_wheel_maximum_depth_order_m": max(wrist_depth_orders),
        "travel_wheel_depth_order_range_m": max(wrist_depth_orders)
        - min(wrist_depth_orders),
    }


def parallel_forearm_failures(metrics: dict[str, Any]) -> list[str]:
    """The travel-signal structural failures, in the compiler's exact order."""

    if "parallel_forearm_max_axis_error_deg" not in metrics:
        return []
    failures: list[str] = []
    if float(metrics["parallel_forearm_max_axis_error_deg"]) > 20.0:
        failures.append("travel-signal forearms form an excessive depth V")
    if float(metrics["parallel_forearm_max_frontal_axis_error_deg"]) > 8.0:
        failures.append(
            "travel-signal forearms do not remain parallel in presentation"
        )
    if float(metrics["parallel_forearm_minimum_separation_m"]) < 0.025:
        failures.append("travel-signal forearms intersect or lose clearance")
    if float(metrics["parallel_forearm_minimum_hand_separation_m"]) < 0.28:
        failures.append(
            "travel-signal hands collapse into the same base position"
        )
    if float(metrics["travel_wheel_cross_body_fraction"]) < 0.95:
        failures.append(
            "travel-signal fists do not remain across by the opposite elbows"
        )
    if float(metrics["travel_wheel_maximum_opposite_elbow_distance_m"]) > 0.20:
        failures.append(
            "travel-signal fists stray too far from the opposite elbows"
        )
    if not (
        float(metrics["travel_wheel_minimum_vertical_order_m"]) <= -0.15
        and float(metrics["travel_wheel_maximum_vertical_order_m"]) >= 0.15
        and float(metrics["travel_wheel_minimum_depth_order_m"]) <= -0.08
        and float(metrics["travel_wheel_maximum_depth_order_m"]) >= 0.08
    ):
        failures.append(
            "travel-signal forearms do not exchange over/under and front/back order"
        )
    return failures


def parallel_forearm_checks(metrics: dict[str, Any]) -> list[CheckResult]:
    if "parallel_forearm_max_axis_error_deg" not in metrics:
        return []
    exchanged = (
        float(metrics["travel_wheel_minimum_vertical_order_m"]) <= -0.15
        and float(metrics["travel_wheel_maximum_vertical_order_m"]) >= 0.15
        and float(metrics["travel_wheel_minimum_depth_order_m"]) <= -0.08
        and float(metrics["travel_wheel_maximum_depth_order_m"]) >= 0.08
    )
    return [
        upper_bound_check(
            "anatomy.travel_wheel.axis_error",
            ANATOMY,
            float(metrics["parallel_forearm_max_axis_error_deg"]),
            20.0,
            scale=40.0,
            detail="travel-signal forearms form an excessive depth V",
        ),
        upper_bound_check(
            "anatomy.travel_wheel.frontal_axis_error",
            ANATOMY,
            float(metrics["parallel_forearm_max_frontal_axis_error_deg"]),
            8.0,
            scale=30.0,
            detail="travel-signal forearms do not remain parallel in presentation",
        ),
        lower_bound_check(
            "anatomy.travel_wheel.forearm_separation",
            ANATOMY,
            float(metrics["parallel_forearm_minimum_separation_m"]),
            0.025,
            scale=0.025,
            detail="travel-signal forearms intersect or lose clearance",
        ),
        lower_bound_check(
            "anatomy.travel_wheel.hand_separation",
            ANATOMY,
            float(metrics["parallel_forearm_minimum_hand_separation_m"]),
            0.28,
            scale=0.28,
            detail="travel-signal hands collapse into the same base position",
        ),
        lower_bound_check(
            "signal.travel_wheel.cross_body_fraction",
            SIGNAL,
            float(metrics["travel_wheel_cross_body_fraction"]),
            0.95,
            scale=0.95,
            detail="travel-signal fists do not remain across by the opposite elbows",
        ),
        upper_bound_check(
            "signal.travel_wheel.opposite_elbow_distance",
            SIGNAL,
            float(metrics["travel_wheel_maximum_opposite_elbow_distance_m"]),
            0.20,
            scale=0.20,
            detail="travel-signal fists stray too far from the opposite elbows",
        ),
        CheckResult(
            id="signal.travel_wheel.order_exchange",
            layer=SIGNAL,
            status="pass" if exchanged else "fail",
            measured={
                "minimum_vertical_order_m": float(
                    metrics["travel_wheel_minimum_vertical_order_m"]
                ),
                "maximum_vertical_order_m": float(
                    metrics["travel_wheel_maximum_vertical_order_m"]
                ),
                "minimum_depth_order_m": float(
                    metrics["travel_wheel_minimum_depth_order_m"]
                ),
                "maximum_depth_order_m": float(
                    metrics["travel_wheel_maximum_depth_order_m"]
                ),
            },
            threshold=None,
            severity=0.0 if exchanged else 1.0,
            detail="travel-signal forearms do not exchange over/under and front/back order",
        ),
    ]
