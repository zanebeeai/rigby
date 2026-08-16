"""Semantic cycle metrics: the observable path a motion-bearing verb promises.

Moved verbatim from ``compiler._semantic_cycle_metrics`` and
``compiler._append_semantic_cycle_failures``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..kinematics import rig_kinematics
from ..models import (
    AssertionSpec,
    ClipFrame,
    Hand,
    MotionProgram,
    TrajectoryKind,
    TrajectoryPlane,
)
from .contract import SIGNAL, CheckResult, lower_bound_check


def semantic_cycle_assertion(program: MotionProgram) -> AssertionSpec | None:
    return next(
        (
            assertion
            for assertion in program.assertions
            if assertion.name.endswith("_trajectory_reversals")
        ),
        None,
    )


def semantic_cycle_metrics(
    frames: list[ClipFrame],
    phase_ranges: list[dict[str, float | str]],
    program: MotionProgram,
) -> dict[str, Any]:
    """Measure the observable path promised by a motion-bearing verb."""

    assertion = semantic_cycle_assertion(program)
    if assertion is None:
        return {}
    cyclic_primitives = [
        primitive
        for primitive in program.primitives
        if primitive.trajectory == TrajectoryKind.OSCILLATE
        and primitive.effectors
        and primitive.parameters.trajectory_cycles > 0.0
    ]
    if not cyclic_primitives:
        return {
            "semantic_cycle_action": assertion.name.removesuffix(
                "_trajectory_reversals"
            ),
            "semantic_cycle_min_reversal_count": 0,
            "semantic_cycle_min_excursion_m": 0.0,
            "semantic_cycle_requested_amplitude_m": 0.0,
        }
    labels = {
        primitive.label or primitive.kind.value for primitive in cyclic_primitives
    }
    intervals = [
        (float(item["start_s"]), float(item["end_s"]))
        for item in phase_ranges
        if str(item.get("label")) in labels
    ]
    hand_axes: dict[Hand, int] = {}
    requested_cycles = 0.0
    requested_amplitude = 0.0
    for primitive in cyclic_primitives:
        axis = {
            TrajectoryPlane.FRONTAL: 0,
            TrajectoryPlane.SAGITTAL: 1,
            TrajectoryPlane.HORIZONTAL: 2,
        }[primitive.trajectory_plane or TrajectoryPlane.FRONTAL]
        for target in primitive.effectors:
            hand_axes[target.hand] = axis
        requested_cycles = max(
            requested_cycles, primitive.parameters.trajectory_cycles
        )
        requested_amplitude = max(
            requested_amplitude,
            primitive.parameters.trajectory_amplitude_m,
        )
    values: dict[Hand, list[float]] = {hand: [] for hand in hand_axes}
    kinematics = rig_kinematics()
    for frame in frames:
        if not any(start <= frame.time_s <= end for start, end in intervals):
            continue
        positions = kinematics.canonical_positions(frame.bones)
        for hand, axis in hand_axes.items():
            # Measure the path in the moving shoulder frame. World-space
            # wrist positions are dominated by root travel when someone
            # waves while walking, which can erase otherwise valid lateral
            # reversals from the metric.
            values[hand].append(
                float(
                    positions[f"{hand.value}Hand"][axis]
                    - positions[f"{hand.value}UpperArm"][axis]
                )
            )

    per_hand: dict[str, dict[str, float | int]] = {}
    for hand, samples in values.items():
        directions: list[int] = []
        for delta in np.diff(np.asarray(samples, dtype=float)):
            if abs(float(delta)) < 5e-4:
                continue
            direction = 1 if delta > 0.0 else -1
            if not directions or direction != directions[-1]:
                directions.append(direction)
        per_hand[hand.value] = {
            "reversal_count": max(0, len(directions) - 1),
            "excursion_m": (
                float(np.ptp(np.asarray(samples, dtype=float)))
                if samples
                else 0.0
            ),
        }
    return {
        "semantic_cycle_action": assertion.name.removesuffix(
            "_trajectory_reversals"
        ),
        "semantic_cycle_requested_cycles": requested_cycles,
        "semantic_cycle_requested_amplitude_m": requested_amplitude,
        "semantic_cycle_min_reversal_count": min(
            (int(item["reversal_count"]) for item in per_hand.values()),
            default=0,
        ),
        "semantic_cycle_min_excursion_m": min(
            (float(item["excursion_m"]) for item in per_hand.values()),
            default=0.0,
        ),
        "semantic_cycle_hands": per_hand,
    }


def _minimum_excursion_m(metrics: dict[str, Any]) -> float:
    requested_amplitude = float(
        metrics.get("semantic_cycle_requested_amplitude_m", 0.0)
    )
    return max(0.045, requested_amplitude * 0.75)


def semantic_cycle_failures(
    program: MotionProgram,
    metrics: dict[str, Any],
) -> list[str]:
    """The semantic-cycle structural failures, in the compiler's exact order."""

    assertion = semantic_cycle_assertion(program)
    if assertion is None:
        return []
    failures: list[str] = []
    required_reversals = int(round(assertion.threshold or 1.0))
    observed_reversals = int(metrics.get("semantic_cycle_min_reversal_count", 0))
    if observed_reversals < required_reversals:
        failures.append(
            f"{metrics.get('semantic_cycle_action', 'cyclic action')} has "
            f"{observed_reversals} visible reversals; {required_reversals} required"
        )
    minimum_excursion = _minimum_excursion_m(metrics)
    observed_excursion = float(metrics.get("semantic_cycle_min_excursion_m", 0.0))
    if observed_excursion < minimum_excursion:
        failures.append(
            f"{metrics.get('semantic_cycle_action', 'cyclic action')} lateral/path "
            f"excursion is {observed_excursion:.3f} m; {minimum_excursion:.3f} m required"
        )
    return failures


def semantic_cycle_checks(
    program: MotionProgram,
    metrics: dict[str, Any],
) -> list[CheckResult]:
    assertion = semantic_cycle_assertion(program)
    if assertion is None:
        return []
    required_reversals = float(int(round(assertion.threshold or 1.0)))
    minimum_excursion = _minimum_excursion_m(metrics)
    return [
        lower_bound_check(
            "signal.semantic_cycle.reversals",
            SIGNAL,
            float(metrics.get("semantic_cycle_min_reversal_count", 0)),
            required_reversals,
            scale=max(required_reversals, 1.0),
            detail="cyclic action does not show the required visible reversals",
        ),
        lower_bound_check(
            "signal.semantic_cycle.excursion",
            SIGNAL,
            float(metrics.get("semantic_cycle_min_excursion_m", 0.0)),
            minimum_excursion,
            scale=minimum_excursion,
            detail="cyclic action does not travel far enough to read",
        ),
    ]
