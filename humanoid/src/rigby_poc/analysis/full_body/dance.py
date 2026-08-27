"""Beat count, lateral sway and alternating foot lifts for the dance action."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

from ...models import BodyAction, Hand
from .selectors import FullBodyPass


def dance_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    hips_positions = fb.hips_positions
    toe_clearances = fb.toe_clearances
    dance_primitives = fb.dance_primitives

    if dance_primitives:
        dance_ranges = [
            item
            for item in phase_ranges
            if item.get("action") == BodyAction.DANCE.value
        ]
        phase_diagnostics: list[dict[str, Any]] = []
        requested_beats = 0
        measured_beats = 0
        alternating_lifts = 0
        lateral_ranges: list[float] = []
        peak_clearances = {
            Hand.LEFT.value: [],
            Hand.RIGHT.value: [],
        }
        for primitive, phase in zip(
            dance_primitives,
            dance_ranges,
            strict=False,
        ):
            assert primitive.body is not None
            beats = max(1, int(round(float(primitive.body.cycles))))
            requested_beats += beats
            indices = [
                index
                for index, frame in enumerate(frames)
                if float(phase["start_s"])
                <= frame.time_s
                <= float(phase["end_s"])
            ]
            if not indices:
                continue
            minimum_peak_distance = max(
                2,
                int(round(len(indices) / beats * 0.50)),
            )
            lift_peaks: list[tuple[int, str, float]] = []
            phase_peak_clearances: dict[str, float] = {}
            for side in (Hand.LEFT.value, Hand.RIGHT.value):
                values = toe_clearances[side][indices]
                phase_peak_clearances[side] = float(np.max(values))
                peak_clearances[side].append(phase_peak_clearances[side])
                local_peaks = find_peaks(
                    values,
                    height=0.025,
                    prominence=0.012,
                    distance=minimum_peak_distance,
                )[0]
                lift_peaks.extend(
                    (
                        indices[int(local_index)],
                        side,
                        float(values[int(local_index)]),
                    )
                    for local_index in local_peaks
                )
            lift_peaks.sort(key=lambda value: value[0])
            measured_beats += len(lift_peaks)
            previous_side: str | None = None
            phase_alternating_lifts = 0
            for _, side, _ in lift_peaks:
                if previous_side is None or side != previous_side:
                    phase_alternating_lifts += 1
                previous_side = side
            alternating_lifts += phase_alternating_lifts
            lateral_range = max(
                float(np.ptp(hips_positions[indices, 0])),
                float(np.ptp(hips_positions[indices, 2])),
            )
            lateral_ranges.append(lateral_range)
            phase_diagnostics.append(
                {
                    "label": primitive.label or BodyAction.DANCE.value,
                    "requested_beats": beats,
                    "measured_beats": len(lift_peaks),
                    "alternating_lift_count": phase_alternating_lifts,
                    "lift_sequence": [side for _, side, _ in lift_peaks],
                    "lateral_root_range_m": lateral_range,
                    "left_foot_peak_clearance_m": phase_peak_clearances.get(
                        Hand.LEFT.value,
                        0.0,
                    ),
                    "right_foot_peak_clearance_m": phase_peak_clearances.get(
                        Hand.RIGHT.value,
                        0.0,
                    ),
                }
            )
        metrics["requested_dance_beats"] = requested_beats
        metrics["measured_dance_beats"] = measured_beats
        metrics["dance_alternating_lift_count"] = alternating_lifts
        metrics["dance_lateral_root_range_m"] = max(lateral_ranges, default=0.0)
        metrics["dance_left_foot_peak_clearance_m"] = max(
            peak_clearances[Hand.LEFT.value],
            default=0.0,
        )
        metrics["dance_right_foot_peak_clearance_m"] = max(
            peak_clearances[Hand.RIGHT.value],
            default=0.0,
        )
        metrics["dance_phase_metrics"] = phase_diagnostics


__all__ = ["dance_metrics"]
