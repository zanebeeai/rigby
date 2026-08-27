"""Climb progress and support quality.

``climb_support_target_max_error_m`` and ``climb_three_point_support_fraction``
compare achieved limb positions against the *commanded* support targets, which
is why the compiler now persists ``climb_support_constraints``: only the frames
are recoverable from a clip, never the intent behind them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...models import BodyAction
from .selectors import FullBodyPass


def climb_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    frames = fb.frames
    phase_ranges = fb.phase_ranges
    world_positions = fb.world_positions
    hips_positions = fb.hips_positions
    scene = fb.scene
    climb_primitives = fb.climb_primitives
    climb_support_constraints = fb.climb_support_constraints

    if climb_primitives:
        climb_ranges = [
            item
            for item in phase_ranges
            if item.get("action") == BodyAction.CLIMB.value
        ]
        requested_height = sum(
            float(primitive.body.height_m)
            for primitive in climb_primitives
            if primitive.body is not None
        )
        requested_cycles = sum(
            float(primitive.body.cycles)
            for primitive in climb_primitives
            if primitive.body is not None
        )
        measured_height = 0.0
        phase_diagnostics: list[dict[str, Any]] = []
        for primitive, phase in zip(climb_primitives, climb_ranges, strict=False):
            assert primitive.body is not None
            indices = [
                index
                for index, frame in enumerate(frames)
                if float(phase["start_s"])
                <= frame.time_s
                <= float(phase["end_s"])
            ]
            if not indices:
                continue
            phase_height = abs(
                float(hips_positions[indices[-1], 1] - hips_positions[indices[0], 1])
            )
            measured_height += phase_height
            phase_diagnostics.append(
                {
                    "label": primitive.label or BodyAction.CLIMB.value,
                    "support_object_id": primitive.body.support_object_id,
                    "direction": primitive.body.climb_direction.value,
                    "requested_height_m": primitive.body.height_m,
                    "measured_height_m": phase_height,
                    "requested_cycles": primitive.body.cycles,
                }
            )
        settled_constraints = [
            item for item in climb_support_constraints if item.get("settled")
        ]
        target_errors: list[float] = []
        supported_samples = 0
        final_supported_limb_count = 0
        for constraint in settled_constraints:
            frame_index = int(constraint["frame_index"])
            errors = [
                float(
                    np.linalg.norm(
                        world_positions[frame_index][name]
                        - np.asarray(target, dtype=float)
                    )
                )
                for name, target in constraint["targets"].items()
            ]
            target_errors.extend(errors)
            supported_samples += int(sum(error <= 0.10 for error in errors) >= 3)
            final_supported_limb_count = sum(error <= 0.10 for error in errors)
        metrics["requested_climb_height_m"] = requested_height
        metrics["measured_climb_height_m"] = measured_height
        metrics["climb_vertical_completion_fraction"] = (
            measured_height / requested_height if requested_height > 1e-8 else 0.0
        )
        metrics["requested_climb_cycles"] = requested_cycles
        metrics["climb_support_target_max_error_m"] = max(
            target_errors,
            default=999.0,
        )
        metrics["climb_three_point_support_fraction"] = (
            supported_samples / len(settled_constraints)
            if settled_constraints
            else 0.0
        )
        metrics["climb_final_supported_limb_count"] = final_supported_limb_count
        metrics["climb_missing_support_object_count"] = sum(
            scene.object_by_id(primitive.body.support_object_id) is None
            for primitive in climb_primitives
            if primitive.body is not None and primitive.body.support_object_id
        )
        metrics["climb_phase_metrics"] = phase_diagnostics


__all__ = ["climb_metrics"]
