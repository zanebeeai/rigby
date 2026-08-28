"""The composite metric pass, moved out of ``_compile_composite``.

Composite is the only path that runs the gesture-structure evaluation **per
hand** and then folds the two into the flat keys every other path emits
directly. The fold is not a tidy-up: ``max_*`` keys take the worse hand,
``active_hand_visibility_fraction`` takes the *minimum*, and
``self_collision_frames`` takes the sum. Reproduced exactly, because a
composite clip's headline numbers are a different aggregation from a gesture
clip's identically-named ones.

Both traps plan 02 §1.5 warned about live here. The presentation window is read
from ``ctx.presentation_ranges``, which reads what the compiler persisted --
re-deriving it moves the window by an ulp. And the bone set fed to the angular
pass comes from ``evaluate_gesture_structure`` per hand rather than from a
path-level list, which is why composite is excluded from the flat
``_angular_metrics`` helper.
"""

from __future__ import annotations

from typing import Any

from ..models import PrimitiveKind
from .contact import (
    intra_hand_contact_failures as _intra_hand_contact_failures,
    intra_hand_contact_metrics as _intra_hand_contact_metrics,
)
from .context import AnalysisContext
from .fingers import curl_values_from_frame as _curl_values_from_frame
from .forearm import (
    parallel_forearm_failures as _parallel_forearm_failures,
    parallel_forearm_metrics as _parallel_forearm_metrics,
)
from .gesture import (
    evaluate_gesture_structure,
    shake_joint_oscillation_metrics,
)
from .safety import clip_contract_violations, safety_metrics as _safety_metrics
from .semantic import (
    semantic_cycle_failures as _semantic_cycle_failures,
    semantic_cycle_metrics as _semantic_cycle_metrics,
)


def composite_metrics(ctx: AnalysisContext) -> dict[str, Any]:
    """Every metric the composite compile path emits, bar the carry-overs."""

    frames = ctx.frames
    program = ctx.program
    phase_ranges = ctx.phase_ranges
    presentation_ranges = ctx.presentation_ranges
    world_positions = ctx.world_positions
    metrics: dict[str, Any] = {}

    metrics["active_hands"] = [hand.value for hand in program.hands]
    metrics["composite_segment_count"] = len(program.primitives)
    metrics["trajectory_cycles"] = max(
        (primitive.parameters.trajectory_cycles for primitive in program.primitives),
        default=0.0,
    )
    metrics["trajectory_amplitude_m"] = max(
        (primitive.parameters.trajectory_amplitude_m for primitive in program.primitives),
        default=0.0,
    )
    metrics["axial_rotation_amplitude"] = max(
        (primitive.parameters.axial_rotation_amplitude for primitive in program.primitives),
        default=0.0,
    )
    metrics.update(
        # The whole context, not just the positions: the contact pass needs
        # fingertip pivots and a head rotation as well, and those come out of
        # the same per-frame world-matrix evaluation the positions do.
        _intra_hand_contact_metrics(frames, phase_ranges, program, ctx=ctx)
    )
    metrics.update(
        _semantic_cycle_metrics(
            frames, phase_ranges, program, world_positions=world_positions
        )
    )
    metrics.update(
        _parallel_forearm_metrics(
            frames, phase_ranges, world_positions=world_positions
        )
    )
    metrics.update(_safety_metrics(frames))
    travel_hand_shapes = {
        effector.hand: effector.hand_shape
        for primitive in program.primitives
        if primitive.label in {
            "parallel_forearm_travel_setup",
            "parallel_forearm_travel_cycle",
        }
        for effector in primitive.effectors
    }
    structures = {
        hand: evaluate_gesture_structure(
            frames,
            hand,
            presentation_ranges,
            travel_hand_shapes.get(hand),
            # Deliberately included, not only the single-hand path: the
            # chest-frame wrist and the rest-pose camera are two errors that
            # partly cancel, so correcting one here and not the other would
            # leave the composite cases measuring the cancelled pair.
        )
        for hand in program.hands
    }
    structural_failures: list[str] = []
    for hand, structure in structures.items():
        structural_failures.extend(
            f"{hand.value}: {failure}" for failure in structure["structural_failures"]
        )
        for key, value in structure.items():
            metrics[f"{hand.value}_{key}"] = value
    metrics.update(
        {
            "max_wrist_swing_rad": max(
                (float(value["max_wrist_swing_rad"]) for value in structures.values()),
                default=0.0,
            ),
            "max_wrist_twist_rad": max(
                (float(value["max_wrist_twist_rad"]) for value in structures.values()),
                default=0.0,
            ),
            "max_forearm_twist_rad": max(
                (float(value["max_forearm_twist_rad"]) for value in structures.values()),
                default=0.0,
            ),
            "self_collision_frames": sum(
                int(value["self_collision_frames"]) for value in structures.values()
            ),
            "active_hand_visibility_fraction": min(
                (float(value["active_hand_visibility_fraction"]) for value in structures.values()),
                default=0.0,
            ),
            "max_angular_velocity_rad_s": max(
                (float(value["max_angular_velocity_rad_s"]) for value in structures.values()),
                default=0.0,
            ),
            "max_angular_acceleration_rad_s2": max(
                (float(value["max_angular_acceleration_rad_s2"]) for value in structures.values()),
                default=0.0,
            ),
            "max_angular_jerk_rad_s3": max(
                (float(value["max_angular_jerk_rad_s3"]) for value in structures.values()),
                default=0.0,
            ),
        }
    )
    cycle_ranges = [
        (float(item["start_s"]), float(item["end_s"]))
        for item in phase_ranges
        if item["kind"] == PrimitiveKind.CYCLE.value
    ]
    if cycle_ranges:
        joint_motion = {
            hand: shake_joint_oscillation_metrics(frames, hand, cycle_ranges)
            for hand in program.hands
        }
        for hand, values in joint_motion.items():
            for key, value in values.items():
                metrics[f"{hand.value}_{key}"] = value
        metrics["forearm_rotation_cycles"] = min(
            values["forearm_rotation_cycles"] for values in joint_motion.values()
        )
        metrics["forearm_rotation_amplitude_rad"] = min(
            values["forearm_rotation_amplitude_rad"] for values in joint_motion.values()
        )
        metrics["wrist_flexion_cycles"] = max(
            values["wrist_flexion_cycles"] for values in joint_motion.values()
        )
        metrics["wrist_deviation_cycles"] = max(
            values["wrist_deviation_cycles"] for values in joint_motion.values()
        )
    structural_failures.extend(_parallel_forearm_failures(metrics))
    if metrics["nan_count"]:
        structural_failures.append("clip contains non-finite transforms")
    if clip_contract_violations(metrics, allow_root_motion=False):
        structural_failures.append("clip exceeds a joint limit")
    structural_failures.extend(_intra_hand_contact_failures(program, metrics))
    structural_failures.extend(_semantic_cycle_failures(program, metrics))
    metrics["structural_failures"] = structural_failures
    metrics["structural_valid"] = not structural_failures
    metrics["finger_assertions"] = {
        f"{hand.value}_shape_defined": all(
            f"{hand.value}{digit}Proximal" in frames[-1].bones
            for digit in ("Index", "Middle", "Ring", "Little")
        )
        for hand in program.hands
    }
    metrics["normalized_finger_curls"] = {
        hand.value: _curl_values_from_frame(frames[-1], hand.value)
        for hand in program.hands
    }
    return metrics


__all__ = ["composite_metrics"]
