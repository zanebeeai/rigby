"""The gesture, strike and grab metric pass, moved out of ``compile_motion``.

What this owns is everything derivable from the finished clip and the program.
What it deliberately does not own is the MuJoCo grasp block: ``simulate_grasp``
runs a proxy gripper disconnected from the rig, and importing ``rigby_poc.physics``
pulls in ``mujoco`` -- which ``tests/test_analysis_imports.py`` exists to prevent,
because a check has to run where MuJoCo is absent. Plan 02 §1.4 item 3 said the
analyzer could simply re-run ``_physics_for``; it cannot, without giving up the
property that makes the layer worth extracting. Those ~10 keys and the GRAB
structural branch that reads ``physics.success`` are deferred **by design rather
than by schedule**.

Two quirks are reproduced rather than cleaned up, because this is a move. The
gesture path folds ``wrist_swing_twist_limit_violations`` into the shared joint
limit counter after the fact and republishes the collision count under a generic
key. And ``forearm_rotation_cycles`` is seeded from the planner's *requested*
``wrist_shake_cycles`` and only overwritten with the measured value on the
GESTURE path -- so a STRIKE program carrying a SHAKE primitive publishes a metric
named as a measurement whose value is the request verbatim. That is reachable
through the API, not merely theoretical; ``strike_shake_echo`` in the fixture is
such a clip. Fixing it moves numbers, so it belongs in its own PR.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..models import ClipFrame, HandShape, Intent, MotionProgram, PrimitiveKind
from ..primitives import forearm_shake_amplitude_rad
from .context import AnalysisContext
from .fingers import curl_values_from_frame as _curl_values_from_frame
from .gesture import (
    evaluate_gesture_structure,
    shake_joint_oscillation_metrics,
    world_arm_landmarks,
    world_hand_samples,
)
from .safety import safety_metrics as _safety_metrics


def final_hand_shape(program: MotionProgram) -> HandShape:
    """Reproduce the shape the compile loop carried out of its last primitive.

    A fold over the program: each primitive inherits the running shape unless it
    names its own, and a recovery on the gesture or strike path forces OPEN
    because recovery returns to a relaxed pose rather than holding the gesture.
    """

    final_shape = HandShape.OPEN
    for primitive in program.primitives:
        shape = primitive.hand_shape or final_shape
        if (
            program.intent in {Intent.GESTURE, Intent.STRIKE}
            and primitive.kind == PrimitiveKind.RECOVER
        ):
            shape = HandShape.OPEN
        final_shape = shape
    return final_shape


def assertion_frame_for(ctx: AnalysisContext) -> ClipFrame:
    """The frame the hand-shape assertions are read from.

    The compiler latches ``frames[-1]`` at the end of every HOLD and STRIKE
    primitive, so what survives the loop is the last frame of the *last* such
    phase -- not the last frame of the clip, when anything follows it. Recovered
    here from ``phase_ranges_s`` rather than persisted, and verified frame-exact
    on every gesture, strike and grab case in the fixture.
    """

    frames = ctx.frames
    latching = [
        item
        for item in ctx.phase_ranges
        if item.get("kind") in {PrimitiveKind.HOLD.value, PrimitiveKind.STRIKE.value}
    ]
    if not latching:
        return frames[-1]
    end = float(latching[-1]["end_s"])
    index = max(
        (i for i, frame in enumerate(frames) if frame.time_s <= end + 1e-9),
        default=len(frames) - 1,
    )
    return frames[index]


def hand_metrics(ctx: AnalysisContext) -> dict[str, Any]:
    """Every metric the gesture/strike/grab path emits, bar physics and carry-overs."""

    frames = ctx.frames
    program = ctx.program
    phase_ranges = ctx.phase_ranges
    presentation_ranges = ctx.presentation_ranges
    final_shape = final_hand_shape(program)
    assertion_frame = assertion_frame_for(ctx)
    metrics: dict[str, Any] = {}

    asserted_shape = final_shape
    if program.intent == Intent.GESTURE:
        gesture_shape = next((item.hand_shape for item in program.primitives if item.kind == PrimitiveKind.HOLD), final_shape)
        asserted_shape = gesture_shape or final_shape
    elif program.intent == Intent.STRIKE:
        strike_shape = next(
            (item.hand_shape for item in program.primitives if item.kind == PrimitiveKind.STRIKE),
            HandShape.FIST,
        )
        asserted_shape = strike_shape or HandShape.FIST
    actual_curls = _curl_values_from_frame(assertion_frame or frames[-1], program.hand.value)
    if asserted_shape == HandShape.HANG_TEN:
        assertions = {
            "thumb_extended": actual_curls["thumb"] < 0.35,
            "little_extended": actual_curls["little"] < 0.35,
            "index_curled": actual_curls["index"] > 0.65,
            "middle_curled": actual_curls["middle"] > 0.65,
            "ring_curled": actual_curls["ring"] > 0.65,
            "all_finger_bones_present": all(
                f"{program.hand.value}{finger}{segment}" in (assertion_frame or frames[-1]).bones
                for finger, segments in {
                    "Thumb": ("Metacarpal", "Proximal", "Distal"),
                    "Index": ("Proximal", "Intermediate", "Distal"),
                    "Middle": ("Proximal", "Intermediate", "Distal"),
                    "Ring": ("Proximal", "Intermediate", "Distal"),
                    "Little": ("Proximal", "Intermediate", "Distal"),
                }.items()
                for segment in segments
            ),
        }
    else:
        assertions = {"shape_defined": len(actual_curls) == 5}
    metrics["finger_assertions"] = assertions
    metrics["normalized_finger_curls"] = actual_curls
    metrics["finger_assertions_computed_from_clip"] = True
    shake_params = next(
        (item.parameters for item in program.primitives if item.kind == PrimitiveKind.SHAKE),
        None,
    )
    if shake_params is not None:
        shake_amplitude_rad = (
            forearm_shake_amplitude_rad(shake_params.duration_s, shake_params.wrist_shake_cycles)
            * shake_params.wrist_shake_amplitude
        )
        metrics["forearm_rotation_cycles"] = shake_params.wrist_shake_cycles
        metrics["forearm_rotation_amplitude_rad"] = shake_amplitude_rad
        # Legacy aliases are kept for old reports and API clients.
        metrics["wrist_shake_cycles"] = shake_params.wrist_shake_cycles
        metrics["wrist_shake_amplitude_rad"] = shake_amplitude_rad
        metrics["shake_duration_s"] = shake_params.duration_s
    metrics.update(_safety_metrics(frames))
    if program.intent in {Intent.GESTURE, Intent.STRIKE}:
        structure = evaluate_gesture_structure(
            frames,
            program.hand,
            presentation_ranges,
            # Not `final_shape`: this path has never passed a hand shape, and
            # doing so now would widen the sampled finger/palm spans on every
            # gesture and strike clip. Only the world data changes here.
            world_samples=world_hand_samples(ctx, program.hand, presentation_ranges),
        )
        metrics.update(structure)
        if program.intent == Intent.GESTURE:
            shake_ranges = [
                (float(item["start_s"]), float(item["end_s"]))
                for item in phase_ranges
                if item["kind"] == PrimitiveKind.SHAKE.value
            ]
            metrics.update(shake_joint_oscillation_metrics(frames, program.hand, shake_ranges))
        else:
            metrics["strike_type"] = program.strike_type.value if program.strike_type else None
            strike_range = next(
                (
                    (float(item["start_s"]), float(item["end_s"]))
                    for item in phase_ranges
                    if item["kind"] == PrimitiveKind.STRIKE.value
                ),
                None,
            )
            # Measured in world, not in the chest frame: the trunk carries part
            # of every punch now, so ``arm_landmarks`` -- which rebuilds the arm
            # from a fixed rest shoulder -- would report the swing relative to a
            # torso that is itself turning. ``ctx.indices_in`` applies the same
            # 1e-8 interval predicate the frame filter used, so the frame set is
            # unchanged.
            strike_indices = ctx.indices_in([strike_range]) if strike_range is not None else []
            landmarks = [
                world_arm_landmarks(ctx, index, program.hand) for index in strike_indices
            ]
            wrists = [item[2] for item in landmarks]
            metrics["strike_wrist_path_length_m"] = float(
                sum(np.linalg.norm(second - first) for first, second in zip(wrists, wrists[1:]))
            )
            metrics["strike_lateral_excursion_m"] = float(
                max((wrist[0] for wrist in wrists), default=0.0)
                - min((wrist[0] for wrist in wrists), default=0.0)
            )
            metrics["strike_forward_excursion_m"] = float(
                max((wrist[2] for wrist in wrists), default=0.0)
                - min((wrist[2] for wrist in wrists), default=0.0)
            )
            if landmarks:
                shoulder, elbow, wrist, _ = landmarks[-1]
                upper = shoulder - elbow
                lower = wrist - elbow
                cosine = float(
                    np.clip(
                        np.dot(upper, lower)
                        / max(np.linalg.norm(upper) * np.linalg.norm(lower), 1e-8),
                        -1.0,
                        1.0,
                    )
                )
                metrics["impact_elbow_angle_deg"] = math.degrees(math.acos(cosine))
        metrics["joint_limit_violations"] += structure["wrist_swing_twist_limit_violations"]
        metrics["unresolved_non_hand_collisions"] = structure["self_collision_frames"]

    return metrics


__all__ = ["assertion_frame_for", "final_hand_shape", "hand_metrics"]
