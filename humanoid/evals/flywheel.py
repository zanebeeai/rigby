from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from evals.capture import capture_result_frames
from rigby_poc.compiler import PROJECT_ROOT, compile_motion
from rigby_poc.judge import RepairPatch, VLMJudge, write_judge_record
from rigby_poc.kinematics import rig_kinematics
from rigby_poc.run_config import effective_configuration
from rigby_poc.observability import (
    NullTracer,
    Tracer,
    current_stage_timeline,
    stage_timeline,
)
from rigby_poc.io_utils import atomic_write_json
from rigby_poc.models import (
    BodyAction,
    ClipResult,
    CompileRequest,
    Hand,
    Intent,
    MotionProgram,
    PlanRequest,
    PrimitiveKind,
    PrimitiveParameters,
    default_scene,
)
from rigby_poc.planner import plan_motion
from rigby_poc.primitives import presentation_arc_amplitude_rad, wrist_flourish_amplitude_rad
from rigby_poc.quality import arm_landmarks
from rigby_poc.store import ResultStore


@dataclass(frozen=True)
class CandidateRecipe:
    name: str
    deltas: dict[str, float]
    scales: dict[str, float]
    duration_scales: dict[str, float]
    purpose: str
    effector_deltas: dict[str, float] = field(default_factory=dict)
    effector_scales: dict[str, float] = field(default_factory=dict)
    body_deltas: dict[str, float] = field(default_factory=dict)
    body_scales: dict[str, float] = field(default_factory=dict)
    pose_scale: float = 1.0
    pose_directional_scale: float = 1.0
    object_deltas: dict[str, float] = field(default_factory=dict)
    object_scales: dict[str, float] = field(default_factory=dict)


BASELINE_SAMPLING_SALT = "rigby-best-of-five-baseline-v2"
BLINDING_SEED_SALT = "rigby-judge-blinding-v1"

# A model cannot select a visibly better candidate when the generator supplies
# five numerical variants of the same motion.  At least one of these measured
# changes must separate every ranked pair.  The values correspond to roughly
# 4 cm of hand travel, 4.5 cm of elbow travel, 10 degrees of hand orientation,
# or four frames of timing at 30 Hz.
PERCEPTUAL_DIVERSITY_THRESHOLDS = {
    "maximum_wrist_separation_m": 0.040,
    "maximum_elbow_separation_m": 0.045,
    "maximum_hand_orientation_separation_rad": 0.18,
    "duration_separation_s": 0.14,
    "shake_amplitude_separation_rad": 0.035,
    "shake_duration_separation_s": 0.12,
    "shake_cycles_separation": 0.75,
    "recovery_duration_separation_s": 0.10,
    "maximum_knee_separation_m": 0.035,
    "maximum_ankle_separation_m": 0.035,
    "maximum_root_vertical_separation_m": 0.020,
    "maximum_object_separation_m": 0.080,
}

RANKED_CANDIDATE_COUNT = 5
MAX_ADAPTIVE_TIMING_SCALE = 2.10


def _primary_parameters(program: MotionProgram) -> PrimitiveParameters:
    """Return parameters suitable for choosing a candidate recipe.

    Ordered programs intentionally own no top-level primitives. Candidate
    recipes are sequence-wide timing variants, so the first executable child
    supplies the bounded parameter schema they require.
    """

    current = program
    while not current.primitives and current.steps:
        current = current.steps[0]
    if not current.primitives:
        raise ValueError("supported program has no parameters for candidate generation")
    return current.primitives[0].parameters


def _repair_parameter_payload(program: MotionProgram) -> dict[str, Any]:
    """Expose every ordered step to the repair judge without flattening it."""

    if program.intent == Intent.SEQUENCE:
        return {
            "intent": program.intent.value,
            "steps": [
                {
                    "step_index": index,
                    "intent": step.intent.value,
                    "parameters": [
                        primitive.parameters.model_dump(mode="json")
                        for primitive in step.primitives
                    ],
                }
                for index, step in enumerate(program.steps, start=1)
            ],
        }
    return _primary_parameters(program).model_dump(mode="json")


def adaptive_timing_recipes() -> list[CandidateRecipe]:
    """Global safety-net variants for every supported motion program.

    These recipes deliberately change only phase timing.  They therefore
    preserve the authored pose, hand shape, object contact, strike path, and
    handedness while reducing kinematic stress.  The primary intent recipes
    still provide pose and style diversity; these variants make candidate
    replenishment reliable when some of those proposals fail physical checks.
    """

    phase_names = tuple(kind.value for kind in PrimitiveKind)
    scales = (1.18, 1.34, 1.52, 1.72, 1.92, MAX_ADAPTIVE_TIMING_SCALE)
    return [
        CandidateRecipe(
            name=f"adaptive_timing_{round(scale * 100):03d}",
            deltas={},
            scales={},
            duration_scales={phase: scale for phase in phase_names},
            purpose=(
                "Preserve the semantic motion exactly while using a globally "
                f"safer {scale:.2f}x phase timing profile."
            ),
        )
        for scale in scales
    ]


def single_sample_baseline_index(prompt: str, *, candidate_count: int = 5) -> int:
    """Choose the independent single-sample baseline before judging.

    A fixed prompt hash makes the choice reproducible without always treating
    the hand-tuned canonical proposal as the baseline to beat.
    """
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    digest = hashlib.sha256(f"{BASELINE_SAMPLING_SALT}\0{prompt}".encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % candidate_count


def blinding_seed(prompt: str, result_ids: Iterable[str]) -> int:
    """Derive the judge's presentation-order seed from the content being judged.

    A round- or match-indexed seed puts recipe *k* in the same slot in round 0 of
    every run of every prompt, so any positional bias in the judge becomes a
    reproducible preference for one recipe.  Sorting the identifiers keeps the
    seed reproducible from content alone, independent of the order the caller
    happens to hold the candidates in.
    """
    joined = "\0".join(sorted(result_ids))
    digest = hashlib.sha256(
        f"{BLINDING_SEED_SALT}\0{prompt}\0{joined}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def grab_candidate_recipes(parameters: PrimitiveParameters) -> list[CandidateRecipe]:
    """Five visibly distinct but physically bounded pickup strategies."""
    return [
        CandidateRecipe("canonical", {}, {}, {}, "Unmodified semantic-plan baseline."),
        CandidateRecipe(
            "deliberate_approach",
            {"elbow_swivel": 0.22, "lateral_offset": 0.12},
            {"grip_force": 1.08, "thumb_opposition": 1.04},
            {"reach": 1.25, "preshape": 1.18, "contact": 1.15, "close": 1.12},
            "Use a slower, wider approach before establishing contact.",
        ),
        CandidateRecipe(
            "compact_pickup",
            {"elbow_swivel": -0.15, "lateral_offset": -0.08},
            {"grip_force": 1.16, "thumb_opposition": 1.08, "lift_height_m": 0.82},
            {"reach": 0.88, "contact": 1.30, "close": 1.30, "lift": 1.15},
            "Keep the reach compact while emphasizing secure contact and closure.",
        ),
        CandidateRecipe(
            "high_clearance_lift",
            {"arm_height": 0.12, "lateral_offset": 0.10, "torso_participation": 0.10},
            {"lift_height_m": 1.32, "grip_force": 1.12},
            {"lift": 1.20, "hold": 1.12, "recover": 1.10},
            "Lift with extra table clearance and a readable elevated hold.",
        ),
        CandidateRecipe(
            "camera_readable_grasp",
            {"lateral_offset": 0.20, "elbow_swivel": 0.12, "arm_depth": -0.10},
            {"grip_force": 1.10, "thumb_opposition": 1.06},
            {"preshape": 1.45, "contact": 1.55, "close": 1.50, "hold": 1.40},
            "Expose preshape, contact, and closure clearly to the egocentric camera.",
        ),
    ]


def object_interaction_candidate_recipes() -> list[CandidateRecipe]:
    """Five visibly distinct executions that preserve the same object lifecycle."""

    return [
        CandidateRecipe("canonical", {}, {}, {}, "Unmodified physical object-motion baseline."),
        CandidateRecipe(
            "compact_controlled",
            {},
            {},
            {"reach": 1.15, "windup": 1.12, "receive": 1.12, "absorb": 1.18, "move": 1.15, "hold": 1.12},
            "Use a compact arc and deliberate hand preparation.",
            object_scales={"distance_m": 0.88, "apex_height_m": 0.78, "spin_turns": 0.70},
        ),
        CandidateRecipe(
            "lofted_readable",
            {},
            {},
            {"release": 1.15, "flight": 1.10, "close": 1.12, "contact": 1.18, "move": 1.28, "hold": 1.20},
            "Expose the release or intercept with a higher, slower arc.",
            object_deltas={"contact_height_m": 0.06},
            object_scales={"apex_height_m": 1.32, "spin_turns": 0.82},
        ),
        CandidateRecipe(
            "flat_decisive",
            {},
            {},
            {"windup": 0.92, "release": 0.90, "absorb": 1.08, "move": 0.88, "hold": 0.90},
            "Use a flatter, more decisive flight with clear follow-through.",
            object_deltas={"contact_depth_m": 0.04},
            object_scales={"distance_m": 1.08, "apex_height_m": 0.70, "spin_turns": 1.18},
        ),
        CandidateRecipe(
            "camera_readable_object",
            {},
            {},
            {"preshape": 1.30, "contact": 1.35, "close": 1.30, "hold": 1.22, "receive": 1.24, "move": 1.18},
            "Keep contact and hand closure readable in both ego and orbit views.",
            object_deltas={"contact_height_m": -0.06, "contact_depth_m": -0.04},
            object_scales={"distance_m": 0.96, "apex_height_m": 1.08, "spin_turns": 0.55},
        ),
    ]


def composite_candidate_recipes(parameters: PrimitiveParameters) -> list[CandidateRecipe]:
    """Five materially different executions of an authored motion program.

    The semantic trajectory, cycle count, active limbs, phase order, and hand
    shapes remain fixed.  Variants change readable spatial extent, body
    participation, and pacing—the dimensions a visual selector can actually
    compare without turning proposal generation into a phrase catalogue.
    """

    return [
        CandidateRecipe(
            "canonical",
            {},
            {},
            {},
            "Unmodified compositional motion baseline.",
        ),
        CandidateRecipe(
            "compact_controlled",
            {},
            {"trajectory_amplitude_m": 0.72, "torso_participation": 0.72},
            {"move": 1.25, "cycle": 1.12, "recover": 1.20},
            "Use a compact, controlled path with restrained torso participation.",
            effector_scales={
                "target_x": 0.88,
                "target_y": 0.94,
                "elbow_swivel": 0.78,
            },
        ),
        CandidateRecipe(
            "wide_readable",
            {},
            {"trajectory_amplitude_m": 1.24, "torso_participation": 1.18},
            {"move": 1.16, "cycle": 1.18, "recover": 1.12},
            "Use a broad, deliberate trajectory that remains readable in first person.",
            effector_scales={
                "target_x": 1.12,
                "target_z": 1.06,
                "elbow_swivel": 1.10,
            },
        ),
        CandidateRecipe(
            "lifted_fluid",
            {"trajectory_amplitude_m": 0.018},
            {"torso_participation": 1.08},
            {"move": 1.22, "cycle": 1.30, "recover": 1.18},
            "Lift the staging slightly and give the coordinated path more fluid timing.",
            effector_deltas={"target_y": 0.12, "target_z": 0.05},
            effector_scales={"elbow_swivel": 0.86},
        ),
        CandidateRecipe(
            "grounded_precise",
            {},
            {"trajectory_amplitude_m": 0.86, "torso_participation": 0.58},
            {"move": 1.32, "cycle": 1.38, "recover": 1.24},
            "Use lower, slower staging with extra control and camera depth.",
            effector_deltas={"target_y": -0.10, "target_z": 0.10},
            effector_scales={"target_x": 0.96, "elbow_swivel": 0.62},
        ),
    ]


def strike_candidate_recipes(parameters: PrimitiveParameters) -> list[CandidateRecipe]:
    """Five materially different, bounded executions of one semantic strike."""
    return [
        CandidateRecipe("canonical", {}, {}, {}, "Unmodified semantic strike baseline."),
        CandidateRecipe(
            "tight_fast_arc",
            {"path_arc": -0.20, "lateral_offset": 0.08, "elbow_swivel": 0.08},
            {"torso_participation": 0.82},
            {"guard": 0.94, "load": 0.90, "strike": 0.88, "follow_through": 0.92, "recover": 0.94},
            "Use a compact, quick strike led by the arm with a restrained follow-through.",
        ),
        CandidateRecipe(
            "torso_power_arc",
            {"path_arc": 0.16, "arm_depth": 0.06, "elbow_swivel": 0.12},
            {"torso_participation": 1.22},
            {"guard": 1.06, "load": 1.08, "strike": 1.10, "follow_through": 1.12, "recover": 1.08},
            "Drive a broader strike from the torso with a deliberate follow-through.",
        ),
        CandidateRecipe(
            "technical_compact",
            {"lateral_offset": 0.14, "arm_height": -0.04, "path_arc": -0.08},
            {"arm_depth": 0.82, "elbow_swivel": 0.88, "torso_participation": 0.72},
            {"guard": 0.98, "load": 1.04, "strike": 1.02, "follow_through": 1.04, "recover": 1.12},
            "Keep the fist path compact and the elbow controlled for a technical boxing strike.",
        ),
        CandidateRecipe(
            "camera_readable_power",
            {"arm_height": 0.10, "arm_depth": 0.10, "path_arc": 0.08, "elbow_swivel": 0.05},
            {"torso_participation": 1.08},
            {"guard": 1.18, "load": 1.16, "strike": 1.16, "follow_through": 1.18, "recover": 1.16},
            "Slow and lift the striking arc so guard, impact, and recovery remain clear in first person.",
        ),
    ]


def candidate_recipes(
    parameters: PrimitiveParameters,
    intent: Intent = Intent.GESTURE,
) -> list[CandidateRecipe]:
    if intent == Intent.SEQUENCE:
        return [
            CandidateRecipe("canonical", {}, {}, {}, "Unmodified ordered action sequence."),
            CandidateRecipe(
                "brisk_sequence", {}, {},
                {phase.value: 0.90 for phase in PrimitiveKind},
                "Preserve every step while using brisk transitions and action timing.",
            ),
            CandidateRecipe(
                "measured_sequence", {}, {},
                {phase.value: 1.12 for phase in PrimitiveKind},
                "Preserve every step with measured timing for clearer action boundaries.",
            ),
            CandidateRecipe(
                "readable_sequence", {}, {},
                {phase.value: 1.26 for phase in PrimitiveKind},
                "Slow each requested action so the complete order remains visually readable.",
            ),
            CandidateRecipe(
                "punctuated_sequence", {}, {},
                {
                    **{phase.value: 1.04 for phase in PrimitiveKind},
                    "hold": 1.24,
                    "follow_through": 1.16,
                    "recover": 1.18,
                },
                "Add readable punctuation at held poses, follow-throughs, and recoveries.",
            ),
        ]
    if intent == Intent.GRAB:
        return grab_candidate_recipes(parameters)
    if intent == Intent.OBJECT_INTERACTION:
        return object_interaction_candidate_recipes()
    if intent == Intent.STRIKE:
        return strike_candidate_recipes(parameters)
    if intent == Intent.COMPOSITE:
        return composite_candidate_recipes(parameters)
    if intent == Intent.FULL_BODY:
        return [
            CandidateRecipe("canonical", {}, {}, {}, "Unmodified procedural body-motion baseline."),
            CandidateRecipe(
                "brisk_body",
                {},
                {},
                {"body": 0.92, "recover": 0.96},
                "Use brisk whole-body timing with a more energetic silhouette.",
                body_scales={"intensity": 1.12},
                pose_scale=1.04,
                pose_directional_scale=1.14,
            ),
            CandidateRecipe(
                "measured_body",
                {},
                {},
                {"body": 1.12, "recover": 1.08},
                "Use measured timing and restrained body lift for clearer support changes.",
                body_scales={"intensity": 0.88},
                pose_scale=0.92,
                pose_directional_scale=0.90,
            ),
            CandidateRecipe(
                "readable_body",
                {},
                {},
                {"body": 1.28, "recover": 1.16},
                "Slow and soften the full-body action so each support phase reads clearly.",
                body_scales={"intensity": 0.76},
                pose_scale=0.82,
                pose_directional_scale=0.80,
            ),
            CandidateRecipe(
                "expressive_body",
                {},
                {},
                {"body": 1.06, "recover": 1.24},
                "Use stronger lift and arm swing with a deliberate balanced settle.",
                body_scales={"intensity": 1.22},
                pose_scale=1.06,
                pose_directional_scale=1.28,
            ),
        ]
    # The first-generation flywheel mostly changed finger splay and elbow pole.
    # Those clips were technically different but looked effectively identical
    # to a human reviewer.  These recipes deliberately span pose, wrist, timing,
    # and silhouette while preserving the sign of every directional parameter.
    roll_direction = 1.0 if parameters.wrist_roll >= 0.20 else -1.0 if parameters.wrist_roll <= -0.20 else 0.0
    if parameters.duration_s <= 0.42:  # quick
        emphasized_timing = {"present": 0.82, "hold": 0.90, "recover": 0.88}
        expressive_timing = {"present": 0.76, "hold": 0.88, "recover": 0.82}
    elif parameters.duration_s >= 0.75:  # slow
        emphasized_timing = {"present": 1.18, "hold": 1.12, "recover": 1.18}
        expressive_timing = {"present": 1.10, "hold": 1.18, "recover": 1.12}
    else:  # balanced
        emphasized_timing = {"present": 0.94, "hold": 1.10, "recover": 0.98}
        expressive_timing = {"present": 0.88, "hold": 1.06, "recover": 0.92}
    return [
        CandidateRecipe(
            name="canonical",
            deltas={},
            scales={},
            duration_scales={},
            purpose="Unmodified semantic-plan baseline.",
        ),
        CandidateRecipe(
            name="semantic_emphasis",
            deltas={
                "finger_splay": 0.45,
                "thumb_curl": -0.22,
                "little_curl": -0.22,
                "path_arc": -0.45,
                "wrist_flourish": roll_direction * 0.30,
            },
            scales={
                "arm_height": 1.45,
                "arm_depth": 1.55,
                "lateral_offset": 1.45,
                "wrist_pitch": 1.45,
                "wrist_yaw": 1.45,
                "wrist_roll": 1.45,
                "elbow_swivel": 1.25,
                "torso_participation": 1.25,
                "wrist_shake_amplitude": 1.05,
            },
            duration_scales={**emphasized_timing, "shake": 0.95},
            purpose="Make the requested spatial, wrist, and timing categories visibly legible.",
        ),
        CandidateRecipe(
            name="grounded_natural",
            deltas={
                "finger_splay": 0.55,
                "thumb_curl": -0.30,
                "little_curl": -0.30,
                "path_arc": -0.80,
            },
            scales={
                "arm_height": 0.72,
                "arm_depth": 0.55,
                "lateral_offset": 0.68,
                "wrist_pitch": 0.45,
                "wrist_yaw": 0.45,
                "wrist_roll": 0.45,
                "elbow_swivel": 0.30,
                "torso_participation": 0.45,
                "wrist_shake_amplitude": 0.78,
            },
            duration_scales={"present": 1.14, "hold": 1.14, "shake": 1.12, "recover": 1.14},
            purpose="Offer a visibly softer, grounded staging with reduced joint extremes.",
        ),
        CandidateRecipe(
            name="expressive_arc",
            deltas={
                "finger_splay": 0.65,
                "thumb_curl": -0.35,
                "little_curl": -0.35,
                "path_arc": 1.00,
                "wrist_flourish": roll_direction * 0.55,
            },
            scales={
                "arm_height": 1.15,
                "arm_depth": 1.25,
                "lateral_offset": 1.20,
                "wrist_pitch": 1.18,
                "wrist_yaw": 1.18,
                "wrist_roll": 1.18,
                "elbow_swivel": 1.30,
                "torso_participation": 1.50,
                "wrist_shake_amplitude": 1.05,
            },
            duration_scales={**expressive_timing, "shake": 0.94},
            purpose="Use a distinct whole-arm arc and stronger style-appropriate body participation.",
        ),
        CandidateRecipe(
            name="camera_silhouette",
            deltas={
                "finger_splay": 0.90,
                "thumb_curl": -0.45,
                "little_curl": -0.45,
                "path_arc": 0.20,
                "wrist_flourish": -roll_direction * 0.35,
            },
            scales={
                "arm_height": 1.00,
                "arm_depth": 0.82,
                "lateral_offset": 1.05,
                "wrist_pitch": 0.70,
                "wrist_yaw": 0.70,
                "wrist_roll": 0.50,
                "elbow_swivel": 0.55,
                "wrist_shake_amplitude": 0.88,
            },
            duration_scales={"present": 1.05, "hold": 1.22, "shake": 1.05, "recover": 1.05},
            purpose="Present a wide, camera-facing hand silhouette with a clearly readable hold.",
        ),
    ]


def candidate_recipe_pool(
    parameters: PrimitiveParameters,
    intent: Intent = Intent.GESTURE,
) -> list[CandidateRecipe]:
    """Primary recipes plus bounded, globally safe replenishment candidates."""
    primary = candidate_recipes(parameters, intent)
    if intent == Intent.SEQUENCE:
        return [*primary, *adaptive_timing_recipes()]
    if intent == Intent.GRAB:
        return [*primary, *adaptive_timing_recipes()]
    if intent == Intent.STRIKE:
        return [
            *primary,
            CandidateRecipe(
                "safe_centered_arc",
                {"lateral_offset": 0.18, "arm_depth": -0.04, "path_arc": -0.12},
                {"torso_participation": 0.68, "elbow_swivel": 0.82},
                {"guard": 1.20, "load": 1.18, "strike": 1.18, "follow_through": 1.20, "recover": 1.20},
                "Use a slow centered arc with additional joint and torso margin.",
            ),
            CandidateRecipe(
                "high_guard_recovery",
                {"arm_height": 0.14, "lateral_offset": 0.10, "path_arc": 0.04},
                {"torso_participation": 0.78},
                {"guard": 1.24, "load": 1.12, "strike": 1.10, "follow_through": 1.12, "recover": 1.30},
                "Emphasize a high guard and clean, fully visible recovery.",
            ),
            CandidateRecipe(
                "measured_power",
                {"arm_depth": 0.04, "elbow_swivel": 0.10, "path_arc": 0.12},
                {"torso_participation": 1.12},
                {"guard": 1.14, "load": 1.20, "strike": 1.22, "follow_through": 1.24, "recover": 1.18},
                "Use measured timing and torso rotation while preserving the requested strike path.",
            ),
            *adaptive_timing_recipes(),
        ]
    if intent == Intent.COMPOSITE:
        return [*primary, *adaptive_timing_recipes()]
    if intent == Intent.FULL_BODY:
        return [*primary, *adaptive_timing_recipes()]
    roll_direction = (
        1.0
        if parameters.wrist_roll >= 0.20
        else -1.0
        if parameters.wrist_roll <= -0.20
        else 0.0
    )
    return [
        *primary,
        CandidateRecipe(
            name="centered_elbow",
            deltas={
                "finger_splay": 0.48,
                "thumb_curl": -0.22,
                "little_curl": -0.22,
                "path_arc": 0.45,
            },
            scales={
                "elbow_swivel": 0.35,
                "wrist_pitch": 0.75,
                "wrist_yaw": 0.75,
                "wrist_roll": 0.72,
            },
            duration_scales={"present": 1.12, "hold": 1.15, "recover": 1.12},
            purpose="Center the elbow and soften the wrist with a distinctly staged, wider silhouette.",
        ),
        CandidateRecipe(
            name="compact_readable",
            deltas={
                "finger_splay": 0.52,
                "thumb_curl": -0.24,
                "little_curl": -0.24,
                "path_arc": 0.30,
            },
            scales={"arm_depth": 0.86, "lateral_offset": 0.86, "wrist_roll": 0.72},
            duration_scales={"hold": 1.08},
            purpose="Reduce collision risk while keeping a wide hand silhouette.",
        ),
        CandidateRecipe(
            name="soft_staging",
            deltas={
                "finger_splay": 0.42,
                "elbow_swivel": -0.20,
                "thumb_curl": -0.25,
                "little_curl": -0.25,
                "path_arc": 0.70,
            },
            scales={
                "arm_height": 0.90,
                "arm_depth": 0.90,
                "lateral_offset": 0.90,
                "wrist_pitch": 0.75,
                "wrist_yaw": 0.75,
                "wrist_roll": 0.75,
            },
            duration_scales={"present": 1.18, "hold": 1.18, "recover": 1.18},
            purpose="Use slower, gentler staging and reduced joint extremes as a distinct safe replacement.",
        ),
        CandidateRecipe(
            name="torso_clearance",
            deltas={
                "elbow_swivel": -0.24,
                "torso_participation": -0.10,
                "finger_splay": 0.36,
                "path_arc": 0.40,
            },
            scales={"wrist_roll": 0.76},
            duration_scales={"present": 1.10, "recover": 1.10},
            purpose="Move an inward elbow away from the torso while preserving hand directions.",
        ),
        CandidateRecipe(
            name="inward_clearance",
            deltas={"elbow_swivel": -0.20, "finger_splay": 0.44, "path_arc": 0.50},
            scales={
                "arm_depth": 0.92,
                "lateral_offset": 0.82,
                "wrist_pitch": 0.86,
                "wrist_yaw": 0.86,
                "wrist_roll": 0.72,
            },
            duration_scales={"present": 1.12, "hold": 1.05, "recover": 1.12},
            purpose="Create torso clearance for low inward reaches without reversing depth or lateral intent.",
        ),
        CandidateRecipe(
            name="deliberate_underhand",
            deltas={
                "finger_splay": 0.62,
                "thumb_curl": -0.30,
                "little_curl": -0.30,
                "path_arc": -1.0,
                "wrist_flourish": -roll_direction * 0.55,
            },
            scales={
                "arm_height": 0.82,
                "arm_depth": 0.78,
                "lateral_offset": 0.72,
                "wrist_pitch": 0.55,
                "wrist_yaw": 0.55,
                "wrist_roll": 0.55,
                "elbow_swivel": -0.30,
                "torso_participation": 0.35,
            },
            duration_scales={"present": 1.55, "hold": 1.28, "recover": 1.42},
            purpose="Use a slow underhand presentation with the elbow cleared away from the torso.",
        ),
        CandidateRecipe(
            name="deliberate_overhand",
            deltas={
                "finger_splay": 0.72,
                "thumb_curl": -0.36,
                "little_curl": -0.36,
                "path_arc": 1.0,
                "wrist_flourish": roll_direction * 0.70,
            },
            scales={
                "arm_height": 0.92,
                "arm_depth": 0.64,
                "lateral_offset": 0.58,
                "wrist_pitch": 0.72,
                "wrist_yaw": 0.72,
                "wrist_roll": 0.68,
                "elbow_swivel": -0.08,
                "torso_participation": 0.25,
            },
            duration_scales={"present": 1.90, "hold": 1.02, "recover": 1.62},
            purpose="Use a distinct overhand sweep and compact cleared elbow before the hold.",
        ),
        *adaptive_timing_recipes(),
    ]


def _semantic_value(original: float, candidate: float) -> float:
    # Directional prompt parameters may change magnitude but never sign.
    if abs(original) >= 0.20 and original * candidate < 0.0:
        return 0.05 if original > 0.0 else -0.05
    return candidate


def _apply_values(
    parameters: PrimitiveParameters,
    deltas: dict[str, float],
    scales: dict[str, float],
    duration_scale: float,
) -> PrimitiveParameters:
    bounds = {
        "arm_height": (-1.0, 1.0),
        "arm_depth": (-1.0, 1.0),
        "lateral_offset": (-1.0, 1.0),
        "wrist_pitch": (-1.0, 1.0),
        "wrist_yaw": (-1.0, 1.0),
        "wrist_roll": (-1.0, 1.0),
        "elbow_swivel": (-1.0, 1.0),
        "torso_participation": (0.0, 1.0),
        "path_arc": (-1.0, 1.0),
        "wrist_flourish": (-1.0, 1.0),
        "wrist_shake_amplitude": (0.0, 1.0),
        "wrist_shake_cycles": (0.0, 6.0),
        "finger_splay": (-1.0, 1.0),
        "thumb_curl": (-1.0, 1.0),
        "little_curl": (-1.0, 1.0),
        "finger_curl": (-1.0, 1.0),
        "thumb_opposition": (0.0, 1.0),
        "grip_force": (0.0, 1.0),
        "lift_height_m": (0.0, 0.80),
        "trajectory_amplitude_m": (0.0, 0.20),
        "trajectory_cycles": (0.0, 8.0),
        "axial_rotation_amplitude": (0.0, 1.0),
    }
    updates: dict[str, float] = {
        "duration_s": _clamp(parameters.duration_s * duration_scale, 0.05, 4.0)
    }
    for name in set(deltas) | set(scales):
        original = float(getattr(parameters, name))
        candidate = original * scales.get(name, 1.0) + deltas.get(name, 0.0)
        if name in {"arm_height", "arm_depth", "lateral_offset", "wrist_pitch", "wrist_yaw", "wrist_roll"}:
            candidate = _semantic_value(original, candidate)
        minimum, maximum = bounds[name]
        updates[name] = _clamp(candidate, minimum, maximum)
    return parameters.model_copy(update=updates)


def _scaled_pose(pose: Any, scale: float, directional_scale: float) -> Any:
    """Scale an authored pose without changing any requested direction."""

    bounds = {
        "root_drop_m": (0.0, 0.85),
        "root_shift_x_m": (-0.45, 0.45),
        "root_shift_z_m": (-0.45, 0.45),
        "pelvis_pitch_deg": (-105.0, 105.0),
        "pelvis_yaw_deg": (-90.0, 90.0),
        "pelvis_roll_deg": (-60.0, 60.0),
        "torso_pitch_deg": (-100.0, 100.0),
        "torso_yaw_deg": (-100.0, 100.0),
        "torso_roll_deg": (-75.0, 75.0),
        "left_hip_pitch_deg": (-120.0, 120.0),
        "right_hip_pitch_deg": (-120.0, 120.0),
        "left_hip_roll_deg": (-55.0, 55.0),
        "right_hip_roll_deg": (-55.0, 55.0),
        "left_knee_flexion_deg": (0.0, 145.0),
        "right_knee_flexion_deg": (0.0, 145.0),
        "left_ankle_pitch_deg": (-65.0, 65.0),
        "right_ankle_pitch_deg": (-65.0, 65.0),
        "left_foot_shift_x_m": (-0.45, 0.45),
        "left_foot_shift_z_m": (-0.55, 0.55),
        "right_foot_shift_x_m": (-0.45, 0.45),
        "right_foot_shift_z_m": (-0.55, 0.55),
    }
    mechanically_limited = {
        "root_drop_m",
        "left_foot_shift_x_m",
        "left_foot_shift_z_m",
        "right_foot_shift_x_m",
        "right_foot_shift_z_m",
    }
    updates = {
        name: _clamp(
            float(getattr(pose, name))
            * (scale if name in mechanically_limited else directional_scale),
            minimum,
            maximum,
        )
        for name, (minimum, maximum) in bounds.items()
    }
    return pose.model_copy(update=updates)


def apply_recipe(program: MotionProgram, recipe: CandidateRecipe, *, seed_offset: int) -> MotionProgram:
    candidate = program.model_copy(deep=True)
    candidate.seed = (candidate.seed + seed_offset) % (2**31 - 1)
    if candidate.intent == Intent.SEQUENCE:
        candidate.steps = [
            apply_recipe(
                step,
                recipe,
                seed_offset=seed_offset * 17 + index,
            )
            for index, step in enumerate(candidate.steps, start=1)
        ]
        return candidate
    for primitive in candidate.primitives:
        travel_signal_segment = primitive.label in {
            "parallel_forearm_travel_setup",
            "parallel_forearm_travel_cycle",
        }
        travel_signal_effectors = (
            [effector.model_copy(deep=True) for effector in primitive.effectors]
            if travel_signal_segment
            else []
        )
        travel_signal_amplitude = primitive.parameters.trajectory_amplitude_m
        phase = primitive.kind.value
        if primitive.kind != PrimitiveKind.RECOVER:
            phase_deltas = {
                key: value
                for key, value in recipe.deltas.items()
                if primitive.kind == PrimitiveKind.SHAKE or not key.startswith("wrist_shake_")
            }
            phase_scales = {
                key: value
                for key, value in recipe.scales.items()
                if primitive.kind == PrimitiveKind.SHAKE or not key.startswith("wrist_shake_")
            }
            primitive.parameters = _apply_values(
                primitive.parameters,
                phase_deltas,
                phase_scales,
                recipe.duration_scales.get(phase, 1.0),
            )
            if primitive.effectors and (recipe.effector_deltas or recipe.effector_scales):
                next_effectors = []
                for effector in primitive.effectors:
                    updates: dict[str, float] = {}
                    for name in set(recipe.effector_deltas) | set(recipe.effector_scales):
                        original = float(getattr(effector, name))
                        candidate_value = (
                            original * recipe.effector_scales.get(name, 1.0)
                            + recipe.effector_deltas.get(name, 0.0)
                        )
                        if name in {"target_x", "target_y", "target_z", "elbow_swivel", "wrist_pitch", "wrist_yaw", "wrist_roll"}:
                            candidate_value = _clamp(candidate_value, -1.0, 1.0)
                        updates[name] = candidate_value
                    next_effectors.append(effector.model_copy(update=updates))
                primitive.effectors = next_effectors
            if travel_signal_segment:
                # The travel signal's bilateral staging is a coupled physical
                # constraint, not a stylistic parameter. Generic per-effector
                # scaling can otherwise pull the elbows into a crossed pose.
                primitive.effectors = [
                    candidate_effector.model_copy(
                        update={
                            "target_x": stable_effector.target_x,
                            "target_y": stable_effector.target_y,
                            "target_z": stable_effector.target_z,
                            "elbow_swivel": stable_effector.elbow_swivel,
                        }
                    )
                    for stable_effector, candidate_effector in zip(
                        travel_signal_effectors,
                        primitive.effectors,
                        strict=False,
                    )
                ]
                primitive.parameters = primitive.parameters.model_copy(
                    update={
                        "trajectory_amplitude_m": (
                            float(
                                np.clip(
                                    primitive.parameters.trajectory_amplitude_m,
                                    0.080,
                                    0.100,
                                )
                            )
                            if primitive.label == "parallel_forearm_travel_cycle"
                            else travel_signal_amplitude
                        )
                    }
                )
            if primitive.body is not None and (recipe.body_deltas or recipe.body_scales):
                body_updates: dict[str, float] = {}
                body_bounds = {
                    "intensity": (0.0, 1.0),
                    "height_m": (0.0, 0.65),
                }
                for name in set(recipe.body_deltas) | set(recipe.body_scales):
                    original = float(getattr(primitive.body, name))
                    body_updates[name] = _clamp(
                        original * recipe.body_scales.get(name, 1.0)
                        + recipe.body_deltas.get(name, 0.0),
                        *body_bounds[name],
                    )
                primitive.body = primitive.body.model_copy(update=body_updates)
            if (
                primitive.body is not None
                and primitive.body.action == BodyAction.POSE
                and (
                    abs(recipe.pose_scale - 1.0) > 1e-8
                    or abs(recipe.pose_directional_scale - 1.0) > 1e-8
                )
            ):
                primitive.body = primitive.body.model_copy(
                    update={
                        "pose": _scaled_pose(
                            primitive.body.pose,
                            recipe.pose_scale,
                            recipe.pose_directional_scale,
                        )
                    }
                )
        elif primitive.kind == PrimitiveKind.RECOVER:
            primitive.parameters = primitive.parameters.model_copy(
                update={
                    "duration_s": _clamp(
                        primitive.parameters.duration_s * recipe.duration_scales.get(phase, 1.0),
                        0.05,
                        4.0,
                    )
                }
            )
    if candidate.object_motion is not None and (
        recipe.object_deltas or recipe.object_scales
    ):
        object_bounds = {
            "distance_m": (0.08, 2.50),
            "apex_height_m": (0.05, 1.20),
            "spin_turns": (-3.0, 3.0),
            "contact_height_m": (0.85, 1.65),
            "contact_depth_m": (0.18, 0.58),
            "landing_height_m": (0.02, 1.30),
        }
        updates: dict[str, float] = {}
        for name in set(recipe.object_deltas) | set(recipe.object_scales):
            original = float(getattr(candidate.object_motion, name))
            updates[name] = _clamp(
                original * recipe.object_scales.get(name, 1.0)
                + recipe.object_deltas.get(name, 0.0),
                *object_bounds[name],
            )
        candidate.object_motion = candidate.object_motion.model_copy(update=updates)
    return candidate


def describe_program_delta(base: MotionProgram, candidate: MotionProgram) -> dict[str, Any]:
    """Record whether a candidate is materially different in authoring space."""
    if base.intent == Intent.SEQUENCE and candidate.intent == Intent.SEQUENCE:
        step_deltas = [
            describe_program_delta(base_step, candidate_step)
            for base_step, candidate_step in zip(
                base.steps, candidate.steps, strict=False
            )
        ]
        changed = [
            f"step_{index}_{name}"
            for index, delta in enumerate(step_deltas, start=1)
            for name in delta.get("materially_changed_dimensions", [])
        ]
        return {
            "sequence_steps": step_deltas,
            "parameter_deltas": {},
            "effector_deltas": {},
            "shake_parameter_deltas": {},
            "phase_duration_deltas_s": {},
            "body_parameter_deltas": {},
            "physical_pose_deltas": {},
            "materially_changed_dimensions": changed,
            "materially_changed_dimension_count": len(changed),
        }
    base_parameters = base.primitives[0].parameters
    candidate_parameters = candidate.primitives[0].parameters
    thresholds = {
        "duration_s": 0.06,
        "arm_height": 0.10,
        "arm_depth": 0.10,
        "lateral_offset": 0.10,
        "wrist_pitch": 0.10,
        "wrist_yaw": 0.10,
        "wrist_roll": 0.10,
        "elbow_swivel": 0.15,
        "torso_participation": 0.08,
        "finger_splay": 0.20,
        "thumb_curl": 0.15,
        "little_curl": 0.15,
        "path_arc": 0.20,
        "wrist_flourish": 0.20,
        "wrist_shake_amplitude": 0.15,
        "wrist_shake_cycles": 0.50,
        "trajectory_amplitude_m": 0.015,
    }
    deltas = {
        name: max(
            (
                abs(
                    float(getattr(candidate_primitive.parameters, name))
                    - float(getattr(base_primitive.parameters, name))
                )
                for base_primitive, candidate_primitive in zip(
                    base.primitives, candidate.primitives, strict=False
                )
            ),
            default=0.0,
        )
        for name in thresholds
    }
    effector_deltas = {
        name: max(
            (
                abs(float(candidate_effector_value) - float(base_effector_value))
                for base_primitive, candidate_primitive in zip(
                    base.primitives, candidate.primitives, strict=False
                )
                for base_effector, candidate_effector in zip(
                    base_primitive.effectors, candidate_primitive.effectors, strict=False
                )
                for base_effector_value, candidate_effector_value in (
                    (
                        getattr(base_effector, name),
                        getattr(candidate_effector, name),
                    ),
                )
            ),
            default=0.0,
        )
        for name in ("target_x", "target_y", "target_z", "elbow_swivel")
    }
    phase_duration_deltas = {
        base_primitive.kind.value: abs(
            float(candidate_primitive.parameters.duration_s)
            - float(base_primitive.parameters.duration_s)
        )
        for base_primitive, candidate_primitive in zip(base.primitives, candidate.primitives)
    }
    body_parameter_deltas = {
        name: max(
            (
                abs(float(getattr(candidate_primitive.body, name)) - float(getattr(base_primitive.body, name)))
                for base_primitive, candidate_primitive in zip(
                    base.primitives, candidate.primitives, strict=False
                )
                if base_primitive.body is not None and candidate_primitive.body is not None
            ),
            default=0.0,
        )
        for name in ("intensity", "height_m")
    }
    base_shake = next(
        (item.parameters for item in base.primitives if item.kind == PrimitiveKind.SHAKE),
        None,
    )
    candidate_shake = next(
        (item.parameters for item in candidate.primitives if item.kind == PrimitiveKind.SHAKE),
        None,
    )
    shake_parameter_deltas = {
        name: abs(float(getattr(candidate_shake, name)) - float(getattr(base_shake, name)))
        if base_shake is not None and candidate_shake is not None
        else 0.0
        for name in ("wrist_shake_amplitude", "wrist_shake_cycles")
    }
    changed = [name for name, delta in deltas.items() if delta >= thresholds[name]]
    changed.extend(
        f"effector_{name}"
        for name, delta in effector_deltas.items()
        if delta >= (0.08 if name.startswith("target_") else 0.12)
    )
    for name, delta in shake_parameter_deltas.items():
        if delta >= thresholds[name]:
            changed.append(f"shake_{name}")
    for phase, delta in phase_duration_deltas.items():
        if delta >= thresholds["duration_s"] and f"{phase}_duration_s" not in changed:
            changed.append(f"{phase}_duration_s")
    if body_parameter_deltas["intensity"] >= 0.06:
        changed.append("body_intensity")
    if body_parameter_deltas["height_m"] >= 0.03:
        changed.append("body_height_m")
    return {
        "parameter_deltas": deltas,
        "effector_deltas": effector_deltas,
        "shake_parameter_deltas": shake_parameter_deltas,
        "phase_duration_deltas_s": phase_duration_deltas,
        "body_parameter_deltas": body_parameter_deltas,
        "physical_pose_deltas": {
            "wrist_height_m": deltas["arm_height"] * 0.18,
            "wrist_depth_m": deltas["arm_depth"] * 0.12,
            "wrist_lateral_m": deltas["lateral_offset"] * 0.12,
            "wrist_pitch_rad": deltas["wrist_pitch"] * 0.35,
            "wrist_yaw_rad": deltas["wrist_yaw"] * 0.28,
            "forearm_roll_rad": deltas["wrist_roll"] * 0.40,
            "presentation_arc_upper_arm_rad": deltas["path_arc"]
            * presentation_arc_amplitude_rad(base_parameters.duration_s),
            "transient_forearm_roll_rad": deltas["wrist_flourish"]
            * wrist_flourish_amplitude_rad(base_parameters.duration_s),
        },
        "materially_changed_dimensions": changed,
        "materially_changed_dimension_count": len(changed),
    }


def apply_repair(program: MotionProgram, repair: RepairPatch) -> MotionProgram:
    patched = program.model_copy(deep=True)
    if patched.intent == Intent.SEQUENCE:
        patched.steps = [apply_repair(step, repair) for step in patched.steps]
        return patched
    deltas = {
        key.removesuffix("_delta"): value
        for key, value in repair.model_dump().items()
        if key.endswith("_delta")
        and key != "easing_delta"
        and not key.startswith("object_")
    }
    duration_scales = {
        "present": repair.present_duration_scale,
        "guard": repair.present_duration_scale,
        "load": repair.present_duration_scale,
        "reach": repair.present_duration_scale,
        "preshape": repair.present_duration_scale,
        "move": repair.present_duration_scale,
        "body": repair.present_duration_scale,
        "hold": repair.hold_duration_scale,
        "strike": repair.hold_duration_scale,
        "follow_through": repair.hold_duration_scale,
        "contact": repair.hold_duration_scale,
        "close": repair.hold_duration_scale,
        "shake": repair.shake_duration_scale,
        "lift": repair.shake_duration_scale,
        "cycle": repair.shake_duration_scale,
        "windup": repair.present_duration_scale,
        "release": repair.hold_duration_scale,
        "flight": repair.shake_duration_scale,
        "receive": repair.present_duration_scale,
        "absorb": repair.hold_duration_scale,
        "recover": repair.recover_duration_scale,
    }
    for primitive in patched.primitives:
        if primitive.kind != PrimitiveKind.RECOVER:
            phase_deltas = {
                key: value
                for key, value in deltas.items()
                if primitive.kind == PrimitiveKind.SHAKE or not key.startswith("wrist_shake_")
            }
            parameters = _apply_values(
                primitive.parameters,
                phase_deltas,
                {},
                duration_scales.get(primitive.kind.value, 1.0),
            )
            primitive.parameters = parameters.model_copy(
                update={"easing": _clamp(parameters.easing + repair.easing_delta, 0.0, 1.0)}
            )
            if patched.intent == Intent.COMPOSITE and primitive.effectors:
                primitive.effectors = [
                    effector.model_copy(
                        update={
                            "target_x": _clamp(
                                effector.target_x
                                + (1.0 if effector.target_x >= 0.0 else -1.0)
                                * repair.lateral_offset_delta,
                                -1.0,
                                1.0,
                            ),
                            "target_y": _clamp(
                                effector.target_y + repair.arm_height_delta,
                                -1.0,
                                1.0,
                            ),
                            "target_z": _clamp(
                                effector.target_z + repair.arm_depth_delta,
                                -1.0,
                                1.0,
                            ),
                            "elbow_swivel": _clamp(
                                effector.elbow_swivel + repair.elbow_swivel_delta,
                                -1.0,
                                1.0,
                            ),
                            "wrist_pitch": _clamp(
                                effector.wrist_pitch + repair.wrist_pitch_delta,
                                -1.0,
                                1.0,
                            ),
                            "wrist_yaw": _clamp(
                                effector.wrist_yaw + repair.wrist_yaw_delta,
                                -1.0,
                                1.0,
                            ),
                            "wrist_roll": _clamp(
                                effector.wrist_roll + repair.wrist_roll_delta,
                                -1.0,
                                1.0,
                            ),
                        }
                    )
                    for effector in primitive.effectors
                ]
            if primitive.body is not None and primitive.body.action == BodyAction.POSE:
                primitive.body = primitive.body.model_copy(
                    update={
                        "pose": _scaled_pose(
                            primitive.body.pose,
                            repair.pose_root_scale,
                            repair.pose_directional_scale,
                        )
                    }
                )
        elif primitive.kind == PrimitiveKind.RECOVER:
            primitive.parameters = primitive.parameters.model_copy(
                update={
                    "duration_s": _clamp(
                        primitive.parameters.duration_s * duration_scales["recover"], 0.05, 4.0
                    ),
                    "easing": _clamp(
                        primitive.parameters.easing + repair.easing_delta, 0.0, 1.0
                    ),
                }
            )
    if patched.object_motion is not None:
        patched.object_motion = patched.object_motion.model_copy(
            update={
                "distance_m": _clamp(
                    patched.object_motion.distance_m * repair.object_distance_scale,
                    0.08,
                    2.50,
                ),
                "apex_height_m": _clamp(
                    patched.object_motion.apex_height_m * repair.object_apex_scale,
                    0.05,
                    1.20,
                ),
                "contact_height_m": _clamp(
                    patched.object_motion.contact_height_m
                    + repair.object_contact_height_delta,
                    0.85,
                    1.65,
                ),
                "contact_depth_m": _clamp(
                    patched.object_motion.contact_depth_m
                    + repair.object_contact_depth_delta,
                    0.18,
                    0.58,
                ),
            }
        )
    return patched


def apply_prefilter_safety_repair(program: MotionProgram) -> MotionProgram:
    """Create a conservative next-round base without requiring a VLM call."""

    phase_names = tuple(kind.value for kind in PrimitiveKind)
    recipe = CandidateRecipe(
        name="deterministic_prefilter_safety",
        deltas={},
        scales={
            "wrist_pitch": 0.82,
            "wrist_yaw": 0.82,
            "wrist_roll": 0.82,
            "elbow_swivel": 0.86,
            "torso_participation": 0.86,
            "path_arc": 0.88,
            "wrist_flourish": 0.80,
            "wrist_shake_amplitude": 0.82,
        },
        duration_scales={phase: 1.22 for phase in phase_names},
        purpose=(
            "Reduce joint extremes and kinematic stress while preserving the planned semantic motion."
        ),
    )
    return apply_recipe(program, recipe, seed_offset=901)


def _motion_hash(clip: ClipResult) -> str:
    payload = json.dumps(
        [frame.model_dump(mode="json") for frame in clip.frames],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def motion_perceptual_descriptor(
    clip: ClipResult,
    hand: Hand | list[Hand],
) -> dict[str, Any]:
    """Summarize rendered arm and whole-body paths at phase-aware points."""
    active_hands = hand if isinstance(hand, list) else [hand]
    primary_hand = active_hands[0]
    ranges = {
        str(item["kind"]): (float(item["start_s"]), float(item["end_s"]))
        for item in clip.metrics.get("phase_ranges_s", [])
        if isinstance(item, dict)
        and item.get("kind") in {
            "present", "shake", "reach", "preshape", "contact", "close",
            "lift", "hold", "recover", "guard", "load", "strike", "follow_through",
            "move", "cycle", "body", "windup", "release", "flight", "receive", "absorb",
        }
        and float(item.get("end_s", 0.0)) > float(item.get("start_s", 0.0))
    }
    points: list[tuple[str, float]] = [
        ("present", 0.30),
        ("present", 0.50),
        ("present", 0.70),
        ("reach", 0.25),
        ("reach", 0.50),
        ("reach", 0.75),
        ("preshape", 0.50),
        ("contact", 0.50),
        ("close", 0.50),
        ("lift", 0.50),
        ("hold", 0.50),
        ("windup", 0.50),
        ("release", 0.50),
        ("flight", 0.15),
        ("flight", 0.35),
        ("flight", 0.55),
        ("flight", 0.75),
        ("flight", 0.95),
        ("receive", 0.50),
        ("absorb", 0.50),
        ("guard", 0.65),
        ("load", 0.50),
        ("strike", 0.25),
        ("strike", 0.50),
        ("strike", 0.75),
        ("follow_through", 0.50),
        ("move", 0.25),
        ("move", 0.50),
        ("move", 0.75),
        ("cycle", 0.10),
        ("cycle", 0.25),
        ("cycle", 0.50),
        ("cycle", 0.75),
        ("cycle", 0.90),
        ("body", 0.10),
        ("body", 0.25),
        ("body", 0.50),
        ("body", 0.75),
        ("body", 0.90),
    ]
    shake_cycles = float(clip.metrics.get("wrist_shake_cycles", 0.0) or 0.0)
    if "shake" in ranges and shake_cycles > 0.0:
        half_cycles = max(1, int(round(shake_cycles * 2.0)))
        points.extend(
            ("shake", (2 * index + 1) / (4.0 * shake_cycles))
            for index in range(half_cycles)
        )
    if "recover" in ranges:
        points.extend((("recover", 0.50), ("recover", 0.95)))
    samples: list[dict[str, Any]] = []
    for active_hand in active_hands:
        for phase, fraction in points:
            if phase not in ranges or not clip.frames:
                continue
            start, end = ranges[phase]
            target_time = start + (end - start) * fraction
            frame = min(clip.frames, key=lambda item: abs(float(item.time_s) - target_time))
            # Trunk-relative despite the `elbow_world_m` key below: `arm_landmarks`
            # rebuilds the arm on a fixed rest shoulder. Left as-is deliberately --
            # this is eval telemetry compared against itself, and moving it would
            # invalidate every stored descriptor without changing any check.
            _, elbow, wrist, hand_world = arm_landmarks(frame, active_hand)
            samples.append(
                {
                    "hand": active_hand.value,
                    "phase": phase,
                    "fraction": fraction,
                    "time_s": float(frame.time_s),
                    "elbow_world_m": [float(value) for value in elbow],
                    "wrist_world_m": [float(value) for value in wrist],
                    "hand_world_xyzw": [float(value) for value in hand_world.as_quat()],
                }
            )
    body_samples: list[dict[str, Any]] = []
    if "body" in ranges and clip.frames:
        body_start, body_end = ranges["body"]
        kinematics = rig_kinematics()
        for fraction in np.linspace(0.05, 0.95, 19):
            target_time = body_start + (body_end - body_start) * float(fraction)
            frame = min(clip.frames, key=lambda item: abs(float(item.time_s) - target_time))
            positions = kinematics.canonical_positions(frame.bones)
            body_samples.append(
                {
                    "fraction": float(fraction),
                    "time_s": float(frame.time_s),
                    "hips_world_m": [float(value) for value in positions["hips"]],
                    "left_knee_world_m": [
                        float(value) for value in positions["leftLowerLeg"]
                    ],
                    "right_knee_world_m": [
                        float(value) for value in positions["rightLowerLeg"]
                    ],
                    "left_ankle_world_m": [
                        float(value) for value in positions["leftFoot"]
                    ],
                    "right_ankle_world_m": [
                        float(value) for value in positions["rightFoot"]
                    ],
                }
            )
    object_samples: list[dict[str, Any]] = []
    if "flight" in ranges and clip.frames and clip.frames[0].objects:
        flight_start, flight_end = ranges["flight"]
        object_id = next(iter(clip.frames[0].objects))
        for fraction in (0.10, 0.30, 0.50, 0.70, 0.90):
            target_time = flight_start + (flight_end - flight_start) * fraction
            frame = min(clip.frames, key=lambda item: abs(float(item.time_s) - target_time))
            transform = frame.objects[object_id]
            object_samples.append(
                {
                    "fraction": fraction,
                    "time_s": float(frame.time_s),
                    "object_id": object_id,
                    "position_world_m": transform.translation.as_list(),
                    "rotation_world_xyzw": transform.rotation.as_list(),
                }
            )
    shake_amplitude = 0.0
    shake_reversals = 0
    if "shake" in ranges and clip.frames:
        shake_start, shake_end = ranges["shake"]
        preceding = [frame for frame in clip.frames if float(frame.time_s) <= shake_start + 1e-8]
        baseline_frame = preceding[-1] if preceding else clip.frames[0]
        forearm_bone = f"{primary_hand.value}LowerArm"
        baseline_rotation = Rotation.from_quat(
            baseline_frame.bones[forearm_bone].rotation.as_list()
        )
        signed_deviations: list[float] = []
        for frame in clip.frames:
            if shake_start - 1e-8 <= float(frame.time_s) <= shake_end + 1e-8:
                current = Rotation.from_quat(frame.bones[forearm_bone].rotation.as_list())
                signed_deviations.append(
                    float((baseline_rotation.inv() * current).as_rotvec()[1])
                )
        shake_amplitude = max((abs(value) for value in signed_deviations), default=0.0)
        signs = [int(np.sign(value)) for value in signed_deviations if abs(value) >= 0.015]
        shake_reversals = sum(first != second for first, second in zip(signs, signs[1:]))

    recovery_wrist_error_m = 0.0
    recovery_hand_error_rad = 0.0
    if clip.frames:
        for active_hand in active_hands:
            _, _, first_wrist, first_hand = arm_landmarks(clip.frames[0], active_hand)
            _, _, final_wrist, final_hand = arm_landmarks(clip.frames[-1], active_hand)
            recovery_wrist_error_m = max(
                recovery_wrist_error_m,
                float(np.linalg.norm(final_wrist - first_wrist)),
            )
            recovery_hand_error_rad = max(
                recovery_hand_error_rad,
                float((first_hand.inv() * final_hand).magnitude()),
            )

    phase_durations = {
        phase: end - start for phase, (start, end) in ranges.items()
    }
    return {
        "duration_s": float(clip.duration_s),
        "active_hands": [active_hand.value for active_hand in active_hands],
        "phase_durations_s": phase_durations,
        "shake_cycles": shake_cycles,
        "shake_amplitude_rad": shake_amplitude,
        "shake_reversal_count": shake_reversals,
        "recovery_endpoint_wrist_error_m": recovery_wrist_error_m,
        "recovery_endpoint_hand_error_rad": recovery_hand_error_rad,
        "samples": samples,
        "body_samples": body_samples,
        "object_samples": object_samples,
    }


def compare_perceptual_descriptors(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    """Return interpretable path differences and the diversity decision."""
    pairs = list(zip(first.get("samples", []), second.get("samples", [])))
    wrist = [
        float(
            np.linalg.norm(
                np.asarray(a["wrist_world_m"], dtype=float)
                - np.asarray(b["wrist_world_m"], dtype=float)
            )
        )
        for a, b in pairs
    ]
    elbow = [
        float(
            np.linalg.norm(
                np.asarray(a["elbow_world_m"], dtype=float)
                - np.asarray(b["elbow_world_m"], dtype=float)
            )
        )
        for a, b in pairs
    ]
    orientation = [
        float(
            (
                Rotation.from_quat(a["hand_world_xyzw"]).inv()
                * Rotation.from_quat(b["hand_world_xyzw"])
            ).magnitude()
        )
        for a, b in pairs
    ]
    body_pairs = list(zip(first.get("body_samples", []), second.get("body_samples", [])))
    knee_separations = [
        max(
            float(
                np.linalg.norm(
                    np.asarray(a[f"{side}_knee_world_m"], dtype=float)
                    - np.asarray(b[f"{side}_knee_world_m"], dtype=float)
                )
            )
            for side in ("left", "right")
        )
        for a, b in body_pairs
    ]
    ankle_separations = [
        max(
            float(
                np.linalg.norm(
                    np.asarray(a[f"{side}_ankle_world_m"], dtype=float)
                    - np.asarray(b[f"{side}_ankle_world_m"], dtype=float)
                )
            )
            for side in ("left", "right")
        )
        for a, b in body_pairs
    ]
    root_vertical_separations = [
        abs(float(a["hips_world_m"][1]) - float(b["hips_world_m"][1]))
        for a, b in body_pairs
    ]
    object_pairs = list(zip(first.get("object_samples", []), second.get("object_samples", [])))
    object_separations = [
        float(
            np.linalg.norm(
                np.asarray(a["position_world_m"], dtype=float)
                - np.asarray(b["position_world_m"], dtype=float)
            )
        )
        for a, b in object_pairs
    ]
    measured = {
        "maximum_wrist_separation_m": max(wrist, default=0.0),
        "maximum_elbow_separation_m": max(elbow, default=0.0),
        "maximum_hand_orientation_separation_rad": max(orientation, default=0.0),
        "duration_separation_s": abs(
            float(first.get("duration_s", 0.0)) - float(second.get("duration_s", 0.0))
        ),
        "shake_amplitude_separation_rad": abs(
            float(first.get("shake_amplitude_rad", 0.0))
            - float(second.get("shake_amplitude_rad", 0.0))
        ),
        "shake_duration_separation_s": abs(
            float(first.get("phase_durations_s", {}).get("shake", 0.0))
            - float(second.get("phase_durations_s", {}).get("shake", 0.0))
        ),
        "shake_cycles_separation": abs(
            float(first.get("shake_cycles", 0.0))
            - float(second.get("shake_cycles", 0.0))
        ),
        "recovery_duration_separation_s": abs(
            float(first.get("phase_durations_s", {}).get("recover", 0.0))
            - float(second.get("phase_durations_s", {}).get("recover", 0.0))
        ),
        "maximum_knee_separation_m": max(knee_separations, default=0.0),
        "maximum_ankle_separation_m": max(ankle_separations, default=0.0),
        "maximum_root_vertical_separation_m": max(
            root_vertical_separations, default=0.0
        ),
        "maximum_object_separation_m": max(object_separations, default=0.0),
    }
    passing_dimensions = [
        name
        for name, threshold in PERCEPTUAL_DIVERSITY_THRESHOLDS.items()
        if measured[name] >= threshold
    ]
    normalized_dimensions = {
        name: measured[name] / threshold
        for name, threshold in PERCEPTUAL_DIVERSITY_THRESHOLDS.items()
    }
    return {
        **measured,
        "thresholds": PERCEPTUAL_DIVERSITY_THRESHOLDS,
        "normalized_dimensions": normalized_dimensions,
        "normalized_diversity_score": max(normalized_dimensions.values(), default=0.0),
        "passing_dimensions": passing_dimensions,
        "perceptually_distinct": bool(pairs and passing_dimensions),
    }


def select_diverse_candidate_batch(
    candidates: list[dict[str, Any]],
    *,
    candidate_count: int = RANKED_CANDIDATE_COUNT,
    baseline_result_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose the strongest batch-level diversity set without making it fatal.

    Structural validity and unique motion data are hard requirements.  The
    perceptual thresholds are an optimization target: if a fully separated
    five-way set exists, it wins; otherwise the most separated five unique
    clips still reach the visual judge.  This prevents a diversity heuristic
    from incorrectly terminating an otherwise judgeable run.
    """

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    structurally_valid = [
        candidate
        for candidate in candidates
        if candidate.get("compile_success")
        and candidate.get("structural_valid") is not False
        and isinstance(candidate.get("perceptual_descriptor"), dict)
    ]
    unique: list[dict[str, Any]] = []
    seen_motion_hashes: set[str] = set()
    duplicate_result_ids: list[str] = []
    for candidate in structurally_valid:
        motion_hash = str(candidate.get("motion_sha256") or candidate.get("result_id"))
        if motion_hash in seen_motion_hashes:
            duplicate_result_ids.append(str(candidate.get("result_id")))
            continue
        seen_motion_hashes.add(motion_hash)
        unique.append(candidate)

    baseline_available = any(
        candidate.get("result_id") == baseline_result_id for candidate in unique
    )
    base_audit: dict[str, Any] = {
        "requested_candidate_count": candidate_count,
        "structurally_valid_count": len(structurally_valid),
        "unique_motion_count": len(unique),
        "duplicate_result_ids": duplicate_result_ids,
        "baseline_result_id": baseline_result_id,
        "baseline_available": baseline_available,
    }
    if len(unique) < candidate_count:
        return unique, {
            **base_audit,
            "status": "insufficient_unique_structural_candidates",
            "selected_result_ids": [candidate.get("result_id") for candidate in unique],
            "hard_pair_count": 0,
            "total_pair_count": candidate_count * (candidate_count - 1) // 2,
            "all_pairs_above_threshold": False,
            "relaxed_diversity": False,
        }

    pairwise: dict[tuple[int, int], dict[str, Any]] = {}
    for first_index, second_index in combinations(range(len(unique)), 2):
        pairwise[(first_index, second_index)] = compare_perceptual_descriptors(
            unique[first_index]["perceptual_descriptor"],
            unique[second_index]["perceptual_descriptor"],
        )

    baseline_index = next(
        (
            index
            for index, candidate in enumerate(unique)
            if candidate.get("result_id") == baseline_result_id
        ),
        None,
    )
    best_indices: tuple[int, ...] | None = None
    best_key: tuple[int, int, float, float, int] | None = None
    best_comparisons: list[dict[str, Any]] = []
    for indices in combinations(range(len(unique)), candidate_count):
        if baseline_index is not None and baseline_index not in indices:
            continue
        comparisons_for_set = [
            pairwise[(min(first, second), max(first, second))]
            for first, second in combinations(indices, 2)
        ]
        normalized = [
            float(comparison["normalized_diversity_score"])
            for comparison in comparisons_for_set
        ]
        hard_pair_count = sum(score >= 1.0 for score in normalized)
        non_adaptive_count = sum(
            not str(unique[index].get("recipe", {}).get("name", "")).startswith(
                "adaptive_timing_"
            )
            for index in indices
        )
        minimum_score = min(normalized, default=0.0)
        average_score = sum(min(score, 3.0) for score in normalized) / max(1, len(normalized))
        # Prefer hard-separated pairs first, then preserve as many semantic
        # pose/style proposals as possible before considering margin size.
        key = (
            hard_pair_count,
            non_adaptive_count,
            minimum_score,
            average_score,
            -sum(indices),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_indices = indices
            best_comparisons = comparisons_for_set

    assert best_indices is not None
    selected = [unique[index] for index in best_indices]
    total_pair_count = candidate_count * (candidate_count - 1) // 2
    hard_pair_count = int(best_key[0]) if best_key is not None else 0
    selected_pairwise: list[dict[str, Any]] = []
    for (first, second), comparison in zip(combinations(best_indices, 2), best_comparisons):
        selected_pairwise.append(
            {
                "first_result_id": unique[first].get("result_id"),
                "second_result_id": unique[second].get("result_id"),
                **comparison,
            }
        )
    normalized_selected = [
        float(comparison["normalized_diversity_score"])
        for comparison in best_comparisons
    ]
    return selected, {
        **base_audit,
        "status": "complete",
        "selected_result_ids": [candidate.get("result_id") for candidate in selected],
        "hard_pair_count": hard_pair_count,
        "total_pair_count": total_pair_count,
        "all_pairs_above_threshold": hard_pair_count == total_pair_count,
        "relaxed_diversity": hard_pair_count < total_pair_count,
        "minimum_normalized_distance": min(normalized_selected, default=0.0),
        "average_normalized_distance": (
            sum(normalized_selected) / max(1, len(normalized_selected))
        ),
        "baseline_included": any(
            candidate.get("result_id") == baseline_result_id for candidate in selected
        ),
        "pairwise": selected_pairwise,
    }


def _write_trace(path: Path, trace: dict[str, Any]) -> None:
    atomic_write_json(path, trace)


ProgressCallback = Callable[[dict[str, Any]], None]


# The correlation keys a stage span carries. Deliberately not all of `data`: a stage
# span must not churn because a human-readable message changed, only because the thing
# being worked on did.
_STAGE_REFS = ("round", "candidate_index", "result_id")


def _progress(
    callback: ProgressCallback | None,
    event: str,
    stage: str,
    message: str,
    **data: Any,
) -> None:
    """Report progress, and advance the run's stage timeline.

    Every call already declares which stage it belongs to, so driving the timeline from
    here gives stage spans that tile the run without bracketing anything by hand -- and
    makes it impossible to nest two stages, which would double count in
    `Transcript.duration_by_stage()`.
    """
    timeline = current_stage_timeline()
    if timeline is not None:
        timeline.enter(stage, **{key: data[key] for key in _STAGE_REFS if key in data})
    if callback is not None:
        callback(
            {
                "event": event,
                "stage": stage,
                "message": message,
                # Plan 01 section 1.4: the CLI path passes no callback, so before this the
                # only wall clock in the system was stamped by the API store on receipt.
                "at": datetime.now(UTC).isoformat(),
                "data": data,
            }
        )


def _candidate_score(candidate: dict[str, Any]) -> tuple[int, int, int, float]:
    parsed = candidate.get("judgment", {})
    return (
        int(parsed.get("overall", 0)),
        int(parsed.get("anatomical_naturalness", 0)),
        int(parsed.get("gesture_recognizability", 0)),
        float(parsed.get("confidence", 0.0)),
    )


def select_five_way_winner(
    ranked_candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    *,
    winner_index: int | None,
    baseline_result_id: str | None,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve a five-way decision without assuming any candidate passed."""
    if (
        winner_index is not None
        and 0 <= winner_index < len(ranked_candidates)
        and ranked_candidates[winner_index].get("accepted")
    ):
        return ranked_candidates[winner_index], "clear_five_way_winner"
    baseline_candidate = next(
        (
            candidate
            for candidate in ranked_candidates
            if candidate.get("result_id") == baseline_result_id
            and candidate.get("accepted")
        ),
        None,
    )
    if baseline_candidate is not None:
        return baseline_candidate, "no_clear_winner_fallback_baseline"
    if accepted:
        return max(accepted, key=_candidate_score), "no_clear_winner_fallback_best_valid"
    return None, "no_candidate_passed_visual_threshold"


def _tournament(
    accepted: list[dict[str, Any]],
    judge: VLMJudge,
    output_dir: Path,
    trace: dict[str, Any],
    *,
    prompt: str,
) -> dict[str, Any]:
    ordered = sorted(accepted, key=_candidate_score, reverse=True)
    champion = ordered[0]
    for match_index, challenger in enumerate(ordered[1:], start=1):
        comparison = judge.compare(
            Path(champion["evidence_manifest"]),
            Path(challenger["evidence_manifest"]),
            random_seed=blinding_seed(
                prompt,
                [str(champion["result_id"]), str(challenger["result_id"])],
            ),
            reverse_check=True,
        )
        comparison_path = output_dir / "tournament" / f"match-{match_index:02d}.json"
        write_judge_record(comparison, comparison_path)
        mapped = comparison["calls"][0]["mapped_winner"]
        if comparison["order_consistent"] and mapped == "second":
            champion = challenger
        elif not comparison["order_consistent"] or mapped in {"tie", "neither"}:
            if _candidate_score(challenger) > _candidate_score(champion):
                champion = challenger
        trace["tournament"].append(
            {
                "match": match_index,
                "first_candidate": comparison["first_result_id"],
                "second_candidate": comparison["second_result_id"],
                "comparison": str(comparison_path),
                "order_consistent": comparison["order_consistent"],
                "mapped_winners": [call["mapped_winner"] for call in comparison["calls"]],
                "champion_after_match": champion["result_id"],
            }
        )
    return champion


def run_best_of_five(
    prompt: str,
    output_dir: Path,
    *,
    provider: str = "openai",
    base_url: str = "http://127.0.0.1:8000",
    max_rounds: int = 2,
    selection_mode: str = "unary_tournament",
    scene_manifest: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    max_model_calls: int | None = 4,
    capture_fn: Callable[..., Path] = capture_result_frames,
    capture_batch_fn: Callable[..., list[Path]] | None = None,
    tracer: Tracer | NullTracer | None = None,
) -> Path:
    """Run the flywheel, recording a stage timeline when a tracer is supplied.

    A thin wrapper rather than a `with` around the body: installing the timeline is two
    lines, and indenting seven hundred lines of control flow to hold them would bury the
    change this PR is actually making.
    """
    with stage_timeline(tracer if tracer is not None else NullTracer()):
        return _run_best_of_five(
            prompt,
            output_dir,
            provider=provider,
            base_url=base_url,
            max_rounds=max_rounds,
            selection_mode=selection_mode,
            scene_manifest=scene_manifest,
            progress_callback=progress_callback,
            max_model_calls=max_model_calls,
            capture_fn=capture_fn,
            capture_batch_fn=capture_batch_fn,
        )


def _run_best_of_five(
    prompt: str,
    output_dir: Path,
    *,
    provider: str = "openai",
    base_url: str = "http://127.0.0.1:8000",
    max_rounds: int = 2,
    selection_mode: str = "unary_tournament",
    scene_manifest: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    max_model_calls: int | None = 4,
    capture_fn: Callable[..., Path] = capture_result_frames,
    capture_batch_fn: Callable[..., list[Path]] | None = None,
) -> Path:
    if selection_mode not in {"unary_tournament", "five_way", "human_pilot"}:
        raise ValueError(
            "selection_mode must be unary_tournament, five_way, or human_pilot"
        )
    scene = scene_manifest or default_scene()
    _progress(
        progress_callback,
        "planning_started",
        "planning",
        "Interpreting the prompt as bounded motion primitives.",
        prompt=prompt,
    )
    outcome = plan_motion(PlanRequest(text=prompt, scene=scene, provider=provider))
    if outcome.program.intent == Intent.UNSUPPORTED:
        reason = outcome.program.unsupported_reason or "The requested motion is unsupported."
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "flywheel-trace.json"
        trace = {
            "schema_version": "1.0",
            "prompt": prompt,
            "planner": {
                "provider": outcome.provider,
                "model": outcome.model,
                "model_calls": outcome.model_calls,
                "response_ids": list(outcome.response_ids),
                "program": outcome.program.model_dump(mode="json"),
            },
            "rounds": [],
            "repairs": [],
            "tournament": [],
            "rankings": [],
            "selection_mode": selection_mode,
            "capture_policy": "no_capture_for_unsupported_motion",
            "winner_result_id": None,
            "unsupported_reason": reason,
            "status": "unsupported_motion",
        }
        _write_trace(trace_path, trace)
        _progress(
            progress_callback,
            "pipeline_unsupported",
            "finalize",
            reason,
            reason=reason,
            model_calls=outcome.model_calls,
        )
        return trace_path
    _progress(
        progress_callback,
        "plan_ready",
        "planning",
        "The semantic plan is ready.",
        provider=outcome.provider,
        model=outcome.model,
        model_calls=outcome.model_calls,
        intent=outcome.program.intent.value,
        hand=outcome.program.hand.value,
        primitive_count=len(outcome.program.primitives),
        program=outcome.program.model_dump(mode="json"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "flywheel-trace.json"
    resumable = False
    if trace_path.is_file():
        prior_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        resumable = bool(
            prior_trace.get("status") in {"no_acceptable_candidate", "running"}
            and prior_trace.get("prompt") == prompt
            and prior_trace.get("selection_mode") == selection_mode == "five_way"
            and len(prior_trace.get("rounds", [])) == 1
            and not prior_trace.get("rankings")
        )
    if resumable:
        trace = prior_trace
        baseline_candidate_index = int(
            trace.get("baseline_selection", {}).get("candidate_index", 1)
        )
        trace["status"] = "running"
        trace.setdefault("resume_events", []).append(
            {
                "reason": "replacement_pool_expanded",
                "preserved_candidate_count": len(trace["rounds"][0].get("candidates", [])),
            }
        )
    else:
        baseline_candidate_index = single_sample_baseline_index(prompt)
        trace = {
            "schema_version": "1.0",
            "prompt": prompt,
            "planner": {
                "provider": outcome.provider,
                "model": outcome.model,
                "model_calls": outcome.model_calls,
                "response_ids": list(outcome.response_ids),
                "program": outcome.program.model_dump(mode="json"),
            },
            "rounds": [],
            "repairs": [],
            "tournament": [],
            "rankings": [],
            "selection_mode": selection_mode,
            "capture_policy": (
                "deferred_until_lightweight_vlm"
                if selection_mode == "human_pilot"
                else "full_phase_ego_orbit"
            ),
            "baseline_result_id": None,
            "baseline_selection": {
                "method": "precommitted_prompt_hash",
                "salt": BASELINE_SAMPLING_SALT,
                "candidate_index": baseline_candidate_index,
                "candidate_recipe": None,
                "selected_before_judging": True,
            },
            "winner_result_id": None,
            "status": "running",
        }
    _write_trace(trace_path, trace)
    judge = (
        None
        if selection_mode == "human_pilot"
        else VLMJudge(max_model_calls=max_model_calls)
    )
    store = ResultStore()
    round_base = outcome.program
    accepted: list[dict[str, Any]] = []
    direct_winner: dict[str, Any] | None = None

    for round_index in range(max_rounds):
        if resumable and round_index == 0:
            round_record = trace["rounds"][0]
        else:
            round_record = {"round": round_index + 1, "candidates": []}
            trace["rounds"].append(round_record)
        base_parameters = _primary_parameters(round_base)
        recipes = (
            candidate_recipe_pool(base_parameters, outcome.program.intent)
            if selection_mode in {"five_way", "human_pilot"}
            else candidate_recipes(base_parameters, outcome.program.intent)
        )
        attempted_recipes = {
            str(candidate.get("recipe", {}).get("name"))
            for candidate in round_record["candidates"]
            if isinstance(candidate, dict) and isinstance(candidate.get("recipe"), dict)
        }
        for candidate_index, recipe in enumerate(recipes, start=1):
            if recipe.name in attempted_recipes:
                continue
            print(
                f"Round {round_index + 1}/{max_rounds}, candidate {candidate_index}/{len(recipes)}: {recipe.name}",
                flush=True,
            )
            _progress(
                progress_callback,
                "candidate_started",
                "candidates",
                f"Building candidate {candidate_index}: {recipe.name.replace('_', ' ')}.",
                round=round_index + 1,
                candidate_index=candidate_index,
                recipe=recipe.name,
                purpose=recipe.purpose,
            )
            program = apply_recipe(
                round_base,
                recipe,
                seed_offset=round_index * 100 + candidate_index,
            )
            request = CompileRequest(scene=scene, program=program, persist=True)
            clip = compile_motion(request)
            provenance = clip.provenance.model_copy(
                update={
                    "planner_provider": outcome.provider,
                    "planner_model": outcome.model,
                    "model_calls": outcome.model_calls,
                    "seed": program.seed,
                }
            )
            clip = clip.model_copy(update={"provenance": provenance})
            result_id = store.persist(request, clip)
            candidate_dir = output_dir / f"round-{round_index + 1}" / f"candidate-{candidate_index}-{recipe.name}"
            candidate: dict[str, Any] = {
                "candidate_index": candidate_index,
                "recipe": asdict(recipe),
                "result_id": result_id,
                "program": program.model_dump(mode="json"),
                "delta_from_round_baseline": describe_program_delta(round_base, program),
                "motion_sha256": _motion_hash(clip),
                "perceptual_descriptor": motion_perceptual_descriptor(
                    clip,
                    (
                        program.hands
                        if program.intent in {Intent.COMPOSITE, Intent.SEQUENCE}
                        and program.hands
                        else program.hand
                    ),
                ),
                "compile_success": clip.success,
                "structural_valid": clip.metrics.get("structural_valid"),
                "structural_failures": clip.metrics.get("structural_failures", []),
                "quality_metrics": {
                    key: clip.metrics.get(key)
                    for key in (
                        "max_wrist_swing_rad",
                        "max_wrist_twist_rad",
                        "max_forearm_twist_rad",
                        "self_collision_frames",
                        "active_hand_visibility_fraction",
                        "max_angular_velocity_rad_s",
                        "max_angular_acceleration_rad_s2",
                        "max_angular_jerk_rad_s3",
                        "wrist_shake_cycles",
                        "wrist_shake_amplitude_rad",
                        "forearm_rotation_cycles",
                        "forearm_rotation_amplitude_rad",
                        "parallel_forearm_max_axis_error_deg",
                        "parallel_forearm_max_frontal_axis_error_deg",
                        "parallel_forearm_minimum_separation_m",
                        "parallel_forearm_minimum_hand_separation_m",
                        "travel_wheel_cross_body_fraction",
                        "travel_wheel_maximum_opposite_elbow_distance_m",
                        "travel_wheel_vertical_order_range_m",
                        "travel_wheel_depth_order_range_m",
                        "shake_duration_s",
                        "root_path_length_m",
                        "root_displacement_m",
                        "root_vertical_min_m",
                        "root_vertical_max_m",
                        "final_root_yaw_deg",
                        "requested_body_cycles",
                        "requested_dance_beats",
                        "measured_dance_beats",
                        "dance_alternating_lift_count",
                        "dance_lateral_root_range_m",
                        "dance_left_foot_peak_clearance_m",
                        "dance_right_foot_peak_clearance_m",
                        "requested_climb_height_m",
                        "measured_climb_height_m",
                        "climb_vertical_completion_fraction",
                        "requested_climb_cycles",
                        "climb_support_target_max_error_m",
                        "climb_three_point_support_fraction",
                        "climb_final_supported_limb_count",
                        "climb_missing_support_object_count",
                        "ground_penetration_m",
                        "maximum_foot_clearance_m",
                        "airborne_frame_count",
                        "max_support_foot_target_error_m",
                        "max_support_foot_slide_per_frame_m",
                        "support_contact_fraction",
                        "final_balanced_leg_error_rad",
                        "requested_push_up_cycles",
                        "measured_push_up_cycles",
                        "push_up_vertical_excursion_m",
                        "plank_hold_vertical_range_m",
                        "push_up_palm_height_range_m",
                        "push_up_toe_support_clearance_m",
                        "push_up_toe_height_range_m",
                        "push_up_toe_position_range_m",
                        "push_up_min_knee_extension_deg",
                        "requested_crawl_cycles",
                        "requested_crawl_distance_m",
                        "crawl_root_displacement_m",
                        "crawl_hand_alternation_range_m",
                        "requested_jumping_jack_cycles",
                        "measured_jumping_jack_cycles",
                        "jumping_jack_foot_spread_excursion_m",
                        "jumping_jack_max_wrist_height_m",
                        "requested_burpee_cycles",
                        "measured_burpee_jump_cycles",
                        "measured_burpee_push_up_cycles",
                        "burpee_floor_support_phase_count",
                        "burpee_airborne_phase_count",
                        "burpee_root_vertical_excursion_m",
                        "burpee_max_wrist_height_m",
                        "requested_squat_cycles",
                        "measured_squat_cycles",
                        "squat_minimum_depth_m",
                        "squat_max_stance_return_error_m",
                        "requested_lunge_cycles",
                        "measured_lunge_cycles",
                        "lunge_minimum_root_drop_m",
                        "lunge_minimum_foot_stagger_m",
                        "lunge_max_stance_return_error_m",
                        "single_leg_balance_phase_count",
                        "single_leg_min_raised_foot_clearance_m",
                        "single_leg_max_support_foot_slide_m",
                        "single_leg_support_contact_fraction",
                        "requested_sit_up_cycles",
                        "measured_sit_up_cycles",
                        "sit_up_minimum_head_lift_m",
                        "sit_up_max_supine_return_error_m",
                        "requested_body_rotation_degrees",
                        "measured_body_rotation_degrees",
                        "minimum_body_rotation_completion_fraction",
                        "minimum_rotation_travel_completion_fraction",
                        "floor_roll_nonfoot_contact_frame_count",
                        "floor_roll_nonfoot_contact_fraction",
                        "cartwheel_hand_contact_frame_count",
                        "cartwheel_hand_contact_fraction",
                        "cartwheel_inverted_frame_count",
                        "cartwheel_max_foot_clearance_m",
                        "cartwheel_minimum_head_clearance_m",
                        "airborne_rotation_airborne_frame_count",
                        "airborne_rotation_airborne_fraction",
                        "airborne_rotation_peak_root_height_m",
                        "obstacle_missing_target_count",
                        "minimum_obstacle_step_foot_clearance_m",
                        "maximum_obstacle_step_crossing_error_m",
                        "minimum_obstacle_avoidance_root_clearance_m",
                        "handoff_dual_contact_duration_s",
                        "handoff_receiver_retained",
                        "handoff_attachment_slip_m",
                        "object_max_step_m",
                        "object_guided_projected_distance_m",
                        "object_guided_lateral_error_m",
                        "object_guided_support_height_error_m",
                        "object_expected_roll_turns",
                        "object_measured_roll_turns",
                        "object_measured_support_spin_turns",
                        "object_placement_horizontal_distance_m",
                        "carried_object_ids",
                        "carried_object_max_step_m",
                        "stateful_object_transition_count",
                        "stateful_object_release_displacement_m",
                        "stateful_object_landing_height_error_m",
                    )
                },
            }
            round_record["candidates"].append(candidate)
            _progress(
                progress_callback,
                "candidate_compiled",
                "structural_checks",
                (
                    f"Candidate {candidate_index} passed deterministic checks."
                    if clip.success and clip.metrics.get("structural_valid", True)
                    else f"Candidate {candidate_index} was rejected by deterministic checks."
                ),
                round=round_index + 1,
                candidate_index=candidate_index,
                recipe=recipe.name,
                result_id=result_id,
                success=clip.success,
                structural_valid=clip.metrics.get("structural_valid"),
                structural_failures=clip.metrics.get("structural_failures", []),
            )
            if round_index == 0 and candidate_index == baseline_candidate_index:
                trace["baseline_result_id"] = result_id
                trace["baseline_selection"]["candidate_recipe"] = recipe.name
            if clip.success and clip.metrics.get("structural_valid", True):
                structural_predecessors = [
                    item
                    for item in round_record["candidates"][:-1]
                    if item.get("compile_success")
                    and item.get("structural_valid") is not False
                    and isinstance(item.get("perceptual_descriptor"), dict)
                ]
                diversity = [
                    {
                        "against_result_id": prior["result_id"],
                        "against_recipe": prior["recipe"]["name"],
                        **compare_perceptual_descriptors(
                            prior["perceptual_descriptor"],
                            candidate["perceptual_descriptor"],
                        ),
                    }
                    for prior in structural_predecessors
                ]
                candidate["perceptual_diversity"] = diversity
                if selection_mode in {"five_way", "human_pilot"}:
                    candidate["candidate_pool_eligible"] = True
                else:
                    indistinguishable = [
                        item for item in diversity if not item["perceptually_distinct"]
                    ]
                    if indistinguishable:
                        candidate["accepted"] = False
                        candidate["rejection_stage"] = "perceptual_diversity_prefilter"
                        candidate["rejection_reason"] = (
                            "candidate is below every motion-separation threshold versus "
                            + ", ".join(item["against_recipe"] for item in indistinguishable)
                        )
                    else:
                        candidate["perceptually_rankable"] = True
                        _progress(
                            progress_callback,
                            "capture_started",
                            "visual_evidence",
                            f"Capturing full-FOV first-person and orbit evidence for candidate {candidate_index}.",
                            round=round_index + 1,
                            candidate_index=candidate_index,
                            result_id=result_id,
                        )
                        manifest = capture_fn(result_id, candidate_dir, base_url=base_url)
                        candidate["evidence_manifest"] = str(manifest)
                        _progress(
                            progress_callback,
                            "capture_ready",
                            "visual_evidence",
                            f"Visual evidence for candidate {candidate_index} is ready.",
                            round=round_index + 1,
                            candidate_index=candidate_index,
                            result_id=result_id,
                            evidence_manifest=str(manifest),
                        )
                        if selection_mode == "unary_tournament":
                            assert judge is not None
                            manifest = Path(candidate["evidence_manifest"])
                            judgment = judge.score(manifest)
                            judgment_path = candidate_dir / "vlm-judgment.json"
                            write_judge_record(judgment, judgment_path)
                            parsed = judgment["call"]["parsed"]
                            candidate.update(
                                {
                                    "judgment_path": str(judgment_path),
                                    "judgment": parsed,
                                    "accepted": bool(parsed["accept"]),
                                }
                            )
                            if parsed["accept"]:
                                accepted.append(candidate)
            else:
                candidate["accepted"] = False
                candidate["rejection_stage"] = "structural_prefilter"
                structural_failures = candidate.get("structural_failures", [])
                candidate["rejection_reason"] = (
                    "; ".join(str(value) for value in structural_failures)
                    if structural_failures
                    else "The candidate failed deterministic physical checks."
                )
            _write_trace(trace_path, trace)
            if selection_mode in {"five_way", "human_pilot"}:
                _, provisional_selection = select_diverse_candidate_batch(
                    round_record["candidates"],
                    baseline_result_id=trace.get("baseline_result_id"),
                )
                round_record["batch_selection"] = provisional_selection
                if provisional_selection.get("all_pairs_above_threshold"):
                    break

        if selection_mode in {"five_way", "human_pilot"}:
            selected_batch, batch_selection = select_diverse_candidate_batch(
                round_record["candidates"],
                baseline_result_id=trace.get("baseline_result_id"),
            )
            round_record["batch_selection"] = batch_selection
            selected_result_ids = {
                str(candidate.get("result_id")) for candidate in selected_batch
            }
            complete_batch = len(selected_batch) == RANKED_CANDIDATE_COUNT
            for candidate in round_record["candidates"]:
                result_id = str(candidate.get("result_id"))
                selected = complete_batch and result_id in selected_result_ids
                candidate["perceptually_rankable"] = selected
                if selected:
                    candidate.pop("accepted", None)
                    candidate.pop("rejection_stage", None)
                    candidate.pop("rejection_reason", None)
                    _progress(
                        progress_callback,
                        "candidate_selected",
                        "candidates",
                        f"Candidate {candidate['candidate_index']} joined the five-way judging batch.",
                        round=round_index + 1,
                        candidate_index=candidate["candidate_index"],
                        result_id=result_id,
                        relaxed_diversity=batch_selection.get("relaxed_diversity", False),
                    )
                elif candidate.get("compile_success") and candidate.get("structural_valid") is not False:
                    candidate["accepted"] = False
                    candidate["rejection_stage"] = "batch_diversity_selection"
                    candidate["rejection_reason"] = (
                        "A more diverse structurally valid candidate was selected for the five-way batch."
                    )
                    _progress(
                        progress_callback,
                        "candidate_filtered",
                        "candidates",
                        f"Candidate {candidate['candidate_index']} was not needed in the most diverse five-way batch.",
                        round=round_index + 1,
                        candidate_index=candidate["candidate_index"],
                        result_id=result_id,
                        rejection_stage="batch_diversity_selection",
                    )

            if complete_batch:
                pending: list[tuple[dict[str, Any], str, Path]] = []
                for candidate in selected_batch:
                    candidate_index = int(candidate["candidate_index"])
                    recipe_name = str(candidate["recipe"]["name"])
                    result_id = str(candidate["result_id"])
                    if selection_mode == "human_pilot":
                        # The human comparison UI renders both synchronized
                        # views directly from the persisted clip. Defer VLM
                        # evidence capture until a human explicitly requests it.
                        candidate["human_review_ready"] = True
                    elif not candidate.get("evidence_manifest"):
                        candidate_dir = (
                            output_dir
                            / f"round-{round_index + 1}"
                            / f"candidate-{candidate_index}-{recipe_name}"
                        )
                        _progress(
                            progress_callback,
                            "capture_started",
                            "visual_evidence",
                            f"Capturing full-FOV first-person and orbit evidence for candidate {candidate_index}.",
                            round=round_index + 1,
                            candidate_index=candidate_index,
                            result_id=result_id,
                        )
                        pending.append((candidate, result_id, candidate_dir))
                # The whole round goes through one browser launch when the caller
                # supplies a batch capture; the per-candidate seam stays the default so
                # every existing injected `capture_fn` keeps working unchanged.
                if pending:
                    if capture_batch_fn is not None:
                        manifests = capture_batch_fn(
                            [(result_id, directory) for _, result_id, directory in pending],
                            base_url=base_url,
                        )
                    else:
                        manifests = [
                            capture_fn(result_id, directory, base_url=base_url)
                            for _, result_id, directory in pending
                        ]
                    for (candidate, result_id, _), manifest in zip(pending, manifests):
                        candidate["evidence_manifest"] = str(manifest)
                        _progress(
                            progress_callback,
                            "capture_ready",
                            "visual_evidence",
                            f"Visual evidence for candidate {int(candidate['candidate_index'])} is ready.",
                            round=round_index + 1,
                            candidate_index=int(candidate["candidate_index"]),
                            result_id=result_id,
                            evidence_manifest=str(manifest),
                        )
            else:
                _progress(
                    progress_callback,
                    "candidate_pool_incomplete",
                    "candidates",
                    (
                        "The adaptive pool did not yet contain five unique structurally valid motions; "
                        "preparing another bounded round."
                    ),
                    round=round_index + 1,
                    structurally_valid_count=batch_selection.get("structurally_valid_count", 0),
                    unique_motion_count=batch_selection.get("unique_motion_count", 0),
                )
            _write_trace(trace_path, trace)

        if selection_mode == "five_way":
            assert judge is not None
            ranked_candidates = [
                candidate
                for candidate in round_record["candidates"]
                if candidate.get("perceptually_rankable")
                and candidate.get("evidence_manifest")
            ]
            if len(ranked_candidates) == 5:
                _progress(
                    progress_callback,
                    "judging_started",
                    "vlm_judge",
                    "The calibrated visual judge is comparing five blinded candidates.",
                    round=round_index + 1,
                    candidate_result_ids=[item["result_id"] for item in ranked_candidates],
                )
                ranking_path = output_dir / f"round-{round_index + 1}" / "five-way-ranking.json"
                if ranking_path.is_file():
                    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
                    if (
                        ranking.get("kind") != "five_way_motion_judgment"
                        or len(ranking.get("mapped_scores", {})) != 5
                    ):
                        raise ValueError("preserved five-way ranking is incomplete")
                else:
                    ranking = judge.rank_five(
                        [Path(candidate["evidence_manifest"]) for candidate in ranked_candidates],
                        random_seed=blinding_seed(
                            prompt,
                            [str(candidate["result_id"]) for candidate in ranked_candidates],
                        ),
                    )
                    write_judge_record(ranking, ranking_path)
                for index, candidate in enumerate(ranked_candidates):
                    parsed = ranking["mapped_scores"].get(
                        index,
                        ranking["mapped_scores"].get(str(index)),
                    )
                    if not isinstance(parsed, dict):
                        raise ValueError(f"five-way ranking is missing candidate {index}")
                    candidate["judgment"] = parsed
                    candidate["accepted"] = bool(parsed["accept"])
                    if parsed["accept"]:
                        accepted.append(candidate)
                winner_index = ranking["mapped_winner_index"]
                direct_winner, selection_reason = select_five_way_winner(
                    ranked_candidates,
                    accepted,
                    winner_index=winner_index,
                    baseline_result_id=trace.get("baseline_result_id"),
                )
                trace["rankings"].append(
                    {
                        "round": round_index + 1,
                        "record": str(ranking_path),
                        "mapped_winner_result_id": direct_winner["result_id"] if direct_winner else None,
                        "selection_reason": selection_reason,
                        "response_id": ranking["call"]["response_id"],
                    }
                )
                _progress(
                    progress_callback,
                    "judging_ready",
                    "vlm_judge",
                    (
                        "The visual judge selected a clear winner."
                        if direct_winner is not None
                        else "No candidate cleared the visual threshold."
                    ),
                    round=round_index + 1,
                    winner_result_id=direct_winner["result_id"] if direct_winner else None,
                    selection_reason=selection_reason,
                    scores=[candidate.get("judgment", {}) for candidate in ranked_candidates],
                    routing=ranking.get("routing", {}),
                )
                _write_trace(trace_path, trace)

        if selection_mode == "human_pilot":
            human_candidates = [
                candidate
                for candidate in round_record["candidates"]
                if candidate.get("human_review_ready")
            ]
            if len(human_candidates) == 5:
                trace["human_pilot_result_ids"] = [
                    candidate["result_id"] for candidate in human_candidates
                ]
                trace["status"] = "awaiting_human_selection"
                _write_trace(trace_path, trace)
                break

        if accepted:
            break
        judged = [
            candidate
            for candidate in round_record["candidates"]
            if isinstance(candidate.get("judgment"), dict)
        ]
        if round_index + 1 >= max_rounds:
            break
        if not judged:
            structurally_valid = [
                candidate
                for candidate in round_record["candidates"]
                if candidate.get("compile_success")
                and candidate.get("structural_valid") is not False
                and isinstance(candidate.get("program"), dict)
            ]
            repair_source = next(
                (
                    candidate
                    for candidate in structurally_valid
                    if candidate.get("result_id") == trace.get("baseline_result_id")
                ),
                structurally_valid[0] if structurally_valid else None,
            )
            source_program = (
                MotionProgram.model_validate(repair_source["program"])
                if repair_source is not None
                else round_base
            )
            round_base = apply_prefilter_safety_repair(source_program)
            repair_entry = {
                "after_round": round_index + 1,
                "kind": "deterministic_prefilter_safety",
                "source_result_id": (
                    repair_source.get("result_id") if repair_source is not None else None
                ),
                "reason": "fewer than five unique structurally valid candidates reached judging",
                "model_calls": 0,
            }
            trace["repairs"].append(repair_entry)
            _progress(
                progress_callback,
                "repair_ready",
                "repair",
                "A deterministic safety repair prepared a fresh candidate pool without spending a model call.",
                round=round_index + 1,
                source_result_id=repair_entry["source_result_id"],
                repair_kind=repair_entry["kind"],
            )
            _write_trace(trace_path, trace)
            continue
        best_rejected = max(judged, key=_candidate_score)
        assert judge is not None
        repair_record = judge.recommend_repair(
            prompt=prompt,
            motion_profile=round_base.motion_profile.model_dump(mode="json")
            if round_base.motion_profile
            else {},
            parameters=_repair_parameter_payload(round_base),
            judgment=best_rejected["judgment"],
            structural_metrics={
                **best_rejected["quality_metrics"],
                "structural_valid": best_rejected["structural_valid"],
                "structural_failures": best_rejected["structural_failures"],
            },
        )
        repair_path = output_dir / f"repair-round-{round_index + 1}.json"
        write_judge_record(repair_record, repair_path)
        repair = RepairPatch.model_validate(repair_record["call"]["parsed"])
        round_base = apply_repair(round_base, repair)
        trace["repairs"].append(
            {
                "after_round": round_index + 1,
                "source_result_id": best_rejected["result_id"],
                "record": str(repair_path),
                "patch": repair.model_dump(mode="json"),
            }
        )
        _progress(
            progress_callback,
            "repair_ready",
            "repair",
            "The judge proposed a bounded parameter repair for the next round.",
            round=round_index + 1,
            source_result_id=best_rejected["result_id"],
            patch=repair.model_dump(mode="json"),
        )
        _write_trace(trace_path, trace)

    if selection_mode == "human_pilot" and trace.get("status") == "awaiting_human_selection":
        pass
    elif accepted:
        champion = (
            direct_winner
            if selection_mode == "five_way" and direct_winner is not None
            else _tournament(accepted, judge, output_dir, trace, prompt=prompt)
        )
        assert champion is not None
        trace["winner_result_id"] = champion["result_id"]
        trace["winner"] = champion
        trace["status"] = "winner_selected"
    else:
        trace["status"] = "no_acceptable_candidate"
    _write_trace(trace_path, trace)
    _progress(
        progress_callback,
        "pipeline_finished",
        "finalize",
        (
            "The final animation is ready."
            if trace.get("winner_result_id")
            else "The pipeline finished without an acceptable animation."
        ),
        status=trace["status"],
        winner_result_id=trace.get("winner_result_id"),
        trace_path=str(trace_path),
    )
    return trace_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rigby's calibrated best-of-five gesture flywheel")
    parser.add_argument("prompt")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--provider", choices=("openai", "offline"), default="openai")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--selection-mode",
        choices=("unary_tournament", "five_way", "human_pilot"),
        default="unary_tournament",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="where the transcript is written; defaults to results/pipeline-runs/",
    )
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args()

    # Plan 01 section 1.5: with no progress callback the CLI path produced no run id, no
    # run record and no event history at all, and every batch eval driver uses it. It now
    # opens the same root span, in the same place, as an API-launched run -- which is what
    # makes section 7's "structurally identical transcripts" checkable rather than a hope.
    run_root = arguments.run_root or (PROJECT_ROOT / "results" / "pipeline-runs")
    tracer = Tracer.open(run_root, run_id=arguments.run_id)
    atomic_write_json(
        tracer.run_dir / "config.json",
        effective_configuration(
            run_id=tracer.run_id,
            prompt=arguments.prompt,
            provider=arguments.provider,
            selection_mode=arguments.selection_mode,
            max_rounds=arguments.max_rounds,
            extra={"launched_by": "cli", "output_dir": str(arguments.output_dir)},
        ),
    )
    with tracer.span(
        "run",
        "pipeline.run",
        prompt=arguments.prompt,
        provider=arguments.provider,
        selection_mode=arguments.selection_mode,
        launched_by="cli",
    ):
        trace_path = run_best_of_five(
            arguments.prompt,
            arguments.output_dir,
            provider=arguments.provider,
            base_url=arguments.base_url,
            max_rounds=arguments.max_rounds,
            selection_mode=arguments.selection_mode,
            tracer=tracer,
        )
    print(trace_path)
    print(f"transcript: {tracer.run_dir}", flush=True)


if __name__ == "__main__":
    main()
