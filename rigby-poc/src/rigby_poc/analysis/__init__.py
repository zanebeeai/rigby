"""Pure post-hoc analysis of a finished clip.

``analyze(clip, program, scene)`` computes metrics from frames alone: no server,
no browser, no API key, no recompilation. ``validate(metrics, program)`` turns
those metrics into individually addressable :class:`CheckResult` verdicts.

Extraction is staged (see ``docs/plans/02-analysis-layer.md`` §4). This module
owns exactly what PR 02a moved; :func:`owned_metric_keys` reports which keys
that is for a given program, and the equivalence harness asserts those keys are
byte-identical to what ``compiler.py`` produces today. The per-action metric
blocks are still computed inside the compiler and are declared, unported, in
:mod:`rigby_poc.analysis.registry`.
"""

from __future__ import annotations

from typing import Any

from ..models import ClipResult, Intent, MotionProgram, PrimitiveKind, SceneManifest
from .contact import (
    intra_hand_contact_checks,
    intra_hand_contact_failures,
    intra_hand_contact_metrics,
)
from .context import AnalysisContext
from .contract import (
    ANATOMY,
    CONTRACT,
    LAYERS,
    PHYSICS,
    SIGNAL,
    CheckResult,
    CheckStatus,
    count_check,
    lower_bound_check,
    saturating_severity,
    skipped,
    upper_bound_check,
)
from .forearm import (
    parallel_forearm_checks,
    parallel_forearm_failures,
    parallel_forearm_metrics,
)
from .gesture import (
    arm_landmarks,
    evaluate_gesture_structure,
    gesture_structure_checks,
    quality_reference,
    shake_joint_oscillation_metrics,
    swing_twist_angles,
    _angular_kinematics,
)
from .registry import (
    BODY_ANALYZERS,
    OBJECT_ANALYZERS,
    Analyzer,
    AnalyzerEntry,
    body_analyzer,
    deferred_actions,
    object_analyzer,
    unregistered_actions,
)
from .rig import RIG_PROFILE, identity_bones, identity_pose, rig_profile
from .safety import safety_checks, safety_metrics
from .semantic import (
    semantic_cycle_assertion,
    semantic_cycle_checks,
    semantic_cycle_failures,
    semantic_cycle_metrics,
)


# Bone sets each compile path feeds to the angular-kinematics pass. Keyed by
# intent because the path, not the action, chooses them.
_FULL_BODY_ANGULAR_BONES = (
    "hips",
    "chest",
    "leftUpperLeg",
    "leftLowerLeg",
    "rightUpperLeg",
    "rightLowerLeg",
    "leftFoot",
    "rightFoot",
    "leftUpperArm",
    "rightUpperArm",
)
_SEQUENCE_ANGULAR_BONES = (
    "hips",
    "chest",
    "leftUpperLeg",
    "leftLowerLeg",
    "rightUpperLeg",
    "rightLowerLeg",
    "leftUpperArm",
    "leftLowerArm",
    "rightUpperArm",
    "rightLowerArm",
)

_ANGULAR_KEYS = (
    "max_angular_velocity_rad_s",
    "max_angular_acceleration_rad_s2",
    "max_angular_jerk_rad_s3",
)


def _angular_metrics(ctx: AnalysisContext) -> dict[str, Any]:
    """Reproduce the angular-kinematics keys for the paths that emit them flat.

    The composite path folds per-hand structures instead, so it is excluded
    here and stays with the compiler until 02c.
    """

    if ctx.intent == Intent.FULL_BODY:
        bones = list(_FULL_BODY_ANGULAR_BONES)
    elif ctx.intent == Intent.SEQUENCE:
        bones = list(_SEQUENCE_ANGULAR_BONES)
    elif ctx.intent == Intent.OBJECT_INTERACTION:
        prefix = ctx.program.hand.value
        bones = [f"{prefix}UpperArm", f"{prefix}LowerArm", f"{prefix}Hand"]
    else:
        return {}
    values = _angular_kinematics(ctx.frames, bones)
    return dict(zip(_ANGULAR_KEYS, values))


def _is_handoff(ctx: AnalysisContext) -> bool:
    from ..models import ObjectAction

    return (
        ctx.intent == Intent.OBJECT_INTERACTION
        and ctx.object_action == ObjectAction.HANDOFF
    )


def analyze(
    clip: ClipResult,
    program: MotionProgram,
    scene: SceneManifest,
) -> dict[str, Any]:
    """Compute every metric the analysis layer owns for a finished clip.

    ``program`` and ``scene`` must be the effective, post-override pair — the
    same one ``compile_motion`` used. Use
    :func:`rigby_poc.analysis.artifacts.load_analysis_inputs` when reading a
    stored result, because the store persists the pre-override program.
    """

    ctx = AnalysisContext.from_clip(clip, program, scene)
    return analyze_context(ctx)


def analyze_context(ctx: AnalysisContext) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if ctx.intent == Intent.UNSUPPORTED or not ctx.frames:
        return metrics

    metrics.update(
        safety_metrics(ctx.frames, allow_root_motion=ctx.allow_root_motion)
    )

    if ctx.intent == Intent.COMPOSITE:
        metrics.update(
            intra_hand_contact_metrics(ctx.frames, ctx.phase_ranges, ctx.program)
        )
        metrics.update(
            semantic_cycle_metrics(ctx.frames, ctx.phase_ranges, ctx.program)
        )
        metrics.update(parallel_forearm_metrics(ctx.frames, ctx.phase_ranges))
    elif ctx.intent == Intent.FULL_BODY:
        metrics.update(
            semantic_cycle_metrics(ctx.frames, ctx.phase_ranges, ctx.program)
        )

    if not _is_handoff(ctx):
        metrics.update(_angular_metrics(ctx))

    if ctx.intent in {Intent.GESTURE, Intent.STRIKE}:
        structure = evaluate_gesture_structure(
            ctx.frames,
            ctx.program.hand,
            ctx.presentation_ranges,
        )
        metrics.update(structure)
        if ctx.intent == Intent.GESTURE:
            metrics.update(
                shake_joint_oscillation_metrics(
                    ctx.frames,
                    ctx.program.hand,
                    ctx.ranges_for_kind(PrimitiveKind.SHAKE.value),
                )
            )
        # The gesture path folds the wrist violations into the shared joint
        # limit counter and republishes the collision count under the generic
        # key. Reproduce both, in that order.
        metrics["joint_limit_violations"] += structure[
            "wrist_swing_twist_limit_violations"
        ]
        metrics["unresolved_non_hand_collisions"] = structure["self_collision_frames"]

    return metrics


def owned_metric_keys(program: MotionProgram) -> frozenset[str]:
    """The metric keys :func:`analyze` is responsible for, for this program.

    Membership is program-dependent because several checks emit nothing when
    their assertion or primitive is absent. The equivalence harness compares
    exactly these keys against the compiler's output.
    """

    from .equivalence import owned_metric_keys as _owned

    return _owned(program)


def validate(metrics: dict[str, Any], program: MotionProgram) -> list[CheckResult]:
    """Turn owned metrics into addressable check verdicts.

    Same thresholds and same ordering as the compiler's ``structural_failures``
    list; this is a typed view of the checks the analysis layer owns, not an
    additional opinion.
    """

    checks: list[CheckResult] = []
    if "max_wrist_swing_rad" in metrics and "quality_limits" in metrics:
        checks.extend(gesture_structure_checks(metrics))
    checks.extend(parallel_forearm_checks(metrics))
    checks.extend(intra_hand_contact_checks(program, metrics))
    checks.extend(semantic_cycle_checks(program, metrics))
    checks.extend(safety_checks(metrics))
    return checks


def structural_failures(
    metrics: dict[str, Any], program: MotionProgram
) -> list[str]:
    """The moved failure strings, in the order the compiler appends them."""

    failures = list(parallel_forearm_failures(metrics))
    failures.extend(intra_hand_contact_failures(program, metrics))
    failures.extend(semantic_cycle_failures(program, metrics))
    return failures


__all__ = [
    "ANATOMY",
    "BODY_ANALYZERS",
    "CONTRACT",
    "LAYERS",
    "OBJECT_ANALYZERS",
    "PHYSICS",
    "RIG_PROFILE",
    "SIGNAL",
    "AnalysisContext",
    "Analyzer",
    "AnalyzerEntry",
    "CheckResult",
    "CheckStatus",
    "analyze",
    "analyze_context",
    "arm_landmarks",
    "body_analyzer",
    "count_check",
    "deferred_actions",
    "evaluate_gesture_structure",
    "gesture_structure_checks",
    "identity_bones",
    "identity_pose",
    "intra_hand_contact_checks",
    "intra_hand_contact_failures",
    "intra_hand_contact_metrics",
    "lower_bound_check",
    "object_analyzer",
    "owned_metric_keys",
    "parallel_forearm_checks",
    "parallel_forearm_failures",
    "parallel_forearm_metrics",
    "quality_reference",
    "rig_profile",
    "safety_checks",
    "safety_metrics",
    "saturating_severity",
    "semantic_cycle_assertion",
    "semantic_cycle_checks",
    "semantic_cycle_failures",
    "semantic_cycle_metrics",
    "shake_joint_oscillation_metrics",
    "skipped",
    "structural_failures",
    "swing_twist_angles",
    "unregistered_actions",
    "upper_bound_check",
    "validate",
]
