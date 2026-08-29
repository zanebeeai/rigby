"""Pure post-hoc analysis of a finished clip.

``analyze(clip, program, scene)`` computes metrics from frames alone: no server,
no browser, no API key, no recompilation. ``validate(metrics, program, frames,
fps=...)`` turns a finished clip into individually addressable
:class:`CheckResult` verdicts -- most of them read only the metrics, and the
per-DOF range-of-motion layer reads the frames.

Extraction is staged (see ``docs/plans/02-analysis-layer.md`` §4). This module
owns exactly what PR 02a moved; :func:`owned_metric_keys` reports which keys
that is for a given program, and the equivalence harness asserts those keys are
byte-identical to what ``compiler.py`` produces today. The per-action metric
blocks are still computed inside the compiler and are declared, unported, in
:mod:`rigby_poc.analysis.registry`.
"""

from __future__ import annotations

from typing import Any

from ..models import ClipFrame, ClipResult, Intent, MotionProgram, SceneManifest
from .anatomy import rom_checks
from .composite import composite_metrics
from .contact import (
    intra_hand_contact_checks,
    intra_hand_contact_failures,
    intra_hand_contact_metrics,
)
from .context import AnalysisContext, root_drift_policy, root_motion_allowed
from .contract import (
    ANATOMY,
    CONTRACT,
    LAYERS,
    PHYSICS,
    SIGNAL,
    CheckResult,
    CheckStatus,
    binary_check,
    count_check,
    lower_bound_check,
    saturating_margin,
    saturating_severity,
    skipped,
    upper_bound_check,
)
from .forearm import (
    parallel_forearm_checks,
    parallel_forearm_failures,
    parallel_forearm_metrics,
)
from .full_body import (
    BODY_ENTRIES,
    commanded_root_yaw_rad,
    full_body_metrics,
)
from .gesture import (
    _angular_kinematics,
    arm_landmarks,
    evaluate_gesture_structure,
    gesture_structure_checks,
    quality_reference,
    shake_joint_oscillation_metrics,
    swing_twist_angles,
)
from .hand import assertion_frame_for, final_hand_shape, hand_metrics
from .objects import carried_object_id, handoff_metrics
from .physics import (
    FootContacts,
    center_of_mass,
    center_of_mass_series,
    foot_contacts,
    ground_height,
    physics_checks,
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
from .safety import clip_contract_violations, safety_checks, safety_metrics
from .score import (
    CompositeScore,
    FamilyScore,
    LayerScore,
    check_family,
    clip_composite,
    composite_score,
)
from .semantic import (
    semantic_cycle_assertion,
    semantic_cycle_checks,
    semantic_cycle_failures,
    semantic_cycle_metrics,
)
from .signal_quality import (
    LIMB_CHAINS,
    bone_activity,
    signal_checks,
    spectral_arc_length,
)

# Bone sets each compile path feeds to the angular-kinematics pass. Keyed by
# intent because the path, not the action, chooses them. The whole-body list
# lives with its own pass in ``full_body.balance``.
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

    Whole body has its own copy inside ``full_body.balance``, because that pass
    owns the ordering of every block it runs. The composite path folds per-hand
    structures instead, so it is excluded here and stays with the compiler until
    02c.
    """

    if ctx.intent == Intent.SEQUENCE:
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

    # A path ported whole computes its own safety block, in the compiler's own
    # order, so it returns before the shared call below rather than extending
    # it. Not tidiness: safety_metrics traverses every frame, and running it
    # twice per analysis was a quarter of the whole-body pass when 02b first
    # measured it. 02c reintroduced the same duplication for composite by
    # returning after the shared call instead of before it.
    if ctx.intent == Intent.FULL_BODY:
        return full_body_metrics(ctx)
    if ctx.intent == Intent.COMPOSITE:
        return composite_metrics(ctx)
    if _is_handoff(ctx):
        # 02d part 1. Not a relocation -- see analysis.objects.
        return handoff_metrics(ctx)

    if ctx.intent in {Intent.GESTURE, Intent.STRIKE, Intent.GRAB}:
        # 02c ported these too. The MuJoCo grasp block and the GRAB structural
        # branch stay with the compiler by design -- see analysis.hand.
        return hand_metrics(ctx)

    metrics.update(safety_metrics(ctx.frames, allow_root_motion=ctx.allow_root_motion))

    if not _is_handoff(ctx):
        metrics.update(_angular_metrics(ctx))

    return metrics


def owned_metric_keys(program: MotionProgram) -> frozenset[str]:
    """The metric keys :func:`analyze` is responsible for, for this program.

    Membership is program-dependent because several checks emit nothing when
    their assertion or primitive is absent. The equivalence harness compares
    exactly these keys against the compiler's output.
    """

    from .equivalence import owned_metric_keys as _owned

    return _owned(program)


def validate(
    metrics: dict[str, Any],
    program: MotionProgram,
    frames: list[ClipFrame],
    *,
    fps: float,
) -> list[CheckResult]:
    """Turn a finished clip into addressable check verdicts.

    For every check with a compiler counterpart this is the same threshold and
    the same ordering as the compiler's ``structural_failures`` list -- a typed
    view of what the analysis layer owns, not an additional opinion. The
    range-of-motion layer (plan 04 §3.6) has no counterpart; it is appended
    after them.

    ``frames`` and ``fps`` are required rather than optional, and that is the
    substance of the wiring rather than a detail of it. 04c shipped ROM
    enforcement **dark** -- 156 limits, 82 of them enforced, and no caller in
    ``src/`` or ``evals/``. An optional ``frames=None`` would have left the one
    real caller, the mutation detection matrix, still not passing them, and ROM
    would have stayed dark behind the appearance of being wired. That is the
    shape ``docs/testing.md`` calls a gate structurally incapable of failing.

    Note what this does **not** do: it does not write ``structural_valid``.
    Nothing in ``src/`` or ``evals/`` calls this function today, so the ROM
    verdicts reach the typed check surface -- which ``evals/mutations/`` and
    plan 10's layers read -- and not the compiler's accept decision. Wiring ROM
    into ``structural_valid`` rejects 41 of 41 expected-valid corpus cases, 37
    of them on one elbow bound; that is a product decision and it is filed as
    one, not taken here.
    """

    checks: list[CheckResult] = []
    if "max_wrist_swing_rad" in metrics and "quality_limits" in metrics:
        checks.extend(gesture_structure_checks(metrics))
    checks.extend(parallel_forearm_checks(metrics))
    checks.extend(intra_hand_contact_checks(program, metrics))
    checks.extend(semantic_cycle_checks(program, metrics))
    checks.extend(
        safety_checks(metrics, policy=root_drift_policy(program))
    )
    checks.extend(rom_checks(frames, fps=fps))
    checks.extend(
        physics_checks(frames, fps=fps, root_policy=root_drift_policy(program))
    )
    checks.extend(signal_checks(frames, fps=fps))
    return checks


def validate_clip(clip: ClipResult, program: MotionProgram) -> list[CheckResult]:
    """:func:`validate` over a clip, which already carries its frames and rate.

    The convenience matters because the two arguments a caller can get wrong
    are the two this reads off the clip. ``fps`` is the clip's declared rate;
    note that frame spacing is piecewise-uniform rather than uniform (plan 02
    §Findings), so it is exact for nothing except the range-of-motion
    ``integral_deg_s``, which only ranks. The gate is the band, and the band
    does not read ``fps`` at all.
    """

    return validate(clip.metrics, program, clip.frames, fps=float(clip.fps))


def structural_failures(metrics: dict[str, Any], program: MotionProgram) -> list[str]:
    """The moved failure strings, in the order the compiler appends them."""

    failures = list(parallel_forearm_failures(metrics))
    failures.extend(intra_hand_contact_failures(program, metrics))
    failures.extend(semantic_cycle_failures(program, metrics))
    return failures


__all__ = [
    "ANATOMY",
    "BODY_ANALYZERS",
    "BODY_ENTRIES",
    "CONTRACT",
    "LAYERS",
    "LIMB_CHAINS",
    "OBJECT_ANALYZERS",
    "PHYSICS",
    "RIG_PROFILE",
    "SIGNAL",
    "AnalysisContext",
    "Analyzer",
    "AnalyzerEntry",
    "CheckResult",
    "CheckStatus",
    "CompositeScore",
    "FamilyScore",
    "FootContacts",
    "LayerScore",
    "analyze",
    "analyze_context",
    "arm_landmarks",
    "assertion_frame_for",
    "binary_check",
    "body_analyzer",
    "bone_activity",
    "carried_object_id",
    "center_of_mass",
    "center_of_mass_series",
    "check_family",
    "clip_composite",
    "clip_contract_violations",
    "commanded_root_yaw_rad",
    "composite_metrics",
    "composite_score",
    "count_check",
    "deferred_actions",
    "evaluate_gesture_structure",
    "final_hand_shape",
    "foot_contacts",
    "full_body_metrics",
    "gesture_structure_checks",
    "ground_height",
    "hand_metrics",
    "handoff_metrics",
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
    "physics_checks",
    "quality_reference",
    "rig_profile",
    "rom_checks",
    "root_drift_policy",
    "root_motion_allowed",
    "safety_checks",
    "safety_metrics",
    "saturating_margin",
    "saturating_severity",
    "semantic_cycle_assertion",
    "semantic_cycle_checks",
    "semantic_cycle_failures",
    "semantic_cycle_metrics",
    "shake_joint_oscillation_metrics",
    "signal_checks",
    "skipped",
    "spectral_arc_length",
    "structural_failures",
    "swing_twist_angles",
    "unregistered_actions",
    "upper_bound_check",
    "validate",
    "validate_clip",
]
