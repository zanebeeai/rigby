"""The detection curve, and the threshold plan 10 §5.2 makes the headline.

`CalibrationEvidence.detection_threshold` had no producer before this module: 10a
built the field and the gate clause, 06b built the sweeps, and nothing joined
them. This joins them with zero model spend — every number here comes from the
deterministic layer reading a mutated clip, so it is the instrument the grader's
own detection curve will later be compared against, not a stand-in for it.

**Two ways to get a plausible, monotonic, wrong curve, both of them live here.**

*Reading `status` instead of `band`.* Since 04c only **82 of 156** DOFs are
enforced, and `rom_checks` sets `status="fail"` only for an excursion that is
`beyond_max` **and** on an enforced DOF (`analysis/anatomy/rom.py:321`). An
out-of-band excursion on an unenforced DOF stays `status="pass"`, so a curve
built on `status` scores those as undetected at every severity — a manufactured
false negative that rises with severity exactly like a real detection curve.
`evals.mutations.checks.rom_detected` reads the `band` inside `measured` and is
the only permitted reader for an `anatomy.rom.*` target.

*Folding a refusal into a zero.* `spec.applies_to(clip)` has four outcomes, not
two, and three of them are not "undetected":

| `ok` | `static_target` | means | enters the denominator |
| --- | --- | --- | --- |
| False | — | the spec cannot perturb this clip | **no** — it is a skip with a reason |
| True | True | the target bone never moves here | **no** — scored apart as capability |
| True | False | a real excursion inside real motion | yes — this is a threshold point |

A refusal counted as a zero depresses the rate at every severity and moves the
threshold up, which reads as a *conservative* result and therefore never gets
questioned. The skip and its reason are carried instead, because "no detector
exists here" and "the harness declined" are different findings — the distinction
four lanes hit independently this push.

Static targets are the subtler half. They are genuinely applicable — the base
holds one pose and the mutated clip holds another, so a detector that can see the
bone separates them perfectly — but there is no motion for a threshold to sit
inside, so the severity axis is meaningless and the curve is a step. Pooling them
inflates the sweep; dropping them discards a real capability measurement. They
are scored, tagged, and reported beside the curve rather than inside it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import inf
from typing import Any

from evals.calibration_stats import (
    DEFAULT_CONFIDENCE,
    Interval,
    ProportionResult,
    clopper_pearson_lower,
    proportion,
)
from evals.mutations.checks import rom_detected
from evals.mutations.spec import MutationSpec, Severity
from evals.mutations.structural import gates_tripped
from rigby_poc.models import ClipResult

#: The rate a detection threshold is defined at (plan 10 §5.2).
DETECTION_LEVEL = 0.5

#: Prefix whose targets must be read through `band` rather than `status`.
BAND_READ_PREFIX = "anatomy.rom."

#: Prefix whose targets emit no ``CheckResult`` at all and must be read from
#: ``metrics["structural_failures"]`` through :func:`gates_tripped`.
#:
#: A **third** channel, not a variant of the other two. `anatomy.rom.*` reports
#: through `measured["band"]` and everything else through `status`, but both are
#: fields on a `CheckResult` that `validate` emitted. These gates append a string
#: to a metrics list and emit nothing, so `results.get(target)` is `None` for them
#: and the missing-target branch below would raise "was not measured" on a target
#: that was measured perfectly well.
STRUCTURAL_READ_PREFIX = "structural."


class DetectionError(RuntimeError):
    """The curve cannot be built honestly from what was measured."""


def clopper_pearson_upper(
    successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE
) -> float:
    """One-sided Clopper-Pearson **upper** bound, by the complement identity.

    `calibration_stats` ships only the lower bound because every gate clause reads
    one. A threshold interval needs both ends, and deriving the upper from the
    lower on the complementary count keeps a single implementation of the beta
    quantile rather than a second one that can drift from it.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    if n == 0:
        return 1.0
    return 1.0 - clopper_pearson_lower(n - successes, n, confidence)


@dataclass(frozen=True)
class PairOutcome:
    """What one `(spec, case)` pair produced, including the reasons it produced nothing.

    `detected` is `None` **exactly when** the pair was not applicable. That is not
    a third value of the same quantity — it is the absence of the measurement, and
    keeping it out of the bool prevents the fold this module exists to prevent.
    """

    case_id: str
    spec_id: str
    severity: Severity
    applicable: bool
    static_target: bool
    detected: bool | None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.applicable and self.detected is None:
            raise DetectionError(
                f"{self.spec_id}/{self.case_id}: an applicable pair must record a "
                f"verdict; None means not measured and would be scored as undetected"
            )
        if not self.applicable and self.detected is not None:
            raise DetectionError(
                f"{self.spec_id}/{self.case_id}: an inapplicable pair carries no "
                f"verdict, but this one records detected={self.detected!r}"
            )
        if not self.applicable and not self.reason:
            raise DetectionError(
                f"{self.spec_id}/{self.case_id}: a skip with no reason cannot tell "
                f"'no detector exists here' from 'the harness declined'"
            )

    @property
    def is_threshold_point(self) -> bool:
        """Applicable, and on a bone this clip actually moves."""
        return self.applicable and not self.static_target


@dataclass(frozen=True)
class LevelResult:
    """One severity level of a sweep, with its three populations kept apart."""

    severity: Severity
    #: Detection over moving-target pairs. The curve is built from this alone.
    threshold_points: ProportionResult
    #: Detection over static-target pairs. A capability result, never pooled in.
    static_target: ProportionResult
    #: Pairs the spec refused, and why. Never a zero.
    skipped: int
    skip_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "threshold_points": self.threshold_points.to_dict(),
            "static_target": self.static_target.to_dict(),
            "skipped": self.skipped,
            "skip_reasons": list(self.skip_reasons),
        }


def check_results_by_id(
    clip: ClipResult,
    program: Any,
    scene: Any,
) -> dict[str, Any]:
    """Every check id a mutated clip can be scored against.

    `analysis.validate` emits `anatomy.rom.*` since lane `analysis`'s 04d wired
    `rom_checks` into it. Before that it did not, and this function merged the
    two sources by hand so that a ROM target could not look like a check that
    never fires. The merge is gone because it is now a duplicate: `validate`
    calls `rom_checks(frames, fps=fps)` with the same arguments, so running it
    again here would decompose all 52 bones twice per clip for identical
    results.

    **Not `validate_clip(clip, program)`, deliberately, and the reason is worse
    than a metric-set difference.** `analyze(...)` is required here because **a
    mutated clip's `metrics` are the unmutated clip's**. `MutationSpec.apply`
    deep-copies and transforms `frames` only — no transform in
    `evals.mutations` touches `metrics`, measured: after a top-severity
    `rom_sweep` on `fullbody-dance`, `mutated.metrics == clip.metrics` is True
    while `mutated.frames != clip.frames`. So the convenience form would read
    the compiler's **pre-mutation** dict: the right metric set from the wrong
    clip. Every non-ROM target would score as undetected at every severity,
    while the ROM half of the curve still rose plausibly with severity — a
    systematic false negative wearing the shape of a real result, which is the
    defect class this module exists to keep out. Keeping the metrics source
    explicit is what makes 04d's "nothing else here moves" true.
    """
    return scored_clip(clip, program, scene).results


@dataclass(frozen=True)
class ScoredClip:
    """Both detector surfaces for one clip, from a single `analyze` pass.

    The two are kept together because they must describe the same clip: `results`
    holds what `validate` emitted and `metrics` holds the analysis dict the
    `structural.*` gates wrote into. Reading one from the mutated clip and the other
    from the base clip is the fold this module exists to prevent, and separate
    accessors make that combination expressible.
    """

    results: dict[str, Any]
    metrics: Mapping[str, Any]


def scored_clip(clip: ClipResult, program: Any, scene: Any) -> ScoredClip:
    """Analyse `clip` once and return both surfaces a target can be read through."""
    from rigby_poc.analysis import analyze, validate

    metrics = analyze(clip, program, scene)
    checks = validate(metrics, program, clip.frames, fps=float(clip.fps))
    return ScoredClip(
        results={item.id: item for item in checks}, metrics=metrics
    )


def target_detected(
    results: Mapping[str, Any],
    spec: MutationSpec,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> bool:
    """Whether any check this spec targets fired on the mutated clip.

    Routes by target id across **three** channels, not two: `anatomy.rom.*` through
    `measured["band"]`, `structural.*` through `metrics["structural_failures"]`, and
    everything else through `status`. A target the run emitted no result for raises —
    it was not measured, and returning `False` would publish an absence as a negative.

    `metrics` is required only for a `structural.*` target and is the **analysis**
    metrics of the clip being scored, never `clip.metrics`. Measured on four full-body
    corpus cases: a 40 deg `rightLowerLeg.flexion` injection trips up to six gates in
    `analyze(...)` output while `clip.metrics["structural_failures"]` stays byte-identical
    to the unmutated clip's, because the compiler wrote it before the mutation existed.
    Scoring from the wrong one would report zero detection at every severity for the whole
    namespace — the same systematic false negative `check_results_by_id` refuses on the
    metric-source axis, arriving through the structural channel instead.

    Omitting `metrics` for a structural target raises rather than defaulting to "clean":
    the whole namespace has a measured base rate of zero, so a silent `False` is exactly
    what a working detector looks like.
    """
    if not spec.targets:
        raise DetectionError(
            f"{spec.id}: declares no targets, so detection has nothing to read"
        )
    detected = False
    for target in spec.targets:
        if target.startswith(STRUCTURAL_READ_PREFIX):
            # Before the `results` lookup: these ids are never keys in it.
            if metrics is None:
                raise DetectionError(
                    f"{spec.id}: target {target!r} is read from "
                    f"metrics['structural_failures'], but no metrics were passed. "
                    f"Scoring it as undetected would be indistinguishable from a "
                    f"working detector, since this namespace's base rate is zero"
                )
            try:
                detected = detected or target in gates_tripped(metrics)
            except KeyError as error:
                raise DetectionError(
                    f"{spec.id}: target {target!r} has no detector on this clip's "
                    f"path -- {error.args[0]}"
                ) from error
            continue
        result = results.get(target)
        if result is None:
            raise DetectionError(
                f"{spec.id}: nothing emitted a result for target {target!r}, so it "
                f"was not measured. A missing target and an undetected mutation "
                f"produce the same cell and must not be scored the same"
            )
        if target.startswith(BAND_READ_PREFIX):
            detected = detected or rom_detected(result.measured)
            continue
        status = str(result.status)
        if status == "skip":
            # A skipped check ran and declined to measure: `contract.clip.root_drift`
            # skips on whole-body and sequence programs, which enable root motion and
            # so have no bound to apply (lane `analysis`, 04e). Scoring that as
            # `False` would fold "not measured" into "measured negative" -- the same
            # fold this module refuses on the ROM path, arriving through `status`
            # instead of through a missing `band`. The caller must exclude the pair
            # as unmeasured rather than count it as a miss.
            raise DetectionError(
                f"{spec.id}: target {target!r} reports status='skip' on this clip, so "
                f"it was not measured. Exclude the pair as unmeasured rather than "
                f"scoring it as undetected"
            )
        detected = detected or status == "fail"
    return detected


def outcome_for(
    spec: MutationSpec,
    clip: ClipResult,
    program: Any,
    scene: Any,
    *,
    case_id: str,
) -> PairOutcome:
    """Apply one spec to one clip and record what it measured, or why it did not."""
    verdict = spec.applies_to(clip)
    if not verdict.ok:
        return PairOutcome(
            case_id=case_id,
            spec_id=spec.id,
            severity=spec.severity,
            applicable=False,
            static_target=verdict.static_target,
            detected=None,
            reason=verdict.reason,
        )
    mutated = spec.apply(clip)
    scored = scored_clip(mutated, program, scene)
    return PairOutcome(
        case_id=case_id,
        spec_id=spec.id,
        severity=spec.severity,
        applicable=True,
        static_target=verdict.static_target,
        detected=target_detected(scored.results, spec, metrics=scored.metrics),
        reason="",
    )


def detection_curve(outcomes: Iterable[PairOutcome]) -> list[LevelResult]:
    """Group outcomes into one `LevelResult` per severity, ascending."""
    by_level: dict[Severity, list[PairOutcome]] = {}
    for outcome in outcomes:
        by_level.setdefault(outcome.severity, []).append(outcome)
    curve: list[LevelResult] = []
    for severity in sorted(by_level):
        items = by_level[severity]
        points = [item for item in items if item.is_threshold_point]
        statics = [item for item in items if item.applicable and item.static_target]
        skips = [item for item in items if not item.applicable]
        curve.append(
            LevelResult(
                severity=severity,
                threshold_points=proportion(
                    sum(1 for item in points if item.detected), len(points)
                ),
                static_target=proportion(
                    sum(1 for item in statics if item.detected), len(statics)
                ),
                skipped=len(skips),
                skip_reasons=tuple(sorted({item.reason for item in skips})),
            )
        )
    return curve


def detection_threshold(curve: Sequence[LevelResult]) -> Interval | None:
    """Severity at 50% detection, bracketed by the per-level confidence bounds.

    The estimate is the mildest level whose observed rate reaches 50%. The bounds
    come from the same levels read through their Clopper-Pearson interval rather
    than from a fitted curve: the **upper** end is the mildest level whose *lower*
    bound already clears 50% — the severity by which detection is established —
    and the **lower** end is the mildest level whose *upper* bound reaches 50% —
    the earliest severity at which it plausibly could be. Both are monotone in the
    level order, so `lower <= estimate <= upper` holds by construction.

    Returns `None` when no level reaches 50%. That is not missing data to be
    skipped: it is a detector that does not detect, and `_detection_criterion`
    fails the gate on it.

    `n` is the number of threshold points at the estimating level, not the sum
    over the sweep — the rate that located the threshold is the one whose sample
    size bounds it, and summing would state a denominator no single rate had.
    """
    if not curve:
        raise DetectionError("no levels: an empty sweep locates no threshold")
    measured = [level for level in curve if level.threshold_points.n > 0]
    if not measured:
        raise DetectionError(
            "no level has a single threshold point: every pair was skipped or "
            "static, so this sweep measures capability and not a threshold"
        )
    estimate_at = next(
        (
            level
            for level in measured
            if level.threshold_points.estimate >= DETECTION_LEVEL
        ),
        None,
    )
    if estimate_at is None:
        return None
    upper_at = next(
        (
            level
            for level in measured
            if level.threshold_points.lower_bound_95 >= DETECTION_LEVEL
        ),
        None,
    )
    lower_at = next(
        (
            level
            for level in measured
            if clopper_pearson_upper(
                level.threshold_points.successes, level.threshold_points.n
            )
            >= DETECTION_LEVEL
        ),
        None,
    )
    return Interval(
        estimate=estimate_at.severity,
        # No level's lower bound clears 50%: detection is observed but not
        # established anywhere in the sweep, so the threshold is unbounded above.
        # `inf` drives `_detection_criterion` to UNDERPOWERED, which is the honest
        # reading -- the sweep is too coarse or too small, not the grader too dull.
        lower_bound_95=lower_at.severity if lower_at else estimate_at.severity,
        upper_bound_95=upper_at.severity if upper_at else inf,
        n=estimate_at.threshold_points.n,
    )


def capability_rate(curve: Sequence[LevelResult]) -> ProportionResult:
    """Detection over every static-target pair in the sweep, pooled across levels.

    Pooling is correct **here and only here**: the severity axis is meaningless
    for a static target, so the levels are repeats of one measurement rather than
    points on a curve. That is the same reason they are kept out of the threshold.
    """
    successes = sum(level.static_target.successes for level in curve)
    n = sum(level.static_target.n for level in curve)
    return proportion(successes, n)


def skip_ledger(curve: Sequence[LevelResult]) -> dict[str, int]:
    """Every distinct refusal reason in the sweep, and how many levels carried it.

    Reported rather than counted: a sweep that skipped most of its pairs can still
    produce a clean-looking threshold from the few that survived, and the reasons
    are what make that visible.
    """
    ledger: dict[str, int] = {}
    for level in curve:
        for reason in level.skip_reasons:
            ledger[reason] = ledger.get(reason, 0) + 1
    return ledger


def unmutated_baseline(
    specs: Sequence[MutationSpec],
    cases: Sequence[tuple[str, ClipResult, Any, Any]],
) -> ProportionResult:
    """Detection rate on the **unmutated** clip, over the cases the sweep applies to.

    The baseline every detection rate has to be stated against, and the half a
    type cannot supply. `ProportionResult` structurally carries an n and a bound;
    nothing structurally carries a baseline, so a detection rate of 0.87 with a
    large n and a tight interval is indistinguishable from a detector that fires
    on everything until this number sits beside it.

    Measured rather than assumed to be zero. For `anatomy.rom.*` it *should* be
    zero, because `rom_guard` refuses a clip whose target DOF is already out of
    band — but that is a property of one family's guard, not of the sweep
    machinery, and the families whose guards do not exclude an already-failing
    clip are exactly the ones where this number will not be zero. Asserting the
    zero rather than measuring it would hide the case it exists to catch.

    Evaluated once per case rather than once per level: the unmutated clip does
    not vary with severity, so a per-level baseline would restate one measurement
    seven times and inflate its n sevenfold.
    """
    if not specs:
        raise DetectionError("no specs: a sweep with no levels has no baseline")
    template = specs[0]
    successes = 0
    n = 0
    for _case_id, clip, program, scene in cases:
        if not template.applies_to(clip).ok:
            continue
        n += 1
        scored = scored_clip(clip, program, scene)
        if target_detected(scored.results, template, metrics=scored.metrics):
            successes += 1
    return proportion(successes, n)


def sweep_outcomes(
    specs: Sequence[MutationSpec],
    cases: Sequence[tuple[str, ClipResult, Any, Any]],
    *,
    on_error: Callable[[str, str, Exception], None] | None = None,
) -> list[PairOutcome]:
    """Every `(spec, case)` outcome for one sweep.

    `cases` is `(case_id, clip, program, scene)`, already compiled by the caller:
    compiling here would hide the cost of the arm inside the arm.
    """
    outcomes: list[PairOutcome] = []
    for case_id, clip, program, scene in cases:
        for spec in specs:
            try:
                outcomes.append(
                    outcome_for(spec, clip, program, scene, case_id=case_id)
                )
            except Exception as error:
                if on_error is None:
                    raise
                on_error(spec.id, case_id, error)
    return outcomes


__all__ = [
    "BAND_READ_PREFIX",
    "DETECTION_LEVEL",
    "STRUCTURAL_READ_PREFIX",
    "DetectionError",
    "LevelResult",
    "PairOutcome",
    "ScoredClip",
    "capability_rate",
    "check_results_by_id",
    "clopper_pearson_upper",
    "detection_curve",
    "detection_threshold",
    "outcome_for",
    "scored_clip",
    "skip_ledger",
    "sweep_outcomes",
    "target_detected",
    "unmutated_baseline",
]
