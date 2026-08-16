"""Calibration statistics and the grader acceptance gate.

Pure functions, zero model calls, no I/O. This module is the scoring half of the
eval redesign (docs/plans/10-eval-redesign.md sections 5.2 through 5.5). It exists
before any grader does, deliberately: every grader built later is built against
scoring logic that already provably rejects constant predictors.

Two ideas carry the whole design.

1. Every published rate is gated on its **95% Clopper-Pearson lower bound**, never on
   its point estimate. A lower bound is a joint statement about the score and the
   sample size, so an undersized suite cannot pass regardless of how well it scores.
   Twenty-eight negatives cannot certify 0.90 even at a perfect score; thirty-two can,
   at 0.911.

2. Discrimination is scored **per stratum** against a computed majority-class
   baseline, never pooled. The bug that invalidated the previous calibration was that
   every scored item shared one ground-truth label, so "always pick the base clip"
   scored 0.9 against a 0.8 gate. Pooling strata A and B does not fix that on its own
   -- see `score_strata`.

Sign convention for the unary arms, which is inverted from the usual defect-detection
reading and matches section 5.5 of the plan: the **positive class is "the clip is
good"**. Sensitivity is therefore the rate of correctly accepting unmutated clips, and
specificity the rate of correctly rejecting mutated ones. "Always accept" drives
specificity to zero; "always reject" drives sensitivity to zero.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite, sqrt
from typing import Any, Literal

from scipy.stats import beta

DEFAULT_CONFIDENCE = 0.95
_REQUIRED_N_SEARCH_LIMIT = 10_000

Choice = Literal["first", "second", "tie"]
Stratum = Literal["A", "B", "C"]

DIRECTIONAL_STRATA: tuple[Stratum, ...] = ("A", "B")


# --------------------------------------------------------------------------- #
# Proportions
# --------------------------------------------------------------------------- #


def clopper_pearson_lower(
    successes: int,
    n: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """One-sided Clopper-Pearson lower confidence bound for a binomial proportion.

    The returned bound `L` is the largest p for which observing `successes` or more
    out of `n` still has probability `1 - confidence`, i.e. it satisfies
    `P(X >= successes | p = L) == 1 - confidence` exactly. It is computed from the
    beta quantile identity `L = Beta(alpha; k, n - k + 1)`.

    Edges: `successes == 0` returns 0.0 (no evidence bounds the rate away from zero);
    `successes == n` returns `alpha ** (1 / n)`, so a perfect 32-of-32 gives 0.911.
    `n == 0` returns 0.0 -- no data supports no claim.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n == 0 or successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    return float(beta.ppf(alpha, successes, n - successes + 1))


@dataclass(frozen=True)
class ProportionResult:
    """A rate reported the only way this codebase permits: with its n and its bound.

    `estimate` is the raw success rate and `lower_bound_95` its one-sided 95%
    Clopper-Pearson lower bound. Gates read `lower_bound_95`; reports print all four
    fields, because a rate without its n and its interval is not a measurement.

    With `n == 0` both `estimate` and `lower_bound_95` are 0.0. That is a placeholder,
    not a measurement -- callers must check `n` before reading the estimate, and the
    gate does so via its minimum-n criterion.
    """

    successes: int
    n: int
    estimate: float
    lower_bound_95: float

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}")
        if not 0 <= self.successes <= self.n:
            raise ValueError(f"successes must be in [0, {self.n}], got {self.successes}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "n": self.n,
            "estimate": self.estimate,
            "lower_bound_95": self.lower_bound_95,
        }


def proportion(successes: int, n: int) -> ProportionResult:
    """Build a `ProportionResult` at 95% confidence."""
    return ProportionResult(
        successes=successes,
        n=n,
        estimate=(successes / n) if n else 0.0,
        lower_bound_95=clopper_pearson_lower(successes, n, DEFAULT_CONFIDENCE),
    )


def minimum_n_for_lower_bound(
    threshold: float,
    rate: float = 1.0,
    *,
    strict: bool = False,
    confidence: float = DEFAULT_CONFIDENCE,
) -> int | None:
    """Smallest n at which an observed `rate` clears `threshold` on its lower bound.

    This is the number that makes an underpowered suite actionable: it answers "how
    many more items would settle this?" rather than just reporting a false. Returns
    `None` when no sample size within the search limit suffices, which is the correct
    answer whenever `rate <= threshold` -- the bound converges to the rate from below,
    so a rate at or under the threshold is unreachable at any n.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    for n in range(1, _REQUIRED_N_SEARCH_LIMIT + 1):
        bound = clopper_pearson_lower(round(rate * n), n, confidence)
        if bound > threshold or (not strict and bound == threshold):
            return n
    return None


# --------------------------------------------------------------------------- #
# Baselines and agreement metrics
# --------------------------------------------------------------------------- #


def majority_class_baseline(labels: Sequence[object]) -> float:
    """Accuracy of the best constant predictor over `labels`.

    This is the number a grader must beat, and it is *computed from the suite* rather
    than assumed. A suite whose labels are all one value has a baseline of 1.0, which
    no grader can strictly exceed -- that is the design working, not a bug: such a
    suite is incapable of distinguishing a grader from a constant.
    """
    if not labels:
        raise ValueError("majority_class_baseline requires at least one label")
    counts = Counter(labels)
    return max(counts.values()) / len(labels)


def _paired(a: Sequence[object], b: Sequence[object], name_a: str, name_b: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{name_a} and {name_b} must be the same length, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError(f"{name_a} and {name_b} must be non-empty")


def balanced_accuracy(truth: Sequence[object], predicted: Sequence[object]) -> float:
    """Mean per-class recall over the classes present in `truth`.

    Unlike plain accuracy this is immune to class imbalance, so a constant predictor
    scores at chance (1 / number of classes) however skewed the suite is.
    """
    _paired(truth, predicted, "truth", "predicted")
    classes = sorted(set(truth), key=repr)
    recalls = []
    for label in classes:
        support = sum(1 for value in truth if value == label)
        hits = sum(1 for value, guess in zip(truth, predicted, strict=True) if value == label and guess == label)
        recalls.append(hits / support)
    return sum(recalls) / len(recalls)


def cohens_kappa(rater_a: Sequence[object], rater_b: Sequence[object]) -> float:
    """Cohen's kappa: agreement corrected for agreement expected by chance.

    Returns 0.0 when chance agreement is total (both raters constant), because in that
    case the observed agreement carries no information -- the same reason the gate
    refuses to credit constant predictors anywhere else.
    """
    _paired(rater_a, rater_b, "rater_a", "rater_b")
    n = len(rater_a)
    observed = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in set(counts_a) | set(counts_b))
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def matthews_corrcoef(truth: Sequence[bool], predicted: Sequence[bool]) -> float:
    """Matthews correlation coefficient for binary labels.

    Zero for every constant predictor regardless of class balance, which is exactly
    why it is reported alongside sensitivity and specificity: it collapses the whole
    degenerate class to a single recognisable number.
    """
    _paired(truth, predicted, "truth", "predicted")
    pairs = list(zip(truth, predicted, strict=True))
    true_positive = sum(1 for actual, guess in pairs if actual and guess)
    true_negative = sum(1 for actual, guess in pairs if not actual and not guess)
    false_positive = sum(1 for actual, guess in pairs if not actual and guess)
    false_negative = sum(1 for actual, guess in pairs if actual and not guess)
    denominator = sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if denominator == 0.0:
        return 0.0
    return (true_positive * true_negative - false_positive * false_negative) / denominator


# --------------------------------------------------------------------------- #
# Judgment records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UnaryJudgment:
    """One grader verdict on one clip.

    `is_good` is ground truth by construction: unmutated corpus clips that pass every
    deterministic gate are good, mutated clips are not. `accepted` is what the grader
    said.
    """

    clip_id: str
    is_good: bool
    accepted: bool


@dataclass(frozen=True)
class PairwiseJudgment:
    """One forced-choice verdict on one presented pair.

    `correct` is the position holding the member that should win, and is `None` only
    in stratum C, where the two members are deterministically equivalent and there is
    no correct answer. Stratum C pairs are presented in both orders and share a
    `pair_id`; strata A and B carry one record per presentation.
    """

    pair_id: str
    stratum: Stratum
    first_id: str
    second_id: str
    correct: Literal["first", "second"] | None
    predicted: Choice

    def chosen_id(self) -> str | None:
        if self.predicted == "first":
            return self.first_id
        if self.predicted == "second":
            return self.second_id
        return None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UnaryScores:
    """Sensitivity, specificity and two imbalance-proof summaries of the same data."""

    sensitivity: ProportionResult
    specificity: ProportionResult
    balanced_accuracy: float
    matthews_corrcoef: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensitivity": self.sensitivity.to_dict(),
            "specificity": self.specificity.to_dict(),
            "balanced_accuracy": self.balanced_accuracy,
            "matthews_corrcoef": self.matthews_corrcoef,
        }


def score_unary(records: Sequence[UnaryJudgment]) -> UnaryScores:
    """Score accept/reject verdicts. Positive class is "the clip is good"."""
    if not records:
        raise ValueError("score_unary requires at least one judgment")
    good = [record for record in records if record.is_good]
    bad = [record for record in records if not record.is_good]
    truth = [record.is_good for record in records]
    predicted = [record.accepted for record in records]
    return UnaryScores(
        sensitivity=proportion(sum(1 for record in good if record.accepted), len(good)),
        specificity=proportion(sum(1 for record in bad if not record.accepted), len(bad)),
        balanced_accuracy=balanced_accuracy(truth, predicted),
        matthews_corrcoef=matthews_corrcoef(truth, predicted),
    )


@dataclass(frozen=True)
class StratumScore:
    """Directional accuracy within one stratum, against that stratum's own baseline.

    `baseline` is the majority-class accuracy over the *correct positions* in this
    stratum: with balanced presentation order it is 0.50, and the gate requires the
    lower bound to exceed it strictly.
    """

    stratum: Stratum
    accuracy: ProportionResult
    baseline: float
    ties: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "accuracy": self.accuracy.to_dict(),
            "baseline": self.baseline,
            "ties": self.ties,
        }


def score_strata(records: Sequence[PairwiseJudgment]) -> dict[str, StratumScore]:
    """Directional accuracy per stratum, never pooled.

    Pooling is what makes constant predictors look competent. Over a balanced A + B
    pool, "always pick the base clip" is correct on all of A and at chance on B, for
    0.75 against a 0.50 pooled baseline -- it beats the pooled baseline comfortably
    while being exactly the predictor the whole design exists to reject. Scored per
    stratum it lands at 0.5 in B and fails there, which is the point of stratum B.

    Ties count as incorrect: in strata A and B a correct answer exists by
    construction, so declining to give one is a miss, not a neutral outcome.
    """
    scores: dict[str, StratumScore] = {}
    for stratum in DIRECTIONAL_STRATA:
        members = [record for record in records if record.stratum == stratum]
        if not members:
            continue
        missing = [record.pair_id for record in members if record.correct is None]
        if missing:
            raise ValueError(f"stratum {stratum} records must carry a correct answer: {missing[0]}")
        hits = sum(1 for record in members if record.predicted == record.correct)
        scores[stratum] = StratumScore(
            stratum=stratum,
            accuracy=proportion(hits, len(members)),
            baseline=majority_class_baseline([record.correct for record in members]),
            ties=sum(1 for record in members if record.predicted == "tie"),
        )
    return scores


def score_fabricated_preference(records: Sequence[PairwiseJudgment]) -> ProportionResult:
    """Stable fabricated preference over stratum C.

    A stratum C pair holds two clips the deterministic layer cannot separate, shown in
    both orders. Naming the same clip both times is a preference the evidence does not
    support -- fabricated, not perceived. The rate is over complete order pairs;
    incomplete pairs are dropped rather than counted as clean.
    """
    by_pair: dict[str, list[PairwiseJudgment]] = {}
    for record in records:
        if record.stratum == "C":
            by_pair.setdefault(record.pair_id, []).append(record)
    complete = [group for group in by_pair.values() if len(group) == 2]
    fabricated = sum(
        1
        for group in complete
        if group[0].chosen_id() is not None and group[0].chosen_id() == group[1].chosen_id()
    )
    return proportion(fabricated, len(complete))


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


class CriterionStatus(StrEnum):
    """Why a criterion is not passing, which matters as much as whether it is.

    An underpowered suite and a failed grader are different states with different
    remedies -- collect more items, versus fix or delete the grader -- and collapsing
    both to `False` is how a suite ends up being quietly enlarged until it passes.
    """

    PASSED = "passed"
    FAILED = "failed"
    UNDERPOWERED = "underpowered"


@dataclass(frozen=True)
class Interval:
    """A two-sided estimate supplied by the caller, e.g. a detection threshold."""

    estimate: float
    lower_bound_95: float
    upper_bound_95: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "lower_bound_95": self.lower_bound_95,
            "upper_bound_95": self.upper_bound_95,
            "n": self.n,
        }


@dataclass(frozen=True)
class Criterion:
    """One clause of the gate, with everything needed to report it honestly."""

    name: str
    status: CriterionStatus
    observed: float
    threshold: float
    comparison: str
    n: int
    detail: str
    required_n: int | None = None

    @property
    def passed(self) -> bool:
        return self.status is CriterionStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": str(self.status),
            "observed": self.observed,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "n": self.n,
            "detail": self.detail,
            "required_n": self.required_n,
        }


@dataclass(frozen=True)
class GateThresholds:
    """Threshold values for the gate. No defaults: plan 08 owns the real numbers.

    `min_arm_n` is a floor applied to every scored arm and stratum independently. The
    lower-bound mechanism already makes small samples fail, so this is belt and
    braces -- but it is what turns "n = 3, scored 3/3" from a near-miss into an
    explicit underpowered verdict.
    """

    max_acceptable_severity: float
    min_sensitivity: float
    min_specificity: float
    min_kappa: float
    max_variance: float
    min_ablation_delta: float
    max_fabricated_preference: float
    min_arm_n: int


PROVISIONAL_GATE_THRESHOLDS = GateThresholds(
    max_acceptable_severity=0.35,
    min_sensitivity=0.85,
    min_specificity=0.85,
    min_kappa=0.60,
    max_variance=0.05,
    min_ablation_delta=0.15,
    max_fabricated_preference=0.30,
    min_arm_n=19,
)
"""PROVISIONAL. Placeholder values so 10d has something to run against.

Only `min_arm_n` is derived: 19 is the smallest n whose *perfect* score clears 0.85
(19 x 0.85 -> lower bound 0.8541; 18 cannot reach it at any score), so a smaller arm
could never satisfy `min_sensitivity` anyway. Every other value is a guess and must be
replaced from `thresholds.v1.json` when plan 08 lands.
"""


@dataclass(frozen=True)
class CalibrationEvidence:
    """Everything the gate reads. Built by `build_evidence` from judgment records.

    `detection_threshold` is `None` when the severity sweep never reached 50%
    detection. That is not missing data to be skipped -- it is a grader that does not
    detect, and the gate fails it.
    """

    grader: str
    unary: UnaryScores
    strata: dict[str, StratumScore]
    fabricated_preference: ProportionResult
    detection_threshold: Interval | None
    deterministic_kappa: float
    stability_variance: float
    ablation_response: float
    degenerate_predictors_all_fail: bool


def build_evidence(
    grader: str,
    *,
    unary: Sequence[UnaryJudgment],
    pairwise: Sequence[PairwiseJudgment],
    detection_threshold: Interval | None,
    deterministic_kappa: float,
    stability_variance: float,
    ablation_response: float,
    degenerate_predictors_all_fail: bool,
) -> CalibrationEvidence:
    """Score raw judgment records into the evidence the gate consumes."""
    return CalibrationEvidence(
        grader=grader,
        unary=score_unary(unary),
        strata=score_strata(pairwise),
        fabricated_preference=score_fabricated_preference(pairwise),
        detection_threshold=detection_threshold,
        deterministic_kappa=deterministic_kappa,
        stability_variance=stability_variance,
        ablation_response=ablation_response,
        degenerate_predictors_all_fail=degenerate_predictors_all_fail,
    )


def _bounded_criterion(
    name: str,
    result: ProportionResult,
    threshold: float,
    *,
    strict: bool,
    min_arm_n: int,
) -> Criterion:
    """Triage one lower-bound criterion into passed / failed / underpowered.

    The bound converges to the point estimate from below as n grows, which gives the
    rule its shape:

    - bound already clears the threshold -> passed;
    - estimate clears it but the bound does not -> underpowered, and `required_n` says
      how many items at this rate would settle it;
    - estimate does not clear it -> failed, because no sample size rescues a rate that
      is itself under the line.

    An arm below `min_arm_n` is underpowered regardless of score: under that floor the
    suite cannot tell a bad grader from an unlucky one, and reporting a confident
    failure would be as dishonest as reporting a confident pass.
    """
    comparison = ">" if strict else ">="
    cleared = result.lower_bound_95 > threshold if strict else result.lower_bound_95 >= threshold
    estimate_clears = result.estimate > threshold if strict else result.estimate >= threshold
    if result.n < min_arm_n:
        return Criterion(
            name=name,
            status=CriterionStatus.UNDERPOWERED,
            observed=result.lower_bound_95,
            threshold=threshold,
            comparison=comparison,
            n=result.n,
            detail=f"n = {result.n} is below the minimum of {min_arm_n} for a scored arm",
            required_n=min_arm_n,
        )
    if cleared:
        return Criterion(
            name=name,
            status=CriterionStatus.PASSED,
            observed=result.lower_bound_95,
            threshold=threshold,
            comparison=comparison,
            n=result.n,
            detail=(
                f"{result.successes}/{result.n} = {result.estimate:.3f}, "
                f"95% LCB {result.lower_bound_95:.3f} {comparison} {threshold:.3f}"
            ),
        )
    if estimate_clears:
        required = minimum_n_for_lower_bound(threshold, result.estimate, strict=strict)
        return Criterion(
            name=name,
            status=CriterionStatus.UNDERPOWERED,
            observed=result.lower_bound_95,
            threshold=threshold,
            comparison=comparison,
            n=result.n,
            detail=(
                f"{result.successes}/{result.n} = {result.estimate:.3f} clears {threshold:.3f} "
                f"but the 95% LCB {result.lower_bound_95:.3f} does not; the suite is too small to tell"
            ),
            required_n=required,
        )
    return Criterion(
        name=name,
        status=CriterionStatus.FAILED,
        observed=result.lower_bound_95,
        threshold=threshold,
        comparison=comparison,
        n=result.n,
        detail=(
            f"{result.successes}/{result.n} = {result.estimate:.3f} does not clear {threshold:.3f}; "
            f"no sample size recovers a rate at or below the threshold"
        ),
    )


def _scalar_criterion(
    name: str,
    observed: float,
    threshold: float,
    *,
    comparison: str,
    detail: str,
    n: int = 0,
) -> Criterion:
    """A criterion with no interval attached: it passes or it fails, nothing between."""
    cleared = observed <= threshold if comparison == "<=" else observed >= threshold
    return Criterion(
        name=name,
        status=CriterionStatus.PASSED if cleared else CriterionStatus.FAILED,
        observed=observed,
        threshold=threshold,
        comparison=comparison,
        n=n,
        detail=detail,
    )


def _detection_criterion(interval: Interval | None, max_severity: float) -> Criterion:
    if interval is None:
        return Criterion(
            name="detection_threshold",
            status=CriterionStatus.FAILED,
            observed=float("inf"),
            threshold=max_severity,
            comparison="<=",
            n=0,
            detail="detection threshold is not estimable: the sweep never reached 50% detection",
        )
    if not isfinite(interval.estimate):
        return Criterion(
            name="detection_threshold",
            status=CriterionStatus.FAILED,
            observed=interval.estimate,
            threshold=max_severity,
            comparison="<=",
            n=interval.n,
            detail="detection threshold is not finite: the grader does not detect at any severity",
        )
    if interval.upper_bound_95 <= max_severity:
        status, detail = (
            CriterionStatus.PASSED,
            f"severity at 50% detection {interval.estimate:.3f}, 95% upper bound "
            f"{interval.upper_bound_95:.3f} <= {max_severity:.3f}, n = {interval.n}",
        )
    elif interval.estimate <= max_severity:
        status, detail = (
            CriterionStatus.UNDERPOWERED,
            f"severity at 50% detection {interval.estimate:.3f} clears {max_severity:.3f} but the "
            f"95% upper bound {interval.upper_bound_95:.3f} does not; the sweep is too coarse or too small",
        )
    else:
        status, detail = (
            CriterionStatus.FAILED,
            f"severity at 50% detection {interval.estimate:.3f} exceeds {max_severity:.3f}: "
            f"the grader only detects defects that are already obvious",
        )
    return Criterion(
        name="detection_threshold",
        status=status,
        observed=interval.upper_bound_95,
        threshold=max_severity,
        comparison="<=",
        n=interval.n,
        detail=detail,
    )


@dataclass(frozen=True)
class GateReport:
    """The gate's verdict, with every clause kept for the report.

    `status` is FAILED if any clause failed, otherwise UNDERPOWERED if any clause was
    underpowered, otherwise PASSED. A genuine failure dominates: a suite that is both
    too small and demonstrably broken reads as broken, because enlarging it is not the
    remedy.
    """

    grader: str
    status: CriterionStatus
    criteria: tuple[Criterion, ...]

    @property
    def passed(self) -> bool:
        return self.status is CriterionStatus.PASSED

    @property
    def underpowered(self) -> bool:
        return self.status is CriterionStatus.UNDERPOWERED

    def criterion(self, name: str) -> Criterion:
        for entry in self.criteria:
            if entry.name == name:
                return entry
        raise KeyError(f"no criterion named {name!r} in {[entry.name for entry in self.criteria]}")

    def failures(self) -> tuple[Criterion, ...]:
        return tuple(entry for entry in self.criteria if entry.status is CriterionStatus.FAILED)

    def underpowered_criteria(self) -> tuple[Criterion, ...]:
        return tuple(entry for entry in self.criteria if entry.status is CriterionStatus.UNDERPOWERED)

    def summary(self) -> str:
        """One line naming the verdict and what to do about it."""
        if self.status is CriterionStatus.PASSED:
            return f"{self.grader}: passed {len(self.criteria)} criteria"
        if self.status is CriterionStatus.FAILED:
            names = ", ".join(entry.name for entry in self.failures())
            return f"{self.grader}: FAILED on {names} -- the grader is not fit for use"
        entries = self.underpowered_criteria()
        needed = [entry.required_n for entry in entries if entry.required_n is not None]
        target = f"; needs n >= {max(needed)}" if needed else ""
        names = ", ".join(entry.name for entry in entries)
        return f"{self.grader}: UNDERPOWERED on {names} -- no verdict, the suite is too small{target}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "grader": self.grader,
            "status": str(self.status),
            "passed": self.passed,
            "summary": self.summary(),
            "criteria": [entry.to_dict() for entry in self.criteria],
        }


def evaluate_gate(evidence: CalibrationEvidence, thresholds: GateThresholds) -> GateReport:
    """The grader acceptance gate of plan 10, section 5.3.

    Every proportion clause is gated on its 95% lower bound, and the discrimination
    clause is expanded to one criterion per directional stratum against that stratum's
    own computed baseline -- pooling the strata would let "always pick the base clip"
    through, which is the exact bug this gate exists to prevent. Stratum C contributes
    a fabricated-preference ceiling when the suite carries it.
    """
    criteria: list[Criterion] = [
        _detection_criterion(evidence.detection_threshold, thresholds.max_acceptable_severity),
        _bounded_criterion(
            "sensitivity",
            evidence.unary.sensitivity,
            thresholds.min_sensitivity,
            strict=False,
            min_arm_n=thresholds.min_arm_n,
        ),
        _bounded_criterion(
            "specificity",
            evidence.unary.specificity,
            thresholds.min_specificity,
            strict=False,
            min_arm_n=thresholds.min_arm_n,
        ),
    ]

    if not evidence.strata:
        criteria.append(
            Criterion(
                name="discrimination",
                status=CriterionStatus.UNDERPOWERED,
                observed=0.0,
                threshold=0.5,
                comparison=">",
                n=0,
                detail="no directional strata were scored",
                required_n=thresholds.min_arm_n,
            )
        )
    for stratum in DIRECTIONAL_STRATA:
        score = evidence.strata.get(stratum)
        if score is None:
            criteria.append(
                Criterion(
                    name=f"discrimination.{stratum}",
                    status=CriterionStatus.UNDERPOWERED,
                    observed=0.0,
                    threshold=0.5,
                    comparison=">",
                    n=0,
                    detail=f"stratum {stratum} is absent; a suite without it cannot exclude constant predictors",
                    required_n=thresholds.min_arm_n,
                )
            )
            continue
        criteria.append(
            _bounded_criterion(
                f"discrimination.{stratum}",
                score.accuracy,
                score.baseline,
                strict=True,
                min_arm_n=thresholds.min_arm_n,
            )
        )

    if evidence.fabricated_preference.n:
        criteria.append(
            _scalar_criterion(
                "fabricated_preference",
                evidence.fabricated_preference.estimate,
                thresholds.max_fabricated_preference,
                comparison="<=",
                n=evidence.fabricated_preference.n,
                detail=(
                    f"{evidence.fabricated_preference.successes}/{evidence.fabricated_preference.n} "
                    f"stratum C pairs drew a stable winner between deterministically equivalent clips"
                ),
            )
        )

    criteria.extend(
        [
            _scalar_criterion(
                "deterministic_kappa",
                evidence.deterministic_kappa,
                thresholds.min_kappa,
                comparison=">=",
                detail=f"kappa {evidence.deterministic_kappa:.3f} against the deterministic layer",
            ),
            _scalar_criterion(
                "stability_variance",
                evidence.stability_variance,
                thresholds.max_variance,
                comparison="<=",
                detail=f"score variance {evidence.stability_variance:.4f} over repeated runs",
            ),
            _scalar_criterion(
                "ablation_response",
                evidence.ablation_response,
                thresholds.min_ablation_delta,
                comparison=">=",
                detail=f"score moved {evidence.ablation_response:.3f} when the evidence was degraded",
            ),
            Criterion(
                name="degenerate_predictors",
                status=(
                    CriterionStatus.PASSED
                    if evidence.degenerate_predictors_all_fail
                    else CriterionStatus.FAILED
                ),
                observed=float(evidence.degenerate_predictors_all_fail),
                threshold=1.0,
                comparison=">=",
                n=0,
                detail="every degenerate predictor must fail this same gate",
            ),
        ]
    )

    if any(entry.status is CriterionStatus.FAILED for entry in criteria):
        status = CriterionStatus.FAILED
    elif any(entry.status is CriterionStatus.UNDERPOWERED for entry in criteria):
        status = CriterionStatus.UNDERPOWERED
    else:
        status = CriterionStatus.PASSED
    return GateReport(grader=evidence.grader, status=status, criteria=tuple(criteria))


def format_rate(result: ProportionResult, baseline: float, *, label: str) -> str:
    """Render a rate the only way the reporting contract (section 7) permits.

    Never print a rate without its n, its baseline and its bound.
    """
    return (
        f"{label} {result.estimate:.2f} (95% LCB {result.lower_bound_95:.2f}) "
        f"against a {baseline:.2f} baseline, n = {result.n}"
    )
