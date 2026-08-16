"""The permanent guard against the bug class that invalidated the old calibration.

Plan 10, section 5.5. Four degenerate predictors are run through the *real* gate on
synthetic judgment records, and every one must fail. Zero model calls.

The old suite scored "always pick the base clip" at 0.9 against a 0.8 gate, because
every scored item shared one ground-truth label. This test is the reason that cannot
recur: it is not a test of the graders, it is a test of the scoring function, and it
runs before any grader exists.

Each degenerate predictor is handed *perfect* results on every criterion its strategy
does not determine -- a flawless detection threshold, kappa, stability and ablation
response, and for the pairwise predictors a flawless accept/reject arm. That makes the
test as hard as possible on the gate: the only thing that can catch these predictors is
the clause section 5.5 names, so if the gate rejects them, it rejects them for the right
reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import pytest

from evals.calibration_stats import (
    CriterionStatus,
    GateReport,
    GateThresholds,
    Interval,
    PairwiseJudgment,
    Stratum,
    UnaryJudgment,
    build_evidence,
    clopper_pearson_lower,
    evaluate_gate,
)

THRESHOLDS = GateThresholds(
    max_acceptable_severity=0.35,
    min_sensitivity=0.85,
    min_specificity=0.85,
    min_kappa=0.60,
    max_variance=0.05,
    min_ablation_delta=0.15,
    max_fabricated_preference=0.30,
    min_arm_n=19,
)

ARM_N = 32
STRATUM_C_PAIRS = 20

Role = Literal["base", "mild", "severe", "sibling"]


# --------------------------------------------------------------------------- #
# A synthetic suite with the stratification of plan section 5.4
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SuiteItem:
    """One presented pair, with the roles a predictor is allowed to cheat on.

    Roles live here rather than in `calibration_stats` on purpose: the scorer must not
    know which member is the base clip, or it could not detect a predictor that does.
    """

    pair_id: str
    stratum: Stratum
    first: tuple[str, Role]
    second: tuple[str, Role]
    correct: Literal["first", "second"] | None


def _stratified_pairs() -> list[SuiteItem]:
    """Strata A, B and C with the winning member alternating between positions.

    Alternating positions is what pins each stratum's majority-class baseline to
    exactly 0.50, so a position-constant predictor scores at chance by construction.
    """
    items: list[SuiteItem] = []
    for index in range(ARM_N):
        winner_first = index % 2 == 0
        good: tuple[str, Role] = (f"a{index}-base", "base")
        bad: tuple[str, Role] = (f"a{index}-severe", "severe")
        items.append(
            SuiteItem(
                pair_id=f"A{index}",
                stratum="A",
                first=good if winner_first else bad,
                second=bad if winner_first else good,
                correct="first" if winner_first else "second",
            )
        )
    for index in range(ARM_N):
        winner_first = index % 2 == 0
        good = (f"b{index}-mild", "mild")
        bad = (f"b{index}-severe", "severe")
        items.append(
            SuiteItem(
                pair_id=f"B{index}",
                stratum="B",
                first=good if winner_first else bad,
                second=bad if winner_first else good,
                correct="first" if winner_first else "second",
            )
        )
    for index in range(STRATUM_C_PAIRS):
        left: tuple[str, Role] = (f"c{index}-base", "base")
        right: tuple[str, Role] = (f"c{index}-sibling", "sibling")
        items.append(SuiteItem(pair_id=f"C{index}", stratum="C", first=left, second=right, correct=None))
        items.append(SuiteItem(pair_id=f"C{index}", stratum="C", first=right, second=left, correct=None))
    return items


def _unary_clips() -> list[tuple[str, bool]]:
    """`ARM_N` unmutated clips and `ARM_N` mutated ones."""
    return [(f"good{index}", True) for index in range(ARM_N)] + [
        (f"bad{index}", False) for index in range(ARM_N)
    ]


PairwisePredictor = Callable[[SuiteItem], Literal["first", "second", "tie"]]
UnaryPredictor = Callable[[str, bool], bool]


def _judge(
    pairwise: PairwisePredictor,
    unary: UnaryPredictor,
) -> tuple[list[UnaryJudgment], list[PairwiseJudgment]]:
    unary_records = [
        UnaryJudgment(clip_id=clip_id, is_good=is_good, accepted=unary(clip_id, is_good))
        for clip_id, is_good in _unary_clips()
    ]
    pairwise_records = [
        PairwiseJudgment(
            pair_id=item.pair_id,
            stratum=item.stratum,
            first_id=item.first[0],
            second_id=item.second[0],
            correct=item.correct,
            predicted=pairwise(item),
        )
        for item in _stratified_pairs()
    ]
    return unary_records, pairwise_records


def _run_gate(
    name: str,
    pairwise: PairwisePredictor,
    unary: UnaryPredictor,
) -> GateReport:
    """Score a predictor with every non-determined criterion set to a perfect value."""
    unary_records, pairwise_records = _judge(pairwise, unary)
    evidence = build_evidence(
        name,
        unary=unary_records,
        pairwise=pairwise_records,
        detection_threshold=Interval(estimate=0.10, lower_bound_95=0.06, upper_bound_95=0.15, n=140),
        deterministic_kappa=0.90,
        stability_variance=0.001,
        ablation_response=0.60,
        degenerate_predictors_all_fail=True,
    )
    return evaluate_gate(evidence, THRESHOLDS)


# --------------------------------------------------------------------------- #
# The four degenerate predictors
# --------------------------------------------------------------------------- #


def _honest_unary(_clip_id: str, is_good: bool) -> bool:
    """A perfect accept/reject arm, so only the pairwise strategy can fail the gate."""
    return is_good


def _pick_base(item: SuiteItem) -> Literal["first", "second", "tie"]:
    """Always name the base clip; fall back to position one where no base exists.

    Stratum B is exactly that fallback case, and it is the whole reason stratum B is in
    the suite.
    """
    if item.first[1] == "base":
        return "first"
    if item.second[1] == "base":
        return "second"
    return "first"


def _pick_first(_item: SuiteItem) -> Literal["first", "second", "tie"]:
    return "first"


def _abstain(_item: SuiteItem) -> Literal["first", "second", "tie"]:
    """A grader that accepts (or rejects) everything expresses no preference either."""
    return "tie"


def test_always_accept_fails_on_specificity() -> None:
    report = _run_gate("always-accept", _abstain, lambda _clip_id, _is_good: True)
    specificity = report.criterion("specificity")
    assert specificity.status is CriterionStatus.FAILED
    assert specificity.observed == 0.0
    assert specificity.n == ARM_N
    assert report.criterion("sensitivity").status is CriterionStatus.PASSED
    assert not report.passed
    assert report.status is CriterionStatus.FAILED


def test_always_reject_fails_on_sensitivity() -> None:
    report = _run_gate("always-reject", _abstain, lambda _clip_id, _is_good: False)
    sensitivity = report.criterion("sensitivity")
    assert sensitivity.status is CriterionStatus.FAILED
    assert sensitivity.observed == 0.0
    assert sensitivity.n == ARM_N
    assert report.criterion("specificity").status is CriterionStatus.PASSED
    assert not report.passed
    assert report.status is CriterionStatus.FAILED


def test_always_base_fails_in_stratum_b_where_the_base_clip_is_absent() -> None:
    report = _run_gate("always-base", _pick_base, _honest_unary)
    stratum_a = report.criterion("discrimination.A")
    stratum_b = report.criterion("discrimination.B")
    assert stratum_a.status is CriterionStatus.PASSED, "stratum A alone cannot catch this predictor"
    assert stratum_b.status is CriterionStatus.FAILED
    assert stratum_b.threshold == 0.5
    fabricated = report.criterion("fabricated_preference")
    assert fabricated.status is CriterionStatus.FAILED
    assert (fabricated.observed, fabricated.n) == (1.0, STRATUM_C_PAIRS)
    assert not report.passed
    assert report.status is CriterionStatus.FAILED


def test_always_first_fails_on_directional_accuracy_in_both_strata() -> None:
    report = _run_gate("always-first", _pick_first, _honest_unary)
    for name in ("discrimination.A", "discrimination.B"):
        criterion = report.criterion(name)
        assert criterion.status is CriterionStatus.FAILED, name
        assert criterion.threshold == 0.5
    # Position balance is what does this: the predictor is right exactly half the time.
    assert report.criterion("discrimination.A").detail.startswith(f"{ARM_N // 2}/{ARM_N}")
    assert not report.passed
    assert report.status is CriterionStatus.FAILED


def test_every_degenerate_predictor_fails_the_same_gate() -> None:
    """The claim in one assertion, so it cannot be weakened one test at a time."""
    predictors: list[tuple[str, PairwisePredictor, UnaryPredictor]] = [
        ("always-accept", _abstain, lambda _clip_id, _is_good: True),
        ("always-reject", _abstain, lambda _clip_id, _is_good: False),
        ("always-base", _pick_base, _honest_unary),
        ("always-first", _pick_first, _honest_unary),
    ]
    verdicts = {name: _run_gate(name, pairwise, unary) for name, pairwise, unary in predictors}
    assert not any(report.passed for report in verdicts.values())
    assert all(report.status is CriterionStatus.FAILED for report in verdicts.values())


def test_pooling_the_strata_would_let_always_base_through() -> None:
    """Why the gate scores strata separately, asserted rather than asserted-in-prose.

    Over a balanced A + B pool "always pick the base clip" is right 3 times in 4 and its
    lower bound clears the 0.50 pooled baseline comfortably. Pooling is not a weaker
    version of this gate; it is the old bug in a new suite.
    """
    _unary_records, pairwise_records = _judge(_pick_base, _honest_unary)
    directional = [record for record in pairwise_records if record.stratum in {"A", "B"}]
    hits = sum(1 for record in directional if record.predicted == record.correct)
    pooled_lower_bound = clopper_pearson_lower(hits, len(directional))
    assert hits / len(directional) == pytest.approx(0.75)
    assert pooled_lower_bound > 0.5

    report = _run_gate("always-base", _pick_base, _honest_unary)
    assert report.criterion("discrimination.B").status is CriterionStatus.FAILED


# --------------------------------------------------------------------------- #
# The other side of the claim: an honest grader at the boundary passes
# --------------------------------------------------------------------------- #


def _minimum_successes(n: int, threshold: float, *, strict: bool) -> int:
    """Fewest successes out of n whose 95% lower bound clears `threshold`."""
    for successes in range(n + 1):
        bound = clopper_pearson_lower(successes, n)
        if bound > threshold or (not strict and bound == threshold):
            return successes
    raise AssertionError(f"n = {n} cannot clear {threshold} at any score")


BOUNDARY_UNARY = _minimum_successes(ARM_N, THRESHOLDS.min_sensitivity, strict=False)
BOUNDARY_DIRECTIONAL = _minimum_successes(ARM_N, 0.5, strict=True)


def _boundary_records(
    *,
    sensitivity_hits: int,
    specificity_hits: int,
    stratum_a_hits: int,
    stratum_b_hits: int,
) -> tuple[list[UnaryJudgment], list[PairwiseJudgment]]:
    """A suite scoring exactly the requested number of successes in each arm."""
    good = [f"good{index}" for index in range(ARM_N)]
    bad = [f"bad{index}" for index in range(ARM_N)]
    unary_records = [
        UnaryJudgment(clip_id=clip_id, is_good=True, accepted=index < sensitivity_hits)
        for index, clip_id in enumerate(good)
    ] + [
        UnaryJudgment(clip_id=clip_id, is_good=False, accepted=index >= specificity_hits)
        for index, clip_id in enumerate(bad)
    ]

    hits_by_stratum = {"A": stratum_a_hits, "B": stratum_b_hits}
    seen: dict[str, int] = {"A": 0, "B": 0}
    pairwise_records: list[PairwiseJudgment] = []
    for item in _stratified_pairs():
        if item.stratum == "C":
            # Alternate the named clip across orders: a grader with no fabricated
            # preference names position one both times, which is a different clip.
            predicted: Literal["first", "second", "tie"] = "first"
        else:
            index = seen[item.stratum]
            seen[item.stratum] += 1
            correct = item.correct
            assert correct is not None
            wrong: Literal["first", "second"] = "second" if correct == "first" else "first"
            predicted = correct if index < hits_by_stratum[item.stratum] else wrong
        pairwise_records.append(
            PairwiseJudgment(
                pair_id=item.pair_id,
                stratum=item.stratum,
                first_id=item.first[0],
                second_id=item.second[0],
                correct=item.correct,
                predicted=predicted,
            )
        )
    return unary_records, pairwise_records


def _boundary_report(
    *,
    sensitivity_hits: int = BOUNDARY_UNARY,
    specificity_hits: int = BOUNDARY_UNARY,
    stratum_a_hits: int = BOUNDARY_DIRECTIONAL,
    stratum_b_hits: int = BOUNDARY_DIRECTIONAL,
) -> GateReport:
    unary_records, pairwise_records = _boundary_records(
        sensitivity_hits=sensitivity_hits,
        specificity_hits=specificity_hits,
        stratum_a_hits=stratum_a_hits,
        stratum_b_hits=stratum_b_hits,
    )
    evidence = build_evidence(
        "boundary",
        unary=unary_records,
        pairwise=pairwise_records,
        detection_threshold=Interval(
            estimate=0.20,
            lower_bound_95=0.14,
            upper_bound_95=THRESHOLDS.max_acceptable_severity,
            n=140,
        ),
        deterministic_kappa=THRESHOLDS.min_kappa,
        stability_variance=THRESHOLDS.max_variance,
        ablation_response=THRESHOLDS.min_ablation_delta,
        degenerate_predictors_all_fail=True,
    )
    return evaluate_gate(evidence, THRESHOLDS)


def test_a_balanced_suite_at_boundary_scores_passes() -> None:
    report = _boundary_report()
    assert report.passed, report.summary()
    assert all(criterion.status is CriterionStatus.PASSED for criterion in report.criteria)
    assert {criterion.name for criterion in report.criteria} == {
        "detection_threshold",
        "sensitivity",
        "specificity",
        "discrimination.A",
        "discrimination.B",
        "fabricated_preference",
        "deterministic_kappa",
        "stability_variance",
        "ablation_response",
        "degenerate_predictors",
    }


@pytest.mark.parametrize(
    ("arm", "boundary"),
    [
        ("sensitivity_hits", BOUNDARY_UNARY),
        ("specificity_hits", BOUNDARY_UNARY),
        ("stratum_a_hits", BOUNDARY_DIRECTIONAL),
        ("stratum_b_hits", BOUNDARY_DIRECTIONAL),
    ],
)
def test_one_fewer_success_in_any_arm_fails(arm: str, boundary: int) -> None:
    assert not _boundary_report(**{arm: boundary - 1}).passed


def test_the_degenerate_predictor_flag_is_itself_gated() -> None:
    """A calibration run that never ran the degenerate suite does not pass."""
    unary_records, pairwise_records = _boundary_records(
        sensitivity_hits=BOUNDARY_UNARY,
        specificity_hits=BOUNDARY_UNARY,
        stratum_a_hits=BOUNDARY_DIRECTIONAL,
        stratum_b_hits=BOUNDARY_DIRECTIONAL,
    )
    evidence = build_evidence(
        "no-degenerate-run",
        unary=unary_records,
        pairwise=pairwise_records,
        detection_threshold=Interval(estimate=0.20, lower_bound_95=0.14, upper_bound_95=0.30, n=140),
        deterministic_kappa=THRESHOLDS.min_kappa,
        stability_variance=THRESHOLDS.max_variance,
        ablation_response=THRESHOLDS.min_ablation_delta,
        degenerate_predictors_all_fail=False,
    )
    report = evaluate_gate(evidence, THRESHOLDS)
    assert report.criterion("degenerate_predictors").status is CriterionStatus.FAILED
    assert not report.passed


# --------------------------------------------------------------------------- #
# Sample size is self-enforcing, and underpowered is not the same as failed
# --------------------------------------------------------------------------- #


def test_a_small_suite_fails_regardless_of_score() -> None:
    """A perfect ten-item suite cannot pass. That is the mechanism, not a side effect."""
    small = 10
    unary_records = [
        UnaryJudgment(clip_id=f"good{index}", is_good=True, accepted=True) for index in range(small)
    ] + [UnaryJudgment(clip_id=f"bad{index}", is_good=False, accepted=False) for index in range(small)]
    pairwise_records = [
        PairwiseJudgment(
            pair_id=f"{stratum}{index}",
            stratum=stratum,
            first_id=f"{stratum}{index}-good",
            second_id=f"{stratum}{index}-bad",
            correct="first",
            predicted="first",
        )
        for stratum in ("A", "B")
        for index in range(small)
    ]
    evidence = build_evidence(
        "tiny-but-perfect",
        unary=unary_records,
        pairwise=pairwise_records,
        detection_threshold=Interval(estimate=0.05, lower_bound_95=0.02, upper_bound_95=0.09, n=20),
        deterministic_kappa=1.0,
        stability_variance=0.0,
        ablation_response=1.0,
        degenerate_predictors_all_fail=True,
    )
    report = evaluate_gate(evidence, THRESHOLDS)
    assert not report.passed
    assert report.status is CriterionStatus.UNDERPOWERED
    assert {criterion.name for criterion in report.underpowered_criteria()} == {
        "sensitivity",
        "specificity",
        "discrimination.A",
        "discrimination.B",
    }
    assert all(criterion.required_n == THRESHOLDS.min_arm_n for criterion in report.underpowered_criteria())


def test_underpowered_and_failed_are_distinguishable_in_the_report() -> None:
    """The distinction the whole triage exists for.

    A grader scoring 30/32 has a rate well above the 0.85 line but an interval too wide
    to prove it; a grader scoring 20/32 is under the line and no sample size saves it.
    Both are "not passing"; only one is fixed by collecting more data.
    """
    underpowered = _boundary_report(sensitivity_hits=BOUNDARY_UNARY - 1).criterion("sensitivity")
    assert underpowered.status is CriterionStatus.UNDERPOWERED
    assert underpowered.required_n is not None and underpowered.required_n > ARM_N
    assert "too small" in underpowered.detail

    failed = _boundary_report(sensitivity_hits=20).criterion("sensitivity")
    assert failed.status is CriterionStatus.FAILED
    assert failed.required_n is None
    assert "no sample size" in failed.detail


def test_the_report_summary_names_the_remedy() -> None:
    passing = _boundary_report()
    assert passing.summary() == "boundary: passed 10 criteria"

    genuine = _boundary_report(sensitivity_hits=20)
    assert genuine.summary().startswith("boundary: FAILED on sensitivity")
    assert "not fit for use" in genuine.summary()

    thin = _boundary_report(sensitivity_hits=BOUNDARY_UNARY - 1)
    assert thin.summary().startswith("boundary: UNDERPOWERED on sensitivity")
    assert "needs n >= " in thin.summary()


def test_a_failure_anywhere_outranks_an_underpowered_arm() -> None:
    """Enlarging the suite is not the remedy for a grader that is measurably broken."""
    report = _boundary_report(sensitivity_hits=BOUNDARY_UNARY - 1, specificity_hits=10)
    assert report.status is CriterionStatus.FAILED
    assert report.criterion("sensitivity").status is CriterionStatus.UNDERPOWERED
    assert report.criterion("specificity").status is CriterionStatus.FAILED


def test_a_grader_that_never_detects_fails_rather_than_being_skipped() -> None:
    unary_records, pairwise_records = _boundary_records(
        sensitivity_hits=BOUNDARY_UNARY,
        specificity_hits=BOUNDARY_UNARY,
        stratum_a_hits=BOUNDARY_DIRECTIONAL,
        stratum_b_hits=BOUNDARY_DIRECTIONAL,
    )
    evidence = build_evidence(
        "never-detects",
        unary=unary_records,
        pairwise=pairwise_records,
        detection_threshold=None,
        deterministic_kappa=THRESHOLDS.min_kappa,
        stability_variance=THRESHOLDS.max_variance,
        ablation_response=THRESHOLDS.min_ablation_delta,
        degenerate_predictors_all_fail=True,
    )
    report = evaluate_gate(evidence, THRESHOLDS)
    assert report.criterion("detection_threshold").status is CriterionStatus.FAILED
    assert not report.passed


def test_a_suite_missing_stratum_b_is_underpowered_not_passing() -> None:
    """Dropping stratum B is how the old suite became scorable by a constant predictor."""
    unary_records, pairwise_records = _boundary_records(
        sensitivity_hits=BOUNDARY_UNARY,
        specificity_hits=BOUNDARY_UNARY,
        stratum_a_hits=BOUNDARY_DIRECTIONAL,
        stratum_b_hits=BOUNDARY_DIRECTIONAL,
    )
    without_b = [record for record in pairwise_records if record.stratum != "B"]
    evidence = build_evidence(
        "no-stratum-b",
        unary=unary_records,
        pairwise=without_b,
        detection_threshold=Interval(estimate=0.20, lower_bound_95=0.14, upper_bound_95=0.30, n=140),
        deterministic_kappa=THRESHOLDS.min_kappa,
        stability_variance=THRESHOLDS.max_variance,
        ablation_response=THRESHOLDS.min_ablation_delta,
        degenerate_predictors_all_fail=True,
    )
    report = evaluate_gate(evidence, THRESHOLDS)
    assert report.criterion("discrimination.B").status is CriterionStatus.UNDERPOWERED
    assert not report.passed


def test_thresholds_are_a_frozen_record() -> None:
    """The gate reads its numbers from one place, so 08 can own them later."""
    stricter = replace(THRESHOLDS, min_sensitivity=0.95)
    unary_records, pairwise_records = _boundary_records(
        sensitivity_hits=ARM_N,
        specificity_hits=ARM_N,
        stratum_a_hits=ARM_N,
        stratum_b_hits=ARM_N,
    )
    evidence = build_evidence(
        "perfect",
        unary=unary_records,
        pairwise=pairwise_records,
        detection_threshold=Interval(estimate=0.10, lower_bound_95=0.06, upper_bound_95=0.15, n=140),
        deterministic_kappa=1.0,
        stability_variance=0.0,
        ablation_response=1.0,
        degenerate_predictors_all_fail=True,
    )
    # 32 perfect items certify 0.911, which clears 0.85 but not 0.95.
    assert evaluate_gate(evidence, THRESHOLDS).passed
    assert evaluate_gate(evidence, stricter).criterion("sensitivity").status is CriterionStatus.UNDERPOWERED


def test_stratum_records_must_carry_a_correct_answer() -> None:
    with pytest.raises(ValueError):
        build_evidence(
            "malformed",
            unary=[UnaryJudgment(clip_id="good0", is_good=True, accepted=True)],
            pairwise=[
                PairwiseJudgment(
                    pair_id="A0",
                    stratum="A",
                    first_id="x",
                    second_id="y",
                    correct=None,
                    predicted="first",
                )
            ],
            detection_threshold=None,
            deterministic_kappa=0.0,
            stability_variance=0.0,
            ablation_response=0.0,
            degenerate_predictors_all_fail=True,
        )
