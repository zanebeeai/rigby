"""Unit tests for the calibration statistics themselves.

The Clopper-Pearson bound is checked two ways: against hand-computed reference values,
and against its own definition -- `P(X >= k | p = L) == alpha` under the binomial,
which is independent of the beta-quantile implementation used to produce it.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import binom

from evals.calibration_stats import (
    CriterionStatus,
    ProportionResult,
    balanced_accuracy,
    clopper_pearson_lower,
    cohens_kappa,
    format_rate,
    majority_class_baseline,
    matthews_corrcoef,
    minimum_n_for_lower_bound,
    proportion,
)

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Clopper-Pearson
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [
        (8, 10, 0.493099),
        (19, 20, 0.783894),
        (27, 30, 0.761402),
        (45, 50, 0.801167),
        (99, 100, 0.953440),
        (1, 20, 0.002561),
    ],
)
def test_clopper_pearson_lower_matches_reference_values(successes: int, n: int, expected: float) -> None:
    assert clopper_pearson_lower(successes, n) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(("successes", "n"), [(8, 10), (27, 30), (45, 50), (1, 20), (60, 60), (99, 100)])
def test_clopper_pearson_lower_satisfies_its_own_definition(successes: int, n: int) -> None:
    """The bound is the p at which observing this many successes or more has probability alpha."""
    bound = clopper_pearson_lower(successes, n)
    assert binom.sf(successes - 1, n, bound) == pytest.approx(0.05, abs=1e-9)


@pytest.mark.parametrize("n", [1, 10, 100])
def test_clopper_pearson_lower_is_zero_with_no_successes(n: int) -> None:
    """k = 0 supports no claim at all: the bound is exactly zero, not a small number."""
    assert clopper_pearson_lower(0, n) == 0.0


@pytest.mark.parametrize("n", [10, 20, 28, 29, 32, 50, 59])
def test_clopper_pearson_lower_at_perfection_is_alpha_root_n(n: int) -> None:
    assert clopper_pearson_lower(n, n) == pytest.approx(0.05 ** (1 / n), rel=1e-12)


def test_clopper_pearson_lower_is_zero_with_no_data() -> None:
    assert clopper_pearson_lower(0, 0) == 0.0


def test_thirty_two_perfect_negatives_certify_zero_nine_and_twenty_eight_do_not() -> None:
    """The sample-size claim the gate rests on (plan section 5.3).

    A perfect score is not enough on its own: certifying 0.90 takes 29 items, and no
    score at n = 28 can get there.
    """
    assert clopper_pearson_lower(32, 32) == pytest.approx(0.9106, abs=5e-5)
    assert clopper_pearson_lower(28, 28) == pytest.approx(0.8985, abs=5e-5)
    assert clopper_pearson_lower(28, 28) < 0.90 <= clopper_pearson_lower(29, 29)


def test_clopper_pearson_lower_increases_with_sample_size_at_a_fixed_rate() -> None:
    bounds = [clopper_pearson_lower(round(0.9 * n), n) for n in (20, 40, 80, 160, 320)]
    assert bounds == sorted(bounds)
    assert bounds[-1] < 0.9


def test_clopper_pearson_lower_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError):
        clopper_pearson_lower(11, 10)
    with pytest.raises(ValueError):
        clopper_pearson_lower(-1, 10)
    with pytest.raises(ValueError):
        clopper_pearson_lower(5, 10, confidence=1.0)


# --------------------------------------------------------------------------- #
# ProportionResult and required-n
# --------------------------------------------------------------------------- #


def test_proportion_carries_estimate_bound_n_and_successes() -> None:
    result = proportion(31, 32)
    assert (result.successes, result.n) == (31, 32)
    assert result.estimate == pytest.approx(31 / 32)
    assert result.lower_bound_95 == pytest.approx(0.860151, abs=1e-6)
    assert result.to_dict()["lower_bound_95"] == result.lower_bound_95


def test_proportion_with_no_data_is_a_placeholder_not_a_measurement() -> None:
    result = proportion(0, 0)
    assert (result.estimate, result.lower_bound_95, result.n) == (0.0, 0.0, 0)


def test_proportion_result_rejects_more_successes_than_trials() -> None:
    with pytest.raises(ValueError):
        ProportionResult(successes=5, n=4, estimate=1.25, lower_bound_95=0.0)


@pytest.mark.parametrize(("threshold", "expected"), [(0.70, 9), (0.80, 14), (0.85, 19), (0.90, 29), (0.95, 59)])
def test_minimum_n_at_perfection_matches_the_closed_form(threshold: float, expected: int) -> None:
    assert minimum_n_for_lower_bound(threshold) == expected
    assert expected == math.ceil(math.log(0.05) / math.log(threshold))


def test_minimum_n_grows_as_the_observed_rate_approaches_the_threshold() -> None:
    assert minimum_n_for_lower_bound(0.85, 1.00) == 19
    assert minimum_n_for_lower_bound(0.85, 0.95) == 40
    assert minimum_n_for_lower_bound(0.85, 0.90) == 134


def test_minimum_n_is_unreachable_when_the_rate_does_not_beat_the_threshold() -> None:
    """The bound converges to the rate from below, so a rate at the line never clears it."""
    assert minimum_n_for_lower_bound(0.90, 0.90) is None
    assert minimum_n_for_lower_bound(0.90, 0.80) is None


def test_minimum_n_beating_a_half_baseline_strictly() -> None:
    assert minimum_n_for_lower_bound(0.5, 1.0, strict=True) == 5
    assert minimum_n_for_lower_bound(0.5, 0.75, strict=True) == 13


# --------------------------------------------------------------------------- #
# Baselines and agreement metrics
# --------------------------------------------------------------------------- #


def test_majority_class_baseline_is_computed_not_assumed() -> None:
    assert majority_class_baseline(["first", "second"] * 8) == 0.5
    assert majority_class_baseline(["base"] * 9 + ["corruption"]) == 0.9
    assert majority_class_baseline(["base"] * 10) == 1.0


def test_majority_class_baseline_rejects_an_empty_suite() -> None:
    with pytest.raises(ValueError):
        majority_class_baseline([])


def test_balanced_accuracy_puts_constant_predictors_at_chance_however_skewed() -> None:
    truth = [True] * 90 + [False] * 10
    assert balanced_accuracy(truth, [True] * 100) == 0.5
    assert balanced_accuracy(truth, [False] * 100) == 0.5
    assert balanced_accuracy(truth, truth) == 1.0


def test_balanced_accuracy_differs_from_plain_accuracy_under_imbalance() -> None:
    truth = [True] * 90 + [False] * 10
    predicted = [True] * 95 + [False] * 5
    plain = sum(1 for a, b in zip(truth, predicted, strict=True) if a == b) / len(truth)
    assert plain == 0.95
    assert balanced_accuracy(truth, predicted) == pytest.approx(0.75)


def test_cohens_kappa_reference_values() -> None:
    """Textbook 2x2: a=20, b=5, c=10, d=15 over 50 items -> kappa = 0.4."""
    rater_a = ["yes"] * 25 + ["no"] * 25
    rater_b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohens_kappa(rater_a, rater_b) == pytest.approx(0.4, abs=1e-12)


def test_cohens_kappa_is_zero_for_chance_agreement_and_one_for_identity() -> None:
    labels = ["a", "b"] * 10
    assert cohens_kappa(labels, labels) == 1.0
    assert cohens_kappa(labels, ["a", "b", "b", "a"] * 5) == pytest.approx(0.0)


def test_cohens_kappa_gives_no_credit_to_two_constant_raters() -> None:
    """Perfect agreement between two raters who never vary is not agreement."""
    assert cohens_kappa(["yes"] * 20, ["yes"] * 20) == 0.0


def test_matthews_corrcoef_reference_and_degenerate_values() -> None:
    truth = [True] * 4 + [False] * 4
    assert matthews_corrcoef(truth, truth) == pytest.approx(1.0)
    assert matthews_corrcoef(truth, [not value for value in truth]) == pytest.approx(-1.0)
    assert matthews_corrcoef(truth, [True] * 8) == 0.0
    assert matthews_corrcoef(truth, [False] * 8) == 0.0


def test_matthews_corrcoef_matches_the_closed_form_on_an_asymmetric_case() -> None:
    truth = [True] * 6 + [False] * 4
    predicted = [True] * 5 + [False] + [True] + [False] * 3
    # TP=5, FN=1, FP=1, TN=3
    expected = (5 * 3 - 1 * 1) / math.sqrt(6 * 6 * 4 * 4)
    assert matthews_corrcoef(truth, predicted) == pytest.approx(expected)


def test_paired_metrics_reject_mismatched_or_empty_inputs() -> None:
    with pytest.raises(ValueError):
        balanced_accuracy([True, False], [True])
    with pytest.raises(ValueError):
        cohens_kappa([], [])
    with pytest.raises(ValueError):
        matthews_corrcoef([True], [])


# --------------------------------------------------------------------------- #
# Reporting contract
# --------------------------------------------------------------------------- #


def test_format_rate_always_prints_n_baseline_and_bound() -> None:
    rendered = format_rate(proportion(49, 60), 0.5, label="Semantic discrimination")
    assert rendered == "Semantic discrimination 0.82 (95% LCB 0.71) against a 0.50 baseline, n = 60"


def test_criterion_status_values_are_stable_strings() -> None:
    """The status is published in eval-report.v2.json, so its spelling is a contract."""
    assert [str(status) for status in CriterionStatus] == ["passed", "failed", "underpowered"]
