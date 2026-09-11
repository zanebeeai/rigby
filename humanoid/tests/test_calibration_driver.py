"""Calibration arms report what the data supports, or say they cannot.

Plan 10 §5. The arms that matter most here are the ones with no data: a driver
that returns a plausible number for an unmeasurable arm is worse than one that
refuses, because the number survives into a report and the refusal does not.
"""

from __future__ import annotations

import json

import pytest

from evals.calibration.driver import (
    CalibrationDataError,
    baseline_for,
    pool_arms,
    score_arm,
)
from evals.calibration_stats import UnaryJudgment

pytestmark = pytest.mark.fast


def _records(good: int, bad: int, *, accept_good: int, accept_bad: int) -> list[UnaryJudgment]:
    records = [
        UnaryJudgment(clip_id=f"g{index}", is_good=True, accepted=index < accept_good)
        for index in range(good)
    ]
    records += [
        UnaryJudgment(clip_id=f"b{index}", is_good=False, accepted=index < accept_bad)
        for index in range(bad)
    ]
    return records


# ------------------------------------------------------- refusing to score


def test_an_empty_arm_is_unmeasured_not_zero() -> None:
    result = score_arm("sensitivity", [])
    assert result.measured is False
    assert result.scores is None
    assert result.reason == "no records"


@pytest.mark.parametrize(
    ("good", "bad", "only"), [(5, 0, "good"), (0, 5, "bad")]
)
def test_an_arm_with_one_class_refuses_rather_than_reporting_a_rate(
    good: int, bad: int, only: str
) -> None:
    """Sensitivity over no good clips is 0/0.

    A proportion there is not a low score, it is an absent measurement, and
    reporting it as 0.0 reads as a grader that never accepts.
    """
    result = score_arm("x", _records(good, bad, accept_good=good, accept_bad=0))
    assert result.measured is False
    assert f"every clip is {only}" in result.reason
    assert result.n == 5


def test_a_two_class_arm_scores() -> None:
    result = score_arm("x", _records(4, 4, accept_good=4, accept_bad=0))
    assert result.measured is True
    assert result.scores.sensitivity.estimate == 1.0
    assert result.scores.specificity.estimate == 1.0
    assert result.n == 8


def test_every_refusal_carries_a_reason() -> None:
    # A skip with no reason cannot distinguish "no detector exists here" from
    # "the harness declined", and those are different findings.
    for result in (score_arm("a", []), score_arm("b", _records(3, 0, accept_good=3, accept_bad=0))):
        assert result.measured is False
        assert result.reason


# ------------------------------------------------- static-target separation


def test_static_target_results_are_carried_apart_from_the_arm() -> None:
    """676 of 964 applicable pairs land on a bone the case never moves.

    Those separate perfectly for a grader that can see the bone, so pooling them
    into a sweep inflates it. They are a capability measurement, reported beside
    the arm and never inside it.
    """
    result = score_arm("specificity", _records(4, 4, accept_good=4, accept_bad=0), static_target_n=676)
    assert result.static_target_n == 676
    assert result.n == 8
    assert result.to_dict()["static_target_n"] == 676


# -------------------------------------------------------------- baselines


def test_a_baseline_is_none_on_an_empty_set_not_a_number() -> None:
    # So a caller cannot emit a rate with a fabricated baseline beside it.
    assert baseline_for([]) is None


def test_the_baseline_is_the_majority_class() -> None:
    assert baseline_for(_records(9, 1, accept_good=9, accept_bad=0)) == pytest.approx(0.9)


# ------------------------------------------------------- the serialised form


def test_an_unmeasured_arm_serialises_as_unmeasured() -> None:
    payload = score_arm("detection_threshold", []).to_dict()
    assert payload["measured"] is False
    assert payload["scores"] is None
    assert payload["reason"]


def test_a_measured_arm_carries_its_n_beside_its_rates() -> None:
    payload = score_arm("x", _records(4, 4, accept_good=3, accept_bad=1)).to_dict()
    assert payload["n"] == 8
    assert payload["scores"]["sensitivity"]["n"] == 4
    assert payload["scores"]["specificity"]["n"] == 4


# ------------------------------------------------------------- denominators


def _arm(name: str, case_ids: list[str]) -> object:
    records = [
        UnaryJudgment(clip_id=case, is_good=index % 2 == 0, accepted=index % 2 == 0)
        for index, case in enumerate(case_ids)
    ]
    return score_arm(name, records)


def test_an_arm_records_the_cases_it_was_measured_over() -> None:
    assert _arm("a", ["c1", "c2"]).cases == frozenset({"c1", "c2"})


def test_pooling_arms_with_different_case_sets_is_refused() -> None:
    """Not every axis reaches every case.

    06b emits the three `contract.clip.*` checks on all 47 corpus cases but the
    anatomy and signal axes on 14, so a rate quoted per case has denominator 14
    or 47 depending on the axis. Averaging them produces a number whose overlap
    subset is unstated — §2.5's warning arriving through the denominator instead
    of the population.
    """
    wide = _arm("contract", ["c1", "c2", "c3", "c4"])
    narrow = _arm("anatomy", ["c1", "c2"])
    with pytest.raises(CalibrationDataError, match="different case sets"):
        pool_arms([wide, narrow])


def test_the_refusal_names_both_denominators() -> None:
    # So a reader can see which arm reaches fewer cases rather than being told
    # only that pooling failed.
    with pytest.raises(CalibrationDataError, match="anatomy over 2.*contract over 4"):
        pool_arms([_arm("contract", ["c1", "c2", "c3", "c4"]), _arm("anatomy", ["c1", "c2"])])


def test_pooling_agreeing_arms_returns_the_shared_case_set() -> None:
    # A caller that pools has to have obtained the shared set, rather than
    # pooling and hoping.
    shared = pool_arms([_arm("a", ["c1", "c2"]), _arm("b", ["c1", "c2"])])
    assert shared == frozenset({"c1", "c2"})


def test_pooling_refuses_a_union_and_an_intersection_alike() -> None:
    # Either choice would be a silent decision about which arm's denominator
    # wins, so neither is offered.
    with pytest.raises(CalibrationDataError):
        pool_arms([_arm("a", ["c1"]), _arm("b", ["c2"])])


def test_pooling_with_no_measured_arm_raises() -> None:
    with pytest.raises(CalibrationDataError, match="no measured arm"):
        pool_arms([score_arm("empty", [])])


# --- 10d: the detection arm wired to 06b's sweeps -----------------------------


def _sweep_result(name: str, threshold, *, measured: bool, reason: str = "", baseline=None):
    from evals.calibration.driver import SweepResult
    from evals.calibration_stats import proportion

    return SweepResult(
        name=name,
        threshold=threshold,
        curve=[],
        capability=proportion(0, 0),
        skips={},
        measured=measured,
        baseline=baseline if baseline is not None else proportion(0, 12),
        reason=reason,
    )


def test_a_sweep_that_measured_nothing_is_not_a_detector_that_failed() -> None:
    # Opposite findings that would otherwise share a cell: "every pair was
    # refused" blames the corpus, "never reached 50%" blames the grader.
    from evals.calibration.driver import detection_report

    never = _sweep_result("jitter", None, measured=True, reason="never reached 50%")
    nothing = _sweep_result("rom", None, measured=False, reason="no threshold points")
    report = detection_report([never, nothing])
    assert report["measured"] == []
    assert report["unmeasured"]["jitter"] != report["unmeasured"]["rom"]
    assert never.measured is True and nothing.measured is False


def test_the_detection_report_refuses_to_pool_sweeps() -> None:
    # Two sweeps, two thresholds, and no combined number anywhere in the payload.
    from evals.calibration.driver import detection_report
    from evals.calibration_stats import Interval

    results = [
        _sweep_result("a", Interval(0.16, 0.08, 0.28, 20), measured=True),
        _sweep_result("b", Interval(0.68, 0.48, 1.00, 5), measured=True),
    ]
    report = detection_report(results)
    assert sorted(report["measured"]) == ["a", "b"]
    assert len(report["sweeps"]) == 2

    # Both thresholds survive intact and separately keyed by sweep. Asserting only
    # that no top-level "threshold" key exists would pass against a report that
    # averaged them under any other name, and asserting "no float at top level"
    # passes against every possible payload here -- a check that cannot fail is
    # the second failure direction in docs/testing.md, wearing the language of
    # the thing it audits.
    estimates = {item["sweep"]: item["threshold"]["estimate"] for item in report["sweeps"]}
    assert estimates == {"a": 0.16, "b": 0.68}

    # The number pooling would produce, named and excluded.
    pooled = (0.16 + 0.68) / 2
    flat = json.dumps(report)
    assert str(pooled) not in flat
    assert "0.42" not in flat


def test_an_empty_detection_report_raises_rather_than_reporting_nothing() -> None:
    from evals.calibration.driver import detection_report

    with pytest.raises(CalibrationDataError, match="no sweeps"):
        detection_report([])


def test_every_reported_threshold_carries_its_baseline() -> None:
    # `ProportionResult` structurally carries an n and a bound; nothing
    # structurally carries a baseline. A detection rate without one cannot be
    # told apart from a check that was already firing before the mutation --
    # found by lane `infra` in the published-rate guard's blind spot.
    from evals.calibration.driver import detection_report
    from evals.calibration_stats import Interval, proportion

    result = _sweep_result(
        "rom", Interval(0.16, 0.08, 0.28, 20), measured=True, baseline=proportion(0, 20)
    )
    report = detection_report([result])
    entry = report["sweeps"][0]
    assert entry["baseline_detection_rate"] is not None
    assert entry["baseline_detection_rate"]["n"] == 20
    assert entry["baseline_detection_rate"]["estimate"] == 0.0


def test_a_baseline_that_already_fires_is_reported_not_hidden() -> None:
    # The case the baseline exists for: the target check fires on most unmutated
    # clips, so a high detection rate says nothing about the mutation. The report
    # must carry the number rather than quietly publishing the curve alone.
    from evals.calibration.driver import detection_report
    from evals.calibration_stats import Interval, proportion

    result = _sweep_result(
        "jitter", Interval(0.04, 0.04, 0.08, 20), measured=True, baseline=proportion(18, 20)
    )
    entry = detection_report([result])["sweeps"][0]
    assert entry["baseline_detection_rate"]["estimate"] == 0.9
    # And it is not silently folded into the threshold.
    assert entry["threshold"]["estimate"] == 0.04
