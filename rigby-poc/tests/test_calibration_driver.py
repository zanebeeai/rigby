"""Calibration arms report what the data supports, or say they cannot.

Plan 10 §5. The arms that matter most here are the ones with no data: a driver
that returns a plausible number for an unmeasurable arm is worse than one that
refuses, because the number survives into a report and the refusal does not.
"""

from __future__ import annotations

import pytest

from evals.calibration.driver import (
    ArmResult,
    baseline_for,
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
