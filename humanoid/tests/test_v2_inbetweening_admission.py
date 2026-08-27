from __future__ import annotations

from dataclasses import replace

import pytest

from rigby_v2.calibration import (
    HumanOutcome,
    InbetweeningTrial,
    evaluate_inbetweening_admission,
)

pytestmark = pytest.mark.fast


def _trial(index: int, outcome: HumanOutcome) -> InbetweeningTrial:
    return InbetweeningTrial(
        trial_id=f"trial-{index:03d}",
        human_outcome=outcome,
        hard_anchor_errors_m={"wrist_contact": 0.0002, "release": 0.0001},
        hard_anchor_tolerance_m=0.001,
        physics_gates={
            "joint_limits": True,
            "contact_order": True,
            "penetration": True,
            "balance": True,
            "object_state": True,
        },
    )


def _strong_trials() -> tuple[InbetweeningTrial, ...]:
    outcomes = (
        [HumanOutcome.LEARNED_PREFERRED] * 72
        + [HumanOutcome.BASELINE_PREFERRED] * 20
        + [HumanOutcome.TIE] * 8
    )
    return tuple(_trial(index, outcome) for index, outcome in enumerate(outcomes))


def test_learned_inbetweening_admitted_only_when_human_outcome_is_better() -> None:
    decision = evaluate_inbetweening_admission(_strong_trials())
    assert decision.admitted
    assert decision.learned_wins == 72
    assert decision.baseline_wins == 20
    assert decision.decisive_trials == 92
    assert decision.learned_preference_ci.lower > 0.5
    assert decision.one_sided_p_value < 0.05
    assert decision.hard_anchors_preserved
    assert decision.all_physics_gates_passed


def test_statistically_weak_human_result_is_rejected() -> None:
    trials = tuple(
        _trial(
            index,
            HumanOutcome.LEARNED_PREFERRED if index < 16 else HumanOutcome.BASELINE_PREFERRED,
        )
        for index in range(30)
    )
    decision = evaluate_inbetweening_admission(trials)
    assert not decision.admitted
    assert any("not statistically better" in reason for reason in decision.reasons)


def test_anchor_or_any_physics_gate_failure_vetoes_strong_human_preference() -> None:
    strong = list(_strong_trials())
    strong[0] = replace(
        strong[0], hard_anchor_errors_m={"wrist_contact": 0.0011}
    )
    anchor_failure = evaluate_inbetweening_admission(tuple(strong))
    assert not anchor_failure.admitted
    assert not anchor_failure.hard_anchors_preserved

    strong = list(_strong_trials())
    strong[1] = replace(
        strong[1], physics_gates={**strong[1].physics_gates, "penetration": False}
    )
    physics_failure = evaluate_inbetweening_admission(tuple(strong))
    assert not physics_failure.admitted
    assert not physics_failure.all_physics_gates_passed


def test_admission_requires_enough_decisive_trials_and_unique_ids() -> None:
    insufficient = tuple(
        _trial(index, HumanOutcome.LEARNED_PREFERRED) for index in range(20)
    )
    decision = evaluate_inbetweening_admission(insufficient)
    assert not decision.admitted
    assert any("at least 30" in reason for reason in decision.reasons)

    duplicate = (_trial(0, HumanOutcome.LEARNED_PREFERRED),) * 30
    with pytest.raises(ValueError, match="unique"):
        evaluate_inbetweening_admission(duplicate)
