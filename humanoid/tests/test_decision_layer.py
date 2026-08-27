"""The explicit decision layer: `deterministic_valid and grader and thresholds`.

Plan 07 §3.3. Before 07d each term lived somewhere different and `grader_verdict`
lived nowhere at all — the model's self-reported boolean was copied verbatim by
five call sites and never checked against the published rule (§1.1). Making the
combination one pure function is what lets the graders be measured against the
deterministic layer rather than quietly agreeing with it.
"""

from __future__ import annotations

import pytest

from rigby_poc.judge import ACCEPTANCE_MINIMUM_SCORES, decide

pytestmark = pytest.mark.fast


def _scores(**overrides: int) -> dict[str, int]:
    scores = dict.fromkeys(ACCEPTANCE_MINIMUM_SCORES, 5)
    scores.update(overrides)
    return scores


# ------------------------------------------------------------- the three terms


def test_all_three_terms_must_hold() -> None:
    assert decide(_scores(), deterministic_valid=True).accepted is True


@pytest.mark.parametrize("dimension", sorted(ACCEPTANCE_MINIMUM_SCORES))
def test_a_failing_dimension_blocks_acceptance(dimension: str) -> None:
    decision = decide(
        _scores(**{dimension: ACCEPTANCE_MINIMUM_SCORES[dimension] - 1}),
        deterministic_valid=True,
    )
    assert decision.accepted is False
    assert "below_published_thresholds" in decision.reasons


def test_a_deterministic_failure_blocks_a_perfect_score() -> None:
    """The term plan §3.3 calls 'always the authority'.

    A clip the structural layer rejects cannot be accepted however the graders
    scored it — which is the whole reason the combination is explicit.
    """
    decision = decide(_scores(), deterministic_valid=False)
    assert decision.accepted is False
    assert decision.meets_thresholds is True
    assert decision.reasons == ("deterministic_invalid",)


def test_an_unjudged_gating_dimension_blocks_acceptance() -> None:
    decision = decide(
        _scores(), deterministic_valid=True, unjudged_dimensions=["anatomical_naturalness"]
    )
    assert decision.accepted is False
    assert decision.fully_judged is False
    assert "unjudged_gating_dimensions:anatomical_naturalness" in decision.reasons


def test_an_unjudged_ungated_dimension_does_not_block() -> None:
    decision = decide(
        _scores(), deterministic_valid=True, unjudged_dimensions=["temporal_readability"]
    )
    assert decision.accepted is True
    assert decision.fully_judged is True


# ---------------------------------------------------------------- the reasons


def test_reasons_are_complete_not_short_circuited() -> None:
    """A clip failing two terms says so.

    A single first-failure reason reads as though the other terms passed, which
    sends the reader to fix one thing and re-run into the second.
    """
    decision = decide(
        _scores(overall=1),
        deterministic_valid=False,
        unjudged_dimensions=["semantic_match"],
    )
    assert decision.reasons == (
        "deterministic_invalid",
        "below_published_thresholds",
        "unjudged_gating_dimensions:semantic_match",
    )


def test_an_accepted_clip_carries_no_reasons() -> None:
    assert decide(_scores(), deterministic_valid=True).reasons == ()


def test_multiple_unjudged_dimensions_are_all_named() -> None:
    decision = decide(
        _scores(),
        deterministic_valid=True,
        unjudged_dimensions=["overall", "semantic_match"],
    )
    assert "unjudged_gating_dimensions:overall,semantic_match" in decision.reasons


# ------------------------------------------------- no defaults for the authority


def test_deterministic_valid_has_no_default() -> None:
    """A default here would be a value indistinguishable from a measurement.

    A caller that forgot to pass it would get a decision that looks complete,
    with the term the plan calls 'always the authority' silently assumed true —
    the absence-becomes-a-value defect, in the one function that must not have it.
    """
    with pytest.raises(TypeError):
        decide(_scores())  # type: ignore[call-arg]


def test_the_record_carries_every_term_not_just_the_verdict() -> None:
    record = decide(_scores(overall=1), deterministic_valid=True).record()
    assert record == {
        "accepted": False,
        "deterministic_valid": True,
        "meets_thresholds": False,
        "fully_judged": True,
        "reasons": ["below_published_thresholds"],
    }
