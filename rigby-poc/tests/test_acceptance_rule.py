"""Regression guard for the published minimum-selected-score rule.

`acceptance_criteria.yaml` requires `semantic_match >= 4`, but the judge's rule
gated only `overall`, `anatomical_naturalness`, and `gesture_recognizability`,
so a clip that scored 1 on semantics satisfied the rule.  See plan 07 §1.1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigby_poc.judge import (
    ACCEPTANCE_MINIMUM_SCORES,
    MotionJudgeScore,
    VLMJudge,
    meets_acceptance_thresholds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _score(**overrides: object) -> MotionJudgeScore:
    payload: dict[str, object] = {
        "semantic_match": 5,
        "gesture_recognizability": 5,
        "anatomical_naturalness": 5,
        "temporal_readability": 5,
        "egocentric_visibility": 5,
        "cross_view_consistency": 5,
        "overall": 5,
        "accept": True,
        "confidence": 0.95,
        "failure_tags": ["none"],
        "evidence": [
            {"snapshot_id": "01-ego", "observation": "Fist closed at impact."},
            {"snapshot_id": "02-orbit", "observation": "Elbow extends cleanly."},
        ],
        "summary": "Readable and structurally sound motion.",
        "suggested_adjustment": "No adjustment needed.",
    }
    payload.update(overrides)
    return MotionJudgeScore.model_validate(payload)


def test_rule_mirrors_the_published_acceptance_criteria() -> None:
    criteria = json.loads(
        (PROJECT_ROOT / "acceptance_criteria.yaml").read_text(encoding="utf-8")
    )
    published = criteria["autonomous_pipeline"]["minimum_selected_scores"]
    assert ACCEPTANCE_MINIMUM_SCORES == published


def test_semantic_match_of_one_is_rejected_with_everything_else_perfect() -> None:
    assert meets_acceptance_thresholds(_score(semantic_match=1)) is False


@pytest.mark.parametrize("dimension", sorted(ACCEPTANCE_MINIMUM_SCORES))
def test_every_published_dimension_is_gated(dimension: str) -> None:
    assert meets_acceptance_thresholds(_score(**{dimension: 3})) is False


def test_all_dimensions_at_threshold_are_accepted() -> None:
    assert meets_acceptance_thresholds(_score(**dict.fromkeys(ACCEPTANCE_MINIMUM_SCORES, 4)))


def test_ungated_dimensions_do_not_affect_the_rule() -> None:
    assert meets_acceptance_thresholds(
        _score(temporal_readability=1, egocentric_visibility=1, cross_view_consistency=1)
    )


def test_missing_dimension_fails_rather_than_being_skipped() -> None:
    complete = _score().model_dump(mode="json")
    assert meets_acceptance_thresholds(complete)
    for dimension in ACCEPTANCE_MINIMUM_SCORES:
        truncated = {key: value for key, value in complete.items() if key != dimension}
        assert meets_acceptance_thresholds(truncated) is False


def test_self_accepted_low_semantic_score_is_flagged_as_inconsistent() -> None:
    judge = VLMJudge(client=object(), model="test-model")
    # A model claiming acceptance on a clip that fails the published rule must
    # be escalated rather than waved through.
    assert (
        judge._unary_escalation(_score(semantic_match=1, accept=True))
        == "acceptance_score_inconsistency"
    )


def test_honouring_the_published_rule_is_not_treated_as_inconsistent() -> None:
    judge = VLMJudge(client=object(), model="test-model")
    # Under the old three-dimension rule this scored as accepted-by-rule, so a
    # correct `accept=False` looked like an inconsistency and burned a fallback
    # model call.
    assert judge._unary_escalation(_score(semantic_match=1, accept=False)) is None
