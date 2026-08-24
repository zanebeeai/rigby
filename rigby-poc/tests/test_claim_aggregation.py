"""The claims-to-score rule, table driven, including all-`cannot_tell`.

Plan 07 §3.2: scores are derived from claims by an explicit, testable aggregation
rule rather than asked for directly, and `cannot_tell` is a first-class answer —
what makes a low-evidence case distinguishable from a failure.

`test_claim_aggregation.py` from the plan's §5 test list.
"""

from __future__ import annotations

import typing

import pytest

from rigby_poc.judge import (
    ACCEPTANCE_MINIMUM_SCORES,
    OVERALL_SOURCE_DIMENSIONS,
    UNJUDGED_DIMENSION_SCORE,
    FailureTag,
    GraderClaims,
    VLMJudge,
    aggregate_grader_dimensions,
    assemble_split_score,
    grader_dimension_verdicts,
    unexpected_claim_ids,
)
from rigby_poc.judge_claims import (
    CRITICAL_FAILURE_CAP,
    INSUFFICIENT_EVIDENCE_FRACTION,
    ClaimSpec,
    aggregate_claims,
    claim_ids,
    claim_specs,
)
from rigby_poc.judge_prompts import FAMILY_NAMES, GRADER_NAMES, GRADER_SPECS


INTENT = "strike"


def _specs(count: int, *, critical_at: int | None = None) -> tuple[ClaimSpec, ...]:
    return tuple(
        ClaimSpec(
            id=f"claim_{index}",
            question=f"Claim {index} holds.",
            critical=index == critical_at,
            tag="none",
        )
        for index in range(count)
    )


def _verdicts(specs: tuple[ClaimSpec, ...], *pattern: str) -> dict[str, str]:
    assert len(pattern) == len(specs)
    return {spec.id: verdict for spec, verdict in zip(specs, pattern)}


# ------------------------------------------------------- the rule, table driven


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (("yes", "yes", "yes", "yes"), 5),
        (("yes", "yes", "yes", "no"), 4),
        (("yes", "yes", "no", "no"), 3),
        (("yes", "no", "no", "no"), 2),
        (("no", "no", "no", "no"), 1),
        # `cannot_tell` is excluded from the denominator, not counted as a pass
        # or a fail: two of four settled and both passed is still a 5.
        (("yes", "yes", "cannot_tell", "cannot_tell"), 5),
        (("yes", "no", "cannot_tell", "cannot_tell"), 3),
        (("no", "no", "cannot_tell", "cannot_tell"), 1),
    ],
)
def test_settled_pass_fraction_maps_onto_one_to_five(
    pattern: tuple[str, ...], expected: int
) -> None:
    specs = _specs(4)
    verdict = aggregate_claims("dim", specs, _verdicts(specs, *pattern))
    assert verdict.score == expected


def test_all_cannot_tell_is_unjudged_rather_than_scored_low() -> None:
    specs = _specs(4)
    verdict = aggregate_claims("dim", specs, _verdicts(specs, *(["cannot_tell"] * 4)))
    assert verdict.score is None
    assert verdict.insufficient_evidence is True
    assert verdict.cannot_tell == 4
    assert verdict.failed_claim_ids == ()


def test_the_insufficient_evidence_boundary_is_a_strict_majority() -> None:
    specs = _specs(4)
    half = aggregate_claims("dim", specs, _verdicts(specs, "yes", "yes", "cannot_tell", "cannot_tell"))
    over = aggregate_claims("dim", specs, _verdicts(specs, "yes", "cannot_tell", "cannot_tell", "cannot_tell"))
    assert INSUFFICIENT_EVIDENCE_FRACTION == 0.5
    assert half.score == 5
    assert over.score is None


def test_a_critical_no_caps_the_dimension_below_the_gate() -> None:
    specs = _specs(10, critical_at=0)
    # Nine of ten pass, which would otherwise round to 5.
    pattern = ["no"] + ["yes"] * 9
    verdict = aggregate_claims("dim", specs, _verdicts(specs, *pattern))
    assert verdict.score == CRITICAL_FAILURE_CAP
    assert verdict.critical_failure_ids == ("claim_0",)
    assert CRITICAL_FAILURE_CAP < min(ACCEPTANCE_MINIMUM_SCORES.values())


def test_volume_cannot_outvote_a_critical_failure() -> None:
    specs = _specs(20, critical_at=3)
    pattern = ["yes"] * 20
    pattern[3] = "no"
    assert aggregate_claims("dim", specs, _verdicts(specs, *pattern)).score == CRITICAL_FAILURE_CAP


def test_a_non_critical_no_does_not_cap() -> None:
    specs = _specs(10)
    pattern = ["no"] + ["yes"] * 9
    verdict = aggregate_claims("dim", specs, _verdicts(specs, *pattern))
    assert verdict.score == 5
    assert verdict.critical_failure_ids == ()
    assert verdict.failed_claim_ids == ("claim_0",)


def test_a_missing_verdict_raises_rather_than_reading_as_cannot_tell() -> None:
    specs = _specs(3)
    verdicts = _verdicts(specs, "yes", "yes", "yes")
    del verdicts["claim_1"]
    with pytest.raises(ValueError, match="claim_1 has no usable verdict"):
        aggregate_claims("dim", specs, verdicts)


def test_an_unparseable_verdict_raises() -> None:
    specs = _specs(2)
    with pytest.raises(ValueError, match="has no usable verdict"):
        aggregate_claims("dim", specs, {"claim_0": "yes", "claim_1": "probably"})


def test_an_empty_claim_set_raises() -> None:
    with pytest.raises(ValueError, match="no claims to aggregate"):
        aggregate_claims("dim", (), {})


# ------------------------------------------------------------ the claim sets


def test_every_claim_id_is_unique_across_every_grader_and_family() -> None:
    for family in FAMILY_NAMES:
        seen: set[str] = set()
        for grader in GRADER_NAMES:
            for claim_id in claim_ids(grader, intent=family):
                assert claim_id not in seen, f"{claim_id} repeats in family {family}"
                seen.add(claim_id)


def test_every_claim_tag_is_a_real_failure_tag() -> None:
    allowed = set(typing.get_args(FailureTag))
    for family in FAMILY_NAMES:
        for grader in GRADER_NAMES:
            for spec in claim_specs(grader, intent=family):
                assert spec.tag in allowed, f"{spec.id} carries unknown tag {spec.tag}"


def test_every_grader_owning_a_gating_dimension_has_a_critical_claim() -> None:
    # A dimension that gates release with no disqualifying observation behind it
    # can only ever fail by volume, which is how a rule stops biting quietly.
    for grader in GRADER_NAMES:
        if not set(GRADER_SPECS[grader].dimensions) & set(ACCEPTANCE_MINIMUM_SCORES):
            continue
        specs = claim_specs(grader, intent=INTENT)
        assert any(spec.critical for spec in specs), grader


def test_anatomy_asks_about_forearm_and_wrist_twist_separately() -> None:
    # They are different bones. One question about "twist" answers about two
    # joints at once, which is a measurement error rather than a wording one.
    ids = claim_ids("anatomy")
    assert "anatomy.forearm.twist" in ids
    assert "anatomy.wrist.twist" in ids


# --------------------------------------------------- graders into dimensions


def _parts(verdict_by_grader: dict[str, str] | None = None) -> dict[str, dict]:
    chosen = verdict_by_grader or {}
    parts: dict[str, dict] = {}
    for grader in GRADER_NAMES:
        parts[grader] = {
            "claims": [
                {
                    "id": spec.id,
                    "verdict": chosen.get(grader, "yes"),
                    "confidence": 0.9,
                    "snapshot_id": "01-ego",
                }
                for spec in claim_specs(grader, intent=INTENT)
            ],
            "summary": f"{grader} summary.",
        }
    parts["semantic"]["suggested_adjustment"] = "Preserve the pose."
    return parts


def _fail_one(parts: dict[str, dict], grader: str, *, critical: bool) -> str:
    """Flip exactly one claim of `grader` to `no` and return its id."""
    target = next(
        spec for spec in claim_specs(grader, intent=INTENT) if spec.critical is critical
    )
    for claim in parts[grader]["claims"]:
        if claim["id"] == target.id:
            claim["verdict"] = "no"
            return target.id
    raise AssertionError(target.id)


def test_a_grader_owning_two_dimensions_derives_both_from_one_claim_set() -> None:
    verdicts = grader_dimension_verdicts("semantic", _parts()["semantic"], intent=INTENT)
    assert set(verdicts) == {"semantic_match", "gesture_recognizability"}


def test_a_missing_grader_raises_rather_than_defaulting() -> None:
    parts = _parts()
    del parts["anatomy"]
    with pytest.raises(ValueError, match="missing grader result: anatomy"):
        aggregate_grader_dimensions(parts, intent=INTENT)


@pytest.mark.parametrize("grader", ["semantic", "anatomy"])
def test_overall_follows_the_gating_dimensions(grader: str) -> None:
    parts = _parts()
    _fail_one(parts, grader, critical=True)
    merged = aggregate_grader_dimensions(parts, intent=INTENT)
    assert merged["overall"].score == CRITICAL_FAILURE_CAP


def test_everything_failing_scores_one_not_the_cap() -> None:
    merged = aggregate_grader_dimensions(_parts({"anatomy": "no"}), intent=INTENT)
    assert merged["anatomical_naturalness"].score == 1


@pytest.mark.parametrize("grader", ["artifact", "timing", "crossview"])
def test_ungated_dimensions_do_not_drag_overall_down(grader: str) -> None:
    # `acceptance_criteria.yaml` deliberately does not gate these; deriving
    # `overall` from them would promote them into gating dimensions by the back door.
    merged = aggregate_grader_dimensions(_parts({grader: "no"}), intent=INTENT)
    assert merged["overall"].score == 5
    assert set(OVERALL_SOURCE_DIMENSIONS) == set(ACCEPTANCE_MINIMUM_SCORES) - {"overall"}


def test_overall_is_unjudged_when_a_source_dimension_is_unjudged() -> None:
    merged = aggregate_grader_dimensions(_parts({"anatomy": "cannot_tell"}), intent=INTENT)
    assert merged["anatomical_naturalness"].score is None
    assert merged["overall"].score is None


def test_invented_claim_ids_are_recorded_not_ignored() -> None:
    parts = _parts()
    parts["timing"]["claims"].append(
        {"id": "vibes_are_good", "verdict": "yes", "confidence": 1.0, "snapshot_id": "01-ego"}
    )
    assert unexpected_claim_ids("timing", parts["timing"]) == ["vibes_are_good"]
    # ...and do not stop the four correct graders from being scored.
    assert aggregate_grader_dimensions(parts, intent=INTENT)["temporal_readability"].score == 5


# --------------------------------------------------------- the derived score


def test_a_fully_passing_clip_is_accepted() -> None:
    score, verdicts = assemble_split_score(_parts(), intent=INTENT)
    assert score.accept is True
    assert score.overall == 5
    assert score.failure_tags == ["none"]


def test_accept_is_derived_not_self_reported() -> None:
    # One critical claim fails and eight pass: the dimension is capped below the
    # gate regardless, and nothing self-reports an `accept`.
    parts = _parts()
    failed = _fail_one(parts, "semantic", critical=True)
    score, verdicts = assemble_split_score(parts, intent=INTENT)
    assert score.accept is False
    assert score.semantic_match == CRITICAL_FAILURE_CAP
    assert verdicts["semantic_match"].critical_failure_ids == (failed,)


def test_a_single_non_critical_failure_does_not_block_acceptance() -> None:
    parts = _parts()
    _fail_one(parts, "semantic", critical=False)
    score, _ = assemble_split_score(parts, intent=INTENT)
    assert score.accept is True


def test_an_unjudged_gating_dimension_is_never_accepted() -> None:
    score, verdicts = assemble_split_score(_parts({"anatomy": "cannot_tell"}), intent=INTENT)
    assert verdicts["anatomical_naturalness"].score is None
    assert score.anatomical_naturalness == UNJUDGED_DIMENSION_SCORE
    assert UNJUDGED_DIMENSION_SCORE < ACCEPTANCE_MINIMUM_SCORES["anatomical_naturalness"]
    assert score.accept is False


def test_unjudged_is_rejected_by_the_rule_not_only_by_the_placeholder_score(
    monkeypatch,
) -> None:
    """The `fully_judged` guard must bite on its own.

    `UNJUDGED_DIMENSION_SCORE = 3` already fails the gate, so the guard and the
    placeholder are two mechanisms enforcing one rule and either alone would make
    the tests pass. That is precisely the shape of §1.1's bug — a check that
    cannot be observed to fail. Raise the placeholder above the gate and the
    guard must still reject.
    """
    monkeypatch.setattr("rigby_poc.judge.UNJUDGED_DIMENSION_SCORE", 5)
    score, verdicts = assemble_split_score(_parts({"anatomy": "cannot_tell"}), intent=INTENT)
    assert verdicts["anatomical_naturalness"].score is None
    assert score.anatomical_naturalness == 5
    assert score.accept is False


def test_an_unjudged_ungated_dimension_still_permits_acceptance() -> None:
    score, verdicts = assemble_split_score(_parts({"timing": "cannot_tell"}), intent=INTENT)
    assert verdicts["temporal_readability"].score is None
    assert score.accept is True


def test_failure_tags_are_derived_from_which_claims_failed() -> None:
    score, _ = assemble_split_score(_parts({"artifact": "no"}), intent=INTENT)
    assert "hand_cropped" in score.failure_tags
    assert "none" not in score.failure_tags


def test_evidence_cites_the_failures_before_the_passes() -> None:
    score, _ = assemble_split_score(_parts({"crossview": "no"}), intent=INTENT)
    assert score.evidence[0].observation.startswith("crossview/")
    assert ": no" in score.evidence[0].observation


def test_evidence_cites_cannot_tell_before_passes() -> None:
    score, _ = assemble_split_score(_parts({"timing": "cannot_tell"}), intent=INTENT)
    assert ": cannot_tell" in score.evidence[0].observation


def test_confidence_is_the_least_confident_grader() -> None:
    parts = _parts()
    for claim in parts["artifact"]["claims"]:
        claim["confidence"] = 0.31
    score, _ = assemble_split_score(parts, intent=INTENT)
    assert score.confidence == pytest.approx(0.31)


# ------------------------------------------------------------- routing hook


def _claims(verdict: str, count: int = 4, confidence: float = 0.9) -> GraderClaims:
    return GraderClaims.model_validate(
        {
            "claims": [
                {
                    "id": f"claim_{index}",
                    "verdict": verdict,
                    "confidence": confidence,
                    "snapshot_id": "01-ego",
                }
                for index in range(count)
            ],
            "summary": "A summary.",
        }
    )


def test_a_grader_that_cannot_settle_its_claims_escalates() -> None:
    judge = VLMJudge(client=object(), model="test-model")
    assert judge._grader_escalation(_claims("cannot_tell")) == "insufficient_evidence"
    assert judge._grader_escalation(_claims("yes")) is None
    assert judge._grader_escalation(_claims("no")) is None
    assert judge._grader_escalation(_claims("yes", confidence=0.2)) == "low_confidence"
