"""The instrument 07d has to report itself with.

Plan 07 §4 requires 07d to show accept rates before and after on the same
corpus, with n, rather than asserting the new path is better. This is that
measurement, built and tested before the change it measures, so the change
cannot be scored by a tool written to flatter it.
"""

from __future__ import annotations

import pytest

from rigby_poc.acceptance_delta import (
    AcceptanceComparison,
    acceptance_delta,
    compare_record,
    compare_records,
)


def _record(*, accept: bool, unjudged: list[str] | None = None, **scores: int) -> dict:
    parsed = {
        "semantic_match": 5,
        "gesture_recognizability": 5,
        "anatomical_naturalness": 5,
        "temporal_readability": 5,
        "egocentric_visibility": 5,
        "cross_view_consistency": 5,
        "overall": 5,
        "accept": accept,
    }
    parsed.update(scores)
    record: dict = {"call": {"parsed": parsed}}
    if unjudged is not None:
        record["insufficient_evidence_dimensions"] = unjudged
    return record


# ------------------------------------------------------------ reading a record


def test_a_self_accepted_breach_reads_as_newly_rejected() -> None:
    # The 07a case: the model said yes on a clip the published rule fails.
    comparison = compare_record("c1", _record(accept=True, semantic_match=1))
    assert comparison.self_reported is True
    assert comparison.rule_derived is False
    assert comparison.direction == "newly_rejected"


def test_a_self_rejected_passing_clip_reads_as_newly_accepted() -> None:
    # The rarer direction, and the one that would be missed by only counting
    # clips the new rule turns down.
    comparison = compare_record("c2", _record(accept=False))
    assert comparison.direction == "newly_accepted"


def test_agreement_reads_as_unchanged() -> None:
    assert compare_record("c3", _record(accept=True)).direction == "unchanged"
    assert compare_record("c4", _record(accept=False, overall=1)).direction == "unchanged"


def test_an_unjudged_gating_dimension_is_never_accepted() -> None:
    comparison = compare_record(
        "c5", _record(accept=True, unjudged=["anatomical_naturalness"])
    )
    assert comparison.rule_derived is False


def test_an_unjudged_ungated_dimension_does_not_block() -> None:
    comparison = compare_record("c6", _record(accept=True, unjudged=["temporal_readability"]))
    assert comparison.rule_derived is True


def test_a_record_without_a_self_reported_accept_raises() -> None:
    # Silently treating a missing flag as False would manufacture a delta.
    with pytest.raises(ValueError, match="no self-reported accept"):
        compare_record("c7", {"call": {"parsed": {"semantic_match": 5}}})


def test_a_record_without_a_parsed_score_raises() -> None:
    with pytest.raises(ValueError, match="no parsed score"):
        compare_record("c8", {"call": {"parsed": None}})


# ------------------------------------------------------------- the delta report


def _comparisons(*, unchanged: int, rejected: int, accepted: int) -> list[AcceptanceComparison]:
    items = [
        AcceptanceComparison(f"u{i}", self_reported=True, rule_derived=True)
        for i in range(unchanged)
    ]
    items += [
        AcceptanceComparison(f"r{i}", self_reported=True, rule_derived=False)
        for i in range(rejected)
    ]
    items += [
        AcceptanceComparison(f"a{i}", self_reported=False, rule_derived=True)
        for i in range(accepted)
    ]
    return items


def test_both_rates_carry_their_n() -> None:
    report = acceptance_delta(_comparisons(unchanged=8, rejected=2, accepted=0))
    assert report["before"].n == 10
    assert report["after"].n == 10
    assert report["before"].estimate == 1.0
    assert report["after"].estimate == 0.8


def test_a_zero_delta_still_reports_the_disagreement() -> None:
    """The reason this is two rates and a breakdown rather than one delta.

    Three clips newly rejected and three newly accepted nets to zero. Reporting
    only the delta would say "07d changed nothing" about a judge that changed
    its mind on six of ten clips.
    """
    report = acceptance_delta(_comparisons(unchanged=4, rejected=3, accepted=3))
    assert report["before"].estimate == report["after"].estimate
    assert report["changed"] == 6
    assert report["newly_rejected"] == 3
    assert report["newly_accepted"] == 3


def test_the_moved_cases_are_named_not_just_counted() -> None:
    report = acceptance_delta(_comparisons(unchanged=1, rejected=2, accepted=1))
    assert report["newly_rejected_case_ids"] == ["r0", "r1"]
    assert report["newly_accepted_case_ids"] == ["a0"]


def test_the_report_says_what_it_does_not_attribute_to() -> None:
    # §6.2's honesty requirement, enforced structurally: removing diagnostics
    # changes what the model sees and cannot be measured from a recorded payload.
    report = acceptance_delta(_comparisons(unchanged=1, rejected=1, accepted=0))
    assert "diagnostics_removal" in report["does_not_attribute_to"]
    assert "decision_layer_authority" in report["attributes_to"]


def test_an_empty_corpus_raises_rather_than_reporting_a_rate() -> None:
    with pytest.raises(ValueError, match="at least one comparison"):
        acceptance_delta([])


def test_records_can_be_compared_end_to_end() -> None:
    report = compare_records(
        [
            ("good", _record(accept=True)),
            ("bad", _record(accept=True, semantic_match=1)),
        ]
    )
    assert report["n"] == 2
    assert report["before"].estimate == 1.0
    assert report["after"].estimate == 0.5
    assert report["newly_rejected_case_ids"] == ["bad"]


# ------------------------------------------------------------- render provenance


def test_a_render_mode_is_read_off_the_manifest_provenance() -> None:
    record = _record(accept=True)
    record["render_provenance"] = {"render_mode": "deterministic"}
    assert compare_record("c9", record).render_mode == "deterministic"


def test_pooling_across_render_modes_is_refused_not_caveated() -> None:
    """AA changes the magnitude of the pixel difference a pose delta produces,
    not just its noise floor, so a before/after that crosses modes measures the
    mode. A caveat on such a rate would be read past; this raises instead."""
    mixed = [
        AcceptanceComparison("a", self_reported=True, rule_derived=True, render_mode="judged"),
        AcceptanceComparison(
            "b", self_reported=True, rule_derived=False, render_mode="deterministic"
        ),
    ]
    with pytest.raises(ValueError, match="cannot pool acceptance across render modes"):
        acceptance_delta(mixed)


def test_one_render_mode_is_named_on_the_report() -> None:
    same = [
        AcceptanceComparison("a", self_reported=True, rule_derived=True, render_mode="judged"),
        AcceptanceComparison("b", self_reported=True, rule_derived=False, render_mode="judged"),
    ]
    assert acceptance_delta(same)["render_mode"] == "judged"
