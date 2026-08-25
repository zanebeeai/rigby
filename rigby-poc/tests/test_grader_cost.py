"""Token cost is only meaningful beside the fraction of it that is attributable.

Three populations lose tokens in different ways — escalation after a bad parse
(attributable), escalation after a network error (not), and transient retries
(not, and it happens on calls that ultimately succeeded). The undercount is
concentrated in exactly the calls that failed hardest, so a total that averages
over it is systematically low wherever the system was struggling most.

Raised by lane `capture` from PR 01b's instrumentation.
"""

from __future__ import annotations

import pytest

from rigby_poc.grader_cost import CallCost, call_cost, cost_report, report_records


def _record(*, dispatches: int, usages: int, retries: int = 0, tokens: int = 100) -> dict:
    return {
        "routing": {"dispatch_count": dispatches, "transient_retry_count": retries},
        "attempts": [
            {"usage": {"input_tokens": tokens, "output_tokens": tokens // 5}}
            for _ in range(usages)
        ],
    }


# --------------------------------------------------------------- one call


def test_a_clean_call_attributes_every_dispatch() -> None:
    cost = call_cost("semantic", _record(dispatches=1, usages=1))
    assert cost.unattributed_dispatches == 0
    assert cost.total_tokens == 120


def test_a_transient_retry_leaves_a_dispatch_unattributed() -> None:
    # Two retries then success: three dispatches, one usage record, and the call
    # looks entirely healthy from its result.
    cost = call_cost("anatomy", _record(dispatches=3, usages=1, retries=2))
    assert cost.dispatches == 3
    assert cost.attributed_attempts == 1
    assert cost.unattributed_dispatches == 2
    assert cost.transient_retries == 2


def test_an_escalation_after_a_bad_parse_attributes_both_attempts() -> None:
    # The primary answered and then failed validation, so its tokens exist.
    cost = call_cost("timing", _record(dispatches=2, usages=2))
    assert cost.unattributed_dispatches == 0


def test_an_escalation_after_a_network_error_attributes_only_the_fallback() -> None:
    # Nothing answered on the primary, so there is nothing to count.
    cost = call_cost("artifact", _record(dispatches=2, usages=1))
    assert cost.unattributed_dispatches == 1


def test_an_attempt_without_a_usage_block_is_not_counted_as_attributed() -> None:
    record = _record(dispatches=2, usages=1)
    record["attempts"].append({"usage": {}})
    assert call_cost("crossview", record).attributed_attempts == 1


def test_a_partial_usage_block_is_not_counted() -> None:
    # Half a usage record is not usage; counting it would attribute a dispatch
    # whose tokens are unknown.
    record = _record(dispatches=1, usages=0)
    record["attempts"] = [{"usage": {"input_tokens": 10}}]
    assert call_cost("semantic", record).attributed_attempts == 0


def test_more_usage_than_dispatches_is_an_instrumentation_fault() -> None:
    """Impossible from the provider's side, so it is our bug, not theirs.

    Raising makes the report a self-check on its own instrumentation rather than
    a consumer that trusts whatever it is handed.
    """
    with pytest.raises(ValueError, match="exceed 1 dispatches"):
        call_cost("semantic", _record(dispatches=1, usages=2))


def test_a_record_without_routing_raises() -> None:
    with pytest.raises(ValueError, match="no routing block"):
        call_cost("semantic", {"attempts": []})


def test_a_record_without_a_dispatch_count_raises() -> None:
    with pytest.raises(ValueError, match="no dispatch_count"):
        call_cost("semantic", {"routing": {}, "attempts": []})


# ------------------------------------------------------------- the report


def test_a_complete_report_says_so() -> None:
    report = cost_report(
        [call_cost(f"g{index}", _record(dispatches=1, usages=1)) for index in range(5)]
    )
    assert report["attribution_ratio"].estimate == 1.0
    assert report["cost_figure_is_complete"] is True
    assert report["calls_with_unattributed_dispatches"] == []


def test_an_incomplete_report_names_where_the_shortfall_is() -> None:
    # Named, not counted: a ratio below 1.0 means the cost is low by an unknown
    # amount concentrated in these specific calls.
    report = report_records(
        [
            ("clean", _record(dispatches=1, usages=1)),
            ("retried", _record(dispatches=3, usages=1, retries=2)),
        ]
    )
    assert report["cost_figure_is_complete"] is False
    assert report["calls_with_unattributed_dispatches"] == ["retried"]
    assert report["attribution_ratio"].successes == 2
    assert report["attribution_ratio"].n == 4


def test_the_attribution_ratio_carries_its_n_and_an_interval() -> None:
    # Same standard every other published rate in the repo has to meet.
    report = cost_report([call_cost("g", _record(dispatches=3, usages=1, retries=2))])
    ratio = report["attribution_ratio"]
    assert ratio.n == 3
    assert ratio.lower_bound_95 <= ratio.estimate


def test_the_token_total_is_the_attributed_total_not_an_estimate() -> None:
    # No extrapolation over the unattributed dispatches. The gap is reported,
    # never filled in.
    report = cost_report([call_cost("g", _record(dispatches=3, usages=1, tokens=100))])
    assert report["total_tokens"] == 120
    assert report["dispatches"] == 3


def test_an_empty_report_raises_rather_than_reporting_a_ratio() -> None:
    with pytest.raises(ValueError, match="at least one call"):
        cost_report([])
