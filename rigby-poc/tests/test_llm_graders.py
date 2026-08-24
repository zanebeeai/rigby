"""The four text-only graders from plan 10 §4.

Nothing in this category existed: planner evaluation is fixture classification
against expected intent, which measures routing, not fidelity.  These graders
read text and JSON only, and the two numbers the rejection auditor produces are
easy to confuse, so both are tested apart.

All against a stubbed client; no model calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evals.criteria import supported_cases, unsupported_cases
from rigby_poc.llm_graders import (
    DEFAULT_LLM_GRADER_MODEL,
    ClauseFidelityReport,
    FailureHonestyReport,
    LLMGraders,
    RejectionAuditReport,
    RejectionOutcome,
    RepairSoundnessReport,
    capability_grammar,
    expected_rejection_verdict,
    score_rejection_audit,
)
from rigby_poc.models import BodyAction, Intent, ObjectAction
from rigby_poc.planner import DEFAULT_PRIMARY_MODEL


class FakeUsage:
    def model_dump(self, mode: str = "json") -> dict[str, int]:
        return {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}


def _output(text_format: type):
    if text_format is ClauseFidelityReport:
        return ClauseFidelityReport.model_validate(
            {
                "clauses": [
                    {
                        "clause": "throw a left hook",
                        "present": True,
                        "correct_order": True,
                        "count_matches": True,
                        "laterality_matches": True,
                        "note": "Present as a strike step.",
                    }
                ],
                "hallucinated_additions": [],
                "summary": "The program encodes the request.",
            }
        )
    if text_format is RejectionAuditReport:
        return RejectionAuditReport.model_validate(
            {
                "verdict": "out_of_scope",
                "confidence": 0.8,
                "nearest_capability": "none",
                "rationale": "The grammar cannot express this request.",
            }
        )
    if text_format is RepairSoundnessReport:
        return RepairSoundnessReport.model_validate(
            {
                "responsive": True,
                "directionally_correct": True,
                "confidence": 0.7,
                "unaddressed_failures": [],
                "rationale": "The patch moves the cited parameters.",
            }
        )
    return FailureHonestyReport.model_validate(
        {
            "consistent": True,
            "confidence": 0.75,
            "rationale": "The reported failure names the gate that fired.",
        }
    )


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_llm",
            model=kwargs["model"],
            usage=FakeUsage(),
            output_parsed=_output(kwargs["text_format"]),
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def _graders() -> tuple[LLMGraders, FakeClient]:
    client = FakeClient()
    return LLMGraders(client=client), client


# ------------------------------------------------------ the capability grammar


def test_the_grammar_is_derived_from_the_enums_not_hand_written() -> None:
    grammar = capability_grammar()
    assert set(grammar["body_actions"]) == {value.value for value in BodyAction}
    assert set(grammar["object_actions"]) == {value.value for value in ObjectAction}
    # A hand-written grammar goes stale the moment someone adds an action, and a
    # rejection auditor reading a stale grammar calls a capability miss
    # "out of scope" — the exact error the grader exists to catch.
    assert set(grammar["intents"]) == {
        value.value for value in Intent if value != Intent.UNSUPPORTED
    }


def test_unsupported_is_not_a_capability() -> None:
    assert Intent.UNSUPPORTED.value not in capability_grammar()["intents"]


def test_graders_use_a_different_model_family_than_the_planner() -> None:
    # Plan 10 §4: an LLM grading a program written by an LLM will happily ratify
    # a misreading it would have made itself.
    planner_family = DEFAULT_PRIMARY_MODEL.split("-")[0]
    grader_family = DEFAULT_LLM_GRADER_MODEL.split("-")[0]
    assert planner_family != grader_family


# --------------------------------------------------------------- the payloads


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("clause_fidelity", {"prompt": "Throw a left hook.", "program": {"intent": "strike"}}),
        ("rejection_audit", {"prompt": "Fly to the moon.", "unsupported_reason": "no such action"}),
        ("repair_soundness", {"critique": {"summary": "wrist bent"}, "patch": {"wrist_pitch_delta": 0.1}}),
        (
            "failure_honesty",
            {"reported_failure": {"tag": "timing_error"}, "gate_output": {"gate": "ground_contact"}},
        ),
    ],
)
def test_no_grader_ever_sends_an_image(method: str, kwargs: dict) -> None:
    graders, client = _graders()
    getattr(graders, method)(**kwargs)
    for call in client.responses.calls:
        for message in call["input"]:
            for item in message["content"]:
                assert item["type"] == "input_text"


def test_the_rejection_auditor_is_given_the_grammar_and_the_reason() -> None:
    graders, client = _graders()
    record = graders.rejection_audit(
        prompt="Fly to the moon.", unsupported_reason="unsupported motion: flight"
    )
    payload = json.loads(client.responses.calls[0]["input"][1]["content"][0]["text"])
    assert payload["user_motion_request"] == "Fly to the moon."
    assert payload["planner_unsupported_reason"] == "unsupported motion: flight"
    assert payload["capability_grammar"] == capability_grammar()
    assert record["kind"] == "rejection_audit"
    assert record["call"]["parsed"]["verdict"] == "out_of_scope"


def test_clause_fidelity_receives_the_program_not_a_render() -> None:
    graders, client = _graders()
    graders.clause_fidelity(
        prompt="Throw a left hook then step over the box.",
        program={"intent": "sequence", "steps": [{"intent": "strike"}]},
    )
    payload = json.loads(client.responses.calls[0]["input"][1]["content"][0]["text"])
    assert payload["motion_program"]["intent"] == "sequence"


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("clause_fidelity", {"prompt": "p", "program": {}}),
        ("rejection_audit", {"prompt": "p", "unsupported_reason": "r"}),
        ("repair_soundness", {"critique": {}, "patch": {}}),
        ("failure_honesty", {"reported_failure": {}, "gate_output": {}}),
    ],
)
def test_every_grader_call_is_routed_and_observable(method: str, kwargs: dict) -> None:
    # §1.6's complaint about `recommend_repair` was that it had no attempts, no
    # routing and no correlation. A new grader shipping without them repeats it.
    graders, _ = _graders()
    record = getattr(graders, method)(**kwargs)
    assert record["routing"]["primary_model"] == DEFAULT_LLM_GRADER_MODEL
    assert record["routing"]["attempt_count"] == 1
    assert record["attempts"]
    assert record["call"]["usage"]["total_tokens"] == 28


def test_the_hard_model_call_budget_is_honoured() -> None:
    from rigby_poc.judge import ModelCallBudgetExhausted

    graders = LLMGraders(client=FakeClient(), max_model_calls=1)
    graders.failure_honesty(reported_failure={}, gate_output={})
    with pytest.raises(ModelCallBudgetExhausted):
        graders.failure_honesty(reported_failure={}, gate_output={})


# ----------------------------------------------------------------- the labels


def test_the_fixture_corpus_is_the_ground_truth_the_plan_claims() -> None:
    # Plan 10 §4 names "curated supported+unsupported fixtures, which already
    # exist (60 + 20)". Verified rather than assumed.
    assert len(supported_cases()) == 60
    assert len(unsupported_cases()) == 20


def test_a_refused_supported_prompt_is_a_capability_miss_by_construction() -> None:
    assert expected_rejection_verdict(supported=True) == "capability_miss"
    assert expected_rejection_verdict(supported=False) == "out_of_scope"


# ---------------------------------------------------------------- the scoring


def _outcomes(
    *, refused_supported: int, supported_total: int = 60, unsupported_total: int = 20,
    supported_verdict: str = "capability_miss", unsupported_verdict: str = "out_of_scope",
) -> list[RejectionOutcome]:
    outcomes = [
        RejectionOutcome(
            case_id=f"s{index:02d}",
            supported=True,
            refused=index < refused_supported,
            verdict=supported_verdict if index < refused_supported else None,
        )
        for index in range(supported_total)
    ]
    outcomes += [
        RejectionOutcome(
            case_id=f"u{index:02d}",
            supported=False,
            refused=True,
            verdict=unsupported_verdict,
        )
        for index in range(unsupported_total)
    ]
    return outcomes


def test_false_rejection_is_a_property_of_the_planner_not_the_grader() -> None:
    # It needs no model at all: the fixtures carry the label. Nothing in the
    # suite measures this today, and a system that over-rejects looks flawless
    # on every other metric.
    scores = score_rejection_audit(_outcomes(refused_supported=3))
    assert scores["false_rejection"].successes == 3
    assert scores["false_rejection"].n == 60
    assert scores["false_rejection_case_ids"] == ["s00", "s01", "s02"]


def test_a_planner_that_refuses_nothing_supported_has_a_zero_false_rejection_rate() -> None:
    scores = score_rejection_audit(_outcomes(refused_supported=0))
    assert scores["false_rejection"].estimate == 0.0
    assert scores["n_refusals"] == 20


def test_auditor_agreement_is_reported_against_its_baseline() -> None:
    scores = score_rejection_audit(_outcomes(refused_supported=2))
    assert scores["auditor_agreement"].n == 22
    assert scores["auditor_agreement"].estimate == 1.0
    # 20 of 22 refusals are genuinely out of scope, so always answering
    # `out_of_scope` already scores 0.909. Reporting agreement without this
    # number would make a constant predictor look like a working grader.
    assert scores["auditor_agreement_baseline"] == pytest.approx(20 / 22)


def test_a_constant_auditor_scores_at_baseline_and_is_visible_as_such() -> None:
    scores = score_rejection_audit(
        _outcomes(refused_supported=2, supported_verdict="out_of_scope")
    )
    assert scores["auditor_agreement"].estimate == pytest.approx(
        scores["auditor_agreement_baseline"]
    )


def test_the_two_rates_move_independently() -> None:
    # A planner that over-rejects and an auditor that spots it: false rejection
    # is high, agreement is perfect. Conflating them would report a healthy system.
    scores = score_rejection_audit(_outcomes(refused_supported=30))
    assert scores["false_rejection"].estimate == 0.5
    assert scores["auditor_agreement"].estimate == 1.0


def test_every_reported_rate_carries_its_n() -> None:
    scores = score_rejection_audit(_outcomes(refused_supported=1))
    for key in ("false_rejection", "auditor_agreement"):
        assert scores[key].n > 0
        assert scores[key].lower_bound_95 <= scores[key].estimate
