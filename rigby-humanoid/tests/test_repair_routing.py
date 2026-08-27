"""`recommend_repair` must be routed like every other judge call.

Plan 07 §1.6/§3.6: it called `_parse_response` directly, so it carried no
`attempts`, no `routing`, no transient retry, and no fallback — the least
observable model call in the system, feeding the repair loop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rigby_poc.judge import RepairPatch, VLMJudge

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


class FakeUsage:
    def model_dump(self, mode: str = "json") -> dict[str, int]:
        return {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}


def _patch(rationale: str) -> RepairPatch:
    return RepairPatch(
        arm_height_delta=0.02,
        arm_depth_delta=0.0,
        lateral_offset_delta=0.0,
        wrist_pitch_delta=0.0,
        wrist_yaw_delta=0.0,
        wrist_roll_delta=0.0,
        elbow_swivel_delta=0.0,
        finger_splay_delta=0.0,
        thumb_curl_delta=0.0,
        little_curl_delta=0.0,
        wrist_shake_amplitude_delta=0.0,
        present_duration_scale=1.0,
        hold_duration_scale=1.0,
        shake_duration_scale=1.0,
        recover_duration_scale=1.0,
        easing_delta=0.0,
        rationale=rationale,
    )


class RecordingResponses:
    """Fails the primary model once, then answers on the fallback."""

    def __init__(self, *, fail_primary: bool) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_primary = fail_primary

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.fail_primary and len(self.calls) == 1:
            raise ValueError("primary refused to emit a patch")
        return SimpleNamespace(
            id=f"resp_repair_{len(self.calls)}",
            model=kwargs["model"],
            usage=FakeUsage(),
            output_parsed=_patch(f"Repair from {kwargs['model']}."),
        )


class RecordingClient:
    def __init__(self, *, fail_primary: bool = False) -> None:
        self.responses = RecordingResponses(fail_primary=fail_primary)


def _recommend(judge: VLMJudge) -> dict[str, Any]:
    return judge.recommend_repair(
        prompt="throw a right jab",
        motion_profile={"intent": "strike"},
        parameters={"arm_height": 0.0},
        judgment={"accept": False, "semantic_match": 2},
        structural_metrics={"structural_valid": True, "structural_failures": []},
    )


def test_repair_record_carries_attempts_and_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_JUDGE_REPAIR_MODEL", raising=False)
    client = RecordingClient()
    judge = VLMJudge(client=client, model="gpt-5.6-luna", fallback_model="gpt-5.6-terra")
    record = _recommend(judge)

    assert record["kind"] == "bounded_motion_repair"
    assert record["routing"]["primary_model"] == "gpt-5.6-luna"
    assert record["routing"]["fallback_model"] == "gpt-5.6-terra"
    assert record["routing"]["escalated"] is False
    assert record["routing"]["attempt_count"] == 1
    assert len(record["attempts"]) == 1
    assert record["call"]["parsed"]["rationale"] == "Repair from gpt-5.6-luna."


def test_repair_falls_back_when_the_primary_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_JUDGE_REPAIR_MODEL", raising=False)
    client = RecordingClient(fail_primary=True)
    judge = VLMJudge(client=client, model="gpt-5.6-luna", fallback_model="gpt-5.6-terra")
    record = _recommend(judge)

    assert [call["model"] for call in client.responses.calls] == [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
    ]
    assert record["routing"]["escalated"] is True
    assert record["routing"]["escalation_reason"] == "primary_error:ValueError"
    assert record["call"]["parsed"]["rationale"] == "Repair from gpt-5.6-terra."


def test_repair_uses_the_configured_repair_model_as_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_JUDGE_REPAIR_MODEL", "gpt-5.6-repair")
    client = RecordingClient()
    judge = VLMJudge(client=client, model="gpt-5.6-luna", fallback_model="gpt-5.6-terra")
    record = _recommend(judge)

    assert [call["model"] for call in client.responses.calls] == ["gpt-5.6-repair"]
    assert record["routing"]["primary_model"] == "gpt-5.6-repair"


def test_repair_never_escalates_on_the_patch_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_JUDGE_REPAIR_MODEL", raising=False)
    client = RecordingClient()
    judge = VLMJudge(client=client, model="gpt-5.6-luna", fallback_model="gpt-5.6-terra")
    record = _recommend(judge)

    # A bounded patch has no quality signal worth a second opinion, so routing
    # must not spend a fallback call on a successful parse.
    assert len(client.responses.calls) == 1
    assert record["routing"]["escalation_requested_reason"] is None
