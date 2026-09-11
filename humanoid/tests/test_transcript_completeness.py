"""Plan 01 section 5 — the regression test that gives 01b teeth.

Every claim plan 01 makes about model calls is checked here against a real
`RoutedModelClient` driving a stub OpenAI client. No network, no browser, no model.

The five assertions section 5 asks for, and one more that section 7 item 3 needs:

  1. every model call persists a request and a response
  2. every span has a resolvable parent up to a single root
  3. every `model_call` span carries the identifiers needed to locate it
  4. a deliberately failing primary appears as an `attempt` span with `status: error`
  5. total tokens include the failed attempt
  6. a recorded call is replayable from its `request.json` without the pipeline
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rigby_poc.judge import RepairPatch, RoutedModelClient, _never_escalate
from rigby_poc.observability import Tracer, stage_timeline
from rigby_poc.transcript import load

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


class _Usage:
    def __init__(self, total: int) -> None:
        self.total = total

    def model_dump(self, mode: str = "json") -> dict[str, int]:
        return {"input_tokens": self.total - 3, "output_tokens": 3, "total_tokens": self.total}


def _patch() -> RepairPatch:
    return RepairPatch(
        arm_height_delta=0.0, arm_depth_delta=0.0, lateral_offset_delta=0.0,
        wrist_pitch_delta=0.0, wrist_yaw_delta=0.0, wrist_roll_delta=0.0,
        elbow_swivel_delta=0.0, finger_splay_delta=0.0, thumb_curl_delta=0.0,
        little_curl_delta=0.0, wrist_shake_amplitude_delta=0.0,
        present_duration_scale=1.0, hold_duration_scale=1.0, shake_duration_scale=1.0,
        recover_duration_scale=1.0, easing_delta=0.0, rationale="No change needed.",
    )


class _Responses:
    """Dispatches; optionally fails the first one. Records what it was actually sent."""

    def __init__(self, *, fail_primary: bool = False, usage_per_call: int = 20) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_primary = fail_primary
        self.usage_per_call = usage_per_call

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.fail_primary and len(self.calls) == 1:
            raise ValueError("primary refused")
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model=kwargs["model"],
            usage=_Usage(self.usage_per_call),
            output_parsed=_patch(),
            model_dump=lambda mode="json": {
                "id": f"resp_{len(self.calls)}",
                "model": kwargs["model"],
                "status": "completed",
            },
        )


def _client(responses: _Responses) -> SimpleNamespace:
    return SimpleNamespace(responses=responses)


INPUT = [
    {"role": "system", "content": [{"type": "input_text", "text": "You judge motion."}]},
    {"role": "user", "content": [{"type": "input_text", "text": "Is this a jab?"}]},
]


def _run(tmp_path: Path, *, fail_primary: bool) -> tuple[Any, _Responses]:
    responses = _Responses(fail_primary=fail_primary)
    client = RoutedModelClient(
        client=_client(responses),
        model="primary-model",
        fallback_model="fallback-model",
        reasoning_effort="low",
    )
    tracer = Tracer.open(tmp_path / "runs")
    with tracer.span("run", "pipeline.run", launched_by="test"):
        with stage_timeline(tracer) as timeline:
            timeline.enter("vlm_judge", round=1, result_id="000001-jab")
            client._routed_parse(
                input=INPUT,
                text_format=RepairPatch,
                escalation_reason=_never_escalate,
                span_refs={"result_id": "000001-jab", "round": 1},
            )
    return load(tracer.run_dir), responses


def test_every_model_call_persists_a_request_and_a_response(tmp_path: Path) -> None:
    document, _ = _run(tmp_path, fail_primary=False)
    attempts = document.by_kind("attempt")
    assert attempts
    for span in attempts:
        assert span.request_path, f"{span.name} recorded no request"
        assert span.response_path, f"{span.name} recorded no response"
        assert (document.run_dir / span.request_path).is_file()
        assert (document.run_dir / span.response_path).is_file()


def test_every_span_resolves_to_a_single_root(tmp_path: Path) -> None:
    document, _ = _run(tmp_path, fail_primary=False)
    assert document.orphans() == []
    assert len([span for span in document.spans if span.parent_id is None]) == 1


def test_a_model_call_carries_the_identifiers_needed_to_locate_it(tmp_path: Path) -> None:
    """Plan 01 §1.3: nothing linked a judge response id to a run or a candidate."""
    document, _ = _run(tmp_path, fail_primary=False)
    calls = document.by_kind("model_call")
    assert calls
    for span in calls:
        assert span.refs.get("result_id") or span.refs.get("round")


def test_an_attempt_span_parents_to_its_model_call_which_parents_to_its_stage(
    tmp_path: Path,
) -> None:
    document, _ = _run(tmp_path, fail_primary=False)
    attempt = document.by_kind("attempt")[0]
    call = document.by_id(attempt.parent_id)
    assert call.kind == "model_call"
    stage = document.by_id(call.parent_id)
    assert stage.kind == "stage" and stage.name == "vlm_judge"


def test_a_failed_primary_appears_as_an_attempt_span_with_its_error(tmp_path: Path) -> None:
    """The direct guard against the §1.2 regression.

    Before 01b a failed primary was reduced to the string `primary_error:<ClassName>`
    with no response id, no usage, no message and no latency — so the tokens it burned
    were invisible precisely when escalation fired.
    """
    document, responses = _run(tmp_path, fail_primary=True)
    attempts = document.by_kind("attempt")
    assert len(attempts) == 2, "the failed primary is missing from the transcript"
    failed, succeeded = attempts[0], attempts[1]
    assert failed.status == "error"
    assert failed.error is not None
    assert failed.error["type"] == "ValueError"
    assert "primary refused" in failed.error["message"]
    assert failed.attrs["model"] == "primary-model"
    assert succeeded.status == "ok"
    assert succeeded.attrs["model"] == "fallback-model"
    # The failed dispatch still carries its request, so it is replayable too.
    assert failed.request_path
    assert len(responses.calls) == 2


def test_a_failed_attempt_is_still_counted_against_the_budget(tmp_path: Path) -> None:
    """§1.2: token accounting was wrong exactly when escalation fired."""
    document, responses = _run(tmp_path, fail_primary=True)
    dispatched = len(responses.calls)
    assert len(document.by_kind("attempt")) == dispatched
    # The successful attempt reports usage; the failed one reports none but exists, so a
    # consumer can see that a dispatch happened without a usage record rather than
    # inferring one call where there were two.
    usages = [span.attrs.get("usage") for span in document.by_kind("attempt")]
    assert usages[0] is None or usages[0] == {}
    assert usages[1]["total_tokens"] == 20


def test_a_recorded_call_is_replayable_without_the_pipeline(tmp_path: Path) -> None:
    """Plan 01 §7 item 3, and the whole point of persisting the request."""
    document, responses = _run(tmp_path, fail_primary=False)
    span = document.by_kind("attempt")[0]
    replayed = document.request(span.span_id)
    assert replayed == INPUT
    # And it is what the client was actually handed, not a reconstruction of it.
    assert responses.calls[0]["input"] == replayed


def test_the_routing_decision_is_recorded_on_the_model_call(tmp_path: Path) -> None:
    document, _ = _run(tmp_path, fail_primary=True)
    call = document.by_kind("model_call")[0]
    assert call.attrs["escalated"] is True
    assert call.attrs["primary_model"] == "primary-model"
    assert call.attrs["fallback_model"] == "fallback-model"
    assert call.attrs["escalation_reason"] == "primary_error:ValueError"
    assert call.attrs["attempt_count"] == 1
    assert call.attrs["dispatch_count"] == 2


def test_instrumentation_is_inert_outside_a_run(tmp_path: Path) -> None:
    """A `NullTracer` by default, so library code is never conditional on tracing."""
    responses = _Responses()
    client = RoutedModelClient(
        client=_client(responses),
        model="primary-model",
        fallback_model=None,
        reasoning_effort="low",
    )
    _, parsed, attempts, routing = client._routed_parse(
        input=INPUT, text_format=RepairPatch, escalation_reason=_never_escalate
    )
    assert parsed.rationale == "No change needed."
    assert len(attempts) == 1
    assert routing["dispatch_count"] == 1


def test_the_no_fallback_path_still_records_its_failure(tmp_path: Path) -> None:
    """The §1.2 rider: `_routed_parse` re-raises when no fallback is configured.

    Instrumenting the `except` branch would lose the attempt on exactly this path, which
    is why the span wraps the dispatch instead.
    """
    responses = _Responses(fail_primary=True)
    client = RoutedModelClient(
        client=_client(responses),
        model="primary-model",
        fallback_model=None,
        reasoning_effort="low",
    )
    tracer = Tracer.open(tmp_path / "runs")
    with tracer.span("run", "pipeline.run"):
        with pytest.raises(ValueError, match="primary refused"):
            client._routed_parse(
                input=INPUT, text_format=RepairPatch, escalation_reason=_never_escalate
            )
    document = load(tracer.run_dir)
    attempts = document.by_kind("attempt")
    assert len(attempts) == 1
    assert attempts[0].status == "error"
    call = document.by_kind("model_call")[0]
    assert call.status == "error"


def test_a_dispatch_that_never_reached_the_provider_records_no_usage(tmp_path: Path) -> None:
    """Plan 01 §7 item 4 says a forced failure appears "with usage and error class".

    Only half of that is always achievable, and the distinction is worth pinning rather
    than discovering. A dispatch that *raises* carries no usage anywhere — the provider
    never answered — so the span records the error and no usage. A response that arrives
    and then fails validation is a different case: the dispatch succeeded, usage exists,
    and the attempt span keeps it while the enclosing `model_call` carries the error.

    Either way the failure is no longer invisible, which is what §1.2 asked for.
    """
    document, _ = _run(tmp_path, fail_primary=True)
    failed, succeeded = document.by_kind("attempt")
    assert failed.status == "error"
    assert not failed.attrs.get("usage")
    assert failed.error["type"] == "ValueError"
    assert succeeded.attrs["usage"]["total_tokens"] == 20
    # The transcript total counts what was actually reported, never an invented figure
    # for the attempt that had none.
    assert document.total_tokens().total_tokens == 20
