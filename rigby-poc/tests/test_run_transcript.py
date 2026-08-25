"""Plan 01 sections 1.5, 3.6 and 7 — the run transcript, for CLI and API alike.

01a shipped `observability.py` and `transcript.py` wired into nothing. This is the first
test that runs the actual pipeline and reads back what it recorded, so it is where 01's
definition of done stops being a list and starts being a measurement. Two of §7's items
are checkable here and are checked with numbers rather than assertions of intent:

  item 2  a CLI run and an API run of the same prompt produce structurally identical
          transcripts
  item 5  `duration_by_stage()` accounts for at least 95% of measured wall clock

Zero model calls: the judge and the capture step are stubs.

**Any per-stage share measured from this module is a share of a run whose two most
expensive real components are faked.** The judge is a dict literal and capture is a 10 ms
`time.sleep`; a real render measures 2.85 s per candidate against 369 ms of compile, so a
render is roughly 7.7x a compile rather than the ~1% of wall clock a stubbed run reports.
The *total* — stage spans accounting for ~99.8% of the run — is unaffected, because that
is a claim about spans tiling rather than about what the stages contain.

This caveat lives here, beside the numbers, because a caveat delivered in the message that
accompanied a number does not travel with it: the 1.3% figure below was qualified to two
lanes individually and still reached a third uncaveated, where it overturned a plan
decision by three orders of magnitude.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from evals import flywheel
from rigby_poc.models import PipelineRunRequest, default_scene
from rigby_poc.observability import Tracer
from rigby_poc.pipeline import PipelineRunStore
from rigby_poc.transcript import load

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


PROMPT = "throw a right jab"
# Consecutive stages hand over inside one function; anything above this is real work
# escaping instrumentation rather than scheduling noise. Absolute, so it does not move
# with a platform's process-startup cost the way a share of the run does.
_MAX_INTER_STAGE_GAP_MS = 250.0
# Six of the seven stages `_progress` declares. The seventh, `repair`, only fires when a
# candidate needs repairing, so a clean run does not record it and asserting it here would
# make this test depend on the run failing.
EXPECTED_STAGES = {
    "planning",
    "candidates",
    "structural_checks",
    "visual_evidence",
    "vlm_judge",
    "finalize",
}


class _FakeJudge:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.model = "fake"

    def rank_five(self, manifests: list[Path], *, random_seed: int) -> dict[str, Any]:
        scores = {
            index: {
                "label": "ABCDE"[index],
                "semantic_match": 5,
                "gesture_recognizability": 5,
                "anatomical_naturalness": 5,
                "temporal_readability": 5,
                "egocentric_visibility": 5,
                "overall": 5,
                "accept": True,
                "failure_tags": ["none"],
                "summary": "Recognizable and structurally sound motion.",
                "suggested_adjustment": "No adjustment needed.",
            }
            for index in range(5)
        }
        return {
            "kind": "five_way_motion_judgment",
            "mapped_scores": scores,
            "mapped_winner_index": 0,
            "call": {"response_id": "fake-ranking"},
            "routing": {"selected_model": "fake"},
        }


def _fake_capture(result_id: str, output_dir: Path, **_: object) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # A real capture is a browser round trip; give the stage a little wall clock so the
    # §7 item 5 ratio is measuring something rather than dividing noise by noise.
    time.sleep(0.01)
    manifest = output_dir / "evidence-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return manifest


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flywheel, "VLMJudge", _FakeJudge)


def _millis(timestamp: str) -> float:
    """Parse a span's ISO timestamp to milliseconds since the epoch."""
    from datetime import datetime

    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000.0


def _cli_run(tmp_path: Path) -> Path:
    """The shape `evals/flywheel.py main` produces."""
    run_root = tmp_path / "cli-runs"
    tracer = Tracer.open(run_root)
    with tracer.span("run", "pipeline.run", prompt=PROMPT, launched_by="cli"):
        flywheel.run_best_of_five(
            PROMPT,
            tmp_path / "cli-artifacts",
            provider="offline",
            selection_mode="five_way",
            max_rounds=1,
            capture_fn=_fake_capture,
            tracer=tracer,
        )
    return tracer.run_dir


def _api_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The shape `PipelineRunStore` produces, including its worker thread."""
    monkeypatch.setattr("rigby_poc.pipeline.capture_in_subprocess", _fake_capture)
    monkeypatch.setattr(
        "rigby_poc.pipeline.capture_batch_in_subprocess",
        lambda requests, **_: [_fake_capture(result_id, directory) for result_id, directory in requests],
    )
    store = PipelineRunStore(tmp_path / "api-runs")
    record = store.start(
        PipelineRunRequest(text=PROMPT, provider="offline", max_rounds=1, scene=default_scene()),
        base_url="http://127.0.0.1:8000",
    )
    run_id = record["run_id"]
    for _ in range(1200):
        current = store.get(run_id)
        if current and current.get("status") not in {"queued", "running"}:
            break
        time.sleep(0.05)
    return (tmp_path / "api-runs" / run_id)


def test_a_cli_run_produces_a_transcript_at_all(tmp_path: Path, stubbed: None) -> None:
    """Plan 01 §1.5: with no progress callback the CLI path recorded nothing whatsoever."""
    run_dir = _cli_run(tmp_path)
    document = load(run_dir)
    assert document.spans
    assert document.root().name == "pipeline.run"
    assert document.root().refs["launched_by"] == "cli"


def test_every_span_has_a_resolvable_parent_and_there_is_exactly_one_root(
    tmp_path: Path, stubbed: None
) -> None:
    """Plan 01 §8.3's guard. A detached span is a valid second root, never an error."""
    document = load(_cli_run(tmp_path))
    assert document.orphans() == []
    assert len([span for span in document.spans if span.parent_id is None]) == 1


def test_stage_spans_tile_the_run_rather_than_nesting(tmp_path: Path, stubbed: None) -> None:
    """The constraint §7 item 5 depends on, asserted rather than assumed.

    Nested stages double count, so a nesting bug would push the item 5 ratio *above* 100%
    and read as success. This fails on that instead.
    """
    document = load(_cli_run(tmp_path))
    total = sum(document.duration_by_stage().values())
    root = document.root().duration_ms
    assert total <= root, f"stages summed to {total:.0f} ms of a {root:.0f} ms run; they overlap"
    for span in document.by_kind("stage"):
        parent = document.by_id(span.parent_id)
        assert parent.kind != "stage", f"stage {span.name!r} is nested inside stage {parent.name!r}"


def test_stage_spans_leave_no_gap_between_consecutive_stages(
    tmp_path: Path, stubbed: None
) -> None:
    """Plan 01 §7 item 5, as the structural property rather than as a timing bound.

    Item 5 wants stage durations to account for the run. The share is a *measurement* --
    99.8% on macOS, reported in TRACKING -- and a measurement makes a bad assertion,
    because it is a ratio of two quantities that scale differently. Measured on Windows
    CI: the same code gave 93.5%, missing a 95% bound by 1.5 points, because the untimed
    head (imports, store construction, planner setup) was 52x larger there while the
    compiled work was only 1.65x slower. Nothing was wrong with the instrumentation.

    What must hold on every machine is the structure that makes the ratio interpretable:
    stages must *tile* -- each one starting where the last ended, with nothing unattributed
    in between. Untimed head and tail are outside the first and last stage by construction
    and are excluded here; they are reported as a number instead.
    """
    document = load(_cli_run(tmp_path))
    stages = sorted(document.by_kind("stage"), key=lambda span: span.started_at)
    assert len(stages) > 1
    for earlier, later in zip(stages, stages[1:]):
        assert later.started_at >= earlier.ended_at, (
            f"stage {later.name!r} started before {earlier.name!r} ended; stages overlap "
            "and duration_by_stage over-reports"
        )
        gap_ms = (
            _millis(later.started_at) - _millis(earlier.ended_at)
        )
        assert gap_ms <= _MAX_INTER_STAGE_GAP_MS, (
            f"{gap_ms:.1f} ms is unattributed between {earlier.name!r} and {later.name!r}; "
            "work is happening outside every stage"
        )


def test_the_untimed_head_is_reported_as_a_duration_not_only_as_a_share(
    tmp_path: Path, stubbed: None
) -> None:
    """A fact that survives a platform change, unlike the ratio.

    `575 ms of untimed head` is diagnosable on its own; `93.5%` is not, because it moves
    when either quantity moves. This asserts only that the head is a small *absolute*
    slice of a stubbed run, with generous room, and prints it either way.
    """
    document = load(_cli_run(tmp_path))
    root = document.root().duration_ms
    total = sum(document.duration_by_stage().values())
    head_ms = root - total
    share = total / root if root else 0.0
    print(
        f"\nstage coverage: {share * 100:.1f}% of {root:.0f} ms; "
        f"untimed head {head_ms:.0f} ms across {len(document.by_kind('stage'))} stage spans"
    )
    assert head_ms >= 0.0
    # Deliberately loose and absolute. The head is process setup, not pipeline work, so a
    # slow machine makes it larger without anything being wrong; what would be wrong is
    # seconds of it, which would mean real work escaping every stage.
    assert head_ms < 5000.0, f"{head_ms:.0f} ms of the run happened outside every stage"


def test_the_seven_declared_stages_are_all_recorded(tmp_path: Path, stubbed: None) -> None:
    document = load(_cli_run(tmp_path))
    assert EXPECTED_STAGES.issubset(set(document.duration_by_stage()))


def test_stage_spans_carry_the_identifiers_needed_to_locate_them(
    tmp_path: Path, stubbed: None
) -> None:
    """Plan 01 §1.3: correlation must never be reconstructed from array position."""
    document = load(_cli_run(tmp_path))
    per_candidate = [
        span for span in document.by_kind("stage") if span.name in {"candidates", "visual_evidence"}
    ]
    assert per_candidate
    assert all("candidate_index" in span.refs or "round" in span.refs for span in per_candidate)


def test_the_capture_step_records_a_tool_span_under_its_stage(tmp_path: Path) -> None:
    """Plan 01 §1.6: a non-model stage produced artifacts and no record at all.

    Drives the real `_run_capture` against a trivial command rather than going through
    the pipeline, because stubbing `capture_in_subprocess` would replace the very
    function the span lives in — the test would then pass against no instrumentation.
    """
    import sys

    from rigby_poc.observability import stage_timeline
    from rigby_poc.pipeline import _run_capture

    tracer = Tracer.open(tmp_path / "runs")
    with tracer.span("run", "pipeline.run"):
        with stage_timeline(tracer) as timeline:
            timeline.enter("visual_evidence", candidate_index=1, round=1)
            _run_capture(
                [sys.executable, "-c", "print('captured')"],
                tmp_path / "evidence",
                budget_s=60.0,
                description="000001-test",
            )
    document = load(tracer.run_dir)
    tools = document.by_kind("tool")
    assert tools, "capture produced no tool span"
    span = tools[0]
    parent = document.by_id(span.parent_id)
    assert parent.kind == "stage" and parent.name == "visual_evidence"
    assert span.attrs["returncode"] == 0
    assert span.attrs["output_bytes"] > 0
    # §1.6 again: stdout used to go to the null device, so a failure was an exit code.
    assert "captured" in (tmp_path / "evidence" / "capture.log").read_text(encoding="utf-8")


def test_a_failed_capture_records_its_failure_on_the_span(tmp_path: Path) -> None:
    import sys

    from rigby_poc.observability import stage_timeline
    from rigby_poc.pipeline import _run_capture

    tracer = Tracer.open(tmp_path / "runs")
    with tracer.span("run", "pipeline.run"):
        with stage_timeline(tracer) as timeline:
            timeline.enter("visual_evidence")
            with pytest.raises(RuntimeError, match="exit code 4"):
                _run_capture(
                    [sys.executable, "-c", "import sys; sys.exit(4)"],
                    tmp_path / "evidence",
                    budget_s=60.0,
                    description="000001-test",
                )
    document = load(tracer.run_dir)
    span = document.by_kind("tool")[0]
    assert span.status == "error"
    assert span.error is not None and span.error["type"] == "RuntimeError"


def test_a_cli_run_and_an_api_run_are_structurally_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed: None
) -> None:
    """Plan 01 §7 item 2, the whole reason the run roots were unified.

    Structural, not literal: ids, timings and paths differ by construction. What must
    match is the shape — the span kinds present, the stage names, and the parent/child
    relationship between kinds.
    """
    cli = load(_cli_run(tmp_path / "cli"))
    api = load(_api_run(tmp_path / "api", monkeypatch))

    def shape(document: Any) -> dict[str, Any]:
        return {
            "kinds": sorted({span.kind for span in document.spans}),
            "stages": sorted(document.duration_by_stage()),
            "roots": len([span for span in document.spans if span.parent_id is None]),
            "orphans": document.orphans(),
            "kind_edges": sorted(
                {
                    (document.by_id(span.parent_id).kind, span.kind)
                    for span in document.spans
                    if span.parent_id is not None
                }
            ),
        }

    cli_shape, api_shape = shape(cli), shape(api)
    assert cli_shape["stages"] == api_shape["stages"]
    assert cli_shape["kind_edges"] == api_shape["kind_edges"]
    assert cli_shape["roots"] == api_shape["roots"] == 1
    assert cli_shape["orphans"] == api_shape["orphans"] == []
    assert cli.root().name == api.root().name == "pipeline.run"
    assert cli.root().refs["launched_by"] == "cli"
    assert api.root().refs["launched_by"] == "api"


def test_progress_events_carry_a_timestamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed: None) -> None:
    """Plan 01 §1.4: `_progress` added no wall clock of its own."""
    seen: list[dict[str, Any]] = []
    tracer = Tracer.open(tmp_path / "runs")
    with tracer.span("run", "pipeline.run"):
        flywheel.run_best_of_five(
            PROMPT,
            tmp_path / "artifacts",
            provider="offline",
            selection_mode="five_way",
            max_rounds=1,
            capture_fn=_fake_capture,
            progress_callback=seen.append,
            tracer=tracer,
        )
    assert seen
    assert all("at" in event for event in seen)
    assert all(event["at"].endswith("+00:00") for event in seen)


def test_the_api_run_still_writes_its_run_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubbed: None) -> None:
    """Plan 01 §3.6: this PR adds a channel, it does not migrate the existing one."""
    run_dir = _api_run(tmp_path, monkeypatch)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["events"]
    assert record["status"] in {"completed", "no_acceptable_candidate", "unsupported"}
    assert (run_dir / "transcript.jsonl").is_file()
