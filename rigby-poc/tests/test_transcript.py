from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from rigby_poc import transcript as transcript_module
from rigby_poc.observability import Tracer
from rigby_poc.transcript import TokenUsage, TranscriptError, load


FIXTURE_RUN_ID = "20260816T042756-f2107423"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "transcript" / FIXTURE_RUN_ID
EGO_SHA = "3b1eb92bb89416b054be048b1eccc08c47b244689092f889100e82e41f36ac05"
ORBIT_SHA = "4f4fd82add85d1c7ff1e66645e95c1cf2782662dd50697ad56400e417cd95c66"


@pytest.fixture()
def transcript():
    return load(FIXTURE)


@pytest.fixture()
def scratch_run(tmp_path: Path) -> Path:
    """A writable copy of the committed fixture, for tests that mutate the file."""

    target = tmp_path / FIXTURE_RUN_ID
    shutil.copytree(FIXTURE, target)
    return target


# ---------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------


def test_the_committed_fixture_loads_with_nothing_skipped(transcript) -> None:
    assert len(transcript) == 8
    assert transcript.skipped_lines == []
    assert {span.kind for span in transcript.spans} == {
        "run", "stage", "model_call", "attempt"
    }


def test_spans_are_returned_in_ulid_order(transcript) -> None:
    identifiers = [span.span_id for span in transcript.spans]

    assert identifiers == sorted(identifiers)
    assert [span.name for span in transcript.spans][0] == "pipeline.run"


def test_the_tree_has_one_root_and_no_orphans(transcript) -> None:
    root = transcript.root()

    assert root.kind == "run"
    assert root.parent_id is None
    assert transcript.roots() == [root]
    assert transcript.orphans() == []


def test_every_span_reaches_the_root_through_its_parents(transcript) -> None:
    root = transcript.root()

    for span in transcript.spans:
        chain = transcript.ancestors(span.span_id)
        assert (span is root) or chain[-1].span_id == root.span_id


def test_children_and_descendants(transcript) -> None:
    root = transcript.root()

    assert [span.name for span in transcript.children(root.span_id)] == ["planning", "judging"]
    assert len(transcript.descendants(root.span_id)) == 7

    (rank_call,) = [span for span in transcript.spans if span.name == "judge.rank_five"]
    assert [span.name for span in transcript.children(rank_call.span_id)] == [
        "judge.rank_five#0",
        "judge.rank_five#1",
    ]


def test_by_kind_and_model_calls(transcript) -> None:
    assert [span.name for span in transcript.model_calls()] == ["planner.plan", "judge.rank_five"]
    assert [span.name for span in transcript.by_kind("stage")] == ["planning", "judging"]
    assert transcript.by_kind("nonexistent") == []


def test_by_id_raises_for_an_unknown_span(transcript) -> None:
    with pytest.raises(TranscriptError):
        transcript.by_id("01M00000000000000000000000")


def test_refs_carry_the_identifiers_that_positional_correlation_lost(transcript) -> None:
    """Section 1.3: `rank_five` stores neither result ids nor manifest paths today."""

    (rank_call,) = [span for span in transcript.spans if span.name == "judge.rank_five"]

    assert rank_call.refs["result_ids"] == ["000005-throw-a-right-jab"]
    assert rank_call.refs["round"] == 1
    assert len(rank_call.refs["evidence_manifest_sha256"]) == 64


# ---------------------------------------------------------------------------------
# Failed attempts (plan section 1.2 / 3.4)
# ---------------------------------------------------------------------------------


def test_the_failed_primary_attempt_survives_as_a_span(transcript) -> None:
    """The direct guard against the section 1.2 regression.

    `judge.py:831-836` sets `response = None; parsed = None` and keeps only the string
    `primary_error:<ClassName>`, so a failed primary leaves no response id, no usage, and
    no message. In the transcript it is an ordinary span with all three.
    """

    (failure,) = transcript.failures()

    assert failure.kind == "attempt"
    assert failure.name == "judge.rank_five#0"
    assert failure.error is not None
    assert failure.error["type"] == "APITimeoutError"
    assert failure.error["message"] == "Request timed out."
    assert failure.attrs["model"] == "gpt-5.6-luna"
    assert failure.usage.total_tokens == 9520


def test_total_tokens_include_the_failed_attempt(transcript) -> None:
    """Section 8.5: totals go up once failures are counted. That is the correction."""

    total = transcript.total_tokens()
    (failure,) = transcript.failures()

    assert total == TokenUsage(
        input_tokens=1820 + 9400 + 9400,
        output_tokens=640 + 120 + 880,
        reasoning_tokens=512 + 96 + 704,
        total_tokens=1820 + 640 + 9400 + 120 + 9400 + 880,
    )
    assert total.total_tokens - failure.usage.total_tokens == 12740


def test_usage_is_not_double_counted_when_a_parent_also_reports_it(scratch_run: Path) -> None:
    """A `model_call` aggregate plus its `attempt` children must not sum twice."""

    path = scratch_run / "transcript.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["name"] == "judge.rank_five":
            record["attrs"]["usage"] = {"input_tokens": 18800, "output_tokens": 1000,
                                        "total_tokens": 19800}
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8", newline="\n"
    )

    assert load(scratch_run).total_tokens() == load(FIXTURE).total_tokens()


def test_usage_is_counted_when_only_the_model_call_reports_it(scratch_run: Path) -> None:
    """The other instrumentation shape: no attempt spans at all, usage on the call."""

    path = scratch_run / "transcript.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    kept = [record for record in records if record["kind"] != "attempt"]
    for record in kept:
        if record["name"] == "judge.rank_five":
            record["attrs"]["usage"] = {"input_tokens": 100, "output_tokens": 7,
                                        "total_tokens": 107}
    path.write_text(
        "\n".join(json.dumps(record) for record in kept) + "\n", encoding="utf-8", newline="\n"
    )

    assert load(scratch_run).total_tokens() == TokenUsage(100, 7, 0, 107)


def test_token_usage_tolerates_a_missing_or_partial_usage_block() -> None:
    assert TokenUsage.from_usage(None) == TokenUsage()
    assert TokenUsage.from_usage({}) == TokenUsage()
    # total_tokens is derived when the provider omits it.
    assert TokenUsage.from_usage({"input_tokens": 3, "output_tokens": 4}).total_tokens == 7


# ---------------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------------


def test_duration_by_stage_sums_milliseconds_per_stage_name(transcript) -> None:
    assert transcript.duration_by_stage() == {"planning": 9800.0, "judging": 28900.0}


def test_stage_durations_account_for_almost_all_of_the_run(transcript) -> None:
    """Definition-of-done item 5: stages must cover >=95% of measured wall clock."""

    root_ms = transcript.root().duration_ms
    covered = sum(transcript.duration_by_stage().values())

    assert root_ms is not None
    assert covered / root_ms >= 0.95


def test_repeated_stages_of_the_same_name_are_summed(scratch_run: Path) -> None:
    path = scratch_run / "transcript.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    extra = json.loads(next(line for line in lines if '"name":"judging"' in line))
    extra["span_id"] = extra["span_id"][:-1] + "Z"
    extra["duration_ms"] = 100.0
    path.write_text(
        "\n".join([*lines, json.dumps(extra)]) + "\n", encoding="utf-8", newline="\n"
    )

    assert load(scratch_run).duration_by_stage()["judging"] == 29000.0


# ---------------------------------------------------------------------------------
# Payloads and $image rehydration
# ---------------------------------------------------------------------------------


def test_request_rehydrates_image_references_into_data_uris(transcript) -> None:
    (rank_call,) = [span for span in transcript.spans if span.name == "judge.rank_five"]

    payload = transcript.request(rank_call.span_id)
    content = payload[1]["content"]

    assert payload[0]["role"] == "system"
    assert content[0] == {"type": "input_text", "text": "candidate 4 of 5"}
    for node, digest in ((content[1], EGO_SHA), (content[2], ORBIT_SHA)):
        assert node["type"] == "input_image"
        assert node["detail"] == "high"
        assert "$image" not in node
        header, _, encoded = node["image_url"].partition(",")
        assert header == "data:image/png;base64"
        assert hashlib.sha256(base64.b64decode(encoded)).hexdigest() == digest


def test_a_rehydrated_request_round_trips_through_the_writer(tmp_path: Path, transcript) -> None:
    """The replay contract: what the reader hands back re-records to the same references."""

    (rank_call,) = [span for span in transcript.spans if span.name == "judge.rank_five"]
    rehydrated = transcript.request(rank_call.span_id)
    tracer = Tracer(tmp_path / "replay", run_id=FIXTURE_RUN_ID, fsync=False)

    with tracer.span("model_call", "judge.rank_five") as call:
        call.record_request(rehydrated)
        stored = call.request_path

    assert stored is not None
    written = json.loads((tracer.run_dir / stored).read_text(encoding="utf-8"))["value"]
    original = transcript.raw_request(rank_call.span_id)

    for rewritten, source in zip(written[1]["content"][1:], original[1]["content"][1:], strict=True):
        assert rewritten["$image"]["sha256"] == source["$image"]["sha256"]
        assert rewritten["$image"]["bytes"] == source["$image"]["bytes"]
        assert rewritten["$image"]["detail"] == source["$image"]["detail"]


def test_raw_request_leaves_the_references_in_place(transcript) -> None:
    (rank_call,) = [span for span in transcript.spans if span.name == "judge.rank_five"]

    node = transcript.raw_request(rank_call.span_id)[1]["content"][1]

    assert node["$image"]["sha256"] == EGO_SHA
    assert node["$image"]["source_snapshot_id"] == "08-impact_pose-ego"
    assert node["$image"]["payload_dimensions_px"] == [16, 9]
    assert "image_url" not in node


def test_response_returns_the_whole_stored_object(transcript) -> None:
    (rank_call,) = [span for span in transcript.spans if span.name == "judge.rank_five"]

    response = transcript.response(rank_call.span_id)

    assert response["id"] == "resp_fixture_rank_five"
    assert response["status"] == "completed"
    assert response["output_parsed"]["order"] == [4, 1, 0, 2, 3]


def test_a_span_without_a_stored_request_raises(transcript) -> None:
    (planning,) = [span for span in transcript.spans if span.name == "planning"]

    with pytest.raises(TranscriptError, match="no stored request"):
        transcript.request(planning.span_id)


def test_rehydration_raises_when_the_image_bytes_are_gone(scratch_run: Path) -> None:
    """Better a loud failure than a payload that only looks replayable."""

    (scratch_run / "payloads" / "images" / f"{EGO_SHA}.png").unlink()
    loaded = load(scratch_run)
    (rank_call,) = [span for span in loaded.spans if span.name == "judge.rank_five"]

    with pytest.raises(TranscriptError, match="not stored in this run"):
        loaded.request(rank_call.span_id)


def test_rehydration_raises_when_the_image_bytes_were_tampered_with(scratch_run: Path) -> None:
    (scratch_run / "payloads" / "images" / f"{EGO_SHA}.png").write_bytes(b"not the png")
    loaded = load(scratch_run)
    (rank_call,) = [span for span in loaded.spans if span.name == "judge.rank_five"]

    with pytest.raises(TranscriptError, match="do not match the recorded sha256"):
        loaded.request(rank_call.span_id)


def test_a_payload_path_escaping_the_run_directory_is_refused(scratch_run: Path) -> None:
    path = scratch_run / "transcript.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["name"] == "judge.rank_five":
            record["request_path"] = "../../../etc/hosts"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8", newline="\n"
    )
    loaded = load(scratch_run)
    (rank_call,) = [span for span in loaded.spans if span.name == "judge.rank_five"]

    with pytest.raises(TranscriptError, match="escapes the run directory"):
        loaded.raw_request(rank_call.span_id)


# ---------------------------------------------------------------------------------
# Damage tolerance
# ---------------------------------------------------------------------------------


def test_a_truncated_final_line_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    """A crashed run is exactly when the transcript matters most.

    The file is append-only and never rewritten, so a process killed mid-append leaves a
    partial final line. Treating that as a parse error would make every crashed run
    unreadable.
    """

    tracer = Tracer(tmp_path / "run", run_id=FIXTURE_RUN_ID, fsync=False)
    for name in ("planning", "capture", "judging"):
        with tracer.span("stage", name):
            pass
    path = tracer.run_dir / "transcript.jsonl"
    intact = path.read_bytes()
    # Cut the last record in half, the way a SIGKILL between write and flush would.
    last_newline = intact.rindex(b"\n", 0, len(intact) - 1)
    truncated = intact[: last_newline + 1 + (len(intact) - last_newline) // 2]
    path.write_bytes(truncated)

    loaded = load(tracer.run_dir)

    assert [span.name for span in loaded.spans] == ["planning", "capture"]
    assert len(loaded.skipped_lines) == 1
    assert loaded.skipped_lines[0].line_number == 3
    assert "invalid json" in loaded.skipped_lines[0].reason


def test_a_damaged_line_is_reported_not_silently_dropped(scratch_run: Path) -> None:
    path = scratch_run / "transcript.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(2, '{"span_id": "no closing brace"')
    lines.insert(4, "[1, 2, 3]")
    lines.insert(6, '{"kind": "stage", "name": "nameless"}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    loaded = load(scratch_run)

    assert len(loaded.spans) == 8
    assert [item.line_number for item in loaded.skipped_lines] == [3, 5, 7]
    assert [item.reason for item in loaded.skipped_lines] == [
        "invalid json: Expecting ',' delimiter",
        "line is not an object",
        "span record has no span_id",
    ]


def test_blank_lines_are_ignored_without_being_reported(scratch_run: Path) -> None:
    path = scratch_run / "transcript.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n\n   \n", encoding="utf-8", newline="\n"
    )

    loaded = load(scratch_run)

    assert len(loaded.spans) == 8
    assert loaded.skipped_lines == []


def test_a_transcript_written_with_windows_line_endings_reads_identically(
    scratch_run: Path,
) -> None:
    """The repo is developed on macOS and Windows; a stray CR must not reach any field."""

    path = scratch_run / "transcript.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    loaded = load(scratch_run)
    reference = load(FIXTURE)

    assert loaded.skipped_lines == []
    assert [span.span_id for span in loaded.spans] == [span.span_id for span in reference.spans]
    assert loaded.root().name == "pipeline.run"
    assert not any("\r" in (span.ended_at or "") for span in loaded.spans)


def test_a_missing_transcript_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(TranscriptError, match="no transcript at"):
        load(tmp_path)


def test_an_empty_transcript_has_no_root(tmp_path: Path) -> None:
    (tmp_path / "transcript.jsonl").write_text("", encoding="utf-8")

    loaded = load(tmp_path)

    assert loaded.spans == []
    with pytest.raises(TranscriptError, match="no root span"):
        loaded.root()


def test_a_detached_subtree_is_reported_as_orphaned(scratch_run: Path) -> None:
    """Section 8.3's failure mode is detectable, not silent."""

    path = scratch_run / "transcript.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    kept = [record for record in records if record["name"] != "judging"]
    path.write_text(
        "\n".join(json.dumps(record) for record in kept) + "\n", encoding="utf-8", newline="\n"
    )

    loaded = load(scratch_run)

    assert [span.name for span in loaded.orphans()] == ["judge.rank_five"]


def test_ancestors_survive_a_parent_cycle(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"span_id": first, "parent_id": second, "kind": "stage", "name": first})
            for first, second in (("A", "B"), ("B", "A"))
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    loaded = load(tmp_path)

    assert [span.span_id for span in loaded.ancestors("A")] == ["B"]


# ---------------------------------------------------------------------------------
# Writer and reader agree end to end
# ---------------------------------------------------------------------------------


def test_the_writer_and_the_reader_agree_on_a_full_trace(tmp_path: Path) -> None:
    """Nothing emits a transcript yet, so this is what exercises the format end to end."""

    tracer = Tracer(tmp_path / "run", run_id=FIXTURE_RUN_ID, fsync=False)
    raw = (FIXTURE / "payloads" / "images" / f"{EGO_SHA}.png").read_bytes()
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

    with tracer.span("run", "pipeline.run") as root:
        root.set(prompt="throw a right jab")
        with tracer.span("stage", "judging"):
            with tracer.span("model_call", "judge.rank_five", round=1) as call:
                call.record_request([{"type": "input_image", "image_url": uri, "detail": "high"}])
                call.set(primary_model="gpt-5.6-luna")
                try:
                    with tracer.span("attempt", "judge.rank_five#0") as attempt:
                        attempt.set(attempt_index=0, usage={"input_tokens": 9400,
                                                            "output_tokens": 120,
                                                            "total_tokens": 9520})
                        raise TimeoutError("Request timed out.")
                except TimeoutError:
                    pass
                with tracer.span("attempt", "judge.rank_five#1") as retry:
                    retry.set(attempt_index=1, usage={"input_tokens": 9400,
                                                      "output_tokens": 880,
                                                      "total_tokens": 10280})
                    retry.record_response({"id": "resp_2", "status": "completed"})

    loaded = load(tracer.run_dir)
    (rank_call,) = loaded.model_calls()

    assert len(loaded) == 5
    assert loaded.skipped_lines == []
    assert loaded.orphans() == []
    assert loaded.root().name == "pipeline.run"
    assert rank_call.refs == {"round": 1}
    assert [span.status for span in loaded.by_kind("attempt")] == ["error", "ok"]
    assert loaded.total_tokens().total_tokens == 19800
    assert loaded.request(rank_call.span_id)[0]["image_url"] == uri
    assert loaded.duration_by_stage().keys() == {"judging"}


def test_module_exports_the_documented_reader_surface() -> None:
    for name in ("load", "Transcript", "TokenUsage", "TranscriptError"):
        assert hasattr(transcript_module, name)
    for name in ("root", "children", "by_kind", "model_calls", "total_tokens",
                 "duration_by_stage", "request"):
        assert callable(getattr(transcript_module.Transcript, name))
