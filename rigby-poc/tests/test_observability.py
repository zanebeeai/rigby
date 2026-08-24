from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import io
import json
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from rigby_poc import observability
from rigby_poc.transcript import load
from rigby_poc.observability import (
    JsonlWriter,
    NullTracer,
    StageTimeline,
    Tracer,
    UlidFactory,
    current_span,
    new_run_id,
    new_ulid,
    propagate,
    substitute_images,
    ulid_timestamp_ms,
)


def read_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tracer_for(tmp_path: Path, **kwargs) -> Tracer:
    return Tracer(tmp_path / "run", run_id="20260816T042756-f2107423", fsync=False, **kwargs)


# ---------------------------------------------------------------------------------
# ULID
# ---------------------------------------------------------------------------------


def test_ulid_is_26_crockford_characters_and_carries_its_timestamp() -> None:
    value = new_ulid()

    assert len(value) == 26
    assert set(value) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
    assert not set(value) & set("ILOU")
    assert ulid_timestamp_ms(value) > 1_700_000_000_000


def test_ulids_sort_in_creation_order_inside_one_millisecond() -> None:
    frozen = UlidFactory(clock=lambda: 1_781_000_000.0)

    minted = [frozen.new() for _ in range(500)]

    assert minted == sorted(minted)
    assert len(set(minted)) == len(minted)


def test_ulid_stays_monotonic_when_the_clock_steps_backwards() -> None:
    ticks = iter([1_781_000_005.0, 1_781_000_001.0, 1_781_000_002.0, 1_781_000_000.0])
    factory = UlidFactory(clock=lambda: next(ticks))

    minted = [factory.new() for _ in range(4)]

    assert minted == sorted(minted)


def test_ulid_factory_is_thread_safe() -> None:
    factory = UlidFactory()

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        minted = list(pool.map(lambda _: factory.new(), range(2000)))

    assert len(set(minted)) == 2000


def test_new_run_id_matches_the_shape_pipeline_already_accepts() -> None:
    from rigby_poc.pipeline import SAFE_RUN_ID

    assert SAFE_RUN_ID.fullmatch(new_run_id())


# ---------------------------------------------------------------------------------
# Span nesting and contextvar parent resolution
# ---------------------------------------------------------------------------------


def test_nested_spans_resolve_their_parent_from_the_contextvar(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    with tracer.span("run", "pipeline.run") as root:
        assert current_span() is root
        with tracer.span("stage", "judging") as stage:
            with tracer.span("model_call", "judge.rank_five", round=1) as call:
                call.set(primary_model="gpt-5.6-luna")
        assert current_span() is root
    assert current_span() is None

    records = {record["name"]: record for record in read_lines(tracer.run_dir / "transcript.jsonl")}

    assert records["pipeline.run"]["parent_id"] is None
    assert records["judging"]["parent_id"] == records["pipeline.run"]["span_id"]
    assert records["judge.rank_five"]["parent_id"] == records["judging"]["span_id"]
    assert records["judge.rank_five"]["refs"] == {"round": 1}
    assert records["judge.rank_five"]["attrs"] == {"primary_model": "gpt-5.6-luna"}


def test_spans_are_written_in_completion_order_and_ids_sort_by_creation(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    with tracer.span("run", "pipeline.run"):
        with tracer.span("stage", "planning"):
            pass
        with tracer.span("stage", "judging"):
            pass

    records = read_lines(tracer.run_dir / "transcript.jsonl")

    assert [record["name"] for record in records] == ["planning", "judging", "pipeline.run"]
    by_creation = sorted(records, key=lambda record: record["span_id"])
    assert [record["name"] for record in by_creation] == ["pipeline.run", "planning", "judging"]


def test_a_raising_span_records_the_error_and_re_raises(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    def dispatch() -> None:
        with tracer.span("attempt", "judge.rank_five#0"):
            raise TimeoutError("Request timed out.")

    with pytest.raises(TimeoutError):
        dispatch()

    (record,) = read_lines(tracer.run_dir / "transcript.jsonl")

    assert record["status"] == "error"
    assert record["error"]["type"] == "TimeoutError"
    assert record["error"]["message"] == "Request timed out."
    assert len(record["error"]["traceback_digest"]) == 64
    # The traceback text carries absolute developer paths and is deliberately not stored.
    assert "traceback" not in record["error"]
    assert record["duration_ms"] >= 0


def test_span_duration_and_timestamps_are_populated(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    with tracer.span("stage", "capture"):
        pass

    (record,) = read_lines(tracer.run_dir / "transcript.jsonl")

    assert record["started_at"].endswith("Z")
    assert record["ended_at"].endswith("Z")
    assert len(record["started_at"]) == len("2026-08-16T04:31:02.114Z")
    assert isinstance(record["duration_ms"], (int, float))


def test_refs_drop_none_so_a_missing_identifier_is_never_stored(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    with tracer.span("candidate", "candidate-4", result_id="000005-jab", round=None):
        pass

    (record,) = read_lines(tracer.run_dir / "transcript.jsonl")

    assert record["refs"] == {"result_id": "000005-jab"}


# ---------------------------------------------------------------------------------
# Thread boundary (plan section 8.3)
# ---------------------------------------------------------------------------------


def test_a_plain_thread_loses_the_parent_context(tmp_path: Path) -> None:
    """The hazard `PipelineRunStore.start` walks into at `pipeline.py:141`.

    `_execute` is handed to a `threading.Thread`, and a `contextvars.ContextVar` set on
    the parent is not visible there. The child span silently becomes a second root rather
    than erroring, which is why this behaviour is pinned rather than left implicit.
    """

    tracer = tracer_for(tmp_path)
    seen: dict[str, object] = {}

    def worker() -> None:
        seen["parent_before"] = current_span()
        with tracer.span("stage", "detached"):
            pass

    with tracer.span("run", "pipeline.run"):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    records = {record["name"]: record for record in read_lines(tracer.run_dir / "transcript.jsonl")}

    assert seen["parent_before"] is None
    assert records["detached"]["parent_id"] is None
    assert records["pipeline.run"]["parent_id"] is None


def test_propagate_carries_the_parent_context_across_a_thread(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    def worker() -> None:
        with tracer.span("stage", "attached"):
            pass

    with tracer.span("run", "pipeline.run"):
        thread = threading.Thread(target=propagate(worker))
        thread.start()
        thread.join()

    records = {record["name"]: record for record in read_lines(tracer.run_dir / "transcript.jsonl")}

    assert records["attached"]["parent_id"] == records["pipeline.run"]["span_id"]


def test_an_explicit_parent_id_crosses_a_thread_without_context_copying(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    def worker(parent_id: str) -> None:
        with tracer.span("stage", "attached", parent=parent_id):
            pass

    with tracer.span("run", "pipeline.run") as root:
        thread = threading.Thread(target=worker, args=(root.span_id,))
        thread.start()
        thread.join()
        root_id = root.span_id

    records = {record["name"]: record for record in read_lines(tracer.run_dir / "transcript.jsonl")}

    assert records["attached"]["parent_id"] == root_id


def test_propagate_also_works_through_a_thread_pool(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    def worker() -> str | None:
        with tracer.span("candidate", "candidate-0") as span:
            return span.parent_id

    with tracer.span("run", "pipeline.run") as root:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            observed = pool.submit(propagate(worker)).result()
        assert observed == root.span_id


def test_sibling_threads_do_not_see_each_others_spans(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)
    parents: dict[str, str | None] = {}
    barrier = threading.Barrier(2)

    def worker(label: str) -> None:
        with tracer.span("candidate", label) as outer:
            barrier.wait(timeout=5)
            with tracer.span("model_call", f"{label}.judge") as inner:
                parents[label] = inner.parent_id
                assert inner.parent_id == outer.span_id

    threads = [threading.Thread(target=propagate(worker), args=(name,)) for name in ("a", "b")]
    with tracer.span("run", "pipeline.run"):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert parents["a"] != parents["b"]


# ---------------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------------


def test_jsonl_writer_appends_one_line_per_record(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path / "nested" / "transcript.jsonl", fsync=False)

    writer.append({"span_id": "a", "text": "line one"})
    writer.append({"span_id": "b", "text": "line two"})

    raw = (tmp_path / "nested" / "transcript.jsonl").read_bytes()

    assert raw.count(b"\n") == 2
    assert [json.loads(line)["span_id"] for line in raw.splitlines()] == ["a", "b"]


def test_jsonl_writer_never_emits_a_carriage_return(tmp_path: Path) -> None:
    """Windows text mode rewrites "\\n" as "\\r\\n" unless newline is pinned.

    This repo is developed on macOS and Windows both. Without `newline="\\n"` the same
    transcript would differ byte for byte between the two, which breaks any hashing or
    byte-offset reader and leaves a stray "\\r" on the final field of every record.
    """

    path = tmp_path / "transcript.jsonl"
    writer = JsonlWriter(path, fsync=False)

    writer.append({"name": "planning", "note": "no trailing carriage return"})

    assert b"\r" not in path.read_bytes()
    assert path.read_bytes().endswith(b"}\n")


def test_jsonl_writer_records_are_single_line_even_with_nested_payloads(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    writer = JsonlWriter(path, fsync=False)

    writer.append({"attrs": {"usage": {"input_tokens": 10}}, "refs": {"round": 1}})

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_writer_retries_a_transient_windows_lock(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "transcript.jsonl"
    slept: list[float] = []
    writer = JsonlWriter(path, fsync=False, sleep=slept.append)
    real_open = Path.open
    calls = 0

    def intermittently_locked(self, *args, **kwargs):
        nonlocal calls
        if self == path:
            calls += 1
            if calls < 3:
                raise PermissionError(5, "Access is denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", intermittently_locked)
    writer.append({"span_id": "a"})

    assert calls == 3
    assert len(slept) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"span_id": "a"}


def test_jsonl_writer_gives_up_after_the_ladder_and_raises(tmp_path: Path, monkeypatch) -> None:
    writer = JsonlWriter(tmp_path / "transcript.jsonl", replace_attempts=3, fsync=False, sleep=lambda _: None)

    def always_locked(self, *args, **kwargs):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "open", always_locked)

    with pytest.raises(PermissionError):
        writer.append({"span_id": "a"})


def test_concurrent_appends_do_not_interleave(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    writer = JsonlWriter(path, fsync=False)

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda index: writer.append({"index": index, "pad": "x" * 400}), range(400)))

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 400
    assert sorted(json.loads(line)["index"] for line in lines) == list(range(400))


# ---------------------------------------------------------------------------------
# Payload capture and $image substitution
# ---------------------------------------------------------------------------------


def png_bytes(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 9), colour).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def data_uri(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def stored_payload(tracer: Tracer, relative: str | None) -> dict:
    assert relative is not None
    return json.loads((tracer.run_dir / relative).read_text(encoding="utf-8"))


def test_record_request_replaces_base64_images_with_hashed_references(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)
    raw = png_bytes((32, 96, 200))
    payload = [
        {"role": "user", "content": [
            {"type": "input_text", "text": "candidate 4 of 5"},
            {"type": "input_image", "image_url": data_uri(raw), "detail": "high"},
        ]},
    ]

    with tracer.span("model_call", "judge.rank_five") as call:
        call.record_request(payload)
        stored = call.request_path

    document = stored_payload(tracer, stored)
    reference = document["value"][0]["content"][1]["$image"]

    # The hash is of the bytes as sent, which is exactly what section 1.1 says is missing:
    # the manifest hashes the source PNG, but the judge resizes before dispatch.
    assert reference["sha256"] == hashlib.sha256(raw).hexdigest()
    assert reference["bytes"] == len(raw)
    assert reference["media_type"] == "image/png"
    assert reference["detail"] == "high"
    assert "base64" not in json.dumps(document)


def test_record_request_stores_payload_image_bytes_content_addressed(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)
    raw = png_bytes((200, 96, 32))
    digest = hashlib.sha256(raw).hexdigest()
    node = {"type": "input_image", "image_url": data_uri(raw), "detail": "high"}

    with tracer.span("model_call", "judge.rank_five") as first:
        first.record_request([node])
    with tracer.span("model_call", "judge.rank_five") as second:
        second.record_request([node])

    images = sorted((tracer.run_dir / "payloads" / "images").iterdir())

    # The same evidence frame goes to several judge calls; content addressing stores it
    # once, so replayability does not cost a copy per dispatch.
    assert [path.name for path in images] == [f"{digest}.png"]
    assert images[0].read_bytes() == raw


def test_image_metadata_is_merged_by_hash_not_by_position(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)
    ego, orbit = png_bytes((1, 2, 3)), png_bytes((4, 5, 6))
    metadata = {
        hashlib.sha256(orbit).hexdigest(): {"source_snapshot_id": "08-impact_pose-orbit"},
        hashlib.sha256(ego).hexdigest(): {"source_snapshot_id": "08-impact_pose-ego"},
    }
    payload = [
        {"type": "input_image", "image_url": data_uri(ego)},
        {"type": "input_image", "image_url": data_uri(orbit)},
    ]

    with tracer.span("model_call", "judge.rank_five") as call:
        call.record_request(payload, image_metadata=metadata)
        stored = call.request_path

    document = stored_payload(tracer, stored)["value"]

    assert document[0]["$image"]["source_snapshot_id"] == "08-impact_pose-ego"
    assert document[1]["$image"]["source_snapshot_id"] == "08-impact_pose-orbit"


def test_substitute_images_leaves_non_image_strings_alone() -> None:
    payload = {"text": "data is not a uri", "url": "https://example.invalid/a.png",
               "nested": [{"type": "input_text", "text": "data:text/plain,hello"}]}

    assert substitute_images(payload) == payload


def test_record_response_stores_the_whole_model_dump(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    class StubResponse:
        def model_dump(self, mode: str = "python") -> dict:
            return {"id": "resp_1", "status": "incomplete", "output_parsed": {"score": 1},
                    "incomplete_details": {"reason": "max_output_tokens"}}

    with tracer.span("attempt", "judge.rank_five#0") as attempt:
        attempt.record_response(StubResponse())
        stored = attempt.response_path

    document = stored_payload(tracer, stored)

    # Section 1.1: keeping only `output_parsed` throws away status and incomplete_details.
    assert document["incomplete_details"] == {"reason": "max_output_tokens"}
    assert document["status"] == "incomplete"


def test_payload_paths_are_relative_to_the_run_directory(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path)

    with tracer.span("model_call", "judge.rank_five") as call:
        call.record_request({"role": "user"})
        call.record_response({"id": "resp_1"})

    (record,) = read_lines(tracer.run_dir / "transcript.jsonl")

    assert record["request_path"] == f"payloads/{record['span_id']}.request.json"
    assert record["response_path"] == f"payloads/{record['span_id']}.response.json"
    assert not Path(record["request_path"]).is_absolute()


def test_store_payload_images_can_be_switched_off(tmp_path: Path) -> None:
    tracer = tracer_for(tmp_path, store_payload_images=False)
    raw = png_bytes((7, 8, 9))

    with tracer.span("model_call", "judge.rank_five") as call:
        call.record_request([{"type": "input_image", "image_url": data_uri(raw)}])
        stored = call.request_path

    document = stored_payload(tracer, stored)["value"]

    assert "path" not in document[0]["$image"]
    assert not (tracer.run_dir / "payloads" / "images").exists()


# ---------------------------------------------------------------------------------
# NullTracer
# ---------------------------------------------------------------------------------


def test_null_tracer_writes_nothing_and_supports_the_same_calls(tmp_path: Path) -> None:
    tracer = NullTracer()

    with tracer.span("run", "pipeline.run", round=1) as root:
        root.set(model="gpt-5.6-luna")
        root.record_request({"role": "user"})
        root.record_response({"id": "resp_1"})
        with tracer.span("stage", "judging") as stage:
            stage.skip("no candidates")

    assert tracer.current() is None
    assert not list(tmp_path.iterdir())


def test_null_tracer_propagates_an_exception_without_swallowing_it() -> None:
    with pytest.raises(ValueError):
        with NullTracer().span("attempt", "judge.rank_five#0"):
            raise ValueError("boom")


def test_the_default_tracer_is_a_null_tracer() -> None:
    assert isinstance(observability.get_tracer(), NullTracer)


def test_tracer_open_mints_a_run_directory_named_for_the_run(tmp_path: Path) -> None:
    tracer = Tracer.open(tmp_path, fsync=False)

    with tracer.span("run", "pipeline.run"):
        pass

    assert tracer.run_dir == tmp_path / tracer.run_id
    assert (tracer.run_dir / "transcript.jsonl").is_file()


# --- plan 01 section 7, item 5: stage durations must tile, never nest ---------------


def test_stage_timeline_closes_the_previous_stage_before_opening_the_next(
    tmp_path: Path,
) -> None:
    """The guard against double counting. Nested stages report >100% of wall clock."""
    tracer = Tracer(tmp_path / "run", run_id="20260824T120000-deadbeef")
    with tracer.span("run", "pipeline"):
        timeline = StageTimeline(tracer)
        timeline.enter("planning")
        time.sleep(0.02)
        timeline.enter("candidates")
        time.sleep(0.02)
        timeline.close()
    document = load(tmp_path / "run")
    stages = document.by_kind("stage")
    assert [span.name for span in stages] == ["planning", "candidates"]
    # Tiling, not nesting: neither stage is an ancestor of the other.
    assert {span.parent_id for span in stages} == {document.root().span_id}
    total = sum(document.duration_by_stage().values())
    assert total <= document.root().duration_ms, (
        f"stages summed to {total:.1f} ms of a {document.root().duration_ms:.1f} ms run; "
        "they overlap, so duration_by_stage over-reports"
    )


def test_re_entering_a_stage_with_new_refs_starts_a_separately_attributable_span(
    tmp_path: Path,
) -> None:
    """Per-candidate attribution, while `duration_by_stage` still sums under one name."""
    tracer = Tracer(tmp_path / "run", run_id="20260824T120000-deadbeef")
    with tracer.span("run", "pipeline"):
        timeline = StageTimeline(tracer)
        for index in (1, 2, 3):
            timeline.enter("candidates", candidate_index=index, round=1)
        timeline.close()
    document = load(tmp_path / "run")
    stages = document.by_kind("stage")
    assert len(stages) == 3
    assert [span.refs["candidate_index"] for span in stages] == [1, 2, 3]
    assert set(document.duration_by_stage()) == {"candidates"}


def test_re_entering_the_same_stage_with_the_same_refs_does_not_churn_spans(
    tmp_path: Path,
) -> None:
    """`_progress` fires several times per stage; each must not become its own span."""
    tracer = Tracer(tmp_path / "run", run_id="20260824T120000-deadbeef")
    with tracer.span("run", "pipeline"):
        timeline = StageTimeline(tracer)
        timeline.enter("vlm_judge", round=1)
        timeline.enter("vlm_judge", round=1)
        timeline.enter("vlm_judge", round=1)
        timeline.close()
    assert len(load(tmp_path / "run").by_kind("stage")) == 1


def test_a_span_opened_during_a_stage_parents_to_that_stage(tmp_path: Path) -> None:
    """A `tool` or `model_call` span must land under the stage that was running."""
    tracer = Tracer(tmp_path / "run", run_id="20260824T120000-deadbeef")
    with tracer.span("run", "pipeline"):
        timeline = StageTimeline(tracer)
        timeline.enter("visual_evidence", candidate_index=4)
        with tracer.span("tool", "capture"):
            pass
        timeline.close()
    document = load(tmp_path / "run")
    stage = document.by_kind("stage")[0]
    tool = document.by_kind("tool")[0]
    assert tool.parent_id == stage.span_id
    # And the tool span is not itself a stage, so it cannot inflate duration_by_stage.
    assert set(document.duration_by_stage()) == {"visual_evidence"}


def test_the_timeline_leaves_no_orphans_and_exactly_one_root(tmp_path: Path) -> None:
    """Plan 01 section 8.3's guard: a detached span is a valid second root, not an error."""
    tracer = Tracer(tmp_path / "run", run_id="20260824T120000-deadbeef")
    with tracer.span("run", "pipeline"):
        with StageTimeline(tracer) as timeline:
            timeline.enter("planning")
            timeline.enter("finalize")
    document = load(tmp_path / "run")
    assert document.orphans() == []
    assert len([span for span in document.spans if span.parent_id is None]) == 1
