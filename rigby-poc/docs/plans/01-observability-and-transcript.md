# PR 01 — Observability and the full run transcript

Status: proposed, not started.
Scope: give every run a complete, replayable transcript — every model call with its
full request and response, every non-model stage, every failed attempt, correlated by
a real trace hierarchy and timed.

Depends on: nothing. This lands first.
Blocks: [10 — eval redesign](10-eval-redesign.md) (all agent-trajectory evals),
and the replay-based grader work in [07 — judge harness](07-judge-harness.md).

Related: [00-overview.md](00-overview.md).

---

## 1. Problem

The pipeline writes a good *outcome* record and almost no *decision* record. You can
see what was chosen; you cannot see what was asked, what came back, what failed, or
how long anything took.

### 1.1 Request payloads are never persisted

`_call_record` (`src/rigby_poc/judge.py:658-664`) stores exactly four fields:
`response_id`, `model`, `usage`, `parsed`. The `input` argument to
`client.responses.parse` (`judge.py:735`) is discarded at every call site.

That means none of the following survives a run:

- the system prompt — module constants at `judge.py:156, 273, 345, 383`, referenced but
  never recorded or hashed, so **prompt drift across git revisions is invisible in the
  artifacts**;
- the user text blocks (motion diagnostics JSON, snapshot captions, timeline captions —
  `judge.py:635, 546-548, 1004-1010`);
- the image bytes actually sent. Base64 data URIs are built at `judge.py:574-581, 630-643`
  and thrown away. Source PNGs survive on disk with per-image sha256 in the manifest
  (`judge.py:502`), but the **resized and composited bytes the model actually saw are
  neither stored nor hashed**;
- the response schema (`text_format`), implied only by the record's `kind` string.

The planner is worse: the entire prompt string (`planner.py:4163+`, roughly 100 lines of
catalog text) is never persisted at any layer, and the planner writes nothing to disk at
all.

### 1.2 Failed attempts are erased

`_routed_parse` (`judge.py:807-877`) keeps an `attempts[]` list, but only of
*successful, parsed* attempts (`judge.py:829, 858`). When the primary call raises — a
dispatch error, `output_parsed is None`, or a validator throwing — `judge.py:831-836`
sets `response = None; parsed = None` and records a single string,
`primary_error:<ClassName>`.

No response id. No usage. No partial output. No exception message. **Tokens burned on a
failed primary are invisible**, so token accounting is wrong precisely when escalation
fires. Transient retries (`judge.py:751-773`) collapse to one integer,
`transient_retry_count` (`judge.py:871-873`).

The planner drops its failure reason the same way: `planner.py:4138-4159` catches the
primary exception, interpolates type and message into the repair prompt
(`planner.py:4143`), and discards it. The trace shows `model_calls: 2` and never says
whether the repair was triggered by a schema violation, a semantic validation failure,
or a network timeout.

### 1.3 There is no trace hierarchy

There is no trace id, span id, parent id, or per-call UUID anywhere in the repository.
Correlation today is positional and path-based:

| Level | Identifier | Source |
| --- | --- | --- |
| Run | `run_id` = `YYYYmmddTHHMMSS-<8 hex>` | `pipeline.py:116` — **API-launched runs only** |
| Event | monotonic `sequence` + ISO `at` | `pipeline.py:96-102` |
| Round | `round` int and `round-N/` directory | `flywheel.py:1984, 2031` |
| Candidate | `candidate_index`, `recipe.name`, `result_id`, `motion_sha256` | `flywheel.py:2032-2038` |
| Model call | `response_id` | `_call_record` |

Nothing links a judge `response_id` to a `run_id`. Nothing links a planner
`response_id` to a candidate. Two records are structurally uncorrelatable on their own:

- `rank_five` (`judge.py:1218-1229`) stores **neither result ids nor manifest paths**.
  Its `order` and `mapped_scores` are keyed by list position, and the position→candidate
  map exists only in the caller's implicit ordering (`flywheel.py:2391-2396, 2420-2428`).
  The five-way ranking file — the record of the single most important decision in the
  run — cannot be interpreted without replaying the caller.
- `recommend_repair` (`judge.py:1153-1157`) carries **no identifier at all**: no round,
  no result id, no prompt, and no `routing` block, so its reasoning effort is unrecorded
  and it has no retry or fallback path.

### 1.4 There is no timing

The only wall-clock source in the system is `PipelineRunStore._now()`
(`pipeline.py:66-67`), stamped onto progress events. `flywheel-trace.json` contains zero
timestamps. Judge records contain zero timestamps and zero latency — `time` is imported
in `judge.py` solely for retry backoff (`judge.py:772`). No `perf_counter` wraps any of
the three model call sites.

So there is no p50/p95, and no way to attribute run duration across planning, capture,
physics, and judging.

### 1.5 CLI runs are unobserved entirely

`_progress` (`flywheel.py:1751-1759`) adds no timestamp of its own, and with
`progress_callback=None` — the `evals/flywheel.py` CLI path — all event history
vanishes. CLI runs produce no `run.json`, no `run_id`, and no event stream. Every batch
eval driver uses this path.

### 1.6 Non-model stages emit no structured record

`compile_motion`, `store.persist`, and `capture_fn` produce artifacts but no event with
inputs, outputs, duration, or exit status. Capture stdout and stderr are explicitly
routed to `/dev/null` (`pipeline.py:44-45`), so a capture failure surfaces only as an
exit code.

### 1.7 Effective configuration is never snapshotted

`OPENAI_JUDGE_MODEL`, `_FALLBACK_MODEL`, `_REASONING_EFFORT`, `_IMAGE_DETAIL`,
`_MAX_IMAGE_DIMENSION_PX`, `OPENAI_PLANNER_MODEL`, `OPENAI_REPAIR_MODEL`
(`judge.py:686-703, 1071`; `planner.py:4117-4118`), plus `escalation_confidence` and
`max_model_calls`, are read from the environment at construction and only partially
echoed into `routing`. No single record states the configuration a run actually used.

---

## 2. Goals and non-goals

**Goals**

- G1. Every model call persists its complete request and response, sufficient to replay
  it against a different model without re-running the pipeline.
- G2. Every attempt is recorded, including failures, with error class, message, usage,
  and latency.
- G3. A real trace hierarchy — `run → stage → candidate → call → attempt` — with stable
  ids, so any record can be located from any other.
- G4. Wall-clock duration on every span.
- G5. One transcript format, identical for API-launched and CLI runs.
- G6. The effective configuration and code version of a run are captured once, completely.
- G7. Retention is bounded and explicit; a full transcript is affordable to keep forever.

**Non-goals**

- Changing any prompt, model routing, threshold, or judging behavior. This PR is
  strictly additive instrumentation. If a recorded value looks wrong, that is a finding
  for a later PR, not a fix here.
- Shipping a UI. The transcript is a file format plus a reader API; visualization is
  out of scope.
- Adopting OpenTelemetry or an external tracing vendor. See §8.1.

---

## 3. Design

### 3.1 One append-only transcript per run

```
results/pipeline-runs/<run_id>/
  run.json                     # unchanged, still the UI-facing summary
  transcript.jsonl             # NEW — append-only, one span record per line
  config.json                  # NEW — effective configuration snapshot
  payloads/
    <span_id>.request.json     # NEW — full request, including image references
    <span_id>.response.json    # NEW — raw response as returned
  artifacts/                   # unchanged
```

JSONL, not one big JSON, for three reasons: it is append-only so a crashed run still
leaves a readable prefix; it can be tailed live; and it never requires rewriting a large
file under a lock the way `run.json` does today.

### 3.2 Span record

```json
{
  "schema_version": "1.0",
  "span_id":   "01J8...",         // ULID, sortable by creation time
  "parent_id": "01J8...",         // null for the root run span
  "run_id":    "20260816T042756-f2107423",
  "kind":      "model_call",      // run | stage | candidate | model_call | attempt | tool
  "name":      "judge.rank_five",
  "started_at":  "2026-08-16T04:31:02.114Z",
  "ended_at":    "2026-08-16T04:31:19.882Z",
  "duration_ms": 17768,
  "status":    "ok",              // ok | error | skipped
  "error":     null,              // {type, message, traceback_digest}
  "attrs":     { },               // span-kind-specific, see below
  "refs":      {                  // stable cross-references, never positional
    "result_id": "000005-throw-a-right-jab",
    "candidate_index": 4,
    "round": 1,
    "evidence_manifest_sha256": "…"
  },
  "request_path":  "payloads/01J8….request.json",
  "response_path": "payloads/01J8….response.json"
}
```

ULIDs rather than UUID4 so a transcript sorts chronologically without parsing
timestamps. `refs` is the fix for §1.3: **every record carries the identifiers needed to
locate it, so no consumer ever reconstructs correlation from array position.**

### 3.3 Request and response capture

`request.json` stores the exact `input` passed to `responses.parse`, with one
substitution: base64 image data URIs are replaced by

```json
{"$image": {"sha256": "…", "bytes": 41233, "media_type": "image/png",
            "source_snapshot_id": "08-impact_pose-orbit",
            "source_path": "artifacts/round-1/candidate-4-…/08-impact_pose-orbit.png",
            "payload_dimensions_px": [960, 540], "detail": "high"}}
```

This keeps transcripts small while making the payload **exactly reconstructible** — and
critically it hashes *the bytes as sent*, closing the gap at §1.1 where only the source
PNG was hashed. Alongside it we store `system_prompt_sha256` and the prompt text itself
once per run in `config.json` (prompts are constants; storing them per call would be
pure duplication).

`response.json` stores `response.model_dump()` — the whole object, not just
`output_parsed`. That recovers reasoning summaries, `status`, `incomplete_details`,
refusals, and `created_at`.

### 3.4 Failed attempts become first-class spans

Every dispatch produces an `attempt` span, successful or not. On failure:

```json
{"kind": "attempt", "status": "error", "duration_ms": 8021,
 "error": {"type": "APITimeoutError", "message": "…"},
 "attrs": {"attempt_index": 0, "model": "gpt-5.6-luna",
           "usage": null, "transient_retry": true}}
```

This directly fixes §1.2. Token accounting becomes correct because a failed attempt that
consumed tokens records them.

### 3.5 Instrumentation surface

A single context-manager API, so instrumenting a call site is one line and impossible to
half-do:

```python
# src/rigby_poc/observability.py
class Tracer:
    def span(self, kind: str, name: str, *, parent: str | None = None,
             **refs: Any) -> AbstractContextManager[Span]: ...
    def current(self) -> Span | None: ...          # contextvar-based

class Span:
    def set(self, **attrs: Any) -> None: ...
    def record_request(self, payload: Any) -> None: ...
    def record_response(self, response: Any) -> None: ...
```

Parent resolution uses a `contextvar`, so `flywheel.py` does not have to thread a span
id through every function signature — which is what would make this PR unlandable.

A `NullTracer` with the same interface is the default when no run is active, so library
code is never conditional on whether tracing is on.

### 3.6 CLI and API converge

`PipelineRunStore.start` (`pipeline.py:114`) and `evals/flywheel.py main` both open a
root span through the same `Tracer`. `run_id` generation moves into the tracer so the
CLI gets one too, fixing §1.5. `run.json` continues to be written exactly as today —
this PR adds a channel, it does not migrate the existing one.

### 3.7 Retention

Measured on a real run (`20260816T042756-f2107423`): **39 MB, 150 PNGs.**

| Policy | Per run | Notes |
| --- | --- | --- |
| Transcript + config + payload JSON | ~1–3 MB | keep forever |
| Evidence PNG, lossless | 39 MB | keep until the run is compacted |
| Evidence WebP q88, after compaction | ~4.5 MB | 215 KB → 23 KB per image, measured |

`evals/compact_run.py` transcodes evidence to WebP **after** a run reaches a terminal
state, records both hashes, and marks the run compacted. Judging always uses lossless
PNG so the evidence contract and its hashes stay meaningful; compaction is archival
only. Pillow 12.1.1 is already a dependency.

Default: compact runs older than 7 days, keep transcripts indefinitely.

---

## 4. Components

### 4.1 `src/rigby_poc/observability.py` (new)

`Tracer`, `Span`, `NullTracer`, ULID generation, JSONL writer with an
`atomic_write_json`-consistent append, contextvar plumbing. No third-party dependency.

### 4.2 `src/rigby_poc/transcript.py` (new)

Reader API — the thing every future eval consumes:

```python
def load(run_dir: Path) -> Transcript: ...

class Transcript:
    spans: list[Span]
    def root(self) -> Span: ...
    def children(self, span_id: str) -> list[Span]: ...
    def by_kind(self, kind: str) -> list[Span]: ...
    def model_calls(self) -> list[Span]: ...
    def total_tokens(self) -> TokenUsage: ...
    def duration_by_stage(self) -> dict[str, float]: ...
    def request(self, span_id: str) -> dict[str, Any]: ...   # rehydrates $image refs
```

Pure, no network, no browser. Every trajectory eval in PR 09 is a function over a
`Transcript`.

### 4.3 Call-site instrumentation

| File | Change |
| --- | --- |
| `judge.py:735` `_parse_response` | wrap in an `attempt` span; `record_request`/`record_response` |
| `judge.py:807-877` `_routed_parse` | emit `model_call` parent span; **record failed attempts** rather than discarding at 831-836 |
| `judge.py:1125-1157` `recommend_repair` | route through `_routed_parse` for parity, or at minimum add refs + routing |
| `judge.py:1218-1229` `rank_five` | add `result_ids` and `evidence_manifest` paths to the record |
| `planner.py:4121, 4141` | wrap both; record the prompt, and **record the primary failure reason** dropped at 4143 |
| `flywheel.py` `_progress` call sites | open `stage` / `candidate` spans; `_progress` gains a timestamp |
| `pipeline.py:37-48` `capture_in_subprocess` | `tool` span; stop discarding stdout/stderr — capture into the span |
| `pipeline.py:114` / `flywheel.py main` | root span; shared `run_id` |

### 4.4 `config.json`

Effective configuration resolved at run start: all judge and planner model env vars,
`escalation_confidence`, `max_model_calls`, `max_rounds`, image detail and max dimension,
the four system prompts with their sha256, `rigby_poc.__version__`, git commit if
available, Python version, and the resolved rig profile id and sha256.

---

## 5. Test plan

All of these run with zero model calls.

- `tests/test_observability.py` — span nesting via contextvar; parent resolution across
  a thread boundary (the pipeline runs in a `threading.Thread`, `pipeline.py:141`);
  ULID monotonicity; JSONL append survives a mid-write crash (truncated last line is
  skipped by the reader, not fatal).
- `tests/test_transcript.py` — reader over a committed fixture transcript; `$image`
  rehydration; token totals; stage durations.
- `tests/test_transcript_completeness.py` — **the regression test that gives this PR
  teeth.** Run the full pipeline against a stub OpenAI client and a stub capture
  function, then assert:
  1. every model call has non-empty `request_path` and `response_path`;
  2. every span has a resolvable `parent_id` up to the root;
  3. every `model_call` span carries `refs.result_id` or `refs.round`;
  4. a deliberately failing primary call **appears** as an `attempt` span with
     `status: error` — this is the direct guard against the §1.2 regression;
  5. total tokens from the transcript equal the stub's dispatched total, including the
     failed attempt.
- `tests/test_compaction.py` — WebP transcode preserves dimensions; both hashes recorded;
  compaction is idempotent; a compacted run still loads.

---

## 6. Sequencing

Four PRs. Each is independently mergeable and leaves the tree green.

| PR | Contents | Risk |
| --- | --- | --- |
| **01a** | `observability.py`, `transcript.py`, tests, `NullTracer` wired nowhere | none — pure addition |
| **01b** | Instrument judge + planner call sites; `config.json` | low — additive, but touches hot paths |
| **01c** | Instrument flywheel stages and capture; unify CLI and API run roots | medium — touches `flywheel.py` control flow |
| **01d** | Compaction tool and retention policy | none — offline tool |

Landing 01a alone is useful: it makes the transcript format reviewable before any call
site depends on it.

---

## 7. Definition of done

1. `uv run pytest -q` green, including `test_transcript_completeness.py`.
2. A run launched from the API and the same prompt run from the CLI produce
   structurally identical transcripts.
3. Every model call in a real run can be replayed from its `request.json` without the
   pipeline.
4. A forced primary-call failure appears in the transcript with usage and error class.
5. `Transcript.duration_by_stage()` accounts for ≥95% of measured wall-clock.
6. One compacted run occupies under 6 MB.
7. No prompt, threshold, or routing behavior changed — diff review confirms additive only.

---

## 8. Risks and open decisions

### 8.1 Open — build or adopt

OpenTelemetry gives free tooling and a standard vocabulary, but adds a heavy dependency,
maps awkwardly onto file-based artifacts, and the span-per-model-call pattern still needs
custom attributes. **Recommendation: build the minimal tracer here.** It is roughly 300
lines, has no dependency, and writes exactly the artifact the evals need. Revisit only if
a live dashboard becomes a requirement. **Decide before 01a.**

### 8.2 Risk — payload storage of images

Storing full base64 per call would multiply run size by roughly the number of judge
calls. The `$image` reference in §3.3 avoids it, at the cost of transcripts being
non-portable away from their run directory. Accepted: transcripts are read in place.

### 8.3 Risk — thread-boundary context loss

`PipelineRunStore._execute` runs in a `threading.Thread` (`pipeline.py:141`). A
`contextvar` set on the parent does not automatically propagate. The root span must be
created *inside* the thread, or explicitly copied. This is the most likely source of a
silently orphaned span tree, and `test_observability.py` covers it directly.

### 8.4 Risk — secret leakage into artifacts

Request payloads must never contain the API key. The key is not in `input`, so this is
safe today, but `config.json` snapshots environment variables and must use an explicit
allowlist, never a dump of `os.environ`.

### 8.5 Note — this PR will make token accounting change

Once failed attempts are recorded, reported token totals will *increase* relative to
today's numbers. That is a correction, not a regression, and should be called out in the
PR description so it is not mistaken for a cost spike.
