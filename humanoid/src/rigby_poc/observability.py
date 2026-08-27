"""Run tracing: spans, ULIDs, and the append-only transcript writer.

This module writes the artifact described in `docs/plans/01-observability-and-transcript.md`
sections 3.1 through 3.5. It has no third-party dependency and it is wired into no call
site: PR 01a ships the format and its reader, PRs 01b and 01c instrument the pipeline.

The public surface is deliberately small:

    tracer = Tracer.open(root)                  # results/pipeline-runs/<run_id>/
    with tracer.span("stage", "planning") as stage:
        stage.set(recipe="five_way")
        with tracer.span("model_call", "planner.plan") as call:
            call.record_request(payload)
            call.record_response(response)

`NullTracer` satisfies the same interface and does nothing, so library code never has to
ask whether tracing is on.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .io_utils import atomic_write_json


__all__ = [
    "SCHEMA_VERSION",
    "SPAN_KINDS",
    "JsonlWriter",
    "NullSpan",
    "NullTracer",
    "Span",
    "SpanLike",
    "Tracer",
    "UlidFactory",
    "current_span",
    "current_span_id",
    "get_tracer",
    "new_run_id",
    "new_ulid",
    "propagate",
    "retry_on_transient_lock",
    "substitute_images",
    "ulid_timestamp_ms",
]

SCHEMA_VERSION = "1.0"

SPAN_KINDS = ("run", "stage", "candidate", "model_call", "attempt", "tool")
"""Span kinds from plan section 3.2. Advisory: unknown kinds are written unchanged."""

TRANSCRIPT_FILENAME = "transcript.jsonl"
PAYLOAD_DIRNAME = "payloads"
IMAGE_DIRNAME = "images"

_DATA_URI_PREFIX = "data:"
_LIFTED_IMAGE_KEYS = ("detail",)
"""Sibling keys moved from the image node onto its `$image` reference."""


# --------------------------------------------------------------------------------------
# ULID
# --------------------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
"""Crockford base32: no I, L, O, or U, so an id survives being read aloud."""

_TIMESTAMP_CHARS = 10  # 48-bit millisecond timestamp
_RANDOM_CHARS = 16  # 80 bits of randomness
_RANDOM_CEILING = (1 << 80) - 1
_TIMESTAMP_CEILING = (1 << 48) - 1


def _encode_crockford(value: int, length: int) -> str:
    characters = [""] * length
    for index in range(length - 1, -1, -1):
        characters[index] = _CROCKFORD[value & 0x1F]
        value >>= 5
    return "".join(characters)


class UlidFactory:
    """Monotonic ULID generator.

    A plain ULID only sorts by millisecond, so two ids minted in the same millisecond can
    sort against creation order — which would silently reorder a transcript. This factory
    keeps the previous timestamp and randomness and increments the random component
    whenever the clock does not advance, which makes the *string* order the creation
    order. The same rule absorbs a backwards clock step (NTP correction, VM resume).
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._last_timestamp = -1
        self._last_random = 0

    def __call__(self) -> str:
        return self.new()

    def new(self) -> str:
        with self._lock:
            timestamp = int(self._clock() * 1000)
            if timestamp <= self._last_timestamp:
                timestamp = self._last_timestamp
                randomness = self._last_random + 1
                if randomness > _RANDOM_CEILING:
                    # 2^80 ids inside one millisecond is not reachable in practice; step
                    # the timestamp rather than wrap and break monotonicity.
                    timestamp += 1
                    randomness = int.from_bytes(os.urandom(10), "big")
            else:
                randomness = int.from_bytes(os.urandom(10), "big")
            if timestamp > _TIMESTAMP_CEILING:
                raise OverflowError("ULID timestamp exceeds 48 bits")
            self._last_timestamp = timestamp
            self._last_random = randomness
        return _encode_crockford(timestamp, _TIMESTAMP_CHARS) + _encode_crockford(
            randomness, _RANDOM_CHARS
        )


_default_ulid_factory = UlidFactory()


def new_ulid() -> str:
    """Mint a monotonic ULID from the process-wide factory."""

    return _default_ulid_factory.new()


def ulid_timestamp_ms(value: str) -> int:
    """Recover the millisecond timestamp encoded in a ULID."""

    if len(value) != _TIMESTAMP_CHARS + _RANDOM_CHARS:
        raise ValueError("not a ULID")
    timestamp = 0
    for character in value[:_TIMESTAMP_CHARS]:
        position = _CROCKFORD.find(character.upper())
        if position < 0:
            raise ValueError("not a ULID")
        timestamp = (timestamp << 5) | position
    return timestamp


def new_run_id(*, clock: Callable[[], datetime] | None = None) -> str:
    """Mint a run id in the shape `PipelineRunStore` already writes.

    Plan section 3.6 moves run id generation here so a CLI run and an API run are named
    the same way. The format is pinned by `pipeline.SAFE_RUN_ID` and a test asserts the
    two agree.
    """

    now = clock() if clock is not None else datetime.now(UTC)
    return f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------


def _iso_millis(moment: datetime) -> str:
    """ISO-8601 with millisecond precision and a `Z` suffix, per plan section 3.2."""

    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


# --------------------------------------------------------------------------------------
# JSONL writer
# --------------------------------------------------------------------------------------


def retry_on_transient_lock(
    operation: Callable[[], Any],
    *,
    attempts: int = 12,
    sleep: Callable[[float], None] | None = None,
) -> Any:
    """Run `operation`, retrying the Windows transient-lock `PermissionError`.

    This mirrors the replace ladder in `io_utils.atomic_write_json`: Windows search
    indexing and OneDrive can hold a just-touched file for a few milliseconds, and the
    failure surfaces as `PermissionError` rather than anything more specific. The ladder
    is duplicated rather than imported because `atomic_write_json` wraps a whole-file
    replace and an append cannot use one; the two should be hoisted into `io_utils`
    together when a later PR is allowed to edit that module.
    """

    pause = sleep or time.sleep
    total = max(1, attempts)
    for attempt in range(total):
        try:
            return operation()
        except PermissionError:
            if attempt + 1 >= total:
                raise
            pause(min(0.02 * (2**attempt), 0.25))
    raise AssertionError("unreachable")


class JsonlWriter:
    """Append-only JSONL sink.

    Append rather than replace because a crashed run must still leave a readable prefix
    (plan section 3.1). The file is reopened per record instead of holding a handle: an
    open handle on Windows blocks another process from replacing or removing the file,
    and appends here are infrequent enough — a few hundred per run, each bracketing a
    multi-second stage — that the open cost is irrelevant.
    """

    def __init__(
        self,
        path: Path,
        *,
        replace_attempts: int = 12,
        fsync: bool = True,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._attempts = replace_attempts
        self._fsync = fsync
        self._sleep = sleep
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            retry_on_transient_lock(
                lambda: self._write(line), attempts=self._attempts, sleep=self._sleep
            )

    def _write(self, line: str) -> None:
        # newline="\n" is load bearing: without it Windows text mode rewrites every "\n"
        # as "\r\n", which changes the bytes on disk between the two platforms this repo
        # is developed on and leaves a stray "\r" on every field a byte-oriented reader
        # parses.
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            if self._fsync:
                os.fsync(stream.fileno())


# --------------------------------------------------------------------------------------
# Image references
# --------------------------------------------------------------------------------------


def _decode_data_uri(value: str) -> tuple[str, bytes] | None:
    """Split a `data:<media_type>;base64,<payload>` URI into its media type and bytes."""

    if not value.startswith(_DATA_URI_PREFIX):
        return None
    header, separator, encoded = value.partition(",")
    if not separator:
        return None
    header = header[len(_DATA_URI_PREFIX) :]
    if not header.endswith(";base64"):
        return None
    media_type = header[: -len(";base64")] or "application/octet-stream"
    try:
        return media_type, base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None


def _image_extension(media_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(media_type, ".bin")


def substitute_images(
    payload: Any,
    *,
    image_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    store: Callable[[str, str, bytes], str | None] | None = None,
) -> Any:
    """Replace inline base64 image data URIs with `$image` references.

    Plan section 3.3. Every node carrying a base64 data URI becomes

        {"$image": {"sha256": ..., "bytes": ..., "media_type": ..., ...}}

    The hash is of *the bytes as sent*, which is the gap section 1.1 describes: the
    evidence manifest hashes the source PNG, but the judge resizes and composites before
    dispatch and those bytes were never hashed anywhere.

    Sibling keys on the original node are preserved — `detail` rides along on the judge's
    `input_image` nodes already — and `image_metadata` supplies whatever the caller knows
    that the URI cannot say (`source_snapshot_id`, `source_path`,
    `payload_dimensions_px`). It is keyed by the payload sha256 rather than by position,
    because positional correlation is the defect section 1.3 exists to remove.
    """

    metadata = image_metadata or {}

    def walk(node: Any) -> Any:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if not isinstance(value, str):
                    continue
                decoded = _decode_data_uri(value)
                if decoded is None:
                    continue
                media_type, raw = decoded
                digest = hashlib.sha256(raw).hexdigest()
                reference: dict[str, Any] = {
                    "sha256": digest,
                    "bytes": len(raw),
                    "media_type": media_type,
                }
                for sibling, sibling_value in node.items():
                    if sibling != key and sibling in _LIFTED_IMAGE_KEYS:
                        reference[sibling] = sibling_value
                reference.update(dict(metadata.get(digest, {})))
                if store is not None:
                    stored = store(digest, media_type, raw)
                    if stored is not None:
                        reference["path"] = stored
                replacement = {
                    inner_key: walk(inner_value)
                    for inner_key, inner_value in node.items()
                    if inner_key != key and inner_key not in _LIFTED_IMAGE_KEYS
                }
                replacement["$image"] = reference
                return replacement
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(item) for item in node]
        return node

    return walk(payload)


# --------------------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------------------


class SpanLike(Protocol):
    """The surface call sites are allowed to touch, shared by `Span` and `NullSpan`."""

    span_id: str | None

    def set(self, **attrs: Any) -> None: ...

    def record_request(self, payload: Any, **kwargs: Any) -> None: ...

    def record_response(self, response: Any) -> None: ...


class Span:
    """One node in the run's trace tree.

    A span is written to the transcript exactly once, when it closes. Nothing is emitted
    on open: a half-written span carries no duration and no status, so it would be a
    record that every consumer has to special-case.
    """

    def __init__(
        self,
        *,
        tracer: Tracer,
        span_id: str,
        parent_id: str | None,
        run_id: str,
        kind: str,
        name: str,
        refs: Mapping[str, Any] | None = None,
    ) -> None:
        self._tracer = tracer
        self.span_id = span_id
        self.parent_id = parent_id
        self.run_id = run_id
        self.kind = kind
        self.name = name
        self.refs: dict[str, Any] = {key: value for key, value in (refs or {}).items() if value is not None}
        self.attrs: dict[str, Any] = {}
        self.status = "ok"
        self.error: dict[str, Any] | None = None
        self.request_path: str | None = None
        self.response_path: str | None = None
        self.started_at = _iso_millis(datetime.now(UTC))
        self.ended_at: str | None = None
        self.duration_ms: float | None = None
        self._started = time.perf_counter()

    # -- mutation -----------------------------------------------------------------

    @property
    def tracer(self) -> Tracer:
        """The tracer this span was opened on; how library code finds the live one."""

        return self._tracer

    def set(self, **attrs: Any) -> None:
        """Merge span-kind-specific attributes (plan section 3.2 `attrs`)."""

        self.attrs.update(attrs)

    def set_refs(self, **refs: Any) -> None:
        """Merge stable cross-references. `None` values are dropped, never stored."""

        self.refs.update({key: value for key, value in refs.items() if value is not None})

    def record_request(
        self,
        payload: Any,
        *,
        image_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Persist the exact request, with base64 images swapped for `$image` refs."""

        document = substitute_images(
            payload,
            image_metadata=image_metadata,
            store=self._tracer._store_image if self._tracer.store_payload_images else None,
        )
        self.request_path = self._tracer._write_payload(self.span_id, "request", document)

    def record_response(self, response: Any) -> None:
        """Persist the whole response object, not just its parsed projection.

        Section 1.1: today only `output_parsed` survives, which discards reasoning
        summaries, `status`, `incomplete_details`, refusals, and `created_at`.
        """

        document = response
        dump = getattr(response, "model_dump", None)
        if callable(dump):
            try:
                document = dump(mode="json")
            except TypeError:
                document = dump()
        self.response_path = self._tracer._write_payload(self.span_id, "response", document)

    def record_error(self, error: BaseException) -> None:
        self.status = "error"
        self.error = _error_record(error)

    def skip(self, reason: str | None = None) -> None:
        self.status = "skipped"
        if reason:
            self.attrs.setdefault("skip_reason", reason)

    # -- serialisation ------------------------------------------------------------

    def close(self) -> None:
        if self.ended_at is not None:
            return
        self.duration_ms = round((time.perf_counter() - self._started) * 1000, 3)
        self.ended_at = _iso_millis(datetime.now(UTC))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attrs": self.attrs,
            "refs": self.refs,
            "request_path": self.request_path,
            "response_path": self.response_path,
        }


def _error_record(error: BaseException) -> dict[str, Any]:
    formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return {
        "type": type(error).__name__,
        "message": str(error),
        # The traceback text itself is not stored: it carries absolute developer paths
        # into an artifact that gets shared. The digest still groups identical failures.
        "traceback_digest": hashlib.sha256(formatted.encode("utf-8")).hexdigest(),
    }


class NullSpan:
    """No-op span. Every method is a sink so call sites never branch on tracing."""

    span_id: str | None = None
    parent_id: str | None = None
    kind = "null"
    name = "null"
    status = "ok"

    def set(self, **attrs: Any) -> None:
        return None

    def set_refs(self, **refs: Any) -> None:
        return None

    def record_request(self, payload: Any, **kwargs: Any) -> None:
        return None

    def record_response(self, response: Any) -> None:
        return None

    def record_error(self, error: BaseException) -> None:
        return None

    def skip(self, reason: str | None = None) -> None:
        return None


# --------------------------------------------------------------------------------------
# Context propagation
# --------------------------------------------------------------------------------------

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "rigby_current_span", default=None
)


def current_span() -> Span | None:
    """The innermost open span in this context, or `None`."""

    return _current_span.get()


def current_span_id() -> str | None:
    span = _current_span.get()
    return span.span_id if span is not None else None


def propagate(function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap `function` so it runs with a copy of the *current* trace context.

    Plan section 8.3, verified: a `contextvars.ContextVar` set on the parent is not
    visible inside a plain `threading.Thread`, and `PipelineRunStore.start`
    (`pipeline.py:141`) hands `_execute` to exactly such a thread. The child therefore
    sees `current_span() is None` and every span it opens becomes a second root — a
    silently orphaned tree rather than an error.

    Two fixes work, and both are supported here. Either capture the context at submit
    time with this wrapper, or pass the parent id explicitly to `Tracer.span(parent=...)`.
    The wrapper must be applied on the *submitting* side, while the intended parent span
    is still open.
    """

    context = contextvars.copy_context()

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return context.run(function, *args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------------------
# Tracers
# --------------------------------------------------------------------------------------


class NullTracer:
    """Default tracer. Same interface, writes nothing, touches no context."""

    run_id: str | None = None
    run_dir: Path | None = None
    store_payload_images = False

    @contextmanager
    def span(self, kind: str, name: str, *, parent: str | None = None, **refs: Any) -> Iterator[NullSpan]:
        yield NullSpan()

    def current(self) -> Span | None:
        return None

    def close(self) -> None:
        return None


class Tracer:
    """Writes one append-only transcript per run.

    Parent resolution is contextvar based (plan section 3.5) so instrumenting a call site
    is one `with` statement and no function signature grows a span argument — threading
    ids through `flywheel.py` by hand is what would make this unlandable. `parent=` is the
    escape hatch for boundaries a contextvar cannot cross; see `propagate`.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        run_id: str,
        ulid_factory: Callable[[], str] | None = None,
        fsync: bool = True,
        store_payload_images: bool = True,
        writer: JsonlWriter | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.store_payload_images = store_payload_images
        self._new_id = ulid_factory or new_ulid
        self._writer = writer or JsonlWriter(self.run_dir / TRANSCRIPT_FILENAME, fsync=fsync)

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> Tracer:
        """Create a tracer for `root/<run_id>/`, minting a run id when none is given."""

        resolved = run_id or new_run_id()
        return cls(Path(root) / resolved, run_id=resolved, **kwargs)

    # -- spans --------------------------------------------------------------------

    @contextmanager
    def span(
        self,
        kind: str,
        name: str,
        *,
        parent: str | None = None,
        **refs: Any,
    ) -> Iterator[Span]:
        parent_id = parent if parent is not None else current_span_id()
        span = Span(
            tracer=self,
            span_id=self._new_id(),
            parent_id=parent_id,
            run_id=self.run_id,
            kind=kind,
            name=name,
            refs=refs,
        )
        token = _current_span.set(span)
        try:
            yield span
        except BaseException as error:
            span.record_error(error)
            raise
        finally:
            _current_span.reset(token)
            span.close()
            self._writer.append(span.to_record())

    def current(self) -> Span | None:
        return current_span()

    def close(self) -> None:
        return None

    # -- payloads -----------------------------------------------------------------

    def _write_payload(self, span_id: str, suffix: str, document: Any) -> str:
        relative = f"{PAYLOAD_DIRNAME}/{span_id}.{suffix}.json"
        target = self.run_dir / PAYLOAD_DIRNAME / f"{span_id}.{suffix}.json"
        # atomic_write_json takes a mapping; a payload can legitimately be a list (the
        # Responses `input` is one), so wrap non-mappings in an envelope that the reader
        # unwraps rather than silently coercing the shape.
        if isinstance(document, Mapping):
            atomic_write_json(target, dict(document))
        else:
            atomic_write_json(target, {"$envelope": "value", "value": document})
        return relative

    def _store_image(self, digest: str, media_type: str, raw: bytes) -> str:
        """Persist payload image bytes content-addressed, so a request can be replayed.

        The `$image` reference hashes the resized bytes actually dispatched, which no
        source PNG on disk reproduces byte for byte. Storing them here — deduplicated by
        content, so the same evidence frame sent to five judge calls is written once — is
        what makes definition-of-done item 3 ("replayed from its request.json without the
        pipeline") true rather than aspirational.
        """

        relative = f"{PAYLOAD_DIRNAME}/{IMAGE_DIRNAME}/{digest}{_image_extension(media_type)}"
        target = self.run_dir / PAYLOAD_DIRNAME / IMAGE_DIRNAME / f"{digest}{_image_extension(media_type)}"
        if target.is_file():
            return relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            retry_on_transient_lock(lambda: os.replace(temporary, target))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # A failed cleanup must not mask the outcome of the write itself.
                pass
        return relative


class StageTimeline:
    """Flat, non-overlapping `stage` spans over a run.

    `Transcript.duration_by_stage()` sums *every* span whose kind is `stage`, whatever its
    depth, so a stage opened inside another stage is counted twice — measured at 151% of
    wall clock for a single nested pair. Plan 01 section 7 asks stage durations to account
    for at least 95% of wall clock, which means the target can be overshot into
    meaninglessness rather than merely missed: nesting sails past 95% by double counting
    and reports it as success.

    This makes that impossible by construction rather than by discipline. Entering a stage
    closes the previous one, so the spans tile the run instead of nesting, and a caller
    cannot get it wrong by bracketing something in the wrong order.

    Re-entering the same stage with different `refs` — the next candidate, the next round —
    starts a fresh span, so each occurrence is separately attributable while
    `duration_by_stage` still sums them under one name.
    """

    def __init__(self, tracer: Tracer | NullTracer, *, parent: str | None = None) -> None:
        self._tracer = tracer
        self._parent = parent
        self._open: ExitStack | None = None
        self._key: tuple[Any, ...] | None = None

    def enter(self, name: str, **refs: Any) -> None:
        key = (name, tuple(sorted((str(k), str(v)) for k, v in refs.items())))
        if key == self._key:
            return
        self.close()
        stack = ExitStack()
        stack.enter_context(self._tracer.span("stage", name, parent=self._parent, **refs))
        self._open, self._key = stack, key

    def close(self) -> None:
        stack, self._open, self._key = self._open, None, None
        if stack is not None:
            stack.close()

    def __enter__(self) -> StageTimeline:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


_current_stage_timeline: contextvars.ContextVar[StageTimeline | None] = contextvars.ContextVar(
    "rigby_stage_timeline", default=None
)


def current_stage_timeline() -> StageTimeline | None:
    """The timeline the running pipeline is reporting stages to, if any."""

    return _current_stage_timeline.get()


@contextmanager
def stage_timeline(tracer: Tracer | NullTracer, *, parent: str | None = None) -> Iterator[StageTimeline]:
    """Install a `StageTimeline` for the duration of a run.

    Contextvar based for the same reason span parenting is (plan section 3.5): the
    alternative is threading a timeline argument through every progress call site in
    `flywheel.py`, which is what would make this unlandable.
    """

    timeline = StageTimeline(tracer, parent=parent)
    token = _current_stage_timeline.set(timeline)
    try:
        yield timeline
    finally:
        _current_stage_timeline.reset(token)
        timeline.close()


_null_tracer = NullTracer()


def get_tracer() -> Tracer | NullTracer:
    """The tracer of whichever span is currently open, or a `NullTracer`.

    Derived from the open span rather than held in a second contextvar, so there is no
    way for "the active tracer" and "the span being parented to" to disagree. Library
    code -- `pipeline.capture_in_subprocess`, for one -- calls this to open a span
    without being handed a tracer, and gets a no-op outside a run.
    """

    span = current_span()
    if span is not None:
        return span.tracer
    return _null_tracer
