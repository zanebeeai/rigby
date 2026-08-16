"""Reader for the run transcript written by `observability.py`.

Plan section 4.2. This is the surface every future eval consumes: a trajectory eval is a
pure function over a `Transcript`. Nothing here touches the network, a browser, or a
model — loading a transcript is reading files out of one run directory.

    transcript = load(run_dir)
    for call in transcript.model_calls():
        payload = transcript.request(call.span_id)   # $image refs rehydrated
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .observability import PAYLOAD_DIRNAME, SCHEMA_VERSION, TRANSCRIPT_FILENAME


__all__ = [
    "SkippedLine",
    "Span",
    "TokenUsage",
    "Transcript",
    "TranscriptError",
    "load",
]


class TranscriptError(RuntimeError):
    """Raised when a transcript cannot answer a question it was asked."""


@dataclass(frozen=True)
class TokenUsage:
    """Token counts summed off the transcript.

    Section 8.5: once failed attempts are recorded these totals go *up* relative to
    today's numbers, because a primary call that burned tokens and then threw is
    currently invisible. That is a correction, not a cost spike.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    @classmethod
    def from_usage(cls, usage: Mapping[str, Any] | None) -> TokenUsage:
        """Read an OpenAI Responses usage block, tolerating absent fields."""

        if not isinstance(usage, Mapping):
            return cls()

        def count(value: Any) -> int:
            return int(value) if isinstance(value, (int, float)) else 0

        details = usage.get("output_tokens_details")
        reasoning = (
            count(details.get("reasoning_tokens")) if isinstance(details, Mapping) else 0
        )
        input_tokens = count(usage.get("input_tokens"))
        output_tokens = count(usage.get("output_tokens"))
        total = count(usage.get("total_tokens")) or input_tokens + output_tokens
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning,
            total_tokens=total,
        )


@dataclass(frozen=True)
class SkippedLine:
    """A transcript line the reader could not parse, kept so it is never silent."""

    line_number: int
    reason: str
    text: str


@dataclass(frozen=True)
class Span:
    """One transcript record. Read-only counterpart to `observability.Span`."""

    span_id: str
    parent_id: str | None
    run_id: str | None
    kind: str
    name: str
    started_at: str | None = None
    ended_at: str | None = None
    duration_ms: float | None = None
    status: str = "ok"
    error: dict[str, Any] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    refs: dict[str, Any] = field(default_factory=dict)
    request_path: str | None = None
    response_path: str | None = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Span:
        span_id = record.get("span_id")
        if not isinstance(span_id, str) or not span_id:
            raise ValueError("span record has no span_id")
        parent_id = record.get("parent_id")
        return cls(
            span_id=span_id,
            parent_id=parent_id if isinstance(parent_id, str) else None,
            run_id=record.get("run_id") if isinstance(record.get("run_id"), str) else None,
            kind=str(record.get("kind") or "unknown"),
            name=str(record.get("name") or ""),
            started_at=record.get("started_at"),
            ended_at=record.get("ended_at"),
            duration_ms=record.get("duration_ms"),
            status=str(record.get("status") or "ok"),
            error=record.get("error") if isinstance(record.get("error"), Mapping) else None,
            attrs=dict(record.get("attrs") or {}),
            refs=dict(record.get("refs") or {}),
            request_path=record.get("request_path"),
            response_path=record.get("response_path"),
            schema_version=str(record.get("schema_version") or SCHEMA_VERSION),
        )

    @property
    def usage(self) -> TokenUsage:
        return TokenUsage.from_usage(self.attrs.get("usage"))

    @property
    def has_usage(self) -> bool:
        return isinstance(self.attrs.get("usage"), Mapping)


class Transcript:
    """An in-memory view of one run's `transcript.jsonl`."""

    def __init__(
        self,
        run_dir: Path,
        spans: list[Span],
        *,
        skipped: list[SkippedLine] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.spans = spans
        self.skipped_lines = skipped or []
        self._by_id = {span.span_id: span for span in spans}
        self._children: dict[str | None, list[Span]] = defaultdict(list)
        for span in spans:
            self._children[span.parent_id].append(span)

    # -- structure ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.spans)

    def by_id(self, span_id: str) -> Span:
        try:
            return self._by_id[span_id]
        except KeyError:
            raise TranscriptError(f"no span {span_id!r} in this transcript") from None

    def roots(self) -> list[Span]:
        """Every span with no parent. More than one means an orphaned subtree."""

        return list(self._children[None])

    def root(self) -> Span:
        roots = self.roots()
        if not roots:
            raise TranscriptError("transcript has no root span")
        return roots[0]

    def children(self, span_id: str) -> list[Span]:
        return list(self._children.get(span_id, ()))

    def descendants(self, span_id: str) -> list[Span]:
        found: list[Span] = []
        pending = list(self._children.get(span_id, ()))
        while pending:
            span = pending.pop()
            found.append(span)
            pending.extend(self._children.get(span.span_id, ()))
        found.sort(key=lambda span: span.span_id)
        return found

    def ancestors(self, span_id: str) -> list[Span]:
        """Walk parent links to the root, nearest first. Cycle-safe."""

        chain: list[Span] = []
        seen = {span_id}
        parent_id = self.by_id(span_id).parent_id
        while parent_id is not None and parent_id not in seen:
            parent = self._by_id.get(parent_id)
            if parent is None:
                break
            chain.append(parent)
            seen.add(parent_id)
            parent_id = parent.parent_id
        return chain

    def orphans(self) -> list[Span]:
        """Spans naming a parent that is not in the transcript.

        Section 8.3's failure mode: a span opened inside a `threading.Thread` that never
        received the parent context does not error, it just detaches. This is how a
        completeness test detects that.
        """

        return [
            span
            for span in self.spans
            if span.parent_id is not None and span.parent_id not in self._by_id
        ]

    def by_kind(self, kind: str) -> list[Span]:
        return [span for span in self.spans if span.kind == kind]

    def model_calls(self) -> list[Span]:
        return self.by_kind("model_call")

    def attempts(self) -> list[Span]:
        return self.by_kind("attempt")

    def failures(self) -> list[Span]:
        return [span for span in self.spans if span.status == "error"]

    # -- aggregates ---------------------------------------------------------------

    def total_tokens(self) -> TokenUsage:
        """Sum usage over the leaves of the usage tree.

        A `model_call` and its `attempt` children can both carry usage — the parent as an
        aggregate, the children as the actual dispatches. Summing every span that has a
        usage block would double count, and summing only `attempt` spans would report
        zero for any call site instrumented without them. Counting a span only when no
        descendant of it carries usage is correct under both shapes.
        """

        total = TokenUsage()
        for span in self.spans:
            if not span.has_usage:
                continue
            if any(child.has_usage for child in self.descendants(span.span_id)):
                continue
            total = total + span.usage
        return total

    def duration_by_stage(self) -> dict[str, float]:
        """Milliseconds of wall clock per `stage` span name, summed across repeats."""

        durations: dict[str, float] = defaultdict(float)
        for span in self.by_kind("stage"):
            if isinstance(span.duration_ms, (int, float)):
                durations[span.name] += float(span.duration_ms)
        return dict(durations)

    # -- payloads -----------------------------------------------------------------

    def raw_request(self, span_id: str) -> Any:
        """The request exactly as stored, with `$image` references left in place."""

        return self._read_payload(self.by_id(span_id).request_path, "request", span_id)

    def request(self, span_id: str) -> Any:
        """The request with every `$image` reference rehydrated to its data URI.

        The result is the payload as dispatched, so it can be replayed against another
        model without the pipeline. Rehydration verifies each image against the sha256
        recorded at send time and raises rather than returning a payload that only looks
        replayable.
        """

        return self._rehydrate(self.raw_request(span_id))

    def response(self, span_id: str) -> Any:
        return self._read_payload(self.by_id(span_id).response_path, "response", span_id)

    def _read_payload(self, relative: str | None, label: str, span_id: str) -> Any:
        if not relative:
            raise TranscriptError(f"span {span_id!r} has no stored {label}")
        path = self.run_dir / Path(relative)
        try:
            resolved = path.resolve()
        except OSError as error:
            raise TranscriptError(f"cannot resolve {label} for {span_id!r}: {error}") from error
        root = self.run_dir.resolve()
        if root not in resolved.parents:
            raise TranscriptError(f"{label} path for {span_id!r} escapes the run directory")
        if not resolved.is_file():
            raise TranscriptError(f"missing {label} payload for {span_id!r}: {relative}")
        document = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(document, Mapping) and document.get("$envelope") == "value":
            return document["value"]
        return document

    def _rehydrate(self, node: Any) -> Any:
        if isinstance(node, Mapping):
            reference = node.get("$image")
            if isinstance(reference, Mapping):
                rebuilt = {key: self._rehydrate(value) for key, value in node.items() if key != "$image"}
                media_type = str(reference.get("media_type") or "image/png")
                raw = self._image_bytes(reference)
                encoded = base64.b64encode(raw).decode("ascii")
                rebuilt["image_url"] = f"data:{media_type};base64,{encoded}"
                for lifted in ("detail",):
                    if lifted in reference:
                        rebuilt[lifted] = reference[lifted]
                return rebuilt
            return {key: self._rehydrate(value) for key, value in node.items()}
        if isinstance(node, list):
            return [self._rehydrate(item) for item in node]
        return node

    def _image_bytes(self, reference: Mapping[str, Any]) -> bytes:
        digest = reference.get("sha256")
        relative = reference.get("path")
        if not isinstance(relative, str) or not relative:
            if isinstance(digest, str):
                relative = f"{PAYLOAD_DIRNAME}/images/{digest}.png"
            else:
                raise TranscriptError("$image reference carries neither a path nor a sha256")
        path = (self.run_dir / Path(relative)).resolve()
        if self.run_dir.resolve() not in path.parents:
            raise TranscriptError("$image path escapes the run directory")
        if not path.is_file():
            raise TranscriptError(f"$image bytes are not stored in this run: {relative}")
        raw = path.read_bytes()
        if isinstance(digest, str) and hashlib.sha256(raw).hexdigest() != digest:
            raise TranscriptError(f"$image bytes do not match the recorded sha256: {relative}")
        return raw


def load(run_dir: Path, *, filename: str = TRANSCRIPT_FILENAME) -> Transcript:
    """Read `run_dir/transcript.jsonl`.

    A line that does not parse is skipped and recorded in `Transcript.skipped_lines`
    rather than raising. The file is append-only and never rewritten, so a process that
    died mid-append leaves a truncated final line; treating that as fatal would make
    every crashed run unreadable at exactly the moment the transcript is most useful.
    Spans are returned in ULID order, which is creation order.
    """

    directory = Path(run_dir)
    path = directory / filename
    if not path.is_file():
        raise TranscriptError(f"no transcript at {path}")

    spans: list[Span] = []
    skipped: list[SkippedLine] = []
    # newline="" leaves line endings untouched so a transcript written on Windows and a
    # transcript written on macOS parse identically; the strip below handles both.
    with path.open("r", encoding="utf-8", newline="") as stream:
        for number, raw in enumerate(stream, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as error:
                skipped.append(SkippedLine(number, f"invalid json: {error.msg}", text))
                continue
            if not isinstance(record, Mapping):
                skipped.append(SkippedLine(number, "line is not an object", text))
                continue
            try:
                spans.append(Span.from_record(record))
            except ValueError as error:
                skipped.append(SkippedLine(number, str(error), text))

    spans.sort(key=lambda span: span.span_id)
    return Transcript(directory, spans, skipped=skipped)
