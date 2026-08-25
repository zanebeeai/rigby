"""Recorded grader payloads, replayed through the scorer a live run uses.

`--offline-scoring` exists so the whole scoring path can be exercised with zero
model spend (plan 10 §8.1). That makes this a stand-in for the real caller, and a
harness fails in a specific way: **"does my harness do what the caller will do?"
is a different question from "is my measurement correct".** Both halves have to
be checked rather than assumed.

**Same code path.** There is no scoring function in this module. The driver calls
`judge.assemble_split_score` — the identical pure, total function a live run
calls — and this module only supplies its arguments. That is deliberate: an
`if offline` branch anywhere in the scorer means the offline number is not the
online number, and the exercise is worthless. The absence of a wrapper is the
guarantee; a wrapper is where the fork would eventually be added.

**Same payload.** PR 01b makes the other half checkable. `Span.record_request`
stores the payload exactly as dispatched, with base64 images replaced by `$image`
refs hashing the resized composited bytes actually sent, and
`Transcript.request(span_id)` raises when stored bytes fail their recorded
sha256. A harness can run the real path on a payload the real caller would never
have sent, so the first half without the second leaves the same gap.

The key is `(case, grader)` and not `(case, grader, dimension)`: `semantic` owns
two dimensions from one claim set, so dimension is not a key and using it would
let the second dimension silently overwrite the first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rigby_poc.io_utils import atomic_write_json
from rigby_poc.judge_prompts import GRADER_NAMES


class ReplayError(RuntimeError):
    """The store cannot answer honestly for this case."""


@dataclass(frozen=True)
class RecordedGrader:
    """One grader's answer for one case, as it came back from the model."""

    case_id: str
    grader: str
    #: `graders.<name>.call.parsed` — the claim list the scorer consumes.
    parsed: dict[str, Any]
    #: `graders.<name>.prompt` — {grader, version, sha256, family}. A calibration
    #: result is only meaningful against a stated prompt version (07 §3.5), so
    #: this travels with the payload rather than being looked up later.
    prompt: dict[str, Any]

    @property
    def key(self) -> tuple[str, str]:
        return (self.case_id, self.grader)


def graders_from_record(case_id: str, record: Mapping[str, Any]) -> list[RecordedGrader]:
    """Split one split-judgment record into its per-grader payloads."""
    if record.get("kind") != "split_motion_judgment":
        raise ReplayError(
            f"{case_id}: replay needs a split_motion_judgment, got {record.get('kind')!r}"
        )
    graders = record.get("graders")
    if not isinstance(graders, Mapping):
        raise ReplayError(f"{case_id}: record carries no graders block")
    recorded: list[RecordedGrader] = []
    for name in GRADER_NAMES:
        entry = graders.get(name)
        if not isinstance(entry, Mapping):
            raise ReplayError(f"{case_id}: no recorded payload for grader {name}")
        parsed = entry.get("call", {}).get("parsed")
        if not isinstance(parsed, Mapping):
            raise ReplayError(f"{case_id}: grader {name} recorded no parsed payload")
        recorded.append(
            RecordedGrader(
                case_id=case_id,
                grader=name,
                parsed=dict(parsed),
                prompt=dict(entry.get("prompt") or {}),
            )
        )
    return recorded


class ReplayStore:
    """Recorded payloads on disk, one file per `(case, grader)`.

    Flat and boring on purpose: the file name *is* the key, so a duplicate key is
    a filesystem collision rather than a silent overwrite inside a dict.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, case_id: str, grader: str) -> Path:
        if "/" in case_id or "\\" in case_id or case_id in {"", ".", ".."}:
            raise ReplayError(f"unsafe case id for a filename: {case_id!r}")
        return self.root / case_id / f"{grader}.json"

    def write(self, recorded: RecordedGrader) -> Path:
        path = self._path(recorded.case_id, recorded.grader)
        atomic_write_json(
            path,
            {
                "schema_version": "1.0",
                "case_id": recorded.case_id,
                "grader": recorded.grader,
                "prompt": recorded.prompt,
                "parsed": recorded.parsed,
            },
        )
        return path

    def write_record(self, case_id: str, record: Mapping[str, Any]) -> list[Path]:
        return [self.write(item) for item in graders_from_record(case_id, record)]

    def cases(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(child.name for child in self.root.iterdir() if child.is_dir())

    def parts(self, case_id: str) -> dict[str, dict[str, Any]]:
        """The `parts` argument `judge.assemble_split_score` takes, unmodified.

        Raises on a partial case rather than scoring what is present: a case
        missing one grader would otherwise produce a score derived from four,
        which is a different instrument reported under the same name.
        """
        parts: dict[str, dict[str, Any]] = {}
        for name in GRADER_NAMES:
            path = self._path(case_id, name)
            if not path.is_file():
                raise ReplayError(f"{case_id}: no recorded payload for grader {name}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            parts[name] = payload["parsed"]
        return parts

    def prompt_versions(self, case_id: str) -> dict[str, dict[str, Any]]:
        versions: dict[str, dict[str, Any]] = {}
        for name in GRADER_NAMES:
            payload = json.loads(self._path(case_id, name).read_text(encoding="utf-8"))
            versions[name] = payload.get("prompt") or {}
        return versions


def prompt_versions_agree(store: ReplayStore, case_ids: list[str]) -> dict[str, set[str]]:
    """The prompt sha256 each grader was recorded against, across cases.

    More than one sha for a grader means the recordings span a prompt change, and
    pooling them reports one calibration for two instruments (07 §3.5). Returned
    rather than raised: the driver decides whether to refuse or to stratify.
    """
    seen: dict[str, set[str]] = {name: set() for name in GRADER_NAMES}
    for case_id in case_ids:
        for name, prompt in store.prompt_versions(case_id).items():
            digest = prompt.get("sha256")
            if digest:
                seen[name].add(str(digest))
    return seen
