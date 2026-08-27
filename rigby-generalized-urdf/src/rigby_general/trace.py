"""A durable, inspectable record of how one prompt became a motion.

Rigby's own lesson, restated: "Planning, proposal generation, structural checks,
evidence capture, VLM scores, repair, and final selection remain visible instead
of being collapsed into a spinner." A pipeline that only reports its verdict is a
pipeline nobody can debug, and one whose refusals cannot be told apart.

So every stage records what it decided and why, including the ones that
succeeded. The two most useful fields are the ones easiest to leave out:

``cues``  which surface phrase fired which schema. When a prompt is read wrongly,
          this is the difference between "the planner got it wrong" and "the word
          *across* is matching a sweep here".
``role_normalized_hash``  the body-neutral reading. Two traces from different
          robots sharing this hash is the whole magnitude-neutrality claim, made
          checkable by eye rather than only by the audit.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rigby_v2.hashing import content_hash

from .config import base_tree_fingerprint


SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One pipeline stage: what it did, how long it took, what it decided."""

    name: str
    status: str
    """``ok``, ``refused`` or ``failed``."""

    elapsed_ms: float
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "summary": self.summary,
            "detail": self.detail,
        }


@dataclass
class RunTrace:
    """Everything that happened, in order."""

    prompt: str
    robot_id: str
    accepted: bool = False
    kind: str = "prompt"
    """``prompt`` for a run driven by language, ``contact_probe`` for a direct
    grasp attempt.

    Kept distinct rather than dressing a grasp probe up as a prompt run. Contact
    schemas are unafforded until a grasp certifies, so asking a robot to "pick it
    up" in words is genuinely refused -- and a probe that bypasses the planner to
    exercise the physics is a different kind of evidence, worth labelling as
    such."""

    stages: list[StageRecord] = field(default_factory=list)
    robot: dict[str, Any] = field(default_factory=dict)
    schema_program: dict[str, Any] | None = None
    role_normalized_hash: str | None = None
    bindings: list[dict[str, Any]] = field(default_factory=list)
    grounded: dict[str, Any] | None = None
    certification: dict[str, Any] | None = None
    grasp: dict[str, Any] | None = None
    trial: dict[str, Any] | None = None
    """Set on an ``environment_trial``: which world, which object, and what
    admission decided before anything was simulated."""

    failure: dict[str, Any] | None = None
    clip: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def trace_id(self) -> str:
        slug = SLUG.sub("-", self.prompt.lower()).strip("-")[:52]
        return f"{self.robot_id}--{slug}" if slug else self.robot_id

    def record(
        self,
        name: str,
        status: str,
        elapsed_ms: float,
        summary: str,
        **detail: Any,
    ) -> None:
        self.stages.append(
            StageRecord(
                name=name,
                status=status,
                elapsed_ms=elapsed_ms,
                summary=summary,
                detail=detail,
            )
        )

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "trace_id": self.trace_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt": self.prompt,
            "robot_id": self.robot_id,
            "accepted": self.accepted,
            "kind": self.kind,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "robot": self.robot,
            "stages": [stage.to_json() for stage in self.stages],
            "schema_program": self.schema_program,
            "role_normalized_hash": self.role_normalized_hash,
            "bindings": self.bindings,
            "grounded": self.grounded,
            "certification": self.certification,
            "grasp": self.grasp,
            "trial": self.trial,
            "failure": self.failure,
            "clip": self.clip,
            "provenance": {"base_tree": base_tree_fingerprint().as_dict()},
        }
        payload["content_sha256"] = content_hash(
            {k: v for k, v in payload.items() if k != "created_at"}
        )
        return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class TraceStore:
    """Traces on disk, one directory each, alongside their clip."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def directory(self, trace_id: str) -> Path:
        if "/" in trace_id or "\\" in trace_id or ".." in trace_id:
            raise ValueError(f"unsafe trace id: {trace_id!r}")
        return self.root / trace_id

    def write(self, trace: RunTrace, *, clip_bytes: bytes | None = None) -> Path:
        directory = self.directory(trace.trace_id)
        if clip_bytes is not None:
            _atomic_write(directory / "clip.gif", clip_bytes)
            trace.clip = "clip.gif"
        elif (directory / "clip.gif").is_file():
            # A rebuild that skips rendering must not orphan the clip already on
            # disk. Leaving ``clip`` unset here silently detached every existing
            # recording from its trace, and the views that fall back to one --
            # contact probes above all -- went blank with the file still sitting
            # right there beside the trace.
            trace.clip = "clip.gif"
        _atomic_write(
            directory / "trace.json",
            json.dumps(trace.to_json(), indent=2, sort_keys=True).encode("utf-8"),
        )
        return directory

    def list_ids(self) -> tuple[str, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(
            sorted(
                entry.name
                for entry in self.root.iterdir()
                if entry.is_dir() and (entry / "trace.json").is_file()
            )
        )

    def load(self, trace_id: str) -> dict[str, Any]:
        path = self.directory(trace_id) / "trace.json"
        if not path.is_file():
            raise KeyError(trace_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def clip_path(self, trace_id: str) -> Path | None:
        candidate = self.directory(trace_id) / "clip.gif"
        return candidate if candidate.is_file() else None

    def index(self) -> list[dict[str, Any]]:
        """A compact listing, plus the cross-robot grouping by schema hash.

        Grouping here rather than in the browser because it is the one view that
        makes the central claim visible at a glance: several rows, different
        bodies, one hash.
        """

        rows: list[dict[str, Any]] = []
        for trace_id in self.list_ids():
            payload = self.load(trace_id)
            rows.append(
                {
                    "trace_id": trace_id,
                    "prompt": payload["prompt"],
                    "robot_id": payload["robot_id"],
                    "accepted": payload["accepted"],
                    "kind": payload.get("kind", "prompt"),
                    "role_normalized_hash": payload.get("role_normalized_hash"),
                    "duration_s": (payload.get("grounded") or {}).get("duration_s"),
                    "has_clip": bool(payload.get("clip")),
                    "failure_stage": (payload.get("failure") or {}).get("stage"),
                    "schema_keys": (payload.get("schema_program") or {}).get(
                        "canonical_keys", []
                    ),
                }
            )
        return rows
