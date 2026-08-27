"""Content-addressed release evidence writers.

The release checklist consumes only immutable JSON evidence whose digest is
recorded in a separate manifest.  This module deliberately does not turn a
partial evidence set into a ship decision; missing requirements remain blockers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from .checklist import ReleaseEvidence


_REQUIREMENT_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_release_evidence(
    evidence_root: Path,
    *,
    requirement_id: str,
    passed: bool,
    payload: Mapping[str, object],
) -> ReleaseEvidence:
    """Write one deterministic evidence document and return its checklist row."""

    if not _REQUIREMENT_ID.fullmatch(requirement_id):
        raise ValueError("release requirement ID is unsafe")
    document = {
        "schema_version": "1.0",
        "requirement_id": requirement_id,
        "passed": bool(passed),
        "payload": dict(payload),
    }
    encoded = _canonical_json(document)
    relative = f"{requirement_id}.json"
    destination = evidence_root.resolve() / relative
    _atomic_write(destination, encoded)
    return ReleaseEvidence(
        requirement_id=requirement_id,
        status="pass" if passed else "fail",
        artifact_path=relative,
        artifact_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def write_release_evidence_manifest(
    path: Path,
    evidence: Sequence[ReleaseEvidence],
) -> None:
    """Write the exact input consumed by ``scripts/v2/release_check.py``."""

    identifiers = [item.requirement_id for item in evidence]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("release evidence manifest contains duplicate requirements")
    payload = [
        {
            "requirement_id": item.requirement_id,
            "status": item.status,
            "artifact_path": item.artifact_path,
            "artifact_sha256": item.artifact_sha256,
        }
        for item in sorted(evidence, key=lambda item: item.requirement_id)
    ]
    _atomic_write(path.resolve(), _canonical_json(payload))
