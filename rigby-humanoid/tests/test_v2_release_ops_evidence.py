from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigby_v2.release_ops import (
    evaluate_release_checklist,
    seal_release_evidence,
    write_release_evidence_manifest,
)

pytestmark = pytest.mark.fast


def test_sealed_evidence_is_canonical_hash_verified_and_still_partial(
    tmp_path: Path,
) -> None:
    checklist = tmp_path / "checklist.json"
    checklist.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "checklist_version": "test",
                "requirements": [
                    {"id": "first_gate", "description": "first", "required": True},
                    {"id": "second_gate", "description": "second", "required": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence_root = tmp_path / "evidence"
    row = seal_release_evidence(
        evidence_root,
        requirement_id="first_gate",
        passed=True,
        payload={"z": 2, "a": 1},
    )
    manifest = tmp_path / "evidence.json"
    write_release_evidence_manifest(manifest, (row,))

    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    assert parsed == [
        {
            "artifact_path": "first_gate.json",
            "artifact_sha256": row.artifact_sha256,
            "requirement_id": "first_gate",
            "status": "pass",
        }
    ]
    decision = evaluate_release_checklist(
        checklist, (row,), evidence_root=evidence_root
    )
    assert decision.can_ship is False
    assert decision.blockers == ("missing:second_gate",)
    assert decision.verified_evidence["first_gate"] == row.artifact_sha256


def test_evidence_writer_rejects_unsafe_ids_and_duplicate_manifest_rows(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        seal_release_evidence(
            tmp_path, requirement_id="../escape", passed=True, payload={}
        )
    row = seal_release_evidence(
        tmp_path, requirement_id="safe_gate", passed=False, payload={"error": "x"}
    )
    with pytest.raises(ValueError, match="duplicate"):
        write_release_evidence_manifest(tmp_path / "manifest.json", (row, row))
