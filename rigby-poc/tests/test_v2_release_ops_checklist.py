from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rigby_v2.release_ops import (
    ReleaseEvidence,
    evaluate_release_checklist,
    load_release_checklist,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "docs" / "v2" / "release-checklist.json"


def _complete_evidence(root: Path) -> tuple[ReleaseEvidence, ...]:
    _, requirements = load_release_checklist(CHECKLIST)
    result = []
    for requirement in requirements:
        path = root / f"{requirement.requirement_id}.json"
        path.write_text(json.dumps({"passed": True, "id": requirement.requirement_id}), encoding="utf-8")
        result.append(
            ReleaseEvidence(
                requirement_id=requirement.requirement_id,
                status="pass",
                artifact_path=path.name,
                artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return tuple(result)


def test_machine_readable_checklist_ships_only_with_complete_hash_verified_evidence(
    tmp_path: Path,
) -> None:
    evidence = _complete_evidence(tmp_path)
    decision = evaluate_release_checklist(CHECKLIST, evidence, evidence_root=tmp_path)
    assert decision.can_ship
    assert len(decision.verified_evidence) == len(evidence)
    assert decision.blockers == ()
    assert decision.to_dict()["can_ship"] is True


def test_missing_failed_unknown_or_tampered_evidence_refuses_ship(tmp_path: Path) -> None:
    complete = list(_complete_evidence(tmp_path))
    missing = evaluate_release_checklist(CHECKLIST, tuple(complete[:-1]), evidence_root=tmp_path)
    assert not missing.can_ship
    assert missing.blockers[0].startswith("missing:")

    complete[0] = ReleaseEvidence(
        requirement_id=complete[0].requirement_id,
        status="fail",
        artifact_path=complete[0].artifact_path,
        artifact_sha256=complete[0].artifact_sha256,
    )
    failed = evaluate_release_checklist(CHECKLIST, tuple(complete), evidence_root=tmp_path)
    assert not failed.can_ship
    assert any(value.startswith("failed:") for value in failed.blockers)

    complete = list(_complete_evidence(tmp_path))
    complete[0] = ReleaseEvidence(
        requirement_id=complete[0].requirement_id,
        status="pass",
        artifact_path=complete[0].artifact_path,
        artifact_sha256="0" * 64,
    )
    tampered = evaluate_release_checklist(CHECKLIST, tuple(complete), evidence_root=tmp_path)
    assert any(value.startswith("hash_mismatch:") for value in tampered.blockers)

    unknown = evaluate_release_checklist(
        CHECKLIST,
        tuple(_complete_evidence(tmp_path))
        + (
            ReleaseEvidence(
                requirement_id="manual_override",
                status="pass",
                artifact_path="fresh_machine_preflight.json",
                artifact_sha256=hashlib.sha256(
                    (tmp_path / "fresh_machine_preflight.json").read_bytes()
                ).hexdigest(),
            ),
        ),
        evidence_root=tmp_path,
    )
    assert "unknown_evidence:manual_override" in unknown.blockers
    assert not unknown.can_ship


def test_path_escaping_evidence_is_rejected(tmp_path: Path) -> None:
    complete = list(_complete_evidence(tmp_path))
    complete[0] = ReleaseEvidence(
        requirement_id=complete[0].requirement_id,
        status="pass",
        artifact_path="../outside.json",
        artifact_sha256=complete[0].artifact_sha256,
    )
    decision = evaluate_release_checklist(CHECKLIST, tuple(complete), evidence_root=tmp_path)
    assert any(value.startswith("unsafe_evidence:") for value in decision.blockers)
    assert not decision.can_ship
