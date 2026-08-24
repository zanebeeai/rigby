"""The autonomous goal audit must run on a clean clone.

Before 03b it read
``results/judge-calibration/005673-v4-pronation/run-02-truthful-joints`` -- a path
from the original author's machine, holding data that never shipped and cannot be
reconstructed.  On any other checkout ``_read`` returned ``None``, the audit turned
that into an empty dict, and reported "automated corruption calibration is not
passing".  So an audit that could not run at all was indistinguishable from one that
ran and failed, which is why plan 10 section 9.3 lists the clean-clone audit as
still outstanding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.autonomous_goal_audit import COMMITTED_EVIDENCE, build_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _judge_gate(audit: dict[str, object]) -> dict[str, object]:
    gates = audit["gates"]
    assert isinstance(gates, list)
    gate = gates[0]
    assert isinstance(gate, dict)
    assert gate["gate"] == "calibrated_autonomous_judge"
    return gate


def test_the_committed_evidence_exists_and_is_tracked() -> None:
    """The whole repoint rests on this file being in the repository."""
    path = PROJECT_ROOT / COMMITTED_EVIDENCE
    assert path.is_file(), path
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["corruption_suite"]["passed"] is True
    assert evidence["human_calibration"]["passed"] is True


def test_the_judge_gate_passes_from_committed_evidence_alone() -> None:
    """No ``results/`` directory, no machine-local path: the gate still resolves."""
    gate = _judge_gate(build_audit(PROJECT_ROOT))
    assert gate["status"] == "pass", gate["failures"]
    measured = gate["measured"]
    assert isinstance(measured, dict)
    assert measured["vlm_human_agreement_rate"] == 0.9
    assert measured["automated_false_accepts"] == 0
    assert measured["human_pair_count"] == 10


def test_the_audit_reads_no_path_under_results_for_the_judge_gate(
    tmp_path: Path,
) -> None:
    """A project root with an empty ``results/`` must not change the judge gate.

    This is what makes the claim "runnable on a clean clone" a test rather than an
    assertion: the gate is computed from a path inside the repository, so pointing
    the audit at a tree with no run data leaves it untouched.
    """
    (tmp_path / "results").mkdir()
    gate = _judge_gate(
        build_audit(tmp_path, evidence_path=PROJECT_ROOT / COMMITTED_EVIDENCE)
    )
    assert gate["status"] == "pass", gate["failures"]


def test_missing_evidence_is_reported_as_missing_not_as_failing(tmp_path: Path) -> None:
    """The distinction the old code collapsed, restored.

    "The evidence is not there" and "the evidence says the judge failed" need
    different fixes, so they must not produce the same message.
    """
    gate = _judge_gate(build_audit(tmp_path, evidence_path=tmp_path / "absent.json"))
    assert gate["status"] == "fail"
    failures = gate["failures"]
    assert isinstance(failures, list)
    assert len(failures) == 1
    assert "is missing at" in failures[0]
    assert "not passing" not in failures[0]


def test_evidence_that_says_it_failed_is_reported_as_failing(tmp_path: Path) -> None:
    evidence = json.loads(
        (PROJECT_ROOT / COMMITTED_EVIDENCE).read_text(encoding="utf-8")
    )
    evidence["human_calibration"]["passed"] = False
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    gate = _judge_gate(build_audit(tmp_path, evidence_path=path))
    assert gate["status"] == "fail"
    assert gate["failures"] == ["frozen final human calibration is not passing"]


@pytest.mark.parametrize("key", ("corruption_suite", "human_calibration"))
def test_each_half_of_the_evidence_is_load_bearing(tmp_path: Path, key: str) -> None:
    evidence = json.loads(
        (PROJECT_ROOT / COMMITTED_EVIDENCE).read_text(encoding="utf-8")
    )
    del evidence[key]
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    gate = _judge_gate(build_audit(tmp_path, evidence_path=path))
    assert gate["status"] == "fail"


def test_the_repository_holds_no_reference_to_the_retired_machine_local_path() -> None:
    """Guards the repoint against being partially reverted.

    ``evals/goal_audit.py`` still defaults to the same directory, but every one of
    its callers passes an explicit path, so it is a dead default rather than a live
    read.  It is left to plan 10, which rewrites that module; naming it here means
    the next person finds it deliberately rather than by accident.
    """
    lines = (
        (PROJECT_ROOT / "evals/autonomous_goal_audit.py")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    # The retired path is still named in the comment on COMMITTED_EVIDENCE, which is
    # where it belongs -- a reader has to be able to see what was replaced.  What
    # must not come back is a live read of it.
    code = [line for line in lines if not line.lstrip().startswith("#")]
    offenders = [line.strip() for line in code if "005673-v4-pronation" in line]
    assert not offenders, offenders
