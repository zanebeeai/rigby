"""Every published acceptance dimension must actually gate, at every call site.

`semantic_match` was documented as part of the release threshold and never gated
(plan 07 §1.1). It was not the only one of its shape: `foot_drift_m` is compared
against a threshold it can never breach because the value is a literal, and
`analysis/safety.py` enforces asymmetric joint limits as if symmetric, so a bend
in the impossible direction passes. Three instances is a class, not a
coincidence.

The weak version of this guard asks "is the criterion read anywhere". That is not
enough — `foot_drift_m` IS read. The property that matters is **can it actually
fail**: for every published dimension, construct input that breaches it and
assert the gate fires. Anything less would have passed all three bugs.

The second half matters as much as the first: the rule is enforced in more than
one place, and a site that hardcodes its own copy of the dimension list will
silently stop matching the published criteria.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evals.autonomous_goal_audit import _audit_run
from rigby_poc.judge import (
    ACCEPTANCE_MINIMUM_SCORES,
    MotionJudgeScore,
    VLMJudge,
    assemble_split_score,
    meets_acceptance_thresholds,
)
from rigby_poc.judge_claims import claim_specs
from rigby_poc.judge_prompts import GRADER_NAMES, GRADER_SPECS

pytestmark = pytest.mark.medium


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRITERIA = json.loads((PROJECT_ROOT / "acceptance_criteria.yaml").read_text(encoding="utf-8"))
PUBLISHED = CRITERIA["autonomous_pipeline"]["minimum_selected_scores"]

#: Every place in the repo that decides, or claims to decide, acceptance.
#: Adding a site without adding it here is what this file exists to prevent.
ACCEPTANCE_SITES = (
    "rigby_poc.judge.meets_acceptance_thresholds",
    "rigby_poc.judge.assemble_split_score",
    "rigby_poc.judge.VLMJudge._unary_escalation",
    "evals.autonomous_goal_audit._audit_run",
)


def _passing_score(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "semantic_match": 5,
        "gesture_recognizability": 5,
        "anatomical_naturalness": 5,
        "temporal_readability": 5,
        "egocentric_visibility": 5,
        "cross_view_consistency": 5,
        "overall": 5,
        "accept": True,
        "confidence": 0.95,
        "failure_tags": ["none"],
        "evidence": [
            {"snapshot_id": "01-ego", "observation": "Hand visible."},
            {"snapshot_id": "01-orbit", "observation": "Arm natural."},
        ],
        "summary": "A readable and structurally sound motion.",
        "suggested_adjustment": "No adjustment needed.",
    }
    payload.update(overrides)
    return payload


def _run(**score_overrides: object) -> dict[str, object]:
    candidate = {
        "result_id": "winner",
        "judgment": _passing_score(**score_overrides),
        "compile_success": True,
        "structural_valid": True,
        "evidence_manifest": "",
    }
    return {
        "run_id": "run-1",
        "status": "completed",
        "winner_result_id": "winner",
        "trace": {
            "status": "winner_selected",
            "planner": {"program": {"intent": "gesture"}},
            "rounds": [{"candidates": [candidate] + [dict(candidate, result_id=f"c{i}") for i in range(4)]}],
        },
    }


# ------------------------------------------- the published set is the contract


def test_the_judge_rule_mirrors_the_published_criteria_exactly() -> None:
    assert ACCEPTANCE_MINIMUM_SCORES == PUBLISHED


def test_the_audit_checks_exactly_the_published_dimensions() -> None:
    """The audit hardcodes its dimension list in a literal tuple.

    That is the §1.1 bug shape waiting to recur: a fifth published dimension
    would leave the audit quietly checking four. Until it reads the criteria
    file, this test is what keeps the two in step.
    """
    source = (PROJECT_ROOT / "evals" / "autonomous_goal_audit.py").read_text(encoding="utf-8")
    listed = re.search(r'for key in \(([^)]*)\):', source)
    assert listed is not None, "the audit no longer iterates a literal dimension tuple"
    checked = set(re.findall(r'"([a-z_]+)"', listed.group(1)))
    assert checked == set(PUBLISHED)


# ------------------------------------ every dimension can actually fail, per site


@pytest.mark.parametrize("dimension", sorted(PUBLISHED))
def test_the_pure_rule_rejects_each_published_dimension(dimension: str) -> None:
    below = PUBLISHED[dimension] - 1
    assert meets_acceptance_thresholds(_passing_score(**{dimension: below})) is False


@pytest.mark.parametrize("dimension", sorted(PUBLISHED))
def test_the_audit_reports_each_published_dimension(dimension: str) -> None:
    below = PUBLISHED[dimension] - 1
    report = _audit_run(PROJECT_ROOT, _run(**{dimension: below}), "gesture")
    assert f"winner {dimension} is below {PUBLISHED[dimension]}" in report["failures"]


@pytest.mark.parametrize("dimension", sorted(PUBLISHED))
def test_a_self_accepted_breach_is_escalated_rather_than_waved_through(dimension: str) -> None:
    judge = VLMJudge(client=object(), model="test-model")
    score = MotionJudgeScore.model_validate(
        _passing_score(**{dimension: PUBLISHED[dimension] - 1, "accept": True})
    )
    assert judge._unary_escalation(score) == "acceptance_score_inconsistency"


@pytest.mark.parametrize("dimension", sorted(set(PUBLISHED) - {"overall"}))
def test_the_split_path_rejects_each_published_dimension(dimension: str) -> None:
    """`overall` is derived on the split path, so it is exercised through its sources."""
    grader = next(
        name for name in GRADER_NAMES if dimension in GRADER_SPECS[name].dimensions
    )
    parts = {
        name: {
            "claims": [
                {
                    "id": spec.id,
                    "verdict": "no" if name == grader and spec.critical else "yes",
                    "confidence": 0.9,
                    "snapshot_id": "01-ego",
                }
                for spec in claim_specs(name, intent="gesture")
            ],
            "summary": f"{name} summary.",
        }
        for name in GRADER_NAMES
    }
    parts["semantic"]["suggested_adjustment"] = "Preserve the pose."
    score, verdicts = assemble_split_score(parts, intent="gesture")
    assert verdicts[dimension].score < PUBLISHED[dimension]
    assert score.accept is False


# --------------------------------------------- no unreviewed acceptance consumer


def test_every_consumer_of_the_accept_flag_is_a_known_one() -> None:
    """`accept` is copied verbatim by five call sites (plan 07 §1.1).

    A new one appearing without review is how the decision layer 07d installs
    gets bypassed. This fails on a new copier so it is a deliberate decision,
    not a diff nobody read.
    """
    known = {
        "evals/flywheel.py",
        "evals/rerank_existing.py",
        "evals/select_structural_sweep.py",
        "evals/calibrate_judge.py",
        "evals/autonomous_goal_audit.py",
    }
    found: set[str] = set()
    for path in sorted((PROJECT_ROOT / "evals").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r'\["accept"\]|\.get\("accept"\)|\.accept\b', text):
            # `as_posix()`, not `str()`: on Windows `relative_to` yields
            # backslashes and every entry in `known` would look unrecognised,
            # so the guard would fail on the platform rather than on a finding.
            found.add(path.relative_to(PROJECT_ROOT).as_posix())
    # This guard fails OPEN on an empty scan: `set() <= known` is True and
    # `all(...)` over nothing is True, so a glob that stops matching -- a moved
    # directory, a renamed package -- reports a clean result rather than a broken
    # one. Assert the scan found something before believing what it found.
    assert found, "the accept-flag scan matched no files; the guard is not running"
    # Pinned so the Windows fix cannot regress unnoticed on a POSIX-only run:
    # `str(relative_to(...))` yields backslashes there and every entry would look
    # unrecognised, failing the guard on the platform rather than on a finding.
    assert all("\\" not in name and "/" in name for name in found), sorted(found)
    assert found <= known, f"unreviewed consumer of the accept flag: {sorted(found - known)}"


def test_the_gate_is_enforced_in_every_site_this_file_knows_about() -> None:
    """A site added to the codebase but not to this file is the failure mode.

    This asserts the inventory is complete as of now; it cannot detect a site
    nobody told it about, which is why the consumer scan above exists too.
    """
    import importlib

    assert len(ACCEPTANCE_SITES) == 4
    for site in ACCEPTANCE_SITES:
        parts = site.split(".")
        # Walk down from the longest importable module prefix, then through
        # attributes, so a class-scoped site resolves the same way as a module one.
        module = None
        rest: list[str] = []
        for index in range(len(parts), 0, -1):
            try:
                module = importlib.import_module(".".join(parts[:index]))
            except ImportError:
                continue
            rest = parts[index:]
            break
        assert module is not None, site
        target = module
        for part in rest:
            target = getattr(target, part)
        assert callable(target), site
