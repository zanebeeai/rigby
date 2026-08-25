"""No new hardcoded threshold in check code, and the existing count only falls.

Plan 08 §3.5, G4. A threshold that lives as a literal in a comparison has no
source, no unit, and no way to be found -- which is how the repository arrived
at three config files and ~20 code sites disagreeing with each other.

**Defining "threshold" precisely is what makes this shippable.** The scanned tree
holds roughly 250 float literals, most of them numerical-stability epsilons,
unit-vector components and normalisation constants. A guard that fires on all of
them is disabled inside a week. The tractable definition is AST-precise: a
threshold is a float literal appearing as an **operand of a comparison**.
``if x > 0.35`` is a gate; ``value * 0.5`` is arithmetic.

That yields 79 sites, and inspection says essentially all of them are genuine
gates. 79 is far too many to repoint in one PR, so this is a **per-file budget**
that may only ever fall -- the same ledger shape as
``test_no_orphan_thresholds.py``. It converts "we should centralise thresholds"
from an aspiration into a monotonically decreasing, enforced work list.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT_ROOT / "src" / "rigby_poc"

#: Scanned trees. `compiler.py` is here deliberately, and it is the difference
#: between a guard that means what it says and one that fires on non-events.
#:
#: The stated meaning is "no new hardcoded threshold in check code". While plan
#: 02 is still moving check code OUT of the compiler, `analysis/` alone is not
#: that set: 02c part 2 moved five HANG_TEN literals from `compiler.py` into
#: `analysis/hand.py`, and an analysis-only scan reported +5 for a relocation
#: that added nothing. Every remaining 02x move would have done the same.
#:
#: That is the dual of the class this file exists to catch. A gate that cannot
#: fire and a gate that fires on non-events are one defect -- the assertion's
#: scope not matching its meaning -- and the second is the more dangerous here,
#: because a guard that reddens main for non-events is the one that gets an
#: allowlist. Scanning both makes a relocation net-zero by construction and lets
#: the total fall monotonically as the moves and the repointing land.
#:
#: Cost: a bigger honest number on day one. Raised by lane `analysis`, who
#: proved the relocation with this file's own counter rather than a second
#: implementation, and asked rather than editing the ledger.
SCANNED = (PACKAGE / "analysis", PACKAGE / "compiler.py")

#: Below this magnitude a literal is float noise, not a gate.
EPSILON_FLOOR = 1e-7

#: Identity values that are never thresholds in a meaningful sense.
IDENTITY_VALUES = frozenset({0.0, 1.0})

#: Hardcoded comparison thresholds per file, recorded 2026-08-24 on `76b2a47`.
#: These are pre-existing. **A number here may only ever go DOWN.** Repointing a
#: site onto config/thresholds.v1.json lowers it; adding a literal raises it and
#: fails. A file absent from this map may have none at all.
#:
#: The concentration in `full_body/failures.py` is not an accident of style: 02b
#: moved the whole-body metric pass out of the compiler, and it arrived carrying
#: 41 uncited gates. Every one is a real limit with no source and no unit.
BUDGET: dict[str, int] = {
    "compiler.py": 21,
    "analysis/anatomy/frame.py": 1,
    "analysis/contact.py": 1,
    "analysis/forearm.py": 11,
    "analysis/full_body/balance.py": 1,
    "analysis/full_body/climb.py": 2,
    "analysis/full_body/exercises.py": 7,
    "analysis/full_body/failures.py": 41,
    "analysis/hand.py": 5,
    "analysis/full_body/ground.py": 2,
    "analysis/full_body/posture.py": 4,
    "analysis/full_body/rotation.py": 2,
    "analysis/full_body/selectors.py": 2,
    "analysis/gesture.py": 4,
    "analysis/semantic.py": 1,
}

TOTAL_BUDGET = 105


def comparison_thresholds(source: str) -> list[tuple[int, float]]:
    """Float literals used as an operand of a comparison, with their line."""

    found: list[tuple[int, float]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        for operand in (node.left, *node.comparators):
            if not isinstance(operand, ast.Constant):
                continue
            value = operand.value
            if not isinstance(value, float) or isinstance(value, bool):
                continue
            if abs(value) < EPSILON_FLOOR or value in IDENTITY_VALUES:
                continue
            found.append((operand.lineno, value))
    return found


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in SCANNED:
        files.extend(sorted(root.rglob("*.py")) if root.is_dir() else [root])
    assert len(files) > 10, (
        f"only {len(files)} files found under {SCANNED}; the scan is broken, not "
        f"the code. A budget compared against an empty measurement always passes."
    )
    return files


def _measured() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in _scanned_files():
        found = comparison_thresholds(path.read_text(encoding="utf-8"))
        if found:
            name = path.relative_to(PROJECT_ROOT / "src" / "rigby_poc").as_posix()
            counts[name] = len(found)
    return counts


def test_no_file_gains_a_hardcoded_threshold() -> None:
    """The gate. A new comparison literal in check code fails the suite."""

    measured = _measured()
    over = {
        name: (count, BUDGET.get(name, 0))
        for name, count in measured.items()
        if count > BUDGET.get(name, 0)
    }
    assert not over, (
        "hardcoded comparison thresholds increased:\n"
        + "\n".join(f"  {n}: {c} > budget {b}" for n, (c, b) in sorted(over.items()))
        + "\nRead the limit from config/thresholds.v1.json instead. If it genuinely "
        "has no config home, add one -- that is the point of plan 08."
    )


def test_the_total_budget_only_falls() -> None:
    assert sum(_measured().values()) <= TOTAL_BUDGET, (
        f"total hardcoded thresholds rose above {TOTAL_BUDGET}"
    )


def test_a_repointed_file_lowers_its_budget() -> None:
    """Leaving a stale budget hides that the work was done, and re-opens room."""

    measured = _measured()
    stale = {
        name: (measured.get(name, 0), budget)
        for name, budget in BUDGET.items()
        if measured.get(name, 0) < budget
    }
    assert not stale, (
        "these files now hold fewer hardcoded thresholds than their budget:\n"
        + "\n".join(f"  {n}: {c} < budget {b}" for n, (c, b) in sorted(stale.items()))
        + "\nLower the budget to the measured value, or the slack becomes room for a "
        "new one."
    )


def test_budget_names_only_files_that_exist() -> None:
    for name in BUDGET:
        assert (PROJECT_ROOT / "src" / "rigby_poc" / name).exists(), (
            f"{name} is budgeted but does not exist; the budget is stale"
        )


# --- the classifier, which is where this guard is easy to get wrong ---


@pytest.mark.parametrize(
    "source",
    [
        "x = value * 0.5",
        "axis = [0.0, 1.0, 0.0]",
        "scaled = raw / 0.35",
        "def f(threshold=0.25): pass",
        "total = sum(v * 0.28 for v in values)",
    ],
)
def test_arithmetic_is_not_a_threshold(source: str) -> None:
    assert comparison_thresholds(source) == []


@pytest.mark.parametrize(
    "source, expected",
    [
        ("if x > 0.35: pass", [0.35]),
        ("if 1.01 <= y <= 1.48: pass", [1.01, 1.48]),
        ("flag = depth <= 0.015", [0.015]),
        ("n = int(clearance > 0.028)", [0.028]),
    ],
)
def test_a_comparison_operand_is_a_threshold(source: str, expected: list[float]) -> None:
    assert [value for _, value in comparison_thresholds(source)] == expected


def test_stability_epsilons_are_not_thresholds() -> None:
    """`if norm < 1e-8` guards a division; it is not a gate on the motion."""

    assert comparison_thresholds("if norm < 1e-8: return") == []
    assert comparison_thresholds("if abs(d) < 1e-12: pass") == []


def test_identity_comparisons_are_not_thresholds() -> None:
    assert comparison_thresholds("if fraction > 0.0: pass") == []
    assert comparison_thresholds("if fraction >= 1.0: pass") == []
