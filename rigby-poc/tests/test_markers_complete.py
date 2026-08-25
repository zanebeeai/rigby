"""Every collected test file declares a tier, and no file declares two.

Plan 09 §3.3. Tiering rots within a month without this: one unmarked file joins
the default run silently, then ten, and `-m fast` stops meaning "the fast ones"
and starts meaning "the ones somebody remembered to mark".

**Keyed on pytest's own collection patterns, not a directory glob** (lane
`capture`, 2026-08-24). ``python_files = ["test_*.py", "backend_*.py"]``, so
``tests/`` also holds files pytest never collects -- ``capture_e2e_server.py``
and ``corpus_offline_probe.py``, which contain no tests and must carry no marker.
A ``tests/*.py`` glob flags both, and a guard with two false positives on day one
acquires an allowlist and stops being read. Same failure as the
module-reachability guard at 19 flags and the hardcoded-threshold guard at 79.
"""

from __future__ import annotations

import ast
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS = PROJECT_ROOT / "tests"
TIERS = ("fast", "medium", "slow")


def _collection_patterns() -> list[str]:
    """The `python_files` patterns pytest itself collects by."""

    configured = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return configured["tool"]["pytest"]["ini_options"]["python_files"]


def collected_files() -> list[Path]:
    """Every file pytest would collect.

    Callers assert over this list, so an empty one must be an error rather than
    a vacuous pass: `all(...)` over nothing is True, and a moved directory or a
    renamed package would otherwise turn the whole tier gate green. The
    collection is the evidence, and the evidence being absent is one of the
    failures this file exists to detect. Rule from lane `judge`.
    """

    patterns = _collection_patterns()
    found = [
        path
        for path in sorted(TESTS.rglob("*.py"))
        if any(fnmatch(path.name, pattern) for pattern in patterns)
        and "__pycache__" not in path.parts
    ]
    assert len(found) > 50, (
        f"only {len(found)} collected test files found under {TESTS}; the scan is "
        f"broken, not the suite. An assertion over an empty collection passes."
    )
    return found


def declared_tiers(source: str) -> set[str]:
    """Tier markers named by a **module-level** ``pytestmark`` assignment.

    ``tree.body`` rather than ``ast.walk``: pytest honours ``pytestmark`` only at
    module scope, so an assignment inside a function or a class does nothing at
    runtime. Walking the whole tree accepted those, which meant the gate could
    certify a file as tiered while pytest treated it as untiered -- the gate
    passing on a property the runner does not have. Found by lane ``groundtruth``
    asking whether placement mattered, before it cost anyone anything.
    """

    found: set[str] = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            continue
        for inner in ast.walk(node.value):
            if isinstance(inner, ast.Attribute) and inner.attr in TIERS:
                found.add(inner.attr)
    return found


def test_the_registered_markers_are_exactly_the_tiers() -> None:
    """A marker registered but not a tier, or a tier not registered, rots this."""

    configured = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    registered = {
        entry.split(":", 1)[0].strip()
        for entry in configured["tool"]["pytest"]["ini_options"]["markers"]
    }
    assert registered == set(TIERS), f"registered markers {sorted(registered)} != {list(TIERS)}"


def test_every_collected_file_declares_a_tier() -> None:
    """The gate."""

    unmarked = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in collected_files()
        if not declared_tiers(path.read_text(encoding="utf-8"))
    ]
    assert not unmarked, (
        "these collected test files declare no tier:\n"
        + "\n".join(f"  {name}" for name in unmarked)
        + "\nAdd `pytestmark = pytest.mark.<fast|medium|slow>` at module scope. "
        "See docs/testing.md for what each tier means."
    )


def test_no_file_declares_two_tiers() -> None:
    """Two tiers on one file means the file runs in both and belongs in neither."""

    conflicted = {
        path.relative_to(PROJECT_ROOT).as_posix(): sorted(tiers)
        for path in collected_files()
        if len(tiers := declared_tiers(path.read_text(encoding="utf-8"))) > 1
    }
    assert not conflicted, f"files declaring more than one tier: {conflicted}"


def test_uncollected_helpers_are_not_required_to_be_marked() -> None:
    """The exemption is earned by pytest's own patterns, not by a hand-list.

    If either helper is ever renamed to match a collection pattern it starts
    being collected, and the guard above should start requiring a tier -- so this
    asserts the mechanism rather than the names.
    """

    collected = {path.name for path in collected_files()}
    helpers = [
        path
        for path in sorted(TESTS.rglob("*.py"))
        if path.name not in collected and "__pycache__" not in path.parts
    ]
    for helper in helpers:
        assert not any(
            fnmatch(helper.name, pattern) for pattern in _collection_patterns()
        ), f"{helper.name} matches a collection pattern but was treated as a helper"


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import pytest\npytestmark = pytest.mark.fast\n", {"fast"}),
        ("import pytest\npytestmark = pytest.mark.medium\n", {"medium"}),
        ("import pytest\npytestmark = [pytest.mark.slow]\n", {"slow"}),
        ("import pytest\npytestmark = pytest.mark.parametrize('x', [1])\n", set()),
        ("x = 1\n", set()),
    ],
)
def test_the_tier_detector(source: str, expected: set[str]) -> None:
    assert declared_tiers(source) == expected


def test_a_function_level_marker_does_not_count_as_a_file_tier() -> None:
    """Tiering is per file. A single marked test does not tier its module."""

    source = "import pytest\n\n@pytest.mark.slow\ndef test_x():\n    pass\n"
    assert declared_tiers(source) == set()


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\ndef f():\n    pytestmark = pytest.mark.fast\n",
        "import pytest\nclass C:\n    pytestmark = pytest.mark.fast\n",
        "import pytest\nif True:\n    pytestmark = pytest.mark.fast\n",
    ],
)
def test_a_pytestmark_that_is_not_module_level_does_not_tier_the_file(source: str) -> None:
    """pytest ignores these, so the gate must too.

    Accepting them would let a file pass the gate while the runner treats it as
    untiered -- the gate certifying a property the thing it guards does not have.
    """

    assert declared_tiers(source) == set()
