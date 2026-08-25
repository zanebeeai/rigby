"""A compile budget for the corpus test files, counted rather than timed.

**Why a count and not a duration.**  Timing on a shared machine is not a
measurement of the code.  Lane `infra` timed one unchanged tier five times and got
25.6, 33.5, 43.6, 44.9 and 42.6 seconds -- nineteen seconds of spread driven by how
many worktrees were compiling.  Measured here on the same day: the same corpus test
file took 21.4 s once and 41.8 s and 45.2 s an hour later, unchanged.  A timing
assertion on this suite would be a coin flip.

A **compile count** is invariant to load, to platform, and to CPU count.  It is also
the quantity that actually matters: a full corpus recompile is 47 compiles, and the
regression this file exists to catch is somebody answering a question with a loop
over `compile_case` when the committed clips already hold the answer.

That regression is not hypothetical.  ``test_corpus_coverage.py`` shipped in 03b
asking which bones each case moves, twice, for all 47 cases -- **94 compiles** to
answer a question the stored frames already answered, and it was found by the suite
crawling rather than by any test.  Reading the committed clip took that file to zero
compiles.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.medium

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Ceilings, not targets.  Each is the measured count plus a little room, so a
#: refactor that shifts work between tests does not fail while a new corpus-wide
#: loop does.  Raising one is a decision that should appear in a diff with a reason.
COMPILE_BUDGET: dict[str, int] = {
    # 47 for the determinism sweep, plus the idempotence test's deliberate second
    # compile and the seed-invariance pair.  Was 130 before the shared compile: the
    # solver-declaration and slim-clip tests each recompiled the same cases.
    "test_corpus_determinism.py": 75,
    # Reads the committed clips.  Zero is the point of it.
    "test_corpus_coverage.py": 0,
    # One module-level compile per case, shared across ~200 parametrised tests.
    # 47 module-level, plus the loader validating a handful of cases twice.
    "test_corpus_known_bad.py": 60,
    "test_corpus_format.py": 30,
    "test_corpus_cli.py": 60,
}

_COUNTER = """
import sys
import rigby_poc.compiler as compiler
import evals.corpus.loader as loader

calls = {"n": 0}
real = compiler.compile_motion


def counting(*args, **kwargs):
    calls["n"] += 1
    return real(*args, **kwargs)


# `loader` imported the symbol by value, so both bindings need replacing.
compiler.compile_motion = counting
loader.compile_motion = counting

import pytest


class _Ran:
    # Counts tests that actually executed, not tests that were collected.
    def __init__(self):
        self.n = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.n += 1


ran = _Ran()
code = pytest.main(["-q", "-p", "no:warnings", "--no-header", sys.argv[1]], plugins=[ran])
print(f"COMPILES={calls['n']} RAN={ran.n} EXIT={int(code)}")
"""


def _count_compiles(test_file: str) -> tuple[int, int, int]:
    completed = subprocess.run(
        [sys.executable, "-c", _COUNTER, f"tests/{test_file}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    marker = [line for line in completed.stdout.splitlines() if line.startswith("COMPILES=")]
    assert marker, completed.stdout[-2000:] + completed.stderr[-2000:]
    compiles, ran, exit_code = marker[-1].split()
    return (
        int(compiles.removeprefix("COMPILES=")),
        int(ran.removeprefix("RAN=")),
        int(exit_code.removeprefix("EXIT=")),
    )


@pytest.mark.parametrize("test_file", sorted(COMPILE_BUDGET))
def test_a_corpus_test_file_stays_within_its_compile_budget(test_file: str) -> None:
    compiles, ran, exit_code = _count_compiles(test_file)
    assert exit_code == 0, f"{test_file} did not pass while being counted"
    # A budget met by running nothing is a guard that cannot fail, and the child
    # inherits `addopts` -- currently `-m 'fast or medium'` -- so a file retiered to
    # `slow` would be deselected entirely and report zero compiles against every
    # ceiling.
    #
    # Measured: pytest exits **5** on that path, so the exit assertion above already
    # catches today's version of it. (An earlier comment here claimed it exits 0,
    # from a shell pipeline that read `tail`'s status rather than pytest's.) This
    # assertion is the direct statement of the property rather than a consequence
    # of pytest's exit-code convention, which is not ours to depend on.
    assert ran > 0, (
        f"{test_file} ran no tests while being counted, so its zero compiles mean "
        f"nothing. The child inherits addopts (-m 'fast or medium'); check the "
        f"file's tier marker."
    )
    budget = COMPILE_BUDGET[test_file]
    assert compiles <= budget, (
        f"{test_file} performs {compiles} compiles against a budget of {budget}. "
        f"A full corpus recompile is 47. If this is a new corpus-wide loop, the "
        f"committed clip.slim.json.gz probably already holds the answer -- see "
        f"tests/test_corpus_coverage.py::_bones_moved. If the work is genuinely "
        f"needed, raise the budget here and say why."
    )
