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

import pathlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.medium

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Ceilings, not targets.  Each is the measured count plus a little room, so a
#: refactor that shifts work between tests does not fail while a new corpus-wide
#: loop does.  Raising one is a decision that should appear in a diff with a reason.
#:
#: Measuring a file costs a subprocess pytest session, so not every corpus-touching
#: file is measured -- but every one is **declared**, here or in
#: :data:`UNBUDGETED`, and ``test_every_corpus_touching_file_is_declared`` fails on
#: a new one.  A name list nobody maintains goes stale invisibly, which is the exact
#: failure lane `capture` hit when this corpus grew 12 -> 47 and their snapshot bound
#: went stale with nothing connecting the two.
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
    # One corpus-wide pass in a module fixture, to pin the emitted check-id registry
    # against what the analyzer actually emits. Measured at 55: the 47-case pass plus
    # the loader revalidating a handful. Budgeted rather than exempted because it is
    # exactly the corpus-wide loop this file exists to keep visible -- 06a's target
    # ids went unchecked for a whole PR, and the fix for that must not itself be the
    # thing nobody is counting.
    "test_mutation_check_registry.py": 60,
    # Two vacuous-metric pins that need the object-interaction path's own metrics,
    # which the committed clips do not carry -- only hashes. Measured at 22: nine
    # object cases plus the file's thirteen synthetic single-program compiles.
    # Scoped to the object cases rather than the corpus: the first version looped
    # all 47 to reach nine, and this guard caught it.
    "test_vacuous_metrics.py": 30,
    # 10d's detection curve against real compiled clips. Measured at 4: one
    # module fixture compiling four named full-body cases, shared across five
    # tests. Scoped to four rather than looping the corpus because the sweeps
    # are per `(bone, dof)` and four moving cases exercise every branch --
    # threshold point, static target and refusal. Budgeted rather than exempted
    # so that scoping stays a decision somebody has to re-make in a diff if this
    # file ever reaches for `load_corpus()` in a loop.
    "test_detection_curve_on_the_corpus.py": 8,
}

#: Corpus-touching files deliberately not measured, with the reason.  Being here is
#: a decision, not an oversight -- which is the whole difference between this and an
#: undeclared file.
UNBUDGETED: dict[str, str] = {
    "conftest.py": "the session fixture itself; its compiles are attributed to callers",
    "corpus_offline_probe.py": "a helper run in a subprocess by the offline test, not collected",
    "test_analysis_equivalence.py": "lane analysis owns its budget; ours would constrain their PRs",
    "test_anatomical_frame.py": "lane anatomy",
    "test_rom_detects_injected_violation.py": "lane anatomy",
    "test_rom_enforcement.py": "lane anatomy",
    "test_rig_provenance.py": "lane anatomy",
    "test_rom_table.py": "lane anatomy",
    "test_capture_sampling_bound.py": "lane capture; one corpus pass, measured at 41 by them",
    "test_session_fixture_isolation.py": "proves the fixture copies; compiles are the subject",
    "test_corpus_loads_offline.py": "runs its own subprocess probe; counting here would double-count",
    "test_mutation_contract.py": "two module-scoped compiles",
    "test_mutation_determinism.py": "one, plus two subprocesses",
    "test_mutation_injector.py": "two module-scoped fixtures",
    "test_mutation_sweep.py": "one",
    "test_mutation_legacy_port.py": "one corpus pass for the applicability tally",
    "test_mutation_families.py": "three module-scoped compiles; measured at 3",
    "corpus_seed.py": (
        "a helper, not a test file; it compiles nothing and imports nothing from "
        "evals.corpus -- it matches only because its docstring names `load_corpus` "
        "while explaining the validation failure it exists to prevent"
    ),
}

#: What marks a file as touching the corpus.  Deliberately broad: a false positive
#: costs one line in ``UNBUDGETED``, a false negative costs an invisible regression.
#: ``evals.corpus`` catches the CLI path -- ``test_corpus_cli.py`` compiles through
#: ``python -m evals.corpus verify`` and names none of the functions directly.
CORPUS_MARKERS = (
    "load_corpus",
    "compile_case",
    "compile_corpus_case",
    "evals.corpus",
)


def test_every_corpus_touching_file_is_declared() -> None:
    """A new file that compiles the corpus must be budgeted or exempted by name.

    Static and cheap -- no subprocess -- so it can cover every file while the
    measurement covers a subset.  The point is that a corpus-wide loop cannot arrive
    in a file nobody listed, which is how ``test_corpus_coverage`` reached 94
    compiles unnoticed.
    """
    tests = pathlib.Path(__file__).parent
    touching = {
        path.name
        for path in sorted(tests.glob("*.py"))
        if any(marker in path.read_text(encoding="utf-8") for marker in CORPUS_MARKERS)
        and path.name != pathlib.Path(__file__).name
    }
    declared = set(COMPILE_BUDGET) | set(UNBUDGETED)
    undeclared = sorted(touching - declared)
    assert not undeclared, (
        f"these files compile the corpus and are neither budgeted nor exempted: "
        f"{undeclared}. Add a ceiling to COMPILE_BUDGET, or an entry to UNBUDGETED "
        f"saying why not. A full corpus recompile is 47 compiles."
    )
    # Only ``UNBUDGETED`` can go stale invisibly: a budgeted file is measured, so a
    # file that stopped compiling would show up as a count of zero and fail the
    # `ran > 0` assertion rather than sit here unnoticed.
    stale = sorted(set(UNBUDGETED) - touching)
    assert not stale, (
        f"exempted but no longer touching the corpus: {stale}. Remove the entry, or "
        f"the exemption outlives the reason for it."
    )


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
