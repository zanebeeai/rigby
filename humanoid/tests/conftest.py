"""Shared fixtures and the tier machinery.

Plan 09 §3.2 and §3.3. There was no ``conftest.py`` in this repository before
this file, and no fixture of any kind.

Two things here are load-bearing beyond convenience.

**Compiled clips are shared but not shared objects.** ``compile_case`` recompiles
on every call, which is the cost a session fixture removes -- but ``ClipResult``
is a mutable Pydantic model with a mutable ``metrics`` dict, and handing one
instance to every test that asks for a case invites the cross-test contamination
plan 09 §6.1 warns about. So the compile is cached once per session and every
request gets ``model_copy(deep=True)``. ``test_session_fixture_isolation.py``
proves it: it mutates a clip in one test and asserts the next test sees the
original, which is red against a naive session fixture.

**The rig cache is not touched.** ``kinematics.rig_kinematics()`` is
``lru_cache``d on a no-argument function, so the GLB parses once per process.
That is not a performance detail -- it is the one-skeleton-per-process invariant
that PR 04b's asset-digest assertion rests on. Calling ``cache_clear()`` between
tests is the most natural thing in the world for a fixture author to reach for
and it would silently break provenance. Do not add it here without talking to
lane ``anatomy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:  # pragma: no cover - typing only
    from evals.corpus import CorpusCase
    from rigby_poc.models import ClipResult


@pytest.fixture(scope="session")
def rig():
    """The parsed rig. Process-global via ``lru_cache``; never cleared."""

    from rigby_poc.kinematics import rig_kinematics

    return rig_kinematics()


@pytest.fixture(scope="session")
def rig_profile() -> dict:
    from rigby_poc.analysis.rig import rig_profile as _rig_profile

    return _rig_profile()


@pytest.fixture(scope="session")
def corpus() -> list[CorpusCase]:
    """Every corpus case, loaded once. ``load_corpus`` re-validates from disk."""

    from evals.corpus import load_corpus

    return load_corpus()


@pytest.fixture(scope="session")
def corpus_by_id(corpus: list[CorpusCase]) -> dict[str, CorpusCase]:
    return {case.entry.id: case for case in corpus}


@pytest.fixture(scope="session")
def _compiled_cache() -> dict[str, ClipResult]:
    """The shared compile cache. Private: tests take copies, never this."""

    return {}


def _shared_compile(case_id: str, cache: dict, by_id: dict) -> ClipResult:
    """The one place a corpus case is compiled. Returns the shared instance."""

    from evals.corpus import compile_case

    if case_id not in by_id:
        raise KeyError(
            f"{case_id!r} is not a corpus case; known ids include "
            f"{sorted(by_id)[:3]}..."
        )
    if case_id not in cache:
        cache[case_id] = compile_case(by_id[case_id])
    return cache[case_id]


@pytest.fixture(scope="session")
def compile_corpus_case(_compiled_cache, corpus_by_id):
    """Compile a corpus case once per session, handing back a private copy.

    The compile is shared -- that is the saving. The object is not, because
    ``ClipResult`` and its ``metrics`` dict are mutable and a shared instance
    would leak a mutation from one test into every later one.

    Session-scoped so that a ``scope="module"`` fixture can request it. The
    factory copies on every call, so the scope of the factory itself carries no
    isolation meaning -- ``test_session_fixture_isolation.py`` is what proves
    that, and it is why this may be widened without widening the sharing.
    """

    def _compile(case_id: str) -> ClipResult:
        return _shared_compile(case_id, _compiled_cache, corpus_by_id).model_copy(deep=True)

    return _compile


@pytest.fixture(scope="session")
def compile_whole_corpus(_compiled_cache, corpus_by_id):
    """``() -> {case id: ClipResult}`` for the whole corpus, as private copies.

    Nine modules used to open with ``for case in load_corpus(): compile_case(case)``
    in a ``scope="module"`` fixture -- the same 47 compiles, nine times over, for
    about 190s of a 19-minute suite. They share one compile now and still each
    get their own objects, because the callers store the clips alongside derived
    verdicts and some of them mutate.
    """

    def _all() -> dict[str, ClipResult]:
        return {
            case_id: _shared_compile(case_id, _compiled_cache, corpus_by_id).model_copy(deep=True)
            for case_id in corpus_by_id
        }

    return _all


# ---------------------------------------------------------------------------
# Tier timing, reported from the one full run rather than from a second one.
#
# CI used to run `pytest -m fast` and then `pytest`, whose addopts are
# `-m 'fast or medium'` -- so the fast tier, 703 tests, ran twice on every macOS
# job to produce a single duration. These hooks record the same number as a
# by-product of the full run: the summed duration of every `fast`-marked test,
# plus collection, which is what `pytest -m fast` wall-clock was measuring.
#
# Collection is counted deliberately. A module-level corpus compile is paid at
# import time by every invocation, `-m fast` included, and that is exactly the
# regression that took the tier to 131s against a 90s ceiling while every test
# in the file was being deselected. A tier metric that skipped collection would
# have reported all-clear through it.
_TIER = {"fast_seconds": 0.0, "collect_seconds": 0.0}


@pytest.hookimpl(hookwrapper=True)
def pytest_collection(session):
    import time

    start = time.perf_counter()
    yield
    _TIER["collect_seconds"] = time.perf_counter() - start


def pytest_runtest_logreport(report) -> None:
    if "fast" in report.keywords:
        _TIER["fast_seconds"] += report.duration


def pytest_sessionfinish(session, exitstatus) -> None:
    import json
    import os
    import pathlib

    target = os.environ.get("RIGBY_TIER_REPORT")
    if not target:
        return
    payload = {
        "fast_seconds": round(_TIER["fast_seconds"], 1),
        "collect_seconds": round(_TIER["collect_seconds"], 1),
        "fast_tier_seconds": round(_TIER["fast_seconds"] + _TIER["collect_seconds"], 1),
    }
    pathlib.Path(target).write_text(json.dumps(payload), encoding="utf-8")
