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


@pytest.fixture
def compile_corpus_case(_compiled_cache, corpus_by_id):
    """Compile a corpus case once per session, handing back a private copy.

    The compile is shared -- that is the saving. The object is not, because
    ``ClipResult`` and its ``metrics`` dict are mutable and a shared instance
    would leak a mutation from one test into every later one.
    """

    from evals.corpus import compile_case

    def _compile(case_id: str) -> ClipResult:
        if case_id not in corpus_by_id:
            raise KeyError(
                f"{case_id!r} is not a corpus case; known ids include "
                f"{sorted(corpus_by_id)[:3]}..."
            )
        if case_id not in _compiled_cache:
            _compiled_cache[case_id] = compile_case(corpus_by_id[case_id])
        return _compiled_cache[case_id].model_copy(deep=True)

    return _compile
