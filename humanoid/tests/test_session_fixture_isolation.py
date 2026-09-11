"""A session-scoped compile must not leak a mutation between tests.

Plan 09 §6.1 names this risk and says to decide it deliberately rather than
discover it as flakiness. The decision is in ``conftest.py``: share the compile,
copy the object. This file is the proof, and it is written so that it is **red
against a naive session fixture** rather than merely green against the real one.

The scenario is lane ``groundtruth``'s, from the corpus side: `compile_case`
returns a fresh `ClipResult` only because it recompiles, so removing the
recompile is exactly what would start sharing the instance. Their
``test_corpus_known_bad.py`` builds a module-level dict of those metrics dicts
and reads it across roughly 200 parametrised tests -- nothing mutates them today
and nothing stops it either.

Test order matters here: the mutating test must run before the observer. They are
named so that pytest's file order runs them that way, and the observer also
asserts the sentinel is absent rather than only checking the value, so a reorder
turns into a failure rather than a false pass.
"""

from __future__ import annotations

import pytest


CASE = "gesture-shaka-playful-right"
SENTINEL = "__leaked_from_a_previous_test__"


pytestmark = pytest.mark.medium


def test_a_mutates_the_clip_it_was_given(compile_corpus_case) -> None:
    """Mutate hard: add a key, and change an existing one."""

    clip = compile_corpus_case(CASE)
    clip.metrics[SENTINEL] = True
    clip.metrics["root_drift_m"] = 999.0
    assert clip.metrics[SENTINEL] is True


def test_b_sees_no_trace_of_the_previous_mutation(compile_corpus_case) -> None:
    """The gate. Red against a shared instance, green against a deep copy."""

    clip = compile_corpus_case(CASE)
    assert SENTINEL not in clip.metrics, (
        "a mutation from an earlier test survived into this one; the session "
        "fixture is handing out a shared object instead of a copy"
    )
    assert clip.metrics["root_drift_m"] != 999.0


def test_two_requests_in_one_test_are_independent(compile_corpus_case) -> None:
    """Two handles to the same case must not alias each other."""

    first = compile_corpus_case(CASE)
    second = compile_corpus_case(CASE)
    assert first is not second
    assert first.metrics is not second.metrics
    first.metrics[SENTINEL] = True
    assert SENTINEL not in second.metrics


def test_the_frames_are_copied_too_not_just_the_metrics(compile_corpus_case) -> None:
    """A shallow copy would share the frame list, which is the bigger object."""

    first = compile_corpus_case(CASE)
    second = compile_corpus_case(CASE)
    assert first.frames is not second.frames
    assert first.frames[0] is not second.frames[0]


def test_the_shared_compile_is_actually_shared(_compiled_cache, compile_corpus_case) -> None:
    """The saving must be real, or the copying is pure cost.

    If this passes while the isolation tests also pass, the fixture is doing what
    it claims: one compile, many private copies.
    """

    _compiled_cache.pop(CASE, None)
    compile_corpus_case(CASE)
    assert CASE in _compiled_cache
    cached = _compiled_cache[CASE]
    compile_corpus_case(CASE)
    assert _compiled_cache[CASE] is cached, "the case was recompiled on the second request"


def test_the_cache_itself_is_never_handed_out(_compiled_cache, compile_corpus_case) -> None:
    """The private cache entry must not be reachable through the public fixture."""

    handed_out = compile_corpus_case(CASE)
    assert handed_out is not _compiled_cache[CASE]


def test_an_unknown_case_id_fails_loudly(compile_corpus_case) -> None:
    with pytest.raises(KeyError, match="is not a corpus case"):
        compile_corpus_case("no-such-case")


def test_the_rig_cache_is_not_cleared_between_tests(rig) -> None:
    """One skeleton per process is a provenance invariant, not an optimisation.

    PR 04b's asset-digest assertion rests on ``rig_kinematics`` parsing once per
    process. A fixture that called ``cache_clear()`` for isolation would break it
    silently, so this pins that nothing here does.
    """

    from rigby_poc.kinematics import rig_kinematics

    assert rig_kinematics() is rig
    assert rig_kinematics.cache_info().currsize == 1
