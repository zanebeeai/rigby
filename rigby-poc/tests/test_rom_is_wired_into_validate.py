"""The per-DOF range-of-motion layer is reachable from the analysis layer's checks.

04c shipped ROM enforcement **dark**. It computed 156 limits, enforced 82 of
them, and rejected 46 of 47 corpus cases -- and nothing called it. The only
references to :func:`~rigby_poc.analysis.anatomy.rom.rom_checks` in the whole
repository were its own import and its own ``__all__``. A capability with no
caller and a gate that cannot fire produce the same evidence: nothing.

This file is the guard that the wiring stays. It is deliberately written against
the *public* check surface -- :func:`rigby_poc.analysis.validate` -- rather than
against ``rom_checks`` directly, because ``rom_checks`` was already thoroughly
covered by lane `anatomy`'s tests while being unreachable from any caller.
Testing the producer is what let the gap exist, so testing the producer again
would not close it.

The load-bearing measurement is in
:func:`test_the_layer_rejects_clips_the_legacy_counter_passes`: over all 47
corpus cases the legacy ``joint_limit_violations`` counter is **0 of 47** while
the per-DOF layer fails on **18 of 47**, on the same clips and the same bones.
(It was 46 of 47 before the humeral-roll fix; the fix removed the spurious
elbow abduction that made nearly every case fail, and the remaining 18 are the
cases with genuine excursions.) That is the evidence the legacy check may be
deleted -- it is inert, not permissive -- and it is why this PR must land
before that deletion rather than after it.
"""

from __future__ import annotations

import pytest

from evals.corpus import load_corpus
from evals.corpus.loader import compile_case
from rigby_poc.analysis import validate, validate_clip
from rigby_poc.analysis.anatomy.rom import rom_limits
from rigby_poc.analysis.contract import CheckResult

#: One corpus compile per case in a module fixture; see test_corpus_compile_budget.py.
pytestmark = pytest.mark.medium

ROM_PREFIX = "anatomy.rom."

#: The case that compiles to no frames at all. Named rather than discovered so the
#: assertion below fails loudly if it stops being the unmeasured one.
NO_FRAMES_CASE = "knownbad-eigenvalues-unsupported"


@pytest.fixture(scope="module")
def verdicts() -> dict[str, tuple[dict, list[CheckResult]]]:
    """``case id -> (compiler metrics, every check validate emits)``."""

    collected: dict[str, tuple[dict, list[CheckResult]]] = {}
    for case in load_corpus():
        clip = compile_case(case)
        collected[case.entry.id] = (clip.metrics, validate_clip(clip, case.program))
    return collected


@pytest.fixture(scope="module")
def rom_verdicts(
    verdicts: dict[str, tuple[dict, list[CheckResult]]],
) -> dict[str, list[CheckResult]]:
    return {
        case_id: [c for c in checks if c.id.startswith(ROM_PREFIX)]
        for case_id, (_metrics, checks) in verdicts.items()
    }


def test_the_probe_saw_a_real_corpus(
    verdicts: dict[str, tuple[dict, list[CheckResult]]],
) -> None:
    # Every assertion in this file iterates a collection the file did not
    # construct, and each passes vacuously on an empty one. docs/testing.md,
    # "Guards fail in two directions".
    assert len(verdicts) >= 47, f"the corpus produced {len(verdicts)} cases"
    assert NO_FRAMES_CASE in verdicts


def test_validate_emits_a_verdict_for_every_committed_rom_limit(
    rom_verdicts: dict[str, list[CheckResult]],
) -> None:
    """One addressable id per ``(bone, dof)``, on every case, from the public API.

    Equality rather than containment: a subset assertion would stay green if the
    wiring were narrowed to a handful of bones, which is a state this file
    exists to detect.
    """

    expected = {f"{ROM_PREFIX}{bone}.{dof}" for bone, dof in rom_limits()}
    assert len(expected) == 156
    for case_id, checks in rom_verdicts.items():
        assert {c.id for c in checks} == expected, case_id


def test_the_layer_rejects_clips_the_legacy_counter_passes(
    verdicts: dict[str, tuple[dict, list[CheckResult]]],
) -> None:
    """The measurement that licenses deleting the legacy joint-limit check.

    ``joint_limit_violations`` is not permissive, it is **inert**: zero on every
    corpus case, including the root-drift term ``analysis/safety.py`` folds into
    the same counter. The per-DOF layer fails on 16 of 47 cases over the same
    clips and the same bones. Tightening the legacy check would therefore move
    nothing, and deleting it removes a counter that has never once been
    non-zero -- but only once these verdicts exist, which is why the assertion
    lives here rather than in the deletion PR.
    """

    legacy_nonzero = []
    rom_failing = []
    for case_id, (metrics, checks) in verdicts.items():
        if int(metrics.get("joint_limit_violations", 0)) != 0:
            legacy_nonzero.append(case_id)
        if any(c.id.startswith(ROM_PREFIX) and c.failed for c in checks):
            rom_failing.append(case_id)

    total = len(verdicts)
    assert legacy_nonzero == [], (
        f"the legacy counter fired on {legacy_nonzero}; it was inert on all "
        f"{total} cases when this was measured. If it has become live, the "
        "deletion this measurement licenses needs re-deciding."
    )
    assert len(rom_failing) == 16, (
        f"the per-DOF layer failed on {len(rom_failing)} of {total} cases; it "
        "failed on 16 of 47 when this was measured after the humeral-roll fix "
        "plus the swing-twist-coordinated arm interpolation (18 of 47 with "
        "the roll fix alone, 46 of 47 before it). A moved count means the ROM "
        "landscape changed: re-measure and re-pin rather than widening this "
        "to a floor."
    )


def test_a_failing_rom_verdict_is_ranked_as_well_as_addressable(
    rom_verdicts: dict[str, list[CheckResult]],
) -> None:
    """A fail carries a non-zero severity, so a composite can order the failures.

    ``CheckResult`` forbids a non-zero severity on anything but a fail
    (``contract.py``), so a severity-based score is zero for every clean clip by
    construction. The half it *can* rank has to actually be ranked, or the
    quantity is zero everywhere and ranks nothing at all -- which is the
    degenerate oracle plan 10 §3.3 has to avoid.
    """

    failures = [c for checks in rom_verdicts.values() for c in checks if c.failed]
    assert failures, "no case produced a failing ROM verdict"
    assert all(c.severity > 0.0 for c in failures)
    assert all(c.layer == "anatomy" for c in failures)


def test_an_unmeasured_clip_skips_rather_than_passing(
    rom_verdicts: dict[str, list[CheckResult]],
) -> None:
    """A clip with no frames must not publish 156 clean verdicts.

    ``knownbad-eigenvalues-unsupported`` compiles to no frames at all, so every
    bone is unmeasured. The honest output is 156 skips; the dangerous one is 156
    passes -- the "absence becoming a value" shape in docs/testing.md, which
    this very module produced before lane `anatomy` fixed it. Asserted through
    ``validate`` because the fix was made, and could be lost, one layer below.
    """

    statuses = {c.status for c in rom_verdicts[NO_FRAMES_CASE]}
    assert statuses == {"skip"}
    assert len(rom_verdicts[NO_FRAMES_CASE]) == 156


def test_validate_cannot_silently_drop_the_anatomy_layer() -> None:
    """``frames`` is required, so no caller can lose ROM by omission.

    This is the whole reason the signature changed rather than gaining an
    optional parameter. An optional ``frames=None`` would have left the one real
    caller -- the mutation detection matrix, which reads ``clip.metrics`` --
    still not passing them, and the wiring would have been an appearance rather
    than a fact. Pinned as a test because "we will remember to pass frames" is
    the assumption that produced the dark gate in the first place.
    """

    case = next(iter(load_corpus()))
    with pytest.raises(TypeError):
        validate({}, case.program)  # type: ignore[call-arg]
