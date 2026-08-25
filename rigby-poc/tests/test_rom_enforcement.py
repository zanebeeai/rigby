"""Per-DOF range-of-motion enforcement. Plan 04c.

04b measured; this gates. The decision it encodes is that **the elbow bound is
right and the compiler is wrong on every clip**, reported as a 100% rejection
rate with its n rather than softened until the number looks survivable. Plan 04
§6.1l records the measurement that settles it: the humerus rolls under a third
of a degree while the forearm departs its plane by up to 96, so there is nothing
upstream to justify the excursion.
"""

from __future__ import annotations

import pytest

from evals.corpus import load_corpus
from evals.corpus.loader import compile_case
from rigby_poc.analysis.anatomy.authored_rom import ROOT, is_enforced
from rigby_poc.analysis.anatomy.rom import (
    SEVERITY_SCALE_DEG,
    enforceability,
    rom_checks,
    rom_limit,
    rom_limits,
    rom_violations,
)

pytestmark = pytest.mark.medium


@pytest.fixture(scope="module")
def corpus_failures() -> dict[str, list[str]]:
    """Per case: the DOF ids that fail under enforcement. Empty means accepted.

    Cases that compile to zero frames are excluded rather than counted as
    passes -- there is nothing to measure in them, and folding them in would
    inflate the accept rate with clips nobody rendered.
    """

    failures: dict[str, list[str]] = {}
    for case in load_corpus():
        clip = compile_case(case)
        if not clip.frames:
            continue
        failures[case.entry.id] = sorted(
            result.id.removeprefix("anatomy.rom.")
            for result in rom_checks(clip.frames, fps=clip.fps)
            if result.status == "fail"
        )
    return failures


# --------------------------------------------------------------------------
# What is enforced, and why
# --------------------------------------------------------------------------


def test_enforcement_follows_the_rule_rather_than_a_hand_picked_list() -> None:
    """The config must agree with the predicate that generated it.

    A hand-picked list of enforced DOFs would drift from its rationale and a new
    entry would inherit an oversight rather than a decision.
    """

    for (bone, dof), limit in rom_limits().items():
        assert limit.enforced == is_enforced(bone, limit.source["kind"]), f"{bone}.{dof}"


def test_a_provisional_bound_is_never_enforced() -> None:
    """Gating on a value that declares itself unvalidated would publish a
    rejection rate that means nothing."""

    provisional = [
        key for key, limit in rom_limits().items() if limit.source["kind"] == "provisional"
    ]

    assert len(provisional) == 69
    assert not any(rom_limits()[key].enforced for key in provisional)


def test_a_mutation_only_bone_is_never_enforced() -> None:
    for (bone, dof), limit in rom_limits().items():
        if enforceability(bone) == "mutation_only":
            assert not limit.enforced, f"{bone}.{dof}"


def test_the_root_is_never_enforced() -> None:
    """`hips` is whole-body orientation, not a joint angle.

    Its bounds are wide sanity checks, and a cartwheel legitimately rotates it
    past them. Enforcing would reject real motion for a reason that is not
    anatomical, and plan 04 §6.1k says a `hips` failure is not anatomical
    evidence -- so it must not become one here either.
    """

    for bone in ROOT:
        for dof in ("flexion", "abduction", "twist"):
            assert not rom_limit(bone, dof).enforced


def test_both_hinges_enforce_abduction_and_neither_enforces_twist() -> None:
    """G4, and the §3.3 correction, as a gating statement."""

    for side in ("left", "right"):
        for bone in (f"{side}LowerArm", f"{side}LowerLeg"):
            assert rom_limit(bone, "abduction").enforced
            assert rom_limit(bone, "abduction").hard_assert
            assert not rom_limit(bone, "twist").hard_assert


# --------------------------------------------------------------------------
# The rejection rate, with its n
# --------------------------------------------------------------------------


def test_enforcement_rejects_every_corpus_case(corpus_failures) -> None:
    """Plan §6.4's intended outcome, pinned with its n rather than softened.

    46 of 46 cases with frames. This is not a regression and it must not be
    fixed by widening the bound: §6.1l tested the one hypothesis that would have
    exonerated the compiler -- that humeral roll justifies the off-plane forearm
    -- and measured under a third of a degree of roll against up to 96 degrees
    of departure.
    """

    rejected = [case for case, failing in corpus_failures.items() if failing]

    assert len(corpus_failures) == 46
    assert len(rejected) == 46


def test_the_elbow_is_the_whole_story_and_the_rest_is_the_variance(corpus_failures) -> None:
    """What lane `judge` consumes: the split, not the rate.

    A 100% rate carries no information on its own. The usable axis is that most
    cases fail on exactly one thing and a minority fail on more.
    """

    elbow_only = [
        case
        for case, failing in corpus_failures.items()
        if failing and all(dof.endswith("LowerArm.abduction") for dof in failing)
    ]

    assert len(elbow_only) == 37
    assert len(corpus_failures) - len(elbow_only) == 9


def test_every_case_fails_on_both_elbows(corpus_failures) -> None:
    for case, failing in corpus_failures.items():
        assert "leftLowerArm.abduction" in failing, case
        assert "rightLowerArm.abduction" in failing, case


# --------------------------------------------------------------------------
# How it gates
# --------------------------------------------------------------------------


def test_no_duration_tolerance_is_applied_because_there_are_no_blips() -> None:
    """Plan §3.5 proposed gating on the integral to separate a blip from a
    sustained excursion. Measured over the corpus, there are no blips to
    separate: the shortest violation runs 4 frames and the median occupies 99%
    of its clip. A duration gate would filter nothing and would be an
    unmotivated constant, so the band gates and the integral only ranks.

    If a future corpus does contain single-frame excursions, this test fails and
    the duration gate becomes justified -- which is the point of pinning it.
    """

    shortest = None
    for case in load_corpus():
        clip = compile_case(case)
        if not clip.frames:
            continue
        for violation in rom_violations(clip.frames, fps=clip.fps):
            if violation.band != "beyond_max":
                continue
            if shortest is None or violation.frames < shortest:
                shortest = violation.frames

    assert shortest is not None
    assert shortest >= 4


def test_a_report_only_breach_does_not_fail(corpus_failures) -> None:
    """The half of the split that makes the other half meaningful.

    `rightLittleProximal.abduction` is enforced and fails in three cases. The
    thumb entries breach nothing because their bounds are wide, but the
    structural claim is what matters: a beyond_max on an unenforced DOF is
    recorded and does not gate.
    """

    case = next(item for item in load_corpus() if item.entry.id == "fullbody-cartwheel")
    clip = compile_case(case)

    results = {item.id: item for item in rom_checks(clip.frames, fps=clip.fps)}
    breached = {
        f"anatomy.rom.{v.bone}.{v.dof}"
        for v in rom_violations(clip.frames, fps=clip.fps)
        if v.band == "beyond_max"
    }
    unenforced_breaches = [
        key for key in breached if not rom_limits()[tuple(key.split(".")[2:4])].enforced
    ]

    assert unenforced_breaches, "expected at least one unenforced breach to exist"
    for key in unenforced_breaches:
        assert results[key].status == "pass"
        assert results[key].severity == 0.0
        assert results[key].measured["band"] == "beyond_max"


def test_severity_ranks_failures_and_never_decides_one() -> None:
    """`SEVERITY_SCALE_DEG` is arbitrary and that is safe only while it cannot
    gate. This pins that property: changing the scale must not change any
    status."""

    case = next(item for item in load_corpus() if item.entry.id == "fullbody-dance")
    clip = compile_case(case)

    before = {item.id: (item.status, item.severity) for item in rom_checks(clip.frames, fps=clip.fps)}
    failing = {key: value for key, value in before.items() if value[0] == "fail"}

    assert failing
    assert all(0.0 < severity <= 1.0 for _, severity in failing.values())
    assert all(severity == 0.0 for status, severity in before.values() if status != "fail")
    assert SEVERITY_SCALE_DEG > 0.0
