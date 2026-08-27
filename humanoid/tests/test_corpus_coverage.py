"""Adding an enum member without a corpus case must fail here.

A member in neither the covered nor the deferred list is a hole, which is exactly
the state a newly added enum member lands in.

03a guarded two axes and covered four of the seven executable intents and eight of
the twelve body actions.  03b closes both and adds three more axes the corpus
already consumed unguarded -- ``ObjectAction`` (9 members), ``StrikeType`` (4) and
``HandShape`` (7).  Every axis is now fully covered with zero deferrals, so the next
enum member added anywhere in that set is a build failure.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
from evals.corpus import load_corpus, load_manifest
from evals.corpus.loader import read_slim_clip
from evals.corpus.freeze import (
    body_actions_of,
    hand_shapes_of,
    object_actions_of,
    rebuild_coverage,
    strike_types_of,
)
from evals.corpus.models import CoverageAxis
from evals.corpus.seed_cases import SEED_CASES_BY_ID
from pydantic import ValidationError
from rigby_poc.models import BodyAction, HandShape, Intent, ObjectAction, StrikeType

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium

MANIFEST = load_manifest()
CASES = load_corpus()
CASES_BY_ID = {case.id: case for case in CASES}

#: ``Intent.UNSUPPORTED`` produces no frames, so it can never carry a motion hash.
#: It is still required to be *declared*, so that its absence is a decision on the
#: record rather than an oversight.
#: One row per :class:`~evals.corpus.models.Coverage` field: the enum it guards, and
#: how a loaded case reports the members it covers.
AXES = tuple(
    (name, [member.value for member in enum], getattr(MANIFEST.coverage, name), reader)
    for name, enum, reader in (
        ("intent", Intent, lambda case: [case.program.intent]),
        ("body_action", BodyAction, lambda case: body_actions_of(case.program)),
        ("object_action", ObjectAction, lambda case: object_actions_of(case.program)),
        ("strike_type", StrikeType, lambda case: strike_types_of(case.program)),
        ("hand_shape", HandShape, lambda case: hand_shapes_of(case.program)),
    )
)
AXIS_IDS = [row[0] for row in AXES]


@pytest.mark.parametrize(("name", "members", "axis", "reader"), AXES, ids=AXIS_IDS)
def test_every_enum_member_is_covered_or_deferred_with_a_reason(
    name: str, members: list[str], axis: CoverageAxis, reader: object
) -> None:
    declared = set(axis.covered) | set(axis.deferred)
    undeclared = sorted(set(members) - declared)
    assert not undeclared, (
        f"{name} member(s) {undeclared} have no corpus case and no deferral reason. "
        f"Add a case with `python -m evals.corpus freeze`, or record why not under "
        f"coverage.{name}.deferred in evals/corpus/manifest.json."
    )
    unknown = sorted(declared - set(members))
    assert not unknown, f"coverage.{name} names non-members: {unknown}"


@pytest.mark.parametrize(("name", "members", "axis", "reader"), AXES, ids=AXIS_IDS)
def test_deferred_members_really_have_no_case(
    name: str, members: list[str], axis: CoverageAxis, reader: object
) -> None:
    """A member cannot be deferred once a case covers it, or the list rots."""
    covered_now = set(rebuild_coverage(MANIFEST).model_dump()[name]["covered"])
    stale = sorted(set(axis.deferred) & covered_now)
    assert not stale, f"{name} deferred but covered by a case: {stale}"


def test_declared_coverage_matches_the_cases_on_disk() -> None:
    assert rebuild_coverage(MANIFEST).model_dump() == MANIFEST.coverage.model_dump()


@pytest.mark.parametrize(("name", "members", "axis", "reader"), AXES, ids=AXIS_IDS)
def test_every_covered_member_is_reachable_from_a_case(
    name: str, members: list[str], axis: CoverageAxis, reader: object
) -> None:
    """The declaration must be readable back off the programs, not just asserted.

    Reading through the *programs on disk* rather than the manifest rows is the
    point: the rows are what ``rebuild_coverage`` derives from, so checking the
    declaration against them would be circular.
    """
    observed = {member.value for case in CASES for member in reader(case)}
    assert observed == set(axis.covered)


def test_every_axis_is_fully_covered_with_no_deferrals() -> None:
    """03b's headline: no enum member anywhere is waiting for a case."""
    holes = {
        name: sorted(axis.deferred)
        for name, members, axis, _ in AXES
        if set(axis.covered) != set(members)
    }
    assert not holes, holes


def test_every_case_records_the_prompt_it_was_planned_from() -> None:
    """The manifest and ``seed_cases.py`` must not drift apart."""
    assert {case.id for case in CASES} == set(SEED_CASES_BY_ID)
    for case in CASES:
        assert case.entry.source_prompt == SEED_CASES_BY_ID[case.id].prompt
        assert case.entry.family == SEED_CASES_BY_ID[case.id].family


def test_families_span_every_family_the_corpus_defines() -> None:
    from evals.corpus.models import Family

    assert {case.entry.family.value for case in CASES} == {
        member.value for member in Family
    }


def test_a_member_cannot_be_both_covered_and_deferred() -> None:
    with pytest.raises(ValidationError):
        CoverageAxis(covered=["walk"], deferred={"walk": "later"})


def test_a_deferral_needs_a_reason() -> None:
    with pytest.raises(ValidationError):
        CoverageAxis(deferred={"walk": "  "})


#: Bones no corpus case moves, and no prompt can move: nothing in ``compiler.py``
#: ever assigns a rotation to any of them.  See the test below.
# Was five until the strike trunk-yaw work: strikes now rotate `spine` and
# `upperChest` (and `chest`, which always moved), so those two joined the
# covered set exactly as this test's docstring hoped.
IMMOBILE_BONES = frozenset({"leftToes", "rightToes", "neck"})

#: Quaternion component difference below which a bone counts as unmoved.
BONE_MOTION_EPSILON = 1e-6


@lru_cache(maxsize=None)
def _bones_moved(case_id: str) -> frozenset[str]:
    """Bone names this case rotates away from its own frame 0.

    Reads the **committed slim clip** rather than recompiling.  Two tests ask this
    of all 47 cases, and recompiling would be 94 compiles -- around nineteen
    minutes -- to answer a question the stored frames already answer.

    Sound because ``test_corpus_determinism`` proves the stored clip is identical to
    a fresh compile on this platform, and because the question is about *coverage*
    rather than about the hash.  This is the first consumer of the stored clip, which
    plan 03 section 3.3 noted did not yet exist.
    """
    frames = read_slim_clip((CASES_BY_ID[case_id]).slim_clip_path)["frames"]
    if not frames:
        return frozenset()
    rest = frames[0]["bones"]
    moved: set[str] = set()
    for frame in frames:
        for name, bone in frame["bones"].items():
            reference, rotation = rest[name]["rotation"], bone["rotation"]
            if any(
                abs(rotation[axis] - reference[axis]) > BONE_MOTION_EPSILON
                for axis in ("w", "x", "y", "z")
            ):
                moved.add(name)
    return frozenset(moved)


def test_three_bones_are_immobile_in_every_case_and_in_the_compiler() -> None:
    """Three of the 52 canonical bones can never move, whatever the prompt.

    This is a **compiler** gap, not a corpus gap, and no corpus case can close it:
    ``compiler.py`` never assigns a rotation to ``leftToes``, ``rightToes`` or
    ``neck`` on any path.  It matters downstream -- a ROM limit on those bones
    can never fire, and an anatomy grader can never be calibrated on them from
    generated motion, only from mutation.

    The set was five until the strike trunk-yaw work started rotating ``spine``
    and ``upperChest``. If this test goes red because a bone started moving,
    that is good news: delete it from the set and the corpus has gained a joint
    class.
    """
    still = set(IMMOBILE_BONES)
    for case in CASES:
        still -= _bones_moved(case.id)
    assert still == IMMOBILE_BONES, (
        f"these bones now move somewhere: {sorted(IMMOBILE_BONES - still)}"
    )


def test_the_clavicles_are_covered_by_exactly_one_case() -> None:
    """``fullbody-shrug-shoulders`` is the only route to the clavicle joint class.

    ``planner.py``'s ``shrug`` token is the sole producer of
    ``BodyPoseTarget.left_shoulder_elevation_deg``, and without it
    ``leftShoulder``/``rightShoulder`` move in zero cases -- an entire joint class
    with n = 0, which cannot be calibrated by any amount of grader work.  n = 1 is
    thin and is stated as such; it is the difference between thin and impossible.
    """
    movers = sorted(
        case.id
        for case in CASES
        if {"leftShoulder", "rightShoulder"} <= _bones_moved(case.id)
    )
    assert movers == ["fullbody-shrug-shoulders"], movers
