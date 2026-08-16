"""Adding an Intent or a BodyAction without a corpus case must fail here.

03a covers four of the seven executable intents and eight of the twelve body
actions; the rest are declared deferred with a reason. A member in neither list is
a hole, which is exactly the state a newly added enum member lands in.
"""

from __future__ import annotations

import pytest
from evals.corpus import load_corpus, load_manifest
from evals.corpus.freeze import body_actions_of, rebuild_coverage
from evals.corpus.models import CoverageAxis
from evals.corpus.seed_cases import SEED_CASES_BY_ID
from pydantic import ValidationError
from rigby_poc.models import BodyAction, Intent

MANIFEST = load_manifest()
CASES = load_corpus()

#: ``Intent.UNSUPPORTED`` produces no frames, so it can never carry a motion hash.
#: It is still required to be *declared*, so that its absence is a decision on the
#: record rather than an oversight.
AXES = (
    ("intent", [member.value for member in Intent], MANIFEST.coverage.intent),
    ("body_action", [member.value for member in BodyAction], MANIFEST.coverage.body_action),
)


@pytest.mark.parametrize(("name", "members", "axis"), AXES, ids=[item[0] for item in AXES])
def test_every_enum_member_is_covered_or_deferred_with_a_reason(
    name: str, members: list[str], axis: CoverageAxis
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


@pytest.mark.parametrize(("name", "members", "axis"), AXES, ids=[item[0] for item in AXES])
def test_deferred_members_really_have_no_case(
    name: str, members: list[str], axis: CoverageAxis
) -> None:
    """A member cannot be deferred once a case covers it, or the list rots."""
    covered_now = set(rebuild_coverage(MANIFEST).model_dump()[name]["covered"])
    stale = sorted(set(axis.deferred) & covered_now)
    assert not stale, f"{name} deferred but covered by a case: {stale}"


def test_declared_coverage_matches_the_cases_on_disk() -> None:
    assert rebuild_coverage(MANIFEST).model_dump() == MANIFEST.coverage.model_dump()


def test_covered_intents_and_actions_are_reachable_from_a_case() -> None:
    intents = {case.program.intent.value for case in CASES}
    actions = {action.value for case in CASES for action in body_actions_of(case.program)}
    assert intents == set(MANIFEST.coverage.intent.covered)
    assert actions == set(MANIFEST.coverage.body_action.covered)


def test_every_case_records_the_prompt_it_was_planned_from() -> None:
    """The manifest and ``seed_cases.py`` must not drift apart."""
    assert {case.id for case in CASES} == set(SEED_CASES_BY_ID)
    for case in CASES:
        assert case.entry.source_prompt == SEED_CASES_BY_ID[case.id].prompt
        assert case.entry.family == SEED_CASES_BY_ID[case.id].family


def test_families_span_the_four_that_03a_claims() -> None:
    assert {case.entry.family.value for case in CASES} == {
        "gesture",
        "strike",
        "composite",
        "full_body",
    }


def test_a_member_cannot_be_both_covered_and_deferred() -> None:
    with pytest.raises(ValidationError):
        CoverageAxis(covered=["walk"], deferred={"walk": "later"})


def test_a_deferral_needs_a_reason() -> None:
    with pytest.raises(ValidationError):
        CoverageAxis(deferred={"walk": "  "})
