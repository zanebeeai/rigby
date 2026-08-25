"""The 28 legacy specs must survive the port unchanged in motion.

Plan 06 section 2 absorbs them as the severe tier rather than replacing them, so the
port is only honest if the motion it produces is what the previous calibration used.
Anything else silently reinterprets every number that suite ever produced.
"""

from __future__ import annotations

import collections

import pytest

from evals.corpus import load_case, load_corpus
from evals.corpus.loader import compile_case
from evals.corruptions import corrupt_clip, corruption_specs
from evals.mutations.family import MutationFamily, Tier
from evals.mutations.legacy import _hand_of, legacy_specs

GESTURE_CASE = "gesture-hangten-shake-right"


@pytest.fixture(scope="module")
def clip():
    return compile_case(load_case(GESTURE_CASE))


@pytest.fixture(scope="module")
def specs():
    return legacy_specs()


def test_all_twenty_eight_are_ported(specs) -> None:
    assert len(specs) == len(corruption_specs()) == 28
    assert len({spec.id for spec in specs}) == 28


@pytest.mark.parametrize("legacy", corruption_specs(), ids=lambda item: item.id)
def test_the_port_produces_identical_motion(clip, legacy) -> None:
    """Byte-identical frames against ``evals.corruptions.corrupt_clip``.

    This is the gate on the port. If it drifts, every number the previous
    calibration produced means something different, and nothing else would say so.
    """
    spec = next(item for item in legacy_specs() if item.params["legacy_id"] == legacy.id)
    assert spec.applies_to(clip), spec.id
    mine = spec.apply(clip)
    theirs = corrupt_clip(clip.model_copy(deep=True), _hand_of(clip), legacy)
    assert [frame.model_dump(mode="json") for frame in mine.frames] == [
        frame.model_dump(mode="json") for frame in theirs.frames
    ]


def test_every_legacy_spec_is_severe(specs) -> None:
    """The mildest is 0.85 rad, about 49 degrees.

    Plan 06 section 1.3: a grader that detects a 49-degree wrist error has
    demonstrated almost nothing. Labelling them all severe is what makes a report
    that covers only this tier visibly a report that covers only this tier.
    """
    assert {spec.tier for spec in specs} == {Tier.SEVERE}
    assert all(spec.severity == 1.0 for spec in specs)


def test_the_families_match_what_each_kind_actually_breaks(specs) -> None:
    counts = collections.Counter(spec.family for spec in specs)
    assert counts == {
        MutationFamily.ANATOMY: 12,
        MutationFamily.SEMANTIC: 12,
        MutationFamily.TIMING: 4,
    }


def test_the_wrong_joint_spec_is_semantic_because_it_stays_inside_the_gate(specs) -> None:
    """Its own docstring says so, and the classification has to follow it.

    ``_corrupt_wrong_joint_shake``: "Its magnitude stays inside the broad structural
    wrist limit on purpose: this is a visual-anatomy corruption the VLM must reject
    rather than an easy deterministic-gate failure." Filing it under anatomy with a
    deterministic target would score a check that cannot fire as having missed.
    """
    wrong_joint = [
        spec for spec in specs if spec.params["legacy_kind"] == "wrong_joint_shake"
    ]
    assert len(wrong_joint) == 4
    for spec in wrong_joint:
        assert spec.family is MutationFamily.SEMANTIC
        assert spec.targets == ()
        assert spec.asserts, "a semantic mutation needs an asserted program difference"


def test_applicability_is_resolved_per_case_not_per_spec(specs) -> None:
    """Every legacy spec is gesture-shaped; most corpus cases are not.

    Reported as counts rather than asserted case by case, because the interesting
    number is how much of the corpus these 28 specs can say anything about at all.
    """
    tally = collections.Counter()
    for case in load_corpus():
        case_clip = compile_case(case)
        for spec in specs:
            verdict = spec.applies_to(case_clip)
            if not verdict:
                tally["not applicable"] += 1
            elif verdict.static_target:
                tally["capability only"] += 1
            else:
                tally["threshold"] += 1
    total = sum(tally.values())
    assert total == 47 * 28
    # Most pairs are not threshold measurements, and a suite that pooled them would
    # be reporting mostly capability results as if they were detection thresholds.
    assert tally["threshold"] < total // 2, tally
    assert tally["threshold"] > 0, tally
    assert tally["not applicable"] > 0, tally
    assert tally["capability only"] > 0, tally


def test_the_severe_tier_alone_cannot_locate_a_threshold(specs) -> None:
    """Plan 06 section 1.3, as a property of the ported set rather than a comment.

    Every one of these sits at severity 1.0, so they span no range. A detection
    threshold needs a sweep; these 28 give a single point, which is exactly why
    06b's graded families exist.
    """
    assert len({spec.severity for spec in specs}) == 1
