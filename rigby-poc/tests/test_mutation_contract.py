"""The contract a mutated clip must satisfy, and the label leak it must not repeat.

Plan 06 section 5's ``test_mutation_preserves_contract``: a mutated clip is still
schema-valid, finite and correctly framed, so downstream capture and export do not
fail for the wrong reason.  Plus the three rules from section 6.5, which exist
because ``evals/corruptions.py`` breaks all three.
"""

from __future__ import annotations

import math

import pytest
from evals.corpus import load_case
from evals.corpus.loader import compile_case
from evals.mutations.legacy import legacy_specs
from evals.mutations.spec import NotApplicable
from rigby_poc.models import ClipResult

GESTURE_CASE = "gesture-hangten-shake-right"
FULL_BODY_CASE = "fullbody-walk-forward"


@pytest.fixture(scope="module")
def gesture_clip() -> ClipResult:
    return compile_case(load_case(GESTURE_CASE))


@pytest.fixture(scope="module")
def specs():
    return legacy_specs()


@pytest.fixture(scope="module")
def mutated(gesture_clip, specs):
    return {spec.id: spec.apply(gesture_clip) for spec in specs}


def test_every_legacy_spec_applies_to_the_gesture_it_was_built_for(gesture_clip, specs):
    """Pins the fixture the rest of this file rests on."""
    assert len(specs) == 28
    unusable = [spec.id for spec in specs if not spec.applies_to(gesture_clip)]
    assert not unusable, unusable


def test_a_mutated_clip_does_not_announce_that_it_is_mutated(gesture_clip, mutated):
    """Rule 1 and 2 of plan 06 section 6.5.

    ``corrupt_clip`` sets ``success=False``, attaches a ``Failure`` naming the
    corruption, and writes ``metrics["deliberate_corruption"]``.  Then
    ``calibrate_judge.py:224`` scores ``parsed["accept"] and structural_valid`` -- a
    conjunction whose deterministic term the corruption was built to force false, so
    the recorded outcome was not a function of the grader's verdict at all.
    """
    for spec_id, clip in mutated.items():
        assert "deliberate_corruption" not in clip.metrics, spec_id
        assert clip.success == gesture_clip.success, spec_id
        assert clip.failure == gesture_clip.failure, spec_id
        blob = clip.model_dump_json()
        assert spec_id not in blob, f"{spec_id} leaks its own id into the clip"
        assert "corruption" not in blob.lower(), f"{spec_id} leaks the word corruption"


def test_a_mutated_clip_is_still_schema_valid(mutated):
    for spec_id, clip in mutated.items():
        ClipResult.model_validate(clip.model_dump(mode="json")), spec_id


def test_a_mutated_clip_stays_finite(mutated):
    for spec_id, clip in mutated.items():
        for index, frame in enumerate(clip.frames):
            for name, pose in frame.bones.items():
                for value in pose.rotation.as_list():
                    assert math.isfinite(value), f"{spec_id} frame {index} bone {name}"


def test_a_mutated_clip_keeps_its_quaternions_normalised(mutated):
    """Capture and GLB export both assume unit quaternions."""
    for spec_id, clip in mutated.items():
        for index, frame in enumerate(clip.frames):
            for name, pose in frame.bones.items():
                norm = math.sqrt(sum(value * value for value in pose.rotation.as_list()))
                assert norm == pytest.approx(1.0, abs=1e-6), (
                    f"{spec_id} frame {index} bone {name} norm {norm}"
                )


def test_a_mutated_clip_keeps_its_frame_count_and_timebase(gesture_clip, mutated):
    """A mutation perturbs motion, not the clip's shape.

    A timing mutation reorders or repeats poses; it must not change how many frames
    there are or when they occur, or the mutated clip is a different *recording*
    rather than a different *motion*, and a downstream length comparison would score
    the difference for free.
    """
    for spec_id, clip in mutated.items():
        assert len(clip.frames) == len(gesture_clip.frames), spec_id
        assert clip.fps == gesture_clip.fps, spec_id
        assert clip.duration_s == pytest.approx(gesture_clip.duration_s), spec_id
        for index, (before, after) in enumerate(zip(gesture_clip.frames, clip.frames)):
            assert after.time_s == pytest.approx(before.time_s), f"{spec_id} frame {index}"


def test_a_mutation_that_does_not_apply_raises_rather_than_returning_the_clip(specs):
    """A silently unmutated clip scored as a mutation is a false negative.

    And it looks exactly like a genuine gap in check coverage, which is the failure
    plan 06 section 6.1 is about.
    """
    full_body = compile_case(load_case(FULL_BODY_CASE))
    inapplicable = [spec for spec in specs if not spec.applies_to(full_body)]
    assert inapplicable, "expected the shake/present specs to be inapplicable here"
    for spec in inapplicable:
        with pytest.raises(NotApplicable, match=spec.id):
            spec.apply(full_body)


def test_applying_a_mutation_does_not_mutate_its_input(gesture_clip, specs):
    """``apply`` works on a deep copy; the corpus clip is shared across tests."""
    before = gesture_clip.model_dump(mode="json")
    for spec in specs:
        spec.apply(gesture_clip)
    assert gesture_clip.model_dump(mode="json") == before


def test_a_static_target_is_tagged_rather_than_skipped(specs):
    """Plan 06 section 6.1, corrected during 06a.

    A bone that never moves still yields a *capability* result -- base holds one
    pose, mutated holds another -- but not a threshold, because there is no motion
    for a threshold to sit inside.  Applied and tagged, never silently pooled.
    """
    full_body = compile_case(load_case(FULL_BODY_CASE))
    verdicts = [spec.applies_to(full_body) for spec in specs]
    applicable = [item for item in verdicts if item]
    assert applicable, "expected the wrist and finger specs to apply here"
    assert all(item.static_target for item in applicable), (
        "the walk case never moves its hand, so every applicable spec is a "
        "capability measurement rather than a threshold one"
    )
