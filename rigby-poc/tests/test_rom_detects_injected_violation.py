"""Inject a known excursion; assert the check reports that excursion.

Plan 04 §5. This is the metamorphic test, and it is the only one in the plan
that proves the range-of-motion check *measures what it claims to measure*
rather than merely producing numbers. Everything else in 04b establishes that
the table is complete and cited; this establishes that it is connected to the
motion.

The injection is deliberately built from :func:`compose`, so the amount
injected is stated in anatomical degrees rather than as a quaternion someone
believed corresponded to one. Lane `groundtruth` owns the general mutation
library (plan 06); this helper is the narrowest thing that tests the check, and
06b converges it onto ``MutationSpec`` once that exists -- sequencing agreed
with them, since 06a is gated behind 03b and their 06b behind this PR.
"""

from __future__ import annotations

import math

import pytest

from evals.corpus import load_corpus
from evals.corpus.loader import compile_case
from rigby_poc.analysis.anatomy.frame import (
    DofAngles,
    bone_anatomical_frame,
    compose,
    decompose,
)
from rigby_poc.analysis.anatomy.rom import rom_violations
from rigby_poc.models import BonePose, Quat

pytestmark = pytest.mark.medium

#: The clip the corpus-based assertions use. It is chosen for one property and
#: `test_the_chosen_case_has_a_moving_elbow` pins it: **the elbow must actually
#: move**. Lane `groundtruth` found that several full-body cases hold the elbow
#: at a single value to the last decimal for their whole duration --
#: `fullbody-step-over-hurdle`, which this test originally used, sits at
#: -41.849 degrees for all 76 frames, std 0.000. Injecting into a constant is a
#: clean, perfectly detected result that measures the injector rather than the
#: check.
CASE = "fullbody-dance"
DEG = math.pi / 180.0


@pytest.fixture(scope="module")
def clip():
    case = next(item for item in load_corpus() if item.entry.id == CASE)
    return compile_case(case)


@pytest.fixture(scope="module")
def rest_clip(clip):
    """A clip of the rig at its rest pose, borrowing a real clip's timing.

    The plan asks for the injection to go into a corpus clip. It cannot, for
    the elbow, and the reason is 04b's headline measurement: **no corpus case is
    clean on elbow abduction.** All twelve exceed the 5-degree hinge bound, the
    quietest of them peaking at 42.2 degrees. There is no unmutated baseline to
    inject into, so a size assertion against one would be measuring the sum of
    the injection and a defect.

    So the control is synthetic and clean by construction, and
    :func:`test_a_corpus_clip_reports_the_injected_increase` covers the realism
    the plan was reaching for by asserting the *delta* instead.
    """

    identity = Quat()
    frames = [
        frame.model_copy(
            update={"bones": {name: BonePose(rotation=identity) for name in frame.bones}}
        )
        for frame in clip.frames
    ]
    return frames


def _inject(frames, bone: str, *, dof: str, degrees: float):
    """Add a stated anatomical excursion to one DOF, on every frame.

    Decompose, add, recompose -- **not** a post-multiplied delta rotation.
    Post-multiplying looks equivalent and is not: quaternion composition is not
    addition in the decomposed coordinates, so when the bone already carries
    flexion the abduction actually delivered differs from the abduction asked
    for. Measured on `fullbody-dance` frame 0, a 30-degree post-multiplied
    injection produced a 43.2-degree change. It went unnoticed for as long as
    the test used a clip whose elbow was static in a near-pure abduction pose.

    Decompose-add-recompose is exact because `decompose` and `compose` round
    trip to 1e-9, and it is also the right *meaning* for a range-of-motion
    mutation: "this DOF, plus X degrees", not "times this rotation".
    """

    frame_obj = bone_anatomical_frame(bone)
    mutated = []
    for frame in frames:
        bones = dict(frame.bones)
        angles = decompose(bones[bone].rotation.as_list(), frame_obj)
        shifted = DofAngles(
            **{
                f"{name}_rad": getattr(angles, f"{name}_rad")
                + (degrees * DEG if name == dof else 0.0)
                for name in ("flexion", "abduction", "twist")
            }
        )
        value = compose(shifted, frame_obj)
        bones[bone] = BonePose(
            rotation=Quat(
                x=float(value[0]), y=float(value[1]), z=float(value[2]), w=float(value[3])
            ),
            position=bones[bone].position,
        )
        mutated.append(frame.model_copy(update={"bones": bones}))
    return mutated


def _violation(violations, bone: str, dof: str):
    return next((v for v in violations if v.bone == bone and v.dof == dof), None)


def test_the_rest_pose_clip_violates_nothing_at_all(rest_clip, clip) -> None:
    """The control. Without it, every detection below proves nothing.

    Also the strongest single statement that the rest offsets are right: if any
    limit were on the wrong band, the rig standing still would trip it.
    """

    assert rom_violations(rest_clip, fps=clip.fps) == []


def test_the_chosen_case_has_a_moving_elbow(clip) -> None:
    """Guards this file's own instrument against a silent regression.

    A corpus clip whose elbow never moves would make every assertion below pass
    for the wrong reason. 42 of the corpus's cases would fail this.
    """

    frame_obj = bone_anatomical_frame("leftLowerArm")
    values = [
        math.degrees(decompose(frame.bones["leftLowerArm"].rotation.as_list(), frame_obj).abduction_rad)
        for frame in clip.frames
    ]

    assert len({round(value, 6) for value in values}) > 20
    assert max(values) - min(values) > 20.0


def test_no_corpus_case_is_clean_on_elbow_abduction(clip) -> None:
    """04b's headline measurement, pinned as the reason the control is synthetic.

    Every case with frames in it exceeds the 5-degree hinge bound on at least one
    elbow. Asserted as "all of them" rather than as a count, so growing the
    corpus does not turn this red for the wrong reason -- and because the count
    was never the claim. Lane `groundtruth` re-measured over 47 cases: still all
    of them, the only exception being a known-bad case that compiles to zero
    frames.

    This is report-only, so nothing fails today. 04c is where it starts
    rejecting, and plan §6.4 says that is the intended outcome.
    """

    clean, dirty = [], []
    for case in load_corpus():
        compiled = compile_case(case)
        if not compiled.frames:
            continue
        violations = rom_violations(compiled.frames, fps=compiled.fps)
        target = dirty if any(
            v.dof == "abduction" and v.band == "beyond_max" and v.bone.endswith("LowerArm")
            for v in violations
        ) else clean
        target.append(case.entry.id)

    assert dirty, "no corpus case compiled any frames"
    assert clean == [], f"cases with a clean elbow now exist: {clean}"


@pytest.mark.parametrize("degrees", [30.0, 60.0])
def test_an_injected_off_axis_elbow_rotation_is_reported_at_its_own_size(rest_clip, clip, degrees) -> None:
    """Plan §5's row, with the size asserted rather than just the detection."""

    mutated = _inject(rest_clip, "leftLowerArm", dof="abduction", degrees=degrees)

    found = _violation(rom_violations(mutated, fps=clip.fps), "leftLowerArm", "abduction")

    assert found is not None, f"{degrees} deg of elbow abduction went undetected"
    assert found.band == "beyond_max"
    assert found.hard_assert
    assert abs(found.peak_deg) == pytest.approx(degrees, abs=1.5)


def test_severity_is_ordered_by_size_not_merely_present(rest_clip, clip) -> None:
    """A graded report, not a boolean. Plan §3.5."""

    small = _violation(
        rom_violations(_inject(rest_clip, "leftLowerArm", dof="abduction", degrees=15.0),
                       fps=clip.fps), "leftLowerArm", "abduction")
    large = _violation(
        rom_violations(_inject(rest_clip, "leftLowerArm", dof="abduction", degrees=60.0),
                       fps=clip.fps), "leftLowerArm", "abduction")

    assert abs(small.peak_deg) < abs(large.peak_deg)
    assert small.integral_deg_s < large.integral_deg_s


def test_a_single_frame_blip_and_a_sustained_excursion_are_distinguishable(rest_clip, clip) -> None:
    """The distinction the boolean check this replaces cannot express.

    Same peak, two orders of magnitude apart in integral. Plan §3.5 gates on
    the integral and triages on the peak precisely for this.
    """

    sustained = _inject(rest_clip, "leftLowerArm", dof="abduction", degrees=40.0)
    blip = list(rest_clip)
    blip[len(blip) // 2] = _inject(
        [blip[len(blip) // 2]], "leftLowerArm", dof="abduction", degrees=40.0
    )[0]

    sustained_v = _violation(rom_violations(sustained, fps=clip.fps), "leftLowerArm", "abduction")
    blip_v = _violation(rom_violations(blip, fps=clip.fps), "leftLowerArm", "abduction")

    assert blip_v is not None, "a one-frame excursion must still be reported"
    assert blip_v.frames == 1
    assert sustained_v.frames > 50
    assert abs(blip_v.peak_deg) == pytest.approx(abs(sustained_v.peak_deg), abs=2.0)
    assert sustained_v.integral_deg_s > 20.0 * blip_v.integral_deg_s


def test_a_corpus_clip_reports_the_injected_increase(clip) -> None:
    """The realism half, on a clip that already carries a real defect.

    The absolute peak is meaningless here because the baseline is not clean, so
    the assertion is on the increase -- which is the quantity the injection
    actually controls.

    The injection is signed to match the existing excursion. Injecting +30 into
    a clip whose elbow already sits at -41.8 degrees of abduction *reduces* the
    peak by 30, which is correct behaviour and a useless assertion.
    """

    before = _violation(rom_violations(clip.frames, fps=clip.fps), "leftLowerArm", "abduction")
    direction = math.copysign(30.0, before.peak_deg)
    mutated = _inject(clip.frames, "leftLowerArm", dof="abduction", degrees=direction)
    after = _violation(rom_violations(mutated, fps=clip.fps), "leftLowerArm", "abduction")

    assert before is not None and after is not None
    assert abs(after.peak_deg) - abs(before.peak_deg) == pytest.approx(30.0, abs=3.0)


def test_injection_into_one_dof_does_not_leak_into_the_others(clip) -> None:
    """If it leaked, the decomposition would not be a decomposition."""

    mutated = _inject(clip.frames, "leftLowerLeg", dof="flexion", degrees=25.0)
    baseline = {
        (v.bone, v.dof) for v in rom_violations(clip.frames, fps=clip.fps)
    }
    after = {(v.bone, v.dof) for v in rom_violations(mutated, fps=clip.fps)}

    introduced = after - baseline
    assert all(bone == "leftLowerLeg" for bone, _ in introduced), introduced
    assert all(dof == "flexion" for _, dof in introduced), introduced


def test_the_injection_helper_actually_injects_what_it_says(clip) -> None:
    """Guards the test's own instrument. A broken injector fakes a pass."""

    frame_obj = bone_anatomical_frame("leftLowerArm")
    before = decompose(clip.frames[0].bones["leftLowerArm"].rotation.as_list(), frame_obj)
    mutated = _inject(clip.frames, "leftLowerArm", dof="abduction", degrees=30.0)
    after = decompose(mutated[0].bones["leftLowerArm"].rotation.as_list(), frame_obj)

    assert math.degrees(after.abduction_rad - before.abduction_rad) == pytest.approx(30.0, abs=1e-6)
