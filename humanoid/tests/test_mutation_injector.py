"""Instrument tests for the injector, which measure the harness not the code.

Four defects found across 03b, 04a and 04b share one shape: plausible, well-formed
output, no exception, no NaN, and a severity curve that still rises monotonically --
just wrongly.  ``test_mutation_severity_monotonic`` cannot catch any of them.  These
tests exist for no other purpose than to pin the injector, and each corresponds to a
defect that every other test in this suite would pass through.
"""

from __future__ import annotations

import math

import pytest

from evals.corpus import load_case
from evals.corpus.loader import compile_case
from evals.mutations.inject import (

    add_dof,
    bone_dof_series,
    is_static,
    peak_dof,
    signed_magnitude,
)

#: Compiles corpus cases, so `medium` by input rather than by duration.
pytestmark = pytest.mark.medium

#: The elbow carries flexion as well as abduction here, which is what makes it the
#: case that distinguishes adding in DOF coordinates from post-multiplying.
MOVING_CASE = "fullbody-dance"
#: The same bone, in a case where it never moves: 0.609 degrees in every frame.
#: (42.23 degrees before the humeral-roll fix; the constant was the roll artifact.)
STATIC_CASE = "fullbody-walk-forward"
ELBOW = "rightLowerArm"
#: The signing test needs a DOF whose largest-magnitude excursion is *negative*.
#: The elbow's abduction lost that property with the humeral-roll fix (its extremum
#: is now +9.69 deg on the dance clip), so the shoulder carries the fixture:
#: leftUpperArm.abduction swings to -55.65 deg at peak, measured on this clip.
SHOULDER = "leftUpperArm"


@pytest.fixture(scope="module")
def moving_clip():
    return compile_case(load_case(MOVING_CASE))


@pytest.fixture(scope="module")
def static_clip():
    return compile_case(load_case(STATIC_CASE))


@pytest.mark.parametrize("requested_deg", [5.0, 15.0, 30.0, 60.0])
def test_the_injector_delivers_the_magnitude_it_was_asked_for(moving_clip, requested_deg):
    """The defect this pins: ``existing * delta`` is not addition in DOF coordinates.

    Lane ``anatomy`` measured a 30-degree abduction injection delivering **43.2
    degrees** on this exact clip, because post-multiplication is exact only when the
    bone carries nothing but the target DOF.  Adding in DOF coordinates is exact.

    A sweep built on post-multiplication has levels that are non-linear in the
    requested magnitude while still rising with it, so the curve looks fine and the
    x-axis is wrong.
    """
    before = peak_dof(moving_clip, ELBOW, "abduction")
    mutated = add_dof(
        moving_clip.model_copy(deep=True), ELBOW, "abduction", math.radians(requested_deg)
    )
    after = peak_dof(mutated, ELBOW, "abduction")
    delivered_deg = math.degrees(after - before)
    assert delivered_deg == pytest.approx(requested_deg, abs=1e-6), (
        f"asked for {requested_deg} deg, delivered {delivered_deg:.3f}"
    )


def test_the_elbow_in_this_case_really_does_carry_a_second_dof(moving_clip):
    """Pins the *fixture*: the test above is only meaningful on a multi-DOF bone.

    On a bone carrying one DOF, post-multiplication and DOF addition agree, so the
    test above would pass against a broken injector.
    """
    flexion = [abs(value) for value in bone_dof_series(moving_clip, ELBOW, "flexion")]
    abduction = [abs(value) for value in bone_dof_series(moving_clip, ELBOW, "abduction")]
    assert max(flexion) > math.radians(5.0), math.degrees(max(flexion))
    assert max(abduction) > math.radians(5.0), math.degrees(max(abduction))


def test_a_delta_is_signed_to_increase_the_excursion_it_perturbs():
    """The defect this pins: +30 into a bone at -41.8 reduces the peak by 30.

    Correct arithmetic, useless measurement, and in a sweep it reads as a detection
    *failure* at high severity -- the shape of a real finding.
    """
    assert signed_magnitude([-0.7, -0.2, 0.1], 0.5) == pytest.approx(-0.5)
    assert signed_magnitude([0.7, 0.2, -0.1], 0.5) == pytest.approx(0.5)
    assert signed_magnitude([], 0.5) == pytest.approx(0.5)
    # The sign follows the extremum of largest magnitude, not the last value.
    assert signed_magnitude([-0.9, 0.3], 0.5) == pytest.approx(-0.5)


def test_signing_against_the_excursion_increases_the_peak_rather_than_cancelling_it(
    moving_clip,
):
    """The end-to-end version of the rule above, on a real clip."""
    before = peak_dof(moving_clip, SHOULDER, "abduction")
    signed = add_dof(
        moving_clip.model_copy(deep=True), SHOULDER, "abduction", math.radians(20.0)
    )
    unsigned = add_dof(
        moving_clip.model_copy(deep=True),
        SHOULDER,
        "abduction",
        math.radians(20.0),
        sign_from_clip=False,
    )
    assert peak_dof(signed, SHOULDER, "abduction") > before
    # The clip's shoulder abduction is negative at peak, so an unsigned +20 cancels.
    assert peak_dof(unsigned, SHOULDER, "abduction") < before


def test_the_signing_fixture_really_swings_negative(moving_clip):
    """Pins the *fixture* for the test above: the signing test only distinguishes
    signed from unsigned on a DOF whose largest-magnitude excursion is negative.
    The elbow held that role until the humeral-roll fix moved its extremum to
    +9.69 degrees, which silently made signed and unsigned agree there."""
    series = bone_dof_series(moving_clip, SHOULDER, "abduction")
    extremum = max(series, key=abs)
    assert extremum < 0.0, math.degrees(extremum)
    assert abs(extremum) > math.radians(20.0), math.degrees(extremum)


def test_a_static_bone_is_detected_as_static(static_clip, moving_clip):
    """The defect this pins: mutating a bone that never moves perturbs a constant.

    Detection would be trivially perfect and would measure nothing.  The median
    corpus case moves 18 of 52 bones; the quietest moves 4.
    """
    assert is_static(static_clip, ELBOW)
    assert not is_static(moving_clip, ELBOW)


def test_the_static_case_really_is_static_to_the_last_decimal(static_clip):
    """Pins the *fixture* for the test above, the way `anatomy` pinned theirs."""
    series = bone_dof_series(static_clip, ELBOW, "abduction")
    assert len(series) > 50
    assert len({round(value, 9) for value in series}) == 1
    assert math.degrees(abs(series[0])) == pytest.approx(0.609, abs=0.01)


def test_injection_is_deterministic(moving_clip):
    """Same input, byte-identical output.

    An injector that varies run to run cannot be used to measure detection: a
    changed verdict would be indistinguishable from a changed mutation.
    """
    first = add_dof(moving_clip.model_copy(deep=True), ELBOW, "abduction", 0.3)
    second = add_dof(moving_clip.model_copy(deep=True), ELBOW, "abduction", 0.3)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_injection_leaves_other_bones_untouched(moving_clip):
    mutated = add_dof(moving_clip.model_copy(deep=True), ELBOW, "abduction", 0.3)
    for index, (before, after) in enumerate(zip(moving_clip.frames, mutated.frames)):
        for name in before.bones:
            if name == ELBOW:
                continue
            assert before.bones[name].rotation.as_list() == after.bones[name].rotation.as_list(), (
                f"frame {index} bone {name} moved"
            )
