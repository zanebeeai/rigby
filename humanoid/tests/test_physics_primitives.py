"""Plan 10 §3.1's two primitives, and the datum the plan got wrong.

Everything here is synthetic: a pose dict is built from the rig's neutral
canonical positions and perturbed. No compile, no corpus -- so this is the
``fast`` tier and it says nothing about any real clip. The corpus-level
behaviour of the checks lives in ``test_physics_layer.py``.

**Every check here is posed to make the thing fire.** A penetration check
measured against the wrong datum reads zero on a clean rig, and so does a
penetration check wired to a constant; asserting "the clean rig does not
penetrate" cannot tell those apart. So the tests below remove the reason the
answer would be zero -- they push the rig through the floor and require a
number back. Named by lane `judge` as the expected-zero class.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigby_poc.analysis.physics import (
    POINT_MASSES,
    SEGMENTS,
    TOTAL_MASS_FRACTION,
    PhysicsError,
    _convex_hull,
    _margin_to_hull,
    center_of_mass,
    center_of_mass_series,
    foot_contacts,
    ground_height,
    ground_penetration_check,
    lowest_joint_height,
    support_polygon,
)
from rigby_poc.analysis.rig import rig_profile
from rigby_poc.kinematics import rig_kinematics
from rigby_poc.models import BonePose

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def neutral_pose() -> dict[str, np.ndarray]:
    kinematics = rig_kinematics()
    bones = {name: BonePose() for name in rig_profile()["bone_map"]}
    return kinematics.canonical_positions(bones)


def _shifted(pose: dict[str, np.ndarray], delta: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value, dtype=float) + delta for name, value in pose.items()
    }


# -- the mass model ---------------------------------------------------------------------


def test_the_segment_masses_sum_to_exactly_one() -> None:
    """Winter's table closes, so the CoM is a weighted mean and not a scaled one.

    If this drifts, every CoM in the layer is divided by the wrong total and the
    error is a uniform scaling -- which looks plausible and is wrong everywhere.
    """

    assert SEGMENTS, "the segment table is empty"
    assert POINT_MASSES, "the point-mass table is empty"
    assert TOTAL_MASS_FRACTION == pytest.approx(1.0, abs=1e-9)


def test_every_segment_names_bones_the_rig_actually_has() -> None:
    """A typo here would silently drop a segment's mass at runtime."""

    known = set(rig_profile()["bone_map"])
    assert known, "the rig profile exposes no bones"
    for proximal, distal, _mass, _along in SEGMENTS:
        assert proximal in known, proximal
        assert distal in known, distal
    for bone, _mass in POINT_MASSES:
        assert bone in known, bone


# -- the centre of mass -----------------------------------------------------------------


def test_the_neutral_centre_of_mass_sits_at_the_lumbar_spine(neutral_pose) -> None:
    """Validation against an independent landmark, not against itself.

    Plan 04's lesson, which applies verbatim here: a wrong frame produces
    confident garbage that looks like it is working. The same is true of a
    wrong mass model -- it returns a plausible point somewhere in the torso
    whatever the numbers are.

    So this asserts something anatomy fixes independently of the code: the
    standing whole-body centre of mass lies at the lumbar spine. Measured on
    this rig it is **1.0185 m against a spine joint at 1.0174 m, a 1.1 mm
    agreement** across a 1.55 m skeleton -- which no arrangement of wrong mass
    fractions produces by chance.

    A first pass compared it against 55-57% of stature and read 64.6%, which
    looked like a failure and was a bad denominator: the rig's `head` joint is
    the atlanto-occipital, not the crown, and the rig is T-posed, which raises a
    real centre of mass by about two points of stature.
    """

    com = center_of_mass(neutral_pose)
    spine = np.asarray(neutral_pose["spine"], dtype=float)
    assert abs(float(com[1]) - float(spine[1])) < 0.02, (
        f"CoM y={float(com[1]):.4f} against spine y={float(spine[1]):.4f}"
    )
    hips = float(neutral_pose["hips"][1])
    chest = float(neutral_pose["chest"][1])
    assert hips < float(com[1]) < chest, "the CoM left the pelvis-to-chest band"


def test_a_rigid_translation_moves_the_centre_of_mass_by_exactly_that_translation(
    neutral_pose,
) -> None:
    """The defining property of a centre of mass, and it is basis-independent.

    A structural relation rather than a magnitude -- ``docs/testing.md`` prefers
    those, and this one holds on any rig, any pose and any platform.
    """

    delta = np.asarray([0.37, -1.21, 0.05], dtype=float)
    before = center_of_mass(neutral_pose)
    after = center_of_mass(_shifted(neutral_pose, delta))
    assert after == pytest.approx(before + delta, abs=1e-9)


def test_a_symmetric_pose_has_no_lateral_centre_of_mass_offset(neutral_pose) -> None:
    """The neutral rig is mirror-symmetric, so the CoM must sit on the midline.

    This is what catches a left/right mass asymmetry -- a mistyped fraction on
    one arm shifts x while leaving the height, which the landmark test above
    would not see.
    """

    assert float(center_of_mass(neutral_pose)[0]) == pytest.approx(0.0, abs=1e-6)


def test_a_missing_segment_raises_rather_than_returning_a_partial_centre_of_mass(
    neutral_pose,
) -> None:
    """A CoM over some of the body is not a CoM.

    Returning one anyway is the absence-becoming-a-value shape; it would put a
    number in the output that is present and not derived from what it claims to
    describe, and every balance verdict downstream would be confidently wrong.
    """

    partial = dict(neutral_pose)
    del partial["leftUpperLeg"]
    with pytest.raises(PhysicsError, match="leftUpperLeg"):
        center_of_mass(partial)


def test_an_empty_clip_yields_an_empty_centre_of_mass_track() -> None:
    assert center_of_mass_series([]).shape == (0, 3)


# -- the ground plane, which the plan got wrong -----------------------------------------


def test_the_ground_plane_is_the_neutral_toe_and_is_not_the_origin() -> None:
    """Plan 10 §3.1 says "no collider below y = 0". The rig does not use y = 0.

    The floor is the lowest toe of the neutral pose. Measured: 0.015201 m. An
    origin datum reads every frame of every clip as 15.2 mm further above the
    floor than it is -- the *permissive* direction, so a penetration check on
    y=0 would sit at zero and its zero would be a claim about the datum rather
    than about the motion.
    """

    floor = ground_height()
    assert floor > 0.0, "the ground plane collapsed onto the origin"
    assert floor == pytest.approx(0.015201, abs=1e-5)


def test_a_rig_pushed_through_the_floor_reports_the_depth_it_was_pushed(
    neutral_pose,
) -> None:
    """The test that separates a real measurement from a constant zero.

    A clean rig reads zero penetration whether the check works or is wired to a
    literal, so the assertion has to remove the reason it is zero: drop the
    whole skeleton 40 mm and require 40 mm back.
    """

    floor = ground_height()
    drop = 0.040
    sunk = _shifted(neutral_pose, np.asarray([0.0, -drop, 0.0], dtype=float))

    clean = lowest_joint_height([neutral_pose], ground_height=floor)
    assert float(clean[0]) == pytest.approx(0.0, abs=1e-9)

    below = lowest_joint_height([sunk], ground_height=floor)
    assert float(below[0]) == pytest.approx(-drop, abs=1e-9)

    verdict = ground_penetration_check([sunk], floor=floor)
    assert verdict.failed, "a 40 mm penetration did not fail the check"
    assert float(verdict.measured) == pytest.approx(drop, abs=1e-9)
    assert verdict.headroom is not None and verdict.headroom < 0.0


def test_the_penetration_check_uses_every_joint_not_only_the_toes(
    neutral_pose,
) -> None:
    """A knee through the floor is a penetration the toes cannot see.

    Scoped to one bone so the test fails if the scan narrows back to the feet.
    """

    floor = ground_height()
    sunk_knee = dict(neutral_pose)
    sunk_knee["leftLowerLeg"] = np.asarray(
        neutral_pose["leftLowerLeg"], dtype=float
    ) - np.asarray([0.0, 0.60, 0.0], dtype=float)
    verdict = ground_penetration_check([sunk_knee], floor=floor)
    assert verdict.failed, "a knee driven below the floor was not detected"


def test_an_empty_clip_skips_the_penetration_check_rather_than_passing_it() -> None:
    verdict = ground_penetration_check([], floor=ground_height())
    assert verdict.status == "skip"
    assert verdict.headroom is None


# -- foot contact -----------------------------------------------------------------------


def test_a_zero_contact_band_is_refused(neutral_pose) -> None:
    """Testing for exact ground height reports contact on no frame at all.

    A gate that cannot fire, refused at the point it would be configured rather
    than discovered later as a clip with no stance phase.
    """

    with pytest.raises(PhysicsError, match="positive band"):
        foot_contacts([neutral_pose], ground_height=ground_height(), threshold_m=0.0)


def test_the_neutral_rig_is_standing_on_both_feet(neutral_pose) -> None:
    contacts = foot_contacts(
        [neutral_pose], ground_height=ground_height(), threshold_m=0.02
    )
    assert bool(contacts.left[0]) and bool(contacts.right[0])
    assert not bool(contacts.airborne[0])
    assert contacts.duty_factor == {"left": 1.0, "right": 1.0}


def test_a_lifted_foot_is_not_in_contact(neutral_pose) -> None:
    lifted = dict(neutral_pose)
    lifted["leftToes"] = np.asarray(neutral_pose["leftToes"], dtype=float) + np.asarray(
        [0.0, 0.25, 0.0], dtype=float
    )
    contacts = foot_contacts([lifted], ground_height=ground_height(), threshold_m=0.02)
    assert not bool(contacts.left[0])
    assert bool(contacts.right[0])
    assert not bool(contacts.airborne[0]), "one foot down is not airborne"


def test_a_jump_leaves_both_feet_off_the_ground(neutral_pose) -> None:
    airborne = _shifted(neutral_pose, np.asarray([0.0, 0.5, 0.0], dtype=float))
    contacts = foot_contacts(
        [airborne], ground_height=ground_height(), threshold_m=0.02
    )
    assert bool(contacts.airborne[0])
    assert contacts.duty_factor == {"left": 0.0, "right": 0.0}


def test_a_frame_missing_a_toe_raises_rather_than_guessing_contact(
    neutral_pose,
) -> None:
    partial = dict(neutral_pose)
    del partial["leftToes"]
    with pytest.raises(PhysicsError, match="leftToes"):
        foot_contacts([partial], ground_height=ground_height(), threshold_m=0.02)


# -- the support polygon ----------------------------------------------------------------


def test_the_hull_and_its_signed_margin_agree_with_hand_computed_geometry() -> None:
    """Unit square, four answers that can be checked without running anything."""

    hull = _convex_hull(np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    assert hull.shape[0] == 4
    assert _margin_to_hull(hull, np.asarray([0.5, 0.5])) == pytest.approx(0.5)
    assert _margin_to_hull(hull, np.asarray([0.1, 0.5])) == pytest.approx(0.1)
    assert _margin_to_hull(hull, np.asarray([-0.2, 0.5])) == pytest.approx(-0.2)
    # Nearest boundary point is the corner (1, 1), so the margin is -sqrt(2)/2.
    assert _margin_to_hull(hull, np.asarray([1.5, 1.5])) == pytest.approx(-(2**0.5) / 2)


def test_an_empty_support_region_cannot_contain_anything() -> None:
    assert _margin_to_hull(np.zeros((0, 2)), np.asarray([0.0, 0.0])) == float("-inf")


def test_a_single_foot_stance_still_has_a_support_area(neutral_pose) -> None:
    """The reason the footprint is modelled rather than taken as the two joints.

    The rig's foot is a two-joint chain with no width, so a bare-joint support
    polygon for single support is a line segment: the margin is non-positive by
    construction and the check could never pass in the one case it exists for.
    """

    single = support_polygon(neutral_pose, left=True, right=False)
    assert single.shape[0] >= 3, "single-foot support degenerated to a line"
    ankle = np.asarray(neutral_pose["leftFoot"], dtype=float)
    under_foot = np.asarray([ankle[0], ankle[2]], dtype=float)
    assert _margin_to_hull(single, under_foot) > 0.0, (
        "a point directly under the supporting ankle is outside its own footprint"
    )


def test_both_feet_give_a_wider_support_region_than_one(neutral_pose) -> None:
    """Double support must strictly contain single support at the midline."""

    both = support_polygon(neutral_pose, left=True, right=True)
    single = support_polygon(neutral_pose, left=True, right=False)
    midline = np.asarray([0.0, 0.0], dtype=float)
    assert _margin_to_hull(both, midline) > _margin_to_hull(single, midline)


def test_no_feet_in_contact_yields_no_support_region(neutral_pose) -> None:
    assert support_polygon(neutral_pose, left=False, right=False).shape == (0, 2)
