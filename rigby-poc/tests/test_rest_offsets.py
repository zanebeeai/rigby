"""The rest-to-neutral offsets, checked against an independent instrument.

Plan 04 §6.4. Every limit in ``rom.v1.json`` is quoted anatomically and
converted through these offsets, so an error here shifts an entire limit band
rather than nudging one number. The failure mode is the one this plan already
walked into once: an earlier draft reported 60 degrees of elbow hyperextension
in motion whose elbow never passes straight, because it compared a
rest-relative delta against an anatomical reference.

The independent instrument is **clinical goniometry** -- the angle between two
segments read off their world positions, which is what a clinician measures and
which owes nothing to any rotation-axis convention in this codebase.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rigby_poc.analysis.anatomy.conventions import joint_class
from rigby_poc.analysis.anatomy.neutral import (
    NO_NEUTRAL,
    REST_IS_NEUTRAL,
    rest_offset,
    rest_offsets,
    to_rest_relative,
)
from rigby_poc.analysis.rig import canonical_bone_names, identity_bones
from rigby_poc.kinematics import rig_kinematics

pytestmark = pytest.mark.medium


def _goniometer(proximal: str, joint: str, distal: str) -> float:
    """Included angle at ``joint``, in degrees. 180 is a straight chain."""

    positions = rig_kinematics().canonical_positions(identity_bones())
    first = positions[joint] - positions[proximal]
    second = positions[distal] - positions[joint]
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return 180.0 - math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


@pytest.mark.parametrize(
    ("bone", "chain", "expected_deg"),
    [
        ("leftLowerArm", ("leftUpperArm", "leftLowerArm", "leftHand"), -2.6),
        ("rightLowerArm", ("rightUpperArm", "rightLowerArm", "rightHand"), -2.6),
        ("leftLowerLeg", ("leftUpperLeg", "leftLowerLeg", "leftFoot"), -11.5),
        ("rightLowerLeg", ("rightUpperLeg", "rightLowerLeg", "rightFoot"), -11.5),
    ],
)
def test_hinge_offsets_match_clinical_goniometry(bone, chain, expected_deg) -> None:
    """The two hinges are where goniometry and the derivation must agree exactly.

    A hinge's rest offset is a plain included angle, so there is no room for the
    two methods to differ for a defensible reason.
    """

    measured = _goniometer(*chain) - 180.0
    derived = math.degrees(rest_offset(bone).flexion_rad)

    assert measured == pytest.approx(expected_deg, abs=0.2)
    assert derived == pytest.approx(measured, abs=0.2)


def test_the_t_pose_is_ninety_degrees_of_shoulder_abduction() -> None:
    """The largest offset in the body, and the one §5 exists to warn about."""

    for side in ("left", "right"):
        offset = rest_offset(f"{side}UpperArm")
        assert math.degrees(offset.abduction_rad) == pytest.approx(-89.7, abs=0.5)
        assert abs(math.degrees(offset.flexion_rad)) < 1.0


def test_the_ankle_rests_plantarflexed() -> None:
    """Measured against a *neutral* shin, not the rest shin.

    A goniometer reads 18.2 degrees at rest; the offset is 27.1, and the gap is
    the shin's own 8.9 degrees of rest flexion. The second is the one a limit
    needs, because in the anatomical position the shin is vertical.
    """

    for side in ("left", "right"):
        assert math.degrees(rest_offset(f"{side}Foot").flexion_rad) == pytest.approx(
            -27.1, abs=0.5
        )
        included = _goniometer(f"{side}LowerLeg", f"{side}Foot", f"{side}Toes")
        assert 180.0 - included == pytest.approx(71.8, abs=0.5)


def test_the_spine_is_curved_at_neutral_so_its_offset_is_zero() -> None:
    """Not a shortcut -- an anatomical fact, and getting it wrong was a bug.

    The vertebral column carries a thoracic kyphosis and a cervical lordosis in
    the anatomical position, and this rig's bind pose reproduces them. Treating
    the chain as straight charged the neck a 40.9 degree flexion offset it does
    not have, which put the rig's own rest pose outside its own typical band.
    """

    for bone in canonical_bone_names():
        if joint_class(bone) not in REST_IS_NEUTRAL:
            continue
        offset = rest_offset(bone)
        assert offset is not None
        assert max(abs(value) for value in offset.as_degrees()) < 1e-6, bone

    # The curvature it would otherwise have charged is real and large.
    kinematics = rig_kinematics()
    directions = {}
    for bone in ("upperChest", "neck"):
        node = kinematics.node_by_canonical[bone]
        child = kinematics.nodes[node]["children"][0]
        vector = kinematics.rest_world[child][:3, 3] - kinematics.rest_world[node][:3, 3]
        directions[bone] = vector / np.linalg.norm(vector)
    between = math.degrees(
        math.acos(float(np.clip(np.dot(directions["upperChest"], directions["neck"]), -1, 1)))
    )
    assert between == pytest.approx(40.9, abs=0.5)


def test_the_thumb_has_no_offset_and_says_so() -> None:
    thumbs = [bone for bone in canonical_bone_names() if joint_class(bone) in NO_NEUTRAL]

    assert len(thumbs) == 6
    for bone in thumbs:
        assert rest_offset(bone) is None
        with pytest.raises(ValueError, match="no anatomical neutral"):
            to_rest_relative(bone, "flexion", 60.0)


def test_every_other_bone_has_an_offset() -> None:
    missing = [
        bone
        for bone, offset in rest_offsets().items()
        if offset is None and joint_class(bone) not in NO_NEUTRAL
    ]

    assert missing == []


def test_the_hand_axis_is_the_middle_metacarpal_not_the_first_child() -> None:
    """The rig lists the index metacarpal first; it is 13.8 degrees off the palm.

    Using it charged the wrist a 25 degree deviation offset that is an artifact
    of the child ordering. Plan §1.5 records the same fact as a 0.8984 cosine.
    """

    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(identity_bones())
    for side in ("left", "right"):
        node = kinematics.node_by_canonical[f"{side}Hand"]
        first_child = kinematics.nodes[node]["children"][0]
        assert kinematics.nodes[first_child]["name"].startswith("index")

        origin = positions[f"{side}Hand"]
        by_child = kinematics.rest_world[first_child][:3, 3] - origin
        by_middle = positions[f"{side}MiddleProximal"] - origin
        by_child /= np.linalg.norm(by_child)
        by_middle /= np.linalg.norm(by_middle)
        apart = math.degrees(math.acos(float(np.clip(np.dot(by_child, by_middle), -1, 1))))
        assert apart == pytest.approx(13.8, abs=0.5)

        assert abs(math.degrees(rest_offset(f"{side}Hand").abduction_rad)) < 12.0


def test_a_published_limit_converts_into_the_rest_frame() -> None:
    """``rest_relative = anatomical + offset``, the direction that bit once."""

    knee = rest_offset("leftLowerLeg")
    assert math.degrees(knee.flexion_rad) == pytest.approx(-11.5, abs=0.2)
    # 135 degrees of anatomical knee flexion is 123.5 rest-relative, because the
    # rig already rests 11.5 degrees into that range.
    assert to_rest_relative("leftLowerLeg", "flexion", 135.0) == pytest.approx(123.5, abs=0.3)


def test_the_shoulder_band_is_shifted_by_a_right_angle() -> None:
    """The offset with the most consequence, asserted as a converted bound.

    Written after an anti-tautology run showed the "rest pose is inside every
    typical band" check barely discriminates: most bands are wide enough to
    contain zero whether or not the offset is applied. This one does not have
    that problem -- 185 and 95 are not confusable.
    """

    from rigby_poc.analysis.anatomy.rom import rom_limit

    limit = rom_limit("leftUpperArm", "abduction")

    assert limit.max_deg == (-60.0, 185.0)  # anatomical, as published
    assert limit.to_rest_relative(185.0) == pytest.approx(95.3, abs=0.5)
    assert limit.to_rest_relative(-60.0) == pytest.approx(-149.7, abs=0.5)


def test_ignoring_the_offset_would_reject_ordinary_shoulder_motion() -> None:
    """What the conversion actually buys, stated as a consequence.

    Shipped shoulder abduction reaches 86 degrees rest-relative. Against the
    correctly shifted band that is comfortably inside. Against the unshifted
    anatomical band it is outside, so every arm-raising clip in the corpus
    would be rejected by a limit that is nominally 185 degrees wide.
    """

    from rigby_poc.analysis.anatomy.rom import rom_limit

    limit = rom_limit("leftUpperArm", "abduction")
    observed = -86.0

    assert limit.band_of(observed) == "within_typical"
    assert observed < limit.max_deg[0]  # outside the same band left unconverted
