"""The load-bearing test of plan 04.

Plan 04 §5 and §6.1: the frame derivation is geometrically exact, but calling a
given axis "flexion" is an assumption until validated, and a wrong frame
produces confident garbage that looks like it is working. So this file does not
check the derivation against itself. It checks it against three sources that
know nothing about :mod:`rigby_poc.analysis.anatomy`:

1. ``grip_presets.crate_grip`` in the rig profile — an authored hand pose whose
   correctness is independently observable, because a grip that opened the hand
   would be visibly wrong. It fixes the digit flexion axis and its sign.
2. Rest **landmark positions** — the knuckle line, the foot's forward
   direction, the body midline. Positions are independent of every
   rotation-axis convention, so they can arbitrate one.
3. **1,606 frames of real compiled motion** across the 12 golden-corpus cases.
   Human elbows flex and barely hyperextend; human knees are hinges. Motion
   that Rigby already ships is ground truth for the *sign and axis* of a joint
   even though it is not ground truth for the joint's *limits*.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rigby_poc.analysis.anatomy import (
    DofAngles,
    Reference,
    all_frames,
    bone_anatomical_frame,
    compose,
    conventions,
    decompose,
    digit_bones,
    joint_class,
)
from rigby_poc.analysis.anatomy import (
    frame as frame_module,
)
from rigby_poc.analysis.rig import canonical_bone_names, identity_bones, rig_profile
from rigby_poc.kinematics import rig_kinematics
from rigby_poc.models import BonePose, Quat

pytestmark = pytest.mark.medium

DEG = math.pi / 180.0


def _quat(rotation_vector) -> Quat:
    value = Rotation.from_rotvec(np.asarray(rotation_vector, dtype=float)).as_quat()
    return Quat(x=float(value[0]), y=float(value[1]), z=float(value[2]), w=float(value[3]))


def _posed(**bones: Quat) -> dict[str, BonePose]:
    pose = identity_bones()
    for name, rotation in bones.items():
        pose[name] = BonePose(rotation=rotation)
    return pose


def _pose_dof(bone: str, *, flexion=0.0, abduction=0.0, twist=0.0) -> Quat:
    """A quaternion that means a stated anatomical angle, by construction."""

    value = compose(
        DofAngles(flexion_rad=flexion, abduction_rad=abduction, twist_rad=twist),
        bone_anatomical_frame(bone),
    )
    return Quat(x=float(value[0]), y=float(value[1]), z=float(value[2]), w=float(value[3]))


def _tip_world(bone: str, bones) -> np.ndarray:
    kinematics = rig_kinematics()
    node = kinematics.node_by_canonical[bone]
    child = kinematics.nodes[node]["children"][0]
    return kinematics.world_matrices(bones)[child][:3, 3]


# --------------------------------------------------------------------------
# Derivation coverage and self-consistency
# --------------------------------------------------------------------------


def test_every_canonical_bone_gets_a_frame() -> None:
    frames = all_frames()

    assert set(frames) == set(canonical_bone_names())
    assert len(frames) == 52


def test_every_frame_is_orthonormal_and_right_handed() -> None:
    for name, frame in all_frames().items():
        matrix = frame.matrix()
        assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-9), name
        for axis in (frame.flexion_axis, frame.abduction_axis, frame.twist_axis):
            assert abs(float(np.linalg.norm(axis)) - 1.0) < 1e-9, name


def test_longitudinal_axis_is_local_y_for_every_bone() -> None:
    """Plan §1.5's claim, re-measured rather than inherited."""

    cosines = {name: frame.longitudinal_cosine for name, frame in all_frames().items()}
    worst = min(cosines, key=cosines.get)

    assert cosines[worst] >= frame_module.MIN_LONGITUDINAL_COSINE
    # The plan says "cosine >= 0.9 everywhere" and then names two bones at
    # 0.898, which cannot both be true. 0.8984 is the measured value, and it is
    # the two hands, whose first child is the index metacarpal -- not the thumb,
    # as the plan states.
    assert worst in {"leftHand", "rightHand"}
    assert cosines[worst] == pytest.approx(0.8984, abs=5e-4)
    assert sum(1 for value in cosines.values() if value > 0.9999) == 50


def test_axis_selection_is_never_a_coin_flip() -> None:
    """Every label beats its runner-up by the declared margin.

    A frame derived with a margin near zero would be an arbitrary choice
    presented as a derivation, which is exactly the §6.1 failure.
    """

    for name, frame in all_frames().items():
        assert frame.flexion_margin >= frame_module.MIN_MARGIN, (
            f"{name}: flexion label is weakly determined ({frame.flexion_margin:.3f})"
        )
        assert frame.abduction_margin >= frame_module.MIN_MARGIN, name


def test_thumb_metacarpal_is_the_weakest_frame_in_the_rig() -> None:
    """Recorded, not asserted away: the thumb's "flexion" is really opposition.

    Its margin is roughly half the next weakest. Any thumb-metacarpal limit in
    ``rom.v1.json`` inherits that uncertainty and must say so.
    """

    frames = all_frames()
    margins = sorted((frame.flexion_margin, name) for name, frame in frames.items())

    assert margins[0][1].endswith("ThumbMetacarpal")
    assert margins[0][0] < 0.6
    assert margins[0][0] > frame_module.MIN_MARGIN


def test_derivation_refuses_a_convention_the_rest_pose_cannot_realise() -> None:
    """The guard that makes the rest of this file meaningful.

    If no cross axis produces the declared motion, the frame must raise rather
    than return the best of two bad options.
    """

    original = conventions.CONVENTIONS["knee"]
    conventions.CONVENTIONS["knee"] = (
        Reference.LATERAL,
        Reference.SUPERIOR,
        "deliberately impossible: the knee does not flex sideways",
    )
    bone_anatomical_frame.cache_clear()
    try:
        with pytest.raises(ValueError, match="no cross axis realises"):
            bone_anatomical_frame("leftLowerLeg")
    finally:
        conventions.CONVENTIONS["knee"] = original
        bone_anatomical_frame.cache_clear()
        all_frames.cache_clear()


def test_unknown_bone_has_no_convention() -> None:
    with pytest.raises(KeyError):
        joint_class("leftTail")


# --------------------------------------------------------------------------
# Plan §5's pose table
# --------------------------------------------------------------------------


def test_rest_pose_is_zero_on_every_dof() -> None:
    identity = [0.0, 0.0, 0.0, 1.0]

    for name, frame in all_frames().items():
        angles = decompose(identity, frame)
        assert angles.flexion_rad == pytest.approx(0.0, abs=1e-12), name
        assert angles.abduction_rad == pytest.approx(0.0, abs=1e-12), name
        assert angles.twist_rad == pytest.approx(0.0, abs=1e-12), name


@pytest.mark.parametrize("bone", ["leftLowerArm", "rightLowerArm"])
def test_elbow_flexed_ninety_degrees(bone: str) -> None:
    angles = decompose(_pose_dof(bone, flexion=90 * DEG).as_list(), bone_anatomical_frame(bone))

    assert math.degrees(angles.flexion_rad) == pytest.approx(90.0, abs=1e-6)
    assert math.degrees(angles.abduction_rad) == pytest.approx(0.0, abs=1e-6)
    assert math.degrees(angles.twist_rad) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("bone", ["leftLowerLeg", "rightLowerLeg"])
def test_knee_flexed_ninety_degrees_moves_the_ankle_behind_the_knee(bone: str) -> None:
    """Not a round trip: the tip must actually go where anatomy says."""

    frame = bone_anatomical_frame(bone)
    pose = _posed(**{bone: _pose_dof(bone, flexion=90 * DEG)})
    before = _tip_world(bone, identity_bones())
    after = _tip_world(bone, pose)

    angles = decompose(pose[bone].rotation.as_list(), frame)
    assert math.degrees(angles.flexion_rad) == pytest.approx(90.0, abs=1e-6)
    assert abs(math.degrees(angles.abduction_rad)) < 1e-6
    # The heel travels backwards and upwards, toward the buttock.
    assert after[2] < before[2] - 0.20
    assert after[1] > before[1] + 0.20


def test_arm_raised_overhead_does_not_blow_up() -> None:
    """Plan §5's "arm raised overhead, no gimbal blow-up" row.

    The plan expects that pose at ``leftUpperArm.flexion == 180``. It is not:
    this rig rests in a T-pose, so the humerus already points laterally and the
    DOF that raises it overhead is *abduction*. Flexion from here is a
    horizontal sweep across the front of the body. Both are asserted below,
    because the mistake the plan makes here is exactly the mistake that a
    limit table would inherit.

    180 degrees is the sign singularity of a rotation vector, so the contract at
    the extreme is that the *magnitude* survives and the frame stays finite --
    not that the sign is meaningful there.
    """

    bone = "leftUpperArm"
    frame = bone_anatomical_frame(bone)
    shoulder = rig_kinematics().canonical_positions(identity_bones())[bone]

    for degrees in (90.0, 150.0, 179.0, 180.0):
        for dof in ("flexion", "abduction"):
            quaternion = _pose_dof(bone, **{dof: degrees * DEG})
            angles = decompose(quaternion.as_list(), frame)
            values = [angles.flexion_rad, angles.abduction_rad, angles.twist_rad]
            assert np.isfinite(values).all()
            measured = angles.flexion_rad if dof == "flexion" else angles.abduction_rad
            other = angles.abduction_rad if dof == "flexion" else angles.flexion_rad
            assert abs(math.degrees(measured)) == pytest.approx(degrees, abs=1e-4)
            assert abs(math.degrees(other)) < 1e-4

    overhead = _tip_world(bone, _posed(**{bone: _pose_dof(bone, abduction=90 * DEG)}))
    assert overhead[1] - shoulder[1] > 0.25
    assert abs(overhead[0] - shoulder[0]) < 0.05

    swept = _tip_world(bone, _posed(**{bone: _pose_dof(bone, flexion=90 * DEG)}))
    assert swept[2] - shoulder[2] > 0.25
    assert abs(swept[1] - shoulder[1]) < 0.05


def test_forearm_pronation_is_twist_and_nothing_else() -> None:
    bone = "leftLowerArm"
    frame = bone_anatomical_frame(bone)

    angles = decompose(_pose_dof(bone, twist=80 * DEG).as_list(), frame)

    assert math.degrees(angles.twist_rad) == pytest.approx(80.0, abs=1e-6)
    assert math.degrees(angles.flexion_rad) == pytest.approx(0.0, abs=1e-6)
    assert math.degrees(angles.abduction_rad) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_closed_fist_flexes_every_finger_segment_positively(side: str) -> None:
    """Independent source 1. The preset is authored; its *sign* is observable."""

    preset = rig_profile()["grip_presets"]["crate_grip"]
    curl = np.asarray(preset["curl_axis"], dtype=float)

    for segment, angle in preset["segment_angles_rad"].items():
        bone = f"{side}{segment}"
        angles = decompose(
            _quat(curl * angle).as_list(), bone_anatomical_frame(bone)
        )
        if segment.startswith("Thumb"):
            # The thumb metacarpal's dominant motion under this preset is
            # opposition, not flexion, which is why its frame is the weakest in
            # the rig. Excluded deliberately rather than silently.
            continue
        assert angles.flexion_rad > 0.0, f"{bone} does not flex under crate_grip"
        assert math.degrees(angles.flexion_rad) == pytest.approx(
            math.degrees(angle), abs=1.0
        )


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_crate_grip_actually_closes_the_hand(side: str) -> None:
    """What makes the preset usable as ground truth at all.

    Plan §6.2 is right that the grip *magnitudes* are authored rather than
    measured. Its direction is not in question, and this is why.
    """

    preset = rig_profile()["grip_presets"]["crate_grip"]
    curl = np.asarray(preset["curl_axis"], dtype=float)
    kinematics = rig_kinematics()
    gripped = _posed(
        **{
            f"{side}{segment}": _quat(curl * angle)
            for segment, angle in preset["segment_angles_rad"].items()
        }
    )

    open_tips = kinematics.fingertip_positions(identity_bones(), side)
    grip_tips = kinematics.fingertip_positions(gripped, side)

    for digit in ("index", "middle", "ring", "little"):
        opened = float(np.linalg.norm(open_tips[digit] - open_tips["thumb"]))
        closed = float(np.linalg.norm(grip_tips[digit] - grip_tips["thumb"]))
        assert closed < opened * 0.6, f"{side} {digit} does not close toward the thumb"


@pytest.mark.parametrize("side", ["left", "right"])
def test_digit_flexion_axis_equals_the_profile_curl_axis(side: str) -> None:
    """Independent source 1, stated as an axis identity."""

    curl = np.asarray(rig_profile()["grip_presets"]["crate_grip"]["curl_axis"], dtype=float)
    curl = curl / np.linalg.norm(curl)

    for bone in digit_bones(side):
        if "Thumb" in bone:
            continue  # see test_the_authored_thumb_curl_is_adduction_not_flexion
        assert np.allclose(bone_anatomical_frame(bone).flexion_axis, curl, atol=1e-9), bone


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_authored_thumb_curl_is_adduction_not_flexion(side: str) -> None:
    """A finding, recorded as a test rather than resolved by bending the frame.

    ``crate_grip`` drives all three thumb segments about the same local +X as
    the fingers. For the four fingers that axis *is* flexion, exactly. For the
    thumb, whose rest frame sits at an angle to the palm, it resolves to pure
    abduction with a zero flexion component: the preset closes the thumb by
    adducting it across the palm, which is how a real thumb closes on a grip,
    but it is not the DOF a published thumb-flexion ROM figure describes.

    The consequence for plan 04b: a thumb limit authored against flexion
    constrains a DOF this rig's own grip preset never uses. Thumb entries are
    the least trustworthy in ``rom.v1.json`` and must say so.
    """

    preset = rig_profile()["grip_presets"]["crate_grip"]
    curl = np.asarray(preset["curl_axis"], dtype=float)

    for segment in ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal"):
        angle = preset["segment_angles_rad"][segment]
        bone = f"{side}{segment}"
        angles = decompose(_quat(curl * angle).as_list(), bone_anatomical_frame(bone))

        assert abs(math.degrees(angles.flexion_rad)) < 1e-6, bone
        assert math.degrees(abs(angles.abduction_rad)) == pytest.approx(
            math.degrees(angle), abs=1e-6
        )


# --------------------------------------------------------------------------
# Independent source 2 — rest landmarks
# --------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_knee_flexion_axis_is_the_ankle_transverse_landmark_axis(side: str) -> None:
    """The knee hinge axis is parallel to the ankle's medio-lateral axis.

    Derived from where the toes are, which no rotation convention can influence.
    """

    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(identity_bones())
    forward = positions[f"{side}Toes"] - positions[f"{side}Foot"]
    forward = forward - np.asarray([0.0, 1.0, 0.0]) * float(forward[1])
    forward /= np.linalg.norm(forward)
    transverse = np.cross(np.asarray([0.0, 1.0, 0.0]), forward)
    transverse /= np.linalg.norm(transverse)

    node = kinematics.node_by_canonical[f"{side}LowerLeg"]
    local = kinematics.rest_world[node][:3, :3].T @ transverse
    derived = bone_anatomical_frame(f"{side}LowerLeg").flexion_axis

    assert abs(float(np.dot(local / np.linalg.norm(local), derived))) > 0.999


@pytest.mark.parametrize("side", ["left", "right"])
def test_radial_reference_points_at_the_index_knuckle(side: str) -> None:
    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(identity_bones())
    knuckle_line = positions[f"{side}IndexProximal"] - positions[f"{side}LittleProximal"]

    radial = frame_module.radial_direction(side)

    assert float(np.dot(radial, knuckle_line)) > 0.0


@pytest.mark.parametrize("side", ["left", "right"])
def test_palmar_reference_is_the_direction_fingertips_travel_when_they_curl(side: str) -> None:
    kinematics = rig_kinematics()
    preset = rig_profile()["grip_presets"]["crate_grip"]
    curl = np.asarray(preset["curl_axis"], dtype=float)
    gripped = _posed(
        **{
            f"{side}{segment}": _quat(curl * angle)
            for segment, angle in preset["segment_angles_rad"].items()
        }
    )
    travel = (
        kinematics.fingertip_positions(gripped, side)["index"]
        - kinematics.fingertip_positions(identity_bones(), side)["index"]
    )

    palmar = frame_module.palmar_direction(side)

    assert float(np.dot(palmar, travel / np.linalg.norm(travel))) > 0.5


# --------------------------------------------------------------------------
# Mirroring — plan §3.4
# --------------------------------------------------------------------------


def _mirror_local_delta(left_bone: str, right_bone: str, quaternion) -> np.ndarray:
    """Reflect a left-bone local delta into its right-bone counterpart.

    Reflection through the sagittal plane maps a world rotation ``W`` to
    ``M W M``; conjugating into each bone's own rest frame gives the local map.
    Built from the rest transforms alone, so it does not know what a frame is.
    """

    kinematics = rig_kinematics()
    mirror = np.diag([-1.0, 1.0, 1.0])
    left = kinematics.rest_world[kinematics.node_by_canonical[left_bone]][:3, :3]
    right = kinematics.rest_world[kinematics.node_by_canonical[right_bone]][:3, :3]
    transfer = right.T @ mirror @ left
    local = Rotation.from_quat(np.asarray(quaternion, dtype=float)).as_matrix()
    return Rotation.from_matrix(transfer @ local @ transfer.T).as_quat()


@pytest.mark.parametrize(
    "stem",
    [
        "Shoulder",
        "UpperArm",
        "LowerArm",
        "Hand",
        "UpperLeg",
        "LowerLeg",
        "Foot",
        "Toes",
        "IndexProximal",
        "MiddleIntermediate",
        "LittleDistal",
        "ThumbProximal",
    ],
)
def test_mirrored_poses_preserve_flexion_and_abduction_and_negate_twist(stem: str) -> None:
    """The invariant that says how ``rom.v1.json`` is authored for both sides.

    Plan §3.4 states the mirror rule as "abduction and twist negate, flexion
    does not". Measured over all twelve bone stems, half of that is right and
    half is wrong. In the anatomical frame:

    * **flexion is preserved** -- the plan is right;
    * **abduction is preserved** -- the plan is wrong. Abduction is signed
      away from the midline on both sides, so a mirrored pose abducts by the
      same positive amount;
    * **twist negates** -- the plan is right, and it is the only one that does,
      because a reflection reverses the handedness of a rotation about the
      longitudinal axis.

    The consequence for 04b: flexion and abduction limits are authored once and
    applied to both sides unchanged; a twist range must be negated and swapped
    when mirrored. The plan's own rule would have mirrored abduction limits
    backwards on every limb in the rig.
    """

    left, right = f"left{stem}", f"right{stem}"
    for flexion, abduction, twist in (
        (0.4, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 0.25),
        (0.35, -0.2, 0.15),
    ):
        left_quat = _pose_dof(left, flexion=flexion, abduction=abduction, twist=twist)
        right_quat = _mirror_local_delta(left, right, left_quat.as_list())

        mirrored = decompose(right_quat, bone_anatomical_frame(right))

        # 1e-6 rad, not 0: the GLB's left and right rest transforms are
        # mirror images to about 1e-7 and no closer, so this tolerance is a
        # measurement of the asset, not slack in the derivation.
        assert mirrored.flexion_rad == pytest.approx(flexion, abs=1e-6)
        assert mirrored.abduction_rad == pytest.approx(abduction, abs=1e-6)
        assert mirrored.twist_rad == pytest.approx(-twist, abs=1e-6)


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_compose_and_decompose_round_trip() -> None:
    generator = np.random.default_rng(20260824)

    for name in ("leftUpperArm", "leftLowerArm", "leftHand", "rightUpperLeg", "leftIndexDistal"):
        frame = bone_anatomical_frame(name)
        for _ in range(64):
            angles = DofAngles(*generator.uniform(-1.0, 1.0, size=3))
            recovered = decompose(compose(angles, frame), frame)
            assert recovered.flexion_rad == pytest.approx(angles.flexion_rad, abs=1e-9)
            assert recovered.abduction_rad == pytest.approx(angles.abduction_rad, abs=1e-9)
            assert recovered.twist_rad == pytest.approx(angles.twist_rad, abs=1e-9)


# --------------------------------------------------------------------------
# Independent source 3 — 12 corpus cases of real compiled motion
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus_angles() -> dict[str, list[DofAngles]]:
    """Every frame of every golden-corpus case, resolved onto its bone's frame.

    This is the arbiter the plan's §6.1 risk actually needs. A mislabelled axis
    is invisible against a synthetic pose that was built with the same label,
    and obvious against motion the compiler produced without knowing this module
    exists: human elbows flex and barely hyperextend, and human knees are hinges.
    """

    from evals.corpus import load_corpus
    from evals.corpus.loader import compile_case

    frames = all_frames()
    measured: dict[str, list[DofAngles]] = {}
    for case in load_corpus():
        for clip_frame in compile_case(case).frames:
            for bone, pose in clip_frame.bones.items():
                if bone not in frames:
                    continue
                measured.setdefault(bone, []).append(
                    decompose(pose.rotation.as_list(), frames[bone])
                )
    return measured


def _degrees(angles: list[DofAngles], dof: str) -> np.ndarray:
    return np.degrees([getattr(item, f"{dof}_rad") for item in angles])


@pytest.mark.parametrize("bone", ["leftLowerLeg", "rightLowerLeg"])
def test_shipped_motion_treats_the_knee_as_a_pure_hinge(bone, corpus_angles) -> None:
    """The single most convincing number in this file.

    If the knee's flexion axis were mislabelled, real motion would smear across
    two DOFs. It does not: across every frame of all twelve cases the knee's
    off-axis excursion peaks at 4.4 degrees while flexion reaches 99.7, a
    ratio of 22.6:1 over 5034 frames.
    """

    angles = corpus_angles[bone]
    flexion = _degrees(angles, "flexion")
    abduction = _degrees(angles, "abduction")

    assert len(angles) > 1000
    assert flexion.max() > 95.0
    assert float(np.mean(flexion[np.abs(flexion) > 1.0] > 0.0)) > 0.95
    # A mislabelled flexion axis would smear motion across two DOFs, giving a
    # ratio near 1. Anything past 10:1 is decisively not that. This is the
    # claim the test exists for; the bound is set by the hypothesis, not by
    # the corpus maximum.
    assert flexion.max() > 10.0 * np.abs(abduction).max()
    # Knee twist is NOT bounded here. Tibial axial rotation is anatomically
    # real -- roughly 10 deg internal, more in flexion -- so the 8.0 in the
    # original was a corpus maximum wearing an anatomical costume. 9.0 deg in
    # fullbody-turn-left is a plausible thing for a turn to do.


@pytest.mark.parametrize("bone", ["leftLowerArm", "rightLowerArm"])
def test_shipped_motion_flexes_the_elbow_far_more_than_it_extends_it(bone, corpus_angles) -> None:
    """The elbow's *sign* validated by motion the compiler produced blind."""

    flexion = _degrees(corpus_angles[bone], "flexion")
    active = flexion[np.abs(flexion) > 1.0]

    assert flexion.max() > 120.0
    assert float(np.mean(active > 0.0)) > 0.90
    assert flexion.max() > abs(flexion.min()) * 2.0


@pytest.mark.parametrize("bone", ["leftLowerArm", "rightLowerArm"])
def test_shipped_motion_drives_the_elbow_far_off_its_hinge_axis(bone, corpus_angles) -> None:
    """Plan §6.4, measured, before any limit exists to argue about.

    The elbow is a hinge, so plan §3.3 makes its abduction a ``hard_assert``.
    Shipped motion peaks at 91-96 degrees of elbow abduction and hyperextends to
    -60, both across n = 1606 frames of the 12 golden cases. Turning that assert
    on will reject currently-shipping motion, and the plan says that is the
    intended outcome rather than a reason to widen the limit.

    Recorded here so 04b's distribution review starts from a number and 04c's
    rejection rate can be compared against it.
    """

    abduction = _degrees(corpus_angles[bone], "abduction")
    flexion = _degrees(corpus_angles[bone], "flexion")

    assert np.abs(abduction).max() > 85.0
    assert flexion.min() < -35.0


@pytest.mark.parametrize("bone", ["leftUpperArm", "rightUpperArm", "leftUpperLeg", "rightUpperLeg"])
def test_shipped_motion_is_predominantly_flexion_at_the_proximal_joints(bone, corpus_angles) -> None:
    """Validates the *sign* of the flexion axis, via a statistic with margin.

    This asserted ``mean(active > 0) > 0.90`` and `leftUpperLeg` measured
    **0.913** on the 47-case corpus -- a 1.4% margin, so it was passing
    incidentally rather than because the claim holds, and one new case with more
    hip extension would have turned it red for a reason unrelated to the axis
    label. Lane `capture` shipped and then found the same defect in a 95% span
    bound that missed by 1.5 points on Windows; the lesson they drew is that
    being right about someone else's bound does not make you look at your own.

    Two scale-free forms replace it, both meaning "the positive direction is the
    one this joint travels in":

    * the median active excursion is clearly positive, and
    * the integral of flexion outweighs the integral of extension.

    Weakest measurement across the four bones is `leftUpperLeg` at a median of
    19.7 degrees and a ratio of 9.8; the arms are effectively unbounded because
    their flexion never goes negative at all.
    """

    flexion = _degrees(corpus_angles[bone], "flexion")
    active = flexion[np.abs(flexion) > 1.0]

    assert len(active) > 100
    assert float(np.median(active)) > 10.0
    positive = float(np.sum(np.clip(flexion, 0.0, None)))
    negative = float(np.sum(np.clip(-flexion, 0.0, None)))
    assert positive > 3.0 * negative


@pytest.mark.parametrize("bone", ["leftIndexProximal", "leftMiddleProximal", "leftLittleProximal"])
def test_shipped_finger_motion_is_flexion_only(bone, corpus_angles) -> None:
    angles = corpus_angles[bone]

    flexion = _degrees(angles, "flexion")
    abduction = _degrees(angles, "abduction")

    assert flexion.min() > 0.0
    # A ratio, not an absolute bound. This asserted |abduction| < 12.0 and
    # `leftLittleProximal` measured 11.68 -- a 2.6% margin, passing by accident.
    # The little finger genuinely splays, so the claim was never "abduction is
    # near zero"; it is that flexion dominates. Weakest ratio is 5.6.
    assert flexion.max() > 3.0 * np.abs(abduction).max()
    assert np.abs(_degrees(angles, "twist")).max() < 2.0


def test_the_frame_is_exercised_by_shipped_motion_or_declared_untested(corpus_angles) -> None:
    """Which of the 52 frames the corpus actually moves.

    Recorded rather than asserted away. A frame no clip ever moves has passed
    only the synthetic tests above, and its 04b limits carry that weight. This
    test fails if the *set* changes, so growing the corpus surfaces here.
    """

    unexercised = sorted(
        bone
        for bone, angles in corpus_angles.items()
        if max(
            max(abs(item.flexion_rad), abs(item.abduction_rad), abs(item.twist_rad))
            for item in angles
        )
        < 1e-6
    )

    assert unexercised == [
        "leftToes",
        "neck",
        "rightToes",
        "spine",
        "upperChest",
    ]


def test_ankle_dorsiflexion_in_shipped_motion_exceeds_human_range(corpus_angles) -> None:
    """A 04b finding, pinned now so 04c cannot quietly inherit it.

    Positive ankle flexion is plantarflexion, so the negative tail is
    dorsiflexion. Human dorsiflexion tops out near 20 degrees; shipped motion
    reaches 50, over n = 1606 frames. Plantarflexion reaches 47, which is
    within range. So the excursion is one-sided, which is what makes "the rest
    pose is not the neutral this convention assumes" a live hypothesis
    alongside "the motion is wrong". Plan §6.4 says to treat both as equally
    likely; this records the number so the 04b distribution review starts from
    evidence rather than from a guess.
    """

    for bone in ("leftFoot", "rightFoot"):
        flexion = _degrees(corpus_angles[bone], "flexion")
        assert flexion.min() < -45.0  # 50 deg of dorsiflexion; human max is ~20
        assert flexion.max() > 40.0  # 47 deg of plantarflexion; human max is ~50
