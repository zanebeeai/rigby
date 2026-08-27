"""Morphology and manifest contracts refuse incoherent bodies.

Every rejection here is a body that would otherwise reach the bake and waste
twenty minutes before failing, or -- worse -- certify a primitive against a claim
nothing measured.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rigby_v2.contracts import Vec3

from conftest import build_manifest, build_morphology, make_site
from rigby_general.contracts import (
    DirectionV1,
    EffectorKind,
    EffectorV1,
    FrameSource,
    IntrinsicFrameV1,
    JointRole,
    MorphologyClass,
    RobotScaleV1,
    SiteSemantic,
)


# --------------------------------------------------------------------------
# Directions and frames
# --------------------------------------------------------------------------


def test_direction_must_be_a_unit_vector() -> None:
    with pytest.raises(ValidationError, match="unit vector"):
        DirectionV1(x=1.0, y=1.0, z=0.0)


def test_intrinsic_frame_must_be_orthogonal_and_right_handed() -> None:
    up = DirectionV1(x=0.0, y=0.0, z=1.0)
    front = DirectionV1(x=1.0, y=0.0, z=0.0)

    with pytest.raises(ValidationError, match="orthogonal"):
        IntrinsicFrameV1(
            up=up,
            front=up,
            lateral=front,
            source=FrameSource.DERIVED,
            confidence=0.5,
        )

    with pytest.raises(ValidationError, match="right-handed"):
        IntrinsicFrameV1(
            up=up,
            front=front,
            lateral=DirectionV1(x=0.0, y=-1.0, z=0.0),
            source=FrameSource.DERIVED,
            confidence=0.5,
        )


def test_operator_confirmed_frame_is_certain_by_definition() -> None:
    with pytest.raises(ValidationError, match="certain by definition"):
        IntrinsicFrameV1(
            up=DirectionV1(x=0.0, y=0.0, z=1.0),
            front=DirectionV1(x=1.0, y=0.0, z=0.0),
            lateral=DirectionV1(x=0.0, y=1.0, z=0.0),
            source=FrameSource.OPERATOR_CONFIRMED,
            confidence=0.8,
        )


# --------------------------------------------------------------------------
# Effectors
# --------------------------------------------------------------------------


def test_a_grasping_effector_needs_measured_closure_evidence() -> None:
    """Naming something a gripper is not evidence that it closes."""

    with pytest.raises(ValidationError, match="requires grip joints"):
        EffectorV1(
            name="claims_to_grip",
            kind=EffectorKind.PARALLEL_JAW,
            chain_id="arm",
            tip_body="tool",
            member_bodies=("jaw_a", "jaw_b"),
            site_names=("arm_tip",),
        )


def test_a_tool_tip_cannot_claim_closure_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot declare closure evidence"):
        EffectorV1(
            name="rigid_tip",
            kind=EffectorKind.TOOL_TIP,
            chain_id="arm",
            tip_body="tool",
            grip_joints=("grip_a",),
            site_names=("arm_tip",),
        )


def test_a_parallel_jaw_needs_two_opposition_groups() -> None:
    with pytest.raises(ValidationError, match="two opposition groups"):
        EffectorV1(
            name="one_sided",
            kind=EffectorKind.PARALLEL_JAW,
            chain_id="arm",
            tip_body="tool",
            member_bodies=("jaw_a", "jaw_b"),
            grip_joints=("grip_a",),
            opposition_groups=(("jaw_a", "jaw_b"),),
            max_aperture_m=0.08,
            site_names=("arm_tip",),
        )


def test_a_member_cannot_oppose_itself() -> None:
    with pytest.raises(ValidationError, match="two opposition groups"):
        EffectorV1(
            name="confused",
            kind=EffectorKind.PARALLEL_JAW,
            chain_id="arm",
            tip_body="tool",
            member_bodies=("jaw_a", "jaw_b"),
            grip_joints=("grip_a",),
            opposition_groups=(("jaw_a", "jaw_a"),),
            max_aperture_m=0.08,
            site_names=("arm_tip",),
        )


def test_multifinger_requires_three_converging_members() -> None:
    with pytest.raises(ValidationError, match="at least 3 converging members"):
        EffectorV1(
            name="two_fingers",
            kind=EffectorKind.MULTIFINGER,
            chain_id="arm",
            tip_body="tool",
            member_bodies=("jaw_a", "jaw_b"),
            grip_joints=("grip_a",),
            opposition_groups=(("jaw_a",), ("jaw_b",)),
            max_aperture_m=0.08,
            site_names=("arm_tip",),
        )


@pytest.mark.parametrize(
    "kind, expected_grasp",
    [
        (EffectorKind.PARALLEL_JAW, True),
        (EffectorKind.MULTIFINGER, True),
        (EffectorKind.TOOL_TIP, False),
    ],
)
def test_grasp_capability_follows_effector_kind(
    kind: EffectorKind, expected_grasp: bool
) -> None:
    built = build_morphology(effector_kind=kind)
    assert built.effectors[0].can_grasp is expected_grasp
    assert bool(built.grasping_effectors) is expected_grasp


# --------------------------------------------------------------------------
# Morphology consistency
# --------------------------------------------------------------------------


def test_morphology_rejects_a_chain_naming_an_unknown_joint(morphology) -> None:
    broken = morphology.model_dump(mode="python")
    broken["chains"][0]["joints"] = ("shoulder", "not_a_joint")

    with pytest.raises(ValidationError, match="unknown joints"):
        type(morphology).model_validate(broken)


def test_morphology_rejects_an_effector_naming_an_unknown_site(morphology) -> None:
    broken = morphology.model_dump(mode="python")
    broken["effectors"][0]["site_names"] = ("arm_tip", "imaginary_site")

    with pytest.raises(ValidationError, match="unknown sites"):
        type(morphology).model_validate(broken)


def test_bimanual_morphology_requires_two_grasping_chains(morphology) -> None:
    """Two hands is what makes a robot bimanual -- not a mirror plane.

    Symmetry is evidence about handedness and nothing more. Plenty of real
    dual-arm rigs are not mirrored, and left from right can still be told apart
    by the intrinsic lateral axis.
    """

    broken = morphology.model_dump(mode="python")
    broken["morphology_class"] = MorphologyClass.FIXED_BASE_BIMANUAL.value

    with pytest.raises(ValidationError, match="two grasping chains"):
        type(morphology).model_validate(broken)


def test_a_measured_mirror_plane_must_name_real_chains(morphology) -> None:
    from rigby_general.contracts import DirectionV1, SymmetryV1

    broken = morphology.model_dump(mode="python")
    broken["symmetry"] = SymmetryV1(
        plane_normal=DirectionV1(x=0.0, y=1.0, z=0.0),
        plane_point_m=Vec3(x=0.0, y=0.0, z=0.0),
        left_chain_id="arm",
        right_chain_id="a_chain_that_does_not_exist",
        residual_m=0.001,
    ).model_dump(mode="python")

    with pytest.raises(ValidationError, match="not a known chain"):
        type(morphology).model_validate(broken)


def test_scale_rejects_a_link_longer_than_the_whole_reach() -> None:
    with pytest.raises(ValidationError, match="cannot exceed total reach"):
        RobotScaleV1(
            reach_radius_m=0.5,
            characteristic_length_m=0.9,
            neutral_speed_mps=0.4,
            base_footprint_m=0.2,
            payload_kg=3.0,
            total_mass_kg=18.0,
            workspace_centroid_m=Vec3(x=0.0, y=0.0, z=0.0),
            reach_samples=128,
        )


def test_joint_lookup_reports_unknown_names(morphology) -> None:
    assert morphology.joint("elbow").role is JointRole.MAJOR_POSITION
    with pytest.raises(KeyError, match="unknown joint"):
        morphology.joint("no_such_joint")


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def test_manifest_dofs_must_match_the_measured_joints(morphology) -> None:
    built = build_manifest(morphology)
    broken = built.model_dump(mode="python")
    broken["dofs"] = broken["dofs"][:-1]
    broken["actuator_order"] = broken["actuator_order"][:-1]

    with pytest.raises(ValidationError, match="match the measured morphology joints"):
        type(built).model_validate(broken)


def test_manifest_sites_must_match_the_derived_sites(morphology) -> None:
    built = build_manifest(morphology)
    broken = built.model_dump(mode="python")
    broken["sites"] = list(broken["sites"]) + [
        make_site("hand_authored", "tool", SiteSemantic.TASK).model_dump(mode="python")
    ]

    with pytest.raises(ValidationError, match="match the derived morphology sites"):
        type(built).model_validate(broken)


def test_manifest_refuses_a_floating_base(morphology) -> None:
    built = build_manifest(morphology)
    broken = built.model_dump(mode="python")
    broken["fixed_base"] = False

    with pytest.raises(ValidationError, match="fixed-base robots only"):
        type(built).model_validate(broken)


def test_site_vocabulary_narrows_to_the_v2_human_names(morphology) -> None:
    """Downstream v2 code still expects fingertip/palm/foot/gaze/task."""

    narrowed = {site.name: site.semantic for site in build_manifest(morphology).rig_sites()}

    assert narrowed["arm_tip"] == "task"
    assert narrowed["arm_grasp_center"] == "palm"
    assert narrowed["robot_base"] == "task"
