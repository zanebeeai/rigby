"""Morphology is measured, not named.

The zoo is built specifically to punish an analyser that reads link names or
assumes a shape: reach spans a factor of five, effector counts run from one rigid
tip to three articulated digits, and one robot carries a camera that must not be
mistaken for a hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigby_general.contracts import (
    EffectorKind,
    JointRole,
    MorphologyClass,
    SiteSemantic,
)
from rigby_general.pipeline import ingest_robot


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"
ZOO_IDS = sorted(
    directory.name for directory in ZOO_ROOT.iterdir() if (directory / "robot.urdf").is_file()
)


def ground_truth(robot_id: str) -> dict:
    return json.loads(
        (ZOO_ROOT / robot_id / "ground_truth.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def ingested() -> dict:
    return {
        robot_id: ingest_robot(ZOO_ROOT / robot_id / "robot.urdf", robot_id=robot_id)
        for robot_id in ZOO_IDS
    }


def test_the_zoo_is_actually_varied() -> None:
    """A zoo of six near-identical arms would prove nothing about generality."""

    assert len(ZOO_IDS) >= 6


@pytest.mark.parametrize("robot_id", ZOO_IDS)
def test_every_robot_ingests(robot_id: str, ingested: dict) -> None:
    robot = ingested[robot_id]
    assert robot.integrity.passed
    assert robot.manifest.rig_id == robot_id
    assert robot.manifest.dofs
    assert robot.mjcf_xml.startswith("<mujoco")


@pytest.mark.parametrize("robot_id", ZOO_IDS)
def test_derived_sites_match_ground_truth(robot_id: str, ingested: dict) -> None:
    """Requirement ``site_derivation_accuracy``.

    Keyed on ``(semantic, body)`` rather than site name: the names are the
    analyser's own output, so checking them against itself would prove nothing.
    """

    truth = ground_truth(robot_id)
    expected = {(item["semantic"], item["body"]) for item in truth["expected_sites"]}
    produced = {
        (site.semantic.value, site.body) for site in ingested[robot_id].morphology.sites
    }

    missing = expected - produced
    accuracy = len(expected & produced) / len(expected)
    assert accuracy >= 0.95, f"{robot_id} missing derived sites: {sorted(missing)}"


@pytest.mark.parametrize("robot_id", ZOO_IDS)
def test_effector_classification_matches_ground_truth(
    robot_id: str, ingested: dict
) -> None:
    """Requirement ``effector_classification``."""

    truth = ground_truth(robot_id)
    expected = sorted(item["kind"] for item in truth["expected_effectors"])
    produced = sorted(
        effector.kind.value for effector in ingested[robot_id].morphology.effectors
    )
    assert produced == expected


@pytest.mark.parametrize("robot_id", ZOO_IDS)
def test_morphology_class_matches_ground_truth(robot_id: str, ingested: dict) -> None:
    truth = ground_truth(robot_id)
    assert ingested[robot_id].morphology.morphology_class.value == truth["morphology_class"]


def test_the_zoo_covers_every_effector_kind(ingested: dict) -> None:
    """Requirement ``effector_classification`` demands the coverage, not just accuracy."""

    kinds = {
        effector.kind
        for robot in ingested.values()
        for effector in robot.morphology.effectors
    }
    assert EffectorKind.PARALLEL_JAW in kinds
    assert EffectorKind.MULTIFINGER in kinds
    assert EffectorKind.TOOL_TIP in kinds
    assert EffectorKind.SENSOR in kinds


def test_reach_spans_a_wide_range(ingested: dict) -> None:
    """Magnitude neutrality is only tested by bodies of very different size."""

    reaches = [robot.morphology.scale.reach_radius_m for robot in ingested.values()]
    assert max(reaches) / min(reaches) >= 4.0


# --------------------------------------------------------------------------
# The behavioural gripper test
# --------------------------------------------------------------------------


def test_a_rigid_tip_is_not_a_gripper(ingested: dict) -> None:
    """Nothing on the tool arm closes, so nothing on it may be called a jaw."""

    effectors = ingested["zoo_tool_arm"].morphology.effectors
    assert [effector.kind for effector in effectors] == [EffectorKind.TOOL_TIP]
    assert not effectors[0].can_grasp
    assert effectors[0].grip_joints == ()


def test_a_camera_is_not_an_effector(ingested: dict) -> None:
    """The long arm's wrist camera must stay a gaze source, not become a finger."""

    by_kind = {
        effector.kind: effector
        for effector in ingested["zoo_long_arm"].morphology.effectors
    }
    assert EffectorKind.SENSOR in by_kind
    assert not by_kind[EffectorKind.SENSOR].can_grasp
    gaze = [
        site
        for site in ingested["zoo_long_arm"].morphology.sites
        if site.semantic is SiteSemantic.GAZE
    ]
    assert gaze and "name_hint" in gaze[0].derivation


def test_parallel_jaw_opposition_is_one_finger_against_the_other(
    ingested: dict,
) -> None:
    effector = ingested["zoo_jaw_arm"].morphology.effectors[0]
    assert effector.kind is EffectorKind.PARALLEL_JAW
    assert len(effector.opposition_groups) == 2
    assert all(len(group) == 1 for group in effector.opposition_groups)
    assert effector.max_aperture_m and effector.max_aperture_m > 0.0


def test_tripod_hand_opposes_one_digit_against_two(ingested: dict) -> None:
    """The grip a three-finger hand actually makes, recovered from geometry."""

    effector = ingested["zoo_hand_arm"].morphology.effectors[0]
    assert effector.kind is EffectorKind.MULTIFINGER
    assert len(effector.opposition_groups) == 2
    assert sorted(len(group) for group in effector.opposition_groups) == [1, 2]


# --------------------------------------------------------------------------
# Joint roles
# --------------------------------------------------------------------------


def test_base_yaw_is_a_positioning_joint_not_a_wrist(ingested: dict) -> None:
    """Measured at the parked pose alone, a base yaw looks like it does nothing.

    A vertically parked arm puts its tip on the base rotation axis, so sweeping
    that joint from rest moves the tip by millimetres. It is still the joint that
    sweeps the entire workspace.
    """

    for robot_id in ("zoo_jaw_arm", "zoo_hand_arm", "zoo_tool_arm"):
        joints = {
            joint.name: joint for joint in ingested[robot_id].morphology.joints
        }
        assert joints["joint_0"].role is JointRole.MAJOR_POSITION, robot_id


def test_a_seven_axis_arm_has_exactly_one_redundant_joint(ingested: dict) -> None:
    """Kinematic redundancy against the full six-row pose Jacobian."""

    roles = [joint.role for joint in ingested["zoo_hand_arm"].morphology.joints]
    assert roles.count(JointRole.REDUNDANT) == 1


@pytest.mark.parametrize("robot_id", ["zoo_jaw_arm", "zoo_tool_arm", "zoo_long_arm"])
def test_non_redundant_arms_report_no_redundancy(robot_id: str, ingested: dict) -> None:
    roles = [joint.role for joint in ingested[robot_id].morphology.joints]
    assert JointRole.REDUNDANT not in roles


@pytest.mark.parametrize("robot_id", ZOO_IDS)
def test_grip_joints_are_classified_by_closure(robot_id: str, ingested: dict) -> None:
    morphology = ingested[robot_id].morphology
    grip_named = {
        joint.name for joint in morphology.joints if joint.role is JointRole.GRIP
    }
    declared = {
        name for effector in morphology.effectors for name in effector.grip_joints
    }
    assert grip_named == declared


# --------------------------------------------------------------------------
# Frames and scale
# --------------------------------------------------------------------------


def test_up_is_always_against_gravity(ingested: dict) -> None:
    for robot in ingested.values():
        up = robot.morphology.intrinsic_frame.up
        assert pytest.approx(up.z, abs=1e-6) == 1.0


def test_a_symmetric_pedestal_arm_admits_no_derived_front(ingested: dict) -> None:
    """The honest answer, and the reason a person confirms it at upload.

    A shoulder that can pitch the arm behind the base makes the reachable set
    almost symmetric, so no direction is distinguished. Reporting a confident
    front here would mean rendering a confidently backwards motion later.
    """

    frame = ingested["zoo_jaw_arm"].morphology.intrinsic_frame
    assert frame.confidence == 0.0
    assert "no_evidence_default_axis" in frame.evidence


def test_a_two_armed_robot_derives_front_from_its_mirror_plane(ingested: dict) -> None:
    frame = ingested["zoo_dual_arm"].morphology.intrinsic_frame
    assert frame.confidence >= 0.5
    assert "mirror_plane" in frame.evidence
    assert ingested["zoo_dual_arm"].morphology.symmetry is not None


def test_operator_confirmation_makes_a_frame_certain(ingested: dict) -> None:
    from rigby_general.contracts import DirectionV1, FrameSource
    from rigby_general.morphology import confirm_frame

    proposed = ingested["zoo_jaw_arm"].morphology.intrinsic_frame
    confirmed = confirm_frame(proposed, front=DirectionV1(x=0.0, y=1.0, z=0.0))

    assert confirmed.source is FrameSource.OPERATOR_CONFIRMED
    assert confirmed.confidence == 1.0
    assert pytest.approx(confirmed.front.y, abs=1e-9) == 1.0
    assert pytest.approx(confirmed.lateral.x, abs=1e-9) == -1.0


def test_confirmation_rejects_a_front_parallel_to_up(ingested: dict) -> None:
    from rigby_general.contracts import DirectionV1
    from rigby_general.morphology import confirm_frame

    proposed = ingested["zoo_dual_arm"].morphology.intrinsic_frame
    with pytest.raises(ValueError, match="parallel to up"):
        confirm_frame(proposed, front=DirectionV1(x=0.0, y=0.0, z=1.0))


@pytest.mark.parametrize("robot_id", ZOO_IDS)
def test_scale_is_internally_coherent(robot_id: str, ingested: dict) -> None:
    scale = ingested[robot_id].morphology.scale
    assert 0.0 < scale.characteristic_length_m <= scale.reach_radius_m
    assert scale.neutral_speed_mps > 0.0
    assert scale.total_mass_kg > 0.0
    assert scale.reach_samples >= 1000


def test_measurement_is_deterministic() -> None:
    """Two ingests of one file must measure the same body.

    Every primitive is certified against these numbers. If reach drifted between
    runs, a library baked yesterday would describe a different robot than the one
    running today.
    """

    first = ingest_robot(ZOO_ROOT / "zoo_jaw_arm" / "robot.urdf", robot_id="zoo_jaw_arm")
    second = ingest_robot(ZOO_ROOT / "zoo_jaw_arm" / "robot.urdf", robot_id="zoo_jaw_arm")
    assert first.morphology.content_hash() == second.morphology.content_hash()
    assert first.manifest.content_hash() == second.manifest.content_hash()


# --------------------------------------------------------------------------
# The finalized model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("robot_id", ZOO_IDS)
def test_finalized_model_carries_every_site_and_actuator(
    robot_id: str, ingested: dict
) -> None:
    import mujoco

    robot = ingested[robot_id]
    model = robot.finalized.model

    for site in robot.morphology.sites:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site.name) >= 0
    assert model.nu == len(robot.morphology.joints)


@pytest.mark.parametrize("robot_id", ZOO_IDS)
def test_actuators_are_plain_motors(robot_id: str, ingested: dict) -> None:
    """A bias term would make the v2 inverse-dynamics controller refuse the model."""

    import mujoco

    model = ingested[robot_id].finalized.model
    for index in range(model.nu):
        assert model.actuator_gaintype[index] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[index] == mujoco.mjtBias.mjBIAS_NONE
        assert model.actuator_dyntype[index] == mujoco.mjtDyn.mjDYN_NONE
