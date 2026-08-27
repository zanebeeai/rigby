"""Shared fixtures.

The synthetic robot here is deliberately not any real machine: a two-link arm
with a wrist and a parallel jaw, named ``synthetic_arm``. Using an invented body
in unit tests keeps requirement ``zero_per_robot_code`` honest -- no benchmark
robot's identifiers leak into the source tree through a fixture.
"""

from __future__ import annotations

import hashlib

import pytest
from rigby_core.contracts import ArtifactRefV1, Vec3

from rigby_general.contracts import (
    DirectionV1,
    EffectorKind,
    EffectorV1,
    FrameSource,
    IntrinsicFrameV1,
    JointKind,
    JointRole,
    KinematicChainV1,
    MorphologyClass,
    RobotAssetManifestV1,
    RobotJointV1,
    RobotMorphologyV1,
    RobotScaleV1,
    RobotSiteV1,
    SiteSemantic,
)


def fake_sha256(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def make_joint(
    name: str,
    *,
    body: str,
    role: JointRole,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    kind: JointKind = JointKind.HINGE,
    minimum: float = -3.0,
    maximum: float = 3.0,
    tip_translation_m: float = 0.4,
    tip_rotation_rad: float = 0.2,
) -> RobotJointV1:
    return RobotJointV1(
        name=name,
        kind=kind,
        body=body,
        axis=DirectionV1(x=axis[0], y=axis[1], z=axis[2]),
        minimum=minimum,
        maximum=maximum,
        velocity_limit=2.0,
        effort_limit=80.0,
        role=role,
        tip_translation_m=tip_translation_m,
        tip_rotation_rad=tip_rotation_rad,
    )


def make_site(name: str, body: str, semantic: SiteSemantic) -> RobotSiteV1:
    return RobotSiteV1(
        name=name,
        body=body,
        semantic=semantic,
        position_m=Vec3(x=0.4, y=0.0, z=0.5),
        derivation=f"test_fixture.{semantic.value}",
    )


def build_morphology(
    *,
    robot_id: str = "synthetic_arm",
    effector_kind: EffectorKind = EffectorKind.PARALLEL_JAW,
) -> RobotMorphologyV1:
    joints = [
        make_joint("shoulder", body="link_a", role=JointRole.MAJOR_POSITION),
        make_joint(
            "elbow", body="link_b", role=JointRole.MAJOR_POSITION, axis=(0.0, 1.0, 0.0)
        ),
        make_joint(
            "wrist",
            body="wrist",
            role=JointRole.WRIST_ORIENT,
            axis=(1.0, 0.0, 0.0),
            tip_translation_m=0.01,
            tip_rotation_rad=2.4,
        ),
    ]
    sites = [
        make_site("arm_tip", "tool", SiteSemantic.TIP),
        make_site("arm_grasp_center", "tool", SiteSemantic.GRASP_CENTER),
        make_site("robot_base", "base", SiteSemantic.BASE),
    ]

    if effector_kind is EffectorKind.TOOL_TIP:
        effector = EffectorV1(
            name="arm_effector",
            kind=EffectorKind.TOOL_TIP,
            chain_id="arm",
            tip_body="tool",
            site_names=("arm_tip",),
        )
    else:
        member_bodies = ("jaw_a", "jaw_b")
        grip_joints = ["grip_a", "grip_b"]
        opposition = (("jaw_a",), ("jaw_b",))
        if effector_kind is EffectorKind.MULTIFINGER:
            member_bodies = ("jaw_a", "jaw_b", "jaw_c")
            grip_joints = ["grip_a", "grip_b", "grip_c"]
            opposition = (("jaw_a",), ("jaw_b", "jaw_c"))
        for grip in grip_joints:
            joints.append(
                make_joint(
                    grip,
                    body=grip.replace("grip", "jaw"),
                    role=JointRole.GRIP,
                    kind=JointKind.SLIDE,
                    minimum=0.0,
                    maximum=0.04,
                    tip_translation_m=0.04,
                    tip_rotation_rad=0.0,
                )
            )
        effector = EffectorV1(
            name="arm_effector",
            kind=effector_kind,
            chain_id="arm",
            tip_body="tool",
            member_bodies=member_bodies,
            grip_joints=tuple(grip_joints),
            opposition_groups=opposition,
            max_aperture_m=0.08,
            site_names=("arm_tip", "arm_grasp_center"),
        )

    # A flat envelope: this synthetic arm reaches the same distance in every
    # direction. Real robots do not, which is exactly why the field exists -- but
    # a contract fixture only needs it to be well formed.
    azimuth_bins, elevation_bins = 12, 7
    chain = KinematicChainV1(
        chain_id="arm",
        bodies=("base", "link_a", "link_b", "wrist", "tool"),
        joints=("shoulder", "elbow", "wrist"),
        tip_body="tool",
        root_body="link_a",
        reach_radius_m=0.9,
        workspace_centroid_m=Vec3(x=0.4, y=0.0, z=0.5),
        working_direction=DirectionV1(x=1.0, y=0.0, z=0.0),
        reach_envelope_m=tuple(0.9 for _ in range(azimuth_bins * elevation_bins)),
        envelope_azimuth_bins=azimuth_bins,
        envelope_elevation_bins=elevation_bins,
        positioning_dof=2,
        orienting_dof=1,
    )

    return RobotMorphologyV1(
        robot_id=robot_id,
        morphology_class=MorphologyClass.FIXED_BASE_ARM,
        base_body="base",
        joints=tuple(joints),
        chains=(chain,),
        effectors=(effector,),
        sites=tuple(sites),
        intrinsic_frame=IntrinsicFrameV1(
            up=DirectionV1(x=0.0, y=0.0, z=1.0),
            front=DirectionV1(x=1.0, y=0.0, z=0.0),
            lateral=DirectionV1(x=0.0, y=1.0, z=0.0),
            source=FrameSource.OPERATOR_CONFIRMED,
            confidence=1.0,
            evidence=("reach_centroid", "base_principal_axis"),
        ),
        scale=RobotScaleV1(
            reach_radius_m=0.9,
            characteristic_length_m=0.35,
            neutral_speed_mps=0.4,
            base_footprint_m=0.2,
            payload_kg=3.0,
            total_mass_kg=18.0,
            workspace_centroid_m=Vec3(x=0.4, y=0.0, z=0.5),
            reach_samples=4096,
        ),
        self_collision_pairs=(("link_a", "tool"),),
    )


def build_manifest(morphology: RobotMorphologyV1) -> RobotAssetManifestV1:
    return RobotAssetManifestV1(
        rig_id=morphology.robot_id,
        mjcf=ArtifactRefV1(
            sha256=fake_sha256(f"{morphology.robot_id}.mjcf"),
            size_bytes=4096,
            media_type="application/xml",
            filename="robot.xml",
        ),
        source_asset=ArtifactRefV1(
            sha256=fake_sha256(f"{morphology.robot_id}.urdf"),
            size_bytes=2048,
            media_type="application/xml",
            filename="robot.urdf",
        ),
        source_format="urdf",
        dofs=tuple(joint.to_dof_spec() for joint in morphology.joints),
        actuator_order=tuple(joint.name for joint in morphology.joints),
        rest_qpos=tuple(0.0 for _ in morphology.joints),
        sites=morphology.sites,
        morphology=morphology,
    )


@pytest.fixture
def morphology() -> RobotMorphologyV1:
    return build_morphology()


@pytest.fixture
def manifest(morphology: RobotMorphologyV1) -> RobotAssetManifestV1:
    return build_manifest(morphology)
