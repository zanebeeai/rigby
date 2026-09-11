from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from rigby_core.contracts import (
    ArtifactRefV1,
    CandidateTrajectoryV1,
    CaptureConfigV1,
    CoordinateFrame,
    DofSpecV1,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    Quaternion,
    QuaternionConvention,
    QuaternionMeaning,
    QuaternionOrder,
    RigAssetManifestV1,
    SceneManifestV2,
    SimulationJobV1,
    TrajectorySampleV1,
    Vec3,
)
from rigby_core.hashing import canonical_json, content_hash
from rigby_core.records import AnimationRecordV1, CertificationStatus, DatasetSplit, FailureRecordV1
from rigby_core.errors import FailureCode

pytestmark = pytest.mark.fast


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def quaternion() -> Quaternion:
    return Quaternion(
        values=(1.0, 0.0, 0.0, 0.0),
        convention=QuaternionConvention(
            order=QuaternionOrder.WXYZ,
            frame=CoordinateFrame.LOCAL,
            meaning=QuaternionMeaning.REST_DELTA,
        ),
    )


def motion_program() -> MotionProgramV2:
    return MotionProgramV2(
        program_id="program-1",
        source_text="Reach out and press the button.",
        duration_s=1.0,
        rig_id="rigby-human-v1",
        scene_id="button-scene",
        seed=7,
        phases=(MotionPhaseV2(phase_id="act", kind=PhaseKind.ACTION, start_s=0, end_s=1),),
        tracks=(
            MotionTrackV2(
                track_id="right-hand",
                target="right_palm",
                owner="right_arm",
                keyframes=(
                    MotionKeyframeV2(
                        time_s=0,
                        position=Vec3(x=0.1, y=0.2, z=1.0),
                        rotation=quaternion(),
                    ),
                    MotionKeyframeV2(
                        time_s=1,
                        position=Vec3(x=0.1, y=-0.2, z=1.0),
                        rotation=quaternion(),
                    ),
                ),
            ),
        ),
    )


def simulation_job(*, job_id: str = "job-1") -> SimulationJobV1:
    artifacts = {
        "program_artifact": ArtifactRefV1(sha256=HASH_A, size_bytes=1),
        "scene_artifact": ArtifactRefV1(sha256=HASH_B, size_bytes=1),
        "rig_artifact": ArtifactRefV1(sha256=HASH_C, size_bytes=1),
        "candidate_artifact": ArtifactRefV1(sha256=HASH_D, size_bytes=1),
    }
    return SimulationJobV1(
        job_id=job_id,
        program_hash=HASH_A,
        scene_hash=HASH_B,
        rig_hash=HASH_C,
        candidate_hash=HASH_D,
        **artifacts,
        capture=CaptureConfigV1(cameras=("orbit", "ego")),
        seed=11,
        submitted_at=datetime(2026, 8, 10, tzinfo=UTC),
    )


def test_contracts_are_extra_forbidding_immutable_and_versioned() -> None:
    program = motion_program()
    assert program.schema_version == "2.0"
    with pytest.raises(ValidationError):
        MotionPhaseV2(
            phase_id="bad", kind=PhaseKind.ACTION, start_s=0, end_s=1, surprise=True
        )
    with pytest.raises(ValidationError):
        program.duration_s = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable"):
        program.metadata["changed"] = True
    with pytest.raises(ValidationError):
        MotionProgramV2(**{**program.model_dump(), "schema_version": "1.0"})


def test_quaternion_requires_explicit_convention_and_unit_norm() -> None:
    with pytest.raises(ValidationError, match="normalized"):
        Quaternion(
            values=(1.0, 1.0, 0.0, 0.0),
            convention=quaternion().convention,
        )
    with pytest.raises(ValidationError, match="convention"):
        Quaternion.model_validate({"values": [1, 0, 0, 0]})


def test_program_rejects_non_monotonic_and_out_of_bounds_timelines() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        MotionTrackV2(
            track_id="bad",
            target="hand",
            owner="arm",
            keyframes=(
                MotionKeyframeV2(time_s=1, joint_values={"wrist": 0.0}),
                MotionKeyframeV2(time_s=0, joint_values={"wrist": 0.1}),
            ),
        )
    data = motion_program().model_dump()
    data["duration_s"] = 0.5
    with pytest.raises(ValidationError, match="beyond program duration"):
        MotionProgramV2.model_validate(data)


def test_scene_and_rig_manifests_bind_assets_to_explicit_coordinates() -> None:
    artifact = ArtifactRefV1(sha256=HASH_A, size_bytes=12, filename="rig.mjz")
    rig = RigAssetManifestV1(
        rig_id="rigby-human-v1",
        mjcf=artifact,
        dofs=(
            DofSpecV1(
                name="right_elbow",
                joint="right_elbow",
                minimum=0,
                maximum=2.4,
                velocity_limit=6,
                effort_limit=80,
            ),
        ),
        actuator_order=("right_elbow",),
        rest_qpos=(0.0,),
    )
    scene = SceneManifestV2(scene_id="empty", rig_asset_hash=rig.content_hash())
    assert rig.free_root is True
    assert scene.coordinate_system.up_axis == "+Z"
    with pytest.raises(ValidationError, match="free root"):
        RigAssetManifestV1(
            rig_id="fixed",
            mjcf=artifact,
            free_root=False,
            dofs=rig.dofs,
            actuator_order=rig.actuator_order,
            rest_qpos=rig.rest_qpos,
        )


def test_canonical_hash_is_order_independent_and_timezone_normalized() -> None:
    instant_utc = datetime(2026, 8, 10, 12, tzinfo=UTC)
    instant_offset = instant_utc.astimezone(timezone(timedelta(hours=-4)))
    left = {"b": [2, 3], "a": {"instant": instant_utc, "value": -0.0}}
    right = {"a": {"value": 0.0, "instant": instant_offset}, "b": [2, 3]}
    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)
    assert motion_program().content_hash() == content_hash(motion_program())
    with pytest.raises(ValueError, match="NaN"):
        canonical_json({"bad": float("nan")})


def test_records_capture_release_license_lineage_and_typed_failures() -> None:
    artifact = ArtifactRefV1(sha256=HASH_A, size_bytes=1)
    record = AnimationRecordV1(
        record_id="animation-1",
        prompt="wave",
        program_hash=HASH_A,
        scene_hash=HASH_B,
        rig_hash=HASH_C,
        result_hash=HASH_D,
        status=CertificationStatus.STAGED,
        release="N",
        staged_for_release="N+1",
        split=DatasetSplit.TRAIN,
        artifacts=(artifact,),
        evaluation_version="eval-v1",
    )
    failure = FailureRecordV1(
        failure_id="failure-1",
        job_id="job-1",
        code=FailureCode.INFEASIBLE,
        message="Target is outside the reachable workspace",
        stage="compile",
    )
    assert record.status is CertificationStatus.STAGED
    assert failure.code is FailureCode.INFEASIBLE
    assert simulation_job().schema_version == "1.0"


def test_candidate_trajectory_rejects_inconsistent_generalized_state() -> None:
    with pytest.raises(ValidationError, match="consistent qpos"):
        CandidateTrajectoryV1(
            candidate_id="candidate",
            rig_id="rig",
            samples=(
                TrajectorySampleV1(time_s=0.0, qpos=(0.0, 1.0), qvel=(0.0,)),
                TrajectorySampleV1(time_s=1.0, qpos=(0.0,), qvel=(0.0,)),
            ),
        )
