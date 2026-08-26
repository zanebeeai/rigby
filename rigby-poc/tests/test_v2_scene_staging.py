from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.config import RuntimeSettings
from rigby_v2.contracts import (
    CandidateTrajectoryV1,
    CaptureConfigV1,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    SimulationJobV1,
    TrajectorySampleV1,
)
from rigby_v2.jobs import SQLiteJobStore
from rigby_v2.records import JobState
from rigby_v2.rigging.manifest_builder import stage_canonical_rig
from rigby_v2.scenes import stage_object_pack_scene
from rigby_v2.simulation import NativeMujocoRuntime
from rigby_v2.worker import SimulationWorker, WorkerDependencies


def test_staged_scene_binds_pack_rig_mjcf_and_self_contained_mjz(
    tmp_path: Path,
) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    staged = stage_object_pack_scene(
        "drawer",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )

    manifest = staged.manifest
    assert artifacts.exists(staged.reference, verify=True)
    assert manifest.rig_asset_hash == rig.reference.sha256
    assert manifest.pack_id == "drawer"
    assert manifest.compiled_mjcf is not None
    assert manifest.compiled_mjz is not None
    assert artifacts.exists(manifest.compiled_mjcf, verify=True)
    assert artifacts.exists(manifest.compiled_mjz, verify=True)
    assert manifest.compiled_mjcf.sha256 == staged.compiled.xml_sha256
    assert manifest.compiled_mjz.sha256 == staged.compiled.mjz_sha256
    assert manifest.objects[0].articulated
    assert manifest.affordances
    assert manifest.state_predicates

    model = mujoco.MjSpec.from_zip(
        io.BytesIO(artifacts.read_bytes(manifest.compiled_mjz))
    ).compile()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis") >= 0
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "obj__drawer_unit__drawer_slide"
        )
        >= 0
    )


def test_staging_rejects_a_mismatched_rig_profile(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("small", artifacts=artifacts)
    try:
        stage_object_pack_scene(
            "drawer",
            profile="large",
            rig_reference=rig.reference,
            artifacts=artifacts,
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched rig/scene profile was accepted")


def test_worker_runs_rig_trajectory_inside_staged_object_scene(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "drawer",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    program = MotionProgramV2(
        program_id="drawer-neutral",
        source_text="Stand neutrally in front of the drawer.",
        duration_s=0.025,
        rig_id=rig.manifest.rig_id,
        scene_id=scene.manifest.scene_id,
        seed=4,
        phases=(
            MotionPhaseV2(
                phase_id="hold",
                kind=PhaseKind.HOLD,
                start_s=0.0,
                end_s=0.025,
            ),
        ),
        tracks=(
            MotionTrackV2(
                track_id="neutral-spine",
                target="spine_flex",
                owner="posture",
                keyframes=(
                    MotionKeyframeV2(time_s=0.0, joint_values={"spine_flex": 0.0}),
                    MotionKeyframeV2(time_s=0.025, joint_values={"spine_flex": 0.0}),
                ),
            ),
        ),
    )
    program_ref = artifacts.put_json(program, filename="drawer-neutral-program.json")
    rig_model = mujoco.MjModel.from_xml_string(
        artifacts.read_bytes(rig.manifest.mjcf).decode("utf-8")
    )
    candidate = CandidateTrajectoryV1(
        candidate_id="drawer-neutral-candidate",
        rig_id=rig.manifest.rig_id,
        samples=(
            TrajectorySampleV1(
                time_s=0.0,
                qpos=tuple(rig_model.qpos0),
                qvel=tuple(np.zeros(rig_model.nv)),
            ),
            TrajectorySampleV1(
                time_s=0.025,
                qpos=tuple(rig_model.qpos0),
                qvel=tuple(np.zeros(rig_model.nv)),
            ),
        ),
    )
    candidate_ref = artifacts.put_json(
        candidate, filename="drawer-neutral-candidate.json"
    )
    job = SimulationJobV1(
        job_id="drawer-scene-job",
        program_hash=program_ref.sha256,
        scene_hash=scene.reference.sha256,
        rig_hash=rig.reference.sha256,
        candidate_hash=candidate_ref.sha256,
        program_artifact=program_ref,
        scene_artifact=scene.reference,
        rig_artifact=rig.reference,
        candidate_artifact=candidate_ref,
        capture=CaptureConfigV1(cameras=("orbit",)),
        seed=4,
        submitted_at=datetime(2026, 8, 10, tzinfo=UTC),
    )
    jobs = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    jobs.submit(job)
    completed = SimulationWorker(
        WorkerDependencies(
            jobs=jobs,
            artifacts=artifacts,
            settings=RuntimeSettings(
                project_root=Path(__file__).resolve().parents[1],
                artifact_root=tmp_path / "artifacts",
                worker_id="scene-worker",
            ),
            runtime=NativeMujocoRuntime(),
        )
    ).run_once()

    assert completed is not None and completed.state is JobState.SUCCEEDED
    assert completed.result is not None
    assert completed.result.reproducibility.model_hash == scene.compiled.mjz_sha256
    with np.load(artifacts.resolve(completed.result.trace)) as trace:
        scene_model = scene.compiled.load_model()
        assert trace["qpos"].shape[1] == scene_model.nq
        assert scene_model.nq > rig_model.nq
