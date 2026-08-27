from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.config import RuntimeSettings
from rigby_core.contracts import (
    CandidateTrajectoryV1,
    CaptureConfigV1,
    DofSpecV1,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    RigAssetManifestV1,
    SceneManifestV2,
    SimulationJobV1,
    SolverConfigV1,
    TrajectorySampleV1,
)
from rigby_core.errors import FailureCode
from rigby_core.jobs import SQLiteJobStore
from rigby_core.records import JobState
from rigby_v2.rigging.canonical_human import xml_for_profile
from rigby_v2.simulation import NativeMujocoRuntime
from rigby_v2.worker import SimulationWorker, WorkerDependencies
import pytest

pytestmark = pytest.mark.medium


def _stage_job(
    artifacts: ContentAddressedArtifactStore,
    *,
    job_id: str,
) -> SimulationJobV1:
    xml = xml_for_profile("medium")
    model = mujoco.MjModel.from_xml_string(xml)
    mjcf_ref = artifacts.put_bytes(
        xml.encode("utf-8"), media_type="application/mjcf+xml", filename="human.xml"
    )
    dofs = []
    actuator_order = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        assert joint_name is not None and actuator_name is not None
        minimum, maximum = map(float, model.jnt_range[joint_id])
        dofs.append(
            DofSpecV1(
                name=joint_name,
                joint=joint_name,
                minimum=minimum,
                maximum=maximum,
                velocity_limit=20.0,
                effort_limit=150.0,
            )
        )
        actuator_order.append(actuator_name)
    rig = RigAssetManifestV1(
        rig_id="rigby-canonical-human-medium",
        mjcf=mjcf_ref,
        dofs=tuple(dofs),
        actuator_order=tuple(actuator_order),
        rest_qpos=tuple(float(value) for value in model.qpos0),
    )
    rig_ref = artifacts.put_json(rig, filename="rig-manifest.json")
    scene = SceneManifestV2(
        scene_id="empty-stage",
        rig_asset_hash=rig_ref.sha256,
    )
    scene_ref = artifacts.put_json(scene, filename="scene.json")
    program = MotionProgramV2(
        program_id="stationary-program",
        source_text="Stand in the neutral pose.",
        duration_s=0.05,
        rig_id=rig.rig_id,
        scene_id=scene.scene_id,
        seed=1,
        phases=(
            MotionPhaseV2(
                phase_id="hold", kind=PhaseKind.HOLD, start_s=0.0, end_s=0.05
            ),
        ),
        tracks=(
            MotionTrackV2(
                track_id="spine-hold",
                target="spine_flex",
                owner="posture",
                keyframes=(
                    MotionKeyframeV2(time_s=0.0, joint_values={"spine_flex": 0.0}),
                    MotionKeyframeV2(time_s=0.05, joint_values={"spine_flex": 0.0}),
                ),
            ),
        ),
    )
    program_ref = artifacts.put_json(program, filename="program.json")
    zero_velocity = tuple(float(value) for value in np.zeros(model.nv))
    neutral_qpos = tuple(float(value) for value in model.qpos0)
    candidate = CandidateTrajectoryV1(
        candidate_id="neutral-candidate",
        rig_id=rig.rig_id,
        samples=(
            TrajectorySampleV1(time_s=0.0, qpos=neutral_qpos, qvel=zero_velocity),
            TrajectorySampleV1(time_s=0.05, qpos=neutral_qpos, qvel=zero_velocity),
        ),
    )
    candidate_ref = artifacts.put_json(candidate, filename="candidate.json")
    return SimulationJobV1(
        job_id=job_id,
        program_hash=program_ref.sha256,
        scene_hash=scene_ref.sha256,
        rig_hash=rig_ref.sha256,
        candidate_hash=candidate_ref.sha256,
        program_artifact=program_ref,
        scene_artifact=scene_ref,
        rig_artifact=rig_ref,
        candidate_artifact=candidate_ref,
        capture=CaptureConfigV1(cameras=("orbit",)),
        seed=1,
        submitted_at=datetime(2026, 8, 10, tzinfo=UTC),
    )


def test_job_survives_restart_executes_and_replays_exactly(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    database = tmp_path / "jobs.sqlite3"
    first_store = SQLiteJobStore(database)
    first_store.submit(_stage_job(artifacts, job_id="first-run"))

    # Recreate both API-side durable objects before the worker sees the job.
    restarted_store = SQLiteJobStore(database)
    restarted_artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    worker = SimulationWorker(
        WorkerDependencies(
            jobs=restarted_store,
            artifacts=restarted_artifacts,
            settings=RuntimeSettings(
                project_root=Path(__file__).resolve().parents[1],
                artifact_root=tmp_path / "artifacts",
                worker_id="test-worker",
            ),
            runtime=NativeMujocoRuntime(),
        )
    )
    completed = worker.run_once()
    assert completed is not None and completed.state is JobState.SUCCEEDED
    assert completed.result is not None
    assert completed.result.success is False
    assert completed.result.outcome == "simulation_completed_uncertified"
    assert restarted_artifacts.exists(completed.result.trace, verify=True)
    assert completed.result.reproducibility.solver == completed.job.solver
    assert completed.result.reproducibility.requested_capture == completed.job.capture
    assert completed.result.reproducibility.cameras == ()

    replay_job = _stage_job(restarted_artifacts, job_id="replay-run")
    restarted_store.submit(replay_job)
    replayed = worker.run_once()
    assert replayed is not None and replayed.result is not None
    assert replayed.result.trace.sha256 == completed.result.trace.sha256
    assert replayed.result.metrics == completed.result.metrics


def test_worker_fails_closed_without_lock_or_with_solver_drift(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")

    missing_lock_store = SQLiteJobStore(tmp_path / "missing-lock.sqlite3")
    missing_lock_store.submit(_stage_job(artifacts, job_id="missing-lock"))
    missing_lock = SimulationWorker(
        WorkerDependencies(
            jobs=missing_lock_store,
            artifacts=artifacts,
            settings=RuntimeSettings(
                project_root=tmp_path,
                artifact_root=tmp_path / "artifacts",
                worker_id="missing-lock-worker",
            ),
            runtime=NativeMujocoRuntime(),
        )
    ).run_once()
    assert missing_lock is not None and missing_lock.state is JobState.FAILED
    assert missing_lock.failure is not None
    assert missing_lock.failure.code is FailureCode.INVALID_CONTRACT
    assert "dependency lock is missing" in missing_lock.failure.message

    drift_store = SQLiteJobStore(tmp_path / "solver-drift.sqlite3")
    drifted = _stage_job(artifacts, job_id="solver-drift").model_copy(
        update={"solver": SolverConfigV1(iterations=17)}
    )
    drift_store.submit(drifted)
    drift = SimulationWorker(
        WorkerDependencies(
            jobs=drift_store,
            artifacts=artifacts,
            settings=RuntimeSettings(
                project_root=Path(__file__).resolve().parents[1],
                artifact_root=tmp_path / "artifacts",
                worker_id="solver-drift-worker",
            ),
            runtime=NativeMujocoRuntime(),
        )
    ).run_once()
    assert drift is not None and drift.state is JobState.FAILED
    assert drift.failure is not None
    assert drift.failure.code is FailureCode.INVALID_CONTRACT
    assert "does not match the compiled MuJoCo model" in drift.failure.message
