from __future__ import annotations

from pathlib import Path

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.config import RuntimeSettings
from rigby_core.jobs import SQLiteJobStore
from rigby_core.records import JobState
from rigby_v2.simulation import (
    ConstantTarget,
    ControlTarget,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationFailureCode,
    SimulationRequest,
)
from rigby_v2.rigging.canonical_human import xml_for_profile
from rigby_v2.worker import SimulationWorker, WorkerDependencies

from test_v2_worker_vertical import _stage_job
import pytest

pytestmark = pytest.mark.medium


def test_native_runtime_stops_at_a_bounded_cancellation_poll() -> None:
    import mujoco

    xml = xml_for_profile("medium")
    model = mujoco.MjModel.from_xml_string(xml)
    result = NativeMujocoRuntime().simulate(
        SimulationRequest(
            model_xml=xml,
            trajectory=ConstantTarget(ControlTarget.stationary(model.qpos0, model.nv)),
            config=SimulationConfig(duration_s=1.0, free_root_joint_name="pelvis_free"),
            cancel_check=lambda: True,
        )
    )

    assert not result.completed
    assert result.failure is not None
    assert result.failure.code is SimulationFailureCode.CANCELLED
    assert result.failure.step == 1
    assert len(result.trace.times_s) == 1


def test_cancelled_leased_job_is_not_completed_or_failed(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    jobs = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    jobs.submit(_stage_job(artifacts, job_id="cancel-during-run"))
    leased = jobs.lease("cancel-worker")
    assert leased is not None
    cancelled = jobs.cancel(leased.job.job_id)
    assert cancelled.state is JobState.CANCELLED

    result = SimulationWorker(
        WorkerDependencies(
            jobs=jobs,
            artifacts=artifacts,
            settings=RuntimeSettings(
                project_root=tmp_path,
                artifact_root=tmp_path / "artifacts",
                worker_id="cancel-worker",
            ),
            runtime=NativeMujocoRuntime(),
        )
    ).execute(leased)

    assert result.state is JobState.CANCELLED
    assert result.result is None
    assert result.failure is None
