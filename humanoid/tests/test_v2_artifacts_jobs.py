from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_core.contracts import (
    ArtifactRefV1,
    CaptureConfigV1,
    ReproducibilityManifestV1,
    SimulationJobV1,
    SimulationResultV1,
    SolverConfigV1,
)
from rigby_core.errors import ArtifactIntegrityError, FailureCode, InvalidJobTransitionError, JobLeaseError
from rigby_core.jobs import JobStore, SQLiteJobStore
from rigby_core.records import FailureRecordV1, JobState

pytestmark = pytest.mark.fast


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
START = datetime(2026, 8, 10, 12, tzinfo=UTC)


def simulation_job(job_id: str) -> SimulationJobV1:
    return SimulationJobV1(
        job_id=job_id,
        program_hash=HASH_A,
        scene_hash=HASH_B,
        rig_hash=HASH_C,
        candidate_hash=HASH_D,
        program_artifact=ArtifactRefV1(sha256=HASH_A, size_bytes=1),
        scene_artifact=ArtifactRefV1(sha256=HASH_B, size_bytes=1),
        rig_artifact=ArtifactRefV1(sha256=HASH_C, size_bytes=1),
        candidate_artifact=ArtifactRefV1(sha256=HASH_D, size_bytes=1),
        capture=CaptureConfigV1(cameras=("orbit",)),
        seed=3,
        submitted_at=START,
    )


def test_content_addressed_store_deduplicates_and_detects_corruption(tmp_path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    first = store.put_bytes(b"rigby", media_type="text/plain", filename="note.txt")
    second = store.put_bytes(b"rigby", media_type="text/plain")
    assert first.sha256 == second.sha256
    assert store.read_bytes(first) == b"rigby"
    assert len(list((tmp_path / "artifacts" / "objects" / "sha256").rglob("*"))) == 2

    store.resolve(first, verify=False).write_bytes(b"tampered")
    assert store.exists(first, verify=True) is False
    with pytest.raises(ArtifactIntegrityError):
        store.resolve(first)
    with pytest.raises(ArtifactIntegrityError):
        store.put_bytes(b"rigby")


def test_content_addressed_store_streams_files(tmp_path) -> None:
    source = tmp_path / "capture.mp4"
    source.write_bytes(b"frame" * 100_000)
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    reference = store.put_file(source, media_type="video/mp4")
    assert reference.filename == "capture.mp4"
    assert reference.size_bytes == source.stat().st_size
    assert store.read_bytes(reference) == source.read_bytes()


def test_store_satisfies_protocol_and_survives_process_recreation(tmp_path) -> None:
    database = tmp_path / "jobs.sqlite3"
    first = SQLiteJobStore(database)
    assert isinstance(first, JobStore)
    submitted = first.submit(simulation_job("durable"), idempotency_key="request-1")
    assert submitted.state is JobState.QUEUED

    recreated = SQLiteJobStore(database)
    restored = recreated.get("durable")
    assert restored.job == submitted.job
    assert restored.idempotency_key == "request-1"
    duplicate = recreated.submit(simulation_job("other-id"), idempotency_key="request-1")
    assert duplicate.job.job_id == "durable"


def test_leases_heartbeat_priority_and_crash_recovery(tmp_path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = SQLiteJobStore(database)
    store.submit(simulation_job("low"), priority=1, available_at=START)
    store.submit(simulation_job("high"), priority=10, available_at=START)

    leased = store.lease("worker-a", lease_seconds=10, now=START)
    assert leased is not None
    assert leased.job.job_id == "high"
    assert leased.state is JobState.LEASED
    assert leased.attempts == 1
    heartbeaten = store.heartbeat(
        "high", "worker-a", lease_seconds=20, now=START + timedelta(seconds=5)
    )
    assert heartbeaten.lease_expires_at == START + timedelta(seconds=25)
    with pytest.raises(JobLeaseError):
        store.heartbeat("high", "worker-b", now=START + timedelta(seconds=6))

    restarted = SQLiteJobStore(database)
    assert restarted.recover(now=START + timedelta(seconds=24)) == 0
    assert restarted.recover(now=START + timedelta(seconds=26)) == 1
    recovered = restarted.lease("worker-b", now=START + timedelta(seconds=26))
    assert recovered is not None
    assert recovered.job.job_id == "high"
    assert recovered.attempts == 2


def test_cancellation_and_typed_failure_are_terminal(tmp_path) -> None:
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    store.submit(simulation_job("cancel"), available_at=START)
    store.lease("worker", now=START)
    cancelled = store.cancel("cancel", now=START + timedelta(seconds=1))
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.cancel_requested is True
    with pytest.raises(InvalidJobTransitionError):
        store.fail(
            "cancel",
            "worker",
            FailureRecordV1(
                job_id="cancel",
                code=FailureCode.CANCELLED,
                message="cancelled",
                stage="simulation",
            ),
            now=START + timedelta(seconds=2),
        )

    store.submit(simulation_job("failure"), available_at=START)
    leased = store.lease("worker", now=START)
    assert leased is not None and leased.job.job_id == "failure"
    failure = FailureRecordV1(
        job_id="failure",
        code=FailureCode.SIMULATION_FAILED,
        message="MuJoCo diverged",
        stage="simulation",
        retryable=True,
    )
    failed = store.fail("failure", "worker", failure, now=START + timedelta(seconds=1))
    assert failed.state is JobState.FAILED
    assert failed.failure == failure
    assert store.list(states=(JobState.FAILED,))[0].job.job_id == "failure"


def test_completion_persists_replay_manifest_across_restart(tmp_path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = SQLiteJobStore(database)
    store.submit(simulation_job("success"), available_at=START)
    leased = store.lease("worker", now=START)
    assert leased is not None
    artifact = ArtifactRefV1(sha256=HASH_A, size_bytes=100)
    result = SimulationResultV1(
        result_id="result-1",
        job_id="success",
        success=True,
        outcome="button_pressed",
        trace=artifact,
        reproducibility=ReproducibilityManifestV1(
            os="Windows 11",
            python_version="3.12.12",
            mujoco_version="3.11.0",
            model_hash=HASH_B,
            solver=SolverConfigV1(),
            seeds=(3,),
            cameras=(),
            dependency_lock_hash=HASH_C,
        ),
        completed_at=START + timedelta(seconds=1),
    )
    completed = store.complete(
        "success", "worker", result, now=START + timedelta(seconds=1)
    )
    assert completed.state is JobState.SUCCEEDED
    assert completed.result == result
    assert completed.lease_owner is None

    restored = SQLiteJobStore(database).get("success")
    assert restored.result is not None
    assert restored.result.reproducibility.mujoco_version == "3.11.0"
    assert restored.result.content_hash() == result.content_hash()
