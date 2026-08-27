from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import psycopg
import pytest

from rigby_core.contracts import (
    ArtifactRefV1,
    CaptureConfigV1,
    ReproducibilityManifestV1,
    SimulationJobV1,
    SimulationResultV1,
    SolverConfigV1,
)
from rigby_core.jobs import JobStore
from rigby_core.postgres_jobs import PostgresJobStore
from rigby_core.records import JobState


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("RIGBY_TEST_POSTGRES") != "1",
        reason="set RIGBY_TEST_POSTGRES=1 with the local v2 database running",
    ),
]

DATABASE_URL = os.getenv(
    "RIGBY_V2_DATABASE_URL",
    "postgresql://rigby:rigby-local-only@127.0.0.1:54329/rigby",
)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
START = datetime(2026, 8, 10, 12, tzinfo=UTC)


def _job(job_id: str) -> SimulationJobV1:
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
        seed=5,
        submitted_at=START,
    )


def _result(job_id: str) -> SimulationResultV1:
    artifact = ArtifactRefV1(sha256=HASH_A, size_bytes=1)
    return SimulationResultV1(
        job_id=job_id,
        success=True,
        outcome="smoke_passed",
        trace=artifact,
        reproducibility=ReproducibilityManifestV1(
            os="Windows 11",
            python_version="3.12",
            mujoco_version="3.11.0",
            model_hash=HASH_B,
            solver=SolverConfigV1(),
            seeds=(5,),
            cameras=(),
            dependency_lock_hash=HASH_C,
        ),
        completed_at=START + timedelta(seconds=5),
    )


def test_postgres_store_claims_recovers_and_persists_results() -> None:
    prefix = f"integration-{uuid4()}"
    job_ids = [f"{prefix}-low", f"{prefix}-high"]
    store = PostgresJobStore(DATABASE_URL)
    assert isinstance(store, JobStore)
    assert store.health()["vector_version"] == "0.8.2"
    try:
        store.submit(_job(job_ids[0]), priority=1, available_at=START)
        store.submit(
            _job(job_ids[1]),
            priority=10,
            available_at=START,
            idempotency_key=f"{prefix}-request",
        )
        duplicate = store.submit(
            _job(f"{prefix}-ignored"), idempotency_key=f"{prefix}-request"
        )
        assert duplicate.job.job_id == job_ids[1]

        leased = store.lease("postgres-worker", lease_seconds=10, now=START)
        assert leased is not None and leased.job.job_id == job_ids[1]
        assert leased.state is JobState.LEASED
        assert store.recover(now=START + timedelta(seconds=11)) == 1

        reclaimed = store.lease("postgres-worker-2", now=START + timedelta(seconds=11))
        assert reclaimed is not None and reclaimed.job.job_id == job_ids[1]
        completed = store.complete(
            job_ids[1],
            "postgres-worker-2",
            _result(job_ids[1]),
            now=START + timedelta(seconds=12),
        )
        assert completed.state is JobState.SUCCEEDED
        assert PostgresJobStore(DATABASE_URL).get(job_ids[1]).result == completed.result
    finally:
        with psycopg.connect(DATABASE_URL) as connection:
            connection.execute(
                "DELETE FROM rigby_v2.simulation_jobs WHERE job_id = ANY(%s)",
                (job_ids,),
            )
