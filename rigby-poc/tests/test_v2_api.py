from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from rigby_v2.app import ApiContext, create_app
from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.config import RuntimeSettings
from rigby_v2.contracts import (
    ArtifactRefV1,
    CaptureConfigV1,
    SimulationJobV1,
    SubmitSimulationJobRequestV1,
)
from rigby_v2.jobs import SQLiteJobStore
import pytest

pytestmark = pytest.mark.fast


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _job() -> SimulationJobV1:
    return SimulationJobV1(
        job_id="api-job",
        program_hash=HASH_A,
        scene_hash=HASH_B,
        rig_hash=HASH_C,
        candidate_hash=HASH_D,
        program_artifact=ArtifactRefV1(sha256=HASH_A, size_bytes=1),
        scene_artifact=ArtifactRefV1(sha256=HASH_B, size_bytes=1),
        rig_artifact=ArtifactRefV1(sha256=HASH_C, size_bytes=1),
        candidate_artifact=ArtifactRefV1(sha256=HASH_D, size_bytes=1),
        capture=CaptureConfigV1(cameras=("orbit",)),
        seed=1,
        submitted_at=datetime(2026, 8, 10, tzinfo=UTC),
    )


def _context(tmp_path: Path) -> ApiContext:
    return ApiContext(
        jobs=SQLiteJobStore(tmp_path / "jobs.sqlite3"),
        artifacts=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        settings=RuntimeSettings(
            project_root=tmp_path,
            artifact_root=tmp_path / "artifacts",
            worker_id="api-test",
        ),
    )


def test_v2_api_submits_persists_lists_and_cancels(tmp_path: Path) -> None:
    request = SubmitSimulationJobRequestV1(
        job=_job(), priority=8, idempotency_key="api-request"
    )
    first = TestClient(create_app(_context(tmp_path)))
    submitted = first.post("/api/v2/jobs", json=request.model_dump(mode="json"))
    assert submitted.status_code == 202
    assert submitted.json()["state"] == "queued"

    # A fresh app instance sees the same durable job.
    restarted = TestClient(create_app(_context(tmp_path)))
    detail = restarted.get("/api/v2/jobs/api-job")
    assert detail.status_code == 200
    assert detail.json()["priority"] == 8
    listed = restarted.get("/api/v2/jobs", params={"state": "queued"})
    assert [item["job"]["job_id"] for item in listed.json()["jobs"]] == ["api-job"]
    assert restarted.post("/api/v2/jobs/api-job/replay").status_code == 409

    rig = restarted.post("/api/v2/rigs/canonical/medium/stage")
    assert rig.status_code == 200
    assert rig.json()["manifest"]["rig_id"] == "rigby-canonical-human-medium"
    assert restarted.post("/api/v2/rigs/canonical/giant/stage").status_code == 404
    scene = restarted.post(
        "/api/v2/scenes/packs/grasp_place_block/medium/stage"
    )
    assert scene.status_code == 200
    assert scene.json()["scene_manifest"]["pack_id"] == "grasp_place_block"
    assert scene.json()["scene_manifest"]["compiled_mjz"]["sha256"]
    assert restarted.post("/api/v2/scenes/packs/not_a_pack/medium/stage").status_code == 404

    cancelled = restarted.post("/api/v2/jobs/api-job/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert restarted.get("/api/v2/health").json()["status"] == "ok"
