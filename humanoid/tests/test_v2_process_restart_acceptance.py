from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
import mujoco
import numpy as np
import psycopg
import pytest
from psycopg import sql

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_core.contracts import (
    CandidateTrajectoryV1,
    CaptureConfigV1,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    SceneManifestV2,
    SimulationJobV1,
    SubmitSimulationJobRequestV1,
    TrajectorySampleV1,
)
from rigby_v2.rigging import stage_canonical_rig
from rigby_core.hashing import hash_file
from rigby_v2.release_ops import seal_release_evidence


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("RIGBY_TEST_PROCESS_RESTART") != "1",
        reason="set RIGBY_TEST_PROCESS_RESTART=1 with the local PostgreSQL service running",
    ),
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATABASE_URL = os.getenv(
    "RIGBY_V2_DATABASE_URL",
    "postgresql://rigby:rigby-local-only@127.0.0.1:54329/rigby",
)


def _database_url(database: str) -> str:
    parsed = urlsplit(BASE_DATABASE_URL)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"/{database}", parsed.query, parsed.fragment)
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_process(
    code: str,
    *,
    env: dict[str, str],
    log_path: Path,
    stack: ExitStack,
) -> subprocess.Popen[bytes]:
    log = stack.enter_context(log_path.open("ab", buffering=0))
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return process


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_until(predicate, *, timeout_s: float, description: str):
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception as error:  # Services may be between socket bind and readiness.
            last_error = error
        time.sleep(0.1)
    detail = f"; last error={last_error!r}" if last_error is not None else ""
    raise AssertionError(f"timed out waiting for {description}{detail}")


def _wait_api(client: httpx.Client) -> None:
    def healthy() -> bool:
        response = client.get("/api/v2/health")
        return response.status_code == 200 and response.json()["status"] == "ok"

    _wait_until(healthy, timeout_s=20, description="live API health")


def _wait_job(client: httpx.Client, job_id: str, state: str, timeout_s: float = 30):
    def reached():
        response = client.get(f"/api/v2/jobs/{job_id}")
        if response.status_code != 200:
            return None
        record = response.json()
        return record if record["state"] == state else None

    return _wait_until(
        reached, timeout_s=timeout_s, description=f"job {job_id}={state}"
    )


def _stage_short_job(artifact_root: Path) -> SimulationJobV1:
    artifacts = ContentAddressedArtifactStore(artifact_root)
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = SceneManifestV2(
        scene_id="restart-acceptance-empty-stage",
        rig_asset_hash=rig.reference.sha256,
    )
    scene_ref = artifacts.put_json(scene, filename="restart-scene.json")
    program = MotionProgramV2(
        program_id="restart-acceptance-neutral",
        source_text="Stand neutrally during the process restart acceptance run.",
        duration_s=0.05,
        rig_id=rig.manifest.rig_id,
        scene_id=scene.scene_id,
        seed=41,
        phases=(
            MotionPhaseV2(
                phase_id="hold",
                kind=PhaseKind.HOLD,
                start_s=0.0,
                end_s=0.05,
            ),
        ),
        tracks=(
            MotionTrackV2(
                track_id="neutral-spine",
                target="spine_flex",
                owner="posture",
                keyframes=(
                    MotionKeyframeV2(time_s=0.0, joint_values={"spine_flex": 0.0}),
                    MotionKeyframeV2(time_s=0.05, joint_values={"spine_flex": 0.0}),
                ),
            ),
        ),
    )
    program_ref = artifacts.put_json(program, filename="restart-program.json")
    rig_xml = artifacts.read_bytes(rig.manifest.mjcf).decode("utf-8")
    model = mujoco.MjModel.from_xml_string(rig_xml)
    candidate = CandidateTrajectoryV1(
        candidate_id="restart-acceptance-candidate",
        rig_id=rig.manifest.rig_id,
        samples=(
            TrajectorySampleV1(
                time_s=0.0,
                qpos=tuple(float(value) for value in model.qpos0),
                qvel=tuple(float(value) for value in np.zeros(model.nv)),
            ),
            TrajectorySampleV1(
                time_s=0.05,
                qpos=tuple(float(value) for value in model.qpos0),
                qvel=tuple(float(value) for value in np.zeros(model.nv)),
            ),
        ),
    )
    candidate_ref = artifacts.put_json(candidate, filename="restart-candidate.json")
    return SimulationJobV1(
        job_id=f"restart-acceptance-{uuid4()}",
        program_hash=program_ref.sha256,
        scene_hash=scene_ref.sha256,
        rig_hash=rig.reference.sha256,
        candidate_hash=candidate_ref.sha256,
        program_artifact=program_ref,
        scene_artifact=scene_ref,
        rig_artifact=rig.reference,
        candidate_artifact=candidate_ref,
        capture=CaptureConfigV1(cameras=("orbit",)),
        seed=41,
        submitted_at=datetime.now(UTC),
    )


def test_live_api_worker_process_restart_recovers_and_replays_exactly(
    tmp_path: Path,
) -> None:
    database_name = "rigby_restart_" + uuid4().hex
    admin_url = _database_url("postgres")
    database_url = _database_url(database_name)
    artifact_root = tmp_path / "artifacts"
    pause_marker = (tmp_path / "leased.marker").resolve()
    processes: list[subprocess.Popen[bytes]] = []
    acceptance: dict[str, object] | None = None

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name))
        )
    try:
        initialization = (PROJECT_ROOT / "db" / "init" / "001_rigby_v2.sql").read_text(
            encoding="utf-8"
        )
        with psycopg.connect(database_url, autocommit=True) as database:
            database.execute(initialization)

        common_env = os.environ.copy()
        existing_pythonpath = common_env.get("PYTHONPATH")
        common_env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        common_env.update(
            {
                "RIGBY_V2_DATABASE_URL": database_url,
                "RIGBY_V2_ARTIFACT_DIR": str(artifact_root),
                "RIGBY_V2_PROJECT_ROOT": str(PROJECT_ROOT),
                "RIGBY_V2_LEASE_SECONDS": "5",
                "RIGBY_V2_RESTART_ACCEPTANCE": "1",
            }
        )
        job = _stage_short_job(artifact_root)

        with ExitStack() as stack:
            first_api_env = common_env | {"RIGBY_V2_API_PORT": str(_free_port())}
            first_api = _start_process(
                "from rigby_v2.app import run; run()",
                env=first_api_env,
                log_path=tmp_path / "api-first.log",
                stack=stack,
            )
            processes.append(first_api)
            first_base_url = f"http://127.0.0.1:{first_api_env['RIGBY_V2_API_PORT']}"
            with httpx.Client(base_url=first_base_url, timeout=5) as client:
                _wait_api(client)
                response = client.post(
                    "/api/v2/jobs",
                    json=SubmitSimulationJobRequestV1(
                        job=job,
                        idempotency_key=f"restart-submit-{job.job_id}",
                    ).model_dump(mode="json"),
                )
                assert response.status_code == 202, response.text

                paused_env = common_env | {
                    "RIGBY_V2_WORKER_ID": "restart-worker-interrupted",
                    "RIGBY_V2_PAUSE_AFTER_LEASE_FILE": str(pause_marker),
                }
                interrupted_worker = _start_process(
                    "from rigby_v2.worker import run; run()",
                    env=paused_env,
                    log_path=tmp_path / "worker-interrupted.log",
                    stack=stack,
                )
                processes.append(interrupted_worker)
                _wait_until(
                    lambda: (
                        pause_marker.is_file()
                        and pause_marker.read_text(encoding="utf-8") == job.job_id
                    ),
                    timeout_s=20,
                    description="worker pause after PostgreSQL lease",
                )
                leased = _wait_job(client, job.job_id, "leased")
                assert leased["lease_owner"] == "restart-worker-interrupted"
                lease_expires_at = datetime.fromisoformat(
                    leased["lease_expires_at"].replace("Z", "+00:00")
                )
                _stop(interrupted_worker)

                wait_seconds = max(
                    0.0, (lease_expires_at - datetime.now(UTC)).total_seconds()
                )
                time.sleep(wait_seconds + 0.25)
                replacement_env = common_env | {
                    "RIGBY_V2_WORKER_ID": "restart-worker-replacement"
                }
                replacement = _start_process(
                    "from rigby_v2.worker import run; run()",
                    env=replacement_env,
                    log_path=tmp_path / "worker-replacement.log",
                    stack=stack,
                )
                processes.append(replacement)
                completed = _wait_job(client, job.job_id, "succeeded")
                _stop(replacement)
                assert completed["attempts"] == 2
                first_outcome = completed["result"]["outcome"]
                first_trace = completed["result"]["trace"]["sha256"]

            _stop(first_api)

            second_api_env = common_env | {"RIGBY_V2_API_PORT": str(_free_port())}
            restarted_api = _start_process(
                "from rigby_v2.app import run; run()",
                env=second_api_env,
                log_path=tmp_path / "api-restarted.log",
                stack=stack,
            )
            processes.append(restarted_api)
            second_base_url = f"http://127.0.0.1:{second_api_env['RIGBY_V2_API_PORT']}"
            with httpx.Client(base_url=second_base_url, timeout=5) as client:
                _wait_api(client)
                durable = _wait_job(client, job.job_id, "succeeded")
                assert durable["result"]["trace"]["sha256"] == first_trace
                replay_response = client.post(f"/api/v2/jobs/{job.job_id}/replay")
                assert replay_response.status_code == 202, replay_response.text
                replay_id = replay_response.json()["job"]["job_id"]

                replay_worker = _start_process(
                    "from rigby_v2.worker import run; run()",
                    env=common_env | {"RIGBY_V2_WORKER_ID": "restart-replay-worker"},
                    log_path=tmp_path / "worker-replay.log",
                    stack=stack,
                )
                processes.append(replay_worker)
                replayed = _wait_job(client, replay_id, "succeeded")
                _stop(replay_worker)
                assert replayed["result"]["outcome"] == first_outcome
                assert replayed["result"]["trace"]["sha256"] == first_trace
                acceptance = {
                    "live_http_submission": True,
                    "worker_terminated_after_lease": True,
                    "lease_recovered_by_replacement": True,
                    "attempts": completed["attempts"],
                    "api_restarted_in_new_process": True,
                    "live_replay_endpoint": True,
                    "outcome": first_outcome,
                    "authoritative_trace_sha256": first_trace,
                    "replay_trace_sha256": replayed["result"]["trace"]["sha256"],
                    "test_source_sha256": hash_file(Path(__file__)),
                    "dependency_lock_sha256": hash_file(PROJECT_ROOT / "uv.lock"),
                }
    finally:
        for process in reversed(processes):
            _stop(process)
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    sql.Identifier(database_name)
                )
            )
        evidence_root = os.getenv("RIGBY_RESTART_EVIDENCE_ROOT")
        if acceptance is not None and evidence_root:
            seal_release_evidence(
                Path(evidence_root),
                requirement_id="deterministic_replay",
                passed=True,
                payload={**acceptance, "disposable_database_cleaned": True},
            )
