from __future__ import annotations

import io
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient

from rigby_v2.app import ApiContext, create_app
from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.config import RuntimeSettings
from rigby_core.contracts import (
    ArtifactRefV1,
    CaptureConfigV1,
    SimulationJobV1,
    SubmitSimulationJobRequestV1,
)
from rigby_core.jobs import SQLiteJobStore
from rigby_v2.observability import JsonLogFormatter, configure_json_logging, log_event
import pytest

pytestmark = pytest.mark.fast


def _context(tmp_path: Path) -> ApiContext:
    return ApiContext(
        jobs=SQLiteJobStore(tmp_path / "jobs.sqlite3"),
        artifacts=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        settings=RuntimeSettings(
            project_root=tmp_path,
            artifact_root=tmp_path / "artifacts",
            worker_id="observability-test",
        ),
    )


def _submission() -> SubmitSimulationJobRequestV1:
    hashes = tuple(character * 64 for character in "abcd")
    artifacts = tuple(
        ArtifactRefV1(sha256=value, size_bytes=1) for value in hashes
    )
    job = SimulationJobV1(
        job_id="observed-job",
        program_hash=hashes[0],
        scene_hash=hashes[1],
        rig_hash=hashes[2],
        candidate_hash=hashes[3],
        program_artifact=artifacts[0],
        scene_artifact=artifacts[1],
        rig_artifact=artifacts[2],
        candidate_artifact=artifacts[3],
        capture=CaptureConfigV1(cameras=("orbit",)),
        seed=1,
        submitted_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    return SubmitSimulationJobRequestV1(
        job=job,
        priority=2,
        idempotency_key="observed-request",
    )


@contextmanager
def _captured_api(tmp_path: Path) -> Iterator[tuple[TestClient, io.StringIO]]:
    app = create_app(_context(tmp_path))
    logger = logging.getLogger("rigby_v2")
    old_handlers = logger.handlers[:]
    old_level = logger.level
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    try:
        yield TestClient(app), stream
    finally:
        logger.handlers[:] = old_handlers
        logger.setLevel(old_level)


def _events(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


def test_request_and_job_events_have_stable_shape_and_propagate_correlation(
    tmp_path: Path,
) -> None:
    with _captured_api(tmp_path) as (client, stream):
        response = client.post(
            "/api/v2/jobs",
            headers={"X-Request-ID": "correlation-123"},
            json=_submission().model_dump(mode="json"),
        )

    assert response.status_code == 202
    assert response.headers["X-Request-ID"] == "correlation-123"
    events = _events(stream)
    assert [event["event"] for event in events] == [
        "request_started",
        "job_submission_completed",
        "request_completed",
    ]
    assert {event["request_id"] for event in events} == {"correlation-123"}
    assert {event["route"] for event in events} == {"/api/v2/jobs"}
    completed = events[-1]
    assert set(completed) == {
        "duration_ms",
        "event",
        "failure_code",
        "http_status",
        "level",
        "logger",
        "message",
        "method",
        "request_id",
        "route",
        "status",
        "timestamp",
    }
    assert completed["status"] == "succeeded"
    assert completed["http_status"] == 202
    assert completed["failure_code"] is None
    assert isinstance(completed["duration_ms"], float)


def test_typed_staging_failure_redacts_request_surfaces(tmp_path: Path) -> None:
    with _captured_api(tmp_path) as (client, stream):
        response = client.post(
            "/api/v2/scenes/packs/not_a_pack/medium/stage?api_key=QUERY-SECRET",
            headers={
                "X-Request-ID": "failure-123",
                "Authorization": "Bearer AUTH-SECRET",
                "Cookie": "session=COOKIE-SECRET",
            },
            json={"prompt": "PROMPT-SECRET", "body": "BODY-SECRET"},
        )

    assert response.status_code == 404
    events = _events(stream)
    failed = next(event for event in events if event["event"] == "scene_staging_failed")
    completed = next(event for event in events if event["event"] == "request_completed")
    assert failed["failure_code"] == "unsupported_asset"
    assert completed["failure_code"] == "unsupported_asset"
    assert failed["route"] == "/api/v2/scenes/packs/{pack_id}/{profile}/stage"
    encoded = stream.getvalue()
    for secret in (
        "QUERY-SECRET",
        "AUTH-SECRET",
        "COOKIE-SECRET",
        "PROMPT-SECRET",
        "BODY-SECRET",
    ):
        assert secret not in encoded


def test_formatter_defensively_redacts_credentials_prompts_and_exceptions() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger = logging.getLogger("rigby_v2.test.redaction")
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    log_event(
        logger,
        "redaction_probe",
        authorization="Bearer AUTH-VALUE",
        cookie="session=COOKIE-VALUE",
        prompt="PROMPT-VALUE",
        request_body={"api_key": "KEY-VALUE"},
        database_url="postgresql://user:DB-PASSWORD@localhost/rigby",
        nested={"credential": "CREDENTIAL-VALUE"},
    )
    payload = json.loads(stream.getvalue())
    assert payload["authorization"] == "[REDACTED]"
    assert payload["cookie"] == "[REDACTED]"
    assert payload["prompt"] == "[REDACTED]"
    assert payload["request_body"] == "[REDACTED]"
    assert payload["database_url"] == "[REDACTED]"
    assert payload["nested"] == {"credential": "[REDACTED]"}
    assert not any(
        secret in stream.getvalue()
        for secret in (
            "AUTH-VALUE",
            "COOKIE-VALUE",
            "PROMPT-VALUE",
            "KEY-VALUE",
            "DB-PASSWORD",
            "CREDENTIAL-VALUE",
        )
    )


def test_json_logging_configuration_does_not_duplicate_owned_handlers() -> None:
    logger = logging.getLogger("rigby_v2")
    configure_json_logging()
    configure_json_logging()
    owned = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_rigby_v2_json_handler", False)
    ]
    assert len(owned) == 1


def test_service_startup_is_json_logged_without_changing_health_semantics(
    tmp_path: Path,
) -> None:
    with _captured_api(tmp_path) as (client, stream):
        with client:
            response = client.get(
                "/api/v2/health",
                headers={"X-Request-ID": "health-123"},
            )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    events = _events(stream)
    assert sum(event["event"] == "service_started" for event in events) == 1
    assert sum(event["event"] == "service_stopped" for event in events) == 1
    request = next(event for event in events if event["event"] == "request_completed")
    assert request["route"] == "/api/v2/health"
    assert request["request_id"] == "health-123"
