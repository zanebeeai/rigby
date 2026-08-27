from __future__ import annotations

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.routing import Match

from .artifacts import ContentAddressedArtifactStore
from .config import RuntimeSettings
from .contracts import SubmitSimulationJobRequestV1, utc_now
from .errors import ArtifactIntegrityError, FailureCode, JobNotFoundError
from .jobs import JobStore
from .observability import configure_json_logging, log_event
from .postgres_jobs import PostgresJobStore
from .records import JobRecord, JobState
from .rigging.manifest_builder import stage_canonical_rig
from .scenes import SceneAssetError, stage_object_pack_scene


_LOGGER = logging.getLogger("rigby_v2.api")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and isinstance(getattr(route, "path", None), str):
        return route.path
    partial: str | None = None
    for candidate in request.app.router.routes:
        match, _ = candidate.matches(request.scope)
        path = getattr(candidate, "path", None)
        if not isinstance(path, str):
            continue
        if match is Match.FULL:
            return path
        if match is Match.PARTIAL and partial is None:
            partial = path
    return partial or "<unmatched>"


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    return supplied if _REQUEST_ID.fullmatch(supplied) else str(uuid4())


def _duration_ms(started: float) -> float:
    return round(max(0.0, time.perf_counter() - started) * 1000.0, 3)


def _failure_for_status(status_code: int) -> str | None:
    if status_code < 400:
        return None
    if status_code == 422:
        return FailureCode.INVALID_CONTRACT.value
    if status_code >= 500:
        return FailureCode.INTERNAL_ERROR.value
    return "http_error"


def _mark_failure(request: Request, code: FailureCode | str) -> str:
    value = code.value if isinstance(code, FailureCode) else code
    request.state.failure_code = value
    return value


def _log_lifecycle(
    request: Request,
    event: str,
    started: float,
    *,
    status: str,
    failure_code: str | None = None,
    **identifiers: object,
) -> None:
    log_event(
        _LOGGER,
        event,
        request_id=request.state.request_id,
        route=_route_template(request),
        status=status,
        duration_ms=_duration_ms(started),
        failure_code=failure_code,
        **identifiers,
    )


@dataclass(slots=True)
class ApiContext:
    jobs: JobStore
    artifacts: ContentAddressedArtifactStore
    settings: RuntimeSettings


def create_context(settings: RuntimeSettings | None = None) -> ApiContext:
    resolved = settings or RuntimeSettings.from_env()
    return ApiContext(
        jobs=PostgresJobStore(resolved.database_url),
        artifacts=ContentAddressedArtifactStore(resolved.artifact_root),
        settings=resolved,
    )


def _dump(record: JobRecord) -> dict[str, object]:
    return record.model_dump(mode="json")


def create_app(context: ApiContext | None = None) -> FastAPI:
    configure_json_logging()
    dependencies = context or create_context()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        log_event(_LOGGER, "service_started", service="api", status="ready")
        try:
            yield
        finally:
            log_event(_LOGGER, "service_stopped", service="api", status="stopped")

    app = FastAPI(title="Rigby v2 API", version="2.0.0", lifespan=lifespan)
    app.state.rigby_v2 = dependencies

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        started = time.perf_counter()
        request.state.request_id = _request_id(request)
        route = _route_template(request)
        log_event(
            _LOGGER,
            "request_started",
            request_id=request.state.request_id,
            method=request.method,
            route=route,
            status="started",
            duration_ms=0.0,
            failure_code=None,
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            code = getattr(exc, "code", FailureCode.INTERNAL_ERROR)
            failure_code = _mark_failure(request, str(code))
            log_event(
                _LOGGER,
                "request_completed",
                level=logging.ERROR,
                request_id=request.state.request_id,
                method=request.method,
                route=_route_template(request),
                status="failed",
                http_status=500,
                duration_ms=_duration_ms(started),
                failure_code=failure_code,
                exception_type=type(exc).__name__,
            )
            raise
        failure_code = getattr(
            request.state,
            "failure_code",
            _failure_for_status(response.status_code),
        )
        response.headers["X-Request-ID"] = request.state.request_id
        log_event(
            _LOGGER,
            "request_completed",
            level=logging.WARNING if response.status_code >= 400 else logging.INFO,
            request_id=request.state.request_id,
            method=request.method,
            route=_route_template(request),
            status="failed" if response.status_code >= 400 else "succeeded",
            http_status=response.status_code,
            duration_ms=_duration_ms(started),
            failure_code=failure_code,
        )
        return response

    @app.get("/api/v2/health")
    def health() -> dict[str, object]:
        database: dict[str, str] | None = None
        database_error: str | None = None
        health_method = getattr(dependencies.jobs, "health", None)
        if callable(health_method):
            try:
                database = health_method()
            except Exception as exc:  # Health must remain diagnostic when DB is down.
                database_error = f"{type(exc).__name__}: {exc}"
        return {
            "status": "ok" if database_error is None else "degraded",
            "version": "2.0.0",
            "job_store": type(dependencies.jobs).__name__,
            "database": database,
            "database_error": database_error,
            "runtime": dependencies.settings.reproducibility_snapshot(),
        }

    @app.post("/api/v2/jobs", status_code=202)
    def submit(
        request: SubmitSimulationJobRequestV1,
        http_request: Request,
    ) -> dict[str, object]:
        started = time.perf_counter()
        try:
            record = dependencies.jobs.submit(
                request.job,
                priority=request.priority,
                idempotency_key=request.idempotency_key,
            )
        except ValueError as exc:
            code = _mark_failure(http_request, FailureCode.INVALID_CONTRACT)
            _log_lifecycle(
                http_request,
                "job_submission_failed",
                started,
                status="failed",
                failure_code=code,
                job_id=request.job.job_id,
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _log_lifecycle(
            http_request,
            "job_submission_completed",
            started,
            status="succeeded",
            job_id=record.job.job_id,
        )
        return _dump(record)

    @app.post("/api/v2/rigs/canonical/{profile}/stage")
    def stage_rig(profile: str, http_request: Request) -> dict[str, object]:
        started = time.perf_counter()
        if profile not in {"small", "medium", "large"}:
            code = _mark_failure(http_request, FailureCode.UNSUPPORTED_ASSET)
            _log_lifecycle(
                http_request,
                "rig_staging_failed",
                started,
                status="failed",
                failure_code=code,
                profile=profile,
            )
            raise HTTPException(status_code=404, detail="unknown canonical rig profile")
        staged = stage_canonical_rig(profile, artifacts=dependencies.artifacts)
        _log_lifecycle(
            http_request,
            "rig_staging_completed",
            started,
            status="succeeded",
            profile=profile,
            rig_id=staged.manifest.rig_id,
        )
        return {
            "reference": staged.reference.model_dump(mode="json"),
            "manifest": staged.manifest.model_dump(mode="json"),
        }

    @app.post("/api/v2/scenes/packs/{pack_id}/{profile}/stage")
    def stage_scene(
        pack_id: str,
        profile: str,
        http_request: Request,
    ) -> dict[str, object]:
        started = time.perf_counter()
        if profile not in {"small", "medium", "large"}:
            code = _mark_failure(http_request, FailureCode.UNSUPPORTED_ASSET)
            _log_lifecycle(
                http_request,
                "scene_staging_failed",
                started,
                status="failed",
                failure_code=code,
                pack_id=pack_id,
                profile=profile,
            )
            raise HTTPException(status_code=404, detail="unknown canonical rig profile")
        try:
            rig = stage_canonical_rig(profile, artifacts=dependencies.artifacts)
            scene = stage_object_pack_scene(
                pack_id,
                profile=profile,
                rig_reference=rig.reference,
                artifacts=dependencies.artifacts,
            )
        except SceneAssetError as exc:
            code = _mark_failure(http_request, exc.code)
            _log_lifecycle(
                http_request,
                "scene_staging_failed",
                started,
                status="failed",
                failure_code=code,
                pack_id=pack_id,
                profile=profile,
            )
            raise HTTPException(status_code=404, detail=exc.message) from exc
        _log_lifecycle(
            http_request,
            "scene_staging_completed",
            started,
            status="succeeded",
            pack_id=pack_id,
            profile=profile,
            scene_id=scene.manifest.scene_id,
        )
        return {
            "rig_reference": rig.reference.model_dump(mode="json"),
            "rig_manifest": rig.manifest.model_dump(mode="json"),
            "scene_reference": scene.reference.model_dump(mode="json"),
            "scene_manifest": scene.manifest.model_dump(mode="json"),
        }

    @app.get("/api/v2/jobs")
    def list_jobs(
        state: list[JobState] | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, object]:
        records = dependencies.jobs.list(states=state, limit=limit)
        return {"jobs": [_dump(record) for record in records]}

    @app.get("/api/v2/jobs/{job_id}")
    def job_detail(job_id: str) -> dict[str, object]:
        try:
            return _dump(dependencies.jobs.get(job_id))
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/v2/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, object]:
        try:
            return _dump(dependencies.jobs.cancel(job_id))
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/api/v2/jobs/{job_id}/replay", status_code=202)
    def replay(job_id: str, http_request: Request) -> dict[str, object]:
        started = time.perf_counter()
        try:
            source = dependencies.jobs.get(job_id)
        except JobNotFoundError as exc:
            code = _mark_failure(http_request, "job_not_found")
            _log_lifecycle(
                http_request,
                "replay_submission_failed",
                started,
                status="failed",
                failure_code=code,
                source_job_id=job_id,
            )
            raise HTTPException(status_code=404, detail="job not found") from exc
        if source.state is not JobState.SUCCEEDED or source.result is None:
            code = _mark_failure(http_request, "job_not_replayable")
            _log_lifecycle(
                http_request,
                "replay_submission_failed",
                started,
                status="failed",
                failure_code=code,
                source_job_id=job_id,
            )
            raise HTTPException(
                status_code=409, detail="only completed jobs can be replayed"
            )
        replay_job = source.job.model_copy(
            update={"job_id": str(uuid4()), "submitted_at": utc_now()}
        )
        replay_record = dependencies.jobs.submit(
            replay_job,
            priority=source.priority,
            idempotency_key=f"replay:{job_id}:{replay_job.job_id}",
        )
        _log_lifecycle(
            http_request,
            "replay_submission_completed",
            started,
            status="succeeded",
            source_job_id=job_id,
            job_id=replay_record.job.job_id,
        )
        return _dump(replay_record)

    @app.get("/api/v2/jobs/{job_id}/trace")
    def trace(job_id: str) -> FileResponse:
        try:
            record = dependencies.jobs.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if record.result is None:
            raise HTTPException(status_code=409, detail="job has no result trace")
        try:
            path = dependencies.artifacts.resolve(record.result.trace)
        except (FileNotFoundError, ArtifactIntegrityError) as exc:
            raise HTTPException(
                status_code=500, detail="result trace failed integrity check"
            ) from exc
        return FileResponse(
            path,
            media_type=record.result.trace.media_type,
            filename=record.result.trace.filename or f"{job_id}-trace.npz",
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    port = int(os.environ.get("RIGBY_V2_API_PORT", "8010"))
    if not 1 <= port <= 65535:
        raise ValueError("RIGBY_V2_API_PORT must be between 1 and 65535")
    uvicorn.run("rigby_v2.app:app", host="127.0.0.1", port=port, reload=False)
