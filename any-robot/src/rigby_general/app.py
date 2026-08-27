"""FastAPI surface for the general pipeline.

Phase 0 ships ``/api/v3/health`` only. It exists this early because the health
payload is where the base-tree fingerprint becomes observable: without it, a path
dependency on an untracked ``rigby_core`` has no identity, and a certified
primitive could not be traced back to the solver and gate code that produced it.

Model routing follows the v2 rule from ``benchmark/live_configuration.py``:
report *presence* booleans and a fingerprint over the non-secret routing, never
the values themselves.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rigby_core.hashing import content_hash

from . import __version__
from .config import GeneralSettings, base_tree_fingerprint
from .contracts import SUPPORTED_MORPHOLOGY_CLASSES, DirectionV1
from .errors import RigbyGeneralError, RobotNotFoundError
from .robots import RobotRegistry
from .trace import TraceStore
from .schema import program as schema_program


_ROUTING_KEYS = (
    "OPENAI_PLANNER_MODEL",
    "OPENAI_JUDGE_MODEL",
    "OPENAI_JUDGE_FALLBACK_MODEL",
    "OPENAI_JUDGE_REASONING_EFFORT",
)


def provider_readiness(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Describe model routing without disclosing a key or a model name."""

    values = os.environ if env is None else env
    routing = {key: values.get(key, "") for key in _ROUTING_KEYS}
    return {
        "openai_key_present": bool(values.get("OPENAI_API_KEY", "").strip()),
        "routing_configured": {
            key: bool(value.strip()) for key, value in routing.items()
        },
        "configuration_fingerprint": content_hash(routing),
    }


def health_payload(settings: GeneralSettings) -> dict[str, Any]:
    base_tree = base_tree_fingerprint()
    return {
        "service": "rigby-general",
        "version": __version__,
        "schema_versions": {
            "motion_schema_program": schema_program.SCHEMA_VERSION,
        },
        "base_tree": base_tree.as_dict(),
        "supported_morphology_classes": sorted(
            item.value for item in SUPPORTED_MORPHOLOGY_CLASSES
        ),
        "runtime": {
            "physics_hz": settings.physics_hz,
            "render_fps": settings.render_fps,
            "bake_budget_seconds": settings.bake_budget_seconds,
            "bake_max_attempts": settings.bake_max_attempts,
        },
        "providers": provider_readiness(),
    }


def create_app(settings: GeneralSettings | None = None) -> FastAPI:
    resolved = settings or GeneralSettings.from_env()
    app = FastAPI(title="Rigby General", version=__version__)
    registry = RobotRegistry(resolved.robot_root)

    @app.get("/api/v3/health")
    def health() -> dict[str, Any]:
        return health_payload(resolved)

    @app.get("/api/v3/robots")
    def list_robots() -> dict[str, Any]:
        return {
            "robots": [
                registry.load(robot_id).summary() for robot_id in registry.list_ids()
            ]
        }

    @app.post("/api/v3/robots", status_code=201)
    async def upload_robot(
        file: UploadFile, robot_id: str | None = None
    ) -> dict[str, Any]:
        payload = await file.read()
        try:
            record = registry.register_bytes(
                payload, filename=file.filename or "robot.urdf", robot_id=robot_id
            )
        except RigbyGeneralError as error:
            # A model that will not load is a typed refusal naming the rule it
            # broke, never a 500 with a stack trace.
            raise HTTPException(
                status_code=422,
                detail={
                    "failure_code": error.code.value,
                    "message": error.message,
                    **error.details,
                },
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return record.summary()

    @app.get("/api/v3/robots/{robot_id}")
    def get_robot(robot_id: str) -> dict[str, Any]:
        return _load(robot_id).summary()

    @app.get("/api/v3/robots/{robot_id}/morphology")
    def get_morphology(robot_id: str) -> dict[str, Any]:
        record = _load(robot_id)
        return {
            "morphology": record.morphology.model_dump(mode="json"),
            "provenance": record.provenance,
        }

    @app.get("/api/v3/robots/{robot_id}/model.xml")
    def get_model(robot_id: str) -> FileResponse:
        record = _load(robot_id)
        return FileResponse(record.model_path, media_type="application/xml")

    @app.post("/api/v3/robots/{robot_id}/frame")
    def confirm_frame(
        robot_id: str, front: list[float] | None = Body(default=None, embed=True)
    ) -> dict[str, Any]:
        """Accept or correct the proposed front direction.

        The single place a person is genuinely needed. A pedestal arm has no
        derivable front, and every deictic or intrinsic-frame schema resolves
        against it, so the answer has to come from whoever installed the robot.
        """

        _load(robot_id)
        direction = None
        if front is not None:
            if len(front) != 3:
                raise HTTPException(
                    status_code=422, detail="front must be three components"
                )
            try:
                direction = DirectionV1(x=front[0], y=front[1], z=front[2])
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            record = registry.confirm_front(robot_id, direction)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return record.summary()

    def _load(robot_id: str):
        try:
            return registry.load(robot_id)
        except (RobotNotFoundError, ValueError) as error:
            raise HTTPException(
                status_code=404, detail=f"unknown robot: {robot_id}"
            ) from error

    # -- results and the trace studio -------------------------------------
    #
    # Served from the app the way Motion Studio is in v1, so a run can be
    # inspected without leaving the process that produced it. The same files
    # also open straight from disk with no server at all.

    results_root = resolved.project_root / "results"
    traces = TraceStore(results_root)

    @app.get("/api/v3/results")
    def list_results() -> dict[str, Any]:
        return {"results": traces.index()}

    @app.get("/api/v3/results/{trace_id}")
    def get_result(trace_id: str) -> dict[str, Any]:
        try:
            return traces.load(trace_id)
        except (KeyError, ValueError) as error:
            raise HTTPException(
                status_code=404, detail=f"unknown result: {trace_id}"
            ) from error

    @app.get("/api/v3/results/{trace_id}/clip.gif")
    def get_clip(trace_id: str) -> FileResponse:
        try:
            path = traces.clip_path(trace_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="unknown result") from error
        if path is None:
            raise HTTPException(status_code=404, detail="this result has no clip")
        return FileResponse(path, media_type="image/gif")

    @app.get("/", include_in_schema=False)
    def studio() -> FileResponse:
        page = results_root / "studio.html"
        if not page.is_file():
            raise HTTPException(
                status_code=404,
                detail="no studio page yet; run scripts/build_studio.py",
            )
        return FileResponse(page, media_type="text/html")

    if results_root.is_dir():
        # The page references clips as ./<trace_id>/clip.gif, which resolves the
        # same whether it is opened from disk or served from here.
        app.mount("/", StaticFiles(directory=results_root), name="results")

    app.state.settings = resolved
    app.state.registry = registry
    app.state.traces = traces
    return app


def run() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    settings = GeneralSettings.from_env()
    uvicorn.run(create_app(settings), host="127.0.0.1", port=settings.api_port)
