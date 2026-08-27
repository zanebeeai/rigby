"""FastAPI surface for the general pipeline.

Phase 0 ships ``/api/v3/health`` only. It exists this early because the health
payload is where the base-tree fingerprint becomes observable: without it, a path
dependency on an untracked ``rigby_v2`` has no identity, and a certified
primitive could not be traced back to the solver and gate code that produced it.

Model routing follows the v2 rule from ``benchmark/live_configuration.py``:
report *presence* booleans and a fingerprint over the non-secret routing, never
the values themselves.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rigby_v2.hashing import content_hash

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


def _register_bundled_robots(registry: RobotRegistry, settings: GeneralSettings) -> None:
    """Make the robots already on disk addressable without re-uploading them.

    The registry starts empty, so every asset robot -- the zoo, the third-party
    set, the operator's own hardware -- was invisible to the API even though the
    studio listed all of them. A home page offering eleven robots and an endpoint
    that knows none of them is worse than either alone.

    A model that will not ingest is skipped rather than raised: three of these
    are *meant* to be refused, and the refusal is recorded in the studio's intake
    view, not here.
    """

    roots = (
        settings.project_root / "assets" / "general" / "zoo",
        settings.project_root / "assets" / "general" / "exotic",
        settings.project_root / "assets" / "general" / "irl",
    )
    known = set(registry.list_ids())
    for root in roots:
        if not root.is_dir():
            continue
        for directory in sorted(root.iterdir()):
            source = directory / "robot.urdf"
            if not source.is_file() or directory.name in known:
                continue
            try:
                # From the path, not from the bytes: a URDF names its meshes
                # relative to its own directory, and bytes have no directory.
                registry.register_source(
                    source,
                    robot_id=directory.name,
                )
            except Exception:  # noqa: BLE001 - a refused model simply is not offered
                continue


def create_app(settings: GeneralSettings | None = None) -> FastAPI:
    resolved = settings or GeneralSettings.from_env()
    app = FastAPI(title="Rigby General", version=__version__)

    # The trace studio is a static file, usually opened from a small local
    # server on another port, and its home page posts prompts here. Without this
    # the browser blocks that call and the page reports the pipeline down while
    # it is plainly running. Loopback only -- this is a local tool, and the
    # allowance should not outlive the machine it is on.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://(127\.0\.0\.1|localhost)(:\d+)?$",
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    registry = RobotRegistry(resolved.robot_root)
    _register_bundled_robots(registry, resolved)

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

    @app.get("/api/v3/environments")
    def list_environments() -> dict[str, Any]:
        """The authored worlds, for a caller choosing where to run something."""

        from .scenes import available_environments, load_environment

        rows = []
        for name in available_environments():
            environment = load_environment(name)
            rows.append(
                {
                    "environment_id": environment.environment_id,
                    "description": environment.description,
                    "objects": [
                        {
                            "name": item.name,
                            "span_m": round(item.span_m, 4),
                            "mass_kg": round(item.mass_kg, 4),
                            "position_m": [round(v, 4) for v in item.position_m],
                        }
                        for item in environment.objects
                    ],
                }
            )
        return {"environments": rows}

    @app.post("/api/v3/runs", status_code=201)
    def submit_run(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Answer one prompt on one robot, and record the trace either way.

        A refusal is a 201 with ``accepted: false``, not an error status. The
        stage that refused and the measurement it refused on are the result --
        turning them into a 4xx would throw away the part worth reading.
        """

        from .run import answer
        from .schema.inventory import load_inventory
        from .primitives import PrimitiveLibrary

        prompt = str(payload.get("prompt") or "").strip()
        robot_id = str(payload.get("robot_id") or "").strip()
        if not prompt or not robot_id:
            raise HTTPException(
                status_code=422,
                detail="both `prompt` and `robot_id` are required",
            )

        record = _load(robot_id)
        library = PrimitiveLibrary(resolved.robot_root)
        # The record keeps the compiled MJCF on disk rather than in memory, so
        # the model is loaded per run. That also means a run always uses the file
        # the registry actually holds, not a copy that drifted from it.
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(record.model_path))
        try:
            result = answer(
                prompt,
                record.manifest,
                model,
                load_inventory(),
                library.load(robot_id),
                library.load_failures(robot_id),
            )
        except RigbyGeneralError as error:
            raise HTTPException(
                status_code=422,
                detail={
                    "failure_code": error.code.value,
                    "message": error.message,
                    **error.details,
                },
            ) from error

        traces.write(result.trace)
        return {
            "trace_id": result.trace.trace_id,
            "accepted": bool(result.accepted),
            "failure_stage": result.failure_stage,
            "prompt": prompt,
            "robot_id": robot_id,
            "trace": result.trace.to_json(),
        }

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
