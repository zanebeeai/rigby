from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .compiler import PROJECT_ROOT, RIG_PROFILE, apply_overrides, compile_motion
from .models import (
    CompileRequest,
    CompileResponse,
    ExportRequest,
    PlanRequest,
    PlanResponse,
    PipelineRunRequest,
)
from .pipeline import PipelineRunStore
from .planner import plan_motion, provider_status
from .store import ResultStore


store = ResultStore()
pipeline_runs = PipelineRunStore()
app = FastAPI(title="Rigby POC API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    return {"status": "ok", "version": __version__, "provider": provider_status()}


@app.get("/api/v1/assets")
def assets() -> dict[str, object]:
    manifest = json.loads((PROJECT_ROOT / "assets" / "manifest.json").read_text(encoding="utf-8"))
    manifest["rig_profile_url"] = "/api/v1/assets/rig-profile"
    return manifest


@app.get("/api/v1/assets/rig-profile")
def rig_profile() -> dict[str, object]:
    return json.loads(RIG_PROFILE.read_text(encoding="utf-8"))


@app.post("/api/v1/plan", response_model=PlanResponse)
def plan(request: PlanRequest) -> PlanResponse:
    outcome = plan_motion(request)
    return PlanResponse(
        program=outcome.program,
        provider=outcome.provider,
        model=outcome.model,
        model_calls=outcome.model_calls,
    )


@app.post("/api/v1/compile", response_model=CompileResponse)
def compile_program(request: CompileRequest) -> CompileResponse:
    clip = compile_motion(request)
    result_id = None
    if request.persist:
        effective_scene, effective_program = apply_overrides(request)
        effective_request = CompileRequest(scene=effective_scene, program=effective_program, persist=True)
        result_id = store.persist(effective_request, clip)
    return CompileResponse(result_id=result_id, clip=clip)


@app.post("/api/v1/export")
def export(request: ExportRequest) -> FileResponse:
    path = store.animation_path(request.result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="result not found")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{request.result_id}.glb")


@app.get("/api/v1/results")
def list_results() -> dict[str, object]:
    return {"results": [item.model_dump(mode="json") for item in store.list()]}


@app.get("/api/v1/results/{result_id}")
def result_detail(result_id: str) -> dict[str, object]:
    detail = store.detail(result_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="result not found")
    return detail


@app.get("/api/v1/results/{result_id}/animation.glb")
def result_animation(result_id: str) -> FileResponse:
    path = store.animation_path(result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="result not found")
    return FileResponse(path, media_type="model/gltf-binary", filename="animation.glb")


@app.post("/api/v1/pipeline-runs", status_code=202)
def start_pipeline(request: PipelineRunRequest, http_request: Request) -> dict[str, object]:
    return pipeline_runs.start(request, base_url=str(http_request.base_url).rstrip("/"))


@app.get("/api/v1/pipeline-runs/{run_id}")
def pipeline_detail(run_id: str) -> dict[str, object]:
    detail = pipeline_runs.get(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="pipeline run not found")
    return detail


app.mount("/assets", StaticFiles(directory=PROJECT_ROOT / "assets"), name="assets")
results_root = PROJECT_ROOT / "results"
results_root.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=results_root, html=True), name="results-replay")
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/static", StaticFiles(directory=frontend_dist), name="frontend-static")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


def run() -> None:
    import uvicorn

    uvicorn.run("rigby_poc.app:app", host="127.0.0.1", port=8000, reload=False)
