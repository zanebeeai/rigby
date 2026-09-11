from __future__ import annotations

import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .compiler import PROJECT_ROOT, RIG_PROFILE, apply_overrides, compile_motion
from .gripper_runs import GripperRunRequest, GripperRunStore
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
gripper_runs = GripperRunStore()
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


def _demo_repo_root() -> Path:
    """The checkout whose demos/ registry a saved result goes into.

    A function rather than a constant so a test can point it at a scratch
    checkout instead of writing into the real registry.
    """

    from rigby_core.demos import find_repo_root

    return find_repo_root(PROJECT_ROOT)


@app.post("/api/v1/results/{result_id}/demo", status_code=201)
def register_result_demo(result_id: str, payload: dict[str, object] | None = Body(default=None)) -> dict[str, object]:
    """Put a compiled result in the shared demo registry (`demos/` at the repo root).

    The entry carries the clip itself as a `humanoid-bones-v1` payload, so the
    registry viewer can replay it once its bone player exists, plus who asked
    (the machine's git identity, or ``who`` in the body), the commit and branch
    this server runs from, the prompt, and the planner that answered it. A GIF
    is not rendered here: that needs a browser and the capture page, which is
    `evals.render_demo_gif`'s job; add its output to the entry afterwards.
    """

    from rigby_core.demos import DemoSpec, find_repo_root, register, write_index

    folder = store.root / result_id
    if not folder.is_dir() or folder.parent.resolve() != store.root.resolve():
        raise HTTPException(status_code=404, detail="result not found")
    clip_path = folder / "clip.json"
    if not clip_path.is_file():
        raise HTTPException(status_code=409, detail="this result has no clip.json to register")
    try:
        root = _demo_repo_root()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    body: dict[str, object] = payload or {}
    program = json.loads((folder / "program.json").read_text(encoding="utf-8"))
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8")) if (folder / "provenance.json").is_file() else {}
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8")) if (folder / "metrics.json").is_file() else {}
    clip = json.loads(clip_path.read_text(encoding="utf-8"))
    prompt = str(program.get("source_text") or body.get("prompt") or result_id)
    verdict = metrics.get("accepted", metrics.get("structural_valid"))
    outcome = None
    if verdict is not None:
        outcome = {"state": "ok" if verdict else "failed", "text": "accepted" if verdict else "rejected by the structural gates"}
    planner = f"{provenance.get('planner_provider', '?')}/{provenance.get('planner_model', '?')}"
    spec = DemoSpec(
        title=str(body.get("title") or prompt[:72]),
        prompt=prompt,
        tier="humanoid",
        embodiment=str(provenance.get("rig_id") or RIG_PROFILE.get("id", "mesh2motion-human-vrm1")),
        kind="humanoid-bones-v1",
        how=str(body.get("how") or f"uv run rigby-humanoid; POST /api/v1/pipeline-runs {json.dumps({'prompt': prompt})} -> result {result_id} ({planner}, seed {provenance.get('seed', 0)})"),
        payload=clip_path,
        who=str(body["who"]) if body.get("who") else None,
        outcome=outcome,
        notes=str(body.get("notes") or f"{clip.get('fps')} fps, {clip.get('duration_s')} s, planner {planner}"),
        tags=["humanoid", f"result:{result_id}", str(program.get("intent", ""))],
    )
    try:
        entry_path = register(spec, root)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    write_index(root)
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    return {
        "id": entry["id"],
        "registry": entry_path.relative_to(root).as_posix(),
        "who": entry["who"],
        "source": entry["source"],
        "index": "demos/index.html",
    }


@app.post("/api/v1/pipeline-runs", status_code=202)
def start_pipeline(request: PipelineRunRequest, http_request: Request) -> dict[str, object]:
    return pipeline_runs.start(request, base_url=str(http_request.base_url).rstrip("/"))


@app.get("/api/v1/pipeline-runs/{run_id}")
def pipeline_detail(run_id: str) -> dict[str, object]:
    detail = pipeline_runs.get(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="pipeline run not found")
    return detail


@app.post("/api/v1/gripper-runs", status_code=202)
def start_gripper_run(request: GripperRunRequest) -> dict[str, object]:
    """Start one isolated, persistent VLM-directed gripper attempt."""
    return gripper_runs.start(request)


@app.get("/api/v1/gripper-runs")
def list_gripper_runs() -> dict[str, object]:
    return {"runs": gripper_runs.list()}


@app.get("/api/v1/gripper-runs/{run_id}")
def gripper_run_detail(run_id: str) -> dict[str, object]:
    detail = gripper_runs.get(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="gripper run not found")
    return detail


@app.get("/api/v1/gripper-runs/{run_id}/clip")
def gripper_run_clip(run_id: str) -> FileResponse:
    clip = gripper_runs.clip_path(run_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="gripper run clip not found")
    return FileResponse(clip, media_type="application/json")


@app.post("/api/v1/gripper-runs/{run_id}/abort")
def abort_gripper_run(run_id: str) -> dict[str, object]:
    detail = gripper_runs.abort(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="gripper run not found")
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
