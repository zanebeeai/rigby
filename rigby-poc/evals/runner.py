from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .archive import AcceptanceLedger, ResultArchive, utc_now
from .criteria import FIXTURES, POC_ROOT, fixture_errors, grasp_trials, load_criteria, load_json, supported_cases, unsupported_cases
from .evidence import (
    export_gate,
    gesture_diversity_gate,
    gesture_gate,
    grasp_and_physical_gates,
    latency_gate,
    lookup,
    parametric_gate,
    planner_gate,
    safety_gate,
    structural_physics_gate,
)
from .glb import check_glb
from .http_api import ApiResponse, RigbyApi
from .models import AcceptanceReport, GateResult, Status
from .orientation import camera_orientation_gate, inspect_egocentric_camera_source, semantic_forward_gate
from .report import render_report
from .review import write_review_bundle


DEFAULT_SCENE = {
    "schema_version": "1.0",
    "rig": {
        "id": "mesh2motion-human-vrm1",
        "asset_uri": "assets/models/human-male.glb",
        "profile_uri": "config/rig_profiles/mesh2motion-human-vrm1.json",
        "fixed_root": True,
    },
    "objects": [
        {
            "id": "block",
            "kind": "block",
            "transform": {
                "translation": {"x": 0.00, "y": 1.05, "z": 0.29},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "dimensions_m": {"x": 0.06, "y": 0.08, "z": 0.06},
            "mass_kg": 0.20,
            "friction": 1.0,
            "sockets": [{
                "id": "front_center",
                "transform": {
                    "translation": {"x": 0.0, "y": 0.0, "z": -0.03},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "approach_normal": {"x": 0.0, "y": 0.0, "z": -1.0},
                "grasp_span_m": 0.06,
            }],
        }
    ],
    "fps": 30,
    "reachable_radius_m": 0.72,
}


def _program(body: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    value = body.get("program", body)
    return value if isinstance(value, dict) else None


def _clip(body: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    value = body.get("clip") or body.get("result") or body
    return value if isinstance(value, dict) else {}


def _metrics(body: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    value = body.get("metrics")
    if not isinstance(value, dict):
        clip = _clip(body)
        value = clip.get("metrics") if isinstance(clip, dict) else None
    return value if isinstance(value, dict) else {}


def _provenance(body: dict[str, Any] | None) -> dict[str, Any]:
    value = lookup(body or {}, ["provenance"])
    return value if isinstance(value, dict) else {}


def _glb(response: ApiResponse) -> bytes | None:
    if response.binary:
        return response.binary
    encoded = lookup(response.body or {}, ["glb_base64", "animation_glb_base64"])
    if not isinstance(encoded, str):
        return None
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error):
        return None


def _archive_clip(
    archive: ResultArchive,
    *,
    label: str,
    prompt: str,
    scene: dict[str, Any],
    program: dict[str, Any],
    compile_response: ApiResponse,
    overrides: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    body = compile_response.body or {}
    clip = _clip(body)
    persisted_id = body.get("result_id") if isinstance(body, dict) else None
    if isinstance(persisted_id, str):
        return persisted_id, clip
    result_id = archive.add(
        label=label,
        request={"prompt": prompt, "parameter_overrides": overrides or {}},
        scene=scene,
        program=program,
        clip=clip,
        metrics=_metrics(body),
        provenance={
            **_provenance(body),
            "api_status": compile_response.status,
            "compile_elapsed_ms": compile_response.elapsed_ms,
        },
    )
    return result_id, clip


def _unverified_report(reason: str, criteria: dict[str, Any], started: str, source: str) -> AcceptanceReport:
    names = [
        "planner", "gesture", "gesture_diversity", "grasp_robustness", "physical_proof", "safety_and_quality", "structural_physics",
        "semantic_forward_space", "egocentric_camera_orientation",
        "parametric_control", "overall_latency", "glb_export",
    ]
    return AcceptanceReport(
        run_id="pending",
        started_at=started,
        completed_at=utc_now(),
        source=source,
        synthetic=False,
        gates=[GateResult(name, Status.UNVERIFIED, reason, required=criteria.get({
            "grasp_robustness": "grasp", "physical_proof": "physical_proof",
            "safety_and_quality": "safety", "structural_physics": "structural_physics",
            "semantic_forward_space": "semantic_forward_space", "egocentric_camera_orientation": "egocentric_camera_orientation",
            "overall_latency": "latency", "glb_export": "export",
        }.get(name, name), {}), failures=[reason]) for name in names],
    )


def execute(base_url: str, review_path: Path | None, timeout_s: float, provider: str = "openai") -> tuple[AcceptanceReport, dict[str, Any]]:
    started = utc_now()
    criteria = load_criteria()
    errors = fixture_errors(criteria)
    if errors:
        report = _unverified_report("invalid acceptance fixture set", criteria, started, f"live-public-api:{base_url};provider={provider}")
        report.gates[0] = GateResult("fixture_integrity", Status.ERROR, "Acceptance fixtures are invalid", failures=errors)
        return report, {}

    api = RigbyApi(base_url, timeout_s=timeout_s, provider=provider)
    archive = ResultArchive(POC_ROOT / "results")
    supported_fixture = supported_cases()
    unsupported_fixture = unsupported_cases()

    # A single probe avoids hundreds of repeated connection failures when the app is not running.
    probe = api.plan(supported_fixture[0]["prompt"], DEFAULT_SCENE)
    if probe.status is None:
        return _unverified_report(f"public API unavailable: {probe.error}", criteria, started, f"live-public-api:{base_url};provider={provider}"), {}

    supported_results: list[dict[str, Any]] = []
    compiled_supported: list[dict[str, Any]] = []
    latency_samples: list[float] = []
    for index, case in enumerate(supported_fixture):
        response = probe if index == 0 else api.plan(case["prompt"], DEFAULT_SCENE)
        expected = {key: case[key] for key in ("intent", "hand", "primitive", "object_id")}
        if "motion_profile" in case:
            expected["motion_profile"] = case["motion_profile"]
        supported_results.append({**case, "expected": expected, **response.evidence()})
        program = _program(response.body)
        if program is None or not (response.status and 200 <= response.status < 300):
            continue
        compiled = api.compile(program, DEFAULT_SCENE)
        latency_samples.append(response.elapsed_ms + compiled.elapsed_ms)
        item = {"id": case["id"], "program": program, "scene": DEFAULT_SCENE, "clip": _clip(compiled.body), "metrics": _metrics(compiled.body), "response": compiled}
        if compiled.body is not None and compiled.status and 200 <= compiled.status < 300:
            result_id, _ = _archive_clip(archive, label=case["id"], prompt=case["prompt"], scene=DEFAULT_SCENE, program=program, compile_response=compiled)
            item["result_id"] = result_id
        compiled_supported.append(item)

    unsupported_results = []
    for case in unsupported_fixture:
        response = api.plan(case["prompt"], DEFAULT_SCENE)
        unsupported_results.append({**case, **response.evidence()})

    # Compile the complete 5 x 5 x 2 x 3 physical matrix. Plans are object-local,
    # so identical text reuses a plan while every scene receives its own rollout.
    grasp_results: list[dict[str, Any]] = []
    grasp_plan_cache: dict[str, tuple[dict[str, Any] | None, ApiResponse]] = {}
    for trial in grasp_trials(criteria):
        if trial["prompt"] not in grasp_plan_cache:
            plan_response = api.plan(trial["prompt"], trial["scene"])
            grasp_plan_cache[trial["prompt"]] = (_program(plan_response.body), plan_response)
        program, plan_response = grasp_plan_cache[trial["prompt"]]
        item = {"id": trial["id"], "scene": trial["scene"], "program": program or {}, "metrics": {}}
        if program is None:
            grasp_results.append(item)
            continue
        compiled = api.compile(program, trial["scene"])
        item.update({"clip": _clip(compiled.body), "metrics": _metrics(compiled.body), "response": compiled})
        if compiled.body is not None and compiled.status and 200 <= compiled.status < 300:
            result_id, _ = _archive_clip(archive, label=trial["id"], prompt=trial["prompt"], scene=trial["scene"], program=program, compile_response=compiled)
            item["result_id"] = result_id
        grasp_results.append(item)

    if review_path is not None:
        raise ValueError("reviews cannot be applied during a new automated run; use `python -m evals.review bind` then `finalize`")
    review = None

    planner = planner_gate(supported_results, unsupported_results, criteria["planner"], provider)
    gesture_items = [item for item in compiled_supported if item["id"] in {f"s{i:02d}" for i in range(1, 21)}]
    gesture = gesture_gate(gesture_items, review, criteria["gesture"])
    gesture_diversity = gesture_diversity_gate(gesture_items, criteria["gesture_diversity"])
    grasp, physical = grasp_and_physical_gates(grasp_results, criteria["grasp"], criteria["physical_proof"])
    safety = safety_gate(compiled_supported + grasp_results, criteria["safety"])
    structural = structural_physics_gate(compiled_supported + grasp_results, criteria["structural_physics"])
    rig_profile = load_json(POC_ROOT / "config" / "rig_profiles" / "mesh2motion-human-vrm1.json")
    semantic_forward = semantic_forward_gate(compiled_supported + grasp_results, rig_profile, criteria["semantic_forward_space"])
    camera_source = POC_ROOT / criteria["egocentric_camera_orientation"]["source"]
    camera_orientation = camera_orientation_gate(
        inspect_egocentric_camera_source(camera_source), rig_profile, criteria["egocentric_camera_orientation"]
    )

    # Verify every advertised slider through three deterministic recompiles.
    param_cases: list[dict[str, Any]] = []
    gesture_base = next((item for item in compiled_supported if item["id"] == "s01"), None)
    grasp_base = next((item for item in grasp_results if item.get("program")), None)
    grasp_controls = {"grip_force", "lift_height_m", "hold_duration_s", "block_width_m", "block_height_m", "block_depth_m", "block_mass_kg", "block_friction", "block_x_m", "block_y_m", "block_z_m"}
    scene_paths = {
        "block_width_m": ("dimensions_m", "x"), "block_height_m": ("dimensions_m", "y"),
        "block_depth_m": ("dimensions_m", "z"), "block_mass_kg": (None, "mass_kg"),
        "block_friction": (None, "friction"), "block_x_m": ("transform", "translation", "x"),
        "block_y_m": ("transform", "translation", "y"), "block_z_m": ("transform", "translation", "z"),
    }
    for control in criteria["parametric_control"]["controls"]:
        slider = control["name"]
        base = grasp_base if slider in grasp_controls else gesture_base
        case_result: dict[str, Any] = {
            "control": slider,
            "observed": [],
            "compile_ms": [],
            "planner_calls": [],
            "successful_recompiles": [],
        }
        if not base or not base.get("program"):
            param_cases.append(case_result)
            continue
        for level in control["levels"]:
            scene = copy.deepcopy(base["scene"])
            program = copy.deepcopy(base["program"])
            overrides: dict[str, Any] = {}
            if control["target"] == "scene":
                path = scene_paths[slider]
                cursor: dict[str, Any] = scene["objects"][0]
                for key in path[:-1]:
                    if key is not None:
                        cursor = cursor[key]
                cursor[path[-1]] = level
            elif slider == "handedness":
                overrides["hand"] = level
            else:
                overrides[slider] = level
            response = api.compile(program, scene, overrides or None)
            case_result["compile_ms"].append(response.elapsed_ms)
            observables = lookup(response.body or {}, ["slider_observables", "parametric_observables", "observables"])
            observed = observables.get(control["observable"]) if isinstance(observables, dict) else None
            if isinstance(observed, (int, float)):
                case_result["observed"].append(float(observed))
            provenance = _provenance(response.body)
            planner_calls = lookup(provenance, ["planner_calls", "model_calls"])
            if isinstance(planner_calls, (int, float)):
                case_result["planner_calls"].append(float(planner_calls))
            succeeded = lookup(response.body or {}, ["success"])
            if isinstance(succeeded, bool):
                case_result["successful_recompiles"].append(succeeded)
            if response.body is not None and response.status and 200 <= response.status < 300:
                _archive_clip(
                    archive, label=f"param-{slider}-{level}", prompt="parameter recompilation",
                    scene=scene, program=program, compile_response=response,
                    overrides=overrides or {slider: level},
                )
        param_cases.append(case_result)
    parametric = parametric_gate(param_cases, criteria["parametric_control"])
    latency = latency_gate(latency_samples, criteria["latency"])

    # One gesture plus one center-pose trial for each shape and hand = 11 exports.
    selected = gesture_items[:1]
    selected.extend(
        item for item in grasp_results
        if "center-near" in item["id"] and item["id"].endswith("p1")
    )
    selected = selected[: criteria["export"]["sample_count"]]
    export_checks = []
    node_aliases = {source: canonical for canonical, source in rig_profile["bone_map"].items()}
    for item in selected:
        result_id = item.get("result_id")
        if not result_id:
            export_checks.append({"id": item["id"], "valid": False, "has_provenance": False, "compared_samples": 0, "failures": ["clip was not archived"]})
            continue
        response = api.export(result_id)
        data = _glb(response)
        if data is None:
            export_checks.append({"id": item["id"], "valid": False, "has_provenance": False, "compared_samples": 0, "failures": [response.error or "export did not return GLB bytes"]})
            continue
        check = check_glb(
            data,
            item.get("clip", {}),
            translation_tolerance=criteria["export"]["max_translation_error_m"],
            rotation_tolerance=criteria["export"]["max_rotation_error"],
            node_aliases=node_aliases,
        )
        export_checks.append({"id": item["id"], **check.__dict__})
        if item.get("result_id"):
            archive.attach_glb(item["result_id"], data)
    export_result = export_gate(export_checks, criteria["export"])

    report = AcceptanceReport(
        run_id="pending",
        started_at=started,
        completed_at=utc_now(),
        source=f"live-public-api:{base_url};provider={provider}",
        synthetic=False,
        gates=[planner, gesture, gesture_diversity, grasp, physical, safety, structural, semantic_forward, camera_orientation, parametric, latency, export_result],
        artifacts=[str(archive.index_path.relative_to(POC_ROOT))],
    )
    return report, {
        "parametric_cases": param_cases,
        "export_checks": export_checks,
        "gesture_review_records": [
            {
                "clip_id": item["id"],
                "result_id": item.get("result_id"),
                "available": bool(item.get("result_id")),
            }
            for item in gesture_items
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the complete Rigby acceptance harness against its public API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--review", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds")
    parser.add_argument("--provider", choices=("openai", "offline", "auto"), default="openai")
    args = parser.parse_args(argv)
    if args.review is not None:
        parser.error("--review is unsafe for a new clip set; use `python -m evals.review bind` and `python -m evals.review finalize`")
    report, details = execute(args.base_url, args.review, args.timeout, args.provider)
    criteria = load_criteria()
    ledger = AcceptanceLedger(POC_ROOT / "results" / "acceptance-runs")
    report_dict = report.to_dict()
    preliminary_html = render_report(report_dict)
    index = ledger.record(report_dict, preliminary_html, criteria["runs_required"])
    run_id = index["runs"][-1]["id"]
    report_dict["run_id"] = run_id
    run_root = ledger.root / run_id
    review_records = details.get("gesture_review_records", [])
    if review_records:
        manifest_path, template_path = write_review_bundle(run_root, review_records)
        report_dict.setdefault("artifacts", []).extend([
            str(manifest_path.relative_to(POC_ROOT)),
            str(template_path.relative_to(POC_ROOT)),
        ])
    report_path = ledger.root / run_id / "report.json"
    report_path.write_text(json.dumps(report_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ledger.root / run_id / "report.html").write_text(render_report(report_dict, index), encoding="utf-8")
    print(json.dumps({
        "run_id": run_id,
        "passed": report_dict["passed"],
        "certified": index["certified"],
        "consecutive_passes": index["consecutive_passes"],
        "report": str(report_path),
    }, indent=2))
    return 0 if index["certified"] else 1


if __name__ == "__main__":
    sys.exit(main())
