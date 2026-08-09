from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rigby_poc.compiler import PROJECT_ROOT


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _candidate(trace: dict[str, Any], result_id: str) -> dict[str, Any] | None:
    for round_record in trace.get("rounds", []):
        for candidate in round_record.get("candidates", []):
            if candidate.get("result_id") == result_id:
                return candidate
    return None


def _audit_run(project_root: Path, run: dict[str, Any], expected_intent: str) -> dict[str, Any]:
    failures: list[str] = []
    trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
    winner_id = str(run.get("winner_result_id") or "")
    rounds = trace.get("rounds", []) if isinstance(trace.get("rounds"), list) else []
    candidates = [item for round_record in rounds for item in round_record.get("candidates", [])]
    planner = trace.get("planner") if isinstance(trace.get("planner"), dict) else {}
    intent = (planner.get("program") or {}).get("intent")
    if run.get("status") != "completed" or trace.get("status") != "winner_selected":
        failures.append("run did not finish with a selected winner")
    if intent != expected_intent:
        failures.append(f"planner intent is {intent!r}, expected {expected_intent!r}")
    if len(candidates) < 5:
        failures.append(f"only {len(candidates)}/5 candidates were recorded")
    winner = _candidate(trace, winner_id) if winner_id else None
    if not winner:
        failures.append("winner candidate is missing from the trace")
    else:
        judgment = winner.get("judgment") if isinstance(winner.get("judgment"), dict) else {}
        for key in ("semantic_match", "gesture_recognizability", "anatomical_naturalness", "overall"):
            if int(judgment.get(key, 0)) < 4:
                failures.append(f"winner {key} is below 4")
        if not judgment.get("accept"):
            failures.append("winner was not accepted by the visual judge")
        if winner.get("compile_success") is not True or winner.get("structural_valid") is not True:
            failures.append("winner did not pass deterministic structural checks")
        manifest = _read(Path(str(winner.get("evidence_manifest", ""))))
        contract = manifest.get("capture_contract", {}) if manifest else {}
        if contract.get("width_px") != 1600 or contract.get("height_px") != 900:
            failures.append("winner evidence does not preserve the 1600x900 source frame")
        if contract.get("egocentric_vertical_fov_deg") != 94.0 or not contract.get("raw_canvas_only"):
            failures.append("winner evidence does not preserve the complete egocentric FOV")
    result_dir = project_root / "results" / winner_id
    if winner_id and not (result_dir / "animation.glb").is_file():
        failures.append("winner GLB is missing")
    return {
        "run_id": run.get("run_id"),
        "intent": intent,
        "status": "pass" if not failures else "fail",
        "winner_result_id": winner_id or None,
        "candidate_count": len(candidates),
        "failures": failures,
    }


def build_audit(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    calibration_root = project_root / "results/judge-calibration/005673-v4-pronation/run-02-truthful-joints"
    automated = _read(calibration_root / "calibration-report.json") or {}
    frozen_human = _read(calibration_root / "human-review/judge-human-score.json") or {}
    judge_failures: list[str] = []
    if not automated.get("summary", {}).get("passed"):
        judge_failures.append("automated corruption calibration is not passing")
    if not frozen_human.get("passed"):
        judge_failures.append("frozen final human calibration is not passing")

    runs: list[dict[str, Any]] = []
    root = project_root / "results/pipeline-runs"
    for path in sorted(root.glob("*/run.json"), reverse=True):
        record = _read(path)
        if not record:
            continue
        trace_path = path.parent / "artifacts/flywheel-trace.json"
        trace = _read(trace_path)
        if trace:
            record["trace"] = trace
        runs.append(record)
    gesture = next(
        (run for run in runs if (run.get("trace", {}).get("planner", {}).get("program", {}).get("intent") == "gesture") and run.get("status") == "completed"),
        None,
    )
    grab = next(
        (run for run in runs if (run.get("trace", {}).get("planner", {}).get("program", {}).get("intent") == "grab") and run.get("status") == "completed"),
        None,
    )
    gates = [
        {
            "gate": "calibrated_autonomous_judge",
            "status": "pass" if not judge_failures else "fail",
            "human_reviews_required_after_calibration": 0,
            "measured": {
                "vlm_human_agreement_rate": frozen_human.get("vlm_human_agreement_rate"),
                "ab_order_consistency_rate": frozen_human.get("ab_order_consistency_rate"),
                "automated_false_accepts": automated.get("summary", {}).get("false_accepts"),
            },
            "failures": judge_failures,
        },
        _audit_run(project_root, gesture, "gesture") if gesture else {
            "gate": "gesture_pipeline_smoke", "status": "incomplete", "failures": ["no completed autonomous gesture run"]
        },
        _audit_run(project_root, grab, "grab") if grab else {
            "gate": "pickup_pipeline_smoke", "status": "incomplete", "failures": ["no completed autonomous pickup run"]
        },
    ]
    gates[1]["gate"] = "gesture_pipeline_smoke"
    gates[2]["gate"] = "pickup_pipeline_smoke"
    passed = all(gate["status"] == "pass" for gate in gates)
    return {
        "schema_version": "1.0",
        "objective": "autonomous_text_to_egocentric_animation",
        "evaluation_policy": "no_future_human_judging",
        "status": "pass" if passed else "not_ready",
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Rigby's autonomous authoring goal")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("results/autonomous-goal-audit.json"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.project_root / args.output
    audit = build_audit(args.project_root.resolve())
    _atomic(output, audit)
    print(output)


if __name__ == "__main__":
    main()
