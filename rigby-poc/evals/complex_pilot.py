from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.flywheel import run_best_of_five
from evals.heldout import heldout_complex_hangten_cases
from evals.variant_review import create_multi_variant_review
from rigby_poc.quality import quality_reference


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _existing_trace(path: Path, prompt: str) -> bool:
    if not path.is_file():
        return False
    trace = _read(path)
    return bool(
        trace.get("prompt") == prompt
        and trace.get("selection_mode") == "human_pilot"
        and trace.get("status") == "awaiting_human_selection"
    )


def _case_report(case: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    trace = _read(trace_path)
    candidates = [
        candidate
        for round_record in trace.get("rounds", [])
        for candidate in round_record.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    rankable = [candidate for candidate in candidates if candidate.get("human_review_ready")]
    baseline = next(
        (
            candidate
            for candidate in candidates
            if candidate.get("result_id") == trace.get("baseline_result_id")
        ),
        None,
    )
    limits = quality_reference()["hard_limits"]
    quality_failures: list[str] = []
    for candidate in rankable:
        metrics = candidate.get("quality_metrics", {})
        checks = {
            "max_wrist_swing_rad": ("maximum", limits["wrist_swing_rad"]),
            "max_wrist_twist_rad": ("maximum", limits["wrist_twist_rad"]),
            "max_forearm_twist_rad": ("maximum", limits["forearm_twist_rad"]),
            "max_angular_velocity_rad_s": ("maximum", limits["angular_velocity_rad_s"]),
            "max_angular_acceleration_rad_s2": (
                "maximum",
                limits["angular_acceleration_rad_s2"],
            ),
            "max_angular_jerk_rad_s3": ("maximum", limits["angular_jerk_rad_s3"]),
            "active_hand_visibility_fraction": (
                "minimum",
                limits["minimum_active_hand_visibility_fraction"],
            ),
        }
        for name, (direction, bound) in checks.items():
            value = metrics.get(name)
            if value is None or (
                direction == "maximum" and float(value) > float(bound) + 1e-8
            ) or (
                direction == "minimum" and float(value) < float(bound) - 1e-8
            ):
                quality_failures.append(f"{candidate.get('result_id')}: {name}={value}")
        if int(metrics.get("self_collision_frames", -1)) != 0:
            quality_failures.append(
                f"{candidate.get('result_id')}: self_collision_frames="
                f"{metrics.get('self_collision_frames')}"
            )

    semantic_failures: list[str] = []
    expected_cycles = float(case["expected_cycles"])
    for candidate in rankable:
        descriptor = candidate.get("perceptual_descriptor", {})
        cycles = float(descriptor.get("shake_cycles", 0.0))
        reversals = int(descriptor.get("shake_reversal_count", 0))
        if abs(cycles - expected_cycles) > 1e-8:
            semantic_failures.append(
                f"{candidate.get('result_id')}: cycles={cycles}, expected={expected_cycles}"
            )
        if reversals < max(1, int(round(expected_cycles * 2.0)) - 1):
            semantic_failures.append(
                f"{candidate.get('result_id')}: reversals={reversals}"
            )
        if float(descriptor.get("recovery_endpoint_wrist_error_m", 1.0)) > 1e-6:
            semantic_failures.append(
                f"{candidate.get('result_id')}: recovery wrist endpoint mismatch"
            )
        if float(descriptor.get("recovery_endpoint_hand_error_rad", 1.0)) > 1e-6:
            semantic_failures.append(
                f"{candidate.get('result_id')}: recovery hand endpoint mismatch"
            )

    motion_hashes = [str(candidate.get("motion_sha256")) for candidate in rankable]
    failures: list[str] = []
    if trace.get("planner", {}).get("model_calls") != 0:
        failures.append("planner made a model call")
    if len(rankable) != 5:
        failures.append(f"expected five rankable candidates, found {len(rankable)}")
    if len(set(motion_hashes)) != len(motion_hashes):
        failures.append("rankable candidate motion hashes are not unique")
    if baseline is None or not baseline.get("compile_success") or not baseline.get("structural_valid"):
        failures.append("precommitted single-sample baseline is not structurally valid")
    failures.extend(quality_failures)
    failures.extend(semantic_failures)
    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "hand": case["hand"],
        "expected_cycles": expected_cycles,
        "trace": str(trace_path),
        "baseline_result_id": trace.get("baseline_result_id"),
        "baseline_recipe": trace.get("baseline_selection", {}).get("candidate_recipe"),
        "attempted_candidates": len(candidates),
        "rankable_candidates": len(rankable),
        "rankable_result_ids": [candidate["result_id"] for candidate in rankable],
        "unique_rankable_motion_hashes": len(set(motion_hashes)),
        "local_rejections": [
            {
                "result_id": candidate.get("result_id"),
                "recipe": candidate.get("recipe", {}).get("name"),
                "stage": candidate.get("rejection_stage"),
                "reason": candidate.get("rejection_reason"),
            }
            for candidate in candidates
            if candidate.get("rejection_stage")
        ],
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def run_complex_human_pilot(
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_paths: list[Path] = []
    case_reports: list[dict[str, Any]] = []
    for case in heldout_complex_hangten_cases():
        case_dir = output_dir / str(case["id"])
        trace_path = case_dir / "flywheel-trace.json"
        if not _existing_trace(trace_path, str(case["prompt"])):
            trace_path = run_best_of_five(
                str(case["prompt"]),
                case_dir,
                provider="offline",
                base_url=base_url,
                max_rounds=1,
                selection_mode="human_pilot",
            )
        trace_paths.append(trace_path)
        case_reports.append(_case_report(case, trace_path))

    review_dir = output_dir / "human-review"
    manifest_path, answer_path = create_multi_variant_review(trace_paths, review_dir)
    manifest = _read(manifest_path)
    report = {
        "schema_version": "1.0",
        "kind": "complex_hangten_no_model_human_pilot",
        "status": (
            "awaiting_human_review"
            if all(case["status"] == "pass" for case in case_reports)
            else "generation_failed"
        ),
        "model_calls": 0,
        "case_count": len(case_reports),
        "comparison_count": len(manifest.get("records", [])),
        "cases": case_reports,
        "review_manifest": str(manifest_path),
        "answer_key": str(answer_path),
        "review_url": (
            f"{base_url.rstrip('/')}/comparison.html?manifest=/"
            + str(manifest_path).replace("\\", "/")
        ),
    }
    report_path = output_dir / "pilot-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Rigby's complex Hang Ten human pilot")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    arguments = parser.parse_args()
    print(run_complex_human_pilot(arguments.output_dir, base_url=arguments.base_url))


if __name__ == "__main__":
    main()
