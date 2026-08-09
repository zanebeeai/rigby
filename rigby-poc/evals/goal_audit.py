from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_METRICS = {
    "max_wrist_swing_rad": "wrist_swing_rad",
    "max_wrist_twist_rad": "wrist_twist_rad",
    "max_forearm_twist_rad": "forearm_twist_rad",
    "max_angular_velocity_rad_s": "angular_velocity_rad_s",
    "max_angular_acceleration_rad_s2": "angular_acceleration_rad_s2",
    "max_angular_jerk_rad_s3": "angular_jerk_rad_s3",
}


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _resolve(project_root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else project_root / path


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _quality_failures(metrics: dict[str, Any], limits: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for measured_key, limit_key in QUALITY_METRICS.items():
        value = metrics.get(measured_key)
        if not isinstance(value, (int, float)):
            failures.append(f"missing {measured_key}")
        elif float(value) > float(limits[limit_key]) + 1e-9:
            failures.append(f"{measured_key}={value} exceeds {limits[limit_key]}")
    collision_frames = metrics.get("self_collision_frames")
    if not isinstance(collision_frames, int):
        failures.append("missing self_collision_frames")
    elif collision_frames != 0:
        failures.append(f"self_collision_frames={collision_frames}")
    visibility = metrics.get("active_hand_visibility_fraction")
    minimum_visibility = float(limits["minimum_active_hand_visibility_fraction"])
    if not isinstance(visibility, (int, float)):
        failures.append("missing active_hand_visibility_fraction")
    elif float(visibility) + 1e-9 < minimum_visibility:
        failures.append(
            f"active_hand_visibility_fraction={visibility} below {minimum_visibility}"
        )
    return failures


def _winner_traces(project_root: Path, batch: dict[str, Any] | None) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    if not batch:
        return []
    return [
        (record, _read(_resolve(project_root, record.get("trace"))))
        for record in batch.get("records", [])
        if isinstance(record, dict) and record.get("status") == "winner_selected"
    ]


def _audit_structural(
    project_root: Path,
    batch: dict[str, Any] | None,
    limits: dict[str, float],
) -> dict[str, Any]:
    traces = _winner_traces(project_root, batch)
    failures: list[str] = []
    expected_fixture = "heldout_hangten_v4_anatomical_shake_precommitted_perceptual_diversity"
    if batch and batch.get("fixture") != expected_fixture:
        failures.append("structural winners come from an obsolete held-out fixture")
    checked = 0
    for record, trace in traces:
        case_id = str(record.get("case_id", "unknown"))
        if not trace or trace.get("status") != "winner_selected":
            failures.append(f"{case_id}: winner trace is missing or incomplete")
            continue
        winner = trace.get("winner")
        if not isinstance(winner, dict):
            failures.append(f"{case_id}: trace has no winner record")
            continue
        checked += 1
        if winner.get("structural_valid") is not True:
            failures.append(f"{case_id}: winner is not structurally valid")
        metrics = winner.get("quality_metrics")
        if not isinstance(metrics, dict):
            failures.append(f"{case_id}: winner has no quality metrics")
            continue
        failures.extend(f"{case_id}: {failure}" for failure in _quality_failures(metrics, limits))
    coverage_complete = len(traces) >= 30 and checked >= 30
    status = "incomplete" if not coverage_complete else "fail" if failures else "pass"
    if len(traces) < 30:
        failures.insert(0, f"only {len(traces)}/30 held-out winners are available")
    return {
        "gate": "structural_validity",
        "status": status,
        "required": {
            "heldout_winners": 30,
            "wrist_limit_violations": 0,
            "self_collision_frames": 0,
            "minimum_active_hand_visibility_fraction": limits[
                "minimum_active_hand_visibility_fraction"
            ],
            "quality_limits": {key: limits[value] for key, value in QUALITY_METRICS.items()},
        },
        "measured": {"winner_traces": len(traces), "quality_records_checked": checked},
        "failures": failures,
    }


def _audit_generator_readiness(sweep: dict[str, Any] | None) -> dict[str, Any]:
    expected_fixture = "heldout_hangten_v4_anatomical_shake_precommitted_perceptual_diversity"
    if not sweep:
        return {
            "gate": "complex_generator_readiness",
            "status": "incomplete",
            "required": {
                "fixture": expected_fixture,
                "heldout_prompts": 30,
                "rankable_candidates": 150,
                "unique_rankable_motion_hashes_within_prompt": 150,
                "wording_coverage": ["anatomical", "natural_shake", "palm_rotation"],
                "cycle_coverage": [2.0, 3.0, 4.0],
                "model_calls": 0,
            },
            "measured": None,
            "missing": ["30-prompt complex no-model structural sweep"],
            "failures": [],
        }
    failures: list[str] = []
    if sweep.get("fixture") != expected_fixture:
        failures.append("structural sweep uses an obsolete held-out fixture")
    if sweep.get("status") != "pass" or sweep.get("ready_for_paid_selection") is not True:
        failures.append("structural sweep did not pass")
    if int(sweep.get("requested_prompts", 0)) < 30 or int(sweep.get("passing_prompts", 0)) < 30:
        failures.append("fewer than 30 complex held-out prompts passed")
    if int(sweep.get("rankable_candidates", 0)) < 150:
        failures.append("fewer than five rankable candidates per held-out prompt")
    if int(sweep.get("unique_rankable_motion_hashes_within_prompt", 0)) < 150:
        failures.append("rankable candidate motion hashes collapsed within prompts")
    if int(sweep.get("model_calls", -1)) != 0:
        failures.append("structural readiness sweep made model calls")
    if set(sweep.get("wording_coverage", [])) != {
        "anatomical",
        "natural_shake",
        "palm_rotation",
    }:
        failures.append("complex shake wording coverage is incomplete")
    if {float(value) for value in sweep.get("cycle_coverage", [])} != {2.0, 3.0, 4.0}:
        failures.append("complex shake cycle coverage is incomplete")
    case_failures = [
        str(case.get("id", "unknown"))
        for case in sweep.get("cases", [])
        if not isinstance(case, dict) or case.get("status") != "pass" or case.get("failures")
    ]
    if case_failures:
        failures.append("failed sweep cases: " + ", ".join(case_failures))
    return {
        "gate": "complex_generator_readiness",
        "status": "fail" if failures else "pass",
        "required": {
            "fixture": expected_fixture,
            "heldout_prompts": 30,
            "rankable_candidates": 150,
            "unique_rankable_motion_hashes_within_prompt": 150,
            "wording_coverage": ["anatomical", "natural_shake", "palm_rotation"],
            "cycle_coverage": [2.0, 3.0, 4.0],
            "model_calls": 0,
        },
        "measured": {
            key: sweep.get(key)
            for key in (
                "fixture",
                "status",
                "ready_for_paid_selection",
                "requested_prompts",
                "passing_prompts",
                "rankable_candidates",
                "unique_rankable_motion_hashes_within_prompt",
                "attempted_candidates",
                "local_rejections",
                "wording_coverage",
                "cycle_coverage",
                "model_calls",
            )
        },
        "missing": [],
        "failures": failures,
    }


def _audit_judge(
    calibration: dict[str, Any] | None,
    human_score: dict[str, Any] | None,
    heldout_judge_score: dict[str, Any] | None,
) -> dict[str, Any]:
    failures: list[str] = []
    missing: list[str] = []
    summary = calibration.get("summary", {}) if calibration else {}
    if not calibration:
        missing.append("28-case anatomical corruption calibration report")
    else:
        if int(summary.get("completed_wrist_corruptions", 0)) < 12:
            failures.append("fewer than 12 wrist corruptions were calibrated")
        if int(summary.get("completed_wrong_joint_shake_corruptions", 0)) < 4:
            failures.append("fewer than 4 wrong-joint shake corruptions were calibrated")
        if int(summary.get("completed_timing_corruptions", 0)) < 4:
            failures.append("fewer than 4 timing corruptions were calibrated")
        if float(summary.get("false_acceptance_rate", 1.0)) >= 0.05:
            failures.append("hybrid false acceptance is not under 5%")
        if float(summary.get("ab_order_consistency", 0.0)) < 0.90:
            failures.append("A/B-order consistency is below 90%")
    if not human_score:
        missing.append("obvious-pair human calibration score")
    else:
        if int(human_score.get("complete", 0)) < 10:
            failures.append("fewer than 10 obvious human comparisons are complete")
        if float(human_score.get("vlm_human_agreement_rate", 0.0)) < 0.80:
            failures.append("obvious-pair human agreement is below 80%")
        if float(human_score.get("ab_order_consistency_rate", 0.0)) < 0.90:
            failures.append("human-calibrated A/B consistency is below 90%")
    heldout_metrics = heldout_judge_score.get("metrics", {}) if heldout_judge_score else {}
    if not heldout_judge_score:
        missing.append("held-out VLM/human comparison score")
    else:
        completed = int(heldout_metrics.get("completed_pairs", 0))
        total = int(heldout_metrics.get("total_pairs", 0))
        if completed < 18 or completed != total:
            failures.append("held-out VLM/human comparison is incomplete")
        if float(heldout_metrics.get("vlm_human_agreement_rate", 0.0)) < 0.70:
            failures.append("held-out VLM/human agreement is below 70%")
    status = "incomplete" if missing else "fail" if failures else "pass"
    return {
        "gate": "judge_validity",
        "status": status,
        "required": {
            "wrist_corruptions": 12,
            "wrong_joint_shake_corruptions": 4,
            "timing_corruptions": 4,
            "maximum_false_acceptance_rate": 0.05,
            "minimum_obvious_human_agreement": 0.80,
            "minimum_heldout_human_agreement": 0.70,
            "minimum_ab_order_consistency": 0.90,
        },
        "measured": {
            "calibration_summary": summary or None,
            "obvious_human_score": human_score,
            "heldout_judge_score": heldout_judge_score,
        },
        "missing": missing,
        "failures": failures,
    }


def _candidate_trace_failures(case_id: str, trace: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    motion_hashes: list[str] = []
    rankable: list[dict[str, Any]] = []
    candidates = [
        candidate
        for round_record in trace.get("rounds", [])
        if isinstance(round_record, dict)
        for candidate in round_record.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    if len(candidates) < 5:
        failures.append(f"{case_id}: only {len(candidates)} candidates are preserved")
    required = {
        "recipe",
        "result_id",
        "motion_sha256",
        "compile_success",
        "structural_valid",
        "structural_failures",
        "quality_metrics",
        "accepted",
        "perceptual_descriptor",
    }
    for index, candidate in enumerate(candidates, start=1):
        absent = sorted(required.difference(candidate))
        if absent:
            failures.append(f"{case_id}: candidate {index} misses {', '.join(absent)}")
        motion_hash = candidate.get("motion_sha256")
        if isinstance(motion_hash, str) and motion_hash:
            motion_hashes.append(motion_hash)
        if candidate.get("structural_valid") is True:
            if candidate.get("rejection_stage") == "perceptual_diversity_prefilter":
                if not candidate.get("rejection_reason"):
                    failures.append(
                        f"{case_id}: near-duplicate candidate {index} lacks its rejection reason"
                    )
                comparisons = candidate.get("perceptual_diversity")
                if not isinstance(comparisons, list) or not any(
                    isinstance(item, dict) and item.get("perceptually_distinct") is False
                    for item in comparisons
                ):
                    failures.append(
                        f"{case_id}: near-duplicate candidate {index} lacks measured evidence"
                    )
            else:
                for field in ("evidence_manifest", "judgment"):
                    if field not in candidate:
                        failures.append(f"{case_id}: valid candidate {index} misses {field}")
                if candidate.get("evidence_manifest"):
                    rankable.append(candidate)
        elif candidate.get("rejection_stage") != "structural_prefilter":
            failures.append(f"{case_id}: invalid candidate {index} lacks structural rejection")
    if len(set(motion_hashes)) < min(5, len(candidates)):
        failures.append(f"{case_id}: candidate motion hashes collapsed")
    if len(rankable) < 5:
        failures.append(f"{case_id}: only {len(rankable)}/5 perceptually rankable candidates")
    for rank_index, candidate in enumerate(rankable):
        comparisons = candidate.get("perceptual_diversity")
        if not isinstance(comparisons, list) or len(comparisons) != rank_index:
            failures.append(
                f"{case_id}: rankable candidate {rank_index + 1} lacks complete pairwise diversity evidence"
            )
            continue
        if any(
            not isinstance(item, dict) or item.get("perceptually_distinct") is not True
            for item in comparisons
        ):
            failures.append(
                f"{case_id}: rankable candidate {rank_index + 1} is not perceptually distinct"
            )
    for repair in trace.get("repairs", []):
        if not isinstance(repair, dict) or not {"source_result_id", "record", "patch"}.issubset(repair):
            failures.append(f"{case_id}: repair provenance is incomplete")
    return failures, motion_hashes


def _audit_best_of_five(
    project_root: Path,
    batch: dict[str, Any] | None,
    preference_score: dict[str, Any] | None,
) -> dict[str, Any]:
    traces = _winner_traces(project_root, batch)
    failures: list[str] = []
    winner_hashes: list[str] = []
    complete_traces = 0
    baseline_selected = 0
    fair_baselines = 0
    perceptually_rankable = 0
    expected_fixture = "heldout_hangten_v4_anatomical_shake_precommitted_perceptual_diversity"
    protocol_current = bool(batch and batch.get("fixture") == expected_fixture)
    if batch and not protocol_current:
        failures.append(
            "held-out batch predates the fair perceptual-diversity protocol"
        )
    for record, trace in traces:
        case_id = str(record.get("case_id", "unknown"))
        if not trace or trace.get("status") != "winner_selected":
            failures.append(f"{case_id}: missing winner trace")
            continue
        complete_traces += 1
        if protocol_current:
            trace_failures, _ = _candidate_trace_failures(case_id, trace)
            failures.extend(trace_failures)
            baseline = trace.get("baseline_selection", {})
            if (
                isinstance(baseline, dict)
                and baseline.get("method") == "precommitted_prompt_hash"
                and baseline.get("selected_before_judging") is True
                and isinstance(baseline.get("candidate_index"), int)
                and 1 <= int(baseline["candidate_index"]) <= 5
            ):
                fair_baselines += 1
            else:
                failures.append(f"{case_id}: single-sample baseline was not fairly precommitted")
            rankable_count = sum(
                bool(candidate.get("evidence_manifest"))
                for round_record in trace.get("rounds", [])
                if isinstance(round_record, dict)
                for candidate in round_record.get("candidates", [])
                if isinstance(candidate, dict)
            )
            perceptually_rankable += int(rankable_count >= 5)
        winner = trace.get("winner", {})
        winner_hash = winner.get("motion_sha256") if isinstance(winner, dict) else None
        if isinstance(winner_hash, str) and winner_hash:
            winner_hashes.append(winner_hash)
        if trace.get("baseline_result_id") == trace.get("winner_result_id"):
            baseline_selected += 1
    if len(winner_hashes) != len(set(winner_hashes)):
        failures.append("winner motion hashes are duplicated across held-out prompts")
    missing: list[str] = []
    if complete_traces < 30:
        missing.append(f"only {complete_traces}/30 complete best-of-five traces")
    if not preference_score:
        missing.append("blinded baseline-versus-winner human score")
    elif int(preference_score.get("complete", 0)) < 30:
        failures.append("human uplift review has fewer than 30 completed prompts")
    elif float(preference_score.get("winner_preference_rate", 0.0)) < 0.70:
        failures.append("winner preference rate is below 70%")
    status = "incomplete" if missing else "fail" if failures else "pass"
    return {
        "gate": "best_of_five_uplift",
        "status": status,
        "required": {
            "heldout_prompts": 30,
            "candidates_per_prompt": 5,
            "minimum_human_winner_preference": 0.70,
            "complete_trace_provenance": True,
            "fair_precommitted_single_sample_baseline": True,
            "pairwise_perceptual_diversity_evidence": True,
        },
        "measured": {
            "winner_records": len(traces),
            "complete_traces": complete_traces,
            "unique_winner_motion_hashes": len(set(winner_hashes)),
            "baseline_selected_automatic_ties": baseline_selected,
            "fair_precommitted_baselines": fair_baselines,
            "traces_with_five_perceptually_rankable_candidates": perceptually_rankable,
            "preference_score": preference_score,
        },
        "missing": missing,
        "failures": failures,
    }


def _gate_by_name(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next(
        (
            gate
            for gate in report.get("gates", [])
            if isinstance(gate, dict) and gate.get("gate") == name
        ),
        None,
    )


def _audit_acceptance_run(project_root: Path, record: dict[str, Any], limits: dict[str, float]) -> tuple[list[str], dict[str, Any]]:
    run_id = str(record.get("id", "unknown"))
    report_path = _resolve(project_root / "results" / "acceptance-runs", record.get("report"))
    report = _read(report_path)
    failures: list[str] = []
    measured: dict[str, Any] = {"run_id": run_id, "report": str(report_path)}
    if not report:
        return [f"run {run_id}: report is missing"], measured
    gesture = _gate_by_name(report, "gesture")
    diversity = _gate_by_name(report, "gesture_diversity")
    measured["gesture"] = gesture
    measured["gesture_diversity"] = diversity
    if not gesture or gesture.get("status") != "pass":
        failures.append(f"run {run_id}: gesture review did not pass")
    else:
        gesture_measured = gesture.get("measured", {})
        if int(gesture_measured.get("reviews_complete", 0)) < 20:
            failures.append(f"run {run_id}: fewer than 20 clips were reviewed")
        if int(gesture_measured.get("review_passes", 0)) < 16:
            failures.append(f"run {run_id}: fewer than 16 clips scored 4+ in both views")
    if not diversity or diversity.get("status") != "pass":
        failures.append(f"run {run_id}: diversity gate did not pass")
    elif int(diversity.get("measured", {}).get("unique_motion_hashes", 0)) < 20:
        failures.append(f"run {run_id}: fewer than 20 unique motion hashes")

    manifest_path = report_path.parent / "blinded-gesture-manifest.json"
    manifest = _read(manifest_path)
    structural_checked = 0
    metrics_missing = 0
    structurally_invalid = 0
    quality_failure_clips = 0
    quality_failure_kinds: set[str] = set()
    if not manifest or len(manifest.get("records", [])) < 20:
        failures.append(f"run {run_id}: 20-clip manifest is missing")
    else:
        for clip in manifest["records"]:
            result_id = str(clip.get("result_id", ""))
            metrics = _read(project_root / "results" / result_id / "metrics.json")
            if not metrics:
                metrics_missing += 1
                continue
            structural_checked += 1
            if metrics.get("structural_valid") is not True:
                structurally_invalid += 1
            clip_quality_failures = _quality_failures(metrics, limits)
            if clip_quality_failures:
                quality_failure_clips += 1
                quality_failure_kinds.update(clip_quality_failures)
    if metrics_missing:
        failures.append(f"run {run_id}: metrics are missing for {metrics_missing}/20 clips")
    if structurally_invalid:
        failures.append(
            f"run {run_id}: structural_valid is not true for {structurally_invalid}/20 clips"
        )
    if quality_failure_clips:
        failures.append(
            f"run {run_id}: {quality_failure_clips}/20 clips lack or violate quality metrics"
        )
    measured["structural_clips_checked"] = structural_checked
    measured["metrics_missing"] = metrics_missing
    measured["structurally_invalid"] = structurally_invalid
    measured["quality_failure_clips"] = quality_failure_clips
    measured["quality_failure_kinds"] = sorted(quality_failure_kinds)
    return failures, measured


def _audit_final_acceptance(
    project_root: Path,
    acceptance_index: dict[str, Any] | None,
    limits: dict[str, float],
) -> dict[str, Any]:
    eligible = [
        record
        for record in (acceptance_index or {}).get("runs", [])
        if isinstance(record, dict) and not record.get("synthetic")
    ]
    latest = eligible[-2:]
    failures: list[str] = []
    measured_runs: list[dict[str, Any]] = []
    if len(latest) == 2:
        try:
            if int(latest[1]["id"]) != int(latest[0]["id"]) + 1:
                failures.append("latest two acceptance runs are not consecutive")
        except (KeyError, TypeError, ValueError):
            failures.append("acceptance run IDs are invalid")
        for record in latest:
            run_failures, measured = _audit_acceptance_run(project_root, record, limits)
            failures.extend(run_failures)
            measured_runs.append(measured)
    missing = [] if len(latest) == 2 else [f"only {len(latest)}/2 non-synthetic acceptance runs"]
    status = "incomplete" if missing else "fail" if failures else "pass"
    return {
        "gate": "final_rigby_acceptance",
        "status": status,
        "required": {
            "consecutive_runs": 2,
            "clips_per_run": 20,
            "minimum_clips_scoring_4_plus_in_both_views": 16,
            "unique_motion_hashes_per_run": 20,
            "all_structural_gates": True,
        },
        "measured": {"runs": measured_runs},
        "missing": missing,
        "failures": failures,
    }


def build_goal_audit(
    *,
    project_root: Path = PROJECT_ROOT,
    calibration_report: Path | None = None,
    judge_human_score: Path | None = None,
    batch_report: Path | None = None,
    preference_score: Path | None = None,
    acceptance_index: Path | None = None,
    structural_sweep_report: Path | None = None,
    heldout_judge_score: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    calibration_root = (
        project_root / "results/judge-calibration/005673-v4-pronation/run-02-truthful-joints"
    )
    calibration_report = calibration_report or calibration_root / "calibration-report.json"
    judge_human_score = judge_human_score or calibration_root / "human-review/judge-human-score.json"
    batch_report = batch_report or (
        project_root / "results/heldout-flywheel/complex-v4-selection/batch-report.json"
    )
    preference_score = preference_score or (
        project_root
        / "results/heldout-flywheel/complex-v4-selection/human-review/preference-score.json"
    )
    acceptance_index = acceptance_index or project_root / "results/acceptance-runs/index.json"
    structural_sweep_report = structural_sweep_report or (
        project_root
        / "results/complex-hangten-structural-sweep-v4/structural-sweep-report.json"
    )
    heldout_judge_score = heldout_judge_score or (
        project_root
        / "results/heldout-flywheel/complex-v4-selection/human-review/preference-score.json"
    )
    quality = _read(project_root / "config/motion_quality_reference.json")
    if not quality or not isinstance(quality.get("hard_limits"), dict):
        raise ValueError("motion quality reference is missing")
    limits = quality["hard_limits"]
    calibration = _read(calibration_report)
    human_score = _read(judge_human_score)
    batch = _read(batch_report)
    preference = _read(preference_score)
    gates = [
        _audit_generator_readiness(_read(structural_sweep_report)),
        _audit_structural(project_root, batch, limits),
        _audit_judge(calibration, human_score, _read(heldout_judge_score)),
        _audit_best_of_five(project_root, batch, preference),
        _audit_final_acceptance(project_root, _read(acceptance_index), limits),
    ]
    ready = all(gate["status"] == "pass" for gate in gates)
    return {
        "schema_version": "1.0",
        "objective": "hangten_before_block_pickup",
        "status": "pass" if ready else "not_ready",
        "ready_for_block_pickup": ready,
        "sources": {
            "quality_reference": str(project_root / "config/motion_quality_reference.json"),
            "calibration_report": str(calibration_report),
            "judge_human_score": str(judge_human_score),
            "batch_report": str(batch_report),
            "preference_score": str(preference_score),
            "acceptance_index": str(acceptance_index),
            "structural_sweep_report": str(structural_sweep_report),
            "heldout_judge_score": str(heldout_judge_score),
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit every gate required before Rigby block pickup")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--judge-human-score", type=Path)
    parser.add_argument("--batch-report", type=Path)
    parser.add_argument("--preference-score", type=Path)
    parser.add_argument("--acceptance-index", type=Path)
    parser.add_argument("--structural-sweep-report", type=Path)
    parser.add_argument("--heldout-judge-score", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/hangten-goal-audit.json"))
    arguments = parser.parse_args()
    audit = build_goal_audit(
        project_root=arguments.project_root,
        calibration_report=arguments.calibration_report,
        judge_human_score=arguments.judge_human_score,
        batch_report=arguments.batch_report,
        preference_score=arguments.preference_score,
        acceptance_index=arguments.acceptance_index,
        structural_sweep_report=arguments.structural_sweep_report,
        heldout_judge_score=arguments.heldout_judge_score,
    )
    output = arguments.output
    if not output.is_absolute():
        output = arguments.project_root / output
    _atomic_write(output, audit)
    print(output)


if __name__ == "__main__":
    main()
