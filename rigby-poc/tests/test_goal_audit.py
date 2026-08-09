from __future__ import annotations

import json
from pathlib import Path

from evals.goal_audit import _audit_judge, _candidate_trace_failures, build_goal_audit


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _project(tmp_path: Path) -> Path:
    _write(
        tmp_path / "config/motion_quality_reference.json",
        {
            "hard_limits": {
                "angular_velocity_rad_s": 9.5,
                "angular_acceleration_rad_s2": 82.0,
                "angular_jerk_rad_s3": 2100.0,
                "wrist_swing_rad": 0.48,
                "wrist_twist_rad": 0.18,
                "forearm_twist_rad": 1.35,
                "minimum_active_hand_visibility_fraction": 1.0,
            }
        },
    )
    _write(tmp_path / "results/acceptance-runs/index.json", {"runs": []})
    return tmp_path


def test_goal_audit_never_converts_missing_evidence_into_pass(tmp_path: Path) -> None:
    root = _project(tmp_path)
    audit = build_goal_audit(project_root=root)
    assert audit["ready_for_block_pickup"] is False
    assert audit["status"] == "not_ready"
    assert {gate["status"] for gate in audit["gates"]} == {"incomplete"}


def test_judge_gate_requires_timing_humans_and_heldout_agreement(tmp_path: Path) -> None:
    root = _project(tmp_path)
    calibration = _write(
        root / "calibration.json",
        {
            "summary": {
                "completed_wrist_corruptions": 12,
                "completed_timing_corruptions": 0,
                "false_acceptance_rate": 0.0,
                "ab_order_consistency": 0.9,
            }
        },
    )
    audit = build_goal_audit(project_root=root, calibration_report=calibration)
    judge = next(gate for gate in audit["gates"] if gate["gate"] == "judge_validity")
    assert judge["status"] == "incomplete"
    assert "fewer than 4 timing corruptions were calibrated" in judge["failures"]
    assert "obvious-pair human calibration score" in judge["missing"]
    assert "held-out VLM/human comparison score" in judge["missing"]


def test_judge_gate_reuses_nonautomatic_uplift_pairs_for_heldout_agreement() -> None:
    gate = _audit_judge(
        {
            "summary": {
                "completed_wrist_corruptions": 12,
                "completed_wrong_joint_shake_corruptions": 4,
                "completed_timing_corruptions": 4,
                "false_acceptance_rate": 0.0,
                "ab_order_consistency": 1.0,
            }
        },
        {
            "complete": 10,
            "vlm_human_agreement_rate": 0.8,
            "ab_order_consistency_rate": 1.0,
        },
        {
            "metrics": {
                "completed_pairs": 29,
                "total_pairs": 29,
                "vlm_human_agreement_rate": 0.72,
                "automatic_baseline_ties_excluded": 1,
            }
        },
    )
    assert gate["status"] == "pass"
    assert gate["failures"] == []


def test_final_acceptance_requires_two_reviewed_diverse_structural_runs(tmp_path: Path) -> None:
    root = _project(tmp_path)
    runs = []
    quality = {
        "structural_valid": True,
        "max_wrist_swing_rad": 0.2,
        "max_wrist_twist_rad": 0.05,
        "max_forearm_twist_rad": 0.7,
        "self_collision_frames": 0,
        "active_hand_visibility_fraction": 1.0,
        "max_angular_velocity_rad_s": 4.0,
        "max_angular_acceleration_rad_s2": 40.0,
        "max_angular_jerk_rad_s3": 500.0,
    }
    for run_number in (1, 2):
        run_id = f"{run_number:06d}"
        run_dir = root / "results/acceptance-runs" / run_id
        records = []
        for clip_number in range(20):
            result_id = f"{run_number * 100 + clip_number:06d}-motion"
            _write(root / "results" / result_id / "metrics.json", quality)
            records.append({"clip_id": f"s{clip_number + 1:02d}", "result_id": result_id})
        _write(run_dir / "blinded-gesture-manifest.json", {"records": records})
        _write(
            run_dir / "report.json",
            {
                "gates": [
                    {
                        "gate": "gesture",
                        "status": "pass",
                        "measured": {"reviews_complete": 20, "review_passes": 16},
                    },
                    {
                        "gate": "gesture_diversity",
                        "status": "pass",
                        "measured": {"unique_motion_hashes": 20},
                    },
                ]
            },
        )
        runs.append(
            {
                "id": run_id,
                "synthetic": False,
                "report": f"{run_id}/report.json",
            }
        )
    index = _write(root / "results/acceptance-runs/index.json", {"runs": runs})
    audit = build_goal_audit(project_root=root, acceptance_index=index)
    final = next(gate for gate in audit["gates"] if gate["gate"] == "final_rigby_acceptance")
    assert final["status"] == "pass"
    assert final["failures"] == []


def test_best_of_five_trace_requires_measured_pairwise_perceptual_diversity() -> None:
    candidates = []
    for index in range(5):
        candidates.append(
            {
                "recipe": {"name": f"recipe-{index}"},
                "result_id": f"result-{index}",
                "motion_sha256": f"hash-{index}",
                "compile_success": True,
                "structural_valid": True,
                "structural_failures": [],
                "quality_metrics": {},
                "accepted": True,
                "perceptual_descriptor": {"samples": [{}]},
                "perceptual_diversity": [
                    {
                        "against_result_id": f"result-{prior}",
                        "perceptually_distinct": True,
                    }
                    for prior in range(index)
                ],
                "evidence_manifest": f"candidate-{index}/evidence-manifest.json",
                "judgment": {"accept": True},
            }
        )
    trace = {"rounds": [{"candidates": candidates}], "repairs": []}
    failures, hashes = _candidate_trace_failures("h01", trace)
    assert failures == []
    assert len(hashes) == 5

    candidates[4]["perceptual_diversity"][0]["perceptually_distinct"] = False
    failures, _ = _candidate_trace_failures("h01", trace)
    assert any("is not perceptually distinct" in failure for failure in failures)


def test_complex_generator_gate_requires_full_no_model_coverage(tmp_path: Path) -> None:
    root = _project(tmp_path)
    sweep = _write(
        root / "sweep.json",
        {
            "fixture": "heldout_hangten_v4_anatomical_shake_precommitted_perceptual_diversity",
            "status": "pass",
            "ready_for_paid_selection": True,
            "model_calls": 0,
            "requested_prompts": 30,
            "passing_prompts": 30,
            "rankable_candidates": 150,
            "unique_rankable_motion_hashes_within_prompt": 150,
            "attempted_candidates": 210,
            "local_rejections": 60,
            "wording_coverage": ["anatomical", "natural_shake", "palm_rotation"],
            "cycle_coverage": [2.0, 3.0, 4.0],
            "cases": [
                {"id": f"hx{index + 1:02d}", "status": "pass", "failures": []}
                for index in range(30)
            ],
        },
    )
    audit = build_goal_audit(project_root=root, structural_sweep_report=sweep)
    gate = next(
        item for item in audit["gates"] if item["gate"] == "complex_generator_readiness"
    )
    assert gate["status"] == "pass"
    assert gate["failures"] == []
