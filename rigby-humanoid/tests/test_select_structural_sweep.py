from __future__ import annotations

import json
from pathlib import Path

from evals.goal_audit import _candidate_trace_failures
from evals.select_structural_sweep import FIXTURE, run_sweep_selection

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _fixture(tmp_path: Path) -> Path:
    cases = []
    for case_index in range(30):
        case_id = f"hx{case_index + 1:02d}"
        trace_path = tmp_path / "sources" / case_id / "flywheel-trace.json"
        trace_path.parent.mkdir(parents=True)
        candidates = [
            {
                "result_id": f"{case_id}-candidate-{candidate_index}",
                "recipe": {"name": f"recipe-{candidate_index}"},
                "motion_sha256": f"{case_id}-hash-{candidate_index}",
                "compile_success": True,
                "structural_valid": True,
                "structural_failures": [],
                "quality_metrics": {"structural_valid": True},
                "perceptual_descriptor": {"samples": [{}]},
                "perceptual_diversity": [
                    {
                        "against_result_id": f"{case_id}-candidate-{prior_index}",
                        "perceptually_distinct": True,
                    }
                    for prior_index in range(candidate_index)
                ],
                "human_review_ready": True,
            }
            for candidate_index in range(5)
        ]
        trace_path.write_text(
            json.dumps(
                {
                    "status": "awaiting_human_selection",
                    "selection_mode": "human_pilot",
                    "baseline_result_id": candidates[0]["result_id"],
                    "rounds": [{"candidates": candidates}],
                }
            ),
            encoding="utf-8",
        )
        cases.append(
            {
                "id": case_id,
                "prompt": f"Prompt {case_index + 1}",
                "hand": "right" if case_index % 2 == 0 else "left",
                "motion_profile": {"style": "neutral"},
                "expected_cycles": float(2 + case_index % 3),
                "shake_wording": ("natural_shake", "palm_rotation", "anatomical")[
                    case_index % 3
                ],
                "trace": str(trace_path),
                "status": "pass",
                "failures": [],
            }
        )
    sweep = tmp_path / "structural-sweep-report.json"
    sweep.write_text(
        json.dumps(
            {
                "fixture": FIXTURE,
                "status": "pass",
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return sweep


def test_zero_prompt_preflight_constructs_no_judge_and_captures_nothing(
    tmp_path: Path,
) -> None:
    sweep = _fixture(tmp_path)

    def forbidden_capture(*_: object, **__: object) -> Path:
        raise AssertionError("zero-call preflight attempted evidence capture")

    report_path = run_sweep_selection(
        sweep,
        tmp_path / "selection",
        max_new_prompts=0,
        capture_fn=forbidden_capture,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["summary"] == {
        "requested": 30,
        "winners_selected": 0,
        "pending": 30,
        "errors": 0,
        "automatic_baseline_ties": 0,
    }
    assert report["cost_control"]["new_prompt_attempts"] == 0
    assert report["cost_control"]["max_new_model_calls_this_run"] == 0
    assert report["cost_control"]["model_calls_dispatched_this_run"] == 0
    assert report["cost_control"]["maximum_additional_five_way_calls"] == 30
    assert not (tmp_path / "selection" / "human-review").exists()


def test_bounded_selector_reuses_candidates_and_creates_review_after_thirty(
    tmp_path: Path,
) -> None:
    sweep = _fixture(tmp_path)
    captured: list[str] = []

    def fake_capture(result_id: str, output_dir: Path, **_: object) -> Path:
        captured.append(result_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "evidence-manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return manifest

    class FakeJudge:
        def rank_five(self, manifests: list[Path], *, random_seed: int) -> dict:
            assert len(manifests) == 5
            assert random_seed >= 90_001
            return {
                "mapped_winner_index": 2,
                "mapped_scores": {
                    index: {"accept": True, "overall": 4}
                    for index in range(5)
                },
                "call": {"response_id": f"response-{random_seed}"},
                "attempts": [
                    {
                        "model": "test-model",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "total_tokens": 12,
                        },
                    }
                ],
            }

    report_path = run_sweep_selection(
        sweep,
        tmp_path / "selection",
        max_new_prompts=30,
        judge=FakeJudge(),  # type: ignore[arg-type]
        capture_fn=fake_capture,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["summary"]["winners_selected"] == 30
    assert report["summary"]["automatic_baseline_ties"] == 0
    assert all(
        record["winner_result_id"].endswith("candidate-2")
        for record in report["records"]
    )
    first_trace = json.loads(Path(report["records"][0]["trace"]).read_text(encoding="utf-8"))
    assert first_trace["status"] == "winner_selected"
    assert first_trace["source_no_model_trace_sha256"]
    assert first_trace["winner"]["result_id"].endswith("candidate-2")
    traced_candidates = first_trace["rounds"][0]["candidates"]
    assert all(candidate["evidence_manifest"] for candidate in traced_candidates)
    assert all(candidate["judgment"]["accept"] for candidate in traced_candidates)
    failures, hashes = _candidate_trace_failures("hx01", first_trace)
    assert failures == []
    assert len(hashes) == 5
    assert len(captured) == 150
    assert report["cost_control"]["usage_this_run"] == {
        "attempts": 30,
        "input_tokens": 300,
        "output_tokens": 60,
        "total_tokens": 360,
    }
    manifest = json.loads(
        (tmp_path / "selection" / "human-review" / "comparison-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(manifest["records"]) == 30

    def forbidden_capture(*_: object, **__: object) -> Path:
        raise AssertionError("completed selection did not resume immutably")

    original_hash = report_path.read_bytes()
    resumed = run_sweep_selection(
        sweep,
        tmp_path / "selection",
        max_new_prompts=0,
        capture_fn=forbidden_capture,
    )
    assert resumed.read_bytes() == original_hash


def test_model_call_ceiling_overrides_a_larger_prompt_limit(tmp_path: Path) -> None:
    sweep = _fixture(tmp_path)
    captured: list[str] = []

    def fake_capture(result_id: str, output_dir: Path, **_: object) -> Path:
        captured.append(result_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = output_dir / "evidence-manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        return manifest

    class FakeJudge:
        def rank_five(self, manifests: list[Path], *, random_seed: int) -> dict:
            return {
                "mapped_winner_index": 1,
                "mapped_scores": {
                    index: {"accept": True, "overall": 4}
                    for index in range(5)
                },
                "call": {"response_id": f"response-{random_seed}"},
                "attempts": [
                    {
                        "model": "test-model",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "total_tokens": 12,
                        },
                    }
                ],
            }

    report_path = run_sweep_selection(
        sweep,
        tmp_path / "selection",
        max_new_prompts=30,
        max_new_model_calls=1,
        judge=FakeJudge(),  # type: ignore[arg-type]
        capture_fn=fake_capture,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["winners_selected"] == 1
    assert report["summary"]["pending"] == 29
    assert len(captured) == 5
    assert report["cost_control"]["model_calls_dispatched_this_run"] == 1
    assert report["cost_control"]["remaining_model_calls_this_run"] == 0
