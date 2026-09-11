from __future__ import annotations

import json
from pathlib import Path

from evals.complex_judge_calibration import (
    _ordered_records,
    run_complex_judge_calibration,
    summarize_complex_judge_calibration,
)

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


def _comparison(winner: str, *, consistent: bool = True) -> dict:
    calls = [
        {"mapped_winner": winner, "attempts": []},
        {"mapped_winner": winner if consistent else "second", "attempts": []},
    ]
    return {"order_consistent": consistent, "calls": calls}


def test_calibration_summary_applies_heldout_and_order_gates() -> None:
    human = {
        "p01v01": "challenger",
        "p02v01": "baseline",
        "p03v01": "tie",
        "p04v01": "challenger",
    }
    comparisons = {
        "p01v01": _comparison("first"),
        "p02v01": _comparison("second"),
        "p03v01": _comparison("tie"),
        "p04v01": _comparison("second"),
    }
    summary = summarize_complex_judge_calibration(
        human_outcomes=human,
        comparison_records=comparisons,
        total_pairs=8,
    )
    assert summary["vlm_human_agreement_rate"] == 0.75
    assert summary["ab_order_consistency_rate"] == 1.0
    assert summary["pilot_pass"] is True
    assert summary["full_pass"] is False
    assert summary["recommendation"] == "resume_remaining_pairs"

    comparisons["p04v01"] = _comparison("first", consistent=False)
    failed = summarize_complex_judge_calibration(
        human_outcomes=human,
        comparison_records=comparisons,
        total_pairs=8,
    )
    assert failed["ab_order_consistency_rate"] == 0.75
    assert failed["pilot_pass"] is False
    assert failed["recommendation"] == "stop_and_recalibrate"


def test_pair_order_interleaves_prompts_before_second_variant() -> None:
    answer = {
        "records": [
            {"clip_id": "p02v02"},
            {"clip_id": "p01v02"},
            {"clip_id": "p02v01"},
            {"clip_id": "p01v01"},
        ]
    }
    assert [item["clip_id"] for item in _ordered_records(answer)] == [
        "p01v01",
        "p02v01",
        "p01v02",
        "p02v02",
    ]


def test_zero_pair_preflight_scores_review_without_model_or_capture(
    tmp_path: Path,
) -> None:
    records = []
    submissions = []
    for index in range(1, 5):
        clip_id = f"p{index:02d}v01"
        records.append(
            {
                "clip_id": clip_id,
                "challenger_label": "A",
                "baseline_label": "B",
                "challenger_result_id": f"challenger-{index}",
                "baseline_result_id": f"baseline-{index}",
                "challenger_recipe": "expressive_arc",
                "baseline_recipe": "canonical",
            }
        )
        submissions.append({"clip_id": clip_id, "choice": "A"})
    answer = {
        "manifest_sha256": "manifest-hash",
        "records": records,
    }
    submission = {
        "manifest_sha256": "manifest-hash",
        "records": submissions,
    }
    answer_path = tmp_path / "answer.json"
    submission_path = tmp_path / "submission.json"
    answer_path.write_text(json.dumps(answer), encoding="utf-8")
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    report_path = run_complex_judge_calibration(
        submission_path,
        answer_path,
        tmp_path / "calibration",
        max_new_pairs=0,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["cost_control"]["completed_pairs_total"] == 0
    assert report["cost_control"]["remaining_pairs"] == 4
    assert report["cost_control"]["minimum_additional_primary_calls"] == 8
    assert report["cost_control"]["maximum_additional_calls_with_escalation"] == 16
    assert report["routing"]["primary_model"] is None


def test_bounded_calibration_records_usage_and_can_pass_four_pair_pilot(
    tmp_path: Path,
) -> None:
    records = []
    submissions = []
    for index in range(1, 5):
        clip_id = f"p{index:02d}v01"
        records.append(
            {
                "clip_id": clip_id,
                "challenger_label": "A",
                "baseline_label": "B",
                "challenger_result_id": f"challenger-{index}",
                "baseline_result_id": f"baseline-{index}",
                "challenger_recipe": "expressive_arc",
                "baseline_recipe": "canonical",
            }
        )
        submissions.append({"clip_id": clip_id, "choice": "A"})
    answer_path = tmp_path / "answer.json"
    submission_path = tmp_path / "submission.json"
    answer_path.write_text(
        json.dumps({"manifest_sha256": "hash", "records": records}),
        encoding="utf-8",
    )
    submission_path.write_text(
        json.dumps({"manifest_sha256": "hash", "records": submissions}),
        encoding="utf-8",
    )

    captured: list[str] = []

    def fake_capture(result_id: str, output_dir: Path, **_: object) -> Path:
        captured.append(result_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "evidence-manifest.json"
        path.write_text("{}", encoding="utf-8")
        return path

    class FakeJudge:
        model = "gpt-5.6-luna"
        fallback_model = "gpt-5.6-terra"
        reasoning_effort = "low"
        image_detail = "high"
        max_image_dimension_px = 960

        def compare(self, first: Path, second: Path, **kwargs: object) -> dict:
            assert kwargs["reverse_check"] is True
            assert kwargs["routing_policy"] == "lightweight_routed"
            return {
                "first_result_id": first.parent.name,
                "second_result_id": second.parent.name,
                "order_consistent": True,
                "calls": [
                    {
                        "mapped_winner": "first",
                        "attempts": [
                            {
                                "model": "gpt-5.6-luna",
                                "usage": {
                                    "input_tokens": 10,
                                    "output_tokens": 2,
                                    "total_tokens": 12,
                                },
                            }
                        ],
                    },
                    {
                        "mapped_winner": "first",
                        "attempts": [
                            {
                                "model": "gpt-5.6-luna",
                                "usage": {
                                    "input_tokens": 10,
                                    "output_tokens": 2,
                                    "total_tokens": 12,
                                },
                            }
                        ],
                    },
                ],
            }

    report_path = run_complex_judge_calibration(
        submission_path,
        answer_path,
        tmp_path / "calibration",
        max_new_pairs=4,
        judge=FakeJudge(),  # type: ignore[arg-type]
        capture_fn=fake_capture,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["metrics"]["full_pass"] is True
    assert report["metrics"]["vlm_human_agreement_rate"] == 1.0
    assert report["cost_control"]["usage_by_model"]["gpt-5.6-luna"] == {
        "attempts": 8,
        "input_tokens": 80,
        "output_tokens": 16,
        "total_tokens": 96,
    }
    assert len(captured) == 8
