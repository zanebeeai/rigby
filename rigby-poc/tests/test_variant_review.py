from __future__ import annotations

import json
from pathlib import Path

from evals.variant_review import (
    create_multi_variant_review,
    create_revision_review,
    create_variant_review,
    score_variant_review,
)

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def test_variant_review_is_blinded_and_scores_recipes(tmp_path: Path) -> None:
    trace = {
        "prompt": "Make a shaka.",
        "baseline_result_id": "000001-baseline",
        "rounds": [
            {
                "candidates": [
                    {
                        "result_id": "000001-baseline",
                        "recipe": {"name": "grounded_natural"},
                    },
                    {
                        "result_id": "000002-challenger-a",
                        "recipe": {"name": "expressive_arc"},
                        "evidence_manifest": "candidate-a/evidence-manifest.json",
                    },
                    {
                        "result_id": "000003-challenger-b",
                        "recipe": {"name": "camera_silhouette"},
                        "evidence_manifest": "candidate-b/evidence-manifest.json",
                    },
                    {
                        "result_id": "000004-near-duplicate",
                        "recipe": {"name": "near_duplicate"},
                        "rejection_stage": "perceptual_diversity_prefilter",
                    },
                ]
            }
        ],
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    manifest_path, answer_path = create_variant_review(trace_path, tmp_path / "review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 2
    assert "recipe" not in json.dumps(manifest).lower()
    submission = {
        "manifest_sha256": manifest["manifest_sha256"],
        "records": [
            {"clip_id": answer["records"][0]["clip_id"], "choice": answer["records"][0]["challenger_label"]},
            {"clip_id": answer["records"][1]["clip_id"], "choice": "tie"},
        ],
    }
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    score = score_variant_review(submission_path, answer_path)
    assert score["complete"] == 2
    assert score["challenger_preferred"] == 1
    assert score["ties"] == 1
    assert score["records"][0]["challenger_recipe"] == "expressive_arc"


def test_multi_prompt_review_combines_blinded_human_ready_traces(tmp_path: Path) -> None:
    traces = []
    for trace_index in range(2):
        prompt = f"Prompt {trace_index + 1}"
        baseline = f"00000{trace_index + 1}-baseline"
        trace = {
            "prompt": prompt,
            "baseline_result_id": baseline,
            "rounds": [
                {
                    "candidates": [
                        {"result_id": baseline, "recipe": {"name": "canonical"}},
                        {
                            "result_id": f"00001{trace_index}-challenger",
                            "recipe": {"name": "expressive_arc"},
                            "human_review_ready": True,
                        },
                    ]
                }
            ],
        }
        trace_path = tmp_path / f"trace-{trace_index}.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        traces.append(trace_path)
    manifest_path, answer_path = create_multi_variant_review(traces, tmp_path / "multi")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "blinded_multi_prompt_variant_pilot"
    assert len(manifest["records"]) == 2
    assert {record["prompt"] for record in manifest["records"]} == {"Prompt 1", "Prompt 2"}
    assert len({record["clip_id"] for record in manifest["records"]}) == 2
    assert "recipe" not in json.dumps(manifest).lower()
    assert answer["kind"] == "multi_prompt_variant_pilot_answer_key"


def test_revision_review_matches_same_recipe_and_blinds_implementation(tmp_path: Path) -> None:
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    for case_index in range(2):
        case_id = f"cx{case_index + 1:02d}"
        for root, suffix in ((before_dir, "old"), (after_dir, "new")):
            case_dir = root / case_id
            case_dir.mkdir(parents=True)
            trace = {
                "prompt": f"Prompt {case_index + 1}",
                "rounds": [
                    {
                        "candidates": [
                            {
                                "result_id": f"{case_id}-{suffix}",
                                "recipe": {"name": "canonical"},
                                "compile_success": True,
                                "structural_valid": True,
                            }
                        ]
                    }
                ],
            }
            (case_dir / "flywheel-trace.json").write_text(json.dumps(trace), encoding="utf-8")

    manifest_path, answer_path = create_revision_review(
        before_dir, after_dir, tmp_path / "revision"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "blinded_matched_revision_review"
    assert len(manifest["records"]) == 2
    assert "pronation" not in json.dumps(manifest).lower()
    assert all("-new" in record["challenger_result_id"] for record in answer["records"])
    assert answer["kind"] == "matched_revision_answer_key"
