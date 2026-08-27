from __future__ import annotations

import json
from pathlib import Path

from evals.preference_review import create_preference_review, score_preference_review

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


def _write_batch(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "case_id": f"h{index:02d}",
                        "prompt": f"Prompt {index}",
                        "status": "winner_selected",
                        "baseline_result_id": f"{index:06d}-motion-a",
                        "winner_result_id": f"{index + 100:06d}-motion-b",
                    }
                    for index in range(1, 31)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_preference_manifest_is_blinded_and_scores_thirty_choices(tmp_path: Path) -> None:
    batch = tmp_path / "batch.json"
    _write_batch(batch)
    manifest_path, key_path = create_preference_review(batch, tmp_path / "review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 30
    assert "winner" not in json.dumps(manifest).lower()
    assert {record["winner_label"] for record in key["records"]} == {"A", "B"}

    submission = {
        "schema_version": "1.0",
        "manifest_sha256": manifest["manifest_sha256"],
        "records": [
            {"clip_id": record["clip_id"], "choice": record["winner_label"]}
            for record in key["records"]
        ],
    }
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    result = score_preference_review(submission_path, key_path)
    assert result["complete"] == 30
    assert result["winner_preference_rate"] == 1.0
    assert result["metrics"]["completed_pairs"] == 30
    assert result["metrics"]["total_pairs"] == 30
    assert result["metrics"]["vlm_human_agreement_rate"] == 1.0
    assert result["passed"] is True


def test_baseline_selected_case_counts_as_automatic_tie(tmp_path: Path) -> None:
    batch = tmp_path / "batch.json"
    _write_batch(batch)
    value = json.loads(batch.read_text(encoding="utf-8"))
    value["records"][7]["winner_result_id"] = value["records"][7]["baseline_result_id"]
    batch.write_text(json.dumps(value), encoding="utf-8")
    manifest_path, key_path = create_preference_review(batch, tmp_path / "review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 29
    automatic = [record for record in key["records"] if "automatic_outcome" in record]
    assert len(automatic) == 1
    submission = {
        "manifest_sha256": manifest["manifest_sha256"],
        "records": [
            {"clip_id": record["clip_id"], "choice": record["winner_label"]}
            for record in key["records"]
            if "winner_label" in record
        ],
    }
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    result = score_preference_review(submission_path, key_path)
    assert result["complete"] == 30
    assert result["winner_preferred"] == 29
    assert result["ties"] == 1
    assert result["winner_preference_rate"] == 29 / 30
    assert result["metrics"]["completed_pairs"] == 29
    assert result["metrics"]["total_pairs"] == 29
    assert result["metrics"]["vlm_human_agreement_rate"] == 1.0
    assert result["metrics"]["automatic_baseline_ties_excluded"] == 1
    assert "single-sample baseline" in result["records"][7]["reason"]
