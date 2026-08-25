from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.judge_human_review import create_judge_human_review, score_judge_human_review

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


def _write_fixture(root: Path, *, unstable: int = 0) -> Path:
    run = root / "calibration" / "run-01"
    (run / "base-evidence").mkdir(parents=True)
    (run / "pairwise").mkdir()
    base = "000001-motion"
    (run / "base-evidence" / "evidence-manifest.json").write_text(
        json.dumps({"prompt": "Make a readable right-handed shaka."}),
        encoding="utf-8",
    )
    unary = []
    pairwise = []
    for index in range(1, 11):
        corruption_id = f"broken-{index}"
        corrupt = f"{index + 1:06d}-motion"
        comparison = run / "pairwise" / f"{index:02d}-{corruption_id}.json"
        is_unstable = index <= unstable
        calls = [
            {"mapped_winner": "first"},
            {"mapped_winner": "second" if is_unstable else "first"},
        ]
        comparison.write_text(
            json.dumps(
                {
                    "first_result_id": base,
                    "second_result_id": corrupt,
                    "calls": calls,
                    "order_consistent": not is_unstable,
                }
            ),
            encoding="utf-8",
        )
        unary.append({"corruption": {"id": corruption_id}, "result_id": corrupt})
        pairwise.append(
            {
                "corruption": {"id": corruption_id},
                "comparison": str(comparison),
            }
        )
    report = run / "calibration-report.json"
    report.write_text(
        json.dumps(
            {
                "base_result_id": base,
                "unary_corruptions": unary,
                "pairwise_checks": pairwise,
            }
        ),
        encoding="utf-8",
    )
    return report


def _base_submission(manifest: dict, key: dict) -> dict:
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "records": [
            {"clip_id": record["clip_id"], "choice": record["base_label"]}
            for record in key["records"]
        ],
    }


def test_obvious_pair_review_is_blinded_and_scores_human_agreement(tmp_path: Path) -> None:
    report = _write_fixture(tmp_path)
    manifest_path, key_path = create_judge_human_review(report, tmp_path / "review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 10
    serialized_manifest = json.dumps(manifest).lower()
    assert "corruption" not in serialized_manifest
    assert "broken" not in serialized_manifest
    assert [record["base_label"] for record in key["records"]].count("A") == 5
    assert [record["base_label"] for record in key["records"]].count("B") == 5

    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps(_base_submission(manifest, key)), encoding="utf-8")
    score = score_judge_human_review(submission_path, key_path)
    assert score["complete"] == 10
    assert score["vlm_human_agreement_rate"] == 1.0
    assert score["human_base_preference_rate"] == 1.0
    assert score["ab_order_consistency_rate"] == 1.0
    assert score["passed"] is True


def test_obvious_pair_review_fails_order_consistency_and_rejects_wrong_manifest(tmp_path: Path) -> None:
    report = _write_fixture(tmp_path, unstable=2)
    manifest_path, key_path = create_judge_human_review(report, tmp_path / "review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    submission = _base_submission(manifest, key)
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    score = score_judge_human_review(submission_path, key_path)
    assert score["vlm_human_agreement_rate"] == 0.8
    assert score["ab_order_consistency_rate"] == 0.8
    assert score["passed"] is False

    submission["manifest_sha256"] = "wrong"
    submission_path.write_text(json.dumps(submission), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        score_judge_human_review(submission_path, key_path)
