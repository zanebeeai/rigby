from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_preference_review(batch_report: Path, output_dir: Path) -> tuple[Path, Path]:
    batch = json.loads(batch_report.read_text(encoding="utf-8"))
    records = batch.get("records")
    if not isinstance(records, list) or len(records) < 30:
        raise ValueError("human preference review requires at least 30 held-out records")
    complete = [record for record in records if record.get("status") == "winner_selected"]
    if len(complete) < 30:
        raise ValueError(f"only {len(complete)}/30 held-out records have winners")
    rng = random.Random(20260807)
    manifest_records: list[dict[str, Any]] = []
    answer_records: list[dict[str, Any]] = []
    for index, record in enumerate(complete[:30], start=1):
        baseline = record["baseline_result_id"]
        winner = record["winner_result_id"]
        if not baseline or not winner:
            raise ValueError(f"{record.get('case_id')} lacks a baseline or winner")
        if baseline == winner:
            answer_records.append(
                {
                    "clip_id": f"automatic-{record['case_id']}",
                    "case_id": record["case_id"],
                    "winner_result_id": winner,
                    "baseline_result_id": baseline,
                    "automatic_outcome": "tie",
                    "reason": "best-of-five selected the precommitted single-sample baseline itself",
                }
            )
            continue
        winner_label = rng.choice(("A", "B"))
        baseline_label = "B" if winner_label == "A" else "A"
        by_label = {winner_label: winner, baseline_label: baseline}
        opaque_id = f"p{index:02d}"
        manifest_records.append(
            {
                "clip_id": opaque_id,
                "prompt": record["prompt"],
                "candidates": {
                    label: {
                        "result_url": f"/api/v1/results/{result_id}",
                    }
                    for label, result_id in by_label.items()
                },
            }
        )
        answer_records.append(
            {
                "clip_id": opaque_id,
                "case_id": record["case_id"],
                "winner_label": winner_label,
                "baseline_label": baseline_label,
                "winner_result_id": winner,
                "baseline_result_id": baseline,
            }
        )
    manifest_core = {
        "schema_version": "1.0",
        "kind": "blinded_pairwise_preference",
        "source_batch_sha256": _hash(batch_report),
        "instructions": "Choose A, B, or Tie using the prompt and both synchronized views.",
        "records": manifest_records,
    }
    manifest_bytes = json.dumps(manifest_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {
        **manifest_core,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    answer_key = {
        "schema_version": "1.0",
        "kind": "preference_answer_key",
        "manifest_sha256": manifest["manifest_sha256"],
        "threshold": {"minimum_winner_preference_rate": 0.70, "minimum_prompts": 30},
        "records": answer_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "comparison-manifest.json"
    key_path = output_dir / "comparison-answer-key.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    key_path.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path, key_path


def score_preference_review(submission_path: Path, answer_key_path: Path) -> dict[str, Any]:
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    if submission.get("manifest_sha256") != answer_key.get("manifest_sha256"):
        raise ValueError("submission does not match the comparison manifest")
    answers = {
        str(record.get("clip_id")): record
        for record in submission.get("records", [])
        if isinstance(record, dict)
    }
    winner_preferred = 0
    baseline_preferred = 0
    ties = 0
    complete = 0
    human_compared = 0
    judge_human_agreements = 0
    scored: list[dict[str, Any]] = []
    for expected in answer_key["records"]:
        if expected.get("automatic_outcome") == "tie":
            complete += 1
            ties += 1
            scored.append(
                {
                    "clip_id": expected["clip_id"],
                    "choice": "automatic",
                    "outcome": "tie",
                    "reason": expected.get("reason"),
                }
            )
            continue
        actual = answers.get(expected["clip_id"], {})
        choice = actual.get("choice")
        if choice not in {"A", "B", "tie"}:
            continue
        complete += 1
        human_compared += 1
        outcome = (
            "tie"
            if choice == "tie"
            else "winner"
            if choice == expected["winner_label"]
            else "baseline"
        )
        winner_preferred += int(outcome == "winner")
        judge_human_agreements += int(outcome == "winner")
        baseline_preferred += int(outcome == "baseline")
        ties += int(outcome == "tie")
        scored.append({"clip_id": expected["clip_id"], "choice": choice, "outcome": outcome})
    preference_rate = winner_preferred / complete if complete else 0.0
    heldout_total = sum(
        expected.get("automatic_outcome") != "tie"
        for expected in answer_key["records"]
    )
    heldout_agreement_rate = (
        judge_human_agreements / human_compared if human_compared else 0.0
    )
    threshold = answer_key["threshold"]
    passed = bool(
        complete >= int(threshold["minimum_prompts"])
        and preference_rate >= float(threshold["minimum_winner_preference_rate"])
    )
    return {
        "schema_version": "1.0",
        "manifest_sha256": answer_key["manifest_sha256"],
        "complete": complete,
        "winner_preferred": winner_preferred,
        "baseline_preferred": baseline_preferred,
        "ties": ties,
        "winner_preference_rate": preference_rate,
        "metrics": {
            "completed_pairs": human_compared,
            "total_pairs": heldout_total,
            "vlm_human_agreements": judge_human_agreements,
            "vlm_human_agreement_rate": heldout_agreement_rate,
            "automatic_baseline_ties_excluded": complete - human_compared,
        },
        "threshold": threshold,
        "passed": passed,
        "records": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or score a blinded baseline-vs-winner review")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("batch_report", type=Path)
    create.add_argument("output_dir", type=Path)
    score = subparsers.add_parser("score")
    score.add_argument("submission", type=Path)
    score.add_argument("answer_key", type=Path)
    score.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "create":
        print(*create_preference_review(arguments.batch_report, arguments.output_dir), sep="\n")
    else:
        result = score_preference_review(arguments.submission, arguments.answer_key)
        arguments.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(arguments.output)


if __name__ == "__main__":
    main()
