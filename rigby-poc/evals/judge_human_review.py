from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


MINIMUM_OBVIOUS_PAIRS = 10
MINIMUM_HUMAN_AGREEMENT = 0.80
MINIMUM_BASE_PREFERENCE = 0.80
MINIMUM_ORDER_CONSISTENCY = 0.90


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _hash_payload(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _resolve_evidence_path(value: object, calibration_report: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute() and path.is_file():
        return path
    candidates = [Path.cwd() / path]
    candidates.extend(parent / path for parent in calibration_report.resolve().parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def create_judge_human_review(
    calibration_report: Path,
    output_dir: Path,
    *,
    random_seed: int = 20260808,
) -> tuple[Path, Path]:
    report = _read(calibration_report)
    base_result_id = str(report.get("base_result_id", ""))
    pairwise = report.get("pairwise_checks")
    unary = report.get("unary_corruptions")
    if not base_result_id:
        raise ValueError("calibration report has no base result")
    if not isinstance(pairwise, list) or len(pairwise) < MINIMUM_OBVIOUS_PAIRS:
        raise ValueError("judge human review requires at least ten obvious pairwise checks")
    if not isinstance(unary, list):
        raise ValueError("calibration report has no unary corruption records")

    result_by_corruption = {
        str(record.get("corruption", {}).get("id")): str(record.get("result_id", ""))
        for record in unary
        if isinstance(record, dict) and isinstance(record.get("corruption"), dict)
    }
    base_manifest = calibration_report.parent / "base-evidence" / "evidence-manifest.json"
    if not base_manifest.is_file():
        raise ValueError(f"base evidence manifest is missing: {base_manifest}")
    prompt = str(_read(base_manifest).get("prompt", "")).strip()
    if not prompt:
        raise ValueError("base evidence manifest has no prompt")

    rng = random.Random(random_seed)
    base_labels = ["A"] * (MINIMUM_OBVIOUS_PAIRS // 2)
    base_labels.extend(["B"] * (MINIMUM_OBVIOUS_PAIRS - len(base_labels)))
    rng.shuffle(base_labels)
    manifest_records: list[dict[str, Any]] = []
    answer_records: list[dict[str, Any]] = []
    for index, record in enumerate(pairwise[:MINIMUM_OBVIOUS_PAIRS], start=1):
        if not isinstance(record, dict) or "error" in record:
            raise ValueError(f"pairwise check {index} is incomplete")
        corruption = record.get("corruption")
        if not isinstance(corruption, dict) or not corruption.get("id"):
            raise ValueError(f"pairwise check {index} has no corruption identity")
        corruption_id = str(corruption["id"])
        corruption_result_id = result_by_corruption.get(corruption_id, "")
        if not corruption_result_id:
            raise ValueError(f"pairwise check {corruption_id} has no result")

        comparison_path = _resolve_evidence_path(record.get("comparison"), calibration_report)
        if not comparison_path.is_file():
            raise ValueError(f"pairwise evidence is missing: {comparison_path}")
        comparison = _read(comparison_path)
        if comparison.get("first_result_id") != base_result_id:
            raise ValueError(f"{corruption_id} does not compare the expected base first")
        if comparison.get("second_result_id") != corruption_result_id:
            raise ValueError(f"{corruption_id} does not compare the expected corruption second")
        calls = comparison.get("calls")
        if not isinstance(calls, list) or len(calls) != 2:
            raise ValueError(f"{corruption_id} lacks the two A/B-order calls")
        mapped_winners = [str(call.get("mapped_winner")) for call in calls]
        if any(winner not in {"first", "second", "tie"} for winner in mapped_winners):
            raise ValueError(f"{corruption_id} has an invalid mapped winner")
        order_consistent = bool(comparison.get("order_consistent"))
        consensus = (
            "base"
            if order_consistent and mapped_winners == ["first", "first"]
            else "corruption"
            if order_consistent and mapped_winners == ["second", "second"]
            else "tie"
            if order_consistent and mapped_winners == ["tie", "tie"]
            else "unstable"
        )

        base_label = base_labels[index - 1]
        corruption_label = "B" if base_label == "A" else "A"
        result_by_label = {
            base_label: base_result_id,
            corruption_label: corruption_result_id,
        }
        clip_id = f"j{index:02d}"
        manifest_records.append(
            {
                "clip_id": clip_id,
                "prompt": prompt,
                "candidates": {
                    label: {"result_url": f"/api/v1/results/{result_id}"}
                    for label, result_id in result_by_label.items()
                },
            }
        )
        answer_records.append(
            {
                "clip_id": clip_id,
                "corruption_id": corruption_id,
                "base_label": base_label,
                "corruption_label": corruption_label,
                "base_result_id": base_result_id,
                "corruption_result_id": corruption_result_id,
                "vlm_mapped_winners": mapped_winners,
                "vlm_order_consistent": order_consistent,
                "vlm_consensus": consensus,
            }
        )

    manifest_core = {
        "schema_version": "1.0",
        "kind": "blinded_obvious_pair_judge_calibration",
        "instructions": "Choose A, B, or Tie using the prompt and both synchronized views.",
        "records": manifest_records,
    }
    manifest = {**manifest_core, "manifest_sha256": _hash_payload(manifest_core)}
    answer_key = {
        "schema_version": "1.0",
        "kind": "obvious_pair_judge_calibration_answer_key",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_calibration_report": str(calibration_report),
        "thresholds": {
            "minimum_pairs": MINIMUM_OBVIOUS_PAIRS,
            "minimum_vlm_human_agreement": MINIMUM_HUMAN_AGREEMENT,
            "minimum_human_base_preference": MINIMUM_BASE_PREFERENCE,
            "minimum_ab_order_consistency": MINIMUM_ORDER_CONSISTENCY,
        },
        "records": answer_records,
    }
    manifest_path = output_dir / "judge-human-manifest.json"
    answer_key_path = output_dir / "judge-human-answer-key.json"
    _atomic_write(manifest_path, manifest)
    _atomic_write(answer_key_path, answer_key)
    return manifest_path, answer_key_path


def score_judge_human_review(submission_path: Path, answer_key_path: Path) -> dict[str, Any]:
    submission = _read(submission_path)
    answer_key = _read(answer_key_path)
    if submission.get("manifest_sha256") != answer_key.get("manifest_sha256"):
        raise ValueError("submission does not match the judge calibration manifest")
    submitted = {
        str(record.get("clip_id")): record
        for record in submission.get("records", [])
        if isinstance(record, dict)
    }
    complete = 0
    base_preferred = 0
    corruption_preferred = 0
    ties = 0
    agreements = 0
    order_consistent = 0
    scored: list[dict[str, Any]] = []
    for expected in answer_key.get("records", []):
        actual = submitted.get(str(expected.get("clip_id")), {})
        choice = actual.get("choice")
        if choice not in {"A", "B", "tie"}:
            continue
        complete += 1
        human_outcome = (
            "tie"
            if choice == "tie"
            else "base"
            if choice == expected["base_label"]
            else "corruption"
        )
        base_preferred += int(human_outcome == "base")
        corruption_preferred += int(human_outcome == "corruption")
        ties += int(human_outcome == "tie")
        vlm_consensus = str(expected["vlm_consensus"])
        agrees = human_outcome == vlm_consensus
        agreements += int(agrees)
        order_consistent += int(bool(expected["vlm_order_consistent"]))
        scored.append(
            {
                "clip_id": expected["clip_id"],
                "choice": choice,
                "human_outcome": human_outcome,
                "vlm_consensus": vlm_consensus,
                "vlm_order_consistent": bool(expected["vlm_order_consistent"]),
                "agrees": agrees,
            }
        )

    denominator = complete or 1
    agreement_rate = agreements / denominator if complete else 0.0
    base_preference_rate = base_preferred / denominator if complete else 0.0
    order_consistency_rate = order_consistent / denominator if complete else 0.0
    thresholds = answer_key["thresholds"]
    passed = bool(
        complete >= int(thresholds["minimum_pairs"])
        and agreement_rate >= float(thresholds["minimum_vlm_human_agreement"])
        and base_preference_rate >= float(thresholds["minimum_human_base_preference"])
        and order_consistency_rate >= float(thresholds["minimum_ab_order_consistency"])
    )
    return {
        "schema_version": "1.0",
        "manifest_sha256": answer_key["manifest_sha256"],
        "complete": complete,
        "base_preferred": base_preferred,
        "corruption_preferred": corruption_preferred,
        "ties": ties,
        "vlm_human_agreements": agreements,
        "vlm_human_agreement_rate": agreement_rate,
        "human_base_preference_rate": base_preference_rate,
        "ab_order_consistency_rate": order_consistency_rate,
        "thresholds": thresholds,
        "passed": passed,
        "records": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or score blinded human calibration of the VLM judge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("calibration_report", type=Path)
    create.add_argument("output_dir", type=Path)
    score = subparsers.add_parser("score")
    score.add_argument("submission", type=Path)
    score.add_argument("answer_key", type=Path)
    score.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "create":
        print(*create_judge_human_review(arguments.calibration_report, arguments.output_dir), sep="\n")
        return
    result = score_judge_human_review(arguments.submission, arguments.answer_key)
    _atomic_write(arguments.output, result)
    print(arguments.output)


if __name__ == "__main__":
    main()
