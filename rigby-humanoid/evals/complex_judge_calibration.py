from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from evals.capture import capture_result_frames
from evals.variant_review import score_variant_review
from rigby_poc.judge import VLMJudge, write_judge_record


MINIMUM_HELDOUT_AGREEMENT = 0.70
MINIMUM_ORDER_CONSISTENCY = 0.90
DEFAULT_PILOT_PAIRS = 4


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _ordered_records(answer: dict[str, Any]) -> list[dict[str, Any]]:
    """Interleave prompts so the bounded first pass samples every prompt."""

    def key(record: dict[str, Any]) -> tuple[int, int, str]:
        clip_id = str(record.get("clip_id", ""))
        try:
            prompt_index = int(clip_id.split("v", 1)[0].removeprefix("p"))
            variant_index = int(clip_id.split("v", 1)[1])
        except (ValueError, IndexError):
            return (10_000, 10_000, clip_id)
        return (variant_index, prompt_index, clip_id)

    return sorted(
        [record for record in answer.get("records", []) if isinstance(record, dict)],
        key=key,
    )


def _human_outcomes(
    submission: dict[str, Any],
    answer: dict[str, Any],
) -> dict[str, str]:
    choices = {
        str(record.get("clip_id")): record.get("choice")
        for record in submission.get("records", [])
        if isinstance(record, dict)
    }
    outcomes: dict[str, str] = {}
    for expected in answer.get("records", []):
        if not isinstance(expected, dict):
            continue
        clip_id = str(expected.get("clip_id", ""))
        choice = choices.get(clip_id)
        if choice not in {"A", "B", "tie"}:
            continue
        outcomes[clip_id] = (
            "tie"
            if choice == "tie"
            else "challenger"
            if choice == expected.get("challenger_label")
            else "baseline"
        )
    return outcomes


def _vlm_outcome(comparison: dict[str, Any]) -> str:
    calls = comparison.get("calls", [])
    if not comparison.get("order_consistent") or len(calls) != 2:
        return "unstable"
    winners = [str(call.get("mapped_winner")) for call in calls]
    if winners == ["first", "first"]:
        return "challenger"
    if winners == ["second", "second"]:
        return "baseline"
    if winners == ["tie", "tie"]:
        return "tie"
    if winners == ["neither", "neither"]:
        return "neither"
    return "unstable"


def _usage_by_model(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    for comparison in comparisons:
        for call in comparison.get("calls", []):
            for attempt in call.get("attempts", []):
                model = str(attempt.get("model", "unknown"))
                usage = attempt.get("usage", {})
                totals[model]["attempts"] += 1
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    totals[model][key] += int(usage.get(key, 0) or 0)
    return dict(totals)


def summarize_complex_judge_calibration(
    *,
    human_outcomes: dict[str, str],
    comparison_records: dict[str, dict[str, Any]],
    total_pairs: int,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for clip_id, human_outcome in human_outcomes.items():
        comparison = comparison_records.get(clip_id)
        if comparison is None:
            continue
        vlm_outcome = _vlm_outcome(comparison)
        order_consistent = bool(comparison.get("order_consistent"))
        scored.append(
            {
                "clip_id": clip_id,
                "human_outcome": human_outcome,
                "vlm_outcome": vlm_outcome,
                "agrees": human_outcome == vlm_outcome,
                "order_consistent": order_consistent,
            }
        )
    completed = len(scored)
    agreement_rate = (
        sum(record["agrees"] for record in scored) / completed if completed else 0.0
    )
    order_rate = (
        sum(record["order_consistent"] for record in scored) / completed
        if completed
        else 0.0
    )
    full_complete = completed == total_pairs
    pilot_sufficient = completed >= min(DEFAULT_PILOT_PAIRS, total_pairs)
    pilot_pass = bool(
        pilot_sufficient
        and agreement_rate >= MINIMUM_HELDOUT_AGREEMENT
        and order_rate >= MINIMUM_ORDER_CONSISTENCY
    )
    return {
        "completed_pairs": completed,
        "total_pairs": total_pairs,
        "vlm_human_agreements": sum(record["agrees"] for record in scored),
        "vlm_human_agreement_rate": agreement_rate,
        "order_consistent_pairs": sum(record["order_consistent"] for record in scored),
        "ab_order_consistency_rate": order_rate,
        "thresholds": {
            "minimum_heldout_human_agreement": MINIMUM_HELDOUT_AGREEMENT,
            "minimum_ab_order_consistency": MINIMUM_ORDER_CONSISTENCY,
            "pilot_pairs_before_expansion": min(DEFAULT_PILOT_PAIRS, total_pairs),
        },
        "pilot_pass": pilot_pass,
        "full_pass": bool(full_complete and pilot_pass),
        "recommendation": (
            "heldout_judge_gate_passed"
            if full_complete and pilot_pass
            else "stop_and_recalibrate"
            if pilot_sufficient and not pilot_pass
            else "resume_remaining_pairs"
            if pilot_pass
            else "run_bounded_pilot"
        ),
        "records": scored,
    }


def run_complex_judge_calibration(
    submission_path: Path,
    answer_key_path: Path,
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    max_new_pairs: int = DEFAULT_PILOT_PAIRS,
    judge: VLMJudge | None = None,
    capture_fn: Callable[..., Path] = capture_result_frames,
) -> Path:
    if max_new_pairs < 0:
        raise ValueError("max_new_pairs cannot be negative")
    submission = _read(submission_path)
    answer = _read(answer_key_path)
    if submission.get("manifest_sha256") != answer.get("manifest_sha256"):
        raise ValueError("submission does not match the complex pilot answer key")
    human_score = score_variant_review(submission_path, answer_key_path)
    total_pairs = len(answer.get("records", []))
    if human_score["complete"] != total_pairs:
        raise ValueError(
            f"human review is incomplete: {human_score['complete']}/{total_pairs} pairs"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / "human-score.json", human_score)
    human_outcomes = _human_outcomes(submission, answer)

    comparison_dir = output_dir / "comparisons"
    evidence_dir = output_dir / "evidence"
    comparison_records: dict[str, dict[str, Any]] = {}
    for expected in _ordered_records(answer):
        clip_id = str(expected["clip_id"])
        comparison_path = comparison_dir / f"{clip_id}.json"
        if comparison_path.is_file():
            prior = _read(comparison_path)
            if (
                prior.get("first_result_id") == expected.get("challenger_result_id")
                and prior.get("second_result_id") == expected.get("baseline_result_id")
                and len(prior.get("calls", [])) == 2
            ):
                comparison_records[clip_id] = prior

    pending = [
        record
        for record in _ordered_records(answer)
        if str(record["clip_id"]) not in comparison_records
    ]
    selected = pending[:max_new_pairs]
    if selected and judge is None:
        judge = VLMJudge()

    errors: list[dict[str, str]] = []
    for pair_offset, expected in enumerate(selected, start=1):
        clip_id = str(expected["clip_id"])
        challenger_id = str(expected["challenger_result_id"])
        baseline_id = str(expected["baseline_result_id"])
        try:
            manifests: dict[str, Path] = {}
            for result_id in (challenger_id, baseline_id):
                manifest_path = evidence_dir / result_id / "evidence-manifest.json"
                if not manifest_path.is_file():
                    manifest_path = capture_fn(
                        result_id,
                        evidence_dir / result_id,
                        base_url=base_url,
                    )
                manifests[result_id] = manifest_path
            assert judge is not None
            comparison = judge.compare(
                manifests[challenger_id],
                manifests[baseline_id],
                random_seed=81_000 + pair_offset,
                reverse_check=True,
                routing_policy="lightweight_routed",
            )
            write_judge_record(comparison, comparison_dir / f"{clip_id}.json")
            comparison_records[clip_id] = comparison
        except Exception as error:  # preserve the first failure and stop spending
            errors.append(
                {
                    "clip_id": clip_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            break

    summary = summarize_complex_judge_calibration(
        human_outcomes=human_outcomes,
        comparison_records=comparison_records,
        total_pairs=total_pairs,
    )
    comparisons = list(comparison_records.values())
    remaining = total_pairs - len(comparison_records)
    report = {
        "schema_version": "1.0",
        "kind": "complex_hangten_lightweight_vlm_calibration",
        "status": (
            "error"
            if errors
            else "complete"
            if remaining == 0
            else "pilot_complete"
            if len(comparison_records) >= min(DEFAULT_PILOT_PAIRS, total_pairs)
            else "ready"
        ),
        "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        "answer_key_sha256": hashlib.sha256(answer_key_path.read_bytes()).hexdigest(),
        "human_score": str(output_dir / "human-score.json"),
        "routing": {
            "policy": "lightweight_routed",
            "primary_model": getattr(judge, "model", None),
            "fallback_model": getattr(judge, "fallback_model", None),
            "reasoning_effort": getattr(judge, "reasoning_effort", None),
            "image_detail": getattr(judge, "image_detail", None),
            "max_image_dimension_px": getattr(judge, "max_image_dimension_px", None),
            "reverse_order_check": True,
        },
        "cost_control": {
            "max_new_pairs_this_run": max_new_pairs,
            "new_pairs_attempted": len(selected),
            "completed_pairs_total": len(comparison_records),
            "remaining_pairs": remaining,
            "minimum_additional_primary_calls": remaining * 2,
            "maximum_additional_calls_with_escalation": remaining * 4,
            "usage_by_model": _usage_by_model(comparisons),
        },
        "metrics": summary,
        "errors": errors,
    }
    report_path = output_dir / "calibration-report.json"
    _atomic_write(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Rigby's lightweight VLM on the complex human pilot"
    )
    parser.add_argument("submission", type=Path)
    parser.add_argument("answer_key", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-new-pairs", type=int, default=DEFAULT_PILOT_PAIRS)
    arguments = parser.parse_args()
    print(
        run_complex_judge_calibration(
            arguments.submission,
            arguments.answer_key,
            arguments.output_dir,
            base_url=arguments.base_url,
            max_new_pairs=arguments.max_new_pairs,
        )
    )


if __name__ == "__main__":
    main()
