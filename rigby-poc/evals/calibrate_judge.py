from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.capture import capture_result_frames
from rigby_poc.judge import VLMJudge, write_judge_record


def calibration_summary(
    unary: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
) -> dict[str, Any]:
    completed_unary = [item for item in unary if "accepted" in item]
    false_accepts = sum(bool(item["accepted"]) for item in completed_unary)
    false_acceptance_rate = false_accepts / len(completed_unary) if completed_unary else None
    completed_pairs = [item for item in pairwise if "base_won" in item]
    base_wins = sum(bool(item["base_won"]) for item in completed_pairs)
    obvious_agreement = base_wins / len(completed_pairs) if completed_pairs else None
    consistent = sum(bool(item["order_consistent"]) for item in completed_pairs)
    order_consistency = consistent / len(completed_pairs) if completed_pairs else None
    completed_wrist = sum(
        item.get("corruption", {}).get("kind") == "wrist_rotation"
        for item in completed_unary
    )
    completed_wrong_joint = sum(
        item.get("corruption", {}).get("kind") == "wrong_joint_shake"
        for item in completed_unary
    )
    completed_timing = sum(
        item.get("corruption", {}).get("kind") == "timing"
        for item in completed_unary
    )
    thresholds = {
        "minimum_unary_corruptions": 28,
        "minimum_wrist_corruptions": 12,
        "minimum_wrong_joint_shake_corruptions": 4,
        "minimum_timing_corruptions": 4,
        "minimum_obvious_pairs": 10,
        "maximum_false_acceptance_rate": 0.05,
        "minimum_obvious_pairwise_agreement": 0.80,
        "minimum_ab_order_consistency": 0.90,
    }
    passed = bool(
        false_acceptance_rate is not None
        and obvious_agreement is not None
        and order_consistency is not None
        and len(completed_unary) >= thresholds["minimum_unary_corruptions"]
        and completed_wrist >= thresholds["minimum_wrist_corruptions"]
        and completed_wrong_joint >= thresholds["minimum_wrong_joint_shake_corruptions"]
        and completed_timing >= thresholds["minimum_timing_corruptions"]
        and len(completed_pairs) >= thresholds["minimum_obvious_pairs"]
        and false_acceptance_rate < thresholds["maximum_false_acceptance_rate"]
        and obvious_agreement >= thresholds["minimum_obvious_pairwise_agreement"]
        and order_consistency >= thresholds["minimum_ab_order_consistency"]
    )
    return {
        "completed_unary_corruptions": len(completed_unary),
        "completed_wrist_corruptions": completed_wrist,
        "completed_wrong_joint_shake_corruptions": completed_wrong_joint,
        "completed_timing_corruptions": completed_timing,
        "false_accepts": false_accepts,
        "false_acceptance_rate": false_acceptance_rate,
        "completed_obvious_pairs": len(completed_pairs),
        "base_wins": base_wins,
        "obvious_pairwise_agreement": obvious_agreement,
        "order_consistent_pairs": consistent,
        "ab_order_consistency": order_consistency,
        "thresholds": thresholds,
        "passed": passed,
    }


def structural_prefilter_record(record: dict[str, Any]) -> dict[str, Any]:
    """Record a zero-cost rejection already proven by deterministic gates."""
    if bool(record.get("structural_valid")):
        raise ValueError("structural prefilter only applies to invalid corruptions")
    return {
        "corruption": record["corruption"],
        "result_id": record["result_id"],
        "vlm_accepted": None,
        "structural_valid": False,
        "accepted": False,
        "rejection_source": "deterministic_structural_prefilter",
        "structural_failures": list(record.get("structural_failures", [])),
    }


def _obvious_comparisons(
    available: list[tuple[dict[str, Any], Path]],
    limit: int,
) -> list[tuple[dict[str, Any], Path]]:
    timing = sorted(
        (item for item in available if item[0]["corruption"]["kind"] == "timing"),
        key=lambda item: item[0]["corruption"]["id"],
    )
    wrong_joint = sorted(
        (item for item in available if item[0]["corruption"]["kind"] == "wrong_joint_shake"),
        key=lambda item: item[0]["corruption"]["id"],
    )
    if not timing:
        return sorted(
            available,
            key=lambda item: (
                item[0]["corruption"]["kind"] == "wrist_rotation",
                -abs(float(item[0]["corruption"].get("angle_rad", 0.0))),
            ),
        )[:limit]
    fingers = sorted(
        (
            item
            for item in available
            if item[0]["corruption"]["kind"] in {"fist_shape", "open_middle_fingers"}
        ),
        key=lambda item: (
            -abs(float(item[0]["corruption"].get("angle_rad", 0.0))),
            item[0]["corruption"]["id"],
        ),
    )
    wrists = sorted(
        (item for item in available if item[0]["corruption"]["kind"] == "wrist_rotation"),
        key=lambda item: (
            -abs(float(item[0]["corruption"].get("angle_rad", 0.0))),
            item[0]["corruption"]["id"],
        ),
    )
    selected = [*timing[:4], *wrong_joint[:2], *fingers[:2], *wrists[:2]]
    selected_ids = {item[0]["corruption"]["id"] for item in selected}
    selected.extend(
        item
        for item in [*timing[4:], *wrong_joint[2:], *fingers[2:], *wrists[2:]]
        if item[0]["corruption"]["id"] not in selected_ids
    )
    return selected[:limit]


def _ordered_unary_records(records: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Run the newly important anatomy failures before broad legacy cases."""
    priority = {
        "wrong_joint_shake": 0,
        "timing": 1,
        "wrist_rotation": 2,
        "fist_shape": 3,
        "open_middle_fingers": 3,
    }
    indexed = list(enumerate(records, start=1))
    return sorted(
        indexed,
        key=lambda item: (
            priority.get(str(item[1].get("corruption", {}).get("kind")), 99),
            str(item[1].get("corruption", {}).get("id", "")),
        ),
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    report["summary"] = calibration_summary(report["unary_corruptions"], report["pairwise_checks"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _accumulate_attempts(
    usage_by_model: dict[str, dict[str, int]],
    judgment: dict[str, Any],
) -> None:
    attempts = judgment.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        call = judgment.get("call")
        attempts = [call] if isinstance(call, dict) else []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        model = str(attempt.get("model", "unknown"))
        usage = attempt.get("usage") if isinstance(attempt.get("usage"), dict) else {}
        aggregate = usage_by_model.setdefault(
            model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        aggregate["calls"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            aggregate[key] += int(usage.get(key, 0) or 0)


def rebuild_report(suite_path: Path, output_dir: Path, *, pairwise_limit: int = 10) -> Path:
    """Rebuild the aggregate exclusively from immutable per-call artifacts."""
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    records = suite["records"]
    report_path = output_dir / "calibration-report.json"
    base_record = json.loads((output_dir / "base-judgment.json").read_text(encoding="utf-8"))
    usage_by_model: dict[str, dict[str, int]] = {}
    _accumulate_attempts(usage_by_model, base_record)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "suite_path": str(suite_path),
        "base_result_id": suite["base_result_id"],
        "model_role": "Luna unary rejection, Terra positive confirmation and blinded comparison, plus deterministic structural gates",
        "base_judgment": base_record["call"]["parsed"],
        "unary_corruptions": [],
        "pairwise_checks": [],
    }
    available: list[tuple[dict[str, Any], Path]] = []
    for index, record in enumerate(records, start=1):
        corruption = record["corruption"]
        case_dir = output_dir / "corruptions" / f"{index:02d}-{corruption['id']}"
        judgment_path = case_dir / "vlm-judgment.json"
        manifest_path = case_dir / "evidence-manifest.json"
        if not judgment_path.is_file() or not manifest_path.is_file():
            continue
        judgment = json.loads(judgment_path.read_text(encoding="utf-8"))
        _accumulate_attempts(usage_by_model, judgment)
        parsed = judgment["call"]["parsed"]
        structural_valid = bool(record.get("structural_valid"))
        report["unary_corruptions"].append(
            {
                "corruption": corruption,
                "result_id": record["result_id"],
                "evidence_manifest": str(manifest_path),
                "vlm_accepted": parsed["accept"],
                "structural_valid": structural_valid,
                "accepted": parsed["accept"] and structural_valid,
                "overall": parsed["overall"],
                "failure_tags": parsed["failure_tags"],
                "response_id": judgment["call"]["response_id"],
                "usage": judgment["call"]["usage"],
            }
        )
        available.append((record, manifest_path))

    obvious = _obvious_comparisons(available, pairwise_limit)
    for index, (record, _) in enumerate(obvious, start=1):
        corruption = record["corruption"]
        comparison_path = output_dir / "pairwise" / f"{index:02d}-{corruption['id']}.json"
        if not comparison_path.is_file():
            continue
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        for call in comparison.get("calls", []):
            if isinstance(call, dict):
                _accumulate_attempts(usage_by_model, call)
        report["pairwise_checks"].append(
            {
                "corruption": corruption,
                "comparison": str(comparison_path),
                "base_won": all(call["mapped_winner"] == "first" for call in comparison["calls"]),
                "order_consistent": comparison["order_consistent"],
                "mapped_winners": [call["mapped_winner"] for call in comparison["calls"]],
                "response_ids": [call["response_id"] for call in comparison["calls"]],
            }
        )
    report["model_usage"] = {
        "by_model": usage_by_model,
        "calls": sum(item["calls"] for item in usage_by_model.values()),
        "input_tokens": sum(item["input_tokens"] for item in usage_by_model.values()),
        "output_tokens": sum(item["output_tokens"] for item in usage_by_model.values()),
        "total_tokens": sum(item["total_tokens"] for item in usage_by_model.values()),
    }
    _write_report(report_path, report)
    return report_path


def calibrate(
    suite_path: Path,
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    pairwise_limit: int = 10,
    max_new_unary: int | None = None,
    max_new_pairwise: int | None = None,
    max_new_model_calls: int | None = None,
    pairwise_routing_policy: str = "authoritative",
) -> Path:
    if max_new_unary is not None and max_new_unary < 0:
        raise ValueError("max_new_unary cannot be negative")
    if max_new_pairwise is not None and max_new_pairwise < 0:
        raise ValueError("max_new_pairwise cannot be negative")
    if max_new_model_calls is not None and max_new_model_calls < 0:
        raise ValueError("max_new_model_calls cannot be negative")
    if pairwise_routing_policy not in {
        "authoritative",
        "lightweight_routed",
        "lightweight_only",
    }:
        raise ValueError("unsupported pairwise routing policy")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    records = suite.get("records")
    if not isinstance(records, list) or len(records) < 28:
        raise ValueError("calibration suite must contain at least 28 corruptions")
    timing_count = sum(
        isinstance(record, dict)
        and record.get("corruption", {}).get("kind") == "timing"
        for record in records
    )
    if timing_count < 4:
        raise ValueError("calibration suite must contain at least four timing corruptions")
    wrong_joint_count = sum(
        isinstance(record, dict)
        and record.get("corruption", {}).get("kind") == "wrong_joint_shake"
        for record in records
    )
    if wrong_joint_count < 4:
        raise ValueError(
            "calibration suite must contain at least four wrong-joint shake corruptions"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "calibration-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "schema_version": "1.0",
            "suite_path": str(suite_path),
            "base_result_id": suite["base_result_id"],
            "model_role": "hybrid structural prefilter plus blinded visual motion judge",
            "unary_corruptions": [],
            "pairwise_checks": [],
        }
    judge = VLMJudge(max_model_calls=max_new_model_calls)

    base_evidence_dir = output_dir / "base-evidence"
    base_manifest = base_evidence_dir / "evidence-manifest.json"
    if not base_manifest.is_file():
        print("Capturing base evidence", flush=True)
        capture_result_frames(suite["base_result_id"], base_evidence_dir, base_url=base_url)
    base_judgment_path = output_dir / "base-judgment.json"
    if base_judgment_path.is_file():
        base_record = json.loads(base_judgment_path.read_text(encoding="utf-8"))
    else:
        base_record = judge.score(base_manifest)
        write_judge_record(base_record, base_judgment_path)
    report["base_judgment"] = base_record["call"]["parsed"]
    _write_report(report_path, report)

    existing_unary = {
        str(item.get("corruption", {}).get("id")): item
        for item in report["unary_corruptions"]
        if isinstance(item, dict) and isinstance(item.get("corruption"), dict)
    }
    report["unary_corruptions"] = []
    new_unary_attempts = 0
    for index, record in _ordered_unary_records(records):
        corruption = record["corruption"]
        result_id = record["result_id"]
        case_dir = output_dir / "corruptions" / f"{index:02d}-{corruption['id']}"
        manifest = case_dir / "evidence-manifest.json"
        prior = existing_unary.get(corruption["id"])
        if prior and "error" not in prior and "accepted" in prior:
            if prior.get("rejection_source") == "deterministic_structural_prefilter":
                report["unary_corruptions"].append(structural_prefilter_record(record))
                continue
            if not manifest.is_file():
                continue
            vlm_accepted = bool(prior.get("vlm_accepted", prior.get("accepted")))
            prior["vlm_accepted"] = vlm_accepted
            prior["structural_valid"] = bool(record.get("structural_valid"))
            prior["accepted"] = vlm_accepted and bool(record.get("structural_valid"))
            report["unary_corruptions"].append(prior)
            continue
        if not bool(record.get("structural_valid")):
            report["unary_corruptions"].append(structural_prefilter_record(record))
            continue
        if max_new_unary is not None and new_unary_attempts >= max_new_unary:
            continue
        if judge.remaining_model_calls == 0:
            continue
        new_unary_attempts += 1
        try:
            print(f"Unary corruption {index:02d}/{len(records)}: {corruption['id']}", flush=True)
            if not manifest.is_file():
                capture_result_frames(result_id, case_dir, base_url=base_url)
            judgment = judge.score(manifest)
            write_judge_record(judgment, case_dir / "vlm-judgment.json")
            parsed = judgment["call"]["parsed"]
            report["unary_corruptions"].append(
                {
                    "corruption": corruption,
                    "result_id": result_id,
                    "evidence_manifest": str(manifest),
                    "vlm_accepted": parsed["accept"],
                    "structural_valid": bool(record.get("structural_valid")),
                    "accepted": parsed["accept"] and bool(record.get("structural_valid")),
                    "overall": parsed["overall"],
                    "failure_tags": parsed["failure_tags"],
                    "response_id": judgment["call"]["response_id"],
                    "usage": judgment["call"]["usage"],
                }
            )
        except Exception as error:  # preserve the partial calibration trace
            report["unary_corruptions"].append(
                {"corruption": corruption, "result_id": result_id, "error": str(error)}
            )
        _write_report(report_path, report)

    # Pairwise calibration stratifies timing, the original wrong-joint shake
    # failure, broken fingers, and obvious extreme wrist bends. Subtle constant
    # twists remain the deterministic structural gate's responsibility.
    all_manifests = [
        (
            record,
            output_dir
            / "corruptions"
            / f"{index:02d}-{record['corruption']['id']}"
            / "evidence-manifest.json",
        )
        for index, record in enumerate(records, start=1)
    ]
    obvious_manifests = _obvious_comparisons(all_manifests, pairwise_limit)
    existing_pairwise = {
        str(item.get("corruption", {}).get("id")): item
        for item in report["pairwise_checks"]
        if isinstance(item, dict) and isinstance(item.get("corruption"), dict)
    }
    report["pairwise_checks"] = []
    new_pairwise_attempts = 0
    for index, (record, manifest) in enumerate(obvious_manifests, start=1):
        corruption = record["corruption"]
        prior_pair = existing_pairwise.get(corruption["id"])
        if prior_pair and "error" not in prior_pair:
            report["pairwise_checks"].append(prior_pair)
            continue
        if max_new_pairwise is not None and new_pairwise_attempts >= max_new_pairwise:
            continue
        if judge.remaining_model_calls == 0:
            continue
        new_pairwise_attempts += 1
        try:
            print(f"Pairwise reverse-order check {index:02d}/{pairwise_limit}: {corruption['id']}", flush=True)
            if not manifest.is_file():
                case_dir = manifest.parent
                capture_result_frames(record["result_id"], case_dir, base_url=base_url)
            comparison = judge.compare(
                base_manifest,
                manifest,
                random_seed=10_000 + index,
                reverse_check=True,
                routing_policy=pairwise_routing_policy,
            )
            comparison_path = output_dir / "pairwise" / f"{index:02d}-{corruption['id']}.json"
            write_judge_record(comparison, comparison_path)
            report["pairwise_checks"].append(
                {
                    "corruption": corruption,
                    "comparison": str(comparison_path),
                    "base_won": all(call["mapped_winner"] == "first" for call in comparison["calls"]),
                    "order_consistent": comparison["order_consistent"],
                    "mapped_winners": [call["mapped_winner"] for call in comparison["calls"]],
                    "response_ids": [call["response_id"] for call in comparison["calls"]],
                }
            )
        except Exception as error:
            report["pairwise_checks"].append({"corruption": corruption, "error": str(error)})
        _write_report(report_path, report)

    completed_unary = sum("accepted" in item for item in report["unary_corruptions"])
    completed_pairwise = sum(
        "base_won" in item for item in report["pairwise_checks"]
    )
    report["cost_control"] = {
        "max_new_unary_this_run": max_new_unary,
        "max_new_pairwise_this_run": max_new_pairwise,
        "max_new_model_calls_this_run": max_new_model_calls,
        "pairwise_routing_policy_this_run": pairwise_routing_policy,
        "model_calls_made_this_run": judge.model_calls_made,
        "remaining_model_calls_this_run": judge.remaining_model_calls,
        "new_unary_attempts": new_unary_attempts,
        "new_pairwise_attempts": new_pairwise_attempts,
        "completed_unary_total": completed_unary,
        "completed_pairwise_total": completed_pairwise,
        "remaining_unary": max(0, len(records) - completed_unary),
        "remaining_pairwise": max(0, pairwise_limit - completed_pairwise),
        "maximum_additional_calls_with_confirmation": (
            max(0, len(records) - completed_unary) * 2
            + max(0, pairwise_limit - completed_pairwise) * 4
        ),
    }
    _write_report(report_path, report)

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the Rigby visual motion judge")
    parser.add_argument("suite", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--pairwise-limit", type=int, default=10)
    parser.add_argument("--max-new-unary", type=int)
    parser.add_argument("--max-new-pairwise", type=int)
    parser.add_argument("--max-new-model-calls", type=int)
    parser.add_argument(
        "--pairwise-routing-policy",
        choices=("authoritative", "lightweight_routed", "lightweight_only"),
        default="authoritative",
    )
    parser.add_argument("--rebuild-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.rebuild_only:
        print(rebuild_report(arguments.suite, arguments.output_dir, pairwise_limit=arguments.pairwise_limit))
        return
    print(
        calibrate(
            arguments.suite,
            arguments.output_dir,
            base_url=arguments.base_url,
            pairwise_limit=arguments.pairwise_limit,
            max_new_unary=arguments.max_new_unary,
            max_new_pairwise=arguments.max_new_pairwise,
            max_new_model_calls=arguments.max_new_model_calls,
            pairwise_routing_policy=arguments.pairwise_routing_policy,
        )
    )


if __name__ == "__main__":
    main()
