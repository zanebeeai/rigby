from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.flywheel import run_best_of_five
from evals.heldout import heldout_complex_hangten_uplift_cases


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_heldout_batch(
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    limit: int = 30,
) -> Path:
    cases = heldout_complex_hangten_uplift_cases()[:limit]
    report_path = output_dir / "batch-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "schema_version": "1.0",
            "fixture": "heldout_hangten_v4_anatomical_shake_precommitted_perceptual_diversity",
            "selection_mode": "five_way",
            "planner_provider": "offline deterministic profile parser",
            "records": [],
        }
    completed = {record["case_id"]: record for record in report["records"]}
    report["records"] = []
    for index, case in enumerate(cases, start=1):
        run_dir = output_dir / "runs" / case["id"]
        trace_path = run_dir / "flywheel-trace.json"
        prior = completed.get(case["id"])
        if prior and trace_path.is_file() and prior.get("status") == "winner_selected":
            report["records"].append(prior)
            continue
        print(f"Held-out flywheel {index:02d}/{len(cases)}: {case['id']}", flush=True)
        quota_exhausted = False
        try:
            run_best_of_five(
                case["prompt"],
                run_dir,
                provider="offline",
                base_url=base_url,
                max_rounds=2,
                selection_mode="five_way",
            )
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            report["records"].append(
                {
                    "case_id": case["id"],
                    "prompt": case["prompt"],
                    "hand": case["hand"],
                    "motion_profile": case["motion_profile"],
                    "expected_cycles": case["expected_cycles"],
                    "shake_wording": case["shake_wording"],
                    "trace": str(trace_path),
                    "status": trace["status"],
                    "baseline_result_id": trace.get("baseline_result_id"),
                    "winner_result_id": trace.get("winner_result_id"),
                    "winner_recipe": trace.get("winner", {}).get("recipe", {}).get("name"),
                    "round_count": len(trace.get("rounds", [])),
                    "candidate_count": sum(
                        len(round_record.get("candidates", []))
                        for round_record in trace.get("rounds", [])
                    ),
                }
            )
        except Exception as error:
            error_message = str(error)
            quota_exhausted = "credit_balance_exhausted" in error_message
            report["records"].append(
                {
                    "case_id": case["id"],
                    "prompt": case["prompt"],
                    "hand": case["hand"],
                    "motion_profile": case["motion_profile"],
                    "expected_cycles": case["expected_cycles"],
                    "shake_wording": case["shake_wording"],
                    "trace": str(trace_path),
                    "status": "error",
                    "error": error_message,
                }
            )
        report["summary"] = {
            "requested": len(cases),
            "completed": len(report["records"]),
            "winners_selected": sum(
                record.get("status") == "winner_selected" for record in report["records"]
            ),
            "errors": sum(record.get("status") == "error" for record in report["records"]),
        }
        _write(report_path, report)
        if quota_exhausted:
            raise RuntimeError("OpenAI quota exhausted; held-out batch stopped for safe resume")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 30-prompt held-out best-of-five batch")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=30)
    arguments = parser.parse_args()
    print(
        run_heldout_batch(
            arguments.output_dir,
            base_url=arguments.base_url,
            limit=arguments.limit,
        )
    )


if __name__ == "__main__":
    main()
