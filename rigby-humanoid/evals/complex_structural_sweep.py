from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.complex_pilot import _case_report, _existing_trace
from evals.flywheel import run_best_of_five
from evals.heldout import heldout_complex_hangten_uplift_cases


FIXTURE = "heldout_hangten_v4_anatomical_shake_precommitted_perceptual_diversity"


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_complex_structural_sweep(
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    limit: int = 30,
) -> Path:
    """Fill five locally valid/diverse candidates per complex held-out prompt.

    ``human_pilot`` mode is intentional: it constructs no VLM client, performs
    no evidence capture, and makes no API call. The resulting traces are a
    prerequisite audit, not evidence of model or human preference.
    """
    cases = heldout_complex_hangten_uplift_cases()[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    case_reports: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_dir = output_dir / "runs" / str(case["id"])
        trace_path = case_dir / "flywheel-trace.json"
        if not _existing_trace(trace_path, str(case["prompt"])):
            print(f"Complex structural sweep {index:02d}/{len(cases)}: {case['id']}", flush=True)
            trace_path = run_best_of_five(
                str(case["prompt"]),
                case_dir,
                provider="offline",
                base_url=base_url,
                max_rounds=1,
                selection_mode="human_pilot",
            )
        case_report = _case_report(case, trace_path)
        case_report["motion_profile"] = case["motion_profile"]
        case_report["shake_wording"] = case["shake_wording"]
        case_reports.append(case_report)

    passing = [case for case in case_reports if case["status"] == "pass"]
    report = {
        "schema_version": "1.0",
        "kind": "complex_hangten_no_model_structural_sweep",
        "fixture": FIXTURE,
        "status": "pass" if len(passing) == len(cases) else "fail",
        "ready_for_paid_selection": len(passing) == len(cases) == 30,
        "model_calls": 0,
        "requested_prompts": len(cases),
        "passing_prompts": len(passing),
        "rankable_candidates": sum(int(case["rankable_candidates"]) for case in case_reports),
        "unique_rankable_motion_hashes_within_prompt": sum(
            int(case["unique_rankable_motion_hashes"]) for case in case_reports
        ),
        "attempted_candidates": sum(int(case["attempted_candidates"]) for case in case_reports),
        "local_rejections": sum(len(case["local_rejections"]) for case in case_reports),
        "wording_coverage": sorted({str(case["shake_wording"]) for case in case_reports}),
        "cycle_coverage": sorted({float(case["expected_cycles"]) for case in case_reports}),
        "cases": case_reports,
    }
    report_path = output_dir / "structural-sweep-report.json"
    _write(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the no-model 30-prompt complex Hang Ten structural sweep"
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=30)
    arguments = parser.parse_args()
    print(
        run_complex_structural_sweep(
            arguments.output_dir,
            base_url=arguments.base_url,
            limit=arguments.limit,
        )
    )


if __name__ == "__main__":
    main()
