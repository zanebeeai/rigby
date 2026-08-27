from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from evals.capture import capture_result_frames
from evals.preference_review import create_preference_review
from rigby_poc.judge import VLMJudge, write_judge_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "heldout_hangten_v4_anatomical_shake_precommitted_perceptual_diversity"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _resolve(path: object) -> Path:
    value = Path(str(path))
    return value if value.is_absolute() else PROJECT_ROOT / value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_candidates(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        candidate
        for round_record in trace.get("rounds", [])
        if isinstance(round_record, dict)
        for candidate in round_record.get("candidates", [])
        if isinstance(candidate, dict)
    ]
    rankable = [candidate for candidate in candidates if candidate.get("human_review_ready")]
    if len(rankable) != 5:
        raise ValueError(f"source trace has {len(rankable)}/5 rankable candidates")
    if any(
        candidate.get("compile_success") is not True
        or candidate.get("structural_valid") is not True
        or not candidate.get("motion_sha256")
        for candidate in rankable
    ):
        raise ValueError("rankable source candidate lacks structural or hash evidence")
    baseline_id = trace.get("baseline_result_id")
    baseline = next(
        (candidate for candidate in candidates if candidate.get("result_id") == baseline_id),
        None,
    )
    if baseline is None or baseline.get("structural_valid") is not True:
        raise ValueError("precommitted single-sample baseline is missing or structurally invalid")
    return rankable, baseline


def _usage(ranking: dict[str, Any]) -> dict[str, int]:
    totals = {"attempts": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for attempt in ranking.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        totals["attempts"] += 1
        usage = attempt.get("usage", {})
        if isinstance(usage, dict):
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                totals[key] += int(usage.get(key, 0) or 0)
    return totals


def _report(
    *,
    sweep_path: Path,
    records: list[dict[str, Any]],
    max_new_prompts: int,
    max_new_model_calls: int,
    new_attempts: int,
    usage: dict[str, int],
) -> dict[str, Any]:
    complete = sum(record.get("status") == "winner_selected" for record in records)
    errors = sum(record.get("status") == "error" for record in records)
    return {
        "schema_version": "1.0",
        "fixture": FIXTURE,
        "kind": "bounded_five_way_selection_from_structural_sweep",
        "selection_mode": "five_way_from_preserved_candidates",
        "source_sweep": str(sweep_path),
        "source_sweep_sha256": _sha256(sweep_path),
        "status": "complete" if complete == len(records) == 30 else "error" if errors else "ready",
        "records": records,
        "summary": {
            "requested": len(records),
            "winners_selected": complete,
            "pending": len(records) - complete - errors,
            "errors": errors,
            "automatic_baseline_ties": sum(
                record.get("winner_result_id") == record.get("baseline_result_id")
                for record in records
                if record.get("status") == "winner_selected"
            ),
        },
        "cost_control": {
            "max_new_prompts_this_run": max_new_prompts,
            "max_new_model_calls_this_run": max_new_model_calls,
            "new_prompt_attempts": new_attempts,
            "model_calls_dispatched_this_run": usage["attempts"],
            "remaining_model_calls_this_run": max(
                0, max_new_model_calls - usage["attempts"]
            ),
            "completed_prompts_total": complete,
            "remaining_prompts": len(records) - complete,
            "maximum_additional_five_way_calls": len(records) - complete,
            "usage_this_run": usage,
        },
    }


def run_sweep_selection(
    sweep_report: Path,
    output_dir: Path,
    *,
    base_url: str = "http://127.0.0.1:8000",
    max_new_prompts: int = 0,
    max_new_model_calls: int | None = None,
    judge: VLMJudge | None = None,
    capture_fn: Callable[..., Path] = capture_result_frames,
) -> Path:
    """Rank preserved five-candidate sets without recompiling or forced uplift.

    A no-clear-winner decision conservatively selects the precommitted baseline,
    which becomes an automatic tie in the later human uplift review. The judge
    is constructed lazily, so ``max_new_prompts=0`` is a true no-call preflight.
    """
    if max_new_prompts < 0:
        raise ValueError("max_new_prompts cannot be negative")
    if max_new_model_calls is None:
        max_new_model_calls = max_new_prompts
    if max_new_model_calls < 0:
        raise ValueError("max_new_model_calls cannot be negative")
    sweep_report = sweep_report.resolve()
    sweep = _read(sweep_report)
    if sweep.get("fixture") != FIXTURE or sweep.get("status") != "pass":
        raise ValueError("selection requires the passing v4 complex structural sweep")
    cases = sweep.get("cases")
    if not isinstance(cases, list) or len(cases) < 30:
        raise ValueError("selection requires at least 30 swept prompts")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "batch-report.json"
    review_manifest = output_dir / "human-review" / "comparison-manifest.json"
    if report_path.is_file() and review_manifest.is_file():
        prior_complete = _read(report_path)
        if (
            prior_complete.get("status") == "complete"
            and prior_complete.get("source_sweep_sha256") == _sha256(sweep_report)
        ):
            return report_path

    prior_records: dict[str, dict[str, Any]] = {}
    if report_path.is_file():
        prior = _read(report_path)
        prior_records = {
            str(record.get("case_id")): record
            for record in prior.get("records", [])
            if isinstance(record, dict)
        }

    records: list[dict[str, Any]] = []
    new_attempts = 0
    usage = {"attempts": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    stop_after_error = False
    for case_index, case in enumerate(cases[:30], start=1):
        case_id = str(case["id"])
        trace_path = _resolve(case["trace"])
        trace_hash = _sha256(trace_path)
        trace = _read(trace_path)
        if trace.get("status") != "awaiting_human_selection" or trace.get("selection_mode") != "human_pilot":
            raise ValueError(f"{case_id}: source trace is not a completed no-model candidate set")
        rankable, baseline = _source_candidates(trace)
        prior = prior_records.get(case_id)
        if (
            prior
            and prior.get("status") == "winner_selected"
            and prior.get("source_trace_sha256") == trace_hash
            and _resolve(prior.get("ranking_record")).is_file()
        ):
            records.append(prior)
            continue

        pending = {
            "case_id": case_id,
            "prompt": case["prompt"],
            "hand": case["hand"],
            "motion_profile": case["motion_profile"],
            "expected_cycles": case["expected_cycles"],
            "shake_wording": case["shake_wording"],
            "source_trace": str(trace_path),
            "source_trace_sha256": trace_hash,
            "baseline_result_id": baseline["result_id"],
            "status": "pending",
        }
        if (
            stop_after_error
            or new_attempts >= max_new_prompts
            or usage["attempts"] >= max_new_model_calls
        ):
            records.append(pending)
            continue

        new_attempts += 1
        case_dir = output_dir / "runs" / case_id
        try:
            manifests: list[Path] = []
            evidence_records: list[dict[str, str]] = []
            for candidate in rankable:
                result_id = str(candidate["result_id"])
                manifest = case_dir / "evidence" / result_id / "evidence-manifest.json"
                if not manifest.is_file():
                    manifest = capture_fn(
                        result_id,
                        manifest.parent,
                        base_url=base_url,
                    )
                manifests.append(manifest)
                evidence_records.append(
                    {
                        "result_id": result_id,
                        "evidence_manifest": str(manifest),
                    }
                )
            if judge is None:
                judge = VLMJudge(max_model_calls=max_new_model_calls)
            ranking = judge.rank_five(manifests, random_seed=90_000 + case_index)
            ranking_path = case_dir / "five-way-ranking.json"
            write_judge_record(ranking, ranking_path)
            ranking_usage = _usage(ranking)
            for key in usage:
                usage[key] += ranking_usage[key]

            judged_candidates: list[dict[str, Any]] = []
            for index, candidate in enumerate(rankable):
                score = ranking["mapped_scores"][index]
                judged_candidates.append(
                    {
                        "result_id": candidate["result_id"],
                        "recipe": candidate["recipe"],
                        "motion_sha256": candidate["motion_sha256"],
                        "quality_metrics": candidate["quality_metrics"],
                        "judgment": score,
                    }
                )
            winner_index = ranking.get("mapped_winner_index")
            clear_winner = bool(
                isinstance(winner_index, int)
                and 0 <= winner_index < len(rankable)
                and ranking["mapped_scores"][winner_index].get("accept") is True
            )
            winner = rankable[winner_index] if clear_winner else baseline
            selection_trace = json.loads(json.dumps(trace))
            evidence_by_result = {
                record["result_id"]: record["evidence_manifest"]
                for record in evidence_records
            }
            score_by_result = {
                candidate["result_id"]: ranking["mapped_scores"][index]
                for index, candidate in enumerate(rankable)
            }
            selected_winner: dict[str, Any] | None = None
            for round_record in selection_trace.get("rounds", []):
                for candidate in round_record.get("candidates", []):
                    result_id = candidate.get("result_id")
                    if result_id in score_by_result:
                        candidate["evidence_manifest"] = evidence_by_result[result_id]
                        candidate["judgment"] = score_by_result[result_id]
                        candidate["accepted"] = bool(score_by_result[result_id].get("accept"))
                    if result_id == winner["result_id"]:
                        selected_winner = candidate
            if selected_winner is None:
                raise ValueError("selected winner is absent from the preserved source trace")
            selection_trace.update(
                {
                    "selection_mode": "five_way_from_preserved_candidates",
                    "capture_policy": "full_phase_ego_orbit",
                    "source_no_model_trace": str(trace_path),
                    "source_no_model_trace_sha256": trace_hash,
                    "winner_result_id": winner["result_id"],
                    "winner": selected_winner,
                    "status": "winner_selected",
                }
            )
            selection_trace.setdefault("rankings", []).append(
                {
                    "round": 1,
                    "record": str(ranking_path),
                    "mapped_winner_result_id": winner["result_id"],
                    "selection_reason": (
                        "clear_five_way_winner"
                        if clear_winner
                        else "no_clear_winner_conservative_baseline"
                    ),
                    "response_id": ranking.get("call", {}).get("response_id"),
                }
            )
            selection_trace_path = case_dir / "selection-trace.json"
            _write(selection_trace_path, selection_trace)
            record = {
                **pending,
                "status": "winner_selected",
                "winner_result_id": winner["result_id"],
                "winner_recipe": winner["recipe"]["name"],
                "selection_reason": (
                    "clear_five_way_winner"
                    if clear_winner
                    else "no_clear_winner_conservative_baseline"
                ),
                "ranking_record": str(ranking_path),
                "trace": str(selection_trace_path),
                "response_id": ranking.get("call", {}).get("response_id"),
                "evidence": evidence_records,
                "judged_candidates": judged_candidates,
                "model_usage": ranking_usage,
            }
            records.append(record)
        except Exception as error:
            records.append(
                {
                    **pending,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            stop_after_error = True

        interim = _report(
            sweep_path=sweep_report,
            records=[*records, *[
                {
                    "case_id": str(rest["id"]),
                    "prompt": rest["prompt"],
                    "status": "pending",
                }
                for rest in cases[len(records):30]
            ]],
            max_new_prompts=max_new_prompts,
            max_new_model_calls=max_new_model_calls,
            new_attempts=new_attempts,
            usage=usage,
        )
        _write(report_path, interim)

    report = _report(
        sweep_path=sweep_report,
        records=records,
        max_new_prompts=max_new_prompts,
        max_new_model_calls=max_new_model_calls,
        new_attempts=new_attempts,
        usage=usage,
    )
    _write(report_path, report)
    if report["status"] == "complete":
        create_preference_review(report_path, output_dir / "human-review")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bounded five-way selection over the preserved v4 structural sweep"
    )
    parser.add_argument("sweep_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-new-prompts", type=int, default=0)
    parser.add_argument("--max-new-model-calls", type=int)
    arguments = parser.parse_args()
    print(
        run_sweep_selection(
            arguments.sweep_report,
            arguments.output_dir,
            base_url=arguments.base_url,
            max_new_prompts=arguments.max_new_prompts,
            max_new_model_calls=arguments.max_new_model_calls,
        )
    )


if __name__ == "__main__":
    main()
