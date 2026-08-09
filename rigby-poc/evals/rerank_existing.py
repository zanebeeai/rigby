from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from evals.capture import refresh_motion_diagnostics
from rigby_poc.judge import VLMJudge, write_judge_record


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def rerank_cases(batch_report: Path, case_ids: list[str]) -> Path:
    batch = json.loads(batch_report.read_text(encoding="utf-8"))
    records = {
        str(record.get("case_id")): record
        for record in batch.get("records", [])
        if isinstance(record, dict)
    }
    judge = VLMJudge()
    for case_id in case_ids:
        record = records.get(case_id)
        if not record:
            raise ValueError(f"batch report has no {case_id}")
        trace_path = Path(str(record["trace"]))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        candidates = [
            candidate
            for round_record in trace.get("rounds", [])
            for candidate in round_record.get("candidates", [])
            if candidate.get("evidence_manifest")
        ]
        if len(candidates) != 5:
            raise ValueError(f"{case_id} has {len(candidates)} evidence candidates, expected five")
        manifests = [Path(str(candidate["evidence_manifest"])) for candidate in candidates]
        for manifest in manifests:
            refresh_motion_diagnostics(manifest)

        round_dir = trace_path.parent / "round-1"
        ranking_path = round_dir / "five-way-ranking.json"
        prior_ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        prior_model = str(prior_ranking.get("call", {}).get("model", "unknown")).replace("/", "-")
        preserved_path = round_dir / f"five-way-ranking-{prior_model}-preserved.json"
        if not preserved_path.is_file():
            shutil.copy2(ranking_path, preserved_path)

        ranking = judge.rank_five(manifests, random_seed=70_000)
        write_judge_record(ranking, ranking_path)
        for index, candidate in enumerate(candidates):
            parsed = ranking["mapped_scores"][index]
            candidate["judgment"] = parsed
            candidate["accepted"] = bool(parsed["accept"])
        winner_index = ranking["mapped_winner_index"]
        winner = candidates[winner_index] if winner_index is not None else None
        if winner is None or not winner["accepted"]:
            trace["winner"] = None
            trace["winner_result_id"] = None
            trace["status"] = "no_acceptable_candidate"
            record["status"] = "no_acceptable_candidate"
            record["winner_result_id"] = None
            record["winner_recipe"] = None
        else:
            trace["winner"] = winner
            trace["winner_result_id"] = winner["result_id"]
            trace["status"] = "winner_selected"
            record["status"] = "winner_selected"
            record["winner_result_id"] = winner["result_id"]
            record["winner_recipe"] = winner["recipe"]["name"]
        trace.setdefault("rerank_history", []).append(
            {
                "reason": "migrate preserved flagship ranking to lightweight comparison role",
                "prior_record": str(preserved_path),
                "prior_model": prior_ranking.get("call", {}).get("model"),
                "replacement_record": str(ranking_path),
                "replacement_model": ranking["call"]["model"],
            }
        )
        trace["rankings"] = [
            {
                "round": 1,
                "record": str(ranking_path),
                "mapped_winner_result_id": trace.get("winner_result_id"),
                "response_id": ranking["call"]["response_id"],
            }
        ]
        _write(trace_path, trace)
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "status": trace["status"],
                    "winner_recipe": record.get("winner_recipe"),
                    "model": ranking["call"]["model"],
                    "usage": ranking["call"]["usage"],
                }
            ),
            flush=True,
        )
    batch["records"] = [records[str(record["case_id"])] for record in batch["records"]]
    batch["summary"] = {
        "requested": len(batch["records"]),
        "completed": len(batch["records"]),
        "winners_selected": sum(record.get("status") == "winner_selected" for record in batch["records"]),
        "errors": sum(record.get("status") == "error" for record in batch["records"]),
    }
    _write(batch_report, batch)
    return batch_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank preserved five-way evidence with current model roles")
    parser.add_argument("batch_report", type=Path)
    parser.add_argument("case_id", nargs="+")
    arguments = parser.parse_args()
    print(rerank_cases(arguments.batch_report, arguments.case_id))


if __name__ == "__main__":
    main()
