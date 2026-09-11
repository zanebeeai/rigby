from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _payload_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _records_for_trace(
    trace_path: Path,
    *,
    rng: random.Random,
    clip_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    baseline = trace.get("baseline_result_id")
    if not baseline:
        raise ValueError("trace has no precommitted baseline")
    candidates = [
        candidate
        for round_record in trace.get("rounds", [])
        for candidate in round_record.get("candidates", [])
        if candidate.get("result_id")
    ]
    baseline_candidate = next(
        (candidate for candidate in candidates if candidate["result_id"] == baseline),
        None,
    )
    if baseline_candidate is None:
        raise ValueError("trace baseline is not one of its candidates")
    # Only candidates that reached the visual evidence stage belong in the
    # pilot.  Structurally invalid or near-duplicate attempts remain preserved
    # in the source trace, but showing them here would make this a rejection
    # audit rather than a best-of-five perceptibility check.
    challengers = [
        candidate
        for candidate in candidates
        if candidate["result_id"] != baseline
        and (candidate.get("evidence_manifest") or candidate.get("human_review_ready"))
    ]
    if not challengers:
        raise ValueError("trace has no challenger candidates")

    manifest_records: list[dict[str, Any]] = []
    answer_records: list[dict[str, Any]] = []
    for index, challenger in enumerate(challengers, start=1):
        challenger_label = rng.choice(("A", "B"))
        baseline_label = "B" if challenger_label == "A" else "A"
        by_label = {
            challenger_label: challenger["result_id"],
            baseline_label: baseline,
        }
        clip_id = f"{clip_prefix}{index:02d}"
        manifest_records.append(
            {
                "clip_id": clip_id,
                "prompt": trace["prompt"],
                "candidates": {
                    label: {"result_url": f"/api/v1/results/{result_id}"}
                    for label, result_id in by_label.items()
                },
            }
        )
        answer_records.append(
            {
                "clip_id": clip_id,
                "challenger_label": challenger_label,
                "baseline_label": baseline_label,
                "challenger_result_id": challenger["result_id"],
                "baseline_result_id": baseline,
                "challenger_recipe": challenger["recipe"]["name"],
                "baseline_recipe": baseline_candidate["recipe"]["name"],
            }
        )
    return manifest_records, answer_records


def _write_review(
    *,
    output_dir: Path,
    core: dict[str, Any],
    answer_records: list[dict[str, Any]],
    answer_kind: str,
) -> tuple[Path, Path]:
    manifest = {**core, "manifest_sha256": _payload_hash(core)}
    answer_key = {
        "schema_version": "1.0",
        "kind": answer_kind,
        "manifest_sha256": manifest["manifest_sha256"],
        "records": answer_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "variant-manifest.json"
    answer_path = output_dir / "variant-answer-key.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    answer_path.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path, answer_path


def create_variant_review(trace_path: Path, output_dir: Path) -> tuple[Path, Path]:
    manifest_records, answer_records = _records_for_trace(
        trace_path,
        rng=random.Random(20260807),
        clip_prefix="v",
    )

    core = {
        "schema_version": "1.0",
        "kind": "blinded_variant_pilot",
        "source_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "instructions": "Choose A, B, or Tie using the prompt and both synchronized views.",
        "records": manifest_records,
    }
    return _write_review(
        output_dir=output_dir,
        core=core,
        answer_records=answer_records,
        answer_kind="variant_pilot_answer_key",
    )


def create_multi_variant_review(
    trace_paths: list[Path],
    output_dir: Path,
) -> tuple[Path, Path]:
    if len(trace_paths) < 2:
        raise ValueError("multi-prompt review requires at least two traces")
    rng = random.Random(20260808)
    manifest_records: list[dict[str, Any]] = []
    answer_records: list[dict[str, Any]] = []
    source_hashes: list[str] = []
    for trace_index, trace_path in enumerate(trace_paths, start=1):
        records, answers = _records_for_trace(
            trace_path,
            rng=rng,
            clip_prefix=f"p{trace_index:02d}v",
        )
        manifest_records.extend(records)
        answer_records.extend(answers)
        source_hashes.append(hashlib.sha256(trace_path.read_bytes()).hexdigest())
    core = {
        "schema_version": "1.0",
        "kind": "blinded_multi_prompt_variant_pilot",
        "source_trace_sha256": source_hashes,
        "instructions": (
            "Choose A, B, or Tie using the displayed prompt and both synchronized full-FOV views."
        ),
        "records": manifest_records,
    }
    return _write_review(
        output_dir=output_dir,
        core=core,
        answer_records=answer_records,
        answer_kind="multi_prompt_variant_pilot_answer_key",
    )


def _candidate_for_recipe(trace: dict[str, Any], recipe: str) -> dict[str, Any]:
    matches = [
        candidate
        for round_record in trace.get("rounds", [])
        for candidate in round_record.get("candidates", [])
        if candidate.get("recipe", {}).get("name") == recipe
        and candidate.get("result_id")
        and candidate.get("compile_success", True)
        and candidate.get("structural_valid", True)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one structurally valid {recipe!r} candidate, found {len(matches)}")
    return matches[0]


def create_revision_review(
    before_dir: Path,
    after_dir: Path,
    output_dir: Path,
    *,
    recipe: str = "canonical",
) -> tuple[Path, Path]:
    """Blind matched prompts before and after a single implementation change."""
    before_paths = {
        path.parent.name: path for path in sorted(before_dir.glob("*/flywheel-trace.json"))
    }
    after_paths = {
        path.parent.name: path for path in sorted(after_dir.glob("*/flywheel-trace.json"))
    }
    case_ids = sorted(set(before_paths) & set(after_paths))
    if not case_ids or set(before_paths) != set(after_paths):
        raise ValueError("before and after pilot directories must contain the same traced cases")

    rng = random.Random(20260808)
    manifest_records: list[dict[str, Any]] = []
    answer_records: list[dict[str, Any]] = []
    source_hashes: list[dict[str, str]] = []
    for index, case_id in enumerate(case_ids, start=1):
        before_path = before_paths[case_id]
        after_path = after_paths[case_id]
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8"))
        if before.get("prompt") != after.get("prompt"):
            raise ValueError(f"prompt mismatch for {case_id}")
        baseline = _candidate_for_recipe(before, recipe)
        challenger = _candidate_for_recipe(after, recipe)
        challenger_label = rng.choice(("A", "B"))
        baseline_label = "B" if challenger_label == "A" else "A"
        by_label = {
            challenger_label: challenger["result_id"],
            baseline_label: baseline["result_id"],
        }
        clip_id = f"r{index:02d}"
        manifest_records.append(
            {
                "clip_id": clip_id,
                "prompt": after["prompt"],
                "candidates": {
                    label: {"result_url": f"/api/v1/results/{result_id}"}
                    for label, result_id in by_label.items()
                },
            }
        )
        answer_records.append(
            {
                "clip_id": clip_id,
                "challenger_label": challenger_label,
                "baseline_label": baseline_label,
                "challenger_result_id": challenger["result_id"],
                "baseline_result_id": baseline["result_id"],
                "challenger_recipe": f"{recipe}: forearm pronation/supination",
                "baseline_recipe": f"{recipe}: hand-joint deviation",
                "case_id": case_id,
            }
        )
        source_hashes.append(
            {
                "case_id": case_id,
                "before": hashlib.sha256(before_path.read_bytes()).hexdigest(),
                "after": hashlib.sha256(after_path.read_bytes()).hexdigest(),
            }
        )

    core = {
        "schema_version": "1.0",
        "kind": "blinded_matched_revision_review",
        "source_trace_sha256": source_hashes,
        "instructions": (
            "Choose A, B, or Tie using the displayed prompt and both synchronized full-FOV views."
        ),
        "records": manifest_records,
    }
    return _write_review(
        output_dir=output_dir,
        core=core,
        answer_records=answer_records,
        answer_kind="matched_revision_answer_key",
    )


def score_variant_review(submission_path: Path, answer_path: Path) -> dict[str, Any]:
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    if submission.get("manifest_sha256") != answer.get("manifest_sha256"):
        raise ValueError("submission does not match the variant manifest")
    submitted = {
        str(record.get("clip_id")): record
        for record in submission.get("records", [])
        if isinstance(record, dict)
    }
    scored: list[dict[str, Any]] = []
    for expected in answer["records"]:
        choice = submitted.get(expected["clip_id"], {}).get("choice")
        if choice not in {"A", "B", "tie"}:
            continue
        outcome = (
            "tie"
            if choice == "tie"
            else "challenger"
            if choice == expected["challenger_label"]
            else "baseline"
        )
        scored.append(
            {
                "clip_id": expected["clip_id"],
                "choice": choice,
                "outcome": outcome,
                "challenger_recipe": expected["challenger_recipe"],
                "baseline_recipe": expected["baseline_recipe"],
            }
        )
    return {
        "schema_version": "1.0",
        "manifest_sha256": answer["manifest_sha256"],
        "complete": len(scored),
        "challenger_preferred": sum(record["outcome"] == "challenger" for record in scored),
        "baseline_preferred": sum(record["outcome"] == "baseline" for record in scored),
        "ties": sum(record["outcome"] == "tie" for record in scored),
        "records": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or score a blinded within-prompt variant pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("trace", type=Path)
    create.add_argument("output_dir", type=Path)
    create_multi = subparsers.add_parser("create-multi")
    create_multi.add_argument("output_dir", type=Path)
    create_multi.add_argument("traces", type=Path, nargs="+")
    create_revision = subparsers.add_parser("create-revision")
    create_revision.add_argument("before_dir", type=Path)
    create_revision.add_argument("after_dir", type=Path)
    create_revision.add_argument("output_dir", type=Path)
    create_revision.add_argument("--recipe", default="canonical")
    score = subparsers.add_parser("score")
    score.add_argument("submission", type=Path)
    score.add_argument("answer_key", type=Path)
    score.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "create":
        print(*create_variant_review(arguments.trace, arguments.output_dir), sep="\n")
    elif arguments.command == "create-multi":
        print(*create_multi_variant_review(arguments.traces, arguments.output_dir), sep="\n")
    elif arguments.command == "create-revision":
        print(
            *create_revision_review(
                arguments.before_dir,
                arguments.after_dir,
                arguments.output_dir,
                recipe=arguments.recipe,
            ),
            sep="\n",
        )
    else:
        result = score_variant_review(arguments.submission, arguments.answer_key)
        arguments.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(arguments.output)


if __name__ == "__main__":
    main()
