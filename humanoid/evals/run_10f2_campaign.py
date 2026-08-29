"""10f2: the grader calibration re-run, committed and durable this time.

The 10f run's driver and per-clip records lived in ``/tmp`` and are gone —
its aggregates are unauditable and the band-restricted question (368 of 1288
mutated clips are subperceptual by design, so pooled MCC scores the grader
against a non-perceptual oracle) cannot be answered from the report alone.
This driver reproduces the run with three corrections, each a recorded lesson:

1. **Durable records.** One JSON per (case, spec) under a run root inside the
   repo's git-ignored ``results/`` (or any ``--run-root``), plus the full
   judgment via ``write_judge_record`` and a ``ReplayStore`` copy so any
   accept definition can be re-scored offline without a model call.
2. **Terminal-vs-transient status.** A record counts as done only when it
   ``scored`` or failed terminally (``not_applicable``, ``compile_failed``).
   ``render_failed`` and ``grade_failed`` are transient: a resume redoes them.
   The 10f run nearly lost 131 clips to the opposite reading.
3. **A hard call ceiling.** There is no committed price table, so the budget
   is denominated in model calls (10f: 6,670 calls ≈ $9.41 at the
   secondary-sourced rates). The driver stops cleanly at ``--max-calls``.

Bands are recorded per clip (``tier``), because their absence is exactly what
made the pooled-MCC question unanswerable post hoc.

Zero-model-call modes: ``--skip-grading`` exercises compile → persist →
capture → record without constructing a judge, and ``--limit N`` bounds any
run for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.capture import capture_results
from evals.corpus import load_corpus
from evals.corpus.loader import compile_case
from evals.mutations.anatomy import rom_sweep
from evals.mutations.checks import require_known_targets
from evals.mutations.clipping import limb_through_torso_sweep
from evals.mutations.signal import jitter_sweep
from evals.mutations.spec import MutationSpec
from evals.mutations.sweep import tier_for
from evals.mutations.timing import snap_sweep
from rigby_poc.judge import (
    MotionJudgeScore,
    VLMJudge,
    meets_acceptance_thresholds,
    write_judge_record,
)
from rigby_poc.models import CompileRequest
from rigby_poc.store import ResultStore

RECORD_SCHEMA = "10f2.1"
TERMINAL_STATUSES = {"scored", "not_applicable", "compile_failed"}

#: The rom sweep needs a (bone, dof) and the 10f run never recorded its
#: choice, so this one is explicit and non-comparable by construction. The
#: shoulder is the best-calibrated joint in the corpus (n = 42 per the 03d
#: handoff) and its flexion baseline is within_typical on the arm-driven
#: cases, so rom_guard leaves a usable arm.
ROM_BONE = "leftUpperArm"
ROM_DOF = "flexion"


def campaign_specs() -> list[MutationSpec]:
    specs = (
        rom_sweep(ROM_BONE, ROM_DOF)
        + jitter_sweep()
        + limb_through_torso_sweep()
        + snap_sweep()
    )
    require_known_targets(specs)
    return specs


@dataclass
class Job:
    case_id: str
    spec: MutationSpec | None

    @property
    def key(self) -> str:
        spec_id = self.spec.id if self.spec is not None else "unmutated"
        return f"{self.case_id}--{spec_id}".replace(".", "_")

    def record_skeleton(self, run_id: str) -> dict[str, Any]:
        return {
            "schema_version": RECORD_SCHEMA,
            "run_id": run_id,
            "key": self.key,
            "case_id": self.case_id,
            "spec_id": self.spec.id if self.spec else None,
            "family": self.spec.family.value if self.spec else None,
            "severity": self.spec.severity if self.spec else None,
            "tier": tier_for(self.spec.severity).value if self.spec else None,
        }


def write_record(records_dir: Path, job_key: str, record: dict[str, Any]) -> None:
    path = records_dir / f"{job_key}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_terminal_keys(records_dir: Path) -> set[str]:
    done: set[str] = set()
    for path in records_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") in TERMINAL_STATUSES:
            done.add(record["key"])
    return done


def spent_calls(records_dir: Path) -> int:
    total = 0
    for path in records_dir.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total += int(record.get("dispatch_count") or 0)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--env-file", type=Path, default=None,
                        help="dotenv file holding the judge credentials")
    parser.add_argument("--max-calls", type=int, default=12000,
                        help="hard ceiling on grader model calls, resume-aware")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="only this many pending jobs (smoke runs)")
    parser.add_argument("--skip-grading", action="store_true",
                        help="compile/persist/capture/record only; no judge")
    args = parser.parse_args()

    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)

    run_root = args.run_root
    records_dir = run_root / "records"
    judgments_dir = run_root / "judgments"
    evidence_root = run_root / "evidence"
    for directory in (records_dir, judgments_dir, evidence_root):
        directory.mkdir(parents=True, exist_ok=True)
    run_id = run_root.name

    specs = campaign_specs()
    corpus = list(load_corpus())
    store = ResultStore()

    # Compile every case once; a case that cannot compile gets terminal
    # records for every job that names it.
    clips: dict[str, Any] = {}
    cases_by_id = {case.root.name: case for case in corpus}
    jobs: list[Job] = []
    for case_id in sorted(cases_by_id):
        jobs.append(Job(case_id, None))
        jobs.extend(Job(case_id, spec) for spec in specs)

    done = load_terminal_keys(records_dir)
    pending = [job for job in jobs if job.key not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"campaign {run_id}: {len(jobs)} jobs total, {len(done)} terminal, "
        f"{len(pending)} to run; rom target {ROM_BONE}.{ROM_DOF}; "
        f"calls spent so far {spent_calls(records_dir)} / {args.max_calls}"
    )

    judge = None
    if not args.skip_grading:
        judge = VLMJudge(grader_mode="split")

    calls_used = spent_calls(records_dir)
    scored = failed = skipped = 0

    def compile_cached(case_id: str):
        if case_id not in clips:
            try:
                clips[case_id] = compile_case(cases_by_id[case_id])
            except Exception:
                clips[case_id] = None
        return clips[case_id]

    for start in range(0, len(pending), args.chunk_size):
        if calls_used >= args.max_calls:
            print(f"call ceiling reached ({calls_used}); stopping cleanly")
            break
        chunk = pending[start : start + args.chunk_size]

        # Stage 1: compile / mutate / persist.
        renderable: list[tuple[Job, str]] = []
        for job in chunk:
            record = job.record_skeleton(run_id)
            clip = compile_cached(job.case_id)
            if clip is None or not clip.frames:
                record.update(status="compile_failed",
                              error="base case does not compile to frames")
                write_record(records_dir, job.key, record)
                skipped += 1
                continue
            if job.spec is not None:
                applicability = job.spec.applies_to(clip)
                if not applicability.ok:
                    record.update(
                        status="not_applicable",
                        applicable=False,
                        skip_reason=applicability.reason,
                    )
                    write_record(records_dir, job.key, record)
                    skipped += 1
                    continue
                record.update(applicable=True,
                              static_target=applicability.static_target)
                clip = job.spec.apply(clip)
            case = cases_by_id[job.case_id]
            request = CompileRequest(
                scene=case.scene, program=case.program, persist=True
            )
            result_id = store.persist(request, clip)
            record.update(status="pending", mutated_result_id=result_id)
            write_record(records_dir, job.key, record)
            renderable.append((job, result_id))

        if not renderable:
            continue

        # Stage 2: capture, one browser per chunk.
        requests = [
            (result_id, evidence_root / result_id) for _, result_id in renderable
        ]
        try:
            manifests = capture_results(requests, base_url=args.base_url)
        except Exception as error:
            for job, result_id in renderable:
                record = job.record_skeleton(run_id)
                record.update(
                    status="render_failed",
                    mutated_result_id=result_id,
                    error=f"batch capture failed: {error}",
                )
                write_record(records_dir, job.key, record)
                failed += len(renderable)
            print(f"chunk capture failed: {error}")
            time.sleep(10)
            continue

        # Stage 3: grade and record.
        for (job, result_id), manifest_path in zip(renderable, manifests):
            record = job.record_skeleton(run_id)
            record.update(mutated_result_id=result_id)
            if job.spec is not None:
                applicability = job.spec.applies_to(clips[job.case_id])
                record.update(applicable=True,
                              static_target=applicability.static_target)
            if args.skip_grading:
                record.update(status="scored", accept=None,
                              skip_reason="grading skipped by flag")
                write_record(records_dir, job.key, record)
                scored += 1
                continue
            try:
                judgment = judge.score(Path(manifest_path))
                parsed = judgment["call"]["parsed"]
                score = MotionJudgeScore.model_validate(parsed)
                accept = bool(meets_acceptance_thresholds(score))
                # Per-grader dispatch counts, not routing.model_calls_made:
                # that one is the judge instance's CUMULATIVE counter across
                # every clip it has scored, so summing it double-counts the
                # budget. Measured on the smoke run: 5 then 10 for two clips
                # of 5 dispatches each.
                dispatches = sum(
                    int((grader.get("routing") or {}).get("dispatch_count") or 0)
                    for grader in (judgment.get("graders") or {}).values()
                )
                usage = judgment.get("call", {}).get("usage") or {}
                escalated = sorted(
                    name
                    for name, grader in (judgment.get("graders") or {}).items()
                    if (grader.get("routing") or {}).get("escalated")
                )
                record.update(
                    status="scored",
                    accept=accept,
                    dimension_scores=parsed.get("dimension_scores"),
                    escalated_graders=escalated,
                    dispatch_count=dispatches,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    evidence_manifest_sha256=judgment.get(
                        "evidence_manifest_sha256"
                    ),
                )
                write_judge_record(
                    judgment, judgments_dir / f"{job.key}.json"
                )
                calls_used += max(dispatches, 1)
                scored += 1
            except Exception as error:
                record.update(
                    status="grade_failed",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=3),
                )
                failed += 1
                if "RateLimit" in type(error).__name__:
                    print("rate limited; backing off 30 s")
                    time.sleep(30)
            write_record(records_dir, job.key, record)

        # Stage 4: evidence PNGs are the disk cost; the manifest and hashes
        # live on in the judgment record, so drop the pixels once scored.
        for _, result_id in renderable:
            directory = evidence_root / result_id
            for png in directory.glob("*.png"):
                png.unlink(missing_ok=True)

        print(
            f"progress: scored {scored}, transient-failed {failed}, "
            f"skipped {skipped}, calls {calls_used}/{args.max_calls}"
        )

    print(
        f"done: scored {scored}, transient-failed {failed}, skipped {skipped}, "
        f"calls {calls_used}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
