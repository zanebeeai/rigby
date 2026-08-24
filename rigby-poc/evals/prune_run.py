"""Plan 05 section 3.7 — bound what a finished run keeps.

One prompt costs roughly 58 MiB: 39 MiB in the run directory, almost all evidence PNGs,
plus ~19 MiB in the `ResultStore`, which persists a folder for every attempted candidate
rather than only the captured ones. Nothing was ever collected.

This tool handles the `ResultStore` half. A candidate that was compiled, rejected, and
never selected still keeps the record of *what* it was and *why* it lost -- its program,
its metrics, its provenance -- and loses only the two large artifacts that exist to be
replayed: `clip.json` and `animation.glb`. Evidence PNG compaction is PR 01d's, which
transcodes to WebP after a run reaches a terminal state.

Pruning is destructive, so it reports by default and acts only under `--apply`.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rigby_poc.compiler import PROJECT_ROOT


# The record of a losing candidate; everything else about it is reproducible by
# recompiling its program, which is deterministic (plan 03 section 1.1).
RETAINED_FILES = ("request.json", "scene.json", "program.json", "metrics.json", "provenance.json")
# A result id names one directory directly under the store. It arrives from a trace file,
# which is data, so it is validated as a name rather than trusted as a path.
SAFE_RESULT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
PRUNABLE_FILES = ("clip.json", "animation.glb")
TERMINAL_STATUSES = frozenset({"completed", "failed", "unsupported", "no_acceptable_candidate"})


class RunNotTerminal(RuntimeError):
    """The run is still queued or running; pruning it would race the writer."""


@dataclass
class PrunePlan:
    run_id: str
    winner_result_id: str | None
    protected: list[str] = field(default_factory=list)
    prunable: list[tuple[str, Path]] = field(default_factory=list)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(path.stat().st_size for _, path in self.prunable if path.is_file())

    def describe(self) -> str:
        lines = [
            f"run {self.run_id}",
            f"  winner: {self.winner_result_id or '(none)'}",
            f"  protected results: {len(self.protected)}",
            f"  prunable files: {len(self.prunable)}",
            f"  reclaimable: {self.reclaimable_bytes / 1_048_576:.2f} MiB",
        ]
        lines.extend(f"    {path}" for _, path in self.prunable)
        return "\n".join(lines)


def _run_record(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no run.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a run record")
    return value


def _trace(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "artifacts" / "flywheel-trace.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def candidate_result_ids(trace: dict[str, Any]) -> list[str]:
    """Every result id the run produced, in the order the trace records them."""
    seen: dict[str, None] = {}
    for round_record in trace.get("rounds", []):
        if not isinstance(round_record, dict):
            continue
        for candidate in round_record.get("candidates", []):
            if isinstance(candidate, dict) and candidate.get("result_id"):
                seen.setdefault(str(candidate["result_id"]), None)
    return list(seen)


def plan_prune(
    run_dir: Path,
    *,
    store_root: Path | None = None,
    protect: Iterable[str] = (),
) -> PrunePlan:
    """Decide what a finished run may drop, without touching anything."""
    record = _run_record(run_dir)
    status = str(record.get("status", ""))
    if status not in TERMINAL_STATUSES:
        raise RunNotTerminal(f"run {record.get('run_id')} is {status!r}, not terminal")

    trace = _trace(run_dir)
    winner = record.get("winner_result_id") or trace.get("winner_result_id")
    winner_id = str(winner) if winner else None
    root = store_root or PROJECT_ROOT / "results"

    protected = {result_id for result_id in protect if result_id}
    if winner_id:
        protected.add(winner_id)

    plan = PrunePlan(
        run_id=str(record.get("run_id", run_dir.name)),
        winner_result_id=winner_id,
        protected=sorted(protected),
    )
    resolved_root = root.resolve()
    for result_id in candidate_result_ids(trace):
        if result_id in protected:
            continue
        # Two independent checks, because this tool deletes. The first rejects anything
        # that is not a bare directory name, on either platform's separator. The second
        # is containment by ancestry rather than by parent equality: equality would also
        # reject a legitimately nested layout, and on Windows `resolve` normalises case
        # and 8.3 short names, so comparing one component is the wrong instrument.
        if not SAFE_RESULT_ID.fullmatch(result_id) or Path(result_id).name != result_id:
            continue
        folder = root / result_id
        if not folder.is_dir() or resolved_root not in folder.resolve().parents:
            continue
        for name in PRUNABLE_FILES:
            path = folder / name
            if path.is_file():
                plan.prunable.append((result_id, path))
    return plan


def apply_prune(plan: PrunePlan) -> int:
    """Delete what the plan listed. Returns the bytes reclaimed."""
    reclaimed = 0
    for _, path in plan.prunable:
        if path.is_file():
            reclaimed += path.stat().st_size
            path.unlink()
    return reclaimed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="a directory under results/pipeline-runs/")
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument(
        "--protect",
        action="append",
        default=[],
        help="a result id to keep whole, repeatable; the winner is always protected",
    )
    parser.add_argument("--apply", action="store_true", help="delete, rather than report")
    arguments = parser.parse_args()

    plan = plan_prune(
        arguments.run_dir, store_root=arguments.store_root, protect=arguments.protect
    )
    print(plan.describe())
    if not arguments.apply:
        print("\nreport only; pass --apply to delete")
        return
    reclaimed = apply_prune(plan)
    print(f"\nreclaimed {reclaimed / 1_048_576:.2f} MiB")


if __name__ == "__main__":
    main()
