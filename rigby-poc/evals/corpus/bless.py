"""Re-record corpus expectations, deliberately and visibly.

A corpus that silently re-records its own expectations tests nothing (plan 03
section 3.2), so ``bless`` is a dry run by default: it recompiles, prints the diff,
and changes nothing until ``--write`` is passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .freeze import upsert_entry
from .loader import (
    EXPECTED_FILE,
    CorpusCase,
    load_corpus,
    load_manifest,
    manifest_path,
    write_json,
)
from .verify import CaseComparison, Verdict, compare_case, rebless


@dataclass(frozen=True)
class BlessReport:
    comparisons: list[CaseComparison]
    written: list[str]
    dry_run: bool

    @property
    def moved(self) -> list[CaseComparison]:
        return [item for item in self.comparisons if item.verdict is Verdict.MISMATCH]

    @property
    def skipped(self) -> list[CaseComparison]:
        return [
            item for item in self.comparisons if item.verdict is Verdict.UNBLESSED_PLATFORM
        ]

    def render(self) -> str:
        lines = [f"corpus: {len(self.comparisons)} case(s) recompiled"]
        lines.extend(comparison.render() for comparison in self.comparisons)
        lines.append("")
        if not self.moved and not self.skipped:
            lines.append("No expectation moved. Nothing to bless.")
            return "\n".join(lines)
        if self.moved:
            lines.append(f"{len(self.moved)} case(s) moved:")
            lines.extend(f"  - {item.case_id}" for item in self.moved)
            lines.append("")
            lines.append(
                "State in the PR description why each hash moved. A hash that moved "
                "without an intended compiler change is a regression, not a re-bless."
            )
        if self.skipped:
            lines.append(
                f"{len(self.skipped)} case(s) have no hash for this platform; "
                "--write contributes one without touching other platforms."
            )
        lines.append("")
        lines.append(
            "Dry run: nothing written. Re-run with --write to apply."
            if self.dry_run
            else f"Wrote {len(self.written)} expected.json file(s)."
        )
        return "\n".join(lines)


def bless(
    case_ids: list[str] | None = None,
    *,
    root: Path | None = None,
    write: bool = False,
) -> BlessReport:
    cases: list[CorpusCase] = load_corpus(root)
    if case_ids:
        wanted = set(case_ids)
        unknown = sorted(wanted - {case.id for case in cases})
        if unknown:
            raise SystemExit(f"unknown corpus case(s): {unknown}")
        cases = [case for case in cases if case.id in wanted]

    comparisons: list[CaseComparison] = []
    written: list[str] = []
    manifest = load_manifest(root)
    for case in cases:
        comparison = compare_case(case)
        comparisons.append(comparison)
        needs_write = comparison.verdict is not Verdict.MATCH
        if write and needs_write and comparison.observed is not None:
            merged = rebless(case, comparison.observed)
            write_json(case.root / EXPECTED_FILE, merged.model_dump(mode="json"))
            written.append(case.id)
            # Keep the manifest row consistent with what was just recorded. The
            # *diff*, not a stale manifest, is what makes a moved hash visible.
            manifest = upsert_entry(
                manifest,
                case.entry.model_copy(
                    update={"expected_structural_valid": merged.structural_valid}
                ),
            )
    if written:
        write_json(manifest_path(root), manifest.model_dump(mode="json"))
    return BlessReport(comparisons=comparisons, written=written, dry_run=not write)
