"""Audit any file or directory for rates published without their n and baseline.

The test-suite guard (``tests/test_published_rates.py``) is scoped to committed
markdown and report artifacts, because that is what a fresh clone contains and
what CI can enforce. This is the same scanner pointed anywhere, for the things
the gate cannot cover: the plan documents, which are deliberately not committed,
a draft, a PR description, or a report before it is written into the repository.

Run:

    uv run python -m evals.rate_audit docs/plans/07-judge-harness.md
    uv run python -m evals.rate_audit docs/plans/          # every plan
    uv run python -m evals.rate_audit --quiet <path>       # exit code only

Lane ``judge`` audited plan 07 by hand after the committed guard failed on this
repository's own documentation, and found three rates missing their n and one
duplicated paragraph. Their observation is the reason this exists: **the guard is
the only thing in the repository that makes anyone read the prose adversarially.**
The suite proves the code; nothing proves the writing, and theirs had drifted
inside a day.

Exit code is the number of findings, capped at 125, so it composes with a shell
`if`. It is deliberately not wired into the suite: an advisory tool that fails
the build gets its findings waived, and this one is meant to be argued with.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rigby_poc.published_rates import Finding, scan_prose, scan_report


def audit(target: Path) -> list[Finding]:
    """Every finding under ``target``, which may be a file or a directory."""

    if target.is_file():
        paths = [target]
    else:
        paths = sorted(
            path
            for pattern in ("*.md", "*.json")
            for path in target.rglob(pattern)
            if not {".venv", "node_modules", ".git"}.intersection(path.parts)
        )
    findings: list[Finding] = []
    for path in paths:
        scan = scan_report if path.suffix == ".json" else scan_prose
        try:
            findings.extend(scan(path))
        except (ValueError, UnicodeDecodeError):
            continue  # not a document this scanner understands
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--quiet", action="store_true", help="exit code only")
    arguments = parser.parse_args(argv)

    findings: list[Finding] = []
    for target in arguments.targets:
        if not target.exists():
            print(f"no such path: {target}", file=sys.stderr)
            return 126
        findings.extend(audit(target))

    if not arguments.quiet:
        for finding in findings:
            print(f"  {finding}")
        total = len(findings)
        print(
            f"\n{total} rate{'' if total == 1 else 's'} stated without n or baseline."
            + (
                "\nA target ('at least 95%') and a statistical parameter ('95% "
                "confidence') are\nnot published rates and are not reported."
                if total
                else ""
            )
        )
    return min(len(findings), 125)


if __name__ == "__main__":
    raise SystemExit(main())
