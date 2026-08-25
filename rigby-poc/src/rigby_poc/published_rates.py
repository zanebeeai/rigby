"""L3 gate item 3 / plan 10 §9.5: no published rate without its n and its baseline.

A rate on its own is not a result. "90% agreement" is compatible with 9 of 10 and
with 900 of 1000, and those are different claims; and without a baseline it is
compatible with a constant predictor scoring exactly what chance would score. The
previous calibration was invalidated by precisely that: every scored item shared
one ground-truth label, so "always pick the base clip" scored 0.9 against a 0.8
gate. See ``evals/calibration_stats.py``.

This module finds rates that are *published* -- stated in prose or written into a
committed report artifact -- and checks each carries both. It is deliberately
narrow about what counts as publishing:

* A **target** is not a published rate. "at least 95% correct rejection" is a
  requirement; it makes no claim about observed data and needs no n.
* A **statistical parameter** is not a published rate. "95% confidence interval",
  "50% detection threshold" name a method, not a result.
* A **measured claim** is a published rate and must carry both.

Waivers exist, carry an owner and a reason, and are themselves checked: a waived
file may not acquire new deficiencies. See ``WAIVERS``.

What this guard does NOT catch
------------------------------

Stated here rather than in a plan, because a guard that appears to cover a class
while covering only part of it is itself an instance of the defect it exists to
prevent -- something present, authoritative-looking, and not derived from what it
claims to describe.

This guard checks that a rate is *stated* with its n and its baseline. It cannot
check that the underlying values were ever *measured*. Two known holes, both
found by other lanes:

1. **Seeded defaults.** ``compiler._base_metrics`` initialises ten metrics that
   several compile paths never write -- ``max_penetration_m: 0.0`` with nothing
   having looked, ``lost_table_contact: True`` with no table in the scene. A rate
   computed over those carries a perfectly good n and a perfectly good baseline
   while summarising a constant. Pinned as ``UNWRITTEN_BASE_DEFAULTS``; the fix
   is a ``not_measured`` sentinel distinct from a measured zero, not a change
   here.

2. **Carried-over observations.** Mutations recompile nothing, so a mutated clip
   inherits metrics describing the pre-mutation motion -- including
   ``structural_valid``. A rate over "clips that passed structural validation"
   would, for the deferred intents, be computed over a cached answer to a
   different question. Guarded by ``analysis.equivalence``'s
   ``STALE_AFTER_MUTATION``, which raises on read.

3. **A correct n against the wrong reference frame.** A number can be correctly
   computed over a correctly stated sample and still be wrong because it was
   measured from the wrong zero. No static check catches that; the mitigation is
   ``config/thresholds.v1.json`` requiring ``reference_frame`` on any measured
   source.

The common shape is that n and baseline describe the *sample*, and none of these
three is a defect of the sample.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

#: A percentage literal in prose.
RATE_IN_PROSE = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s*(?:%|\bpercent\b)")

#: Phrasing that makes the number a requirement rather than a measurement.
TARGET_PHRASING = re.compile(
    r"\b(at least|at most|no more than|no fewer than|under|below|over|above|"
    r"minimum|maximum|min\.|max\.|target|require|required|requires|must|should|"
    r"goal|criteri|exit|threshold|budget|gate|gated|gates|retain|retains|remain|"
    r"remains|stay|stays|within|gets? to|gate on)\b",
    re.I,
)

#: Phrasing that makes the number the name of a statistical method.
STATISTICAL_PHRASING = re.compile(
    r"\b(confidence|lower bound|upper bound|lcb|ucb|interval|clopper|pearson|"
    r"percentile|quantile|detection|significance|alpha level|two[- ]sided|"
    r"one[- ]sided|p95|p99|p50)\b",
    re.I,
)

#: Evidence that the sample size is stated alongside the rate.
N_PHRASING = re.compile(
    r"(\bn\s*=\s*\d|\b\d[\d,]*[-\s](prompt|case|clip|trial|sample|item|pair|"
    r"frame|run|record|negative|positive)s?\b|\bof\s+\d[\d,]*\b|"
    r"\b\d[\d,]*\s+(prompts|cases|clips|trials|samples|items|pairs|records)\b)",
    re.I,
)

#: Evidence that the comparison point is stated alongside the rate.
BASELINE_PHRASING = re.compile(
    r"\b(baseline|chance|prior|control|majority[- ]class|versus|vs\.?|"
    r"compared (to|with)|relative to|against a|against the|over a)\b",
    re.I,
)

#: JSON keys whose value is a published rate.
RATE_KEY = re.compile(r"(_rate|_fraction|agreement|consistency|accuracy|precision|recall)$", re.I)

#: JSON keys that are gates, not results.
GATE_KEY_PREFIX = re.compile(r"^(minimum_|maximum_|min_|max_|required_|target_|allowed_)", re.I)

#: JSON sibling keys that supply the n.
N_KEY = re.compile(r"(^n$|_count$|_checks$|_pairs$|_corruptions$|_trials$|_total$|^total_)", re.I)

#: JSON sibling keys that supply the baseline.
BASELINE_KEY = re.compile(r"(baseline|chance_level|control_rate)", re.I)


def _display(path: Path) -> str:
    """Repo-relative when the file is in the repository, absolute otherwise."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class Finding:
    """One published rate that is missing its n, its baseline, or both."""

    path: str
    locator: str
    rate: str
    missing: tuple[str, ...]
    context: str

    def __str__(self) -> str:
        return f"{self.path}:{self.locator} states {self.rate} without {' and '.join(self.missing)} -- {self.context}"


#: Files that publish rates without a baseline today, each with an owner and the
#: PR that retires it. A waiver is a recorded deficiency, not an exemption: the
#: recorded count is asserted exactly, so a waived file cannot quietly acquire a
#: new bare rate.
WAIVERS: dict[str, dict[str, Any]] = {
    "rigby-poc/docs/evidence/frozen-judge-calibration.json": {
        "findings": 6,
        "reason": (
            "schema_version 1.0, frozen 2026-08-09. Publishes six rates that each state "
            "an n but no baseline; two of them (pairwise_agreement, ab_order_consistency) "
            "have a chance baseline of 0.50 that is not written down. Regenerating it is "
            "plan 10's eval-report.v2.json, not a repair of the 1.0 artifact."
        ),
        "owner": "plan 10 PR 10f (conductor -- the only PR with model spend)",
        "retires_in": "eval-report.v2.json, plan 10 §9.7",
    },
    "rigby-poc/docs/research-and-roadmap.md": {
        "findings": 1,
        "reason": (
            "One externally cited result (MoVer, 95.1% on a 5,600-prompt synthetic "
            "benchmark) states its n but no baseline. It is someone else's number, so the "
            "honest fix is to cite the paper's baseline or mark it external -- not to "
            "invent one."
        ),
        "owner": "lane infra",
        "retires_in": "08c",
    },
}


def _iter_markdown(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.md")):
        if "node_modules" in path.parts or ".venv" in path.parts:
            continue
        if "plans" in path.parts:
            continue  # plans are specifications and are not committed
        yield path


def scan_prose(path: Path) -> list[Finding]:
    """Published rates in a markdown file that lack their n or their baseline."""

    findings: list[Finding] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for match in RATE_IN_PROSE.finditer(line):
            if TARGET_PHRASING.search(line) or STATISTICAL_PHRASING.search(line):
                continue
            missing = tuple(
                label
                for label, pattern in (("its n", N_PHRASING), ("its baseline", BASELINE_PHRASING))
                if not pattern.search(line)
            )
            if missing:
                findings.append(
                    Finding(
                        path=_display(path),
                        locator=str(number),
                        rate=match.group(0),
                        missing=missing,
                        context=line.strip()[:120],
                    )
                )
    return findings


def scan_report(path: Path) -> list[Finding]:
    """Published rates in a committed JSON report that lack their n or baseline."""

    document = json.loads(path.read_text(encoding="utf-8"))
    findings: list[Finding] = []

    def visit(node: Any, pointer: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{pointer}[{index}]")
            return
        if not isinstance(node, dict):
            return
        siblings = set(node)
        has_n = any(N_KEY.search(key) for key in siblings)
        has_baseline = any(BASELINE_KEY.search(key) for key in siblings)
        for key, item in node.items():
            if isinstance(item, (dict, list)):
                visit(item, f"{pointer}.{key}")
                continue
            if GATE_KEY_PREFIX.match(key) or not RATE_KEY.search(key):
                continue
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                continue
            if not 0.0 <= float(item) <= 1.0:
                continue
            missing = tuple(
                label
                for label, present in (("its n", has_n), ("its baseline", has_baseline))
                if not present
            )
            if missing:
                findings.append(
                    Finding(
                        path=_display(path),
                        locator=f"{pointer}.{key}".lstrip("."),
                        rate=f"{item}",
                        missing=missing,
                        context=f"siblings: {sorted(siblings)}",
                    )
                )

    visit(document, "")
    return findings


def scan_repository() -> list[Finding]:
    """Every published rate in the repository that lacks its n or its baseline."""

    findings: list[Finding] = []
    for path in _iter_markdown(PROJECT_ROOT):
        findings.extend(scan_prose(path))
    for name in ("README.md",):
        candidate = REPO_ROOT / name
        if candidate.exists():
            findings.extend(scan_prose(candidate))
    evidence = PROJECT_ROOT / "docs" / "evidence"
    if evidence.is_dir():
        for path in sorted(evidence.rglob("*.json")):
            findings.extend(scan_report(path))
    return findings


def unwaived(findings: list[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.path not in WAIVERS]


def waived_counts(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {path: 0 for path in WAIVERS}
    for finding in findings:
        if finding.path in counts:
            counts[finding.path] += 1
    return counts
