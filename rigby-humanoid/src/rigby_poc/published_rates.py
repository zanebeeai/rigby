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
* A **prevalence** is not a published rate. "87-94% of frames carry more than 5
  degrees of abduction" describes how often a thing occurs in a population; there
  is no prior state it improved on and no null model it beats, so demanding a
  baseline invites a fabricated comparison ("against a baseline of 0% of frames").
  It still needs its n, and the scanner still requires one. Named by lane
  ``judge`` as the exclusion most likely to tempt someone into inventing a
  baseline to silence the guard, so it is documented as deliberate rather than
  left to look incidental.
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

1a. **And ``success`` itself is partly one of them.** ``finger_assertions`` is a
   tautology for every hand shape but ``HANG_TEN``: the else branch is
   ``{"shape_defined": len(actual_curls) == 5}`` and the curl helper iterates a
   five-entry table, so it returns exactly five or raises -- it cannot be False.
   ``compile_motion`` folds ``all(assertions.values())`` into ``success``, so it
   reads as a load-bearing hand-shape gate and is not one. Measured by lane
   ``analysis`` at 5 of 6 gesture/strike/grab fixture cases.

   This is the worst version of the hole, because ``success`` is the field a
   published rate is most likely to be computed over. **A "success rate" is not
   a rate over successes**; for those shapes it is a rate over a constant folded
   into a conjunction. n and baseline are both correct and the quantity is not
   what its name says.

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

4. **An n that is the wrong quantity.** "99.8% of a 5392 ms run across 44 spans"
   satisfies every check here: it names a number, and 44 looks like an n. But 44
   counts spans *within* one run, not runs -- the claim is *this run was 99.8%
   stage-covered*, not *runs are 99.8% stage-covered*, and only the second is a
   rate. No scanner can separate those from the text; it would have to know what
   the denominator is a denominator of. Named by lane ``judge``.

The common shape of the first three is that n and baseline describe the *sample*,
and none of them is a defect of the sample. The fourth is different: the sample
description is present and is about the wrong thing.

**So this catches rates MISSING their n and baseline. It does not catch a rate
whose n is the wrong quantity.** That is worth stating plainly rather than
letting the name imply the stronger property.
"""

from __future__ import annotations

import json
import math
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

#: An inline code span: a quotation of output or an identifier, never a claim.
INLINE_CODE = re.compile(r"`[^`\n]*`")

#: JSON keys whose value is a published rate.
#: A key whose name says its value is a published rate.
#:
#: **Typed statistical output is out of scope, and that is a decision rather than
#: an oversight.** `calibration_stats.ProportionResult.to_dict()` puts its number
#: under `estimate`, which matches nothing here, so no `ProportionResult` in the
#: repository is ever scanned. The reasoning for leaving it that way: the type
#: cannot be constructed without its `n` and its `lower_bound_95`, so scanning it
#: would re-check what `__post_init__` already enforces.
#:
#: **But the exemption is partial, and the uncovered half is the load-bearing
#: one.** `ProportionResult` guarantees an n and a bound. It does not carry a
#: baseline, and this guard requires both -- rename `estimate` to
#: `detection_rate` and the scan reports "states 0.87 without its baseline".
#: A constant predictor posts a fine rate with a large n and a tight interval;
#: only a baseline shows it is chance. So typed output is exempt from the n half
#: because the type enforces it, and exempt from the baseline half because
#: nothing enforces it anywhere. Plan 10 §10.5 is where that gets closed, not
#: here.
#:
#: Widening this pattern to reach `estimate` would flag every `ProportionResult`
#: for a missing baseline. Measured 2026-08-25: zero such dicts in committed
#: JSON, so the cost today is zero and the decision is about what
#: `eval-report.v2.json` should be required to carry. Raised by lane `judge`,
#: who found the exemption and asked that it be recorded either way.
RATE_KEY = re.compile(r"(_rate|_fraction|agreement|consistency|accuracy|precision|recall)$", re.I)

#: JSON keys that are gates, not results.
GATE_KEY_PREFIX = re.compile(r"^(minimum_|maximum_|min_|max_|required_|target_|allowed_)", re.I)

#: JSON sibling keys that supply the n.
N_KEY = re.compile(r"(^n$|_count$|_checks$|_pairs$|_corruptions$|_trials$|_total$|^total_)", re.I)

#: JSON sibling keys that supply the baseline.
BASELINE_KEY = re.compile(r"(baseline|chance_level|control_rate)", re.I)


def _display(path: Path) -> str:
    """Repo-relative POSIX when the file is in the repository, absolute otherwise.

    ``as_posix`` is not cosmetic here: ``WAIVERS`` is keyed by these strings, and
    a native ``str()`` yields backslashes on Windows, so every waiver silently
    stopped matching and the guard failed there while passing on macOS. Found by
    the Windows CI job -- the exact class of defect that job exists to catch, in
    the guard that was meant to catch defects.
    """

    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


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
    "rigby-humanoid/docs/evidence/frozen-judge-calibration.json": {
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
    "rigby-humanoid/docs/research-and-roadmap.md": {
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


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """Blank-line-delimited blocks, with the 1-based line each starts on.

    The unit of judgement is the paragraph, not the line. Prose wraps, so a rate
    and the n that qualifies it routinely land on adjacent lines -- and a
    line-scoped check reports that as a bare rate, which is a false positive in
    the guard rather than a defect in the writing. It found three in this
    repository's own documentation before this changed.

    Widening the window does weaken the check: a paragraph offers more places for
    an unrelated number to look like an n. That is the right trade, because the
    alternative trains the reader to dismiss it.
    """

    blocks: list[tuple[int, str]] = []
    start = 1
    current: list[str] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            if current:
                blocks.append((start, "\n".join(current)))
                current = []
            continue
        if fenced or line.startswith("    ") and not current:
            # A fenced or indented block is a transcript, not a claim. `97% cpu`
            # in pasted `time` output is not a published rate, and flagging it
            # teaches the reader to skim past real findings.
            continue
        if line.strip():
            if not current:
                start = number
            # An inline code span is a quotation, not a claim -- `[100%]` is
            # pytest's progress output. Blanked rather than removed so column
            # offsets, and therefore reported line numbers, stay correct. Applied
            # per line and after the fence check, because a fence marker is
            # itself backticks.
            current.append(INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line))
        elif current:
            blocks.append((start, "\n".join(current)))
            current = []
    if current:
        blocks.append((start, "\n".join(current)))
    return blocks


def scan_prose(path: Path) -> list[Finding]:
    """Published rates in a markdown file that lack their n or their baseline."""

    findings: list[Finding] = []
    for start, block in _paragraphs(path.read_text(encoding="utf-8")):
        if TARGET_PHRASING.search(block) or STATISTICAL_PHRASING.search(block):
            continue
        missing = tuple(
            label
            for label, pattern in (("its n", N_PHRASING), ("its baseline", BASELINE_PHRASING))
            if not pattern.search(block)
        )
        if not missing:
            continue
        for match in RATE_IN_PROSE.finditer(block):
            offset = block[: match.start()].count("\n")
            findings.append(
                Finding(
                    path=_display(path),
                    locator=str(start + offset),
                    rate=match.group(0),
                    missing=missing,
                    context=" ".join(block.split())[:120],
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
            # A rate on the 0..1 scale and the same rate on the 0..100 scale are
            # the same published claim, and this filter used to check only the
            # first -- so `"detection_rate": 82.0` was skipped while
            # `"detection_rate": 0.82` was caught. A false negative in the
            # direction that hides, inside the guard for L3 gate item 3.
            # Measured before widening: zero rate-named keys sit outside 0..1 in
            # any committed JSON today, so this closes the hole ahead of
            # `eval-report.v2.json` rather than creating churn.
            #
            # Non-finite is still skipped, and that is not the same omission: an
            # `inf` upper bound is an honest reading of "never established
            # anywhere in the sweep" (10d's `Interval`), and a bound is not a
            # rate. Flagged by lane `judge`, which is what prompted this look.
            value = float(item)
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
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
