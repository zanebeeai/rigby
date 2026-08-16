# Slice 1 — worktree session prompts

Five concurrent sessions plus one optional. Each owns a disjoint set of files, so they can
run simultaneously without merge conflicts. Copy one block per fresh Claude Code session,
started from the repo root.

File ownership (do not cross these lines):

| Session | Owns |
| --- | --- |
| 07a | `src/rigby_poc/judge.py`, `evals/flywheel.py` |
| 01a | new files only: `src/rigby_poc/observability.py`, `transcript.py` |
| 02a | `src/rigby_poc/compiler.py`, `quality.py`, new `src/rigby_poc/analysis/` |
| 03a | new files only: `evals/corpus/` |
| 10a | new files only: `evals/calibration_stats.py` |
| 09c | new files only: `.github/workflows/` |

Everyone adds their own new files under `tests/`. Nobody creates `tests/conftest.py` in
slice 1 — that is PR 09b, deliberately excluded to avoid a shared-file conflict.

---

## Session 1 — 07a: acceptance rule and blinding seed

```
Work on PR 07a in an isolated worktree.

Set up:
  cd "/Users/tonypan/Developer/03 - Startups/rigby"
  git worktree add "../rigby-wt/07a-acceptance" -b eval/07a-acceptance
  cd "../rigby-wt/07a-acceptance/rigby-poc"
  uv sync --extra dev

No API key or browser is needed — this work is fully offline.

Read rigby-poc/docs/plans/07-judge-harness.md, sections 1.1, 1.4, 3.4, 3.6, and the 07a
row in section 4. Also read docs/plans/TRACKING.md before you start and again before you
finish.

Scope, exactly three changes:
1. Enforce semantic_match in the acceptance rule. acceptance_criteria.yaml requires
   semantic_match >= 4 but judge.py:893-895 gates only overall, anatomical_naturalness,
   and gesture_recognizability, so a clip scoring 1 on semantics is auto-accepted today.
2. Replace the fixed blinding seed at evals/flywheel.py:2417 (70_000 + round_index) with
   one derived from content, per plan section 3.4. Do the same for the tournament seed.
3. Route judge.py recommend_repair (1125-1157) through _routed_parse so it gains attempts,
   routing, retry, and fallback like every other call.

Do not touch any other file. Other sessions own compiler.py, quality.py, and all new
modules.

Verify the plan's claims yourself before acting on them — read the actual code at every
line reference and confirm the described behaviour. The plan was written from a prior
exploration and may be wrong or stale. If it is wrong, correct the plan document as part
of this PR and note it under Findings in TRACKING.md.

Then run your own exploration of anything the plan does not cover: how acceptance
interacts with the escalation logic around judge.py:898, and whether any test currently
depends on the permissive acceptance rule.

Required tests: tests/test_acceptance_rule.py asserting a score with semantic_match=1 and
everything else at 5 is rejected; tests/test_blinding_seed.py asserting the seed is stable
for identical content, differs across prompts, and that recipe k is NOT always in slot A
in round 0.

Done when: uv run pytest -q is green, the new tests fail against the old behaviour, and
TRACKING.md shows 07a as done with a one-line note. Commit on your branch. Do not push or
open a PR without asking. No Claude attribution in commit messages.
```

---

## Session 2 — 01a: observability and transcript foundation

```
Work on PR 01a in an isolated worktree.

Set up:
  cd "/Users/tonypan/Developer/03 - Startups/rigby"
  git worktree add "../rigby-wt/01a-observability" -b eval/01a-observability
  cd "../rigby-wt/01a-observability/rigby-poc"
  uv sync --extra dev

No API key or browser is needed — this work is fully offline.

Read rigby-poc/docs/plans/01-observability-and-transcript.md in full, and docs/plans/
TRACKING.md before you start and again before you finish.

Scope is 01a only: build src/rigby_poc/observability.py (Tracer, Span, NullTracer, ULID,
JSONL writer, contextvar parent resolution) and src/rigby_poc/transcript.py (the reader
API in section 4.2), plus their tests. Wire them into NOTHING. No call site changes — that
is 01b, and another session owns judge.py this week.

Create only new files. Do not modify any existing module.

Resolve the open decision in section 8.1 first — build a minimal tracer versus adopting
OpenTelemetry. The plan recommends building it (~300 lines, no dependency, writes exactly
the artifact the evals need). Confirm or overturn that with your own investigation, then
record the decision and its reasoning in TRACKING.md under plan 01.

Verify the plan's claims yourself before designing around them. In particular check
section 1.2 (that failed attempts are discarded at judge.py:831-836) and section 8.3 (that
PipelineRunStore._execute runs on a threading.Thread at pipeline.py:141, so a contextvar
set on the parent will not propagate). Both shape the design. If either is wrong, fix the
plan and note it.

Also run your own exploration of how the JSONL writer should interact with the existing
atomic_write_json in src/rigby_poc/io_utils.py, which has a Windows-specific
PermissionError retry ladder. This repo is developed on both macOS and Windows; do not
write file-handling code that only works on one.

Required tests: span nesting via contextvar including across a thread boundary; ULID
monotonicity; a truncated final JSONL line is skipped by the reader rather than fatal;
transcript reader over a committed fixture with $image rehydration.

Done when: uv run pytest -q is green, the transcript format is exercised end to end by
tests even though nothing emits one yet, and TRACKING.md shows 01a done plus the 8.1
decision resolved. Commit on your branch. Do not push or open a PR without asking. No
Claude attribution in commit messages.
```

---

## Session 3 — 02a: analysis layer skeleton (critical path)

```
Work on PR 02a in an isolated worktree. This is the critical path for the whole eval
rebuild — it blocks plans 03, 04, 06 and 10.

Set up:
  cd "/Users/tonypan/Developer/03 - Startups/rigby"
  git worktree add "../rigby-wt/02a-analysis" -b eval/02a-analysis
  cd "../rigby-wt/02a-analysis/rigby-poc"
  uv sync --extra dev

No API key or browser is needed — this work is fully offline.

Read rigby-poc/docs/plans/02-analysis-layer.md in full, and docs/plans/TRACKING.md before
you start and again before you finish.

Scope is 02a only: create the src/rigby_poc/analysis/ package with CheckResult,
AnalysisContext, the registry, and the equivalence harness; then move the already-pure
helpers into it and re-export from their old homes so nothing downstream breaks. The
helpers are listed in section 1.2: compiler.py _safety_metrics (537),
_parallel_forearm_metrics (4777), _intra_hand_contact_metrics (4950),
_semantic_cycle_metrics (5172), and all of quality.py.

Do NOT port the per-action metric blocks. That is 02b through 02e.

This is a move, not a redesign. No metric definition or threshold value changes.

Resolve the open decision in section 6.3 first — whether the compiler keeps populating
ClipResult.metrics. The plan recommends yes (the compiler calls the analyzer and stores
the result), because every downstream consumer depends on it. Confirm or overturn, then
record it in TRACKING.md under plan 02.

The equivalence harness is the safety net that makes this landable, and it needs reference
metrics. Do NOT depend on the golden corpus — another session is building it concurrently.
Instead, freeze the CURRENT metric output for about 12 representative programs into a test
fixture in this PR, and have the harness assert byte-identical output against it. The
corpus supersedes this fixture later.

Verify the plan's claims yourself before relying on them. Specifically: that every metric
block is already a post-hoc pass after the last frames.append (section 1.1 table), that
RigKinematics in kinematics.py:51 is generation-free and covers all 52 canonical bones
(section 1.3), and the two traps in section 1.5 — that store.py:49 persists the
pre-override program so an analyzer must call compiler.apply_overrides, and that
phase_ranges must be read from metrics["phase_ranges_s"] rather than re-derived.

Then run your own exploration of anything the plan does not cover, especially import
cycles between the new analysis package, compiler.py, and quality.py.

Required tests: the equivalence harness; unit tests for at least two moved checks driven
by hand-built synthetic clips; and a registry-coverage test that fails when a BodyAction
or ObjectAction enum member has no registered analyzer.

Done when: uv run pytest -q is green, analysis.analyze runs with no server, browser, or
key, output is byte-identical to today for the fixture programs, and TRACKING.md shows 02a
done plus the 6.3 decision resolved. Commit on your branch. Do not push or open a PR
without asking. No Claude attribution in commit messages.
```

---

## Session 4 — 03a: golden corpus

```
Work on PR 03a in an isolated worktree.

Set up:
  cd "/Users/tonypan/Developer/03 - Startups/rigby"
  git worktree add "../rigby-wt/03a-corpus" -b eval/03a-corpus
  cd "../rigby-wt/03a-corpus/rigby-poc"
  uv sync --extra dev

No API key or browser is needed — this work is fully offline, and that is the point of the
corpus.

Read rigby-poc/docs/plans/03-golden-corpus.md in full, and docs/plans/TRACKING.md before
you start and again before you finish.

Scope is 03a only: the case format, loader, `bless` CLI, freeze.py, and the first 12 cases
covering gesture, strike, composite and full-body. The remaining ~28 cases, the known-bad
cases and the MuJoCo cases are 03b.

Create only new files under evals/corpus/ plus tests. Do not modify compiler.py,
quality.py, or judge.py — other sessions own those this week.

Key design point to verify before building: compilation is claimed to be deterministic,
so the corpus stores programs and recompiles at test time rather than storing clips.
Verify that claim yourself — compile the same program twice in separate processes with
different PYTHONHASHSEED values and compare frame hashes. If it does not hold, the whole
case format changes, so establish it first.

Resolve the open decision in section 6.3 — programs only, or committed slim clips too. The
plan recommends programs only, plus stored clips for MuJoCo and slow cases. Measured
sizes: clip.json is 1140 KB raw, 17 KB as gzipped slim quaternions rounded to 1e-6. Record
the decision in TRACKING.md under plan 03.

Note the cross-platform hazard in section 6.1: MuJoCo determinism holds per platform and
version, not across them. This repo is developed on both macOS and Windows. 03a avoids
MuJoCo cases, but design expected.json so per-platform hashes can be added in 03b without
a format change.

Then run your own exploration of which 12 cases give the best check coverage. The 20
review_clip cases in evals/fixtures/planner_supported.json (ids s01-s20, with pairwise
unique motion_profile blocks) are the natural seed set and are already referenced by
config/motion_quality_reference.json.

Required tests: every case recompiles to its recorded motion_sha256; a coverage test that
fails when an Intent or BodyAction has no case; a test asserting corpus load and analysis
perform no network I/O and start no browser.

Done when: uv run pytest -q is green, 12 cases are committed and reproduce, `bless`
produces a readable diff, and TRACKING.md shows 03a done plus the 6.3 decision resolved.
Commit on your branch. Do not push or open a PR without asking. No Claude attribution in
commit messages.
```

---

## Session 5 — 10a: calibration statistics and the degenerate-predictor guard

```
Work on PR 10a in an isolated worktree. This lands early rather than with the rest of plan
10, because it is the permanent guard against the bug class that invalidated the previous
calibration.

Set up:
  cd "/Users/tonypan/Developer/03 - Startups/rigby"
  git worktree add "../rigby-wt/10a-calibration-stats" -b eval/10a-calibration-stats
  cd "../rigby-wt/10a-calibration-stats/rigby-poc"
  uv sync --extra dev

No API key is needed — everything here is pure functions with zero model calls, by design.

Read rigby-poc/docs/plans/10-eval-redesign.md sections 2, 5.2, 5.3, 5.4 and 5.5, plus the
10a row in section 8. Read docs/plans/00-overview.md section 7 for what the retired
calibration plan contributed. Read docs/plans/TRACKING.md before you start and again
before you finish.

Scope is 10a only: build evals/calibration_stats.py (Clopper-Pearson one-sided lower
bound, majority_class_baseline, balanced_accuracy, cohens_kappa, matthews_corrcoef, and a
ProportionResult record carrying estimate, lower_bound_95, n and successes), plus the gate
function from section 5.3, plus the degenerate-predictor regression test.

Create only new files. Do not modify judge.py, calibrate_judge.py, or judge_human_review.py
— rewriting those is 10f, and another session owns judge.py this week.

The gate function must exist even though no grader exists yet. That is deliberate: it means
every grader built later is built against scoring logic that already provably rejects
constant predictors.

The centrepiece is tests/test_calibration_rejects_degenerate_judges.py. Feed synthetic
judgment records from four degenerate predictors through the real gate and assert every one
FAILS:
  always accept  -> specificity lower bound 0
  always reject  -> sensitivity lower bound 0
  always base    -> absent from stratum B, directional accuracy ~0.5
  always first   -> directional accuracy ~0.5 by position balance
Also assert a balanced suite at boundary scores passes, that one fewer success in any arm
fails, and that n below the per-stratum minimum fails regardless of score.

scipy is already a dependency (scipy==1.18.0); use scipy.stats.beta for the interval.

Verify the statistical claims yourself rather than trusting the plan's numbers — check
clopper_pearson_lower against known reference values including the k=0 and k=n edges, and
sanity-check the sample-size table in the plan (for example that 32 negatives at a perfect
score yields a 95% lower bound of about 0.911, and 28 does not reach 0.90 even at
perfection). If any number in the plan is wrong, fix the plan and note it in TRACKING.md.

Then run your own exploration of how the gate should report a suite that is merely
underpowered versus one where the grader genuinely failed. Those must be distinguishable
in the output, not both just "false".

Done when: uv run pytest -q is green, all four degenerate predictors fail the gate, and
TRACKING.md shows 10a done. Commit on your branch. Do not push or open a PR without
asking. No Claude attribution in commit messages.
```

---

## Session 6 (optional) — 09c: CI with a Windows matrix

```
Work on PR 09c in an isolated worktree.

Set up:
  cd "/Users/tonypan/Developer/03 - Startups/rigby"
  git worktree add "../rigby-wt/09c-ci" -b eval/09c-ci
  cd "../rigby-wt/09c-ci/rigby-poc"
  uv sync --extra dev

Read rigby-poc/docs/plans/09-ci-and-tiering.md, sections 1.5, 3.4, 3.5 and 6, and
docs/plans/TRACKING.md before you start and again before you finish.

Scope is 09c only: a GitHub Actions workflow running the Python suite and the frontend
build on both macOS and Windows. Do not add markers or a conftest.py — that is 09b, and
skipping it now avoids a shared-file conflict with other sessions running this week.

Create only new files under .github/. Do not modify any Python or TypeScript source.

Context worth verifying: the suite is 79 s and needs no server, browser, or API key —
every capture path is injected and every OpenAI client is faked in tests. Confirm that
before designing the workflow, because it determines whether CI needs secrets at all. It
should not.

Note this repo has no CI today and is developed on both macOS and Windows, so expect the
first Windows run to surface real failures — likely around filesystem paths in capture
output directories and the PermissionError retry ladder in src/rigby_poc/io_utils.py.
Finding those is the point. Report them; do not paper over them with retries or skips.

Resolve the open decision in section 6.2 — Windows on every PR, or only on changes to
src/, evals/ or config/. Record it in TRACKING.md under plan 09.

Done when: the workflow is committed, you have reasoned through what it will do on Windows,
and TRACKING.md shows 09c with its status and the 6.2 decision. Commit on your branch. Do
not push or open a PR without asking. No Claude attribution in commit messages.
```
