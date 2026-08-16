# Implementation tracking

Living status for the eval rebuild. One section per plan. The plans in this directory are
the specification; this file is the state.

## How to update this file

Read this before editing. The value of this document is that it stays short enough to read
in one sitting — a tracker nobody reads is worse than no tracker.

- **Status only.** Change a status cell, add a one-line note, record a decision. Do not
  narrate work performed.
- **One line per note.** If it needs a paragraph, it belongs in the plan document, not
  here. Edit the plan and link the section.
- **Findings that change the plan go in the plan.** Then add a one-line pointer here so
  the change is discoverable. A finding that does *not* change the plan does not belong
  here at all.
- **Decisions get recorded once**, in the owning plan's section, with the date and the
  answer. Plans mark open decisions as "**Decide before X**"; resolve them here.
- **Never add a section.** Ten plans, ten sections, fixed. Cross-cutting notes go in
  §Cross-cutting at the bottom.
- **Prune on completion.** When a plan reaches `done`, collapse its PR table to a single
  line and delete its resolved decisions. The plan document retains the detail.

Status values: `not started` · `in progress` · `blocked` · `in review` · `done`

---

## At a glance

| Plan | PRs | Status | Branch / owner | Blocking |
| --- | --- | --- | --- | --- |
| [01 Observability](01-observability-and-transcript.md) | 0/4 | not started | — | trajectory evals |
| [02 Analysis layer](02-analysis-layer.md) | 0/5 | not started | — | 03, 04, 06, 10 |
| [03 Golden corpus](03-golden-corpus.md) | 0/2 | not started | — | 06, 09, 10 |
| [04 Anatomical frame](04-anatomical-frame.md) | 0/3 | not started | — | 06, 10 |
| [05 Capture integrity](05-capture-integrity.md) | 0/1 | not started | — | 07, 10 |
| [06 Mutation library](06-mutation-library.md) | 0/3 | not started | — | 10 |
| [07 Judge harness](07-judge-harness.md) | 1/4 | in progress | `eval/07a-acceptance` | 10 |
| [08 Thresholds](08-threshold-consolidation.md) | 0/3 | not started | — | 10 |
| [09 CI and tiering](09-ci-and-tiering.md) | 0/3 | not started | — | — |
| [10 Eval redesign](10-eval-redesign.md) | 0/7 | not started | — | — |

**Slice 1** (concurrent): 07a · 01a · 02a · 03a · 10a, optionally 09c.
Session prompts and the file-ownership map: [SLICE-1-SESSIONS.md](SLICE-1-SESSIONS.md).

---

## 01 — Observability and transcript

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 01a | `observability.py`, `transcript.py`, tests; wired nowhere | not started | slice 1 |
| 01b | Instrument judge + planner call sites; `config.json` | not started | |
| 01c | Instrument flywheel stages and capture; unify CLI/API run roots | not started | |
| 01d | Compaction tool and retention policy | not started | |

**Open decisions**
- §8.1 Build a minimal tracer or adopt OpenTelemetry — *decide before 01a*. Unresolved.

**Findings**
- _(none yet)_

---

## 02 — Analysis layer

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 02a | Skeleton, `CheckResult`, `AnalysisContext`, registry, equivalence harness; move pure helpers + `quality.py` | not started | slice 1, critical path |
| 02b | Persist IK support targets; port full-body's 13 action blocks | not started | bulk of metric mass |
| 02c | Composite + gesture/strike/grab; converge `arm_landmarks` on `RigKinematics` | not started | |
| 02d | Object interaction + handoff | not started | numeric drift risk |
| 02e | Sequence | not started | hardest path |

**Open decisions**
- §6.3 Does the compiler keep populating `ClipResult.metrics`? — *decide before 02a*.
  Plan recommends yes. Unresolved.

**Findings**
- _(none yet)_

---

## 03 — Golden corpus

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 03a | Case format, loader, `bless` CLI, `freeze.py`, 12 cases | not started | slice 1 |
| 03b | Remaining ~28 cases incl. known-bad and MuJoCo; repoint `goal_audit` | not started | |

**Open decisions**
- §6.3 Commit slim clips for every case, or programs only? — plan recommends programs
  only plus clips for MuJoCo and slow cases. Unresolved.

**Findings**
- _(none yet)_

---

## 04 — Anatomical frame and ROM

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 04a | `AnatomicalFrame` derivation, `decompose`, frame-calibration tests | not started | needs 02a |
| 04b | `rom.v1.json` for 52 bones; checks in report-only mode; mirroring | not started | |
| 04c | Enforcement per DOF after distribution review; delete duplicated rest state | not started | |

**Open decisions**
- §6.3 Box limits only, or add shoulder/hip coupling? — *decide before 04b*. Plan
  recommends box first. Unresolved.

**Findings**
- _(none yet)_

---

## 05 — Capture integrity

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 05 | Fail-loud asset load, render provenance, pose/pixel hash split, seek hook, batching, timeouts, retention | not started | |

**Open decisions**
- §6.2 Add a `--deterministic-render` mode (AA off, hard shadows)? — *decide before 06*.
  Plan recommends yes, as a second mode. Unresolved.

**Findings**
- _(none yet)_

---

## 06 — Mutation library

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 06a | `MutationSpec`, sweeps, composition; port the existing 28 specs as the severe tier | not started | |
| 06b | anatomy / timing / signal / clipping families | not started | |
| 06c | contact / balance / semantic families; detection matrix | not started | |

**Open decisions**
- §6.3 Uniform severity sweep or adaptive bisection? — *decide before 06c*. Plan
  recommends bisection (~3 renders vs 7). Unresolved.

**Findings**
- _(none yet)_

---

## 07 — Judge harness

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 07a | Enforce `semantic_match`; content-derived blinding seed; route `recommend_repair` | done | `eval/07a-acceptance`; 364 pass |
| 07b | Grader split behind a flag; per-family prompt fragments | not started | |
| 07c | Claim-based output + aggregation; `cannot_tell` | not started | |
| 07d | Remove diagnostics from prompts; explicit decision layer | not started | must precede 10f |

**Open decisions**
- §6.1 Does production keep one combined grader while calibration uses split ones? —
  *decide before 07b*. Unresolved.
- §6.3 Does the timing dimension survive calibration, or get deleted? — answered by 10f.

**Findings**
- §1.1 corrected: `judge.py:893-895` is an escalation predicate, not the acceptance
  decision — `accept` is model-self-reported and still unenforced until 07d's §3.3
  decision layer.
- §1.4 corrected: two more positional seeds exist outside `flywheel.py`
  (`rerank_existing.py:53`, `select_structural_sweep.py:244`); §3.4's code sketch did not run.

---

## 08 — Threshold consolidation

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 08a | `thresholds.v1.json` + loader + schema; port config values | not started | |
| 08b | Repoint call sites; resolve contradictions 2, 4, 6, 7 | not started | |
| 08c | `camera.v1.json` + frontend generation; resolve 3, 5; orphan guards | not started | |

**Open decisions**
- §6.2 Who derives `by_family` kinematic ceilings? — plan recommends shipping `default`
  only, marked PROVISIONAL, and deriving per-family values in 10 from corpus percentiles.
  Unresolved.

**Findings**
- _(none yet)_

---

## 09 — CI and tiering

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 09a | Document invocations; pin coverage to nightly with `sysmon` | not started | low value |
| 09b | `conftest.py`, session fixtures, markers | not started | enables sub-30 s fast tier |
| 09c | CI workflow, macOS + Windows matrix | not started | slice 1 (optional) |

**Open decisions**
- §6.2 Windows CI on every PR or only on `src`/`evals`/`config` changes? — plan
  recommends the latter. Unresolved.

**Findings**
- Suite is **79 s** measured (`344 passed, 1:19.39 total`); coverage is not on by default.
  No speed emergency — see plan §1.1.

---

## 10 — Eval redesign

| PR | Scope | Status | Note |
| --- | --- | --- | --- |
| 10a | `calibration_stats.py`, gate function, degenerate-predictor test | not started | slice 1, land early |
| 10b | Physics layer: CoM and everything it unlocks | not started | |
| 10c | Signal layer | not started | |
| 10d | Calibration drivers; `--offline-scoring` | not started | |
| 10e | LLM graders | not started | |
| 10f | VLM calibration run + report; retire `calibrate_judge.py` | not started | only PR with real spend |
| 10g | Trajectory evals | not started | |

**Open decisions**
- §10.3 What replaces "the judge is calibrated" as a release claim? — *decide before 10f*.
  Plan recommends reporting deterministic pass rate and grader validity separately, never
  combined. Unresolved.
- §10.4 If production and calibration use different graders, how is transfer stated?

**Findings**
- _(none yet)_

---

## Cross-cutting

**Platform.** Development is split macOS / Windows. Anything touching MuJoCo hashes,
filesystem paths, or `io_utils.py` atomic writes needs verification on both before merge.

**Repo hygiene.** Additive changes only where possible; the working tree is shared. No
Claude attribution in commits or PRs.

**Superseded.** `docs/judge-calibration-rebuild-plan.md` was retired; its content is mapped
into the current plans in [00-overview.md](00-overview.md) §7.
