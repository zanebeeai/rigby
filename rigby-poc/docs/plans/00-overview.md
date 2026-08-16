# Eval rebuild — overview and dependency map

Status: proposed, not started.
Policy: **no human evaluation at any point**, in the generation loop or in calibration.

This directory holds one plan per change. Nine of them are enabling work; the tenth is the
eval suite itself, which is not buildable until the others land.

---

## 1. Why the enabling work exists

The eval design is not limited by ideas about what to measure. It is limited by six
concrete properties of the current system:

| Limitation | Consequence | Lifted by |
| --- | --- | --- |
| No agent transcript — request payloads, failed attempts, and all latency are discarded | Trajectory evals impossible; a bad judgment cannot be diagnosed after the fact | [01](01-observability-and-transcript.md) |
| Metrics computed inside an 8,446-line compiler | A check cannot run on an arbitrary clip, cannot be unit-tested, and cannot be added without editing the largest file in the repo | [02](02-analysis-layer.md) |
| No committed clip data | Every eval needs a key, a server, a browser, and minutes; the checks themselves have no regression tests | [03](03-golden-corpus.md) |
| No anatomical frame; limits are direction-blind scalars over 8 of 52 bones | Per-DOF range-of-motion checks cannot be written at all | [04](04-anatomical-frame.md) |
| Evidence can silently depict the wrong body; no render provenance | A grader cannot be calibrated on evidence that may be invalid | [05](05-capture-integrity.md) |
| Corruption library is 28 hand/wrist-only, all glaring | Under a no-human policy this is the primary ground-truth source, and it covers a fraction of the body | [06](06-mutation-library.md) |

Plus two that are quality-of-life but compound with everything: the judge cannot be
calibrated per-dimension while it is one 115-line prompt fed deterministic diagnostics
([07](07-judge-harness.md)), and thresholds live in three config files and ~20 code sites
with real contradictions between them ([08](08-threshold-consolidation.md)).

## 2. How ground truth works without humans

The whole design turns on this. Labels are **constructed**, from four independent sources:

1. **Mutation** — apply a known perturbation to a known-good clip; by construction the
   result is worse along that axis. Graded severity yields a detection threshold rather
   than a pass/fail.
2. **Cross-prompt discrimination** — ask which of two prompts a clip depicts. Labels are
   free, the baseline is exactly 0.50, and it measures semantics directly.
3. **Deterministic agreement** — for anything the deterministic layer can measure, it *is*
   ground truth for the model graders.
4. **Stability and ablation controls** — no labels needed; a grader whose score does not
   move when handed the wrong prompt is not reading the prompt.

This is more rigorous than the ten human pairs it replaces, and it regenerates for free on
every model upgrade. What it does **not** establish is human preference — see
[10](10-eval-redesign.md) §2.5. That limitation is stated in every report rather than
papered over.

## 3. The plans

| # | Plan | PRs | Spend | Blocks |
| --- | --- | --- | --- | --- |
| 01 | [Observability and transcript](01-observability-and-transcript.md) | 4 | none | trajectory evals |
| 02 | [Analysis layer](02-analysis-layer.md) | 5 | none | 03, 04, 06, 10 |
| 03 | [Golden corpus](03-golden-corpus.md) | 2 | none | 06, 09, 10 |
| 04 | [Anatomical frame and ROM](04-anatomical-frame.md) | 3 | none | 06, 10 |
| 05 | [Capture integrity](05-capture-integrity.md) | 1 | none | 07, 10 |
| 06 | [Mutation library](06-mutation-library.md) | 3 | none | 10 |
| 07 | [Judge harness](07-judge-harness.md) | 4 | low | 10 |
| 08 | [Threshold consolidation](08-threshold-consolidation.md) | 3 | none | 10 |
| 09 | [CI and tiering](09-ci-and-tiering.md) | 3 | none | — |
| 10 | [Eval redesign](10-eval-redesign.md) | 7 | 10f only | — |

**35 PRs. Exactly one — 10f — spends meaningfully on models.**

## 4. Dependency graph

```
07a (semantic_match bug) ────────── independent, do first, ~1 hour

01 observability ──────────────────────────────┐
                                               │
02a analysis skeleton ──┬── 02b..02e           ├──> 10 eval redesign
                        │                      │
                        ├── 03 corpus ──┬──────┤
                        │               │      │
                        └── 04 anatomy ─┴─ 06 ─┤
                                               │
05 capture ─────────────── 07 judge ───────────┤
                                               │
08 thresholds ─────────────────────────────────┘
```

## 5. Suggested order

**Immediately, independent of everything else** — one roughly one-hour change:

- **07a** — enforce `semantic_match` in the acceptance rule. It is documented in
  `acceptance_criteria.yaml` and has never been enforced in `judge.py:893-895`, so a clip
  scoring 1 on semantics is auto-accepted today.

Note on test speed: the full suite is **79 s** measured (`344 passed, 1:19.39 total`), and
coverage is not enabled by default. There is no speed problem to fix first — see
[09](09-ci-and-tiering.md) §1.1.

**Then, in rough order:** 01a → 02a → 03a → 04a → 09b/09c → 04b → 06a → 08a → 10a.

Note that **10a — the degenerate-predictor regression test — should land early**, not with
the rest of 10. It is pure, costs nothing, and is the permanent guard against the class of
bug that invalidated the previous calibration. Landing it early means every later grader is
built against a scoring function that already provably rejects constant predictors.

## 6. Cross-platform note

Development is split between macOS and Windows. Three items carry real cross-platform risk
and are called out in their plans:

- MuJoCo hash equality in the golden corpus ([03](03-golden-corpus.md) §6.1) — determinism
  is guaranteed per platform and version, not across them.
- Path handling in capture output directories.
- The `PermissionError` retry ladder in `io_utils.py`, which exists precisely because
  Windows differs.

[09c](09-ci-and-tiering.md) puts Windows in the CI matrix from day one, which is the only
mechanism that will catch these.

## 7. Prior art absorbed

An earlier `docs/judge-calibration-rebuild-plan.md` diagnosed the calibration failure
correctly. It has been retired, with everything load-bearing carried into
[10](10-eval-redesign.md):

| From the old plan | Now lives in |
| --- | --- |
| The diagnosis: every scored item shares one label, so a constant predictor scores 0.9 against a 0.8 gate | [10](10-eval-redesign.md) §2, §5.4 |
| Gate on a 95% lower bound against a *computed* majority-class baseline | [10](10-eval-redesign.md) §5.3 |
| Pairwise strata A / B / C, where B excludes the base entirely | [10](10-eval-redesign.md) §5.4 |
| The degenerate-predictor regression test | [10](10-eval-redesign.md) §5.5, promoted to PR 10a |
| Repoint `goal_audit.py:79` off the machine-local path | [03](03-golden-corpus.md) §3.6 |
| Delete `MINIMUM_BASE_PREFERENCE` | [10](10-eval-redesign.md) §9 |
| The fixed blinding seed at `flywheel.py:2417` | [07](07-judge-harness.md) §3.4 |

What was **not** carried over is exactly the human-rater machinery — multi-rater κ, the
labelling bundle, suite validity established by human agreement. §2 above replaces all of
it with constructed ground truth.
