# PR 10 — The eval redesign

Status: proposed, not started.
Scope: the eval suite itself — deterministic check registry, LLM graders, VLM graders,
grader calibration without human labels, trajectory evals, and the reporting contract.

Depends on: **all of** [01](01-observability-and-transcript.md),
[02](02-analysis-layer.md), [03](03-golden-corpus.md), [04](04-anatomical-frame.md),
[05](05-capture-integrity.md), [06](06-mutation-library.md), [07](07-judge-harness.md),
[08](08-threshold-consolidation.md). [09](09-ci-and-tiering.md) is not a hard dependency
but makes this practical to iterate on.

Absorbs: the retired `docs/judge-calibration-rebuild-plan.md`. Its diagnosis and its
statistical mechanism are carried into §2, §5.3 and §5.4 here; its human-rater machinery is
ruled out by policy. See [00-overview.md](00-overview.md) §7 for the full mapping.

---

## 1. What the suite must answer

Four questions, in dependency order. Each must be answerable without a human in the loop.

1. **Is this motion anatomically and physically possible?** — deterministic, certain.
2. **Does it read as biological rather than interpolated?** — deterministic, quantitative.
3. **Does it depict what was asked?** — model graders, and the only place they are
   genuinely required.
4. **Did the pipeline make good decisions getting there?** — trajectory, from the
   transcript.

The current suite answers (1) narrowly, (2) not at all, (3) with an instrument that
cannot be shown to work, and (4) not at all.

---

## 2. The constraint that shapes everything: no human evaluation

Every grader is a measuring instrument, and an uncalibrated instrument is worse than none —
it produces confident numbers that encode nothing. Conventionally, labels come from humans.
That is ruled out.

**Ground truth therefore has to be constructed.** There are four independent sources, and
together they are arguably more rigorous than the ten human pairs the old calibration
used, because they regenerate for free on every model and prompt change.

### 2.1 Construction — mutation

Apply perturbation `T` to known-good clip `C`. By construction `T(C)` is worse than `C`
along `T`'s axis. Labels are free and exact. From [06](06-mutation-library.md), with
graded severity so the output is a **detection threshold**, not a pass/fail.

Covers: anatomy, timing, contact, balance, signal, clipping.

### 2.2 Construction — cross-prompt discrimination

Render the clip for prompt A. Ask the grader which of two prompts it depicts, A or B.
Ground truth is known by construction; the baseline is exactly 0.5 for a forced binary
choice.

Vary the distractor's semantic distance:

| Distance | Example pair | Measures |
| --- | --- | --- |
| Far | "throw a right jab" vs "do a push-up" | gross semantic competence |
| Near | "throw a right jab" vs "throw a right hook" | fine motion discrimination |
| Minimal | "throw a right jab" vs "throw a left jab" | laterality |

**This is the unlock for semantic grading without humans.** It directly measures what
`semantic_match` only gestures at, and it yields a discrimination curve over semantic
distance rather than a single number.

### 2.3 Agreement — the deterministic layer as oracle

For anything L0–L3 can measure, **the deterministic layer is ground truth**. Render clips
with known deterministic properties and check whether the anatomy and artifact graders
agree.

Two things fall out. Systematic disagreement localizes the VLM's perceptual blind spots.
And if a grader only ever agrees with the deterministic layer, **it is redundant and should
be deleted** — a result worth having.

### 2.4 Controls — stability and ablation

No labels needed at all:

- **Stability**: same input, N runs, report variance. An unstable grader with a good mean
  is unusable.
- **Order sensitivity**: flip rate across presentation orders.
- **Paraphrase invariance**: same clip, reworded prompt, same verdict.
- **Ablation response**: degrade the evidence (one view, fewer frames, scrambled prompt)
  and require scores to move in the expected direction. **A grader whose score does not
  change when given the wrong prompt is not reading the prompt** — the single most
  diagnostic control here, and it costs almost nothing.

### 2.5 What this does *not* establish — stated plainly

None of the above establishes human preference. It establishes that a grader is sensitive,
specific, stable, non-degenerate, and appropriately responsive to its inputs. That is an
**instrument-validity claim, not a claim that Rigby's output pleases users.**

Every published number must say so. The failure mode of the old calibration was not only
statistical — it was rhetorical, presenting a degenerate agreement rate as evidence of
quality. That must not recur under a different mechanism.

---

## 3. Deterministic suite

Checks come from [02](02-analysis-layer.md), thresholds from
[08](08-threshold-consolidation.md), anatomy from [04](04-anatomical-frame.md). This PR
adds the layers that do not exist anywhere yet.

### 3.1 New: physics (L2)

Built on a **center of mass** derived from standard segment mass fractions applied to
`RigKinematics.canonical_positions` — one new primitive that unlocks the whole layer.

| Check | Assertion |
| --- | --- |
| `physics.balance.margin` | CoM ground projection inside the support polygon |
| `physics.ballistic.com` | CoM follows a parabola at g during flight |
| `physics.momentum.airborne` | angular momentum conserved while airborne |
| `physics.contact.foot_skate` | no horizontal motion while a foot is in contact |
| `physics.contact.root_consistency` | root travel explainable by foot contacts |
| `physics.ground.penetration` | no collider below y = 0 |
| `physics.gait.duty_factor` | ≈0.6 walk, <0.5 run with a flight phase |
| `physics.gait.cadence_speed` | speed = step length × cadence, within human bounds |

Foot skate and duty factor exist today only as per-family bespoke checks inside individual
primitives; lifting them to a general layer means a new locomotion primitive inherits them
instead of having to reimplement them.

### 3.2 New: signal quality (L3)

The quantitative naturalness proxies — the part usually assumed to need a model.

| Check | Measures | Catches |
| --- | --- | --- |
| `signal.min_jerk` | correlation of end-effector speed profile with the bell-shaped ideal | constant-velocity interpolation — the biggest tell of procedural motion |
| `signal.sparc` | spectral arc length; duration- and noise-robust | steppy, jittery, over-damped motion |
| `signal.power_law` | angular velocity ∝ curvature^(−1/3) on curved paths | paths geometrically right but moving unnaturally along them |
| `signal.dead_limb` | per-bone rotational variance while the body is active | frozen mannequin limbs — very cheap, very high yield |
| `signal.secondary` | spine and head response to limb acceleration | a punch with a rigid torso |
| `signal.anticipation` | counter-motion before a strike; damped oscillation after | motion that starts and stops dead |
| `signal.bilateral_phase` | left–right offset ≈ 50% in gait | broken or in-phase locomotion |

`signal.min_jerk` and `signal.dead_limb` together will likely catch more real "looks wrong"
cases than the VLM does, at zero marginal cost and with a number that can be
regression-tested. **Build this layer before spending on perceptual grading — it changes
what the VLM is needed for.**

### 3.3 Composite scoring

Every check returns `severity ∈ [0,1]` (from [02](02-analysis-layer.md) §3.2). A weighted
composite per clip becomes the **oracle for trajectory evals** in §6 — which is how
selection regret gets measured without a human.

---

## 4. LLM graders

Text and JSON only, no pixels, ~$0.002 per call. Nothing in this category exists today:
planner evaluation is fixture classification against expected intent, which measures
routing, not fidelity.

| Grader | Input | Output | Ground truth |
| --- | --- | --- | --- |
| **Clause fidelity** | prompt + `MotionProgram` | per-clause: requested / present / correct order / count match / laterality / hallucinated additions | semantic mutations ([06](06-mutation-library.md)) |
| **Rejection auditor** | prompt + capability grammar + the `UNSUPPORTED` reason | out-of-scope / capability-miss / ambiguous | curated supported+unsupported fixtures, which already exist (60 + 20) |
| **Repair soundness** | judge critique + `RepairPatch` | responsive? directionally correct? | patches synthesized to be deliberately irrelevant |
| **Failure honesty** | typed failure + the gate that actually fired | consistent? | known-bad corpus cases ([03](03-golden-corpus.md) §3.4) |

The rejection auditor is the highest-value of the four: **it measures the false-rejection
rate, which nothing measures today**, and a system that over-rejects looks flawless on
every other metric in this document.

**Correlated-failure mitigation**: grade with a different model family than the planner
uses. An LLM grading a program written by an LLM will happily ratify a misreading it would
have made itself.

---

## 5. VLM graders

Structure from [07](07-judge-harness.md): five focused graders, claim-based output,
anatomy and artifact blinded to the prompt, no deterministic diagnostics in any prompt.

### 5.1 Calibration per grader

| Grader | Ground truth source | Baseline |
| --- | --- | --- |
| `semantic` | cross-prompt discrimination (§2.2) | 0.50 |
| `anatomy` | anatomy mutations + deterministic agreement (§2.1, §2.3) | majority class, computed |
| `artifact` | clipping mutations + deterministic agreement | computed |
| `timing` | timing mutations | computed |
| `crossview` | view-mismatch mutations | computed |

### 5.2 Metrics reported per grader

- **Detection threshold** — severity at 50% detection, with a confidence interval. The
  headline number, and far more informative than accuracy at one arbitrary severity.
- **Sensitivity** on unmutated corpus clips that pass all deterministic gates. Non-circular
  because selection is by deterministic gates, never by the judge.
- **Specificity** across the mutation severity sweep.
- **Deterministic agreement** — κ against L1–L3 on the overlap subset.
- **Stability** — variance over N repeats.
- **Order-flip rate**.
- **Ablation response** — must be non-zero.

### 5.3 The gate

```python
passed = (
    detection_threshold.upper_bound_95 <= max_acceptable_severity
    and sensitivity.lower_bound_95     >= thresholds["min_sensitivity"]
    and specificity.lower_bound_95     >= thresholds["min_specificity"]
    and discrimination.lower_bound_95  >  majority_class_baseline   # strictly
    and deterministic_kappa            >= thresholds["min_kappa"]
    and stability.variance             <= thresholds["max_variance"]
    and ablation_response              >= thresholds["min_ablation_delta"]
    and degenerate_predictors_all_fail
)
```

Gating on **lower confidence bounds against a computed baseline** is what makes sample size
self-enforcing: a ten-item suite cannot pass regardless of score. This mechanism is carried
over intact from the retired calibration plan.

### 5.4 Pairwise strata, rebuilt without humans

The old suite's fatal flaw was that every scored item had the same ground-truth label, so
"always pick base" scored 0.9 against a 0.8 gate. Composition
([06](06-mutation-library.md) §3.4) fixes this without raters:

| Stratum | Members | Correct answer | Why it matters |
| --- | --- | --- | --- |
| **A** | base vs severe mutation | base | the existing signal, kept |
| **B** | **mild mutation vs severe mutation** | mild | **base is absent**, so "pick base" is impossible |
| **C** | base vs deterministically-equivalent sibling | no stable winner | measures fabricated preference |

With `|A| == |B|`, the majority-class baseline over A ∪ B is exactly 0.50, and no constant
strategy — pick-base, pick-first, pick-second — beats chance. Stratum B is the whole point,
and it requires only the mild tier from [06](06-mutation-library.md).

Stratum C scores **stable fabricated preference**: both orders returning the same non-tie
winner between two clips the deterministic layer cannot separate.

### 5.5 Degenerate-predictor regression test

Permanent, zero model calls. Synthetic judgment records from four degenerate predictors run
through the real scoring function; every one must fail:

| Predictor | Fails on |
| --- | --- |
| always accept | specificity bound = 0 |
| always reject | sensitivity bound = 0 |
| always base | absent from stratum B; directional accuracy ≈ 0.5 |
| always first | directional accuracy ≈ 0.5 by position balance |

**This lands before any suite construction.** It is the permanent guard against the entire
class of bug that invalidated the previous calibration.

---

## 6. Trajectory evals

Pure functions over a `Transcript` ([01](01-observability-and-transcript.md) §4.2). Neither
the planner nor the judge is agentic — both are single structured-output calls — so
"trajectory" means the pipeline's own decision path.

| Metric | Question | Oracle |
| --- | --- | --- |
| **Selection regret** | was the winner the best available candidate? | deterministic composite (§3.3) |
| **Repair efficacy** | does repair beat a random parameter perturbation? | the random control **is** the test |
| **Diversity relevance** | were the varied axes the ones that mattered for this prompt? | per-check severity spread across the batch |
| **Rejection attribution** | structural rejections by cause | check ids |
| **Stage attribution** | semantic / compile / structural / evidence / visual / repair-exhaustion | transcript span kinds |
| **Cost per accepted clip** | calls, tokens, wall clock, dollars | transcript |
| **Escalation precision** | did fallback escalation change the outcome? | compare primary vs fallback verdicts |

Two notes. Repair efficacy needs the **random-perturbation control** or it measures
nothing — a repair loop that jitters parameters will show a nonzero score without it. And
several of these are only computable because PR 01 records failed attempts and latency;
they are the concrete payoff for that work.

The verification run in this repository produced 11 candidates, rejected 3 structurally,
judged 5, and selected a winner with no repair. Not one of those numbers is evaluated
against anything today.

---

## 7. Reporting contract

One rule, and it is the durable fix: **every published rate carries its n, its baseline,
and its interval.**

Replace:

> The final frozen 10-pair calibration achieved 9/10 VLM-human agreement and 10/10 A/B
> order consistency; the 28-case corruption suite produced zero false accepts.

with, for example:

> Semantic discrimination 0.82 (95% LCB 0.71) against a 0.50 baseline, n = 60 forced-choice
> pairs at near semantic distance. Anatomy detection threshold 0.18 severity (95% CI
> 0.14–0.23), n = 140 graded mutations. Deterministic agreement κ = 0.61. Instrument
> validity only; no claim about human preference.

Artifacts: `eval-report.v2.json` plus a rendered HTML view reusing `evals/report.py`.
Per-family breakdown is mandatory — `docs/evaluation.md:64` has named per-family failure
attribution as the missing thing since the beginning, and it stays missing until this
lands.

---

## 8. Sequencing — six PRs

| PR | Contents | Model spend |
| --- | --- | --- |
| **10a** | `calibration_stats.py` — Clopper–Pearson, κ, MCC, majority baseline, balanced accuracy — plus the **degenerate-predictor test** | none |
| **10b** | Physics layer (§3.1): CoM and everything it unlocks | none |
| **10c** | Signal layer (§3.2) | none |
| **10d** | Calibration drivers: mutation sweeps, cross-prompt sets, ablation controls, `--offline-scoring` | none to build |
| **10e** | LLM graders (§4) | low |
| **10f** | VLM calibration run (§5) + report (§7); retire `calibrate_judge.py` and `judge_human_review.py` | high |
| **10g** | Trajectory evals (§6) | none |

**10a–10d and 10g cost nothing to run.** That is most of the suite, and the scoring logic
is proven before it is ever pointed at a paid model. Only 10f spends meaningfully.

---

## 9. Definition of done

1. `uv run pytest -q` green, including the degenerate-predictor test.
2. No grader gates a release without a calibration record carrying n, baseline, and bounds.
3. `docs/evidence/` holds committed evidence at `schema_version: 2.0`; the audit passes on
   a **clean clone**, which it cannot today.
4. Every threshold defined once, with a source ([08](08-threshold-consolidation.md)).
5. No published rate anywhere in the repository stated without its n and its baseline.
6. Per-family failure attribution exists across all seven stages.
7. `MINIMUM_BASE_PREFERENCE` (`judge_human_review.py:13`) and every other human-rating gate
   is **deleted**, not tuned.

---

## 10. Risks and open decisions

### 10.1 Risk — mutation-based ground truth has a blind spot

Mutation measures whether a grader detects *induced* defects. It cannot measure whether the
grader would detect a defect the generator produces naturally but no mutation models.
Mitigation: rejection attribution (§6) surfaces which checks fire in production; a defect
class that appears in production and has no corresponding mutation is a gap in the library,
and that gap is visible rather than silent.

### 10.2 Risk — cross-prompt discrimination may be too easy

If far-distance pairs score near 1.0, the metric saturates and stops discriminating.
Mitigation: report per-distance-band, and treat the **near** and **minimal** bands as the
headline. Saturation at far distance is itself a useful negative result.

### 10.3 Open — what replaces "the judge is calibrated" as a release claim

Under this design the honest claim is instrument validity, not quality. The product-level
question — *is Rigby's output good?* — has no non-human answer, and this plan does not
manufacture one.
**Recommendation: state the deterministic pass rate and the grader validity separately, and
do not combine them into a single quality score.** A composite number would recreate
exactly the rhetorical problem §2.5 warns about. **Decide before 10f**, since it determines
what the report claims.

### 10.4 Open — production versus calibration grader

If [07](07-judge-harness.md) §6.1 concludes that production keeps a single combined grader
while calibration uses the split ones, then production validity rests on calibration
transferring between two different prompts. That is an assumption, and it should be stated
in the report rather than glossed. Measuring it directly costs one extra calibration arm
against the combined grader.

### 10.5 Note — expect the first honest numbers to look worse

Removing diagnostics leakage, enforcing `semantic_match`, adding ROM checks, and measuring
foot drift for real will all reduce apparent quality. Every one of those is a correction to
a measurement that was previously flattering. Say so in the PR descriptions, or the first
honest run will read as a regression.
