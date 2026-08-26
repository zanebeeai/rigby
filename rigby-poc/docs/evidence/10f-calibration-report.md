# 10f — VLM grader calibration, first honest run

Run date 2026-08-26. Model `gpt-5.6-luna`, fallback `gpt-5.6-terra`. Corpus at
`d58b389`. 1,334 clips scored, zero score failures, 126 minutes, **actual spend
$9.41** against a $8.86 projection and a $35.43 authorised ceiling.

Plan 10 §10.3 governs what this document may claim: **the deterministic pass rate
and the grader validity are reported as two separately named numbers and are never
combined into a single quality score.** Both halves fail, separately. That is a
legitimate outcome and it is written as one.

## Verdict

**The grader is not a valid instrument on this corpus.** It accepts most clips
whether or not they were deliberately broken.

| | rate | n | baseline |
| --- | --- | --- | --- |
| accept rate, unmutated clips | 34/41 = 0.829 | n = 41 | majority class 1.000 |
| accept rate, mutated clips | 881/1288 = 0.684 | n = 1288 | majority class 0.000 |

Matthews correlation is **+0.054**, against 0.000 for every constant predictor —
that is the whole finding in one number. Balanced accuracy is 0.573 versus a
chance level of 0.500.

## `deterministic_authored_gate_agreement`

**5 of 5 exact**, n = 5, against an authored-gate baseline that no clip satisfies
by accident. Each `known_bad` case tripped exactly the `must_fail` gates named for
it — no misses and no supersets. Non-circular, because `must_fail` is authored
(`evals/corpus/models.py:213`) rather than derived from the compiler's verdict.

**Certifies nothing, and this is structural rather than a sampling shortfall.**
The floor for a 0.70 threshold is 9 items and the corpus can supply at most 8:
three of eleven `StructuralGate` members are unreachable by generation, and the
rest cluster. Do not read this as "collect more cases" — there are no more to
collect without changing what the label means.

## `grader_specificity_mutation_arm`

Reject rate on mutated clips, decomposed by severity band. The pooled figure is
**not** the headline: the severe band is materially better than the pool, so
quoting the pool alone understates the grader by a third.

| band | rejected | n | rate | LCB95 | baseline |
| --- | --- | --- | --- | --- | --- |
| subperceptual | 94 | 368 | 0.255 | 0.218 | 0.000 |
| mild | 95 | 368 | 0.258 | 0.221 | 0.000 |
| moderate | 140 | 368 | 0.380 | 0.338 | 0.000 |
| **severe** | **78** | **184** | **0.424** | **0.363** | 0.000 |
| pooled | 407 | 1288 | 0.316 | 0.295 | 0.000 |

Pooling across severity is correct in exactly one place —
`evals/calibration/detection.py:363`, where static targets have no severity axis
and the levels are repeats of one measurement. This is not that place. Severity is
the independent variable of a detection curve, so a pooled rate over it averages
seven different experiments.

**Certifies nothing at any band.** The severe band's lower bound of 0.363 is half
the 0.70 floor. The conclusion is unchanged by the decomposition; the number is
not.

## Sensitivity

**Certifies 0.70, n = 41**, against a majority-class baseline of 1.000. 34 of 41
deterministically-valid unmutated clips accepted, lower bound 0.703. The ladder at
this sample size, one-sided 95% Clopper-Pearson:

| observed | LCB95 | certifies |
| --- | --- | --- |
| 41/41 | 0.930 | 0.90 |
| 38/41 | 0.833 | 0.80 |
| 35/41 | 0.731 | 0.70 |
| **34/41** | **0.703** | **0.70** |
| 33/41 | 0.676 | nothing |

It is **one clip from certifying nothing**. A bare "passed at 0.70" would hide
that, which is why the ladder is published beside the figure rather than the
figure alone.

## Detection thresholds — three of four sweeps fail the gate

Severity at 50% detection, per plan 10 §5.2. `detection_threshold=None` is a gate
**failure** and not a skip: a grader whose detection cannot be measured has not
been shown to detect.

| sweep | severity at 50% detection | severe-band rate | n |
| --- | --- | --- | --- |
| rom | not reached at any severity | 22/46 = 0.478 | n = 46 |
| jitter | not reached at any severity | 12/46 = 0.261 | n = 46 |
| clip | not reached at any severity | 21/46 = 0.457 | n = 46 |
| snap | 0.68 | 23/46 = 0.500 | n = 46 |

Each rate above is stated against a baseline of 0.000, the reject rate a constant
accepting grader would score.

## What this run cannot distinguish, stated as a limit

Graders see key-pose stills plus a nine-frame chronological contact sheet per
view. **Jitter is high-frequency frame-to-frame noise and cannot exist in a still**;
at nine sampled frames it is aliased away. So "the grader is weak" and "the
evidence format cannot carry this defect family" are confounded, and **this run
does not separate them, because it varied the defect family and never the evidence
format.**

What the run does constrain: the format-limit hypothesis predicts that spatial
defects, which a single still can carry, should be detected best. The observed
ordering does not match.

| sweep | what it needs to be visible | severe-band rate |
| --- | --- | --- |
| snap | a discontinuity between consecutive frames — temporal | 0.500 |
| rom | a limb past its range in one pose — spatial | 0.478 |
| clip | a limb intersecting the torso in one pose — spatial | 0.457 |
| jitter | noise across many frames — temporal, high frequency | 0.261 |

The best-detected family is temporal and the worst is also temporal, while both
spatial families sit in between. The format explains **jitter** plausibly and
explains nothing about `rom` and `clip`, which a still should carry and which are
still under 0.50. So the single-axis framing is too coarse: the run neither
resolves the confound nor licenses dismissing it.

**The missing arm is one defect family presented under two evidence formats.**
Until that exists, no number here separates instrument weakness from format
limitation, and this document does not claim to.

## Caveats that bound the numbers above

- **The check surface is not uniformly mutation-responsive.** `MutationSpec.apply`
  transforms frames only, so `anatomy.rom.*` recomputes from frames while the
  metric-derived non-ROM checks read pre-mutation values. This bounds the
  *deterministic* comparison, not the grader arm, which reads pixels.
- **Escalation was 336 of 6,670 calls, n = 6670, against an assumed 30%** — the
  measured 0.050 is six times lower than the projection, and the projection was a
  guess rather than a measurement.
- **Escalation in the first tranche is contaminated** by rate-limit retries and is
  excluded from that figure.
- **Rates are from a single run.** Stability over repeats, order-flip and ablation
  response are unmeasured; §5.2 lists them and this run did not buy them.
- **The model rates used for cost are secondary-sourced** and were not verified
  against a primary reference.

## Three grader-contract defects found before the run could score anything

Each was invisible to the suite because every judge in it is a fake client
(`docs/testing.md`), and fakes emit exactly the ids the aggregator wants. The
split path has been default since 07d and had never met a real model.

1. **`Claim.id` was an open string.** The grader rendered the id's final separator
   as an underscore on 4 of 5 calls, n = 5, against a complete-response baseline of
   5 of 5. Closed to a `Literal`; complete after the fix on 5 of 5.
2. **`Claim.snapshot_id` was an open string.** The grader returned a comma-joined
   list of five ids in a scalar field; `Claim` accepted it at 48 characters and
   `MotionJudgeScore` rejected it at 32, two layers downstream.
3. **`_key_pose_labels` declared a label nothing emits.** Its default
   `presented_pose` comes only from the `present` phase, so on full-body, composite
   and sequence clips `crossview` was dispatched with **zero images** and returned
   four claims anyway, while three `evidence="both"` graders silently ran on half
   their declared evidence.

Without these fixes the run would have scored almost nothing, and a broken grader
would have been indistinguishable from a broken harness.
