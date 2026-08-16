# PR 07 — Judge harness: split, unblind, debias

Status: proposed, not started.
Scope: restructure the VLM judge so its dimensions can be independently calibrated, remove
the deterministic-diagnostic leakage that makes it un-measurable, and fix the blinding seed
and one live acceptance bug.

Depends on: [01 — observability](01-observability-and-transcript.md) for per-grader call
records, [05 — capture integrity](05-capture-integrity.md) for trustworthy evidence.
Blocks: [10 — eval redesign](10-eval-redesign.md).

---

## 1. Problem

### 1.1 A live bug: `semantic_match` is never gated

`acceptance_criteria.yaml:12-17` states the release threshold requires
`semantic_match ≥ 4`. But `src/rigby_poc/judge.py:893-895` computes acceptance from only
three dimensions:

```python
score.overall >= 4
and score.anatomical_naturalness >= 4
and score.gesture_recognizability >= 4
```

**A clip scoring `semantic_match = 1` is auto-accepted** as long as the other three pass.
The documented criterion has never been enforced. Worse, the `acceptance_score_inconsistency`
escalation at `judge.py:898` would fire against a model that *did* honour the published
rule.

This is the most consequential single finding in the audit and should be fixed first,
independently of the restructure.

### 1.2 One 115-line mega-prompt

`UNARY_SYSTEM_PROMPT` (`judge.py:156-270`) is a single accreted block covering hang-ten
finger semantics, hook-versus-jab path geometry, object throw/catch/push/roll/spin/place/
drop/handoff lifecycles, jumping jacks, burpees, squats, lunges, sit-ups, cartwheels,
ladder climbing, push-up plank alignment, and more.

Three consequences: it is unmaintainable, it dilutes model attention across unrelated
questions, and **it cannot be calibrated per-dimension** — there is no way to ask whether
the anatomy judgement is any good separately from the semantic one.

### 1.3 Deterministic diagnostics are fed to the judge

The judge receives `motion_diagnostics` — ~130 measured values — alongside the images
(`judge.py:635`, assembled in `capture.py:27-732`). The prompt at `judge.py:169-172`
even instructs the model on how to treat reference bounds.

This leaks the deterministic layer's verdict into a supposedly independent perceptual
signal. It encourages rubber-stamping, and it makes the VLM's actual perceptual ability
unmeasurable: you cannot tell whether it saw a bent wrist or read a number about one.

### 1.4 The blinding seed is a fixed constant

`evals/flywheel.py:2417` seeds five-way blinding with `70_000 + round_index`. Recipe *k*
therefore lands in the same A–E slot in round 0 of **every run of every prompt**. Any
positional bias in the model becomes a systematic, reproducible preference for one recipe.
`judge.py:988, 1167` shuffle from that seed.

The tournament path has the same shape at `90_000 + match_index`.

### 1.5 Scores are flat and correlated

The verification run scored the winner 4/4/4/4/5/4 across six dimensions. Seven correlated
1–5 ratings that move together carry little information and cannot be scored for accuracy
against any ground truth. Binary or ternary claims with cited evidence can be.

### 1.6 `recommend_repair` is unobserved and unguarded

`judge.py:1125-1157` calls `_parse_response` directly: no retry, no fallback, no
`attempts`, no `routing`, no correlation id, and no record of its reasoning effort. It is
the least observable model call in the system and it directly shapes the repair loop.

---

## 2. Goals and non-goals

**Goals**

- G1. Enforce the documented acceptance rule, including `semantic_match`.
- G2. Independent, short, single-purpose graders that can each be calibrated.
- G3. The anatomy and artifact graders are blinded to the prompt.
- G4. No deterministic diagnostics in any grader prompt; combination happens in an
  explicit decision rule.
- G5. Content-derived blinding seeds; order-flip rate reported as a standing metric.
- G6. Every grader call is fully observable via PR 01.

**Non-goals**

- Changing the accept threshold values. Whether 4/5 is right is a question for
  [10](10-eval-redesign.md) once graders are measurable.
- Removing the five-way selection design. Listwise ranking stays; it is the right shape.

---

## 3. Design

### 3.1 Split into focused graders

| Grader | Sees | Blinded to | Output |
| --- | --- | --- | --- |
| `semantic` | prompt + evidence | diagnostics, recipe, other scores | per-clause claims |
| `anatomy` | evidence only | **the prompt** | per-joint claims |
| `artifact` | evidence only | the prompt | clipping / popping / cropping |
| `timing` | chronological strip only | single-pose views | per-transition claims |
| `crossview` | paired ego + orbit | — | consistency claims |

Blinding anatomy and artifact to the prompt is the important one: knowing what was
requested causes a model to rationalize a broken pose as intentional style.

Per-intent guidance moves from one mega-prompt into **per-family fragments** composed only
into the `semantic` grader, which is the only one that needs to know what a hook is.

### 3.2 Verifiable claims instead of 1–5 vibes

```json
{"claims": [
  {"id": "fist_closed_at_impact", "verdict": "yes",
   "confidence": 0.9, "snapshot_id": "08-impact_pose-orbit"},
  {"id": "elbow_bent_on_arc", "verdict": "no",
   "confidence": 0.7, "snapshot_id": "07-strike_midpoint-ego"}
]}
```

Each claim is `yes | no | cannot_tell` with a required snapshot citation. Scores are
derived from claims by an explicit, testable aggregation rule rather than asked for
directly. `cannot_tell` is a first-class answer — it is what makes low-evidence cases
distinguishable from failures, and it is the honest response when 15 stills cannot settle
a timing question.

### 3.3 Diagnostics leave the prompt

The judge receives images and the prompt only. The decision layer combines:

```python
accepted = deterministic_valid and grader_verdict and score_threshold
```

`deterministic_valid` was always the authority; making the combination explicit means the
graders can finally be measured against it (see [10](10-eval-redesign.md), deterministic
agreement).

### 3.4 Content-derived blinding

```python
seed = int.from_bytes(sha256(b"|".join(sorted(result_ids)) + prompt_hash)[:8], "big")
```

Reproducible from content, uncorrelated with recipe order or round index. Both orders are
run and the flip rate recorded per grader as a standing quality metric.

### 3.5 Prompt versioning

Every grader prompt gets an explicit `version` and its sha256 recorded in the run config
(PR 01 §4.4). A calibration result is only meaningful against a stated prompt version, and
today prompt drift across revisions is invisible in the artifacts.

### 3.6 `recommend_repair` routed like everything else

Route it through `_routed_parse` so it gains attempts, routing, retry, fallback, and
correlation refs. Pure consistency fix, no behavioural intent.

---

## 4. Sequencing — four PRs

| PR | Contents | Risk |
| --- | --- | --- |
| **07a** | Fix `semantic_match` gating; content-derived seeds; route `recommend_repair` | low, high value — ship independently and immediately |
| **07b** | Grader split behind a flag; per-family prompt fragments; old path still default | medium |
| **07c** | Claim-based output + aggregation rule; `cannot_tell` | medium |
| **07d** | Remove diagnostics from prompts; explicit decision layer; make split graders default | high — changes acceptance behaviour |

**07a should not wait for the rest.** It fixes a real acceptance bug in a few lines.

07d must not land before [10](10-eval-redesign.md) can measure the result, or you will
have changed the judge without being able to tell whether it improved.

---

## 5. Test plan

- `test_acceptance_rule.py` — a score with `semantic_match = 1` and everything else at 5
  is **rejected**. This is the direct regression guard for §1.1 and should exist before
  anything else in this PR.
- `test_blinding_seed.py` — seed is stable for identical content, differs across prompts,
  and recipe→slot assignment is uniform over many synthetic runs. Explicitly assert that
  recipe *k* is **not** always in slot A in round 0.
- `test_grader_isolation.py` — the anatomy grader's assembled payload contains no prompt
  text and no diagnostics. String-level assertion on the payload, so leakage cannot creep
  back in.
- `test_claim_aggregation.py` — the claims→score rule is a pure function with table-driven
  cases, including all-`cannot_tell`.
- `test_prompt_version_recorded.py` — every grader record carries a prompt version and
  hash.

All run against a stubbed client; no model calls.

---

## 6. Risks and open decisions

### 6.1 Risk — splitting graders multiplies calls

Five graders instead of one is up to 5× the per-candidate cost, against a hard
four-call-per-run budget (`pipeline.py:167`). Mitigations: run `anatomy` and `artifact`
only on candidates that pass deterministic gates (most already do), batch the five-way
listwise call as today, and reserve per-candidate unary grading for calibration rather
than production selection.
**This needs a decision before 07b** — it may mean production keeps a single combined
grader while calibration uses the split ones. That is a legitimate outcome, but it must be
deliberate, and the production grader's validity claim then rests on the split graders'
calibration transferring, which is an assumption to state rather than assume.

### 6.2 Risk — removing diagnostics may lower measured judge quality

Quite likely, and that is the point: current numbers are partly the deterministic layer's
performance wearing the judge's name. Expect scores to drop in 07d and do not treat that
as a regression. It is the first honest measurement.

### 6.3 Open — does the timing dimension survive?

Timing is currently judged from 15 stills per view. It may simply not be a sound
instrument. Rather than assume, calibrate it in [10](10-eval-redesign.md) against timing
mutations; if it cannot beat baseline, either move to dense strips or short video, or
**delete the dimension** rather than keep reporting a number that means nothing.

### 6.4 Note — 07d invalidates prior calibration

Any calibration frozen before 07d describes a different instrument. Sequence 07d **before**
the new calibration is frozen in [10](10-eval-redesign.md), never after.
