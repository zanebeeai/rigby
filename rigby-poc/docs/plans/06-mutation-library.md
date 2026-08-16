# PR 06 — Mutation library: synthetic ground truth

Status: proposed, not started.
Scope: a graded, body-wide clip mutation library. Under a no-human-evaluation policy this
is the primary source of labelled data for the entire suite, and it validates the
deterministic checks as well as the model graders.

Depends on: [02 — analysis layer](02-analysis-layer.md),
[03 — golden corpus](03-golden-corpus.md),
[04 — anatomical frame](04-anatomical-frame.md) for ROM-directed mutations.
Blocks: [10 — eval redesign](10-eval-redesign.md).

---

## 1. Problem

### 1.1 No humans means ground truth must be constructed

The project has ruled out human evaluation entirely. Every grader above the deterministic
floor still needs labelled data, so labels have to come from construction: **apply a known
perturbation `T` to a known-good clip `C`, and you know by construction that `T(C)` is
worse than `C` along the axis `T` targets.** No rater required, and the labels regenerate
for free on every model upgrade.

That promotes mutation from a side suite to load-bearing infrastructure.

### 1.2 The existing corruption library is far too narrow

`evals/corruptions.py:43-99` defines 28 specs across 5 kinds:

- `wrist_rotation` — 12 specs, three axes × `(-1.25, -0.85, 0.85, 1.25)` rad
- `wrong_joint_shake` — 4
- `fist_shape` — 4
- `open_middle_fingers` — 4
- `timing` — 4 (`snap_present`, `stutter_present`, `reverse_present`, `remove_hold`)

Every one is **hand- and wrist-specific**, built for a single hang-ten gesture. There is
nothing for legs, spine, root motion, balance, foot contact, or object interaction — i.e.
nothing for the majority of what Rigby now generates.

### 1.3 Every corruption is glaring

The mildest wrist spec is 0.85 rad ≈ **49°**. A grader that detects a 49° wrist error has
demonstrated almost nothing; a human would never ship that clip. Pass/fail against obvious
corruptions produces the uninformative all-negative suite that invalidated the existing
calibration.

What is needed instead is a **severity sweep** that finds the *detection threshold* — the
magnitude at which a grader starts catching the defect. That is a sensitivity curve, not a
boolean, and it is far more informative per model call.

### 1.4 Mutation also tests the checks

An underused property: mutation validates the deterministic layer too. Inject a known 30°
off-axis elbow rotation and assert the ROM check reports ≈30°. This is metamorphic
testing, and it is the only way to regression-test that a check still fires — a refactor
that silently disables a gate otherwise looks exactly like a clean test run.

---

## 2. Goals and non-goals

**Goals**

- G1. Mutations across the whole body and every check family, not just hand and wrist.
- G2. Every mutation is **graded**, spanning sub-perceptual to egregious.
- G3. Every mutation declares which check family it targets, so detection can be scored
  per axis.
- G4. Mutations are deterministic, parameterized, and composable.
- G5. Mutation drives both grader calibration and deterministic-check regression testing.

**Non-goals**

- Deciding whether a mutation is *perceptible*. That is what the graders are being
  measured on; asserting it up front would be circular.
- Replacing the existing 28 specs. They are absorbed as the severe tier of their families.

---

## 3. Design

### 3.1 Mutation spec

```python
@dataclass(frozen=True)
class MutationSpec:
    id: str
    family: str          # anatomy | timing | contact | balance | semantic | signal
    targets: tuple[str, ...]   # check ids this should trip: ("anatomy.elbow.off_axis",)
    severity: float      # 0.0 … 1.0, monotonic within a family
    params: dict[str, Any]
    def apply(self, clip: ClipResult, program, scene) -> ClipResult: ...
```

`targets` is what makes per-axis scoring possible: a grader is credited only for detecting
what a mutation actually broke.

### 3.2 Severity sweeps

Each mutation family is a **parameterized generator**, not a fixed list:

```python
def wrist_flexion_sweep(levels=(0.05, 0.10, 0.20, 0.35, 0.60, 0.85, 1.25)) -> list[MutationSpec]
#                                 ~3°   ~6°   ~11°  ~20°  ~34°  ~49°  ~72°
```

The bottom of every sweep must be **plausibly sub-perceptual**. That is the whole point:
you cannot locate a detection threshold from above it.

### 3.3 Family coverage

| Family | Mutations | Targets |
| --- | --- | --- |
| **anatomy** | per-DOF ROM violation on any of 52 bones, at graded magnitude; hinge off-axis injection; finger hyperextension | `anatomy.*` |
| **timing** | snap, stutter, freeze, reverse, hold removal, ease inversion, global scaling | `signal.min_jerk`, `signal.sparc` |
| **signal** | additive joint jitter; velocity-profile flattening toward constant velocity; dead-limb (freeze one limb) | `signal.*` |
| **contact** | foot skate injection; ground penetration; float; premature release; attachment slip | `physics.contact.*` |
| **balance** | CoM shift outside support polygon; support-foot removal; ballistic violation during flight | `physics.balance.*` |
| **semantic** | wrong hand mirror; wrong hand shape; phase reorder; drop a requested cycle; wrong travel direction | LLM/VLM semantic graders |
| **clipping** | drive a limb through the torso; finger through palm | `anatomy.self_collision` |

The `semantic` family is special: those mutations are targeted at **model graders only**,
since the deterministic layer cannot judge whether a wave reads as a wave. They are how
the VLM's semantic dimension gets a labelled negative class.

### 3.4 Composition

```python
def compose(*specs: MutationSpec) -> MutationSpec
```

Compounded mutations give a severe tier without pushing any single axis to absurdity, and
they let a directional pair be built from *two mutated* clips rather than base-vs-mutated.
That removes the "always pick the unmutated one" degenerate strategy — see
[10](10-eval-redesign.md) §pairwise strata.

### 3.5 Application layer

Mutations operate on a compiled `ClipResult`, **after** the compiler, never through it.
That keeps them independent of generation, makes them applicable to any corpus case, and
means a mutation cannot be silently repaired by a compiler constraint.

Consequence worth stating: a mutated clip may be physically impossible in ways the
compiler would never produce. That is intentional — the graders are being tested on
detection, not on plausibility of provenance.

### 3.6 Determinism

Mutations take an explicit seed and are pure. `mutate(clip, spec)` must be byte-identical
across runs, so a calibration suite is reproducible.

---

## 4. Sequencing — three PRs

| PR | Contents | Effort |
| --- | --- | --- |
| **06a** | `MutationSpec`, sweep generators, composition, determinism tests; port the existing 28 specs into the new shape as the severe tier | ~2 days |
| **06b** | anatomy / timing / signal / clipping families with graded sweeps | ~3 days |
| **06c** | contact / balance / semantic families; the check-detection matrix in §5 | ~3 days |

---

## 5. Test plan

The centrepiece is **`tests/test_mutation_detection_matrix.py`**: for every mutation spec,
apply it to every applicable corpus case, run the deterministic analyzer, and assert that
**the targeted check fires at high severity and non-targeted checks do not**.

This one test does two jobs at once:

- it proves the deterministic checks actually detect what they claim, and
- it produces the per-check detection-threshold table that [10](10-eval-redesign.md)
  compares model graders against.

A mutation whose target never fires is either a broken mutation or a broken check, and the
matrix says which by showing whether *anything* fired.

Also:

- `test_mutation_determinism.py` — same seed, byte-identical output.
- `test_mutation_severity_monotonic.py` — within a sweep, increasing severity produces
  monotonically increasing measured deviation. Catches sweeps that saturate or wrap.
- `test_mutation_preserves_contract.py` — a mutated clip is still schema-valid, finite,
  and correctly framed, so downstream capture and export do not fail for the wrong reason.

---

## 6. Risks and open decisions

### 6.1 Risk — mutations that the deterministic layer cannot see are unfalsifiable

For the `semantic` family there is no deterministic oracle, so a mutation that "should" be
detectable rests on the assumption that it changed something meaningful. Mitigation: every
semantic mutation must produce a **measurable** program-level difference (different hand,
different cycle count, different phase order) that is asserted independently. If nothing
measurable changed, the mutation is not a valid negative.

### 6.2 Risk — sub-perceptual mutations may be genuinely undetectable

The bottom of a sweep may sit below what *any* grader — or any human — could detect. That
is fine and expected: it is what defines the detection threshold. It becomes a problem
only if it is misread as grader failure. The reporting in [10](10-eval-redesign.md) must
present the curve, never a single pass/fail at one severity.

### 6.3 Open — should mutations be rendered once or per-severity?

Rendering every severity level of every mutation across the corpus is the dominant cost of
the calibration suite: 7 levels × ~20 mutations × ~10 cases × 2 views × 15 frames is a
large number of images. Options: sample severities adaptively (bisect toward the detection
threshold rather than sweeping uniformly), or restrict full sweeps to a small "calibration
subset" of corpus cases and use three levels elsewhere.
**Recommendation: adaptive bisection**, which finds a threshold in ~3 renders per mutation
instead of 7. **Decide before 06c**, since it changes the driver's shape.

### 6.4 Note — this supersedes `evals/corruptions.py`

The existing module becomes a thin compatibility shim re-exporting the severe tier, then
is deleted once `evals/calibrate_judge.py` is rewritten in [10](10-eval-redesign.md). Do
not maintain both.
