# PR 02 — Extract the analysis layer

Status: proposed, not started.
Scope: move metric computation and structural validation out of `compiler.py` into a pure
module that operates on a finished clip, so checks can run on any clip without recompiling
and can be added without touching an 8446-line file.

Depends on: nothing.
Blocks: [03 — golden corpus](03-golden-corpus.md), [04 — anatomical frame](04-anatomical-frame.md),
[06 — mutation library](06-mutation-library.md), [10 — eval redesign](10-eval-redesign.md).

---

## 1. Problem

Every validation metric Rigby computes lives inside `compiler.py`. There is no way to run
a check against an arbitrary clip — a stored result, a mutated clip, a corpus fixture —
without going through compilation. Adding a check means editing the largest file in the
repository, and a check cannot be unit-tested in isolation.

That single fact blocks most of the eval suite: mutation testing, the golden corpus,
per-DOF anatomy checks, and any regression test of the checks themselves.

### 1.1 The good news, established by inspection

The extraction is far more tractable than the file size suggests. **Every metric block is
already a post-hoc pass.** No metric is accumulated inside a per-frame loop anywhere; each
compile path appends its last frame and then runs one contiguous measurement block:

| Path | last `frames.append` | metric block | lines |
| --- | --- | --- | --- |
| `_compile_full_body` (1313) | 2780 | 2853–4726 | ~1875 |
| `_compile_composite` (5304) | 5563 | 5577–5778 | ~200 |
| `_compile_object_handoff` (5947) | 6258 | 6263–6355 | ~93 |
| `_compile_object_interaction` (6356) | 6800 | 6825–7065 | ~241 |
| `_compile_sequence` (7430) | 7617 | 7735–7927 | ~193 |
| `compile_motion` gesture/strike/grab (7952) | 8244 | 8254–8407 | ~154 |

The compiler was written generate-then-measure. The metrics simply never got a file of
their own.

Structural validation is likewise a separate linear pass at six sites
(`compiler.py:4342-4688, 5615-5725, 6281-6303, 6925-6984, 7908-7918, 8369-8377`, plus
`quality.py:291-306`), each a flat sequence of `if metric > threshold: failures.append(...)`
reading `metrics[...]` set earlier in the same block. Never a loop variable. So validation
extraction is nearly free once metrics move: it becomes a pure
`(metrics, program) -> list[str]`.

### 1.2 What is already pure and just misfiled

These have exactly the signature the analysis layer wants and can move mechanically:

- `compiler.py:537` `_safety_metrics(frames, *, allow_root_motion)`
- `compiler.py:4777` `_parallel_forearm_metrics(frames, phase_ranges)`
- `compiler.py:4950` `_intra_hand_contact_metrics(frames, phase_ranges, program)`
- `compiler.py:5172` `_semantic_cycle_metrics(frames, phase_ranges, program)`
- all 324 lines of `quality.py`

### 1.3 The general FK primitive already exists

`kinematics.py:51` `RigKinematics` parses the GLB and evaluates the true hierarchy for
**all 68 nodes / 52 canonical bones including all 30 finger bones**, from a plain
`Mapping[str, BonePose]` — i.e. exactly `frame.bones`. It is completely generation-free:
`world_matrices` (109), `canonical_positions` (144), `fingertip_positions` (151),
`canonical_world_rotation` (174). Cached via `rig_kinematics()` (394).

The full-body path already uses it post-hoc (`compiler.py:2875`). This is the single
biggest enabler and it is already written.

### 1.4 What genuinely cannot be recovered from a clip

Four things, and three have a cheap fix:

1. **IK support targets** — `support_constraints` (`compiler.py:2811-2817`) and
   `climb_support_constraints` (2788-2798) record the *commanded* ankle position per
   frame. Four metrics compare achieved against commanded:
   `max_support_foot_target_error_m`, `max_support_foot_slide_per_frame_m`,
   `support_contact_fraction` (4230-4261), `climb_support_target_max_error_m` (3577).
   Authoring intent is not observable and cannot be inverted out of the clip.
   **Fix: persist them into `metrics`.** They are small — `frame_index`, `side`, a
   3-float target — and this is a ~10-line compiler change that unblocks four metrics.
   Highest-leverage edit in this PR.
2. **`current_yaw`** (`compiler.py:2874` → `final_root_yaw_deg`) — accumulated turn
   *intent*. Achieved yaw is recoverable from the hips quaternion; commanded is not.
   Same fix, or accept the achieved approximation and rename the key honestly.
3. **MuJoCo grasp metrics** — `physics.py:127` `simulate_grasp` simulates a proxy
   cartesian gripper disconnected from the rig clip; ~15 metrics cannot be recovered from
   frames. **But `compiler.py:375` `_physics_for(program, scene)` is pure and
   deterministic**, including its 8-attempt retry ladder, so the analyzer can simply
   re-run it and get identical results. A cost decision, not a correctness blocker.
4. **Sequence step-splitting** — `_compile_sequence` folds *child* `ClipResult.metrics`
   (7656-7678). A post-hoc analyzer must re-split by `metrics["sequence_step_ranges_s"]`
   and re-analyze each span. Children were compiled in local space then yaw-rotated and
   translated into world (`_sequence_frame_in_world`, 7092), and `_blend_sequence_boundary`
   (7376) inserts bridge frames belonging to no step. Expect drift in root-relative
   metrics. This is the hardest of the six paths.

### 1.5 Two traps

- `store.py:49` persists `request.program`, which is the **pre-override** program. An
  analyzer reading stored artifacts must call `compiler.apply_overrides(request)`
  (`compiler.py:193`, already public) to get the effective `(scene, program)`.
- `phase_ranges` timing is **not** `primitive.parameters.duration_s` — it is inflated by
  ~70 lines of per-action frame-count rules (`compiler.py:1419-1492`). Do not re-derive
  it; read `metrics["phase_ranges_s"]`, already persisted at `compiler.py:2854`.

### 1.6 One real inconsistency, surfaced by this work

`compiler.py:8302-8303` sets `forearm_rotation_cycles` from **planner parameters**, then
`compiler.py:8320` overwrites it with the **measured** value from
`shake_joint_oscillation_metrics` — but only for `Intent.GESTURE`. Other intents keep the
parameter-derived value, i.e. a metric that claims to be a measurement is sometimes an
echo of the request. Fix during extraction; note it in the PR.

Relatedly, `quality.py:55` `arm_landmarks` is a hand-rolled arm-only FK using hardcoded
constants from `primitives.py`, and it does **not** agree exactly with
`RigKinematics.canonical_positions` (different rest data sources). Gesture metrics
computed the two ways are not interchangeable. Converge on `RigKinematics`.

---

## 2. Goals and non-goals

**Goals**

- G1. `analysis.analyze(clip, program, scene) -> Metrics` is pure: no server, no browser,
  no key, no recompilation.
- G2. Structural validation is `validate(metrics, program) -> list[CheckResult]`.
- G3. Every check is individually addressable and individually testable.
- G4. Byte-identical metric output versus today, verified against stored results.
- G5. Thresholds become data, not inline literals.

**Non-goals**

- Changing any metric definition or threshold value. This is a move, not a redesign. New
  checks arrive in [04](04-anatomical-frame.md) and [10](10-eval-redesign.md).
- Removing metric computation from the compile path. The compiler keeps calling the
  analyzer so `ClipResult.metrics` is populated exactly as today.

---

## 3. Design

### 3.1 Module layout

```
src/rigby_poc/analysis/
  __init__.py          analyze(), validate()
  context.py           AnalysisContext: FK cache, world positions, phase ranges
  registry.py          check registration + applicability
  safety.py            from compiler._safety_metrics
  kinematics_metrics.py angular velocity/accel/jerk, discontinuity
  gesture.py           from quality.py
  full_body/           one module per BodyAction — 13 of them
  composite.py
  objects.py
  sequence.py
```

### 3.2 The check contract

```python
@dataclass(frozen=True)
class CheckResult:
    id: str                    # "anatomy.elbow.off_axis"
    layer: str                 # contract | anatomy | physics | signal
    status: Literal["pass", "fail", "skip"]
    measured: float | dict[str, Any]
    threshold: float | None
    severity: float            # 0.0 pass … 1.0 egregious
    frames: tuple[int, ...]    # where it fired, for evidence
```

`severity` rather than a boolean is what lets [10](10-eval-redesign.md) rank candidates
and compute selection regret without a human.

### 3.3 Shared context, computed once

`AnalysisContext` holds the expensive derivations so 35 checks do not each recompute FK:

```python
ctx.world_positions   # [kinematics.canonical_positions(f.bones) for f in frames]
ctx.phase_ranges      # from metrics["phase_ranges_s"], never re-derived
ctx.ground_height     # compiler.py:3486
ctx.toe_clearances    # compiler.py:3490
ctx.object_tracks     # from frame.objects
```

### 3.4 Action-keyed registry

The 35 metric-producing branches are already flat, sequential, and self-contained
(`if <action> in program: compute 4-6 keys`). They translate one-to-one into:

```python
BODY_ANALYZERS: dict[BodyAction, Analyzer] = {...}   # 13
OBJECT_ANALYZERS: dict[ObjectAction, Analyzer] = {...}  # ~8
```

Honest accounting: this is 35 units of work, each with its own thresholds. It is not one
refactor.

### 3.5 Equivalence harness

The safety net that makes this landable. A test that, for every stored result in a
fixture set, runs the new analyzer and asserts **exact equality** against the persisted
`metrics.json`:

```python
def test_analysis_matches_persisted_metrics(result_dir):
    scene, program = apply_overrides(load_request(result_dir))
    clip = load_clip(result_dir)
    assert analyze(clip, program, scene) == json.load(result_dir / "metrics.json")
```

Run per extracted path; a path is not done until its equivalence test is green.

---

## 4. Sequencing — five PRs

Phases match the natural seams, each independently mergeable.

| PR | Contents | Effort | Risk |
| --- | --- | --- | --- |
| **02a** | Module skeleton, `CheckResult`, `AnalysisContext`, registry, equivalence harness. Move the 7 already-pure helpers + all of `quality.py`; re-export from `compiler.py` | ~1 day | none — mechanical, zero behavior change |
| **02b** | Persist `support_constraints`, `climb_support_constraints`, `current_yaw` into metrics. Port full-body's 13 action blocks | ~4–6 days | medium — bulk of the metric mass |
| **02c** | Composite + gesture/strike/grab. Converge `arm_landmarks` onto `RigKinematics`; fix the `forearm_rotation_cycles` echo | ~2 days | low — composite is nearly free |
| **02d** | Object interaction + handoff; re-derive lifecycle from `frame.objects` | ~4–5 days | high — numeric drift risk in `rolling_angle_rad`, `attachment_slip` |
| **02e** | Sequence; re-split by `sequence_step_ranges_s` | ~3–4 days | highest — bridge frames, local→world transform |

Total ≈ **2.5–3.5 weeks** for full extraction with verified byte-identical output.

**02a alone unblocks a great deal** — it delivers the contract, the context, and all of
`quality.py` as a testable module, which is enough for [04](04-anatomical-frame.md) to
start. Do not block the anatomy work on 02d/02e landing.

---

## 5. Test plan

- Equivalence tests per path against stored results (§3.5) — the primary gate.
- Unit tests per check with synthetic clips: a hand-built two-frame clip with a known
  violation asserts the check fires with the expected `measured` and `severity`.
- `test_registry_coverage.py` — every `BodyAction` and `ObjectAction` enum member has a
  registered analyzer; a new enum member with no analyzer **fails the suite**. This is
  what stops the deterministic layer silently falling behind the primitive vocabulary.
- Performance: analysis of a 125-frame clip under 100 ms, so the fast tier stays fast.

---

## 6. Risks and open decisions

### 6.1 Risk — numeric drift in 02d

Re-deriving unwrapped `rolling_angle_rad` / `support_spin_angle_rad` from per-frame object
quaternions requires a re-unwrap pass and will not necessarily reproduce the accumulator
bit-for-bit. Mitigation: allow a documented tolerance in the equivalence test for these
specific keys, and record the tolerance in the test rather than silently loosening
equality everywhere.

### 6.2 Risk — ballistic tolerances are intent, not observation

`flight_velocity` feeds `step_reference` (`compiler.py:6841`) as a *tolerance*. Recovering
it post-hoc changes what the check compares against. Treat as a Kind-B carry-over and
persist it, rather than re-deriving.

### 6.3 Open — should the compiler keep populating `ClipResult.metrics`?

Keeping it preserves every downstream consumer (capture diagnostics, flywheel candidate
records, the UI) at the cost of the compiler importing the analyzer. Removing it would be
cleaner but breaks all of them at once.
**Recommendation: keep it.** The compiler calls `analysis.analyze` and stores the result.
The win here is that checks become *runnable elsewhere*, not that the compiler stops
running them. **Decide before 02a.**

### 6.4 Note — 02d and 02e may not be worth finishing immediately

Object interaction and sequence are the two hardest paths and the least load-bearing for
the eval suite, which is centred on anatomy, physics, and signal quality. If time is
short, land 02a–02c, start [04](04-anatomical-frame.md), and leave the last two paths
calling the compiler's inline blocks behind the same registry interface.
