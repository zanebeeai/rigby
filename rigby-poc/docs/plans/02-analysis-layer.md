# PR 02 — Extract the analysis layer

Status: 02a landed; 02b–02e not started.
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

*Verified in 02a.* Every write to `metrics` in `compiler.py` occurs at or after line 2853,
except one inside `_failure_result`. Each path's last `frames.append` precedes its first
metric write. No metric is accumulated in a per-frame loop anywhere.

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

*Corrected in 02a.* Those four do not stand alone. Moving them pulled five more
compiler-private definitions across, each equally pure: `_line_segment_distance`,
`_EGO_NEUTRAL_GAZE`, `_semantic_cycle_assertion`, `_rig_profile` and `_identity_pose`.
The two failure appenders `_append_intra_hand_contact_failures` and
`_append_semantic_cycle_failures`, plus the inline travel-signal threshold block at
`_compile_composite`, came too — they are pure `(program, metrics) -> list[str]` and
splitting them from their metrics would have been the harder move. All are re-exported
from `compiler.py` under their original private names, so no call site changed.

`_line_segment_distance` carries a latent defect, pinned by
`tests/test_analysis_checks.py` rather than fixed here: when the two segments are exactly
parallel the determinant collapses, the solver stops searching along the first segment and
reports the distance from its start point instead of the closest approach. Unreachable for
its one caller — `parallel_forearm_metrics` tolerates 20 degrees of axis error, and at
0.25 m segment lengths the 1e-10 epsilon needs parallelism to about 1e-4.

### 1.3 The general FK primitive already exists

`kinematics.py:51` `RigKinematics` parses the GLB and evaluates the true hierarchy for
**all 68 nodes / 52 canonical bones including all 30 finger bones**, from a plain
`Mapping[str, BonePose]` — i.e. exactly `frame.bones`. It is completely generation-free:
`world_matrices` (109), `canonical_positions` (144), `fingertip_positions` (151),
`canonical_world_rotation` (174). Cached via `rig_kinematics()` (394).

The full-body path already uses it post-hoc (`compiler.py:2875`). This is the single
biggest enabler and it is already written.

*Verified in 02a.* The GLB carries 68 nodes; the rig profile's `bone_map` has 52 entries
and every one resolves to a node, 30 of them finger bones. `RigKinematics.__init__` reads
only the GLB and the profile JSON, and every evaluation method takes a
`Mapping[str, BonePose]`. Confirmed generation-free.

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

### 1.5 Traps

Both confirmed in 02a, with two more found while extracting.

- `store.py:49` persists `request.program`, which is the **pre-override** program — and
  `store.py:48` does the same for the scene. An analyzer reading stored artifacts must
  read `request.json` and call `compiler.apply_overrides(request)` (`compiler.py:193`,
  already public) to get the effective `(scene, program)`.
  `analysis.artifacts.load_analysis_inputs` does exactly that;
  `tests/test_analysis_equivalence.py` demonstrates the divergence with a `hand` override.
- `phase_ranges` timing is **not** `primitive.parameters.duration_s` — it is inflated by
  ~70 lines of per-action frame-count rules (`compiler.py:1419-1492`), then taken as
  `max(duration_s, (frame_count - 1) / fps)`. Do not re-derive it; read
  `metrics["phase_ranges_s"]`, already persisted at `compiler.py:2854`.
  `AnalysisContext.phase_ranges` is the only supported access.
- **`presentation_ranges` is not in `phase_ranges_s`.** It is built while frames are
  emitted and never persisted. For gesture and strike it coincides with the phases of a
  fixed set of kinds, so reading it back is exact. For **composite** it does not: the
  window starts part-way into each phase — `start + duration * 0.50` for the
  `parallel_forearm_travel_setup` label, `start + min(0.15, duration * 0.25)` otherwise,
  and `recover` phases are excluded entirely. Reconstructible from `kind` and `label`,
  which `phase_ranges_s` carries, but not by reading the interval directly. Needed by 02c;
  implemented in `AnalysisContext.presentation_ranges`.
- **The angular-kinematics bone set is per compile path, not per action.** Full body uses
  ten bones including feet, sequence uses ten including both forearms, object interaction
  uses the three bones of the active arm, and gesture/strike get theirs from
  `evaluate_gesture_structure`. All four are derivable from `intent` plus `program.hand`,
  but an analyzer that assumes one list will silently disagree on three paths.

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

As built by 02a; the unwritten modules are the 02b–02e targets.

```
src/rigby_poc/analysis/
  __init__.py          analyze(), validate(), structural_failures()
  contract.py          CheckResult, layers, severity helpers
  context.py           AnalysisContext: FK cache, world positions, phase ranges
  registry.py          action registration + port status
  equivalence.py       which metric keys the layer owns, and what is still deferred
  artifacts.py         load a stored result through apply_overrides
  rig.py               rig profile, identity pose, neutral gaze
  geometry.py          line_segment_distance
  safety.py            from compiler._safety_metrics
  gesture.py           from quality.py, incl. angular velocity/accel/jerk
  forearm.py           from compiler._parallel_forearm_metrics
  contact.py           from compiler._intra_hand_contact_metrics
  semantic.py          from compiler._semantic_cycle_metrics
  full_body/           one module per BodyAction — 12 of them          (02b)
  composite.py                                                          (02c)
  objects.py                                                            (02d)
  sequence.py                                                           (02e)
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
BODY_ANALYZERS: dict[BodyAction, AnalyzerEntry] = {...}    # 12, not 13
OBJECT_ANALYZERS: dict[ObjectAction, AnalyzerEntry] = {...}  # 9, not ~8
```

`AnalyzerEntry` carries `analyzer | None` plus the owning module and the PR that ports it,
so an unported action is *declared* rather than absent. `tests/test_registry_coverage.py`
fails the suite when an enum member has no entry at all.

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

*As built in 02a.* The golden corpus (plan 03) was not available, so the harness runs
against a frozen fixture of 14 programs in `tests/fixtures/analysis_equivalence/`,
covering all seven supported intents and every owned metric family. It makes three
assertions per case:

1. every key `analyze` returns is byte-identical to `clip.metrics` **from the same
   compile run** — this proves the extraction is behaviour-preserving and can never go
   stale;
2. the whole output is byte-identical to the frozen snapshot — this catches a compiler
   change that moves both sides together;
3. the key set sits between the declared `required_metric_keys` and `owned_metric_keys`
   in `analysis/equivalence.py` — this catches a check that silently stops producing.

Comparison is on canonical JSON strings, so `-0.0` and `0.0` are distinguished. The
fixture is regenerated with `uv run python tests/fixtures/analysis_equivalence/freeze.py`
and is superseded by the corpus when 03a lands.

---

## 4. Sequencing — five PRs

Phases match the natural seams, each independently mergeable.

| PR | Contents | Effort | Risk |
| --- | --- | --- | --- |
| **02a** | ✅ Module skeleton, `CheckResult`, `AnalysisContext`, registry, equivalence harness. Moved the 4 pure compiler helpers + 5 supporting definitions + 3 failure blocks + all of `quality.py`; re-exported from `compiler.py` | done | none — mechanical, zero behaviour change |
| **02b** | Persist `support_constraints`, `climb_support_constraints`, `current_yaw` into metrics. Port full-body's 12 action blocks | ~4–6 days | medium — bulk of the metric mass |
| **02c** | Composite + gesture/strike/grab. Converge `arm_landmarks` onto `RigKinematics`; fix the `forearm_rotation_cycles` echo | ~2 days | low — composite is nearly free |
| **02d** | Object interaction + handoff; re-derive lifecycle from `frame.objects` | ~4–5 days | high — numeric drift risk in `rolling_angle_rad`, `attachment_slip` |
| **02e** | Sequence; re-split by `sequence_step_ranges_s` | ~3–4 days | highest — bridge frames, local→world transform |

Total ≈ **2.5–3.5 weeks** for full extraction with verified byte-identical output.

**02a alone unblocks a great deal** — it delivers the contract, the context, and all of
`quality.py` as a testable module, which is enough for [04](04-anatomical-frame.md) to
start. Do not block the anatomy work on 02d/02e landing.

### 4.1 What `analyze` owns after 02a

Per intent, verified byte-identical to `compiler.py`. Everything else in `clip.metrics` is
still computed inside the compiler and is listed in `analysis/equivalence.py` under
`DEFERRED_TO_COMPILER` with its PR.

| Intent | Owned families |
| --- | --- |
| gesture | safety, gesture structure, shake oscillation, angular |
| strike | safety, gesture structure, angular |
| grab | safety |
| composite | safety, intra-hand contact, gaze, semantic cycle, parallel forearm |
| full body | safety, semantic cycle, angular |
| object interaction | safety, angular |
| object handoff | safety |
| sequence | safety, angular |

Two path-specific quirks are reproduced deliberately rather than cleaned up, because 02a
is a move: the gesture/strike paths fold `wrist_swing_twist_limit_violations` into
`joint_limit_violations` after the fact and republish `self_collision_frames` as
`unresolved_non_hand_collisions`; and the handoff path alone skips angular kinematics.

---

## 5. Test plan

- Equivalence tests per path against stored results (§3.5) — the primary gate.
- Unit tests per check with synthetic clips: a hand-built two-frame clip with a known
  violation asserts the check fires with the expected `measured` and `severity`.
- `test_registry_coverage.py` — every `BodyAction` and `ObjectAction` enum member has a
  registered analyzer; a new enum member with no analyzer **fails the suite**. This is
  what stops the deterministic layer silently falling behind the primitive vocabulary.
- Performance: analysis of a 125-frame clip under 100 ms, so the fast tier stays fast.

*Measured in 02a*, best of five, 125 frames, warm rig cache: gesture 17 ms, full body
36 ms, strike 42 ms, sequence 48 ms, composite finger-count 66 ms, composite travel signal
85 ms. The target holds, but the heaviest path has only 15 ms of headroom and the cost is
one full 68-node FK evaluation per frame. The committed assertion is a 300 ms regression
ceiling rather than the target, because a tight gate on unknown CI hardware would flake
instead of informing. If 02b–02e make analysis meaningfully heavier, the fix is to
vectorise `RigKinematics.world_matrices` — it builds 52 rotation matrices one
`scipy` call at a time, which is over half the total.

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

### 6.3 Resolved — the compiler keeps populating `ClipResult.metrics`

**Decided 2026-08-16, during 02a. Recommendation confirmed.** The compiler continues to
populate `ClipResult.metrics`; `analysis.analyze` is a second, independently verified path
to the same numbers, not a replacement.

Three findings from the extraction reinforce the plan's reasoning:

- The blast radius is 352 references across 25 files, including `frontend/src/main.ts:865`,
  which renders `clip.metrics` straight off the compile response with no second round
  trip. Removing population means adding an analysis endpoint and a client change for no
  user-visible gain.
- `ClipResult.metrics` and `ResultSummary.metrics` are both required pydantic fields, and
  `Failure.details` is set to the metrics dict — a compile failure carries its own
  evidence. Dropping population is a schema change plus a migration for every persisted
  result, and it makes failures undiagnosable where they occur.
- The dependency direction costs nothing. `compiler` imports `analysis`; `analysis` imports
  only `models`, `kinematics` and `primitives` — not even `physics`, so checks run where
  MuJoCo is absent. `tests/test_analysis_imports.py` pins that invariant.

### 6.4 Note — 02d and 02e may not be worth finishing immediately

Object interaction and sequence are the two hardest paths and the least load-bearing for
the eval suite, which is centred on anatomy, physics, and signal quality. If time is
short, land 02a–02c, start [04](04-anatomical-frame.md), and leave the last two paths
calling the compiler's inline blocks behind the same registry interface.
