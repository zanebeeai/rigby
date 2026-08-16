# PR 08 — Threshold consolidation

Status: proposed, not started.
Scope: one versioned, cited home for every numeric threshold; fix the contradictions and
vacuous gates the inventory surfaced.

Depends on: [02 — analysis layer](02-analysis-layer.md) (checks must exist before their
thresholds can be centralized).
Blocks: [10 — eval redesign](10-eval-redesign.md).

---

## 1. Problem

Thresholds are spread across three config files and roughly twenty code sites. Some are
duplicated consistently, some contradict each other, several gates are vacuous, and
**18 keys in `acceptance_criteria.yaml` are never read by any Python file at all** — so
editing them changes nothing while appearing authoritative.

### 1.1 Real contradictions

| # | Issue | Detail |
| --- | --- | --- |
| 1 | **`semantic_match` never gated** | `acceptance_criteria.yaml:12-17` requires ≥4; `judge.py:893-895` gates only `overall`, `anatomical_naturalness`, `gesture_recognizability`. Fixed in [07a](07-judge-harness.md) |
| 2 | **`max_root_drift_m` differs 1000×** | criteria `:91` = 0.001 m; `compiler.py:607` rejects at > 1e-6. The published tolerance is meaningless |
| 3 | **`max_foot_drift_m` is vacuous** | criteria `:92` = 0.001, but `compiler.py:612` emits the literal `"foot_drift_m": 0.0`. `evidence.py:481` compares that literal against the criterion. **The gate can never fail** |
| 4 | **Shoulder origin rounded, tolerance zero** | `primitives.py:467` = (±0.174, 1.446, −0.065); profile `:19,23` = (±0.1737, 1.4457, −0.0652). `orientation.py:36-46,80-82` computes reach from the profile and rejects at `reach > max_reach + 0.0`. A target generated against the rounded value can exceed profile reach by ~0.8 mm and fail a zero-tolerance gate |
| 5 | **`required_views` has two vocabularies** | criteria `:10` = `["ego","orbit"]`; `:33` = `["egocentric","orbit"]`. Runtime uses `("ego","orbit")`. Neither key is read — inconsistent dead config |
| 6 | **Discontinuity threshold has no config home** | criteria `:95` says `max_discontinuities: 0`, but what *counts* (0.35 rad/frame) exists only at `compiler.py:580`, with a competing 0.14 rad subdivision rule at `compiler.py:8128` |
| 7 | **Generator margins shadow validator limits** | `primitives.py:143,146` use MAX_FOREARM_TWIST 1.30 / MAX_WRIST_TWIST 0.12 while validators use 1.35 / 0.18. The comment at `:139-142` says the margin is intentional, but it is a magic number rather than a derivation of the config it shadows |

### 1.2 Consistent duplicates — maintenance hazards

Same value, many definitions, no single source:

- **FOV 94.0 — five sites**: criteria `:11`, `motion_quality_reference.json:27`,
  `capture.py:23`, `judge.py:470`, `scene.ts:84` + `capture.ts:101`
- **1600×900 — four sites**: `motion_quality_reference.json:29-30`, `capture.py:21-22`,
  `judge.py:468-469,491,881`, `capture.ts:33-34`
- **wrist/forearm limits — two config files**: `motion_quality_reference.json:21-23` and
  `rig_profiles/…:110-112`. Only the former is read; the rig-profile copy is silently
  unused and free to diverge
- **Neutral gaze (0,−0.65,1) — three sites**: `camera.ts:7`, `compiler.py:4928`,
  `quality.py:104`
- **Arm lengths — two sites**: profile `:29-30` vs `primitives.py:135-136`
- **candidate_count 5 / max_rounds 2 / max_model_calls 4**: criteria `:6-8` vs
  `flywheel.py:76,144`, `models.py:961`, `pipeline.py:167`

### 1.3 No threshold carries a source

Not one numeric limit records where it came from. The clearest casualty is
`motion_quality_reference.json`, whose global velocity, acceleration and jerk ceilings
were derived from **six hand-picked clips of one gesture** and are applied to every motion
family including strikes and locomotion.

---

## 2. Goals and non-goals

**Goals**

- G1. Every threshold defined exactly once, in versioned data.
- G2. Every threshold carries `source`, `derived_from`, and `applies_to`.
- G3. Contradictions §1.1 resolved, each deliberately.
- G4. A test that fails when a threshold is hardcoded in Python or TypeScript.
- G5. Shared values reach the frontend by generation, not by retyping.

**Non-goals**

- Re-deriving threshold *values*. Where a value is wrong (the global kinematic ceilings),
  this PR records that it is provisional and unsourced; correcting it needs the corpus and
  belongs to [10](10-eval-redesign.md).
- Changing generator margins. §1.1 item 7 is documented and left in place, with the margin
  expressed as a derivation rather than a literal.

---

## 3. Design

### 3.1 Layout

```
config/
  thresholds.v1.json      # all numeric gates, cited
  anatomy/rom.v1.json     # from PR 04
  camera.v1.json          # FOV, resolution, gaze — shared with the frontend
  rig_profiles/…          # rig geometry only; limits move out
```

Entry shape:

```json
"physics.root_drift_max_m": {
  "value": 1e-6,
  "unit": "m",
  "applies_to": ["*"],
  "source": "compiler.py:607 — fixed-root contract, exact-zero intent",
  "derived_from": "implementation invariant",
  "note": "supersedes acceptance_criteria.yaml max_root_drift_m 0.001, which was never enforced"
}
```

`source` is mandatory. `"provisional — unvalidated"` is an acceptable and honest value; an
absent one fails the schema.

### 3.2 Per-family thresholds

The single most consequential structural change. Kinematic ceilings become indexed by
motion family:

```json
"signal.angular_velocity_max_rad_s": {
  "default": 9.5,
  "by_family": {"gesture": 9.5, "strike": 22.0, "full_body": 18.0},
  "source": "default from motion_quality_reference.json (n=6 gesture clips) — PROVISIONAL",
  "derived_from": "6 hang-ten clips, runs 000018/000019"
}
```

A punch legitimately exceeds a wave's ceiling, so one global number is simultaneously too
loose for gestures and too tight for strikes.

### 3.3 Frontend generation

`camera.v1.json` is the source; a small build step emits `frontend/src/generated/camera.ts`.
`npm run build` fails if the generated file is stale. This is what stops FOV drifting
across the Python/TypeScript boundary — currently five hand-maintained copies.

### 3.4 Resolutions for §1.1

| # | Resolution |
| --- | --- |
| 1 | Fixed in [07a](07-judge-harness.md); criteria becomes the single source |
| 2 | Adopt 1e-6 as the real contract; delete the 0.001 claim |
| 3 | **Either measure foot drift or delete the gate.** Recommend measuring it in [02](02-analysis-layer.md) — a silently-passing safety gate is worse than none |
| 4 | Delete `primitives.py:467`; read the profile. Keep tolerance at 0.0 once both sides agree |
| 5 | Delete the dead keys, or wire them up. Recommend wiring — they are the documented contract |
| 6 | Move 0.35 rad into `thresholds.v1.json` as `signal.discontinuity_rad`; state its relationship to the 0.14 subdivision rule |
| 7 | Express margins as `validator_limit * margin_factor`, with the factor cited |

### 3.5 Dead-config guard

`test_no_orphan_thresholds.py` asserts every key in the config is read by at least one
call site, and `test_no_hardcoded_thresholds.py` greps for suspicious float literals in
check code. Together they stop the config from drifting back into decoration.

---

## 4. Sequencing — three PRs

| PR | Contents | Risk |
| --- | --- | --- |
| **08a** | `thresholds.v1.json` + loader + schema; port config-file values; no code changes | none |
| **08b** | Repoint Python call sites; resolve §1.1 items 2, 4, 6, 7; delete duplicated constants | medium |
| **08c** | `camera.v1.json` + frontend generation; resolve items 3, 5; orphan/hardcode guards | low |

---

## 5. Test plan

- `test_threshold_schema.py` — every entry has `value`/`by_family`, `unit`, `source`,
  `applies_to`.
- `test_no_orphan_thresholds.py` — every key is read somewhere.
- `test_no_hardcoded_thresholds.py` — no bare float literals in check code outside the
  loader.
- `test_frontend_camera_generated.py` — generated TS matches the JSON.
- `test_threshold_migration_equivalence.py` — for every corpus case, verdicts are
  **identical** before and after migration, except the four cases where §3.4 deliberately
  changes behaviour, which are asserted explicitly by name.

That last test is what makes 08b safe to land.

---

## 6. Risks and open decisions

### 6.1 Risk — item 3 will start failing clips

Foot drift is currently hardcoded to `0.0` and passes always. Measuring it for real will
fail some existing motion. That is the correct outcome but it is a behaviour change, so
land it in report-only mode first, exactly as [04](04-anatomical-frame.md) does for ROM.

### 6.2 Open — who owns `by_family` values

The per-family kinematic ceilings in §3.2 do not exist yet; only the gesture number does,
and it is provisional. Deriving strike and full-body ceilings needs the corpus.
**Recommendation: ship 08 with `default` only plus an explicit `PROVISIONAL` source, and
derive `by_family` in [10](10-eval-redesign.md)** once corpus percentiles exist. Do not
invent numbers here — that is precisely how the current ones came to be.

### 6.3 Note — this PR is a prerequisite for honest reporting

[10](10-eval-redesign.md) requires that every published rate carry its n and its baseline.
The analogous rule for thresholds is that every gate carry its source. Without 08, a
calibration report cites limits nobody can defend.
