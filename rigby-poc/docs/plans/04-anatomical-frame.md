# PR 04 — Anatomical frame and per-DOF range of motion

Status: proposed, not started.
Scope: replace direction-blind scalar joint limits with per-degree-of-freedom anatomical
range-of-motion checks over the whole body, including the 30 finger bones.

Depends on: [02a — analysis layer skeleton](02-analysis-layer.md),
[03 — golden corpus](03-golden-corpus.md) for validation cases.
Blocks: [06 — mutation library](06-mutation-library.md) (ROM mutations need a frame to
mutate along), [10 — eval redesign](10-eval-redesign.md).

---

## 1. Problem

### 1.1 The current check is direction-blind, and worse than it looks

`compiler.py:582-587`:

```python
angle = 2.0 * math.acos(float(np.clip(abs(quat.w), 0.0, 1.0)))
if angle > max(abs(bounds[0]), abs(bounds[1])) + 1e-6:
```

Two separate defects. The quaternion magnitude discards **direction**, so an elbow bent
140° backwards passes `[0.0, 2.75]` exactly as a correct 140° flexion does. And
`max(|lo|, |hi|)` discards **asymmetry**, so the lower bound of `leftLowerArm: [0.0, 2.75]`
is never enforced at all — **elbow hyperextension is currently unpoliced**.

### 1.2 Coverage is 8 bones out of 52

`config/rig_profiles/mesh2motion-human-vrm1.json`:

- `joint_limits_rad` (:115-124) — 8 entries: upper/lower arm and upper/lower leg, both
  sides. Scalar magnitude bounds, as above.
- `anatomical_limits` (:108-114) — wrist and forearm only: `wrist_swing_rad` 0.48,
  `wrist_twist_rad` 0.18, `forearm_twist_rad` 1.35.

**Zero limits exist for** spine, chest, upperChest, neck, head, shoulders, feet, toes, or
any of the 30 finger bones.

### 1.3 `swing_twist_angles` is general; every call site is not

`quality.py:34-52` is fully general — arbitrary quaternion, arbitrary axis. But all three
call sites hardcode axis `[0,1,0]` and are arm-specific (`quality.py:223, 260, 263`). The
sibling wrist metrics at `quality.py:227-229` extract flexion and deviation by raw
`as_rotvec()[0]` / `[2]` — an axis-index shortcut, not an anatomical frame.

### 1.4 The rest data needed already exists — and is currently duplicated

This is the finding that makes the PR tractable. `kinematics.py:51` `RigKinematics` parses
the GLB and exposes, for **all 52 canonical bones including all 30 finger bones**:

- `self.rest[i]` — rest local transform (rotation **and** the child offset that defines
  bone direction)
- `self.rest_world[i]` — rest world transform
- the full parent chain

Two verified facts follow:

1. **`rest_world_pivots_m` (profile :15-27, 11 bones) is redundant.**
   `RigKinematics.canonical_positions({})` reproduces every one of those 11 values exactly.
2. **The six `primitives.py:120-133` constants are redundant.**
   `Rotation.from_matrix(k.rest_world["leftUpperArm"][:3,:3]).as_quat()` equals the
   `_UPPER_ARM_REST_WORLD_XYZW[LEFT]` literal bit-for-bit;
   `_LOWER_ARM_REST_LOCAL_XYZW` and `_HAND_REST_LOCAL_XYZW` match to 1e-7 on both sides.

So the anatomical frame needs **no new asset data**. It is a derivation from data already
parsed at startup.

### 1.5 The rig is axis-consistent

Verified across all 52 bones: **local +Y is the longitudinal bone axis** (cosine against
the child offset ≥ 0.9 everywhere). The only two at 0.898 are `leftHand`/`rightHand`,
purely because their nominal first child is the thumb.

Spot-checked cross-axes are anatomically sensible as-is: `leftUpperLeg`'s rest X-column is
world `[1,0,0]` — the lateral/abduction axis; `leftIndexProximal`'s X-column is
approximately world `-Z` — the finger curl axis.

---

## 2. Goals and non-goals

**Goals**

- G1. A per-bone anatomical frame derived from rest data, for all 52 bones.
- G2. Rotations decomposed into flexion / abduction / twist rather than a scalar.
- G3. A per-DOF limit table covering the whole body, every entry carrying a source.
- G4. Hinge off-axis assertion for elbow and knee.
- G5. Violations reported with severity, duration, and integral — not a boolean.
- G6. Duplicated rest state deleted, with `RigKinematics` the single source.

**Non-goals**

- Coupled ROM envelopes (shoulder rotation as a function of abduction, hip flexion as a
  function of knee angle). Box limits first; coupling is a follow-up once box limits are
  trusted. See §6.3.
- Changing any existing gate's pass/fail behaviour in this PR. New checks land in
  report-only mode. See §3.6.

---

## 3. Design

### 3.1 Derived anatomical frame

```python
# analysis/anatomy/frame.py
@dataclass(frozen=True)
class AnatomicalFrame:
    twist_axis: np.ndarray      # +Y, the longitudinal bone axis
    flexion_axis: np.ndarray    # rest X or Z, per joint convention
    abduction_axis: np.ndarray  # the remaining orthogonal axis
    flexion_sign: float         # +1 / -1, so "flexion" is always positive
```

`bone_anatomical_frame(canonical) -> AnatomicalFrame` derives from `rest[i].rotation` plus
the child offset. Pure, cached, no new data.

### 3.2 Three-DOF decomposition

Generalize `swing_twist_angles` into:

```python
def decompose(local_delta: Quat, frame: AnatomicalFrame) -> DofAngles:
    """-> (flexion_rad, abduction_rad, twist_rad), signed."""
```

Twist about +Y as today; swing projected onto the two derived cross-axes rather than
collapsed to a magnitude.

### 3.3 Limit table

New file `config/anatomy/rom.v1.json`, one entry per `(bone, dof)`:

```json
{
  "leftLowerArm": {
    "flexion":   {"typical": [0, 140], "max": [-10, 145], "unit": "deg",
                  "source": "…", "note": "hyperextension negative"},
    "abduction": {"typical": [0, 0], "max": [-5, 5], "hard_assert": true,
                  "source": "hinge joint — off-axis motion is a defect"},
    "twist":     {"typical": [0, 0], "max": [-5, 5], "hard_assert": true}
  }
}
```

Three deliberate properties:

- **`typical` versus `max`.** Beyond `max` is a hard fail. Sustained occupancy of the band
  between `typical` and `max` is a *naturalness* signal — motion that lives at end-range
  reads as tense — and it is free once this split exists.
- **`hard_assert` for hinges.** This is G4, and it is the cheapest high-value check in the
  whole suite: elbows and knees have one axis, so off-axis motion is unambiguously a
  defect.
- **`source` is required.** An uncited threshold is indistinguishable from a guess, which
  is exactly how the current global velocity ceilings came to be derived from six clips of
  one gesture. A `source` of `"provisional — unvalidated"` is acceptable and honest; an
  absent one is not.

### 3.4 Mirroring

`primitives.py:234-237` applies `side = ±1` for left/right. The limit table is authored
for the left side with an explicit mirror rule — abduction and twist negate, flexion does
not — asserted by a test that mirrored poses produce mirrored decompositions.

### 3.5 Violation records

```python
{"bone": "leftLowerArm", "dof": "abduction",
 "peak_deg": 31.4, "frames": 18, "integral_deg_s": 4.2,
 "band": "beyond_max", "threshold_deg": 5.0}
```

`integral_deg_s` gates; `peak_deg` triages. A 2° single-frame blip is not a 40° sustained
violation, and the current boolean cannot tell them apart.

### 3.6 Rollout in report-only mode

New checks land emitting `CheckResult` with `status="pass"` regardless, and their measured
values recorded. Run across the corpus and the existing `results/` archive, review the
distribution, **then** turn on enforcement per-DOF in a follow-up PR.

Turning 44 bones' worth of new limits on at once would reject a large fraction of
currently-shipping motion, and it would be impossible to tell genuine anatomical
violations from a mis-derived axis. This staging is what makes the PR safe.

### 3.7 Delete the duplicated state

Point `quality.py:55` `arm_landmarks` at `rig_kinematics()`, then delete the six
`primitives.py:120-133` constants and `rest_world_pivots_m` from the profile.

Note this changes gesture metrics slightly — `arm_landmarks` and
`RigKinematics.canonical_positions` do not agree exactly today because they read different
rest sources. Sequence it **with** [02c](02-analysis-layer.md), which already has to
resolve that discrepancy, rather than doing it twice.

---

## 4. Sequencing — three PRs

| PR | Contents | Effort |
| --- | --- | --- |
| **04a** | `AnatomicalFrame` derivation, `decompose`, frame-calibration tests. No limits, no gates | ~2 days |
| **04b** | `rom.v1.json` for all 52 bones; checks in report-only mode; mirroring | ~3 days |
| **04c** | Enforcement per DOF after distribution review; delete duplicated rest state | ~2 days |

---

## 5. Test plan

**`tests/test_anatomical_frame.py` is the load-bearing test.** Everything else in this PR
is meaningless if the frame is wrong, and a wrong frame produces confident garbage that
looks like it is working.

Pose the rig at known configurations and assert the decomposition returns expected angles:

| Pose | Assertion |
| --- | --- |
| Rest / T-pose | every DOF ≈ 0 |
| Elbow flexed 90° | `leftLowerArm.flexion` ≈ 90°, abduction ≈ 0, twist ≈ 0 |
| Arm raised overhead | `leftUpperArm.flexion` ≈ 180°, no gimbal blow-up |
| Forearm fully pronated | twist ≈ 80°, flexion ≈ 0 |
| Fist closed | every finger MCP/PIP/DIP flexion positive and monotonic |
| Knee flexed 90° | `leftLowerLeg.flexion` ≈ 90°, off-axis ≈ 0 |
| Mirrored pair | left and right decompositions mirror exactly |

Plus:

- `test_rom_table_complete.py` — every canonical bone has an entry for every DOF it has;
  every entry has a non-empty `source`.
- `test_rom_detects_injected_violation.py` — inject a known 30° off-axis elbow rotation
  into a corpus clip; assert the check reports `peak_deg ≈ 30`. This is the metamorphic
  test that proves the check measures what it claims.
- `test_rest_state_equivalence.py` — before deleting the duplicated constants, assert
  `RigKinematics` reproduces each one, so the deletion is provably safe.

---

## 6. Risks and open decisions

### 6.1 Risk — the axes are geometrically exact but anatomically labelled by assumption

The derivation is exact; calling a given axis "flexion" is an assumption until validated.
`test_anatomical_frame.py` is the entire mitigation, and it must be written before any
limit value is trusted. If that test does not exist, no number in `rom.v1.json` means
anything.

### 6.2 Risk — finger limits have no calibration source in-repo

`grip_presets` (profile :125-146) is the only finger angle data and it is **authored, not
measured**. Finger ROM values will be provisional, and should be marked so in `source`.
Report-only mode (§3.6) matters most here.

### 6.3 Open — box limits or coupled envelopes

Independent per-axis min/max permits anatomically impossible *combinations*: maximum
shoulder internal rotation is not achievable at maximum abduction, and hip flexion is
limited to roughly 90° with the knee extended versus 120° with it flexed.

**Recommendation: box limits in this PR, with the two highest-value couplings —
shoulder rotation-vs-abduction and hip flexion-vs-knee — added in a follow-up** once the
distribution review shows whether they actually fire. Encoding coupling now, on top of
unvalidated axes, risks debugging two unknowns at once. **Decide before 04b.**

### 6.4 Note — expect this to reject currently-shipping motion

The whole point is that the existing check is nearly vacuous. When enforcement turns on,
some motion Rigby produces today will fail, and some of those failures will be genuine.
Budget time in 04c for triage, and treat "the check is wrong" and "the motion is wrong" as
equally likely hypotheses until the calibration test says otherwise.
