# G05 progress: bounded joint curves

The dual-arm reach/return reference now compiles without exceeding its joint
limits. Its old quintic interpolation overshot between legal IK keyframes
during both the return and recovery. Newly grounded any-robot joint tracks now
use an explicit `bounded_quintic` mode. Legal key values, key times, phase
boundaries, authored durations, physical limits and gate thresholds are
unchanged. Explicit legacy `quintic` retains its previous behavior.

**This completes the bounded-reference prerequisite, not G05.** With the
separately pinned native-clock repair, five bodies pass the existing physical
gates. The dual arm now reaches execution but fails joint-position and actuator
effort gates. These results remain a public, legacy-intake diagnostic; structural
intake, the registered robustness campaign and D05 media are still required.

## Why the curve is bounded

For a zero-acceleration quintic over interval length `h`, endpoint positions
`q0, q1` and velocities `v0, v1`, the Bernstein control values are:

`q0, q0+h*v0/5, q0+2*h*v0/5, q1-2*h*v1/5, q1-h*v1/5, q1`.

Making the velocities follow the secant and limiting their combined magnitude
to `2.5*abs(q1-q0)/h` orders those values. The entire polynomial then stays
between its endpoints. Shared knot tangents only decrease, so satisfying a
later interval cannot invalidate an earlier bound. Extrema and plateaus have
zero tangents; ordinary monotone interiors can retain nonzero velocity.
Shared velocities and zero knot accelerations preserve polynomial C2 continuity.

The runtime trajectory still interpolates its stored position, velocity and
acceleration arrays separately. This change does not make that representation
an exact reconstruction of the C2 polynomial, nor does it prove physical
smoothness, Cartesian intent, collision freedom or bounds after additive
composition/refinement. Existing physical and composed-track gates remain
necessary. Task-space uses of the new mode receive a typed refusal.

## Verification

| Check | Result |
|---|---:|
| Complete any-robot suite | 300 passed, no skips |
| Core suite | 87 passed; one known Windows subprocess-environment test deselected |
| Humanoid compiler, timing, orientation and refinement regressions | 18 passed |
| New bounded-curve tests, included in core count | 29 passed |
| Independent review | 1,800 curves, 14,543 intervals, 615 plateaus |
| Six-body continuous scalar intervals | 1,023 checked |
| Fresh exact-region leaf primitives | 12 certified at duration scale 1.0 |
| Canonical physical compositions | 5 successes, 1 runtime failure |

The deselected core test is the existing Windows PATH issue repaired separately
in PR #27; hashing semantics were not changed here. The any-robot suite reports
the existing Starlette deprecation warning. The independent review found zero
bound/ordering failures, with maximum floating-point bound error
`2.78e-14`. Its [report and reproduction script](g05-curves-review/REPORT.md)
record the exact source hashes reviewed.

The [frozen scalar regression inputs](../../core/tests/fixtures/g05_offending_keyframes.json)
come from the previously published G01 dual-arm program and PR #31 diagnostic.
They remain unchanged. Continuous extrema are found by polynomial derivative
roots, including peaks between sampled times.

| Original failing span | Legacy maximum | Bounded maximum | Upper limit |
|---|---:|---:|---:|
| Return | 2.867910594 rad | 2.849299529 rad | 2.85 rad |
| Recovery | 2.872178258 rad | 2.849299529 rad | 2.85 rad |

## Physical results and remaining failure

The exact prompt was **“reach out as far as you can and then come back”**.
All six used the same semantic program, newly baked exact-region primitives
and zero substitutions. All twelve leaves passed. Three independent executions
per composition produced identical state/control hashes, including the failure.

| Body | Authored duration | Actual native-clock duration | Outcome |
|---|---:|---:|---|
| Compact | 30.509918 s | 30.510 s | Pass |
| Dual arm | 60.591332 s | 60.592 s | Physical gate failure |
| Hand arm | 49.245976 s | 49.246 s | Pass |
| Jaw arm | 47.844854 s | 47.846 s | Pass |
| Long arm | 67.454452 s | 67.456 s | Pass |
| Tool arm | 52.437660 s | 52.438 s | Pass |

The dual arm's reference remains legal, but execution drives `left_grip_left`
to -0.011179 m against a 0 m lower limit, exceeding the unchanged 0.01 margin.
Peak demanded effort reaches 2,922.43 Nm on `left_joint_4_motor` (95 Nm limit)
and 3,194.92 N on `left_grip_left_motor` (45 N limit). Clipped actuator commands
are not substituted for demand when evaluating these limits. The observed
tracking maximum is 0.038904 m. Diagnosis of the physical cause remains separate
from the resolved interpolation defect.

Authored durations match the baseline exactly. The clock repair makes actual
execution roughly twice as long as the old, incorrectly clocked execution;
that distinction must remain explicit when assessing G05's duration criterion.
This prerequisite does not waive that criterion or claim a speed improvement.

## Reproduction and evidence

Source commit: `5a6c0b3bb17eb721da6ba6c144743c24ba872301`.
Baseline grounder: `cfbe16e90544fc22b3acc9d95ce7b2d9cc6a24ba`.
Native-clock source: `8d4d1c7cbeb0986e471dd1d743711d1b8e0d3119` (PR #31).

The probe loads the two historical modules from those Git objects and records
their bytes. It temporarily selects the pinned simulator in process; production
branches remain independent. No API/model calls or sealed holdout inspection
occurred. Source commits must be present locally to regenerate this diagnostic.

```sh
python any-robot/scripts/audit_bounded_joint_curves.py new-output/curves
python docs/results/verify_g05_curves.py
```

The [diagnostic](g05-curves-diagnostic/report.json) includes before/after programs,
continuous extrema, primitive records/refusals, reference arrays, full recorded
native-clock states/controls/demands and all outcomes. The
[validation record](g05-curves-validation.json) pins the payload manifest and
test reports. The verifier recomputes extrema, checks exact key/phase/duration
preservation and validates all six saved physical trace hashes. It does not
replace G01's portable model/world replay or the outstanding D05 videos.
