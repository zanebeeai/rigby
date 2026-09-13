# G05 diagnostic: the dual-arm reach/return refusal

The G01 refusal persists unchanged under G03 structural intake. It comes from **quintic interpolation overshooting a joint limit between legal IK keyframes**, not a jump at the segment boundary. Both standalone leaves compile; the composed warm-started return follows a different legal IK branch near a wrist joint limit. Central-difference tangents push the interpolated reference beyond that limit. Slowing the whole trajectory uniformly cannot remove this geometric overshoot.

This bounded diagnostic used G03 commit `7e1da38328e65d865b07130d33b5ee4d56e748d5`, the public dual-arm URDF, and exact G01 archived task/model/manifest records. It compiled seven references and inspected scalar polynomial extrema. No physical simulation, model calls, sealed holdout identities, expected-site ground truth, or new success claims were used. Only this ignored diagnostic directory was written.

## Reviewed prompt and baseline durations

The exact recorded prompt is `reach out as far as you can and then come back` (no final punctuation). Its frozen role-normalized semantic hash is `09191df11fd0af2d0172d92356e8beaabba12206e701ab1684c730ad7fccd1e6`. The requested sequence is `reach_to_edge/distal` followed by `retract_from_point/medial`, with zero region substitutions and neutral manner. Both dual-arm leaf bakes used `duration_scale=1.0`.

Primary evidence locations:

- Original repo: `docs/research/progress/g00-results/embodiment-probe.json` and `docs/research/progress/g00-baseline.md`.
- G01 worktree: `docs/results/g01-report.md`, `docs/results/g01-release/index.json`, and `docs/results/g01-release/zoo_dual_arm/physical.zip`.
- The archive's `task.json`, `robot.json`, `model.xml`, and `execution.json` retain the exact refusal-generating reference and the two separately certified leaves.

| Public body | G01 reference duration (s) | G01 actual physics duration (s) |
| --- | ---: | ---: |
| Compact | 30.509918 | 14.644 |
| Dual | 60.591332 | 0; compilation refused |
| Hand | 49.245976 | 23.638 |
| Jaw | 47.844854 | 22.966 |
| Long | 67.454452 | 32.378 |
| Tool | 52.437660 | 25.170 |

The existing 240 Hz reference advance versus 500 Hz physics clock explains the approximately 0.48 duration ratio for executed baseline cases. Those durations describe old evidence; the G05 clock correction must use the actual simulator time. The dual-arm reference never reached physics, so its refusal is independent of that clock defect.

Dual standalone leaves are 23.181927 s (`reach_to_edge/distal`) and 39.701185 s (`retract_from_point/medial`), including each leaf's own recovery. Their summed durations are not the composed duration because composition re-grounds both semantic segments and appends one program recovery.

## Exact cause and measurements

The same failure appears in all three comparisons: exact frozen G01 program, current G03 legacy intake, and current G03 structural intake. Structural `axis_0005` maps to the source joint `left_joint_4`; the selected Figure remains the physical left palm. The first refused 240 Hz reference sample is **t = 20.3041666667 s, q = 2.8500972673 rad**, above the unchanged upper bound **2.85 rad**. The full program duration remains **60.591332 s**.

The program's authored phases are action 0 [0, 12.739785], action 1 [12.739785, 32.958493], and recovery [32.958493, 60.591332]. The first violating interpolation span is inside action 1, not at a phase seam. Reference sample time differs from authored knot time because `PhaseRetimer` applies a monotone time law.

| Interval | Authored knot times (s) | In-limit knot values (rad) | Central knot slopes (rad/s) | Analytic maximum (rad) |
| --- | --- | --- | --- | ---: |
| Return | 18.036707 → 19.573017 | 2.826298358 → 2.849299529 | +0.108645716 → -0.014006328 | 2.867910594 |
| Reversed recovery | 44.017016 → 45.440577 | 2.849299529 → 2.826298358 | +0.014578381 → -0.133997069 | 2.872178258 |

The peaks occur at authored times 18.708756677 and 44.843332451 s, exceeding the bound by 0.017910594 and 0.022178258 rad. Those values are roots-of-derivative extrema, not a coarse sampling estimate.

The standalone reach's `left_joint_4` keys peak at 2.091173142 rad and standalone retract keys at 2.181842039 rad. Composition's warm-started keys peak at 2.849299529 rad; all remain individually legal. Both standalone references have zero out-of-limit polynomial intervals, whereas the composed reference has exactly the two intervals above. G03 changes neither these numbers nor the physical selected arm.

Source chain:

1. `any-robot/src/rigby_general/grounding/grounder.py:636`, `ground`, builds the requested sequence. At `:745` it seeds each next segment with the preceding segment's last qpos; `:749` calls `ik.solve_site_path`. This avoids a discontinuous branch switch and is appropriate.
2. `any-robot/src/rigby_general/grounding/ik.py:65`, `solve_site_path`, performs damped site IK with a previous-waypoint preference. `:165` clips proposals to the declared joint limits. Legal pointwise IK solutions do not establish a bounded interpolant.
3. `grounder.py:1044`, `_joint_keyframes`, assigns joint-travel-dependent times; `ground:828` appends the configurations into one track. `_recovery_keyframes` at `:1158` reverses the configurations at `:1200`, creating a second visit through the near-limit neighborhood.
4. `core/src/rigby_core/motion/ownership.py:21`, `JointTrackSeries`, estimates central-difference knot slopes at `:44`. It zeroes tangents only at identical-value plateaus. It does not constrain tangents at local extrema or in nearly saturated monotone spans.
5. `core/src/rigby_core/motion/timing.py:197`, `QuinticSegment`, uses those velocities and zero endpoint acceleration. Its polynomial need not remain between its endpoint values.
6. `core/src/rigby_core/motion/compiler.py:702` correctly refuses a sampled out-of-limit composed value. The wording “Composed tracks” is generic; this example is one joint track and an interior scalar overshoot.

## Compact regression inputs and independent reproduction

`offending-keyframes.json` contains only four neighboring scalar keyframes per interval, the unchanged limits, and expected extrema. It needs no robot, IK, renderer, library bake, or simulator. `polynomial_repro.py` uses NumPy only and independently reconstructs the central slopes and normalized quintic coefficients, solves derivative roots, and asserts both overshoots to 1e-12 rad.

Run `python any-robot/results/g05-transition-diagnostic/polynomial_repro.py` from G03. It also verifies that zero endpoint slopes produce a bounded scalar interpolant through the exact same two keys. That is a diagnostic of repair feasibility; it does not claim that stopping at every waypoint preserves the intended velocity profile or task-space path.

`diagnose.py` is the bounded full-reference reproduction. `summary.json` records all comparisons and archived source hashes; per-profile JSON files retain grounded programs, refusal details, and interval extrema. `phase-endpoints.json` records the separately measured end states.

## General repair suggestions

Use a shape-preserving, bounded joint interpolation rule instead of unconstrained central slopes. A repair should keep the requested schemas, Figure, region, limits, and legal key configurations intact while constraining tangents near extrema and limits. Quintic Bernstein control-point bounds provide one sufficient interval-wide bound; alternatively check polynomial extrema and solve for admissible shared tangents. Keep derivatives continuous across adjacent intervals and independently recheck speed, acceleration, and task-space geometry after changing interpolation.

Add the two scalar regression cases to the compiler tests and include a continuous-interval extrema gate, so a narrow violation between 240 Hz samples cannot disappear simply by changing sample rate. A blanket final qpos clip can conceal invalid interpolation and change velocities/accelerations; it would not establish the requested motion. Uniformly scaling time changes tangent units but leaves the normalized polynomial shape and the geometric overshoot unchanged, so a clock/duration fix alone cannot address this failure.

Keep the G05 clock correction and this interpolation correction as separately attributable changes with separate tests. Preserve the old G01 failure replay and add a newly recorded repaired reference/physical execution with the unchanged prompt. The G03 structural profile should be included in that verification, even though it is numerically identical on this particular case.

For later transition contracts, distinguish Cartesian goal return from the internal joint-state reset. At the end of action 1, the site is 2.914 mm from its starting location while some joint remains 2.85 rad from rest. The appended 27.632839 s recovery reverses the whole prior path to return all joints exactly to rest; its final site and joint errors are zero. This policy explains an extra excursion and repeated exposure to the same near-limit interval. Reusable superprimitives should state their terminal configuration equivalence explicitly rather than assuming that a Cartesian return also restored the joint configuration.

All diagnostic commands completed and no processes remain.
