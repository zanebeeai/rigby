# G05 progress: native physics timing and the remaining joint-limit failure

Certification now follows each model's actual simulator time. Previously it
advanced the reference by 1/240 second per native physics step even when the
model timestep was 0.002 second. A trajectory labeled 0.200 seconds therefore
ran for only 0.096 seconds. The repair preserves the authored reference,
controller settings, model timestep, limits and physical gate thresholds.

Recorded tracking and contact measurements now use positions reconstructed
from the recorded state. MuJoCo's post-integration position/contact cache can
still describe the preceding state; the old code could miss a collision formed
on the final tick. Reconstruction uses separate data and leaves the live
solver state untouched.

## Verification

Eleven real-physics regressions cover multiple timesteps, off-grid reference
ends, reference/control/state timestamp agreement, independently computed
tracking error, final-tick contact detection, repeated trace identity, invalid
physics timesteps and nonzero reference origins. The complete any-robot suite
passed **310 tests** with the existing Starlette deprecation warning.

| Model timestep | Requested reference | Old actual physics duration | Repaired actual duration |
|---|---:|---:|---:|
| 0.002 s | 0.200 s | 0.096 s | 0.200 s |
| 1/240 s | 0.200 s | 0.200 s | 0.200 s |
| 0.003 s | 0.200 s | 0.144 s | 0.201 s |

The last native tick may extend an off-grid reference by less than one
timestep. It is recorded at its actual time, rather than relabeled as the
reference endpoint. All repaired probe timestamps agree exactly with times
independently observed immediately after `mj_step`.

The [diagnostic report](g05-clock-diagnostic/report.json) records source commit
`8d4d1c7cbeb0986e471dd1d743711d1b8e0d3119`, the old source at
`cfbe16e90544fc22b3acc9d95ce7b2d9cc6a24ba`, actual/report clock arrays, controls,
joint state, tracking errors, fresh primitive records and composition outcomes.
[Validation metadata](g05-clock-validation.json) pins the payload manifest and
test reports. Reproduce with:

```sh
python any-robot/scripts/audit_physics_clock.py new-output/clock-diagnostic
python docs/results/g05-transition-diagnostic/polynomial_repro.py
```

The diagnostic loads only public zoo descriptions and already published
baseline cases. It makes no API/model calls. This branch is independent of the
unmerged G01–G03 changes and explicitly uses the legacy intake profile.

## Fresh six-body probe

The exact reviewed prompt was **“reach out as far as you can and then come
back”**. All twelve exact-region leaf primitives certified with duration scale
1.0. Five composed motions passed all existing physical gates, each with three
identical repeated trace hashes and zero region substitutions. The dual-arm
composition remains correctly refused before execution.

| Body | Authored reference duration, approximately | Recorded physical duration | Outcome |
|---|---:|---:|---|
| Compact | 30.510 s | 30.510 s | Certified |
| Dual | 60.591 s | No execution | Joint-limit refusal |
| Hand | 49.246 s | 49.246 s | Certified |
| Jaw | 47.845 s | 47.846 s | Certified |
| Long | 67.454 s | 67.456 s | Certified |
| Tool | 52.438 s | 52.438 s | Certified |

The repair changes actual physical duration relative to the erroneous G01
clock, while leaving the authored duration unchanged. This is **not** a pass
of G05's duration criterion. That criterion still needs an explicit comparison
against both the recorded baseline and the physically consistent reference;
any requested-duration change needs independently justified physical bounds.
No limit, gate or time budget was relaxed to obtain these results.

## Independent dual-arm diagnosis

The [full diagnostic](g05-transition-diagnostic/REPORT.md), captured on G03,
shows that structural ingestion preserves the same physical selected arm and
the same failing composed reference. Four legal neighboring scalar IK keys
produce quintic peaks of **2.867910594 and 2.872178258 rad**, above the unchanged
**2.85 rad** joint limit. The first violation lies inside the return phase,
rather than at its boundary. Both standalone leaves remain legal.

The [compact inputs](g05-transition-diagnostic/offending-keyframes.json) and
[NumPy-only reproduction](g05-transition-diagnostic/polynomial_repro.py) solve
the derivative roots and verify the overshoots independently of Rigby, IK or
simulation. Uniform time scaling leaves the normalized polynomial shape
unchanged and cannot repair this position violation.

The next repair should constrain shared interpolation tangents, preserve legal
keyframes and continuity, and check continuous-interval bounds as well as
velocity, acceleration and task-space geometry. A blanket output-position clip
would not establish a valid reference. The diagnostic also distinguishes a
Cartesian return from an internal joint-state reset: the requested return ends
2.914 mm from the initial site, followed by an additional 27.633-second recovery
that restores joint configuration. Recursive primitive contracts must make
that terminal-state distinction explicit.

## Remaining G05 requirements

**G05 remains in progress.** The following are still required: repair and verify
the dual-arm interpolant, use the structural capability profile and full G01
physics replay, freeze and execute at least 20 start/pace variants per body,
retain the 19/20 feasible-success threshold and typed invalid-request accounting,
resolve the duration comparison, and produce D05's synchronized six-body and
old/repaired dual-arm videos. The complete research catalog remains unchanged.
