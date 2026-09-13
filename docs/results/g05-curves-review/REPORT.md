# Independent review: bounded quintic prerequisite

**No blocking defect found in the scalar tangent limiter or its opt-in integration.** This review covers the uncommitted source hashes recorded in `results.json`. It is not a completion audit of G05 and does not establish physical tracking or manipulation success.

## Mathematical check

For interval length h, endpoint positions q0/q1, zero endpoint acceleration, and endpoint velocities v0/v1, the quintic's Bernstein control values are:

`q0, q0+h*v0/5, q0+2*h*v0/5, q1-2*h*v1/5, q1-h*v1/5, q1`.

Matching both velocity signs to the interval secant and imposing `|v0|+|v1| <= 2.5*|q1-q0|/h` orders those control values. The polynomial therefore stays between its endpoint positions throughout the interval and is monotone there. The implementation first removes inconsistent signs, then only reduces magnitudes. A later reduction shared with a neighboring interval cannot invalidate an earlier bound. Extrema and plateaus receive zero shared tangents. Shared velocities and zero endpoint acceleration preserve C2 continuity between these scalar polynomial pieces.

## Independent probes

`probe.py` constructs source-name-neutral curves and checks the actual implementation coefficients by independently solving all real derivative roots on each normalized interval. It also directly verifies the Bernstein ordering and both sides of every internal derivative seam.

- 1,800 curves; 14,543 intervals; 615 exact plateaus.
- Seed 716310; positions in [-3.2, 3.2]; irregular intervals spanning 0.0001 to approximately 20 seconds, including monotone runs, reversals, and plateaus.
- Zero interval bound failures at 1e-10 tolerance; maximum observed floating-point bound error 2.78e-14.
- Zero Bernstein ordering failures at 1e-12 tolerance.
- Maximum normalized velocity and acceleration seam errors: 2.80e-14 and 9.52e-14.
- The public G01 failing middle spans retain their old maxima under explicit `quintic`: 2.8679105938 and 2.8721782581 rad. Under `bounded_quintic`, both maxima are 2.84929952894 rad with exactly the same key values and timestamps.

These numerical probes complement the sufficient mathematical condition; they are not an exhaustive guarantee over arbitrary IEEE floating-point extremes.

## Integration and claim scope

The new enum value makes the choice explicit in serialized programs. Existing `quintic` behavior remains unchanged; the any-robot grounder deliberately opts newly grounded joint tracks into the bounded mode. The compiler rejects its use on task-space targets, avoiding a misleading bound on an unrelated interpolation path. Existing pointwise limit and velocity gates remain in place.

The guarantee is per scalar polynomial. It does not establish a Cartesian straight line, collision freedom between legal IK keys, bounded additive sums, continuity across ownership changes, or a safe task-space refinement. The source comment correctly preserves those distinctions. Reduced tangents can change Cartesian geometry and derivative peaks, so the root's fresh reference and physical diagnostics remain necessary; old leaf certificates are not evidence about newly grounded curves.

Likewise, the C2 statement applies to the polynomial pieces: `CandidateTrajectoryV1.sample` still linearly interpolates separately stored position, velocity, and acceleration arrays. That existing sampled-target interface should not be described as an exact C2 reconstruction of the polynomial, nor should scalar smoothness be presented as measured physical smoothness. This does not invalidate the position bound on the stored trajectory.

No shared source or tests were edited, no model/API calls or broad suites were run, and all probe processes completed. Reproduction: run `probe.py` with the shared Python and this worktree's `core/src` on `PYTHONPATH`.
