# Hand-tool production acceptance record

Date: 2026-08-11

The hand-tool acceptance search was capped at two authored trajectories. Both
programs used `MotionProgramV2`, a task-space left-palm track, articulated human
finger targets, explicit thumb/handle and index/handle contact windows, and the
production motion compiler. The hammer and pelvis remained free joints. The
scene contained no mocap body, weld, object actuator, attachment, teleport, or
post-initialization generalized-position write.

Both attempts were rejected by the production compiler before simulation with
the typed reason `ik_infeasible`. Attempt 1 reported left-palm residuals up to
0.126270 m. Attempt 2 reported residuals up to 0.188605 m. Removing the authored
contact edges did not make attempt 2 feasible, isolating the current blocker to
the task-space IK/refinement boundary rather than contact-window validation.

No production opposing-contact, lift, strike, penetration, repeat, GLB, or
robustness result exists for either attempt because neither program produced a
simulatable trajectory. In particular, neither attempt is certified.

For comparison only, the separate pre-existing legacy joint-keyframe probe on
the unchanged pack completed three exact repeats with two simultaneous digit
contacts, 0.000 m lift, a 9.385633 N maximum head/support force, and 0.001691 m
maximum penetration. That probe failed grasp persistence, lift, lifecycle, and
the required 10 N tool outcome. It is diagnostic evidence, not production
acceptance, and it was not entered into robustness evaluation.

The temporary unverified mass, grip-radius, and friction calibration considered
between attempts was reverted. The committed object pack therefore retains its
original 1.0 kg hammer, 25 mm handle radius, and 0.85 sliding friction.
