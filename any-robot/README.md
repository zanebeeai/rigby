# rigby-generalized-urdf

Hand it an arbitrary robot as a URDF. It measures what the robot is, bakes a
library of certified motion primitives for that particular body, and then drives
it from plain language.

Rigby already turns a prompt into certified humanoid motion, but it is welded to
one body: the rig manifest maps named VRM bones, the planner is prompted as a
humanoid planner, and the gates are human gates (`PELVIS_DRIFT`, `FOOT_SLIP`,
`FALL`). This package generalizes the pipeline to any fixed-base arm or bimanual
platform, and reuses Rigby's certified motion compiler, contracts and artifact
machinery unchanged.

## The one idea

Rigby's planner emits **metric keyframes** -- positions and joint values in
metres and radians. The language model is guessing numbers. That survives when
one body's dimensions are baked into the prompt catalogue; it cannot survive an
arbitrary upload, because nothing tells the model that this arm reaches 0.39 m
and that one reaches 2.05 m.

Talmy's observation about spatial language is that it never commits to those
numbers in the first place. Closed-class spatial terms are *topological*:
magnitude-neutral and shape-idealized, which is why the same *in* serves a
thimble and a volcano. So the semantic layer need not know the body either. It
emits a magnitude-neutral **schema program**, and a deterministic **grounder**
resolves `distal` and `quickly` against *this robot's measured* reach envelope
and velocity limits.

Two further consequences shape the code:

- **A primitive is a product, not an atom.** Talmy decomposes a motion event into
  Figure, Motion, Path and Ground plus the co-events Manner and Cause, and notes
  that English states Path and Manner separately ("run *in*", "limp *across*").
  So this authors `Figure x Path x Ground x Manner`, never "wave" or "pick up" --
  and a closed inventory of 26 schema bindings generates body-specific
  candidates. Certification yield depends on the body, region and compiler
  version; it is measured rather than implied by the inventory size.
- **Semantics live at segment boundaries.** Tversky & Lee found route directions
  and sketch maps share one skeleton: segments punctuated by reorientations, with
  detail concentrated at the action points. So a program is a segment chain, the
  reusable unit is one segment, and the solver only has to be exact at the nodes.

**The falsifiable claim**, and the load-bearing acceptance requirement: *the same
prompt must produce the same schema program on every robot.* Role-normalize the
Figure/Ground bindings, hash, and require the hashes to match. If they ever
diverge, the semantic layer has leaked body knowledge and the design is wrong.

## Status

The current supported scope is fixed-base arms/bimanual platforms and admitted
dexterous effectors. Floating-base locomotion is not certified by this package.
The September 11 review ran all 299 then-existing tests and freshly baked two
exact free-space segments on each zoo body. All 12 segments certified, but their
complete reach/return composition passed on five bodies: the dual arm exceeded
a joint limit at composition. This is a remaining execution gap, even though the
semantic reading agreed on all six bodies.

| | |
| --- | --- |
| Ingest and morphology | implemented for admitted fixed-base morphologies |
| Schema inventory and grounder | implemented closed vocabulary; 26 entries |
| Bake and certified primitive library | free-space schemas; feasibility varies by body/region |
| Prompt path (recognizer, binder, gates) | implemented; composition and coverage gaps remain |
| Benchmark, audit, demos | partial measurement; deferred acceptance requirements are not passes |
| Contact and grasp primitives | **partial** -- see below |

### Where contact actually stands

Built and tested: a grasp scene generated entirely from the robot's own
measurements (block sized to the measured jaw aperture, placed inside the
measured reach envelope, massed inside the measured payload), a force-controlled
closure loop, opposition detection driven by the measured opposition groups, and
five gates -- `grasp_not_achieved`, `object_not_lifted`, `object_dropped`,
`excessive_penetration`, `hidden_weld`.

The fresh September 11 sweep passes on three of five gripper-bearing bodies.
It uses a different object size and mass for each body. This is a
capability-normalized probe, not a same-world transfer benchmark:

| robot | measured lift | outcome |
| --- | --- | --- |
| `zoo_compact_arm` | 41.3 mm | accepted; 5.0 N grip, 0.47 mm penetration |
| `zoo_jaw_arm` | 113.6 mm | accepted; 25.4 N grip, 0.55 mm penetration |
| `zoo_long_arm` | 173.3 mm | accepted; 35.5 N grip, 0.49 mm penetration |
| `zoo_hand_arm` | 27.0 mm | object dropped / not carried |
| `zoo_dual_arm` | 77.2 mm | object dropped / not carried |

`scripts/grasp_report.py` reproduces this kind of probe. The frozen review
measurements are in [grasp-report.json](../docs/research/grasp-report.json).
Contact-bearing schemas are excluded from the ordinary free-space prompt path.
The studio's environment route has a separate contact-task path; it fits world
geometry, object placement and payload to the selected body. Keep that behavior
explicit when comparing robots.

Collision results apply to the model's declared collision masks and exclusions.
Ingest records inseparable overlapping links as exclusions from the general
self-collision gate. The separate humanoid gripper also filters contact pairs:
its arm segments collide with the table/bin, but not the block or one another.
Neither route establishes unrestricted whole-body collision safety.

Repeated deterministic simulation verifies repeatability, not robustness.
Current claims, evidence limitations and the execution goals are recorded in
[the research review](../docs/research/rigby-capability-review-and-research-plan.md)
and [the goal catalog](../docs/research/rigby-verifiable-goals-and-order.md).

## The zoo

Development runs against six generated URDFs, spread deliberately wide -- a
morphology analyser that only ever sees one shape of arm learns nothing:

| robot | DOF | effector | reach | split |
| --- | --- | --- | --- | --- |
| `zoo_tool_arm` | 5 | rigid tool tip | 1.18 m | development |
| `zoo_jaw_arm` | 6 + 2 | parallel jaw | 1.32 m | development |
| `zoo_hand_arm` | 7 + 3 | three-finger hand | 1.37 m | development |
| `zoo_dual_arm` | 2x(5+2) | two jaws | 1.33 m | development |
| `zoo_compact_arm` | 4 + 2 | parallel jaw | 0.39 m | historical holdout; now inspected |
| `zoo_long_arm` | 6 + 2 | jaw + wrist camera | 2.05 m | historical holdout; now inspected |

Reach spans a factor of five. They are real `.urdf` files ingested through
exactly the path an upload takes, with primitive geometry only, so the suite runs
offline with no mesh downloads or licence questions.

## Robots nobody here wrote

The zoo is the weakness of any generality claim: it was generated by this
repository, so it is tidy in exactly the ways real robot descriptions are not.
Six URDFs were fetched from an outside project, pinned by commit, recorded with
their licence and never committed here:

| robot | outcome | what it is |
| --- | --- | --- |
| `iiwa7` | ingests | seven axes, mesh collision geometry, no gripper |
| `kuka_lwr` | ingests | seven axes; wrist hulls authored intersecting |
| `panda` | ingests | seven axes plus a real parallel-jaw hand |
| `xarm6` | **refused** | one link's inertia violates A + B >= C by 13% |
| `pr2_gripper` | **refused** | a gripper with no arm: zero positioning axes |
| `cartpole` | **refused** | a cart on a rail: one |

```bash
uv run python scripts/fetch_exotic.py
```

An upstream URDF is not an upload, so the fetcher does the packaging a real
uploader would: it rewrites `package://` mesh references to paths inside the
model and drops `<gazebo>` plugin blocks, counting both in `references.json`.
Ingest refuses either of those outright, and rightly -- a sandbox that honours
`package://` is not a sandbox.

### What they broke

Every one of these was a defect in this package, invisible to the zoo, found by
the first real robot to hit it. They are covered by `tests/test_general_exotic.py`.

- **The staged copy was deleted too early.** Ingest compiles twice and stages the
  upload in a scratch directory. Deleting it after the first compile is fine for
  a model built from primitives and fails on the first one with meshes, because
  the spec resolves mesh paths against where it was read.
- **Rest was the zero pose.** The Panda's fourth joint has range `[-3.1416, 0]`,
  so zero *is* its limit and the arm folds through itself; the LWR's zero pose
  buries one link 21 mm inside another. Rest now means a pose the robot can
  actually hold -- walk toward the joint-range midpoint, then coordinate-descend
  on penetration depth.
- **Two different defects were being refused as one.** An arm slumped against its
  torso can rotate out of the contact; two links whose convex hulls are authored
  intersecting cannot, at any configuration. The first is a mechanism that would
  fight itself and is still refused. The second is recorded, excluded from the
  self-collision gate, and shown in the studio -- without that exclusion the LWR
  baked a zero percent yield, failing every binding on a contact present in every
  frame of every motion.
- **A frame marker counted as a finger.** Real URDFs mark the tool centre point
  with a massless, geomless link, and the Panda's sits between the jaws. One
  member without a surface made the closure test abandon the measurement
  entirely, so a real parallel-jaw hand was reported as a rigid tool tip.
- **The reachable set was treated as a ball.** It is a shell: an iiwa7 cannot
  bring its tool closer than 173 mm to its own base, an LWR closer than 236 mm.
  The generated arms fold down to nearly nothing, which is why the assumption
  survived. A degree of remove now spans the shell, so `adjacent` means the
  closest the arm can get rather than a tenth of its maximum reach -- a point
  that, on every real arm, lay inside the machine.
- **An impossible inertia read as a parse error.** MuJoCo refuses a body whose
  principal moments violate the triangle inequality, and that refusal surfaced as
  `unreadable_model`. The check now runs on the source text, so the failure names
  the link and the size of the violation.
- **The refusal path could crash.** An unbindable segment carries its own
  `failure_code`, which collided with the one the trace recorder was passed --
  in the one code path whose entire job is not to crash.

## How it works

```
UPLOAD                                     PROMPT
  URDF/MJCF                                  "sweep slowly across in front of you"
    |                                          |
  1 scan raw text, refuse unsafe assets      6 recognize -> MotionSchemaProgramV1
  2 compile, measure morphology                 (no metres, no joints, no floats)
      chains, effectors, DOF roles,            |
      intrinsic frame, reach envelope        7 bind -> does this robot have a
    |                                             certified primitive for it?
  3 derive sites, write actuators              |
    |                                        8 ground -> MotionProgramV2
  4 enumerate afforded bindings                |
      schema x region                        9 compile (certified v2 compiler)
    |                                          |
  5 bake: ground -> compile -> simulate      10 simulate 3x -> gates -> motion
      x3 -> gate -> certify or refuse             or a typed refusal
```

Everything measured is measured, never declared. A gripper is a set of parts
whose surfaces were observed to converge when a joint was driven; a wrist joint
is one that turns the tip without carrying it anywhere; a redundant joint is one
that adds no rank to the tip Jacobian. Link naming is at most a hint about where
to look, and where a hint is used -- there is exactly one, for cameras, which
have no mechanical signature -- the site records that its derivation was a name
hint rather than a measurement.

## Setup

```powershell
Copy-Item .env.example .env
uv sync --extra dev --link-mode copy
```

Generate the zoo, bake its libraries, and render the demos:

```bash
uv run python scripts/build_zoo.py
```

```bash
uv run python scripts/bake_robot.py --out ./robots
```

```bash
uv run python scripts/build_demos.py
```

Inspect what the analyser measured on each robot:

```bash
uv run python scripts/inspect_zoo.py --verbose
```

## Trace Studio

Every run records what each stage decided, not just the verdict. `results/`
holds one directory per run -- the exact prompt, the stage-by-stage trace, the
body-neutral reading, the measurements it grounded against, the gates, and the
clip -- and `results/studio.html` browses them.

```bash
uv run python scripts/build_studio.py
```

Then open `results/studio.html` directly, or serve it:

```bash
uv run rigby-general
```

The studio is at `http://127.0.0.1:8020/`, alongside `/api/v3/results`,
`/api/v3/results/{id}` and `/api/v3/health`. The page inlines its traces but
references clips relatively, so it works either way.

Five views:

- **Runs** -- prompt, clip, and the full pipeline trace. The planning stage
  records *which surface phrase fired which schema*, which is the difference
  between "the planner got it wrong" and "the word *across* is matching a sweep
  here".
- **Schema invariance** -- runs grouped by role-normalized hash. One group
  spanning six robots is the load-bearing claim holding, in a form you can read
  rather than take on trust.
- **Contact** -- every grasp probe, held or not, each with its clip. Contact
  probes bypass the planner deliberately: contact schemas stay unafforded until a
  grasp certifies, so asking for a pick *in words* is genuinely refused, and the
  probe drives the physics directly so the grasp itself can be measured. All six
  are rendered, because a failed grasp is the more informative clip.
- **Primitive libraries** -- what each robot certified, and what it refused and
  why. The refusals are half the product: they are the robot's declared
  boundary, and the binder quotes them back when asked for something outside it.
- **Audit** -- G-GEN-1, requirement by requirement.

Refused runs are recorded and browsable alongside the successes. A gallery of
things that worked says nothing about what happens when the system is asked for
something it cannot do, and the refusal stages -- firewall, planning, binding,
compilation, certification -- each mean something different.

## Verification

```bash
uv run pytest -q
```

The goal is a machine-checkable contract in `acceptance_criteria.general.yaml`,
reported requirement by requirement, never averaged:

```bash
uv run python -m rigby_general.audit
```

## Relationship to rigby-poc

Path-depends on `../rigby-poc` and reuses its certified machinery rather than
reimplementing it: the `MotionProgramV2` IR, the motion compiler and its phase
retimer, the artifact and hashing contracts. The new layer sits **above**
`MotionProgramV2` -- the grounder's output is an ordinary v2 program.

Four seams are worth knowing about, because each is a place the v2 code could not
be used as-is and the reason is not obvious:

- **`RigAssetManifestV1` refuses `free_root=False`.** A fixed-base arm can never
  satisfy it. Rather than fabricate a free root -- which would defeat the
  `BASE_DRIFT` gate that proves the base stayed bolted down -- this package
  declares `CompilerRigProtocol`, the four members the v2 compiler actually
  reads, and a test AST-scans the base tree to confirm it still reads no more.
- **The v2 simulation runtime needs a free root joint** to track, so it is
  replaced with a fixed-base rollout.
- **The v2 controller carries one global gain** chosen for a 74 kg humanoid.
  Joint inertias across this zoo span four orders of magnitude; applied to a
  2.4 kg desktop arm that gain flings the arm rather than steering it (measured
  peaks of 191 rad/s against a 3.2 rad/s limit). Replaced with computed torque,
  which asks the model for the mass matrix at the current configuration and so
  needs no per-robot tuning at all.
- **The v2 compiler solves IK at every frame**, which is exact and far too slow
  to bake a library: 22 s at 30 Hz, 291 s at 120 Hz, for one two-segment motion.
  The grounder solves at the handful of points where the path actually turns and
  hands the compiler a joint-space track to interpolate -- 0.004 s to ground and
  0.3 s to compile.

Neither v2 tree is tracked by git, so the dependency has no commit to name.
`config.base_tree_fingerprint()` hashes the `rigby_core` sources actually imported,
`/api/v3/health` reports it, and every baked primitive records it.

## Layout

```text
src/rigby_general/
├── contracts.py      robot morphology, manifest, compiler protocol
├── schema/           the magnitude-neutral IR and its sealed inventory
├── ingest/           URDF/MJCF loading, integrity, site and actuator writing
├── morphology/       chains, effectors, frames, reach envelope
├── grounding/        magnitude-neutral -> metric, and the IK behind it
├── gates/            computed-torque rollout, morphology-appropriate gates
├── bake/             binding enumeration and the bake runner
├── primitives/       certified records, refusals, exact-match retrieval
├── planner/          plain language -> schema program
├── binding/          schema program -> this robot's certified primitives
├── run.py            the whole prompt path
├── trace.py          the durable record of how a prompt became a motion
├── audit.py          the goal, requirement by requirement
└── app.py            /api/v3 and the trace studio
```

## Sources

- Talmy, *Semantics and Syntax of Motion* (1975) -- the Figure/Motion/Path/Ground
  decomposition, and the Path/Manner separability that makes a primitive a
  product rather than an atom.
- Talmy, *How Language Structures Space* (1983) -- schematization as topological
  and magnitude-neutral, and the Figure/Ground assignment criteria a kinematic
  tree already satisfies.
- Tversky & Lee, *How Space Structures Language* (1998) -- shared segment
  structure across route directions and maps, and where descriptive detail
  concentrates.
