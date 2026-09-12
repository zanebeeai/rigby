# G02: fixed-world contracts and observation boundaries

G02 is locally achieved. All **840 comparisons** (140 registered public world
variants × six zoo bodies) preserve the resolved fixed-world hash. Twelve short
episodes exercise real RGB/joint sensors, a separate acting worker and an
independent simulator evaluator. Four real payload mutations are detected by
offline evidence verification. All sixteen existing general acceptance
requirements retain their original statements and thresholds.

This result establishes benchmark infrastructure. The episodes send zero control
commands for 100 physical steps each, or 0.2 seconds at the registered timestep.
All twelve complete as unscored protocol probes; none succeeds at the placement
goal. They do not demonstrate manipulation, locomotion, VLM judging, recursive
skills or robustness. G02 does not require a video; its evidence is the registered
protocol, resolved manifests and executable access/mutation tests.

| Frozen criterion | Measured evidence |
|---|---|
| G02.A01: distinct strict and normalized modes; reject fitting and semantic substitutions | Both modes execute through the registered runner. Mutation tests reject world fitting in strict mode, changed requested semantics and undocumented normalized values. All six strict probes share one world hash; six normalized probes disclose six different worlds. |
| G02.A02: resolved geometry/dynamics/placements/lighting/goals/physics/start/sensor hashes; separate body sensors | 140 public world variants compile independently, then each accepts six different body models without changing its world hash. Actual-model mutations in geometry, mass, inertia, friction, placement, collision masks, lighting, camera, gravity, timestep and object initialization are rejected. Body-camera changes alter the body/model identity while preserving the world identity. |
| G02.A03: preregister goals, dwell, budgets, perturbations, feasibility, metrics and splits | Protocol and independent feasibility rules were committed before the final probes. Root placement requires whole geometry, low speed and a two-second release dwell. Attempts/time, PCG64 perturbations, public seeds, separate metrics and confirmatory split-custody rules are recorded. Every current run is unscored. |
| G02.A04: evaluator-owned truth; acting access tests; explicit fully observed baseline | 52 core observation tests plus simulator integrations exercise actual subprocess access, malformed/extra fields, privileged inputs, timeouts/crashes and cleanup. Robot encoders use explicit joint addresses, excluding preceding world-object coordinates. Fully observed packets have a separate type and cannot enter the sensor-only worker. |

The independent reviewer found two concrete evaluator errors, both fixed before
the final evidence. A stationary cube between two margin-enabled robot spheres
received opposing forces above 4 N at positive separation; it now fails release
for the entire dwell. A second regression checks a 9.5 ms simulation cap with
2 ms physics steps: execution stops at 8 ms instead of accepting success at 10 ms.
Success exactly at a registered cap remains eligible. The release clarification
has an explicit protocol amendment; the earlier unscored draft is preserved.
The contact-force interpretation follows MuJoCo's documented
[contact model](https://mujoco.readthedocs.io/en/latest/computation/#contact).

The placement predicate does not require a grasp, lift or carry event. Tasks whose
language requires those events must add corresponding predicates before scoring.
The goal is physically supported by a destination platform; a center point inside
the region is insufficient when part of the object remains outside it.

Evidence and provenance:

- [Machine-readable validation](g02-validation.json), [release index](g02-release/index.json), [offline verifier](verify_g02_release.py).
- [840 comparison records](g02-release/world-conformance.json), [resolved world manifests](g02-release/worlds), [body manifests](g02-release/bodies), [twelve episode records](g02-release/probes).
- [Protocol registration](../../any-robot/assets/general/research-protocols/transfer-bench-v1-r1/registration.json), [registered protocol](../../any-robot/assets/general/research-protocols/transfer-bench-v1-r1/benchmark-protocol.v1.json), [feasibility rules](../../any-robot/assets/general/research-protocols/transfer-bench-v1-r1/feasibility-rules.v1.json), [protocol explanation](../../any-robot/assets/general/research-protocols/README.md).
- [Existing acceptance crosswalk](g02-acceptance-crosswalk.json). None of its sixteen full release requirements is marked proven by these infrastructure probes.

The 248 payload files are pinned by release index SHA-256
`4f0088702fbc4c845838f62fafff62c4ac8a1de53ce380d2f728bba567752303`.
The strict seed-zero probe world SHA-256 is
`d0ae926fd89c6c7441c2b8424d07e9e7107bcad195f0c6c89fc0acbd80b065b0`.
The registration SHA-256 is
`1dab70141652d41aae1e511937696f7c9bbf9cb891aecd19ed7565734f32fc49`.
Full source commit and pinned runtime versions are in
[provenance](g02-release/provenance.json). Source code and registration were clean
and committed before generation. The mutation audit operates on temporary copies
and restores each original payload before checking the next mutation.

Validation: **337 any-robot tests pass**, **112 core tests pass**, and the twelve
focused evaluator/runner regression cases pass. The focused cases are included in
the 337 and must not be added as independent coverage. Core validation deselects
the existing Windows subprocess hash-environment test; its unchanged hashing
semantics and platform fix were separately validated in G00 PR 27. No new test
skip or weakened assertion was introduced. JUnit files preserve the results;
older contract/runner XML snapshots are intermediate checks, not additional tests.

Linux CI now installs OSMesa for the real offscreen RGB tests, using the supported
[MuJoCo backend](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rendering/classic/gl_context.py).
That lane is not yet validated: GitHub Actions billing/allowance has prevented jobs
from starting. Local acceptance is distinct from merging through green CI.

From the repository root, after syncing the workspace, verify the existing bundle:

```powershell
uv run --directory any-robot python -m rigby_general.benchmark.audit verify ../docs/results/g02-release --expected-index-sha256 4f0088702fbc4c845838f62fafff62c4ac8a1de53ce380d2f728bba567752303
uv run --directory any-robot python ../docs/results/verify_g02_release.py
```

Generate a fresh audit into a new directory with
`uv run --directory any-robot python -m rigby_general.benchmark.audit create results/g02-reproduction`.
It refuses to overwrite a directory or publish from uncommitted implementation or
protocol files. Fresh wall-clock metadata changes the evidence index; the recorded
world/body comparisons remain independently checkable. This G02 bundle stores
commands and evaluation records, but does not contain the complete physical
integration states needed for G01-style numerical trajectory replay.

The observation boundary covers trusted policy APIs and clean Python interpreter
state. It is not an OS sandbox against hostile code with the same filesystem,
network or process privileges. Sensor adapters are trusted; tests cannot infer a
numeric value's provenance without auditing its producer. Public protocol files
reveal nominal world parameters. No existing sealed robot identities, prompt
identities or human ground-truth labels were read.

Before scored experiments, integrate G01 physical replay, preregister the full
trial roster and independent feasibility map, enforce shared budgets across
retries, and use independent custody for confirmatory splits. The current public
seeds are engineering splits and do not establish unseen topology or prompt
generalization. Normalization, gravity/time invariants and feasibility evidence
are documented explicitly in the protocol rather than inferred from outcomes.

The next research milestone is G03: derive and validate the body capability
contract needed to bind semantic primitives to unfamiliar morphology. G05 must
also correct the previously identified 240 Hz reference / 500 Hz physics-clock
mismatch before its free-space robustness experiment.
