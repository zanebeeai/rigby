# Rigby v2 completion audit

Audit date: 2026-08-11  
Reference platform: Windows 11 laptop, local MuJoCo, local PostgreSQL/pgvector

This document is the release truth source for the ten implementation goals. A
passing unit test proves only the behavior it exercises. Synthetic ratings,
fabricated benchmark rows, empty databases, and interface-only model adapters
do not satisfy a real execution or evidence gate.

Status meanings:

- **Proven**: implemented and exercised directly with representative evidence.
- **Partial**: useful implementation exists, but the full clause or exit gate is
  not demonstrated.
- **Missing**: the required implementation or evidence does not exist.
- **Contradicted**: current behavior conflicts with the stated requirement.

## Executive status

| Goal | Status | Strongest evidence | Release-critical remainder |
|---|---|---|---|
| 1. Foundation and reproducibility | Proven | Typed immutable contracts, content-addressed artifacts, durable jobs, redacted structured API events, cancellation/leases, applied provenance, and real crash-recovery/replay | No exit-gate remainder |
| 2. Canonical physical human | Proven | Free-root 67-actuator human, articulated fingers, real all-frame/all-profile GLB round trips, explicit audited inertias, fixed-bone and Menagerie reference policies | No exit-gate remainder |
| 3. Scene and object compiler | Partial | Secure MjSpec/MJZ composition, isolated visual/collision assets, six packs, certified physical button and drawer tasks with five robustness survivors each | Successful certified physical completion of the other four task families |
| 4. Timing and motion compiler | Partial | Canonical phase timing, joint/task-space quintics, production SLERP/SQUAD IK, bounded accents, orientation/collision refinement, ownership rejection | Real blinded POC comparison |
| 5. Whole-body MuJoCo | Partial | 240 Hz free-root dynamics, contact-consistent inverse control, stable three-profile standing, native rollout/process isolation, batched exact-five path | Certified physical grasp/lift/hold/release and articulated-task survivors |
| 6. Deterministic certification | Partial | Three exact replays, export/reimport, dynamics/penetration/fall/teleport/object gates, full planned-contact lifecycle/dropout, quaternion and asset/coordinate binding, and two robust production interactions | Certified free-object grasp/lift/hold/release survivor |
| 7. Evidence and best of five | Partial | Motion-effective exact-five joint and task-space variants, 30 FPS three-view evidence, anonymous double-order judging, fallback/call caps | Live model-backed acceptance evidence |
| 8. Certified RAG flywheel | Partial | Hard filters, five namespaces, RRF, N+1 isolation, hard negatives, and a real certified button record in a sealed three-index building release | Admit/run Cosmos before freezing/activating the five-index release; hydrate, re-simulate, and evaluate legacy candidates |
| 9. Human calibration and datasets | Partial | Media-bound blinded packets, retained-file response/attestation verification, Deaf-review artifact binding, strong dataset policies | 200 genuine blinded comparisons/600 ratings, actual Deaf review, real learned-in-betweening admission evidence |
| 10. Release benchmark and migration | Partial | 100-case adversarial firewall pass, resumable evidence-backed supported runner, populated restore, isolated install/startup/rollback, animated renderer p95 under 90 seconds | Configure and execute 300 supported cases; genuine human and RAG A/B evidence |

## Current verification snapshot

- The current v2 suite collects 385 tests: 376 pass and 9 guarded external/live
  integrations are intentionally skipped in the ordinary run. Ruff, compileall,
  and `git diff --check` pass (line-ending notices only).
- The exact current source builds an offline wheel and installs into a new
  isolated environment with the source tree absent from `PYTHONPATH`; both v2
  entry points import and the installed API reports healthy. Wheel SHA-256 is
  `4231141280b15f23eac884ff68897e9fad2486738b66d50a2d2d0b7e7f913dfc`.
- Twelve of fifteen machine-verifiable release requirements pass. Shipping is
  still refused on exactly `supported_benchmark_300`, `human_calibration`, and
  `rag_ab_evaluation`.

## Goal 1 — foundation and reproducibility

Proven:

- Planner, motion, asset/scene, simulation, certification, retrieval, artifact,
  API, and worker responsibilities are separate modules with typed boundaries.
- Frozen versioned contracts, canonical JSON hashing, streaming SHA-256 artifact
  storage, structured failure codes, cancellation, leases, heartbeat, and lease
  recovery are implemented.
- Exact MuJoCo trace replay is independently verifiable and fails closed on
  artifact, environment, and controller drift.
- API/service startup, request, job, replay, and staging lifecycle events are
  structured JSON with propagated request IDs, typed failure codes, route and
  duration fields. Authorization, cookies, queries, bodies, prompts, tokens,
  database URLs, and exception details are redacted.

Exit gate:

- Proven with real subprocesses and a disposable PostgreSQL database: submit by
  HTTP, terminate a worker after it acquires the lease, let the lease expire,
  recover and complete on a replacement worker, restart the API on a new
  process/port, invoke the replay endpoint, and reproduce the exact outcome and
  authoritative trace hash. The disposable database and processes are cleaned.
- Applied solver, resolved capture-camera, and verified dependency-lock
  provenance fail closed on drift.

## Goal 2 — canonical physical human

Proven:

- The existing GLB remains the visual identity while a separate free-root MJCF
  supplies full-body dynamics, palms, articulated digits, colliders, actuators,
  collision exclusions, and semantic sites.
- The visual/physical adapter is not a qpos passthrough, and simulated state can
  be exported back into GLB animation channels.
- Real GLB export/reimport is audited across five representative root,
  full-body, and every-finger frames for small, medium, and large profiles.
  Maximum fingertip error is 0.000045 mm and maximum wrist/foot error is
  0.000037 mm, far below the 8 mm and 15 mm thresholds. Every required source
  bone resolves exactly once; upper chest, shoulders, and toes are verified as
  inherited fixed mappings with no independent animation.

- Every one of the 47 canonical bodies now consumes a versioned explicit
  inertial record. The audit binds the legacy implicit source, current source,
  MuJoCo version, numeric data, and scale law. Medium compiled values are
  byte-identical to the prior dynamics; small/large differ only at floating
  roundoff (at most 1.42e-14 kg and 8.88e-16 kg m2). Composed object bodies
  retain safe geometry inference through `inertiafromgeom="auto"`.

## Goal 3 — scene and object compiler

Proven:

- Scene packs compile through MjSpec into readable MJCF and self-contained MJZ.
- Dynamic mesh collision is restricted to verified content-addressed convex
  pieces; a compiled archive remains loadable after loose source removal.
- Six pack families are declared: grasp/place, drawer, lever/button, hand tool,
  container/lid, and two-handed object.

Open gate:

- Strict three-repeat/export probes cover all six families and reject false
  positives. The human-only button press is genuinely certified: terminal
  travel is -22.525 mm against the -20 mm gate, continuous measured contact
  lasts 186 samples, maximum penetration is 0.366 mm, pelvis drift is 0.141 mm,
  and all five friction/mass/gain/pose/timing variations certify with no object
  actuator, target, or post-initialization state write. The passive drawer now
  also certifies through the production compiler/executor: terminal travel is
  281.7 mm against the 220 mm gate, maximum penetration is 0.158 mm, measured
  handle contact is followed by a clean release, three replays and GLB reimport
  pass, and all five calibrated variations certify without object actuation or
  post-initialization state writes.
- A second bounded interaction wave also stayed fail-closed. The production
  grasp/place program is globally clean (1.131 mm penetration, 1.412 peak speed,
  0.144 mm root drift), but contacts only index/middle/ring with no thumb
  opposition and produces zero lift. That historical drawer failure was
  superseded by the certified pull above. Two production hand-tool programs
  fail typed task-space IK before simulation; the separate
  legacy probe has zero lift and a 9.386 N strike against the 10 N gate. The
  final container/lid program also fails typed IK before simulation: its left
  stabilizer misses by 78.65 mm, its right knob hand by 72.79 mm, and its right
  orientation by 0.1125 rad. Both permitted two-handed-object programs likewise
  fail collision-aware IK; the second removes rack/lumbar and palm/bar conflicts
  but retains 45.33 mm forearm/grip and 27.03 mm forearm/stem conflicts. No
  failed baseline entered replay, robustness, or positive retrieval context.
- Unit/dimension, aggregate geometry, archive expansion, plugin/reference, and
  articulated metadata constraints are now enforced. Verified content-addressed
  OBJ/STL visual meshes are embedded in MJZ with zero mass and zero collision
  masks; collision and visual roles cannot share an asset, so raw concave
  display geometry cannot enter dynamics.

## Goal 4 — timing and motion compiler

Proven:

- Canonical phase kinds/order, phase-local monotone timing, energetic profiles,
  scalar quintic interpolation, hard/contact boundary preservation, and strict
  ownership/priority conflict rejection are implemented.

Open gate:

- Position and world-absolute rotation keyframes now drive task-space IK through
  quintic translation and SLERP/SQUAD orientation, with typed rejection rather
  than silent skipping. Orientation and signed-distance collision residuals are
  production-integrated.
- Bounded overshoot/rebound accents alter joint and task-space motion while
  preserving hard and contact anchors. Target frames beyond world-absolute site
  poses remain a possible extension rather than an exit-gate claim.
- Production task-space refinement now supports explicit coarse optimization
  knots, deterministic sparse LSMR rather than memory-exhausting dense SVD, and
  subtree-aware MJCF collision exclusions. The 240 Hz authoritative rollout is
  unchanged.
- A release-grade v2-versus-POC study path is implemented. It requires at least
  30 unique held-out matched pairs, playable contiguous 30 FPS evidence, raw
  trace summaries for all four dimensions, and certified v2 traces with three
  exact repeats plus export/reimport. Three rater packets contain both randomized
  and reversed presentations; import re-verifies every frame, response, and
  rater attestation, then applies exact one-sided binomial tests and 95% intervals
  independently to phase timing, peak speed, jerk distribution, and hand shape.
  Synthetic fixtures are structurally unable to release.
- The remaining exit gate is evidence, not study plumbing: run the study with
  real retained v2/POC motions and genuine independent human responses. Release
  requires every dimension's lower 95% bound to exceed 0.5 with p < 0.05 and all
  reversal checks to agree.

## Goal 5 — whole-body MuJoCo execution

Proven:

- Evaluated motion uses a free root at fixed 240 Hz with initialization-only
  authoritative state writes.
- Control combines inverse-dynamics feed-forward, bounded joint PD, and
  deterministic COM/pelvis/torso/support-foot tasks.
- Actual qpos, qvel, control, effort, power, and contact state are archived.
- Precomputed controls use native MuJoCo batched rollout; state-dependent
  controllers use spawned processes, and the production exact-five executor now
  preserves ordering/evidence through that batch path.

Proven balance gate:

- Contact-consistent floating-base inverse control applies only legal joint
  motor efforts while MuJoCo supplies the physical support contacts. Across
  three exact runs per profile, five-second small/large and ten-second medium
  standing remain below 0.00061 m pelvis drift, effectively zero penetration,
  0.107 rad/s speed, 12.84 N m effort, and 1.20 W power. A perturbed medium run
  also recovers with zero fall contacts and terminal free-root speed below
  0.001. There is no root effort, weld, mocap, non-foot ground contact, or
  post-initialization state write.

Open gate:

- Demonstrate a genuine articulated hand close, grasp, lift, hold, and release
  against a free object, followed by all calibrated variation reruns.
- Production `MotionProgramV2` grasp attempts now compile through the real
  task-space path and executor. The best bounded redesign achieves genuine
  opposing thumb/index contact at roughly 4.24 N and reaches only 5.24 mm of
  rise against the required 120 mm; integrated reruns also sit at the 2 mm
  penetration boundary. No repeat/export/robustness claim was made because the
  baseline task predicate failed.

## Goal 6 — deterministic certification gates

Proven:

- Physics certification occurs before judging and checks finite state, limits,
  effort/power, penetration, allowed pairs, balance/fall/foot motion, hidden
  weld/mocap/teleport structures, requested object predicates, three repeat
  outcomes, and export/reimport.
- No-success paths return typed infeasible, unsupported, simulation-failed, or
  all-candidates-rejected outcomes.

Proven lifecycle gate:

- Authored contact creation/release timing and order, outside-window persistence,
  edge slip/penetration, short contact dropouts inside a required plateau, every
  free/ball quaternion, and production asset/rig/coordinate bindings are
  enforced. The certified button and drawer now pass these gates; the remaining
  empirical gap is a free-object grasp/lift/hold/release survivor.

## Goal 7 — evidence and best-of-five selection

Proven:

- Exactly five nonduplicate candidates are obtained within twenty compile
  attempts and one repair round.
- Evidence uses authoritative MuJoCo state at 30 FPS from orbit, egocentric, and
  task-closeup views, retaining raw frames, timestamps, intrinsics, and poses.
- Anonymous candidates are judged twice in distinct orders; weakest-dimension
  ranking, fallback triggers, uncertainty rejection, and four-call cap are
  enforced.
- A live `gpt-5.6-luna` structured planner smoke passed against the staged
  canonical rig and drawer scene (seven phases, one valid physical track, no
  invented DOFs, response storage disabled). The smoke also exposed and fixed
  an OpenAI strict-schema incompatibility before release.

Proven diversity gate:

- Joint-value and task-space plans receive deterministic motion-effective route
  changes for path, timing, body, contact, energy, and retrieval seed. A
  canonical-human right-palm test proves five distinct compiled trajectory and
  in-contact fingerprints while hard and contact-boundary anchors remain exact.
  Position changes are bounded to 18 mm and authored rotation changes to 3
  degrees. Static single-keyframe tracks remain intentionally indivisible.

## Goal 8 — certified RAG flywheel

Proven:

- Retrieval hard-filters certification, release, license, schema, rig,
  affordance, limbs, contact, split, and lineage before ranking.
- Five versioned namespaces, reciprocal-rank fusion, compact-context limits,
  release N+1 staging, and separate hard-negative storage are implemented.
- The current button interaction was certified for three exact baseline repeats,
  re-simulated and independently evaluated for three more exact repeats, and
  passed all five calibrated robustness variations. Its baseline authoritative
  trace SHA-256 is
  `cde40bfc63c1005f03d83e8b0637ec5857f7ede356d6c8894c196bee3e821bd2`.
- The live library now has one source `candidate`, one proof-bearing `staged`
  copy, one `building` bootstrap/N+1 release, three registered namespaces, and
  three embeddings in addition to the 6,354 quarantined legacy rows. The
  content-addressed staging evidence is sealed by SHA-256
  `06e934b187ef3d45429274cff2af67503ffef41cd4145e56b6a714f208b144f4`.
- The staged record is ranked through real SigLIP2 keyframe, deterministic
  structured-program, and deterministic phase/contact-motion indexes. RRF with
  `k=60` yields `0.04918032786885246`; compatible hard filters select the record,
  wrong-license and excluded-lineage filters select none, and the production
  retrieval path selects none because the release is not active/certified.
- A narrow evidence-backed context-coverage smoke canary improves from 0/5
  certified anchors without retrieval to 5/5 with the staged compact example.
  This is not the Goal 10 paired-bootstrap RAG quality gate.

Open gate:

- All 6,354 indexed legacy results are now present in the live PostgreSQL library
  as zero-copy `legacy_candidate` rows. Every row is unreleased, has no embedding,
  and is explicitly blocked on source hydration, re-simulation, and independent
  evaluation; a restart-idempotence canary passed. None is positive context.
- Offline-only adapters require exact cached revisions and full snapshot-tree
  digests; runtime registry access is disabled. SigLIP2
  `google/siglip2-base-patch16-224` is pinned to commit
  `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`: 9 files, 1.539 GB, sealed tree
  SHA-256 `6ea3a0eb94d36c6968e9ae10dcfadc76e88b11c937e15134c69cdc1419bd9bc8`.
  A fresh real CUDA run over the current button certification's authoritative
  `task_closeup` PNG emitted a 768-dimensional unit vector twice with zero repeat
  delta; vector SHA-256 is
  `d44f0fb5011c7d72f4d4ba0f45ec08a69f70f2e21d0ac1e9723ae6dab7ad30cf`.
- Cosmos-Embed1 remains explicitly unavailable: its exact repository/commit is
  recorded, but this Windows/Ada 8 GiB laptop is outside the model card's tested
  platform set, custom-code review and project license approval are incomplete,
  and no substitute was loaded. Therefore the two Cosmos text/video namespaces
  were not registered or populated, the release remains `building`, and no
  record was marked `certified`; freeze and activation remain prohibited.

## Goal 9 — human calibration and datasets

Proven:

- A structurally valid 200-pair/600-rating study blueprint, defect coverage,
  blinding/order/reversal controls, agreement statistics, and admission tests
  exist.
- A source-backed package builder verifies every media byte, emits blinded
  content-addressed rater packets, balances presentation order, and keeps source
  identity, defect labels, action family, and ground truth out of rater packets.
- The release-grade importer requires 600 distinct retained response files plus
  one retained attestation per real rater. It verifies safe relative paths,
  exact file hashes, assigned identity/order/timestamps, and explicit
  human/non-synthetic affirmations. The legacy digest-only importer remains
  compatible but is marked non-release-grade.
- Sign-language assets are segregated from generic gesture/production use;
  HUMOTO/OmniRetarget are restricted to object/contact research.

Open gate:

- Existing completed ratings are explicitly synthetic. Conduct and import 200
  genuine blinded comparisons with three independent raters per pair and at
  least thirty pairs per family.
- The importer now requires a verifiable Deaf-review artifact digest rather than
  a nonempty label; the genuine review artifact itself remains external evidence.
- Admit learned in-betweening only from real paired outputs, human verdicts,
  anchor checks, and physics results.

## Goal 10 — release benchmark and migration

Proven:

- The sealed benchmark expands deterministically to 300 supported and 100
  adversarial cases and the evaluator implements every numeric release gate.
- The real deterministic intake firewall classified all 100 adversarial cases
  with zero false accepts, zero typed mismatches, and zero planner/model calls;
  all 300 supported prompts remained allowed. It wrote 100 distinct immutable
  case records and a deterministic release seal.
- A production simulation/evidence measurement for five animated ten-second
  candidates recorded p95 18.21 seconds and 85.67 seconds total wall time,
  excluding VLM latency, on the reference machine.
- Replay, forward-only migrations, backup/restore, preflight, wheel smoke, and a
  fail-closed release checklist are implemented.
- A real Docker PostgreSQL custom-format dump restored all v2 tables—including
  6,354 quarantined animation rows—into a disposable database with identical
  table counts. A content-addressed canary artifact also restored byte-exactly;
  the clone was then removed and the evidence sealed.
- A current wheel was built offline, installed with its exact declared
  dependencies into a new isolated virtual environment, imported with the source
  tree absent from `PYTHONPATH`, and started the installed API to a healthy live
  endpoint. Both v2 entry points were present and the temporary environment was
  removed.
- The v1 migration canary is fully hydrated into CAS with its GLB byte hash
  unchanged; the live v1 replay route returns one animation with 54 channels.
  It remains unreleased/unembedded and still requires v2 re-simulation and
  independent evaluation. The v2 live replay remains trace-exact.
- A live operator drill observed a bounded degraded health response against an
  unreachable disposable database, stopped the bad canary, emitted a
  credential-clean incident bundle, and restored exact-lock healthy API,
  PostgreSQL, and artifact readiness. Protected counts remained unchanged at
  zero jobs, zero releases, and 6,354 quarantined legacy records.
- Twelve of fifteen release-checklist requirements now have verified local
  evidence. The checklist still returns `can_ship=false`.
- A resumable production supported-case runner now binds the sealed manifest,
  budgets, active retrieval release, planner, exact-five compiler/executor,
  nested trace/frame/video artifacts, judge, and final evidence set. Every
  completed artifact is re-hashed on resume; a missing/tampered nested object or
  mismatched artifact store fails closed. The first three communicative cases
  completed deterministic manifest preflight and then stopped with typed
  `external_model_unavailable` because planner/judge configuration was absent:
  3 blocked, 0 failed, 0 complete, and 0 model calls. This is canary evidence,
  not a supported benchmark pass.

Open gate:

- Execute the 300 supported cases through the actual planner/compiler/simulator/
  judge/retrieval runner and seal the source evidence.
- Supply the remaining `human_calibration` and `rag_ab_evaluation` evidence from
  genuine human/VLM agreement and populated certified-library A/B measurements.
- Database migration, populated backup/restore, and an isolated fresh
  install/first start are locally proven. A separate physical machine/image is
  desirable release corroboration but is no longer the only evidence path.
- The release checklist must remain blocking until every content-addressed
  evidence item verifies.

## Ship rule

Rigby v2 is not shippable merely because the repository test suite is green.
Shipping requires every release-critical item above, the immutable benchmark
thresholds, and the machine-verifiable release checklist to pass with genuine,
content-addressed evidence.
