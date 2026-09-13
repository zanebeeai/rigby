# G03 body capability evidence

G03 establishes a reproducible, name-invariant **rigid fixed-base URDF intake
profile**. It derives capability records, preserves uploaded names as aliases,
and explicitly separates measured kinematics from assumptions about actuation,
contact, sensors and physical performance. It does not establish arbitrary
embodiment control, successful object transport or locomotion.

The [offline inspection gallery](g03-release/index.html) contains twelve
six-second GIFs: six procedural zoo bodies, three unchanged third-party robot
descriptions and three independently checked geometry fixtures. Each GIF is a
camera orbit of an archived static pose with chains, sites and sampled workspace.
All 720 frames are labeled as inspection, with zero physical execution steps.

## Acceptance evidence

| Criterion | Evidence and outcome |
|---|---|
| G03.A01: six zoo and three third-party descriptions; no source-name control decisions | All six zoo bodies plus iiwa7, kuka_lwr and SO-ARM101 admitted through the structural profile. The complete source-ID scan has zero zoo-ID hits. Uploaded joint/link names become structural identities before measurement or control; legacy name-based sensor classification is explicitly disabled and guarded by a test. |
| G03.A02: independent geometry, effectors and typed refusals | Three frozen analytic fixtures independently verify 14 named geometric points and two gripping-region centres. All 18 derived site instances are within 1 mm; the maximum discrepancy is 2.78e-17 m. Rigid-tool, parallel-jaw and three-digit classifications match the independent labels. Invalid limits, invalid inertia, missing capabilities and unsupported mimic coupling receive typed errors. Required CONTACT sites are checked separately from the selected TIP. |
| G03.A03: qualified assumptions | Every capability cites recorded evidence or an explicit gap. Source limits, inferred effort, assumed velocity/acceleration, direct simulated motors, convex colliders, collision exclusions, unconfirmed frames, sampled workspace and undeclared sensors remain distinguishable. Geometric closure never enables physical grasp transport, payload, suction, rolling or balance claims. |
| G03.A04: at least 20 permutations per body | 120 distinct variants preserve every mechanical declaration after inverse renaming. Full canonical models, capability hashes and selected figure sites agree. Every variant's reference clock, joint coordinates and all body positions match its baseline exactly at 101 reference times, with zero tolerance. Original-versus-canonical joint axes/ranges, mass and inertia also agree, and link poses agree within 1e-12 at five configurations per zoo body. |
| D03: body inspection gallery | Twelve archived binary models, twelve replayable GIFs, 1,536 saved kinematic samples, per-frame camera maps, body manifests, morphology measurements, capability/uncertainty tables and a named Panda coupling refusal. |

The [audit](g03-release/audit.json), [geometry measurements](g03-release/geometry.json),
[120 invariance records](g03-release/invariance.json),
[model equivalence checks](g03-release/uploaded-model-equivalence.json),
[static scan](g03-release/static-scan.json) and
[contact feasibility map](g03-release/contact-feasibility.json) provide the
machine-readable record. Each permutation includes its actual source, inverse
aliases and numerical reference arrays.

All existing general acceptance requirements and thresholds remain unchanged.
Their previous sealed-label accuracy obligations are retained; this audit does
not open sealed labels or substitute these analytic fixtures for a blind
research evaluation. Independent fixture creation/checking is computational
and analytically reproducible; it is not a human annotation study.

## Corrections supported by the evidence

Legacy ingestion used uploaded names to order symmetric chains and select tip
members, and treated camera-like names as sensor hints. The structural profile
assigns names from mechanical descriptors and topology first. It refuses an
exactly indistinguishable sibling identity instead of using an uploaded label
to break the tie. Existing ingestion remains available for historical replay;
its name invariance is not claimed.

The frozen geometry fixtures initially exposed a 19.09 mm grasp-site error.
The estimator selected the first equally wide sampled opening near a finger
edge. It now uses the centre of equal-clearance witnesses, validates that
proposal against all compiled robot collision solids, and selects a free
witness if a central obstacle occupies the average. The three-digit fixture
retains separated contact patches: an object can bridge those patches even
when a ray through the centre misses a digit. Point clearance remains only a
necessary geometric condition; finite-object fit and physical retention need
separate trials.

Inverted and degenerate bounded joint limits are now rejected before a compiler
can interpret them as unlimited. Continuous joints retain no declared position
bounds; their finite measurement intervals are explicitly sampling assumptions.
URDF mimic mechanisms are refused by this profile because the present lower
runtime cannot preserve those constraints. The Panda refusal is a runtime
coverage limitation, not an invalid-model or task-infeasibility judgment.

## Third-party provenance

The three admitted URDFs and every one of their 41 referenced meshes were
compared byte-for-byte with pinned upstream files. A 6.67 MB offline archive
includes these sources, original license notices and attribution; tests do not
download models. Full URLs, revisions and file digests are in the
[input provenance](../../any-robot/tests/fixtures/g03_third_party/provenance.json).

- iiwa7 and kuka_lwr: bulletphysics/bullet3,
  `63c4d67e337017f9d8b298c900e9aabdb69296e7`,
  [upstream Zlib notice](https://github.com/bulletphysics/bullet3/blob/63c4d67e337017f9d8b298c900e9aabdb69296e7/LICENSE.txt).
- SO-ARM101: TheRobotStudio/SO-ARM100,
  `7629d2ad9853d10fb903093a33ef6114099d97e5`,
  [upstream Apache-2.0 notice](https://github.com/TheRobotStudio/SO-ARM100/blob/7629d2ad9853d10fb903093a33ef6114099d97e5/LICENSE).
- The separate Panda URDF-only refusal fixture preserves its historical
  package-path localization; that change is recorded and its unused meshes
  are not included in the fixture archive.

## Reproduction and validation

Source revisions, release digest, test counts and final verification results
are pinned in [g03-validation.json](g03-validation.json). The full any-robot
suite passed 328 tests; the final compressed-model packaging change also passed
its targeted archive/replay test. The existing Starlette test-client
deprecation warning remains visible. No API/model generation calls were used.

From a workspace environment containing the `core`, `any-robot` and development
dependencies:

```sh
python docs/results/verify_g03_release.py --rerender
python any-robot/scripts/audit_body_capabilities.py new-output/g03
python -m rigby_general.capabilities verify docs/results/g03-release/bodies/three_digits
python -m rigby_general.capabilities render docs/results/g03-release/bodies/three_digits new-output/inspection
```

Verification checks the externally recorded release-index digest, all payloads,
all 120 stored references, the frozen analytic labels, GIF frame durations and
exact replay of all 1,536 saved kinematic samples. `--rerender` regenerates a
complete inspection GIF and checks byte identity. Rendering uses the recorded
MuJoCo version and compressed binary model, without the original assets or an
API connection. The runtime XML alone is not a portable mesh package.

## Remaining scope and next work

New transfer experiments must call `ingest_capability_body`; the default legacy
studio/API has not been migrated to this profile. Floating-base dynamics,
coupled mechanisms, hardware sensors and unsupported source extensions still
need explicit runtime support. Geometric contact witnesses certify only the
declared analytic fixtures; contact feasibility for every zoo/third-party
task pair remains unknown until G06 independently checks object fit and
physical lift, retention and release.

G05 is next: correct the reference/physics clock mismatch, then test generated
free-space primitives and composition across all six bodies under the frozen
trial contract. G06 adds contact evidence before recursive task execution.
CI integration remains conditional on actual green required checks; local
evidence is not a substitute for those checks. The previously reported account
billing restriction is an owner action, not a reason to change the workflows
or relax a research threshold.
