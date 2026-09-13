# G00: executable baseline

**All three G00 verification criteria are met locally.** This establishes a reproducible starting point and preserves known failures; it does not establish the later manipulation, locomotion, hierarchy or judge-calibration goals. Commit and PR integration are tracked in [goal-progress.json](goal-progress.json).

The runs used base commit `cfbe16e90544fc22b3acc9d95ce7b2d9cc6a24ba` plus the baseline fixes in this PR, on Windows 11 with Python 3.12.7, MuJoCo 3.11.0 and NumPy 2.5.1. [Machine-readable evidence](g00-verification-summary.json) records the dependency lock and relevant source hashes. The original review artifacts remain in the parent directory; these fresh runs have separate paths.

## Verification

| Check | Result | Evidence |
|---|---|---|
| Full core suite | 59 passed; no failures or skips | [JUnit](g00-core-tests.xml) |
| Full any-robot suite | 300 passed; no failures or skips | [JUnit](g00-anyrobot-tests.xml) |
| Selected humanoid physical, semantic, judging and export suites | 120 passed, 1 skipped, 3 xfailed, 1 xpassed | [JUnit](g00-humanoid-tests.xml) |
| Gesture, dexterous contact and full-body export smoke | 5 passed | [JUnit](g00-authoring-smoke.xml) |
| Frontend | 36 passed in 6 files; production build passed | Terminal-inspected summary in [metadata](g00-verification-summary.json) |
| Six-body canonical reach/return | 12/12 freshly baked leaves certify; 5/6 compositions certify; one semantic hash; no region substitutions | [Probe](g00-results/embodiment-probe.json) |
| Body-scaled grasp | 3/5 certify; dual-arm and multifinger bodies drop/do not carry the object | [Probe](g00-results/grasp-report.json) |
| Existing acceptance contracts | 75 clauses mapped; full acceptance remains unverified | [Crosswalk](g00-acceptance-crosswalk.md) |

The expected failures are three existing order-dependent closed-loop tests. The unexpected pass is the quarantined Windows production-grasp test: its passing assertions verify that an unsuccessful physical grasp is rejected. The skipped drawer test is the opt-in three-repeat/five-variation robustness run. Basic physical drawer acceptance did run, as did simulated export, all-profile multiframe GLB roundtrip and gripper solidity checks. Passing these selected tests is not a claim that the full humanoid suite passed.

The API key is configured and authenticated through the models endpoint. No generation or paid judge-calibration calls were made in G00. The endpoint check does not establish that any particular generation model is available or scientifically valid.

## Changes and claim boundaries

- The cross-process hash test now preserves the parent's platform environment while varying `PYTHONHASHSEED`. Both child hashes and the parent hash are still compared. Hashing implementation and acceptance assertions are unchanged.
- Demo invariance counts distinct robot IDs separately from distinct semantic hashes. The recorded manifest's aggregate is recomputed from its unchanged clip rows; the canonical prompt has six bodies and one reading. A regression test also verifies that repeated rows do not add bodies and conflicting readings remain visible.
- The any-robot status now reports 26 inventory entries, incomplete composition/contact coverage and body-scaled world fitting. It does not describe the small public zoo as an unseen-topology evaluation.
- Gripper sensing provenance distinguishes the offline colour/bench assumptions from model-identified visual belief and explicitly describes simulated force, range and obstruction instruments. Historical recordings retain their original results with a version notice.
- Collision claims are scoped to declared masks and exclusions. The humanoid gripper's arm segments do not collide with the block or one another; current checks do not establish unrestricted whole-body collision safety.
- General acceptance's contact narrative now cites the measured three-of-five probe. No numeric requirement, test tolerance, refusal, skip or expected-failure marker was weakened.

The general audit currently emits a narrower subset of its acceptance contract and has missing-denominator, replay, membership and coverage gaps. These are static implementation observations, not reproduced false-positive audit runs. The crosswalk retains the stronger requirements and identifies the missing measurements. Old calibration artifacts and synthetic human-study fixtures are not current VLM validation or actual human ratings.

## Reproduction

Use the committed dependency lock and the repository setup in `CONTRIBUTING.md`. These commands use the existing Windows virtual environment; substitute its interpreter path on other platforms. They are offline and do not require the API key. Use a new output directory to retain this baseline.

From the repository root:

```powershell
.venv/Scripts/python.exe docs/research/review_snapshot.py --out docs/research/progress/rerun-g00
```

From `core`:

```powershell
../.venv/Scripts/python.exe -m pytest -o addopts='' -q --junitxml=../docs/research/progress/rerun-core.xml
```

From `any-robot`:

```powershell
../.venv/Scripts/python.exe -m pytest -o addopts='' -q --junitxml=../docs/research/progress/rerun-anyrobot.xml
../.venv/Scripts/python.exe scripts/grasp_report.py --out ../docs/research/progress/rerun-g00
```

From `humanoid`:

```powershell
../.venv/Scripts/python.exe -m pytest -o addopts='' -q tests/test_super_primitives.py tests/test_talmy_motion_situation.py tests/test_vlm_judge.py tests/test_calibration_rejects_degenerate_judges.py tests/test_detection_curve_on_the_corpus.py tests/test_mutation_structural_gates.py tests/test_directed_control.py tests/test_gripper_squeeze.py tests/test_gripper_live_runs.py tests/test_closed_loop.py tests/test_v2_production_grasp_place.py tests/test_v2_production_drawer_acceptance.py tests/test_v2_production_container_lid.py tests/test_v2_two_handed_production_failure.py tests/test_v2_benchmark_execution.py tests/test_v2_human_study_file_evidence.py tests/test_v2_calibration_study.py tests/test_v2_simulated_export.py tests/test_v2_glb_multiframe_roundtrip.py tests/test_gripper_solidity.py --junitxml=../docs/research/progress/rerun-humanoid.xml
../.venv/Scripts/python.exe -m pytest -o addopts='' -q tests/backend_contracts.py::test_handedness_mirrors_gesture_trajectory tests/test_dexterous_action_interpretation.py::test_dexterous_program_compiles_and_measures_real_contacts tests/test_full_body_motion.py::test_glb_export_contains_animated_hips_translation --junitxml=../docs/research/progress/rerun-authoring.xml
```

From `humanoid/frontend`:

```powershell
npm test -- --reporter=dot
npm run build
```

G00 does not require a new demo. G01 supplies full-duration actual-state videos, separate physics replay and re-render commands, tamper detection, and explicit success/failure/refusal examples. Existing capped GIFs remain historical previews.
