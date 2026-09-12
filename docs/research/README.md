# Rigby research review evidence

The goal catalog is now being pursued. [Execution progress](progress/goal-progress.json), the [G00 baseline](progress/g00-baseline.md) and the [acceptance crosswalk](progress/g00-acceptance-crosswalk.md) separate new verification from the original review below.

Review of commit cfbe16e90544fc22b3acc9d95ce7b2d9cc6a24ba, performed on 11 September 2026 in America/Toronto. The fresh probe timestamp is 12 September in UTC.

- [Capability review, related work and research plan](C:/Users/hocke/GitHub/rigby/docs/research/rigby-capability-review-and-research-plan.md)
- [Verifiable goals, pursuit order and prerequisites](C:/Users/hocke/GitHub/rigby/docs/research/rigby-verifiable-goals-and-order.md)
- [Machine-readable goal catalog](C:/Users/hocke/GitHub/rigby/docs/research/rigby-verifiable-goals.json)
- [Verification summary](C:/Users/hocke/GitHub/rigby/docs/research/verification-summary.json)
- [Fresh same-prompt embodiment probe](C:/Users/hocke/GitHub/rigby/docs/research/embodiment-probe.json)
- [Fresh body-scaled grasp sweep](C:/Users/hocke/GitHub/rigby/docs/research/grasp-report.json)
- [Any-robot JUnit](C:/Users/hocke/GitHub/rigby/docs/research/rigby-review-anyrobot-tests.xml)
- [Core JUnit](C:/Users/hocke/GitHub/rigby/docs/research/rigby-review-core-tests.xml)
- [Selected humanoid JUnit](C:/Users/hocke/GitHub/rigby/docs/research/rigby-review-humanoid-tests.xml)

The report distinguishes fresh measurements, historical evidence, paper-reported findings and proposed work. Test passes include correct refusals and detection of failed physical tasks. They are not counts of successful robot tasks.

## Reproduction

Commands below use the existing Windows virtual environment with the repository's packages already installed. Run each group from its stated directory. Reproduction on another platform requires the corresponding Python path and compatible dependencies.

From the repository root:

```powershell
.\.venv\Scripts\python.exe docs/research/review_snapshot.py
```

The probe creates a fresh minimal library for the exact reach/return segments on each of the six zoo bodies. It uses the current private bake API and is intended for the recorded commit, not as a permanent public product interface. It performs no model calls and does not alter the stored product primitive library. Re-running overwrites embodiment-probe.json with the new result and timestamp.

From any-robot:

```powershell
..\.venv\Scripts\python.exe -m pytest -o addopts='' --junitxml=../docs/research/rigby-review-anyrobot-tests.xml -q
..\.venv\Scripts\python.exe scripts/grasp_report.py --out ../docs/research
```

The grasp sweep intentionally builds a body-scaled scene. Do not describe it as a fixed-world test. It excludes the tool arm, which has no grasping effector. Its raw output schema does not contain a timestamp; the paired review metadata records its provenance.

From core:

```powershell
..\.venv\Scripts\python.exe -m pytest -o addopts='' -q --junitxml=../docs/research/rigby-review-core-tests.xml
```

The recorded core run has one Windows subprocess-environment failure. See the raw JUnit for the import traceback; the hash values were not shown to disagree.

From humanoid:

```powershell
..\.venv\Scripts\python.exe -m pytest -o addopts='' -q tests/test_super_primitives.py tests/test_talmy_motion_situation.py tests/test_vlm_judge.py tests/test_calibration_rejects_degenerate_judges.py tests/test_detection_curve_on_the_corpus.py tests/test_mutation_structural_gates.py tests/test_directed_control.py tests/test_gripper_squeeze.py tests/test_gripper_live_runs.py tests/test_closed_loop.py tests/test_v2_production_grasp_place.py tests/test_v2_production_drawer_acceptance.py tests/test_v2_production_container_lid.py tests/test_v2_two_handed_production_failure.py tests/test_v2_benchmark_execution.py tests/test_v2_human_study_file_evidence.py tests/test_v2_calibration_study.py --junitxml=../docs/research/rigby-review-humanoid-tests.xml
```

From humanoid/frontend:

```powershell
npm test -- --reporter=dot
```

Frontend result: six files, 36 tests passed, 2.89 seconds. The terminal output was inspected; no frontend JUnit reporter was configured.

## Interpretation boundaries

The fresh reach/return probe obtained one requested and bound semantic hash across six bodies, zero region substitutions, twelve accepted individual primitive bakes and five accepted compositions. The dual-arm composition failed a joint limit. The accepted motions' repeated simulations were identical; this establishes deterministic repeatability rather than robustness.

The selected humanoid result was 99 passed, one skipped, three expected failures and one unexpected pass. The unexpected pass is a Windows-quarantined test whose assertions confirm a failed grasp is not certified. The skipped test is the opt-in drawer robustness suite.

No paid VLM calibration, hardware trials, external-paper reproduction, complete humanoid suite or new full primitive-library bake was performed. Source citations in the report point to primary research and official documentation, with recent preprints identified as such.
