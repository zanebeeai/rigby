# G00: existing acceptance contracts

The [JSON crosswalk](g00-acceptance-crosswalk.json) covers all sixteen G-GEN-1 clauses, three supporting scope/release clauses, all fifteen active humanoid POC sections with their complete subfields, ten v2 completion goals, fifteen release requirements and sixteen v2 numeric benchmark gates: **75 entries**. Each entry records its source, current measurement implementation, available evidence, status and remaining verification.

**The existing full acceptance contracts are not verified.** This is a static audit of retained evidence, not a new experiment or a revision of any threshold. No sealed robot/prompt manifest or ground-truth identities were opened. No models, new simulations or expensive suites were run in this subtask. Fresh G00 runs and fixes are recorded separately in [the baseline report](g00-baseline.md).

## General audit gaps

These are **source-code observations, not reproduced false-positive runs**. Relevant locations are in [rigby_general.audit](../../../any-robot/src/rigby_general/audit.py); requirements are in [the general contract](../../../any-robot/acceptance_criteria.general.yaml).

- Line 119 scans only robot directory IDs in Python files. The contract also prohibits robot-specific link/joint names and authored configuration.
- Line 141 discovers local URDF directories rather than enforcing pinned upstream benchmark membership, holdouts and morphology/effector coverage. Line 190 reads zoo ground truth without the promised label-hash verification.
- Lines 228 onward skip missing library summaries; missing bodies appear only in detail text and are omitted from coverage/yield minima. Stored summaries are not recertified here.
- Line 258 uses maximum stored bake time, not the declared p95; the 500-attempt ceiling is never measured.
- Line 278 skips each body's planner exceptions during invariance comparisons. The ten public prompts also differ from the required sixty shared benchmark prompts.
- Line 302 checks adversarial prompts only on the first ingested body. Any exception possessing a code attribute counts without matching the expected failure code.
- Line 325 compares two morphology ingests on at most two bodies. It does not establish three-repeat motion-trace identity.
- Line 102 conjuncts only emitted requirements. The stored report has ten passing rows and no deferral notes; the YAML declares sixteen requirements and explicitly defers seven. Its partial release flag cannot mean full acceptance.

## General requirement ledger

“Measured” means an actual measurement exists within its stated limited scope. “Deferred” includes missing or incomplete current measurements, with each gap retained. **No full existing acceptance requirement is marked verified.**

| Requirement | Status | Remaining verification |
|---|---|---|
| zero_per_robot_code | deferred | Check the full sealed vocabulary and configuration provenance through an isolated evaluator. |
| ingest_success | measured | Verify every required unmodified asset, integrity result and manifest hash. |
| site_derivation_accuracy | deferred | Verify independent label hashes, robot coverage and declared missing/extra-site scoring. |
| effector_classification | deferred | Measure all effectors against independent labels and enforce three effector/two morphology families. |
| bake_coverage | deferred | Require a valid current summary and certificates for every body; recompute all afforded coverage and yield. |
| bake_budget | deferred | Retain complete attempt/timing data on the reference machine and compute both declared metrics. |
| schema_invariance | measured | Require every body to parse each supported shared prompt before comparing hashes; count refusals separately. |
| grounding_soundness | deferred | Recompute every DOF/site/contact reference and metric limit against each measured body. |
| supported_certified_winner_rate | deferred | Execute the sealed matrix with physical gates and validated independent judging; retain per-body outcomes. |
| adversarial_typed_outcomes | deferred | Run 100 adversarial cases per body and compare exact expected typed outcomes. |
| critical_deterministic_false_accepts | deferred | Inject failures across selection routes and independently count gate-failing selected candidates. |
| deterministic_replay | deferred | Replay every certified winner three times with pinned physics/controller inputs and compare complete traces. |
| judge_independence | deferred | Declare identities, retain blinded/reversed evidence and measure calibrated judge consistency. |
| latency_p95 | deferred | Record monotonic end-to-end timings over the benchmark, excluding only VLM time. |
| holdout_margin | deferred | Have an isolated evaluator verify heldout membership and per-requirement shortfall without exposing identities. |
| export_roundtrip | deferred | Independently reimport every winner and verify every required link within 1 mm. |

The prior review's six-body probe shares one semantic hash and certifies twelve leaves, but only five complete compositions. It supports narrow semantic grounding, not the full supported-prompt winner rate. The separate body-scaled grasp probe has three successes among five bodies. Contact remains outside the current admitted general inventory; the updated YAML records that three-of-five measurement without admitting contact to the general inventory.

## Humanoid POC and v2

The POC runner has gates for planner, gesture, diversity, grasp, physical proof, safety, structural physics, orientation, parameter controls, latency and export. The JSON retains every active section, threshold and all 28 parameter controls. Constructed metric tests and selected offline compilations establish gate/compiler behavior, not complete physical/model acceptance. The manual-review route and declared autonomous evaluation policy also need an explicit crosswalk.

The current autonomous judge audit still reads the small frozen calibration, and its tests intentionally assert that this artifact passes. The later August 26 corpus report calls the tested grader invalid: 881 of 1,288 corrupted clips accepted, balanced accuracy 0.573. Instrument/evidence generations differ; neither artifact validates a newly configured judge. See [the audit](../../../humanoid/evals/autonomous_goal_audit.py) and [later calibration report](../../../humanoid/docs/evidence/10f-calibration-report.md).

| v2 goal | Evidence boundary | Status |
|---|---|---|
| 1. foundation and reproducibility | Historically proven process crash/recovery/replay, typed contracts and provenance. | deferred |
| 2. canonical physical human | Historical free-root 67-actuator body, explicit inertias and all-profile GLB roundtrip. | deferred |
| 3. scene and object compiler | Two of six task families historically succeed; four lack certified production survivors. | failed |
| 4. timing and motion compiler | Compiler/study plumbing exists; real blinded POC comparison is missing. | deferred |
| 5. whole body execution | Standing succeeds historically; required free-object grasp rises only millimetres against 120 mm. | failed |
| 6. deterministic certification | Gate tests and button/drawer survivors exist; free-object lifecycle survivor is missing. | failed |
| 7. evidence best of five | Exact-five/evidence contracts exist; complete live model-backed acceptance remains unproven. | deferred |
| 8. certified rag | One historical staged button record; three indexes; five-index release building; Cosmos unadmitted. | deferred |
| 9. human calibration and datasets | Study/import contracts exist; completed ratings are synthetic; real review evidence absent. | deferred |
| 10. release benchmark and migration | Historical firewall/restore/install evidence; 300 supported outcomes remain absent. | deferred |

The [v2 completion audit](../../../humanoid/docs/v2/completion-audit.md) reports twelve of fifteen release requirements historically verified. The 300-supported benchmark, genuine human calibration and RAG A/B remain blocking. This crosswalk preserves all fifteen requirements without promoting historical prose into a current release seal.

`test_v2_benchmark_execution.py` constructs outcomes in `FixtureCaseRunner`; `test_v2_calibration_study.py` uses `build_synthetic_completed_study`. Their passing counts are not policy success rates or human accuracy measurements. A passing production-grasp refusal test establishes rejection of a physically unsuccessful attempt. Current numeric gates also differ by pipeline: the POC 15-second end-to-end latency and v2 90-second local-only latency must not be substituted for each other.

## Remaining acceptance work after G00

1. Preserve the separate [fresh baseline](g00-verification-summary.json) and invalidate this static snapshot when source hashes change.
2. Enumerate the whole acceptance contract in reports; distinguish missing measurement, observed failure, explicit deferral and public development-only evidence.
3. Add bounded anti-vacuity tests for the observed audit gaps before relying on its release flag; keep sealed evaluation inputs isolated.
4. Preserve the stronger physics, judge and release gates as explicit empirical work. Green contract tests must not relax their requirements.

The JSON records source SHA-256 values so later implementation edits can invalidate the snapshot explicitly. Output uses LF line endings.
