# Results manifest

Generated 2026-09-10 from commit `0a3c6e2` by `docs/results/build_manifest.py`. One section per place results were ever written, in the order the work happened. Locations marked *untracked* are the residue the 27 Aug tier split left in the checkout; they exist on the machine that ran the experiments and nowhere else, which is the reason this file exists. Media copied into `docs/results/media/` carries its SHA-256 so a copy can be checked against its source.

| set | location | tracked | period | files | size | media |
|---|---|---|---|---|---|---|
| humanoid-runs-aug07-12 | `rigby-poc/results` | untracked | 2026-08-07 to 2026-08-12 | 61102 | 12.9 GB | 21183 |
| humanoid-release-evidence | `rigby-poc/artifacts-v2` | untracked | 2026-08-11 to 2026-08-12 | 197 | 68.5 MB | 0 |
| mjco-sim-workspace | `rigby-mjco-sim` | untracked | 2026-08-12 to 2026-08-18 | 372396 | 14.4 GB | 20487 |
| humanoid-demo-gifs | `humanoid/docs/media` | tracked | 2026-08-27 | 7 | 3.2 MB | 6 |
| humanoid-calibration-evidence | `humanoid/docs/evidence` | tracked | 2026-08-16 to 2026-09-07 | 2 | 10 kB | 0 |
| any-robot-zoo-aug24-27 | `rigby-generalized-urdf/results` | untracked | 2026-08-24 to 2026-08-27 | 229 | 72.7 MB | 36 |
| any-robot-zoo-demos | `rigby-generalized-urdf/docs/media` | untracked | 2026-08-24 to 2026-08-27 | 15 | 5.4 MB | 13 |
| any-robot-environments-aug27-28 | `any-robot/results` | untracked | 2026-08-27 to 2026-08-28 | 363 | 208.9 MB | 42 |
| gripper-milestones | `origin/grasp/auto-lift:rigby-poc` | on the branch only | 2026-08-27 to 2026-09-09 | 27 | 25.3 MB | 14 |

## Humanoid result archive (rigby-poc/results)

- **Location:** `rigby-poc/results` (untracked)
- **Period:** 2026-08-07 to 2026-08-12
- **Produced by:** The humanoid app (`uv run rigby-humanoid`, then `POST /api/v1/pipeline-runs`, or the CLI in `rigby_poc`) writes one immutable six-digit directory per compiled clip: request.json, scene.json, program.json, clip.json, metrics.json, provenance.json, animation.glb. `acceptance-runs/` holds the blinded gesture review runs; `autonomous-goal-audit.json` is the gate summary. See the README inside the directory.
- **Code:** The `rigby-poc` layout that became `humanoid/` on 27 Aug (commit 2d39970). provenance.json records `compiler_version` but not a commit; the compiler versions present date the runs to the 0.1.0 to 0.4.0 compilers of early Aug.
- **On disk:** 61102 files, 12.9 GB, modified 2026-08-07 to 2026-08-12, media {'.glb': 6259, '.png': 14924}
- **Facts:**
  - numbered_runs: 6449
  - first_id: 000001-throw-up-a-hang-ten-sign
  - last_id: 006447-clap-your-hands
  - planner_models: {"compile-only/none": 5447, "offline/rule-planner-v1": 589, "openai/gpt-5.6-luna": 319, "offline/rule-planner-v3": 56, "openai/gpt-5.6-terra": 28, "offline/rule-planner-v5": 10}
  - compiler_versions: {"rigby-compiler-0.1.0": 5576, "rigby-compiler-0.3.0": 733, "rigby-compiler-0.4.0": 109, "rigby-compiler-0.2.0": 31}
  - accepted_or_structural_valid: 985
  - rejected_or_structural_invalid: 164
  - runs_without_verdict: 5300
  - runs_with_glb: 6257
  - runs_with_evidence_pngs: 0
  - acceptance_runs: ["000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009", "000010", "000011", "000012", "000013", "000014", "000015", "000016", "000017", "000018", "000019", "index.json"]
  - autonomous_goal_audit: {"status": "pass", "gates": {"calibrated_autonomous_judge": "pass", "gesture_pipeline_smoke": "pass", "pickup_pipeline_smoke": "pass"}}
  - named_sets: ["acceptance-runs", "capture-lifecycle-smoke", "complex-hang-ten-v1", "complex-hang-ten-v2", "complex-hangten-pilot-v1", "complex-hangten-pilot-v2-pronation", "complex-hangten-pilot-v3-pronation-safe", "complex-hangten-pronation-revision-review", "complex-hangten-structural-sweep-v4", "dexterous-inspection", "flywheel-runs", "full-body-smoke", "grounded-smoke", "handoff-smoke", "heldout-flywheel", "judge-calibration", "judge-evidence", "judge-model-pilot", "object-interaction-smoke", "pickup-final-visual-smoke", "pickup-openai-visual-smoke", "pickup-palm-visual-smoke", "pickup-visual-smoke", "pipeline-runs", "pose-smoke", "sequence-smoke", "stateful-sequence-smoke"]

## Humanoid v2 release evidence and benchmarks (rigby-poc/artifacts-v2)

- **Location:** `rigby-poc/artifacts-v2` (untracked)
- **Period:** 2026-08-11 to 2026-08-12
- **Produced by:** `rigby_v2.release` and `rigby_v2.release_ops` (now under humanoid/src/rigby_v2): each requirement in `release-evidence/manifest.json` is a JSON artefact with its SHA-256. The `supported-live-model-authority-canary-*` runs are the 11 Aug live-model canaries (`run-config.json` names the pipeline id and case ids).
- **Code:** rigby_v2 as tracked by PR 9 (the v2 runtime), pre-rename.
- **On disk:** 197 files, 68.5 MB, modified 2026-08-11 to 2026-08-11
- **Facts:**
  - top_level: ["benchmarks", "library", "objects", "release-evidence", "supported-communicative-canary", "supported-live-model-authority-canary-20260811", "supported-live-model-authority-canary-20260811-v2"]
  - release_evidence: {"adversarial_benchmark_100": "pass", "database_backup_restore": "pass", "dependency_lock_audit": "pass", "deterministic_replay": "pass", "fresh_machine_install": "pass", "fresh_machine_preflight": "pass", "license_dataset_review": "pass", "migration_replay": "pass", "operator_dry_run": "pass", "performance_p95": "pass", "startup_readiness": "pass", "wheel_smoke": "pass"}
  - benchmarks: ["final-integrated"]

## MuJoCo production workspace (rigby-mjco-sim)

- **Location:** `rigby-mjco-sim` (untracked)
- **Period:** 2026-08-12 to 2026-08-18
- **Produced by:** A standalone workspace (own pyproject and uv.lock) whose README calls it 'the clean local-first Rigby production workspace'. `artifacts-v2/` holds the button, drawer, bimanual-grasp and calibration-readiness experiments of 13-17 Aug; `docs/` holds the skinned-avatar SDF probes and the visual-parity audit; `visual/generate_demo_media.py` renders the viewer traces.
- **Code:** Its `rigby_v2` package is the ancestor of what PR 9 tracked, but a large part of this tree exists in no commit (counted below). It is not in git history at all.
- **On disk:** 372396 files, 14.4 GB, modified 1980-01-01 to 2026-08-18, media {'.mp4': 238, '.png': 18954, '.gif': 1221, '.glb': 9, '.jpg': 65}
- **Facts:**
  - artifacts_v2: ["acceptance (2026-08-14)", "allowed-query (2026-08-13)", "benchmark-readiness (2026-08-14)", "benchmark-route-coverage (2026-08-14)", "benchmarks (2026-08-18)", "bimanual-grasp-diag (2026-08-13)", "bimanual-grasp-fresh-raised-ik (2026-08-13)", "bimanual-grasp-pedestal-static (2026-08-13)", "bimanual-grasp-preflight (2026-08-13)", "bimanual-grasp-preflight-2 (2026-08-13)", "bimanual-grasp-preflight-3 (2026-08-13)", "bimanual-grasp-preflight-4 (2026-08-13)", "bimanual-grasp-preflight-5 (2026-08-13)", "bimanual-grasp-preflight-6 (2026-08-13)", "button-contact-v2 (2026-08-14)", "button-ego-last.png (2026-08-13)", "button-exact-five-goal7 (2026-08-14)", "button-exact-five-goal7-execution (2026-08-14)", "button-exact-five-goal7-sequential-v2 (2026-08-14)", "button-exact-five-goal7-sequential-v3 (2026-08-14)", "button-exact-five-goal7-sequential-v4 (2026-08-14)", "button-exact-five-goal7-stati ...
  - docs: ["exact-skinned-dispatch-architecture-v5.md", "migration", "native-exact-skinned-avatar-sdf-probe-v2.md", "native-exact-skinned-avatar-sdf-probe-v3.md", "native-exact-skinned-segment-sdf-probe.md", "tagged-owner-collision-dispatch-probe-v4.md", "v2", "visual-parity-audit.json", "visual-parity.md"]
  - python_files_checked: 1903
  - python_files_in_no_commit: 1770

## Humanoid README demo GIFs (humanoid/docs/media)

- **Location:** `humanoid/docs/media` (tracked)
- **Period:** 2026-08-27
- **Produced by:** `cd humanoid; uv run python -m evals.render_demo_gif <result_id> docs/media/<name>.gif` against a running app on :8000 (8 FPS sample, 480 px ego and orbit panels, 96-colour GIF via FFmpeg). `render_comparison_gif.py` made the before/after strip. `demo-manifest.json` records the result id, prompt, planner and clip length per GIF.
- **Code:** demo-manifest.json says compiler commit d0dbc86 (27 Aug, feat/strike-torso-cross merge).
- **On disk:** 7 files, 3.2 MB, modified 2026-08-28 to 2026-08-28, media {'.gif': 6}
- **Facts:**
  - demo_manifest_keys: ["capture", "comparisons", "compiler_commit", "demos", "generated_on", "schema_version"]
  - demos: [{"file": "left-hook.gif", "result_id": "001055-throw-a-left-hook", "prompt": "throw a left hook", "planner_model": "rule-planner-v1", "duration_s": 3.3833}, {"file": "right-jab.gif", "result_id": "001056-throw-a-right-jab", "prompt": "throw a right jab", "planner_model": "rule-planner-v1", "duration_s": 3.15}, {"file": "step-over-hurdle.gif", "result_id": "000619-step-over-the-hurdle-with-your-right-foot", "prompt": "step over the hurdle with your right foot", "planner_model": "rule-planner-v1", "duration_s": 2.55}, {"file": "hang-ten.gif", "result_id": "000620-throw-up-a-hang-ten-sign-with-your-right-hand-th", "prompt": "Throw up a \"hang-ten\" sign with your right hand, there should be a swift motion up to the main position wherein the middle three fingers are as contracted as possible, the wrist should then shake rapidly back and forth a few times, before returning to default", "plan ...
  - comparisons: [{"file": "arm-fix-throw-comparison.gif", "prompt": "throw the block far and high with your right hand"}]
  - compiler_commit: d0dbc86
  - generated_on: 2026-08-27
- **Media:**

  | file | size | sha256 |
  |---|---|---|
  | `humanoid/docs/media/arm-fix-throw-comparison.gif` | 913 kB | `5a2f601647fb5bc9` |
  | `humanoid/docs/media/finger-count.gif` | 415 kB | `fcd9af30cdb6a191` |
  | `humanoid/docs/media/hang-ten.gif` | 246 kB | `867a450e8d4e804b` |
  | `humanoid/docs/media/left-hook.gif` | 703 kB | `b41bd98021d1a9a3` |
  | `humanoid/docs/media/right-jab.gif` | 630 kB | `9f8ef7ad891c0603` |
  | `humanoid/docs/media/step-over-hurdle.gif` | 322 kB | `363b81bdebc0120d` |

## Judge calibration evidence (humanoid/docs/evidence, humanoid/evals/corpus)

- **Location:** `humanoid/docs/evidence` (tracked)
- **Period:** 2026-08-16 to 2026-09-07
- **Produced by:** `10f-calibration-report.md` and `frozen-judge-calibration.json` come from the calibration campaign driver (`humanoid/evals`, PRs 4, 6, 8 and the 10f2 campaign merge 880735c). The golden corpus under `humanoid/evals/corpus/cases` is blessed by the corpus CLI; `expected.json` per case carries the digest of the blessed clip.
- **Code:** Tracked on main; every change is in git history.
- **On disk:** 2 files, 10 kB, modified 2026-08-28 to 2026-08-28

## Any-robot zoo trials, first pass (rigby-generalized-urdf/results)

- **Location:** `rigby-generalized-urdf/results` (untracked)
- **Period:** 2026-08-24 to 2026-08-27
- **Produced by:** `uv run python scripts/run_trials.py` (every gripper against every authored world), the contact probe in `rigby_general.contact`, and plain prompts through `rigby_general.run`. One `trace.json` per `<robot>--<prompt>` directory.
- **Code:** The `rigby-poc-general` package before it became `any-robot/` (PR 10, PR 12). trace.json provenance hashes the rigby_v2 base tree it ran against (114 files).
- **On disk:** 229 files, 72.7 MB, modified 2026-08-25 to 2026-08-28, media {'.gif': 36}
- **Facts:**
  - traces: 151
  - kinds: {"contact_probe": 16, "prompt": 55, "environment_trial": 80}
  - accepted: 31
  - rejected: 120
  - failure_codes: {"object_too_wide": 33, "unafforded_schema": 24, "object_out_of_reach": 12, "grasp_not_achieved": 11, "object_not_lifted": 10, "no_gripper": 8, "object_inside_reach_hole": 8, "unreachable_object": 8, "ungroundable": 2, "deterministic_gate_failed": 1, "object_dropped": 1, "unsupported_morphology": 1, "excessive_penetration": 1}
  - robots: {"uhand2": 20, "zoo_jaw_arm": 17, "kuka_kr6": 15, "panda": 14, "zoo_compact_arm": 13, "zoo_dual_arm": 13, "zoo_hand_arm": 13, "zoo_long_arm": 13, "eezybotarm_mk1": 5, "beetlebot": 4, "iiwa7": 4, "kuka_lwr": 4, "makerpro_5dof": 4, "so101": 4, "so101_undeclared": 4, "zoo_tool_arm": 4}
  - created: ["2026-08-27", "2026-08-28"]
  - code_base_trees: {"rigby_v2@0effceae5518": 151}
  - top_level_files: ["intake.json", "studio.html", "trials.json"]

## Any-robot zoo demo GIFs and grasp report (rigby-generalized-urdf/docs/media)

- **Location:** `rigby-generalized-urdf/docs/media` (untracked)
- **Period:** 2026-08-24 to 2026-08-27
- **Produced by:** `uv run python scripts/build_demos.py` renders the same plain-language prompts on several zoo robots and writes `demo-manifest.json` with the role-normalized schema hash beside each clip (one prompt, one meaning, many bodies). `uv run python scripts/grasp_report.py --render` writes `grasp-report.json`.
- **Code:** rigby-poc-general at PR 10 to PR 12; the manifest does not record a commit.
- **On disk:** 15 files, 5.4 MB, modified 2026-08-25 to 2026-08-25, media {'.gif': 13}
- **Facts:**
  - demo_manifest_keys: ["clips", "schema_invariance", "schema_version", "shared_prompt"]
  - clips: [{"gif": "zoo_compact_arm-reach-out-as-far-as-you-can-and-then-come-ba.gif", "robot_id": "zoo_compact_arm", "prompt": "reach out as far as you can and then come back", "certified_primitives_used": 2, "tracking_error_m": 0.006391}, {"gif": "zoo_compact_arm-reach-out-quickly,-just-a-little.gif", "robot_id": "zoo_compact_arm", "prompt": "reach out quickly, just a little", "certified_primitives_used": 1, "tracking_error_m": 0.008704}, {"gif": "zoo_dual_arm-reach-out-as-far-as-you-can-and-then-come-ba.gif", "robot_id": "zoo_dual_arm", "prompt": "reach out as far as you can and then come back", "certified_primitives_used": 2, "tracking_error_m": 0.012952}, {"gif": "zoo_dual_arm-lower-the-tool-down-toward-the-table.gif", "robot_id": "zoo_dual_arm", "prompt": "lower the tool down toward the table", "certified_primitives_used": 1, "tracking_error_m": 0.009275}, {"gif": "zoo_hand_arm-reach-out-as- ...
  - grasp_attempts: 5
  - grasp_certified: 1
  - grasp_by_robot: {"zoo_compact_arm": "object_not_lifted, excessive_penetration", "zoo_dual_arm": "object_dropped, excessive_penetration", "zoo_hand_arm": "grasp_not_achieved, object_dropped", "zoo_jaw_arm": "object_not_lifted", "zoo_long_arm": "certified"}
- **Media:**

  | file | size | sha256 |
  |---|---|---|
  | `media/any-robot-zoo-demos/zoo_compact_arm-reach-out-as-far-as-you-can-and-then-come-ba.gif` | 574 kB | `977e85b12d954685` |
  | `media/any-robot-zoo-demos/zoo_compact_arm-reach-out-quickly,-just-a-little.gif` | 513 kB | `46658272c1804a16` |
  | `media/any-robot-zoo-demos/zoo_dual_arm-lower-the-tool-down-toward-the-table.gif` | 219 kB | `ac570a560ed086c3` |
  | `media/any-robot-zoo-demos/zoo_dual_arm-reach-out-as-far-as-you-can-and-then-come-ba.gif` | 254 kB | `3cc5d9f94df43a0d` |
  | `media/any-robot-zoo-demos/zoo_hand_arm-reach-out-as-far-as-you-can-and-then-come-ba.gif` | 457 kB | `7065c044a3542d18` |
  | `media/any-robot-zoo-demos/zoo_hand_arm-trace-a-big-circle.gif` | 466 kB | `5a289467d92bb2f3` |
  | `media/any-robot-zoo-demos/zoo_jaw_arm-reach-out-as-far-as-you-can-and-then-come-ba.gif` | 399 kB | `053d5fa530a2d5bd` |
  | `media/any-robot-zoo-demos/zoo_jaw_arm-wave-three-times.gif` | 397 kB | `0ab37cf17993623a` |
  | `media/any-robot-zoo-demos/zoo_long_arm-pick-up-the-block.gif` | 697 kB | `5c5498229b76caa0` |
  | `media/any-robot-zoo-demos/zoo_long_arm-reach-out-as-far-as-you-can-and-then-come-ba.gif` | 385 kB | `167e3f039dead028` |
  | `media/any-robot-zoo-demos/zoo_long_arm-sweep-across-the-workspace,-then-hold-still.gif` | 400 kB | `791db513029e78de` |
  | `media/any-robot-zoo-demos/zoo_tool_arm-reach-out-as-far-as-you-can-and-then-come-ba.gif` | 384 kB | `c3209522bcc03bce` |
  | `media/any-robot-zoo-demos/zoo_tool_arm-sweep-slowly-across-in-front-of-you.gif` | 355 kB | `700bd61fa28495c5` |
  | `media/any-robot-zoo-demos/demo-manifest.json` | 9 kB | `c01ebb30f5544417` |
  | `media/any-robot-zoo-demos/grasp-report.json` | 2 kB | `26bd364b3d716607` |

## Any-robot environment trials and IRL robots (any-robot/results)

- **Location:** `any-robot/results` (untracked)
- **Period:** 2026-08-27 to 2026-08-28
- **Produced by:** `uv run python scripts/run_trials.py` over the six authored worlds (bench_jig, desk_bench, far_pallet, pallet_cell, raised_shelf, work_table) after the tier split, plus contact probes and prompts on the bench robots (KUKA KR6/LWR, iiwa7, uHand2, EEZYbotARM MK1, Beetlebot, SO-101). `scripts/build_studio.py` and `scripts/export_viewer.py` render any of them.
- **Code:** any-robot on the feat/environments-and-contact line (PR 12, PR 14, and the six 28 Aug commits merged as PR 21). trace.json provenance hashes the rigby_core base tree (19 files).
- **On disk:** 363 files, 208.9 MB, modified 2026-08-28 to 2026-08-28, media {'.gif': 42}
- **Facts:**
  - traces: 241
  - kinds: {"environment_trial": 152, "contact_probe": 16, "prompt": 73}
  - accepted: 78
  - rejected: 163
  - failure_codes: {"unafforded_schema": 32, "grasp_not_achieved": 24, "object_inside_reach_hole": 21, "unreachable_object": 17, "object_too_wide": 17, "object_not_lifted": 16, "unsupported_morphology": 5, "object_out_of_reach": 5, "object_dropped": 5, "scene_not_buildable": 5, "object_too_narrow": 4, "no_gripper": 4, "object_not_carried": 3, "object_not_named": 1, "deterministic_gate_failed": 1, "excessive_penetration": 1, "joint_limit_violation": 1, "ungroundable": 1}
  - robots: {"eezybotarm_mk1": 28, "zoo_jaw_arm": 25, "uhand2": 21, "kuka_kr6": 17, "beetlebot": 15, "panda": 15, "so101": 15, "so101_undeclared": 15, "zoo_long_arm": 15, "iiwa7": 14, "kuka_lwr": 14, "zoo_compact_arm": 13, "zoo_dual_arm": 13, "zoo_hand_arm": 13, "makerpro_5dof": 4, "zoo_tool_arm": 4}
  - created: ["2026-08-27", "2026-08-28"]
  - code_base_trees: {"rigby_core@19622b3502d7": 241}
  - top_level_files: ["intake.json", "studio.html", "trials.json"]

## Gripper milestones and recordings (Angelo, branch grasp/auto-lift)

- **Location:** `origin/grasp/auto-lift:rigby-poc` (on the branch only)
- **Period:** 2026-08-27 to 2026-09-09
- **Produced by:** Each `milestones/runs/<name>.json` is a complete `gripper_clip_v1` recording (every frame's joint poses, link geometry, block pose, contact forces, penetration) from the torque-driven gripper controller in `rigby_poc.gripper` and `rigby_poc.closed_loop`, played by `frontend/public/static/gripper.html?clip=<name>` under `npm run dev --prefix frontend`. The `.webm` files are screen recordings of that viewer; the branch commits no recorder script, so the exact capture settings are Angelo's to state. `RESULTS.txt` is the placed/lift/penetration table.
- **Code:** PR 19 (grasp/auto-lift); rebased copy on the new base is grasp/auto-lift-on-main.
- **Facts:**
  - tip: f00861db5f72404221fc6d92a63250c89e5569ac 2026-09-10 fix(serve): one BLAS thread, set before numpy loads
  - files: 27
  - bytes: 26532983
  - results table:

    ```text
    run                             placed    lift_cm   pen_mm  frames
    ------------------------------------------------------------------
    gripper-behind-the-reach        False        0.00    0.000     780
    gripper-centre-as-tuned         True        28.86    0.036     780
    gripper-far-centred             True        29.73    0.215     780
    gripper-left-and-far            True        32.12    0.344     780
    gripper-left                    True        28.60    0.265     780
    gripper-narrow-block            True        29.28    0.056     780
    gripper-near-centred            True        28.58    0.057     780
    gripper-off-the-scan-grid       True        28.69    0.316     780
    gripper-right-and-near          True        28.94    0.315     780
    gripper-right                   True        28.62    0.290     780
    gripper-run                     True        28.86    0.036     420
    gripper-tall-thin-block         True        25.51    0.034     780
    gripper-wide-block              True        28.44    0.047     780
    ```
- **Media:**

  | file | size | sha256 |
  |---|---|---|
  | `media/gripper-milestones/milestones--README.md` | 5 kB | `2857eaaf60a575a1` |
  | `media/gripper-milestones/milestones--RESULTS.txt` | 1 kB | `186e95296f8a39d9` |
  | `media/gripper-milestones/milestones--SENSING-BOUNDARY.txt` | 1 kB | `09ef9e1607b8512a` |
  | `media/gripper-milestones/milestones--fridge-filmstrip.png` | 416 kB | `e96620a0347cff83` |
  | `media/gripper-milestones/milestones--two-cam-check.png` | 139 kB | `5dcd27d3238db789` |
  | `media/gripper-milestones/milestones--view-room.png` | 130 kB | `9b7a9182b1510f28` |
  | `media/gripper-milestones/milestones--view-wrist.png` | 0 kB | `d73102ce878643f5` |
  | `media/gripper-milestones/milestones--wrist-view.png` | 1 kB | `9347fe86f6b44851` |
  | `media/gripper-milestones/videos--01-lift-and-place-known-positions.webm` | 1.5 MB | `bdd1469e8f43b9e1` |
  | `media/gripper-milestones/videos--02-lift-and-place-camera-only.webm` | 1.6 MB | `eb6354d98eab6f0f` |
  | `media/gripper-milestones/videos--03-cabinet-attempt-not-working.webm` | 2.0 MB | `f57185baf74bebb6` |
  | `media/gripper-milestones/videos--README.md` | 3 kB | `9a0f08fd2fe2b94c` |
  | `media/gripper-milestones/videos--gripper-two-cameras.webm` | 1.7 MB | `c58eb2950a7249f3` |
  | `media/gripper-milestones/videos--gripper-vlm-directed.webm` | 2.5 MB | `9ac71f5d74debd7f` |

## Regenerating

Every set above except the recordings can be regenerated from the tracked code: `run_trials.py`, `build_demos.py` and `grasp_report.py` in `any-robot/scripts`, `render_demo_gif.py` in `humanoid/evals`, and the humanoid app for the archive. Regenerated results will not be byte-identical to the originals where the code moved on; the manifest's `code` field says which line of history each set came from.

