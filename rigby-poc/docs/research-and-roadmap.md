# Rigby research synthesis and development roadmap

## Executive decision

Rigby should not be a direct text-to-joint-rotation model. The practical architecture is:

1. An LLM selects a compact, typed intent: action, hand, hand shape, object, and rejection reason.
2. A deterministic action-semantic layer turns motion-bearing verbs into observable phase requirements and task-specific assertions; it rejects interpretations that preserve only a pose while losing the action.
3. A deterministic smart-primitive layer expands that intent into object-relative phases, sockets, end-effector targets, hand shapes, and hard or soft constraints.
4. A calibrated rig compiler solves arm IK, finger articulation, timing, and retargeting.
5. A fixed verifier proves anatomy, visibility, kinematics, semantic motion obligations, contact, lift, hold, safety, and continuity from the generated trajectory.
6. A full-FOV VLM loop ranks diverse structurally valid candidates, proposes only bounded parameter repairs, and preserves complete traces.
7. A renderer and authoring UI expose the live planning/candidate/judge pipeline, final egocentric/orbit playback, and parameter-only recompilation without another planner call.
8. A learned in-betweening model is added later, only where deterministic interpolation has a measured quality ceiling and sufficient data exists.

This is the smallest architecture that combines the useful ideas in the supplied research while avoiding their stated limitations. The POC began with one expressive hand gesture and one physical pickup; the current revision keeps that calibrated core and extends the same contracts to strikes, paired-arm trajectories, object lifecycles, full-body skills, obstacle-aware motion, and short action sequences.

## What each source contributes

### Agentic game creation lecture notes

The notes describe a useful production pattern rather than a motion algorithm:

- Decompose a complex artifact into hierarchical, parameterized primitives.
- Give the model vector- and helper-level APIs instead of forcing it to reason over low-level geometry.
- Render candidates, judge them, return structured critiques, and retain successful examples as a data flywheel.
- Combine perceptual review with deterministic unit tests.
- Generate multiple candidates while quality is still the priority; retain the traces needed to distill a cheaper one-shot path later.

Rigby adopts the hierarchy, deterministic tests, and a bounded five-candidate VLM loop for each current POC request. Cost and latency remain explicit trace fields so this can later be distilled.

### Ego2Robot

Ego2Robot demonstrates that egocentric human video can be converted into aligned action data at large scale. Its most reusable pipeline ideas are:

- Convert hand keypoints into a compact end-effector position, orientation, and aperture representation.
- Smooth position and width trajectories and use quaternion interpolation for orientation.
- Search for a feasible base placement and validate representative keyframes with IK.
- Represent actions in the camera frame to unify varied egocentric camera placements.
- Curate at three levels: pipeline invariants, statistical outlier/discontinuity filters, and semantic video-text consistency.
- Evaluate visual, scene-layout, embodiment, and semantic perturbations separately instead of hiding failure modes inside one aggregate score.

The paper converts articulated human hands to parallel-jaw grippers and explicitly identifies discarded finger articulation as a limitation. Rigby must not copy that collapse. A Rigby data pipeline should preserve MANO-style or equivalent multi-finger pose, contact labels, object pose, and camera calibration. Ego2Robot is therefore a strong blueprint for future data ingestion and quality curation, not a ready-made dexterous animation dataset.

### MotionBricks

MotionBricks supplies the closest high-level architecture:

- Smart primitives generate target keyframes through one unified interface.
- A shared generative in-betweening backbone accepts a flexible subset of root and pose constraints.
- Smart-object interactions contain intent keyframes plus an interaction binding for detection, sockets/placement, and object-relative anchoring.
- Contact keyframes can be hard constraints, while preparation and exit keyframes can be soft guidance.
- Runtime replanning occurs when commands change or the motion buffer runs low.

This maps directly to Rigby's typed phases and sockets. Reach/preshape/recover are soft phases; contact/close/hold are hard phases. Object-local anchors let one primitive adapt to position, scale, and approach changes.

MotionBricks also explains why Rigby should not start by training a comparable neural backbone. The reported model is supported by roughly 350,000 clips, while the paper says object interactions are not explicitly modeled, the released/open subset lacks finger and object motion, kinematic output can be physically implausible, and high-quality retargeting remains expensive. Rigby should first make the primitive and verifier contract stable, then evaluate an existing in-betweening backbone or train a targeted hand/arm model with contact-rich data.

### MoVer

MoVer shows the value of compiling language into an executable verification program rather than asking a visual model for a single opaque score. Its first-order spatio-temporal predicates produce human-readable, predicate-level failures that can be fed back into correction. On its 5,600-prompt synthetic benchmark, LLM verification-program synthesis reached 95.1%; detailed feedback corrected substantially more animations than no or pass/fail-only feedback.

The warning is equally important: an incorrect generated verifier can confidently approve an incorrect result. Rigby therefore uses a fixed, versioned verification schema for supported primitives. The LLM may choose the intent, but it does not invent the physical success definition. Numeric trajectory checks, scene contacts, and physics are authoritative. Human-readable failed assertions can later drive an offline repair loop.

### Self-Consistency for LLM-Based Motion Trajectory Generation and Verification

This paper samples multiple trajectories, clusters them under task-appropriate geometric invariances, and chooses the dominant family. It reports 4-6 percentage-point generation gains and better verification precision/F1 than VLM baselines. Performance largely stabilizes around 10 samples.

For Rigby, consistency should be measured in task space rather than raw joint space: end-effector path, contact sequence, hand-shape state, relative object motion, and allowed mirror/scale/time warps. Multi-sample consensus is too expensive for the normal authoring path, but it is valuable for:

- discovering new smart primitives from example prompts;
- selecting among ambiguous candidate contact plans;
- mining hard cases for evaluation;
- validating a learned in-betweener against a family of acceptable motions.

It cannot resolve genuinely ambiguous or multi-modal prompts on its own. Rigby should expose ambiguity or ask for a choice instead of treating the largest cluster as truth.

## The POC now implemented

The replacement POC is end to end and intentionally contract-driven:

- Supported program families now include gestures, contact pickup, hooks/jabs/crosses/uppercuts, one- and two-arm trajectories, typed object interactions, full-body locomotion/poses/obstacles, and short action sequences.
- Unsupported or contradictory requests are explicitly rejected rather than silently approximated.
- The OpenAI planner returns compact semantics. Audited local code expands phase graphs, numeric defaults, contacts, support relationships, and recovery.
- Motion-bearing social verbs are interpreted as typed observable actions rather than pose aliases. A greeting wave, for example, requires setup, an open hand, frontal side-to-side cycles, sufficient shoulder-relative excursion and direction reversals, then recovery; the contract composes with whole-body motion.
- The preserved humanoid GLB is calibrated through a rig profile. Clip rotations are rest-relative and exported as GLB 2.0.
- Arm and leg motion uses analytic IK. All five digits are independently posed.
- Pickup uses object sockets and phases: reach, preshape, contact, close, lift, hold, recover.
- MuJoCo simulates a free block and five independently actuated/contactable digit proxies. No weld, hidden attachment, or parallel-gripper shortcut is allowed.
- Additional object lifecycles preserve explicit ownership, release, flight, receive, placement, and landing state in the clip contract.
- Full-body compilers own ground support, foot targets, balance/recovery, obstacle clearance, climbing contacts, and root motion.
- All advertised controls recompile deterministically with zero planner calls.
- Motion Studio provides prompt-to-animation authoring, synchronized first-person/orbit views, timeline scrubbing, controls, history, live pipeline events, metrics, provenance, and export.
- The best-of-five loop applies deterministic validity and diversity gates before full-FOV VLM ranking and bounded repair.

The breadth is an implemented procedural vocabulary, not a claim of unrestricted text-to-motion generalization. Every added skill still needs a named contract, task-specific measurements, held-out prompts, and failure behavior.

The older acceptance harness established useful automated coverage:

- Planning: 60/60 supported prompts and 20/20 unsupported prompts correct.
- Physical pickup: 150/150 trials pass across five block shapes, five poses, both hands, and three paraphrases.
- Safety: 210/210 generated clips pass all invariants.
- Structure: articulated digits and Y-up/Z-up conversion are proven on 210 rollouts.
- Parameter behavior: 28/28 controls move in the expected direction, are model-free, and compile below 3 seconds.
- End-to-end latency: p95 2.57 seconds and 2.98 seconds across 60 samples per run, below the 15-second gate.
- Export: 11/11 GLBs reimport within tolerance with provenance.
- Historical human review showed that the original gesture quality was not close to acceptable: run 000018 had 2/20 clips rated 4+ in both views and run 000019 had 4/20. The corrected primitive and calibrated autonomous flywheel are documented in [vlm-flywheel.md](vlm-flywheel.md).

## End goals and investment gates

### Gate 0 - certify this POC

Pursue the next stage only after the frozen judge calibration and public autonomous gesture/pickup smoke runs pass every versioned gate. No new human review is required.

Required evidence:

- At least 95% correct supported planning and at least 95% correct rejection.
- Finger assertions on 20/20 gesture clips.
- The frozen final calibration retains at least 80% VLM-human agreement and 90% A/B order consistency, while the automated corruption suite stays below 5% false acceptance.
- A real gesture run and a real pickup run each produce five rankable candidates, a structurally valid accepted winner, complete full-FOV evidence, provenance, playback, and GLB export.
- At least 90% of the 150 pickup trials pass; no weld or attachment is permitted.
- Zero joint-limit, fixed-root/foot, unresolved collision, NaN, discontinuity, and penetration violations at the configured thresholds.
- All 28 controls have the correct observable response with no model calls and under 3-second compile time.
- Every long-running stage emits persistent progress and a reload resumes the same run.
- 11/11 representative GLBs reimport within tolerance with provenance.

Stopping rule: if the calibrated judge rejects both bounded rounds, return a typed failure and redesign the diagnosed primitive, hand presentation, timing, mesh skinning, or rig calibration. Do not silently choose a failing candidate.

### Gate 1 - useful authoring alpha

After Gate 0, expand the deterministic library to the smallest useful manipulation vocabulary:

- Hand shapes: open, fist, point, pinch, shaka, relaxed grasp.
- Object actions: grab, place, push, pull, press, turn, and simple handoff-to-socket.
- Interaction model: approach regions, grasp candidates, hard contact constraints, release conditions, and object-relative recovery.
- Editing: every semantic and physical parameter remains reversible and re-compiles without an LLM call.
- Failure UX: unreachable, collision, ambiguous-object, unsupported-sequence, and failed-contact explanations identify the exact predicate/phase.

Alpha exit criteria:

- At least 500 held-out paraphrases across the action vocabulary with 95% correct intent/object/hand selection and 95% correct rejection.
- At least 90% physical success over a versioned matrix of object dimensions, masses, friction, placements, and both hands for every contact action.
- At least 80% of held-out clips clear the calibrated automated visual gate in both full-FOV views, with periodic corruption recalibration preventing judge drift.
- Three materially different humanoid rigs retarget within joint/safety limits, with no per-clip manual fixes.
- Parameter changes respond in under 500 ms at p95 for deterministic recompilation; initial model-backed generation remains under 5 seconds p95.

### Gate 2 - data and learned in-betweening pilot

Start a learned model only after Alpha failures are categorized and at least 30% are clearly interpolation/naturalness failures rather than bad intent, rig, contact, or camera setup.

Pilot data requirements:

- Preserve full finger pose, object 6-DoF pose, hand-object contacts, head/camera pose, and scene scale.
- Use camera-relative end-effector trajectories plus object-relative contact frames.
- Apply Ego2Robot-style L1 invariant filters, L2 distribution/discontinuity filters, and calibrated L3 semantic VLM review.
- Split evaluation by scene, object instance, camera, embodiment, and language paraphrase.
- Maintain licenses and provenance for every source clip.

Pilot model requirements:

- Condition on an arbitrary subset of hard/soft task-space keyframes.
- Generate only the in-betweened pose/timing, never redefine contact success.
- Beat deterministic interpolation under blinded calibrated VLM comparison while matching or improving every safety/physics gate.
- Demonstrate useful performance on held-out objects and at least three rigs before scaling data collection.

If an available MotionBricks-style model can accept the Rigby constraint contract, evaluate it before training from scratch. A bespoke large backbone is not justified until the pilot proves a measurable quality gap and a realistic data path.

### Gate 3 - production-oriented beta

The beta end state is a runtime authoring system, not merely a demo:

- Text, scene state, or game events compile into the same typed motion program.
- Programs remain inspectable, parameterized, serializable, and replayable.
- Continuous runtime replanning preserves active contact constraints.
- A candidate/contact planner handles clutter and multiple valid grasps.
- Verification produces phase-level explanations and safe fallbacks.
- Assets export to GLB and at least one target engine/runtime with equivalent transforms and events.
- Latency, determinism, cost, crash rate, and acceptance results are tracked by version.

Beta go/no-go criteria should include at least 20 interaction types, 10 object families, five humanoid rigs, 1,000 held-out scenes, 95% safe completion on supported tasks, and no regression across two consecutive release-candidate runs.

## Recommended next sequence

1. Keep the passing 10-pair human calibration frozen as the final human evidence; never request another rating in the authoring path.
2. Preserve gesture and physical pickup as the certified regression core while running every expanded intent family through the same public pipeline and trace schema.
3. Build a held-out autonomous matrix by skill, handedness, scene variation, paraphrase, and sequence composition; report per-family failure rates instead of one aggregate score.
4. If a gate misses, repair only the diagnosed primitive, staging, timing, contact, support, capture, or calibration dimension and rerun automatically.
5. Certify additional rig profiles and scene affordances before calling the current procedural breadth an authoring alpha.
6. Measure the deterministic naturalness ceiling before committing to egocentric data conversion or neural in-betweening.

## Non-goals for the next stage

- No claim of unrestricted or learned full-body text-to-motion generation.
- No unverified bimanual dexterity; paired-arm trajectories and typed handoff are narrower contracts.
- No LLM-generated per-frame joint rotations or success thresholds.
- No VLM-only declaration of physical success.
- No fake grasp attachment, welded object, or non-contact teleport.
- No large training run without a versioned data schema, held-out perturbation benchmark, and a deterministic baseline it must beat.

## Supplied research sources

- `Agentic game creation.pdf` - lecture notes, August 4, 2026.
- `2608.02580v1.pdf` - *Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data*.
- `motionbricks_siggraph_2026.pdf` - *MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives*.
- `3731209.pdf` - *MoVer: Motion Verification for Motion Graphics Animations*.
- `Ma_Self-Consistency_for_LLM-Based_Motion_Trajectory_Generation_and_Verification_CVPR_2026_paper.pdf` - *Self-Consistency for LLM-Based Motion Trajectory Generation and Verification*.
