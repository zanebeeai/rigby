# Autonomous VLM authoring flywheel

## Product contract

Rigby accepts a free-text instruction plus the current typed scene. A request either finishes with one structurally valid, visually accepted, editable animation or with a typed inspectable failure. The authoring path never requires a human rating.

1. The planner maps language to a strict `MotionProgram`. It selects intent, active limbs, scene objects, phases, and bounded semantics; it never emits per-frame rotations or success thresholds.
2. A deterministic capability parser reconciles the model output with the implemented grammar. Clear supported prompts are not lost to a model rejection, and unsupported prompts are not forced into an unrelated skill.
3. Rigby expands the semantic program into a pool of named parametric recipes and compiles candidates through the same local primitive, IK, contact, support, and export code used by the editor.
4. Deterministic checks reject invalid anatomy, kinematics, contact, object state, ground support, obstacle, or recovery before any image is sent to a model.
5. Motion hashes and task-space descriptors remove exact or perceptually negligible duplicates. The five-way batch contains five rankable alternatives or the trace records why a complete batch could not be assembled.
6. Every rankable candidate is captured at phase-aware times in the 94-degree egocentric camera and a task-aware orbit camera. Raw evidence is the complete uncropped 1600×900 canvas.
7. The VLM sees blinded candidates, the original prompt, chronological contact sheets, presented/impact/contact phases, and objective diagnostics. It never sees recipe names, seeds, or hidden preference labels.
8. The lightweight Luna route ranks the candidates first. An uncertain or inconsistent decision may use the configured Terra fallback. A product run has a hard four-model-call ceiling across judging and repair.
9. If no candidate passes, the judge may propose one schema-constrained parameter repair and Rigby may run one replacement round. The repair cannot change intent, active side, phase order, primitive vocabulary, or deterministic limits.
10. The winner opens in Motion Studio for synchronized playback, parameter editing, provenance inspection, and GLB export. Exhausting the bounded loop returns `no_acceptable_candidate`.

The same flywheel is used for gestures, strikes, composite arm trajectories, object interactions, full-body skills, and sequences. The deterministic measurements and phase-aware evidence vary by intent; the persistence and judge contract do not.

Every stage is written to `results/pipeline-runs/<run-id>/run.json`. Candidate programs, clips, evidence manifests, judgments, repairs, and `flywheel-trace.json` live in the adjacent `artifacts/` tree. Atomic writes and persistent events let Motion Studio resume the run after a reload.

## Judge validity

The frozen 10-pair calibration passed with:

- 9/10 VLM-human agreement;
- 9/10 human preference for the uncorrupted base motion;
- 10/10 A/B order consistency.

The associated corruption suite passed 28 deliberate wrist, wrong-joint, finger-shape, and timing corruptions with zero false accepts and 10/10 order-consistent base wins. The compact immutable record is checked in at [evidence/frozen-judge-calibration.json](evidence/frozen-judge-calibration.json).

No future generation or acceptance workflow asks for human ratings. This calibration supports the VLM as an automated POC proxy; it does not establish broad human preference or immunity to model drift.

## Cost and stopping rules

- Five candidates are judged together rather than with five independent unary calls.
- Structural and duplicate filtering happens locally before visual inference.
- The primary judge is the lighter configured model; fallback is conditional.
- One run permits two rounds and at most four judge/repair calls.
- A repair patch contains only bounded fields already owned by the compiler.
- Missing evidence, malformed output, a failed score threshold, or exhausted calls is never converted into a pass.

Planner calls are recorded separately in clip provenance. Use `provider: "offline"` when working on deterministic semantics, compilers, or tests without model spend.

## Run the product

```powershell
Push-Location frontend
npm run build
Pop-Location
uv run rigby-poc
```

Open `http://127.0.0.1:8000`, enter a prompt, and choose **Generate 5 & choose**. The live trace shows planning, proposals, structural decisions, capture, VLM scores, any repair, and final selection as they arrive.

## Release gates

- **Planner:** supported prompts produce strict executable programs; contradictory or unsupported requests produce typed reasons.
- **Candidate set:** five structurally valid and non-duplicate candidates reach the judge, or every replenishment/rejection is explained.
- **Evidence:** both complete 1600×900 views, the 94-degree egocentric contract, chronological phase samples, hashes, and diagnostics are present.
- **Judge:** frozen calibration thresholds remain passing and deliberate corruptions remain rejected.
- **Winner:** the selected clip passes deterministic limits and receives at least 4/5 for semantic match, recognizability, anatomy, and overall quality.
- **Artifact:** the final result contains editable parameters, provenance, replayable frames, and a valid GLB.
- **Reliability:** the public asynchronous API persists progress and exposes an explicit terminal state after reload.

See [evaluation.md](evaluation.md) for local regression and model-backed smoke commands.
