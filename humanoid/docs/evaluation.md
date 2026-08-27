# Autonomous acceptance evaluation

Rigby's product path uses the public HTTP API. A run begins at `POST /api/v1/pipeline-runs`, and Motion Studio polls `GET /api/v1/pipeline-runs/{id}` to render persistent events and the terminal result.

No new human ratings are required. Historical reviews are frozen calibration evidence. Current acceptance combines deterministic trajectory/physics gates with the calibrated blinded VLM.

## Local regression suite

From `rigby-humanoid`:

```powershell
uv run pytest -q
Push-Location frontend
npm test
npm run build
Pop-Location
```

The backend suite covers strict schemas, semantic routing, unsupported prompts, action sequences, gesture fingers, strike paths, paired-arm trajectories, full-body support and obstacles, free-object pickup, object lifecycles, coordinate conversion, parameter-only recompilation, atomic persistence, candidate diversity, bounded repair, capture contracts, VLM calibration, GLB validity, and API behavior.

Frontend tests cover terminal pipeline state presentation; the production build type-checks every UI/API/viewer contract.

## Product smoke

Start the server, open `http://127.0.0.1:8000`, and submit representative prompts through **Generate 5 & choose**. At minimum, retain one calibrated gesture and one physical pickup. Expanded releases should also cover strike, composite, object interaction, full-body, obstacle, and sequence families.

Every supported run must visibly progress through:

1. semantic planning;
2. candidate compilation and deterministic checks;
3. a complete five-candidate judging batch or explicit replenishment failures;
4. complete-FOV egocentric and orbit capture;
5. blinded VLM ranking;
6. optional bounded repair;
7. one selected result or a typed terminal failure.

Reloading the page must reconnect to the same persisted run.

## Autonomous release audit

With the server running, an API key configured, and the required smoke evidence present in the local result archive:

```powershell
uv run python -m evals.autonomous_goal_audit
```

The audit requires:

- the 28-case corruption calibration to remain passing with zero false accepts;
- the frozen final calibration to retain at least 80% VLM-human agreement and 90% A/B order consistency;
- completed public gesture and pickup smoke runs;
- at least five recorded candidates per required smoke run;
- an accepted winner scoring at least 4 for semantic match, recognizability, anatomy, and overall quality;
- deterministic structural validity;
- uncropped 1600×900 evidence with the complete 94-degree egocentric FOV;
- persisted provenance and final GLB.

The checked-in [frozen calibration summary](evidence/frozen-judge-calibration.json) documents the immutable human/corruption benchmark. Full local run artifacts remain ignored because they are large and machine-specific.

## Failure policy

Unsupported, contradictory, physically unreachable, incomplete, or visually unacceptable requests end with typed inspectable failures. Missing evidence never becomes a pass. A VLM cannot override deterministic physics or safety. The pipeline never chooses a failing animation merely to return a result.

Per-family held-out reporting should separate semantic routing, compilation, structural rejection, evidence failure, visual rejection, repair exhaustion, and export failure. That separation is necessary before deciding whether the next investment belongs in primitives, scene affordances, rig calibration, capture, VLM calibration, or learned in-betweening.
