# Rigby

The complete architecture, demos, pipeline contracts, setup, APIs, persistence model, verification strategy, limitations, and research links are documented in the [repository README](../README.md).

Rigby v2 is the local-first MuJoCo implementation. It keeps the original GLB as the visual identity while executing and certifying motion on a separate articulated, free-root physical human. The previous POC remains available for migration and comparison; it is not positive context for v2 unless its result is re-simulated and independently certified.

## Local v2 setup

Requirements: Windows 11, Python 3.12, MuJoCo 3.11, ffmpeg, `uv`, and Docker Desktop (or a reachable PostgreSQL/pgvector service).

```powershell
docker compose -f compose.v2.yaml up -d postgres
.\.venv\Scripts\python.exe scripts\v2\preflight.py
.\.venv\Scripts\rigby-v2.exe
```

Start the worker in a second terminal:

```powershell
.\.venv\Scripts\rigby-v2-worker.exe
```

The suite needs no server, no browser, no API key and no network. Every documented
invocation, the tiering markers, and why coverage stays off the default run are in
[docs/testing.md](docs/testing.md).

See [docs/evaluation.md](docs/evaluation.md) for the model-backed autonomous release audit and [docs/vlm-flywheel.md](docs/vlm-flywheel.md) for the judge/repair contract.
The API listens on `http://127.0.0.1:8010`; health is at `/api/v2/health`. Canonical rigs and all six object packs can be staged through the v2 API. See the [operator runbook](docs/v2/operator-runbook.md) for installation, startup, migration, performance, backup/restore, rollback, and release procedures. The [completion audit](docs/v2/completion-audit.md) is the source of truth for which implementation and release gates have genuine evidence.

## What v2 includes

- Immutable typed contracts, content-addressed artifacts, PostgreSQL leases, cancellation, crash recovery, and replay manifests.
- A 67-actuator full-body MJCF with articulated palms and all required phalanges, three body profiles, and audited visual/physical pose conversion.
- Self-contained MjSpec/MJZ scenes for grasp/place, drawer, lever/button, hand tool, container/lid, and bimanual tasks.
- Phase-aware joint motion compilation, an arm/hand refinement foundation, 240 Hz inverse-dynamics/PD control, deterministic balance, object predicates, three-repeat certification, and calibrated survivor variations.
- An exact-five bounded candidate pipeline, audited three-camera evidence, structured anonymous VLM judging, certified-only retrieval rules, and N+1 release isolation.
- A resumable evidence-backed supported benchmark runner, a sealed 100-case adversarial firewall, retained-file human-study verification, and a real offline SigLIP2 keyframe inference canary.
- A sealed 300-supported/100-adversarial benchmark definition, calibration/dataset governance, migrations, backup/restore, wheel smoke, and a fail-closed release checklist.

These bullets describe implemented capabilities, not a ship declaration. The
100-case adversarial firewall and one robust physical button task are sealed,
but the remaining five physical task families, the 300 supported-case run,
complete model-backed indexes, and human calibration evidence remain subject to
the completion audit and release checklist.

## Verification

```powershell
$env:RIGBY_TEST_POSTGRES='1'
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\v2\wheel_smoke.py
.\.venv\Scripts\python.exe scripts\v2\production_renderer_benchmark.py
```

The model-backed planner and judge require an explicitly selected compatible model and API credentials; neither silently falls back or changes model. Embedding weights are separately pinned, sealed, and offline-only. Deterministic simulation and ordinary tests do not require an external model call.

## Legacy POC

The previous app still runs with `rigby-humanoid`, and its own setup and verification
steps are below, unchanged. Import its archive as quarantined records with:

```powershell
.\.venv\Scripts\python.exe scripts\v2\import_legacy_archive.py .\results --limit 10
```

Remove `--limit` only after checking the canary import and available artifact storage.

### Run the POC

Requires Python 3.12, `uv`, Node.js 20.19+, npm, and Google Chrome.

```powershell
uv sync --extra dev
Push-Location frontend
npm ci
npm run build
Pop-Location
uv run rigby-humanoid
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), enter a prompt, and choose **Generate 5 & choose**. The studio displays planning, candidate compilation, deterministic checks, full-FOV capture, VLM judging, bounded repair, and final selection as they happen.

Rigby reads `OPENAI_API_KEY` from this directory's `.env` or the repository-root `.env`. Neither file is committed. Use `provider: "offline"` for deterministic planner/compiler development without an OpenAI planning call.

### Verify the POC

```powershell
uv run pytest -q
Push-Location frontend
npm test
npm run build
Pop-Location
```

The suite needs no server, no browser, no API key and no network. Every documented
invocation, the tiering markers, and why coverage stays off the default run are in
[docs/testing.md](docs/testing.md).

See [docs/evaluation.md](docs/evaluation.md) for the model-backed autonomous release audit and [docs/vlm-flywheel.md](docs/vlm-flywheel.md) for the judge/repair contract.
