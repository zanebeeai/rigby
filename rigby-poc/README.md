# Rigby POC

The complete architecture, demos, pipeline contracts, setup, APIs, persistence model, verification strategy, limitations, and research links are documented in the [repository README](../README.md).

## Run locally

Requires Python 3.12, `uv`, Node.js 20.19+, npm, and Google Chrome.

```powershell
uv sync --extra dev
Push-Location frontend
npm ci
npm run build
Pop-Location
uv run rigby-poc
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), enter a prompt, and choose **Generate 5 & choose**. The studio displays planning, candidate compilation, deterministic checks, full-FOV capture, VLM judging, bounded repair, and final selection as they happen.

Rigby reads `OPENAI_API_KEY` from this directory's `.env` or the repository-root `.env`. Neither file is committed. Use `provider: "offline"` for deterministic planner/compiler development without an OpenAI planning call.

## Verify

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
