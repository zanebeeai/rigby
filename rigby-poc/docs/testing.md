# Running the tests

The suite is hermetic. It needs no server, no browser, no API key and no network.
That is a property worth protecting, so it is stated here and enforced by
`tests/test_invocations.py`.

## The invocations

| What you want | Command |
| --- | --- |
| Everything (the default) | `uv run pytest` |
| One file | `uv run pytest tests/test_full_body_motion.py` |
| One test | `uv run pytest tests/test_full_body_motion.py::test_walk_forward` |
| Frontend | `npm --prefix frontend test` |
| Frontend type-check | `npm --prefix frontend run build` |
| Coverage (rarely — see below) | `COVERAGE_CORE=sysmon uv run pytest --cov=rigby_poc --cov=evals` |

## Tiers

Three markers are registered, and each lane marks its own files:

| Marker | Contents |
| --- | --- |
| `fast` | no compile, no pipeline, no corpus, no subprocess -- pure units, contract, schema |
| `medium` | live compiles, corpus recompiles, flywheel with stubs |
| `slow` | render, real browser, model calls, calibration -- opt-in, never implied |

Declare at module scope, as the first statement after the imports:

```python
import pytest

pytestmark = pytest.mark.medium
```

**Until 09b lands these are declarative only** -- registered so marking a file
emits no warning, but nothing is deselected and the default invocation still runs
everything. 09b adds the default `-m "fast or medium"` selection and a
marker-completeness test that fails collection on an unmarked file.

Measured for 09b, so the budget is not mistaken for a margin: the 30 cheapest of
41 test files total **29.3 s** against the 30 s `fast` budget, on a suite that
went 344 to 920 tests in a single session. It is met but not durable. When it
breaches, re-derive the budget rather than demoting a file to `medium` -- that is
how a tier stops meaning anything.

## Coverage is not on by default, and must not become so

`addopts` is `-q` only. Neither `.pth` hook in site-packages is active:
`a1_coverage.pth` needs `COVERAGE_PROCESS_START` / `COVERAGE_PROCESS_CONFIG` and
`pytest-cov.pth` needs `COV_CORE_SOURCE`. None of those are set, so a plain
`uv run pytest` does no line tracing.

Keep it that way. Measured on `test_action_sequences.py`, adding
`--cov=rigby_poc --cov=evals` takes it from 5.93 s to 23.42 s — a **3.95×**
multiplier, almost all of it tracing the 7,815-line `compiler.py`.

Two rules follow:

1. **Never add `--cov` to `addopts`.** `tests/test_invocations.py` fails the suite
   if anyone does. Coverage belongs to the scheduled job, not to the inner loop.
2. **Always set `COVERAGE_CORE=sysmon` when you do run coverage.** That selects
   Python 3.12's `sys.monitoring` backend instead of the `settrace` one and
   removes most of the multiplier. The nightly CI job sets it.

## Why no secrets are needed

Verified, and re-verified on 2026-08-24 against `dfac5c1`:

- `tests/backend_contracts.py` drives the API through
  `fastapi.testclient.TestClient`, which is in-process ASGI. Nothing binds a port.
- `evals/capture.py` imports Playwright at module scope, so Playwright must be
  *installed* for tests that import it, but no test calls `sync_playwright` — the
  only two call sites are `evals/capture.py:1294` and
  `evals/render_demo_gif.py:127`, and every test injects `capture_fn=fake_capture`.
- `judge.py` imports `OpenAI`, but every test injects a fake client. No test reads
  `OPENAI_API_KEY`.

The measurement: the full suite passes with `OPENAI_API_KEY` unset, no `.env`
present, and every `AF_INET`/`AF_INET6` socket raising on construction.

**CI therefore needs no secrets, and must not be given any.** A CI job that
carries an API key can start depending on one without anyone noticing.

MuJoCo is the only hard external dependency, and only for grasp and pickup
intents. It is a wheel, not a service. Note that importing it shells out to
`sysctl` on macOS, so an offline guard must block sockets rather than
subprocesses wholesale.

### The cost of that, stated plainly

The suite is hermetic **because the two most expensive real components are
faked**, and that is a coverage statement as much as a speed one.

Measured on a real browser: a render is **2.85 s per candidate** — both views, 30
snapshots, batched seek path — against **369 ms** to compile the same candidate.
A render is roughly **7.7x a compile**. The suite's capture double is a 10 ms
sleep, so a per-stage share taken from a suite run says nothing about where real
time goes.

Be precise about what is uncovered, because the gap is narrower than "capture is
untested" and mistaking it invites an expensive fix for the wrong thing.
`tests/test_capture_asset_integrity.py` exercises the real `CaptureSession`
against a fake page — manifest assembly, asset-hash verification, the deadline,
the retry policy — with no browser, and those run everywhere including Windows.

What has **no coverage at all is the browser boundary**: page load, seek, canvas
read, and the PNG bytes. On Windows that boundary has never been exercised, since
the end-to-end test skips cleanly without `npm ci` and a real Chrome on the
runner. Skipping is correct behaviour; it also means the platform most likely to
break that path is the one that never tests it.

Likewise, do not read a green suite as evidence about model behaviour: every
judge is a fake client.

That gap is deliberate: closing it means running a browser in CI, and the cost is
real while the risk is low. It is recorded here rather than left implicit so that
"the suite is green on Windows" cannot harden into a claim about capture.
