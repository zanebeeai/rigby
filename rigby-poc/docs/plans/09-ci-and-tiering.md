# PR 09 — CI, test tiering, and coverage cost

Status: proposed, not started.
Scope: make the suite fast enough to iterate on, tier it, and run it automatically on both
developers' platforms.

Depends on: [03 — golden corpus](03-golden-corpus.md) for the fast tier's inputs.
Blocks: nothing, but every other PR is slower and riskier without it.

---

## 1. Problem

### 1.1 The suite is already fast — 79 seconds

Measured end to end, single process:

```
$ time uv run pytest -q
344 passed
77.01s user  0.67s system  97% cpu  1:19.39 total
```

Per-file timings for the dominant contributors:

```
26.37s  test_full_body_motion.py           (62 tests)
14.93s  test_arbitrary_prompt_coverage.py  (57 tests)
 7.11s  test_pipeline.py                   ( 4 tests)   ← worst per-test
 5.93s  test_action_sequences.py           (16)
 5.92s  test_prompt_family_matrix.py       (22)
 5.30s  test_strike_pipeline.py            ( 8)
 …      20 files under 2 s, 8 files at ~0.15 s
```

**Coverage is not enabled by default and never was.** `addopts` is `-q` only
(`pyproject.toml:39`), and both `.pth` hooks in site-packages are env-gated:
`a1_coverage.pth` requires `COVERAGE_PROCESS_START`/`COVERAGE_PROCESS_CONFIG`, and
`pytest-cov.pth` requires `COV_CORE_SOURCE`. None are set. The stray `.coverage` file in
the tree was an artifact of an explicit ad-hoc run, not the norm.

For the nightly coverage job (§3.1) the cost is real and worth knowing: measured on
`test_action_sequences.py`, `--cov=rigby_poc --cov=evals` takes 5.93 s → 23.42 s, a
**3.95× multiplier** from line-tracing the 8,446-line `compiler.py`. Use
`COVERAGE_CORE=sysmon` (Python 3.12's `sys.monitoring` backend) there.

**Consequence for this plan: there is no speed emergency.** 79 s is a perfectly workable
default. The value in this PR is tiering (so eval iteration can run a sub-30 s subset) and
CI (so Windows breakage is caught) — not raw speed.

### 1.2 No fixtures at all

There is **no `conftest.py` anywhere** in the repository, and `grep -rn "pytest.fixture"
tests/` returns zero hits. No custom fixtures, no session or module scoping, no shared
compiled clips.

Roughly 250 of 344 tests call `compile_motion` directly, so identical clips are recompiled
many times across the suite. The only caching on the hot path is
`kinematics.py:394` `@lru_cache(maxsize=1)` for the rig load, which is process-global and
does help.

### 1.3 No tiering, no markers

`grep -rn "pytest.skip\|skipif\|xfail\|importorskip\|pytestmark" tests/` returns empty.
Every `@pytest.mark.*` in the suite is `parametrize`. There is no way to run "just the fast
ones," which is exactly what eval development needs.

### 1.4 No CI

No `.github/`, no workflow, no automation. Given development is split across macOS and
Windows, nothing currently catches a change that works on one and breaks the other — the
most likely cross-platform hazard being the MuJoCo hash question in
[03](03-golden-corpus.md) §6.1.

### 1.5 The good news

**No test requires a live server, a browser, or an API key.** Verified:

- `tests/backend_contracts.py:459` uses `fastapi.testclient.TestClient`, in-process ASGI.
  No sockets anywhere.
- `evals/capture.py:14-15` imports Playwright at module scope, so Playwright must be
  *installed* for tests that import it, but **no test ever calls `sync_playwright`** —
  every capture path is injected as `capture_fn=fake_capture`.
- `judge.py:14` imports `OpenAI`, but every test injects a fake client. No test reads
  `OPENAI_API_KEY` or makes a network call.

MuJoCo is the only hard external, and only for grasp/pickup intents.

So a fast, hermetic CI job is straightforward — the suite is already hermetic, it just is
not organized.

---

## 2. Goals and non-goals

**Goals**

- G1. A `fast` tier under 30 s for eval iteration. The full 79 s suite stays the default.
- G2. Explicit tiers: fast / medium / slow.
- G3. Shared fixtures so a corpus clip is compiled once per session.
- G4. CI on both macOS and Windows.
- G5. Coverage measured on a schedule, not on every run.

**Non-goals**

- Rewriting tests. Tiering is markers plus a `conftest.py`, not a test rewrite.
- Parallelism via `pytest-xdist`. Revisit only if §3.1 and §3.2 are not enough; a 90 s
  suite does not need it.

---

## 3. Design

### 3.1 Keep coverage off the hot path

No change to the default invocation is needed — it is already coverage-free. What this
plan adds is:

- coverage runs **nightly only**, never on a local invocation or a PR job;
- `COVERAGE_CORE=sysmon` on that nightly job, to avoid the 3.95× tracing cost;
- document the intended invocations in `README.md` so nobody adds `--cov` to `addopts`.

### 3.2 A `conftest.py` with real fixtures

```python
@pytest.fixture(scope="session")
def rig() -> RigKinematics: ...

@pytest.fixture(scope="session")
def corpus() -> Corpus: ...                  # from PR 03

@pytest.fixture(scope="session")
def compiled(corpus) -> dict[str, ClipResult]:
    """Compile every corpus case once per session."""
```

Then the ~250 tests that recompile common motions read from `compiled`. Expected: another
large cut in the two dominant files.

### 3.3 Markers

| Marker | Contents | Budget |
| --- | --- | --- |
| `fast` | pure units, contract, schema, analysis over the pre-compiled corpus | < 30 s |
| `medium` | live compiles, MuJoCo, flywheel with stubs | < 3 min |
| `slow` | render, model calls, calibration | unbounded, opt-in |

Default `addopts` selects `fast and medium`. `slow` requires `-m slow` and is never
implied.

Add a marker-completeness test: every test file declares a `pytestmark`, and an unmarked
test fails collection. Otherwise tiering rots within a month.

### 3.4 CI

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    strategy:
      matrix:
        os: [macos-latest, windows-latest]
    steps:
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev
      - run: uv run pytest -m "fast or medium"
      - run: npm --prefix frontend ci && npm --prefix frontend test
      - run: npm --prefix frontend run build
```

Windows is in the matrix from day one. It is the only mechanism that will catch the
platform divergence risks — MuJoCo determinism, path separators in
`evals/capture.py`'s `output_dir` handling, and the `atomic_write_json` `PermissionError`
retry ladder at `io_utils.py`, which exists precisely because Windows behaves differently.

A separate nightly job runs coverage and the `slow` tier.

### 3.5 Frontend

`npm test` is 23 tests in ~250 ms and `npm run build` type-checks everything. Both belong
in CI as-is; nothing to restructure.

---

## 4. Sequencing — three PRs

| PR | Contents | Effort | Value |
| --- | --- | --- | --- |
| **09a** | Document invocations; pin coverage to the nightly job with `sysmon` | ~1 hour | low — no speed emergency exists |
| **09b** | `conftest.py`, session fixtures, markers, marker-completeness test | ~1 day | high |
| **09c** | CI workflow, macOS + Windows matrix, nightly coverage | ~half day | high |

09b is the one that matters for eval iteration: session-scoped compiled corpus fixtures
plus markers give a sub-30 s `fast` tier. 09c is the one that matters for the split
macOS/Windows setup.

---

## 5. Test plan

- `test_markers_complete.py` — every test file declares a `pytestmark`.
- Timing assertion in CI: the `fast` tier fails if it exceeds 30 s, so the tier cannot rot
  quietly.
- CI itself is the test for 09c: deliberately break something platform-specific on a branch
  and confirm the Windows job catches it.

---

## 6. Risks and open decisions

### 6.1 Risk — session-scoped fixtures can leak state

`ClipResult` is a Pydantic model and tests may mutate it. A session fixture handing the
same object to 250 tests invites cross-test contamination. Mitigation: return deep copies
from `compiled`, or freeze the models. Cheap either way, but decide deliberately rather
than discovering it as flakiness.

### 6.2 Open — is Windows CI worth the minutes?

It roughly doubles CI cost. Given the split development setup and the known Windows-specific
retry logic in `io_utils.py`, **recommendation: yes, on `main` and on PRs touching
`src/`, `evals/`, or `config/`.** Documentation-only changes can skip it.

### 6.3 Note — CI will likely fail on first enable

Expect real cross-platform findings the first time Windows runs, particularly around path
handling in capture output directories and the MuJoCo hashes from
[03](03-golden-corpus.md) §6.1. Budget a day for that rather than treating it as a blocker
on merging the workflow.
