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

**Classify by input, never by duration.** A file that reads a compiled clip is
`medium` at 0.2 s; a file that reads one committed JSON is `fast` at 1.0 s.
Sorting by runtime is how a tier stops meaning anything -- the first time the
`fast` budget breaches, the tempting fix is to demote whatever is slowest, and
after two of those `fast` means "the ones that happen to be quick today".

`tests/test_markers_complete.py` fails if a collected file declares no tier, or
declares two. It keys on pytest's own `python_files` patterns rather than a
directory glob, because `tests/` also holds `capture_e2e_server.py` and
`corpus_offline_probe.py`, which pytest never collects and which must carry no
marker.

## Measured

| Tier | Tests |
| --- | --- |
| `fast` | 291 |
| `medium` | 1292 |
| `slow` | 11 (opt-in only) |
| default (`fast or medium`) | 1583 |

**There is no validated wall-clock number for the `fast` tier, and plan 09 §5's
30 s budget has never been measured on a controlled machine.** Timed five times
on a developer box while the other five worktrees were compiling, the same 291
tests took:

    25.6 s   33.5 s   43.6 s   44.9 s   42.6 s        (load average 5.8 - 7.2)

A 1.9 s change in tier membership moved the figure by 7.9 s in one direction and
19 s across the sample. That is not a measurement of the tier; it is a
measurement of how busy the machine was. The same exposure retired a 300 ms gate
in `analysis/`, which was stable to 1.08x in isolation and went red inside the
full suite.

So the CI step **reports** the duration on every run and fails only above a loose
**90 s rot ceiling**. The ceiling is not the budget — it is there to catch
something expensive entering the tier, and a compile or a corpus load costs
minutes rather than seconds, so it is caught with room to spare. The real budget
should be set from a few CI runs on a dedicated runner and not before.

Asserting 30 s today would have made `main` red for load rather than for rot,
which is worse than having no budget: a gate that fires for reasons unrelated to
what it guards gets disabled, and then the tier really can rot.

**When a real budget exists and it breaches, re-derive the budget — do not
re-sort the tier.** Classify by input. Three files were `medium` in the first
pass of this PR because a regex matched an imported function name and the tier
comment itself; moving them to `fast`, correctly, is what pushed the local figure
over 30 s. Fitting the budget by leaving them misclassified would have made
`fast` mean "the ones that fit" rather than "the ones that read nothing
expensive".

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

## Guards fail in two directions

A guard can be unable to fire, or it can fire correctly except when it matters.
The second is harder to find and this repository has produced both.

**An absence becoming a value.** A gate that is read, documented and structurally
incapable of failing: `semantic_match` documented as gating and never gating,
`max_foot_drift_m` compared against a hardcoded `0.0`, ten never-written
`_base_metrics` defaults published as measurements, an unsigned joint-limit
comparison blind to half its input domain. Greppable at the call site.

**An assertion that fails open.** The check is live but its scope is wider than
its meaning, so it passes when it should not — and it is only ever wrong in the
direction that silences it. `"-gt 30" in command` is satisfied by `-gt 300`, so a
tenfold loosening of the fast-tier budget passed silently. Not greppable.

**A guard that fires for the wrong reason.** The check is live, it fails on a
real measurement, and the measurement is of a quantity unrelated to what the
message claims. A stage-gap assertion read "work is happening outside every
stage" and failed at 22 ms on a loaded runner — but the timeline closes the
previous span and opens the next in one call, so escaped work has **nowhere to
escape to**. The guard could only ever fail on its own close-and-reopen
overhead. Both statements were true and unrelated.

That one is neither greppable nor findable by mutation, and it needs its own
question: **what would the failure mode look like in the data, and can the data
express it?** If the shape the guard exists to catch cannot occur in the
structure being measured, the guard is measuring something else.

The greppable half of the second kind is narrower than "empty collection", and
this is the form worth remembering:

> **An assertion over a collection the test itself did not construct.**
> If the collection is *evidence*, assert it is non-empty before believing what
> it contains.

`all(...)` over nothing is True and `set() <= known` is True, so a moved
directory or a renamed package turns the guard green rather than red. Iterating a
module constant is safe — it is non-empty by construction. Iterating an `rglob`
result, a fake client's recorded calls, or a scan of the source tree is not,
because the collection being empty is precisely the failure the guard exists to
detect.

**Two external answerers, and they are different recommendations.** Both are the
same move — let something you did not choose answer the question — but they catch
different classes, and only one of them is expensive:

| | catches | cost |
| --- | --- | --- |
| **a second platform** | assumptions true where you develop, false elsewhere | a CI matrix |
| **a second dataset** | assumptions true when you wrote them, false now | nearly free |

A stage-timing bound and a path-separator key were both invisible to review and
obvious on Windows. A capture snapshot bound was wrong on **every** platform
equally, so no amount of Windows would have found it — what found it was running
capture against a corpus its author did not own. "Run it on another platform" and
"run it on inputs you did not choose" are not the same advice at two price points.

The operational rider, and it is the part that does the work: **a second dataset
makes staleness findable, it does not make it found.** The corpus that could have
revealed that bound sat there for hours revealing nothing, because nothing asked.
The mechanical form is a **margin guard** — assert the constant covers the data
*and* by how much — so the answer arrives before the constant is outgrown rather
than when it is. Lanes `capture` and `anatomy`.

**How to find them: point the scan at a directory that does not exist and see
who stays green.** Run against this suite's guards, that found two —
`test_markers_complete` reported every file correctly tiered, and
`test_invocations` reported a hermetic suite, both from scanning nothing. Both
now assert their evidence is non-empty first.

The corollary, which costs the most: **a guard that fails in its own vocabulary
is worse than one that crashes.** A path-keyed acceptance guard reported
"unreviewed consumer of the accept flag: [all five]" on Windows — a plausible
finding, in the exact language of the thing it audits. A stack trace would have
been cheaper.

## Measurements: which axis are you crossing?

*Written by lane `capture`, from two failures an hour apart — their timing-ratio
assertion and this file's fast-tier budget. Recorded here rather than in a plan
because it is the rule both of us needed before writing the assertion, not after.*

A number that must hold in a test can fail for reasons unrelated to what it
checks. Two axes cause this in practice, and the two common forms of number fail
on opposite ones.

**Ratios cancel load but not platform.** Two quantities measured in the same
window under the same contention divide the contention out. Across machines they
do not, because numerator and denominator scale differently.
`stage_spans / run_duration` measured 99.8% of a 5392 ms run across 44 spans on
macOS, and 93.5% of an 8881 ms run across 43 spans on Windows CI, for identical
code, against a `>= 0.95` bound. The untimed process head was 52x larger there
(575 ms against 11 ms) while the compiled work was only 1.65x slower.

**Absolutes survive platform but not load.** `575 ms of untimed head` travels
between macOS and Windows as a fact you can read off a log and act on. It does
not survive being measured on a box with seven worktrees compiling: the same 291
tests spanned 25.6 s to 44.9 s here depending only on load average.

**Counts survive both.** A count of events on a deterministic code path is
invariant to how fast the machine is and to which machine it is. "The humanoid
GLB is fetched exactly once per result", "43 spans, one root, zero orphans", "two
attempt spans for two dispatches", "`Rotation` constructions do not scale with
frame count" — none of these move with load, platform, or contention, and each
one pins the property that makes the corresponding timing number meaningful in
the first place.

So, in order of preference for anything that must **assert**:

1. **A count or a structural relation**, where one exists. Prefer it always.
2. **An absolute**, when the axis being crossed is load and the machine is
   controlled.
3. **A ratio**, only when both halves are measured in one window on one machine.

And for anything that must **report** rather than assert, publish the absolute
alongside the ratio. `93.5%` is not diagnosable on its own; `575 ms of untimed
head, n = 43 spans` is.

**Timing is not banned — it is scoped.** The 90 s rot ceiling in CI *is* a
wall-clock assertion, and a legitimate one. The distinction is the size of the
effect against the size of the noise: a compile or a corpus load entering the
`fast` tier costs **minutes**, and the noise floor is tens of seconds, so the
signature is an order of magnitude clear of the variance. A 30 s budget on a
tier that measures 25.6-44.9 s is not. Use timing where the thing you are
catching is orders of magnitude larger than the thing you cannot control, and on
a machine you control; never for a margin.

The corollary that costs the most to learn: **a timing assertion that passes
today may be passing for a reason unrelated to what it checks.** The `>= 0.95` bound
missed by 1.5 points on Windows (0.935 against 0.950, n = 43 spans), which means
every green macOS run had been evidence about the head size rather than about the
bound.

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
