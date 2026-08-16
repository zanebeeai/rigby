# PR 05 — Capture integrity, provenance, and speed

Status: proposed, not started.
Scope: make rendered evidence trustworthy, attributable, and roughly 4× faster to produce.

Depends on: [01 — observability](01-observability-and-transcript.md) for the `tool` span
that will carry capture timing and subprocess output.
Blocks: [07 — judge harness](07-judge-harness.md) and every VLM grader in
[10 — eval redesign](10-eval-redesign.md), because a grader cannot be calibrated on
evidence that may silently depict the wrong body.

---

## 1. Problem

### 1.1 A failed asset load silently substitutes a different body

`frontend/src/scene.ts:422-457` loads the humanoid GLB. The `catch` at line 449 swaps in
a **procedurally generated capsule avatar** and the page still reports
`data-capture-status="ready"` (`capture.ts:134`).

Capture then succeeds, the manifest is written, the hashes are computed, and the VLM
scores a completely different body. Nothing in `evidence-manifest.json` records that the
real asset never loaded.

This is the highest-severity issue in the capture path: it produces confident,
well-formed, entirely invalid evidence. Any calibration run that hits it silently
poisons its own ground truth.

### 1.2 Evidence hashes are not reproducible across machines

`scene.ts:115` enables `antialias: true`; `scene.ts:118-119, 199` use `PCFSoftShadowMap`
with 2048² shadow maps. MSAA resolve and soft-shadow filtering are GPU- and
driver-dependent, so the per-snapshot `sha256` (`capture.py:1361`) differs across
machines for identical motion.

This matters twice over. It means the two of you cannot compare evidence hashes at all,
and it means a future calibration regression cannot be attributed to model drift versus
renderer drift.

Note the useful half: **pose selection *is* deterministic.** `frameAt`
(`frontend/src/motion.ts:149-157`) snaps to the last frame at or before the requested
time with no interpolation, and `rendered_time_s` is recorded. So the *pose* is
reproducible even when the *pixels* are not — which is exactly the distinction the eval
suite needs to encode.

### 1.3 No render provenance is collected at all

Nothing in the frontend reads `THREE.REVISION`, the WebGL vendor/renderer strings, or
the browser build. A grep for `REVISION|UNMASKED|getContext|userAgent` across
`frontend/src/*.ts` returns nothing.

### 1.4 A timeout path leaks Chrome processes

Per snapshot, `capture.py:1315-1340` allows 3 attempts, each with 45 s navigation +
45 s selector wait + 45 s screenshot. Worst case is therefore ~405 s for a single
snapshot against a **300 s subprocess budget** (`pipeline.py:46`).

When that fires, `subprocess.run` raises `subprocess.TimeoutExpired`, which
`capture_in_subprocess` does not catch — it escapes instead of becoming the
`RuntimeError` at `pipeline.py:50`, and the orphaned Chrome process is never reaped.

### 1.5 Capture reloads the entire page once per snapshot

`capture.py:1320` issues a fresh `page.goto` for **every** sample point × view: 30 full
page loads per candidate, each re-fetching the result JSON and re-parsing the 521 KB GLB.

Measured on run `20260816T042756-f2107423`: 6.3–7.2 s per candidate, of which ~5.2 s is
the snapshot loop (~0.17 s each) and only ~1.1–2.0 s is fixed startup.

### 1.6 Storage is unbounded

One prompt costs roughly **58 MiB**: 39 MiB in the run directory (98.4% PNGs) plus
~19 MiB in `ResultStore`, which persists a folder for **all 11 attempted candidates**,
not just the 5 captured. `clip.json` alone is 1.17 MiB per candidate. Nothing is ever
garbage collected.

---

## 2. Goals and non-goals

**Goals**

- G1. A capture either depicts the correct rig or fails loudly. No silent substitution.
- G2. Every manifest records what rendered it — browser build, GPU strings, three.js
  revision, asset hash.
- G3. Pose determinism is asserted and machine-independent; pixel determinism is
  explicitly *not* claimed, and the distinction is encoded in the artifacts.
- G4. Capture is roughly 4× faster via a seek hook rather than per-snapshot reloads.
- G5. Timeouts are internally consistent and never leak a browser process.
- G6. Run storage is bounded by an explicit policy.

**Non-goals**

- Changing which times are sampled. The phase-sampling table
  (`capture.py:843-1265`) encodes a lot of hard-won per-intent tuning — walk/run local
  phases at `capture.py:1202-1207`, dexterous collapse at `1051-1069` — and is out of
  scope. This PR changes *how* frames are produced, never *which*.
- Changing the camera contract, FOV, or resolution.
- Making pixels bit-identical across GPUs. See §6.2.

---

## 3. Design

### 3.1 Fail loudly on asset substitution

Delete the fallback avatar path at `scene.ts:449`. Replace with an explicit failure
state: set `data-capture-status="error"` plus `data-capture-error="<reason>"`, and have
the Python side treat any status other than `ready` as a hard failure.

The procedural avatar has legitimate uses in the interactive studio, so gate it: keep it
for `index.html`, forbid it in `capture.html`. A `strictAssets: true` option on
`RigbyScene` is the smallest expression of that.

Additionally, record the loaded asset's sha256 in the manifest and assert it matches
`assets/manifest.json`. That turns "wrong body" from an invisible failure into a
one-line diff.

### 3.2 Render provenance block

New top-level key in `evidence-manifest.json`, inserted at `capture.py:1386` immediately
after `capture_contract`:

```json
"render_provenance": {
  "browser_version": "151.0.7922.138",
  "browser_channel": "chrome",
  "user_agent": "…",
  "three_revision": "179",
  "webgl_vendor": "Google Inc. (Apple)",
  "webgl_renderer": "ANGLE (Apple, ANGLE Metal Renderer: Apple M-series, …)",
  "webgl_version": "WebGL 2.0 (OpenGL ES 3.0 Chromium)",
  "device_pixel_ratio": 1,
  "antialias": true,
  "shadow_map": "PCFSoft/2048",
  "asset_sha256": "c7c445f4…",
  "asset_id": "mesh2motion-human-male",
  "platform": "darwin"
}
```

Collection points, all already sitting on data they need:

| Value | Where |
| --- | --- |
| `browser_version`, `browser_channel` | `browser.version` on the Playwright object at `capture.py:1295` — more trustworthy than the UA string |
| `three_revision`, `webgl_*`, `antialias`, `shadow_map` | `captureContract()` at `scene.ts:244-253`, which already holds `this.renderer` |
| `user_agent` | `capture.ts:125-132`, alongside the existing `CaptureState` |
| `asset_sha256` | computed once Python-side from `assets/manifest.json` |

`WEBGL_debug_renderer_info` is gated in current Chrome and may return masked strings;
fall back to `gl.getParameter(gl.RENDERER)`, which is always available.

**Placement matters.** Put this at the manifest top level, *not* inside
`snapshot["camera"]` — `src/rigby_poc/judge.py:493-497` validates that dict's shape.
Extra top-level keys are safe: `_manifest` (`judge.py:459-507`) checks
`capture_contract` against a fixed six-key allowlist and otherwise only iterates
`snapshots`. `refresh_motion_diagnostics` (`capture.py:815-828`) round-trips the whole
document, so new keys survive.

### 3.3 Separate pose determinism from pixel determinism

Two distinct hashes per snapshot:

- `pose_sha256` — hash of the *bone rotations applied at that frame*, taken from the clip
  rather than the render. Machine-independent, and the thing determinism tests should
  assert.
- `sha256` — existing pixel hash. Keep it, but demote it in the artifacts to
  `pixel_sha256` and document it as machine-local.

This is the smallest change that makes cross-machine comparison meaningful, and it lets
the eval suite assert reproducibility without pretending GPUs agree.

### 3.4 Seek hook instead of page reload

Add `window.__RIGBY_SEEK__(time_s, view)` to `capture.ts` that re-applies a frame and
re-aims the camera on an already-loaded page, resolving once the render barrier is met.

`capture.py` then loads the page **once per candidate** and issues 30 seeks. Expected:
~6.5 s → ~1.5 s per candidate. As a bonus this fixes §1.1's blast radius — the GLB is
fetched exactly once, so a load failure is a single loud error rather than a
per-snapshot coin flip.

The render barrier moves from a page-level attribute to the promise returned by the seek
call, which is strictly more precise than today's single-`requestAnimationFrame` wait at
`scene.ts:239-241`.

### 3.5 Batch the browser across candidates

`capture_result_frames` (`capture.py:1274-1392`) currently opens and closes its own
browser. Split it:

```python
@contextmanager
def capture_session(base_url: str) -> Iterator[CaptureSession]: ...

class CaptureSession:
    def capture(self, result_id: str, output_dir: Path, *, views=("ego","orbit")) -> Path: ...
```

Then `pipeline.py` spawns one subprocess **per round** (5 candidates) rather than per
candidate. The saving is modest on its own — ~5–8 s per round — but it composes with
§3.4, and both callers already pass `capture_fn` as an injectable seam
(`flywheel.py:1850`, `pipeline.py:168`), so the signature change is contained.

Worth recording why the subprocess exists at all, since it looks removable and is not:
the flywheel runs on a daemon thread inside the FastAPI process (`pipeline.py:141-149`),
and Playwright's **sync** API refuses to run on a thread with a live asyncio event loop.
The subprocess is a workaround for that, not for isolation. Keep it.

### 3.6 Consistent timeout budget

Derive the subprocess budget from the per-snapshot budget rather than hardcoding 300 s:

```
per_snapshot_worst = attempts × (nav + selector + screenshot)
subprocess_budget  = fixed_overhead + snapshot_count × per_snapshot_worst × safety
```

Reduce per-attempt timeouts to 15 s once §3.4 removes the reload (a seek is
sub-second), catch `subprocess.TimeoutExpired` in `capture_in_subprocess`, convert it to
the same `RuntimeError` as other failures, and kill the process group so Chrome is
reaped.

### 3.7 Storage policy

| Item | Policy |
| --- | --- |
| Evidence PNG | lossless until the run is terminal, then WebP q88 via PR 01's compaction (215 KB → 23 KB, measured) |
| `ResultStore` entries for non-selected candidates | prune after the run is terminal, keeping `metrics.json` + `program.json` + `provenance.json` and dropping `clip.json`/`animation.glb` |
| `clip.json` | store gzipped; 1.17 MB → ~50 KB, measured |

Pruning non-selected candidates recovers ~15 MiB of the ~19 MiB `ResultStore` cost per
prompt, and gzipping clips recovers most of the rest.

---

## 4. File-by-file changes

| File | Change |
| --- | --- |
| `frontend/src/scene.ts:422-457` | remove silent fallback; `strictAssets` option; error status |
| `frontend/src/scene.ts:244-253` | extend `captureContract()` with renderer/GL/three provenance |
| `frontend/src/capture.ts:17-25, 125-132` | widen `CaptureState`; add `render_provenance`, UA |
| `frontend/src/capture.ts` | add `window.__RIGBY_SEEK__` |
| `evals/capture.py:1274-1392` | split into `capture_session` + `capture`; single page load; seek loop |
| `evals/capture.py:1295, 1325, 1351, 1386` | collect and emit `render_provenance` |
| `evals/capture.py:1361` | emit `pose_sha256` and `pixel_sha256` |
| `evals/capture.py:1315-1340` | reduce per-attempt timeouts |
| `src/rigby_poc/pipeline.py:21-54` | batch mode; catch `TimeoutExpired`; kill process group; capture stdout/stderr into the PR-01 span |
| `evals/prune_run.py` | new — storage policy from §3.7 |

---

## 5. Test plan

- `tests/test_capture_contract.py` (extend) — manifest carries `render_provenance` with
  all required keys; `capture_contract` shape unchanged so `judge.py:459-507` still
  parses.
- **`tests/test_capture_asset_integrity.py`** (new) — point the scene at a missing GLB
  and assert capture **fails**; assert no manifest is written. This is the direct guard
  against §1.1 and is the most valuable test in this PR.
- `frontend/src/scene.test.ts` (extend) — `strictAssets` throws rather than substituting.
- `tests/test_capture_determinism.py` (new) — two captures of the same result on the same
  machine produce identical `pose_sha256` for every snapshot; `rendered_time_s` matches
  exactly; `pixel_sha256` is *not* asserted equal across machines.
- `tests/test_capture_timeout.py` (new) — a stubbed hanging capture surfaces
  `RuntimeError`, and no child process survives.
- Timing regression: assert per-candidate capture is under 3 s on the golden corpus once
  §3.4 lands.

---

## 6. Risks and open decisions

### 6.1 Risk — removing the fallback avatar breaks the studio

The interactive UI may rely on it during asset load. Mitigation is the `strictAssets`
flag: capture pages forbid substitution, the studio keeps its graceful degradation.
Verify `index.html` still renders during a slow GLB fetch.

### 6.2 Open — disable antialiasing and soft shadows for capture?

Turning both off would make pixel hashes reproducible across GPUs, which would let the
mutation-based calibration in [06](06-mutation-library.md) diff renders directly rather
than going through the VLM. The cost is that evidence looks worse, and the judge has been
calibrated — such as it is — on antialiased renders.

Recommendation: **keep antialiasing on for judged evidence, and add a
`--deterministic-render` capture mode** (AA off, shadows hard, fixed light) used only by
mutation-diff tests. Two modes, each honest about what it is for. **Decide before
[06](06-mutation-library.md).**

### 6.3 Risk — the seek hook diverges from the reload path

If seek does not reset every piece of scene state that a fresh load would, captures will
subtly differ from historical evidence. Mitigation: land §3.4 behind a flag, capture the
same result both ways, and assert `pose_sha256` equality and a bounded pixel difference
before making seek the default.

### 6.4 Note — this PR invalidates existing evidence hashes

Adding `render_provenance` changes the manifest, and `pixel_sha256` renames a field.
Any archived calibration keyed to the old shape must be regenerated. Since
[10](10-eval-redesign.md) regenerates calibration from scratch anyway, do this PR
**before** any new calibration is frozen, not after.
