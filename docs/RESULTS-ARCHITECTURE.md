# The base repository going forward: one registry, one viewer, three lines of work

Written 11 September 2026, after the consolidation in
`INTEGRATION-2026-09.md`. This is the plan for how three people keep
producing demos on three branches and end up with one cumulative record
instead of three incompatible ones. It is grounded in a survey of every
viewer and result format the repository has ever had; the facts from that
survey are listed at the end, with file references, so the plan can be
checked against them.

## What each of us is actually building

The git history sorts cleanly into three lines, and the plan keeps them.

| line | who | tier | what it produces | how it is judged |
|---|---|---|---|---|
| Evaluation and the humanoid compiler | Tony | `humanoid` (and `core`) | the 47-case golden corpus, mutation families, calibration campaigns, the analysis layer, the grasp solver; demos are README GIFs rendered from result ids | corpus digests per platform; the 10f verdict that the VLM grader is not yet an instrument (MCC +0.054) |
| Embodiment and the VLA gripper | Angelo | `humanoid` today, its own package soon | closed-loop decision over super primitives; the torque-driven gripper on one wrist camera; recordings of pick-and-place and the unfinished cabinet | placed / lift / penetration per run; 12 of 13 placed, worst penetration 0.34 mm |
| Arbitrary robots and primitives | Zane | `any-robot` (and `core`) | ingest, measured morphology, the certified primitive library, authored environments, trials across the zoo and the bench robots; demo GIFs per robot per prompt | accepted / refused per trace with a named refusal; 78 of 241 accepted in the last sweep |

Three lines, three tiers of code, and until now three unrelated ways to
write down a result. The code split is right and stays. What was missing
is the layer above it.

## The problem, stated from the survey

1. **Every result directory is gitignored.** `humanoid/results`,
   `humanoid/artifacts-v2`, `any-robot/results`, `any-robot/docs/media`.
   Results could not merge through git because they were never in it. The
   manifest in `docs/results` is the archaeology of that policy: six of nine
   result sets exist on one machine only.
2. **Three frame representations that cannot convert into each other.**
   Humanoid clips are 52 named-bone local delta quaternions against a GLB
   rest pose in glTF Y-up. Any-robot playback is joint-space `qpos` rows
   replayed by forward kinematics over a MuJoCo tree in Z-up. The gripper
   recording is per-frame world-space link segments with no joints at all.
   Any "unified clip format" would be a fourth format nobody produces.
3. **Only one viewer opens from disk.** The any-robot studio is a single HTML
   file with a hand-rolled WebGL2 renderer, chosen precisely so nothing has
   to be served. Motion Studio, capture, review, comparison and the gripper
   page all need the FastAPI server, the GLB over HTTP and `/api/v1`.
4. **Provenance is recorded unevenly and nowhere completely.** The commit is
   recorded in exactly one file in the tree (`humanoid/docs/media/demo-manifest.json`).
   The author is recorded nowhere. Two files named `demo-manifest.json`
   carry different schemas.
5. **Addressing does not merge.** Humanoid results are a monotonic
   six-digit counter, so two branches produce the same `006450`. Any-robot
   traces are `robot--prompt`, so a re-run overwrites. Only content
   addressing survives a merge.

## The design

Two layers, deliberately separate.

### Layer 1: the registry (`demos/`), shipped today

- **One file per demo**, `demos/registry/<id>.json`, schema `rigby.demo/1`
  (`demos/schema/demo.v1.md`). The id is date, prompt slug and eight hex of
  the media digest: two people on two branches never write the same file,
  so `git merge` never conflicts on the registry. That is the whole reason
  it is files rather than one index.
- **Provenance is captured at creation, not remembered later.**
  `demo_tools.py add` writes `who` (git identity mapped to the GitHub
  handle), `source.commit`, `source.branch`, `source.dirty`, `produced_at`,
  and `source.how`, the command that made it. This is the "tag each demo
  with who prompted it" requirement, and it costs the author nothing.
- **Media travels with the entry** under `demos/media/<id>/`, or is
  referenced where it already lives. Eight megabytes per file, twenty-four
  per entry; anything bigger goes on a GitHub release with the URL in
  `notes`. The registry is the one place results are *not* ignored.
- **`kind` is a discriminator, not a wish.** `gif`, `video`, `image` play
  today. `humanoid-bones-v1`, `anyrobot-qpos-v1`, `gripper-links-v1` name
  the three real pose formats; an entry of one of those carries a `payload`
  in that tier's native format, untranslated. The viewer grows a player per
  kind (Layer 2) without the registry changing.
- **CI guards it.** The `demos` job validates every entry, checks every
  media digest, refuses orphans, and fails if `index.html` was not
  regenerated. Seconds, no dependencies, runs on every PR.

The 23 entries seeded from the ledger show the shape: 13 zoo clips by
Zane, 6 README demos by Zane and Tony, 5 gripper recordings by Angelo, each
with its command, its commit where the source kept one, and its number.

### Layer 2: the viewer (`demos/index.html`), phase 1 shipped, phase 2 specified

Phase 1, in this PR: a static page rendered from the registry. Filter by
who, tier, kind, embodiment; search prompt, commit, branch; every card
plays its media, shows its provenance and its command. No server, no
framework, opens from disk, publishes anywhere.

Phase 2, one adapter per pose kind, in this order:

1. **`anyrobot-qpos-v1`** first, because the player already exists:
   `any-robot/scripts/studio_viewer.js` does FK in the browser over the
   `scene` export from `export_viewer.py`, with mesh deduplication by
   content hash. Lift it into `demos/viewer/players/anyrobot.js`, loading
   the entry's `payload` instead of an inlined blob. The studio keeps using
   it; one copy, two callers.
2. **`gripper-links-v1`** second, because it is the simplest geometry:
   segments, fingers and a plate per frame, each flagged `simulated` or
   drawn-for-context. Angelo's `gripper-live.ts` renders it with three.js;
   the static player draws the same tagged union in the same WebGL2
   helper as (1). The `simulated` flag stays visible: it is the honesty of
   the recording.
3. **`humanoid-bones-v1`** last, because it needs the GLB and the bone map:
   `frontend/src/scene.ts` applies local delta quaternions onto the rest
   pose with the `hips` axis swap. The static player needs the GLB inline
   or beside the page (534 kB) and the `BONE_MAP`; the capture contract
   (`?result=&view=&time=`, `__RIGBY_SEEK__`, the asset SHA refusal) stays
   on Motion Studio, which the evals drive, and is not duplicated.

Each adapter is a function `(payload, canvas) -> player` with `seek(t)`,
`play()`, `pause()`. The page dispatches on `kind` and never inspects a
payload to guess what it is.

### What stays where it is

- Motion Studio, the capture page, the blinded review and comparison pages
  keep serving the evals. They are instruments, and `evals/capture.py`
  depends on their exact surface.
- The any-robot studio stays the working page for that tier; the registry
  entry for a demo points at the same viewer export.
- The humanoid archive, `artifacts-v2` and the trace stores stay ignored.
  The registry records the demos that matter, not every run; the manifest
  in `docs/results` records where the bulk lives.

## How the branches work now

- `main` is the only long-lived branch (`CONTRIBUTING.md`). Each of us cuts
  short branches from it: `eval/...` (Tony), `vla/...` or `gripper/...`
  (Angelo), `feat/primitives-...` (Zane). Names are free; the rule is one
  concern, rebased often, merged in days.
- A demo made on a branch is registered on that branch with `add`, and
  merges with the code that made it. Because the entry names the commit,
  the demo can be reproduced from the merged history later, on any
  machine, by anyone.
- Angelo's gripper work lands from `grasp/auto-lift-on-main` (PR 19). Once
  it is on `main`, the gripper becomes a fourth package or a subpackage
  of `humanoid` with its own `results/` ignore rule and its own `add`
  calls; the five recordings already in the registry keep their ids.
- Tony's README demos are registry entries from now on: `render_demo_gif`
  then `add --kind gif --how "..."`; `demo-manifest.json` becomes a view
  over the registry rather than a second record.
- Zane's `build_demos.py` writes its zoo manifest today; its next change is
  to call `add` per clip so the zoo demos register themselves with the
  role-normalized hash in `tags`.

## CI

- The billing block is the first fix, and only the repository owner can
  make it: every Actions job since 7 September was refused with "recent
  account payments have failed or your spending limit needs to be
  increased". Until that clears, nothing below runs.
- The `demos` job (added in this PR) is the gate for the registry: schema,
  digests, orphans, and the rendered page.
- Existing jobs stay: `core`, `any-robot`, `python` (humanoid, macOS and
  Windows), `frontend`, `windows-corpus-hashes` on demand, and the nightly.
- Next: the any-robot studio JavaScript is verified by nothing today. When
  the FK player is lifted into `demos/viewer/players/`, the `demos` job
  gains a headless smoke test that loads each pose payload in the registry
  through its adapter and asserts frame count and duration. That is the
  point at which a code change that breaks replay of anyone's demo turns a
  PR red.

## The survey facts this rests on

1. Frames: humanoid `ClipFrame` is `{time_s, bones{name -> {rotation xyzw, position}}, objects}` with 52 VRM-style names, rotations documented as `local_delta_quaternion_xyzw_relative_to_glb_rest` (`humanoid/config/rig_profiles/mesh2motion-human-vrm1.json`, `humanoid/src/rigby_poc/models.py:838-891`). Any-robot playback is `{nq, frames, times[], qpos[]}` resampled to at most 120 frames (`any-robot/src/rigby_general/viewer/track.py:23-57`) over a `scene` of bodies, joints, geoms and meshes (`viewer/scene.py:74-155`). The gripper clip is `gripper_clip_v1` with per-frame `links[]` of `segment`, `finger` and `plate`, each with a `simulated` flag (branch `grasp/auto-lift`, `rigby-poc/frontend/src/gripper-live.ts:18-21`).
2. Conventions: application glTF Y-up, quaternion xyzw; simulation MuJoCo Z-up with forward -Y, quaternion wxyz. The v1 frontend hardcodes a `[x, -z, y]` swap for `hips` alone (`humanoid/frontend/src/scene.ts:477`); the v2 contracts carry the convention as a field (`core/src/rigby_core/contracts.py:103-111, 153-156`).
3. Static capability: only the any-robot studio (`any-robot/scripts/build_studio.py:317-434`, `studio_viewer.js:1-11`). Motion Studio fetches `/api/v1/results/{id}` and the GLB (`humanoid/frontend/src/api.ts:72-78`, `scene.ts:15`); the gripper page live-tails `run.clip_url`; `rigby-poc/results/replay.html` needs a server and a CDN.
4. Provenance: commit only in `humanoid/docs/media/demo-manifest.json` (`compiler_commit`); planner and model in v1 `provenance.json`; prompt in `program.source_text`, `trace.prompt`, and the gripper `GripperRun.task`; author nowhere; content hash in v2 `ArtifactRefV1` and any-robot `content_sha256`.
5. Addressing: humanoid `NNNNNN-slug` with `index.json` and a `_next_sequence` repair path (`humanoid/src/rigby_poc/store.py:26-82`); any-robot `robot_id--prompt-slug` with `created_at` preserved and `ran_at` moving on overwrite (`any-robot/src/rigby_general/trace.py:93-96, 200-220`); v2 content-addressed `objects/sha256/aa/...` (`core/src/rigby_core/artifacts.py:45-50`).
6. Ignore rules: `.gitignore:13-36` covers every viewer's data directory.
7. Capture is a contract: `capture.html?result=&view=&time=[&render=deterministic]`, `window.__RIGBY_SEEK__`, and a refusal when the GLB SHA disagrees with `humanoid/assets/manifest.json` (`humanoid/evals/capture.py:1342-1344, 1653-1656`).
8. Size has a history: the any-robot studio reached 161 MB before mesh deduplication brought it to 35 MB (`build_studio.py:467-485`); a humanoid `clip.json` is about 1 MB for 98 frames.
9. Frontend tooling: `humanoid/frontend` is Vite 7, three.js 0.179, vitest; `any-robot` has no Node tooling at all and its viewer JS is tested by nothing.
10. CI today: `select-platforms`, `python` (humanoid, macOS always and Windows on code changes, 60-minute hang detector, 90-second fast-tier ceiling), `frontend`, `core`, `any-robot`, `windows-corpus-hashes` on dispatch, plus the nightly slow tier.
