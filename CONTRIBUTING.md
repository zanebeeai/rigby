# Contributing to Rigby

This file exists because, by September 2026, the repository had three
collaborators on four unrelated bases: a rewritten history, a directory
rename, and a local `main` that tracked a deleted branch. Integrating any of
it took a day of archaeology. The rules below are the ones that would have
made that day unnecessary, and the walkthroughs are how the three of us
actually work now. Read the first two sections before touching anything;
the rest is reference.

## 1. The shape of the repository

```
core/        rigby_core     morphology-neutral foundation: contracts, hashing, artifact and job stores,
                            motion compiler, controller, and the demo registrar (rigby_core.demos)
humanoid/    rigby_poc      the humanoid product: planner, compiler, analysis, evals, judge, Motion Studio
             rigby_v2       the certified v2 runtime beneath it
             gripper, closed_loop   the torque-driven gripper and the VLM decision layer (PR 19)
any-robot/   rigby_general  arbitrary-URDF product: ingest, morphology, grounding, contact, primitives,
                            authored environments, the any-robot studio
demos/       the shared registry of demos (one JSON per demo), their media, and the static viewer
docs/        cross-tier documents: the results ledger, the architecture, this consolidation's records
.github/     ci.yml (every PR) and nightly.yml (coverage, the slow tier, weekly macOS and Windows)
```

The three tiers are one `uv` workspace with one `uv.lock` at the root.
`humanoid` and `any-robot` depend on `core`; nothing depends on `humanoid`.
`core/tests/test_core_is_neutral.py` enforces that `core` imports neither
product.

Old names you will meet in history and in older branches:

| before 27 Aug 2026 (commit 2d39970) | now |
|---|---|
| `rigby-poc/` (earlier `rigby-humanoid/`) | `humanoid/` |
| `rigby-poc-general/` | `any-robot/` |
| `rigby-poc/src/rigby_v2/{artifacts,contracts,errors,hashing,jobs,records,postgres_jobs}.py`, `rigby_v2/motion/`, `rigby_v2/simulation/controller.py` | `core/src/rigby_core/` |

Results, rendered media, downloaded robots and virtual environments are
ignored everywhere (`*/results/`, `any-robot/robots/`, `any-robot/docs/media/`,
`.venv/`). The two places results are *not* ignored are `demos/` (the
registry) and `docs/results/` (the record of everything produced before the
registry existed).

## 2. Branches, pull requests, commits

**`main` is the only long-lived branch.** Everything else is a short-lived
branch cut from `origin/main`, merged back through a pull request, and
deleted.

1. **Cut from `origin/main`, today.** `git fetch origin && git switch -c
   <type>/<topic> origin/main`. Never cut from another unmerged branch. If
   your work needs someone else's unmerged work, land theirs first, or agree
   on one shared branch and both push to it.
2. **One concern per branch, and a week at most before it is on `main`.**
   Land the bottom of a stack before building the top. A branch that is a
   hundred commits behind `main` is the problem this file exists to prevent.
3. **Rebase onto `origin/main` before opening the PR and whenever `main`
   moves under you** (`git fetch origin && git rebase origin/main`). Force-push
   your own branch freely; it is yours until it merges.
4. **Never rewrite `main`.** The trailer strip of 27 Aug was the last history
   rewrite; it is what put every collaborator on a different base.
5. **Delete merged branches**, local and remote. `gh pr merge --delete-branch`
   does the remote one.

Branch names: `feat/`, `fix/`, `chore/`, `docs/`, `eval/`, `test/`, then a
short topic. Anyone's branch, same scheme.

**Pull requests.** Open the PR against `main` as soon as the branch has a
shape, as a draft if it is not done. The description says what changed and
why, and how it was verified: which test command, which result ids, which
numbers, in a table. Merge with a merge commit titled `Merge PR N: <title>`,
keeping the branch commits; no squashing, the commit messages are the design
record. Review is a read, not a gate: the author merges once CI is green and
comments are answered.

**Commits.** `type(scope): what changed, in one line`, then a blank line,
then why. The body carries the reasoning; the diff already says what. Types
are the branch prefixes plus `perf` and `refactor`; scope is a tier or a
package inside it (`any-robot`, `compiler`, `grasp`, `studio`, `ci`). No
attribution trailers of any kind.

**Line endings.** `.gitattributes` says `* -text`: git never converts a byte
in this repository, because content hashes are committed beside the content
they hash. Write files as LF. Python's `write_text` uses the platform newline
on Windows, so pass `newline="\n"`; the registrar and every generator in the
tree already do. `git ls-files --eol | grep crlf` finds a slip before it is
pushed.

## 3. Setting up a machine

Python 3.12, [`uv`](https://docs.astral.sh/uv/), Node 20.19+, npm, Chrome
(for the capture and demo renderers), git 2.40 or newer. On Windows also run
`git config --global core.longpaths true` once: the corpus fixtures exceed 260
characters.

From the repository root:

```bash
uv sync --all-packages --all-extras     # one environment for all three tiers
cd humanoid/frontend && npm ci && npm run build && cd ../..
```

Put `OPENAI_API_KEY=...` in a `.env` at the repository root (it is ignored).
The offline planners work without it; live planning, the VLM judge and the
gripper's decision layer need it.

## 4. Running the two studios

**Motion Studio (humanoid).** `cd humanoid && uv run rigby-humanoid`, then
open http://127.0.0.1:8000. Type a prompt, generate, scrub, export a GLB. The
results drawer replays anything in `humanoid/results/`. **Save as demo**
(beside Export GLB) registers the current result in `demos/` with its
`clip.json` as the payload and your name on it.

**Any-robot studio.** `cd any-robot && uv run rigby-general`, then open
http://127.0.0.1:8020. The page is `any-robot/results/studio.html`; build it
once with `uv run python scripts/build_studio.py` if it is missing (this runs
every demo prompt on every robot and takes a while). Submit a prompt against
any ingested robot or upload a URDF; the run lands in the list with its
trace and plays in the world it ran in. **Save as demo** under the preview
registers it with its GIF.

**The demo viewer.** `demos/index.html` opens straight from disk and needs
nothing running. To serve it beside the studios, `uv run python -m
http.server 8765` at the repository root and open
http://127.0.0.1:8765/demos/index.html. The viewer's sidebar links to both
studios for making new demos.

## 5. Making, registering and sharing a demo

A demo is a prompt, a body, a result, and who asked. Register one either
from a studio button, or from the command line:

```bash
uv run python demos/tools/demo_tools.py add \
  --title "so101: pick up the block in the desk bench" \
  --prompt "pick up the block" \
  --tier any-robot --embodiment so101 --kind gif \
  --how "cd any-robot && uv run python scripts/run_trials.py --environment desk_bench" \
  --outcome-state ok --outcome-text "lifted 6.2 cm, penetration 0.4 mm" \
  any-robot/results/so101--pick-up-the-block/clip.gif
uv run python demos/tools/demo_tools.py build
git add demos && git commit -m "demo(any-robot): so101 picks up the block in the desk bench"
```

`add` fills in who you are (git identity, mapped to your GitHub handle),
the commit, the branch, whether the tree was dirty, and copies the media in.
The id is `<date>-<prompt slug>-<8 hex of the media digest>`, so two people
demonstrating the same prompt on the same day still get two files and two
branches never conflict. `build` regenerates `index.html`; commit it with the
entry, CI checks the two agree. `validate` runs what CI runs.

Limits: 8 MB per file, 24 MB per entry, media types gif/webm/mp4/png/jpg. A
longer recording goes on a GitHub release with its URL in `notes` and a still
frame in `media`. Schema: `demos/schema/demo.v1.md`. Design and the plan for
pose playback: `docs/RESULTS-ARCHITECTURE.md`.

Humanoid demos: `cd humanoid && uv run python -m evals.render_demo_gif
<result_id> docs/media/<name>.gif` renders the ego and orbit GIF against a
running Motion Studio; then `add` it with `--kind gif`. README demos are
registry entries too.

## 6. Tests, and what CI runs

From each tier's directory, `uv run pytest`. The humanoid suite selects the
`fast or medium` tiers by default (`-m slow` is the nightly's job: render,
browser, model calls, calibration). `cd humanoid/frontend && npm test && npm
run build` for the frontend. The suites are hermetic: no server, no browser,
no key, no network. Keep them that way; `humanoid/docs/testing.md` says how
that is checked.

`ci.yml` runs on every push to `main` and every PR: `core`, `any-robot`,
`humanoid` (Linux), the frontend build and tests, and the `demos` registry
check. Docs-only PRs skip the humanoid suite. `nightly.yml` runs coverage and
the slow tier on Linux every night, the Postgres-gated tests in a service
container, and the humanoid suite on **macOS and Windows on Sundays** or
whenever the workflow is dispatched; `ci.yml` can also be dispatched with
`cross_platform` ticked. Linux by default is deliberate: GitHub bills macOS
at 10x and Windows at 2x, and running both on every push exhausted the
account's included minutes on 7 September, after which every job was refused
for four days.

If a check shows a billing refusal (jobs with no steps and no logs, an
annotation about payments or spending limits), the fix is in the account's
Billing settings, not in the code; verify locally and say so in the PR.

**Known Windows-only differences** (present on `main` before and after the
September merges, not regressions): `object-throw-far` and
`knownbad-sequence-throw-then-catch` drift past the corpus tolerance because
`win32-amd64` is not a blessed platform; `placing object` in
`test_prompt_family_matrix` reports discontinuities; `test_core_hashing`'s
subprocess uses a Unix `PATH`. `test_corpus_loads_offline` errors when the
checkout path is long.

## 7. The corpus, blessing, and behaviours behind flags

`humanoid/evals/corpus/cases/*` is the golden corpus: 47 programs whose
compiled motion is digested per platform in `expected.json`. Any change that
moves a blessed clip, however small, fails `test_corpus_determinism` until
the corpus is re-blessed on every platform: `uv run python -m evals.corpus
bless --write` on that platform, `python -m evals.bless_diff` to see which
digests moved, and the `windows-corpus-hashes` workflow (dispatch it from
the Actions tab) for the Windows column. A re-bless is a PR of its own, with
the motion change that caused it, so a reviewer can look at the new clips.

Because of that, four behaviours from the closed-loop branch are on `main`
but switched off, each waiting for its own re-bless PR:

| behaviour | where | how to turn on |
|---|---|---|
| angular rate limiter (bounds joint speed after authoring, not strikes) | `rigby_poc.motion_limits`, `features.py` | `RIGBY_ANGULAR_RATE_LIMIT=1` |
| grab gaze (aims neck and head at the object each frame) | `rigby_poc.gaze_controller`, `features.py` | `RIGBY_GRAB_GAZE=1` |
| per-digit full-authority curls and thumb opposition | `rigby_poc.primitives` (see the note at `curls=`) | restore from PR 19's `effective_curls` |
| relaxed rest curl 0.18 instead of 0.04 | same | same |

The hand-rolled quaternion path in `kinematics.py` (a 24 s profile win) was
held back for the same reason. `docs/INTEGRATION-2026-09.md` records the
merge that made these decisions.

Three of `tests/test_closed_loop.py`'s tests are marked `xfail(strict=False)`:
they fail on the branch they came from and pass when the file runs alone.
They are Angelo's to settle and are kept visible rather than deleted.

## 8. Migrating a branch cut before the 27 Aug split

If your branch still has `rigby-poc/` in it, do this once:

```bash
git fetch origin
git rebase --onto 5579c9f 7a570d2            # drops the 22 rewritten commits; 5579c9f has 7a570d2's exact tree
git -c merge.directoryRenames=true merge origin/main   # follows rigby-poc -> humanoid etc.
```

The rebase applies without conflicts. The merge leaves only real conflicts in
files both sides changed.

## 9. Where things stand and who owns what

| line | who | lives in | judged by |
|---|---|---|---|
| evaluation and the humanoid compiler | Tony | `humanoid/evals`, `humanoid/src/rigby_poc/analysis`, the compiler | corpus digests, the calibration campaigns |
| embodiment, the gripper, the VLA decision layer | Angelo | `humanoid/src/rigby_poc/{gripper,closed_loop,...}` | placed / lift / penetration per run |
| arbitrary robots and primitives | Zane | `any-robot/` | accepted / refused per trace, with a named refusal |

The record of everything produced so far, with media, is
`docs/results/RESULTS-MANIFEST.md` and the ledger page beside it. The
design for one viewer over all of it is `docs/RESULTS-ARCHITECTURE.md`.

## 10. Things that bite

- **`uv run` says "trampoline failed to canonicalize"**: the `.venv` was
  created at a path that no longer exists (the checkout moved). `rm -rf .venv
  && uv sync --all-packages --all-extras`.
- **A whole file shows as changed**: line endings. See section 2.
- **`git worktree` on Windows fails with "Filename too long"**: `git -c
  core.longpaths=true worktree add ...`, or set it globally.
- **A test that passes alone fails in the suite**: it is order-dependent.
  Fix the shared state it leaks or mark it `xfail(strict=False)` with the
  reason and your name, never delete it.
- **Motion Studio shows "Connecting"**: the API is not running on :8000, or
  the frontend was not built (`npm run build` in `humanoid/frontend`).
- **The any-robot studio is a 404**: build `results/studio.html` once (section 4).
