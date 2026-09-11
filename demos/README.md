# demos

The shared record of what Rigby can do, kept where three people on three
branches can add to it without ever touching the same file.

- `registry/<id>.json` - one entry per demo: what was asked, on which body,
  who made it, on which commit and branch, with which command, and what came
  out. Schema in `schema/demo.v1.md`.
- `media/<id>/` - the GIF, WebM, MP4 or PNG that shows it. Entries may also
  point at media that already lives elsewhere in the repository.
- `index.html` - the viewer, a single static page generated from the
  registry. Open it from disk; no server, no build step, no dependency.
- `tools/demo_tools.py` - `add`, `validate`, `build`.

## Adding a demo

From the repository root, on your branch, with the media already rendered:

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

`add` fills in who you are, the commit, the branch and whether the tree was
dirty. The id is `<date>-<prompt slug>-<8 hex of the media digest>`, so two
people demonstrating the same prompt on the same day still get two files.
`build` regenerates `index.html`; commit it with the entry, CI checks that
the two agree.

Keep a file under 8 MB and an entry under 24 MB. A longer recording goes on a
GitHub release; put its URL in `notes` and a still frame in `media`.

## From the apps

Both frontends register a result without leaving the page:

- **Motion Studio** (`uv run rigby-humanoid`, http://127.0.0.1:8000): the
  **Save as demo** button beside Export GLB calls
  `POST /api/v1/results/{id}/demo`. The entry carries the result's
  `clip.json` as a `humanoid-bones-v1` payload, the prompt, the planner and
  seed, and the verdict.
- **any-robot studio** (`uv run rigby-general`, then the studio page): the
  **Save as demo** button under a run's preview calls
  `POST /api/v3/results/{trace_id}/demo`. The entry carries the run's
  `clip.gif`, the prompt, the robot and the accept/refuse outcome.

Both stamp who asked from the machine's git identity (or `who` in the
request body), the commit and branch the server is running from, and
whether the tree was dirty, then regenerate `index.html`. What they do not
do is commit: that stays a person's act, on their own branch.

## What CI checks

The `demos` job runs `validate` and `build --check`: every entry has the
required fields, its media exists and matches the recorded SHA-256 and size,
nothing under `media/` is orphaned, and the committed `index.html` is the one
the registry produces.

## Pose payloads

`kind` is `gif`, `video` or `image` today. The three pose kinds
(`humanoid-bones-v1`, `anyrobot-qpos-v1`, `gripper-links-v1`) are reserved
for entries that also carry a `payload`, so the viewer can grow a player per
kind without changing the registry. `docs/RESULTS-ARCHITECTURE.md` says what
each player must do.
