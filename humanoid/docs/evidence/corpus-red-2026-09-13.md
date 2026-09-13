# `main` is red on the corpus, and two different things are causing it

Measured 2026-09-13 on clean `main` at `cfbe16e`, darwin-arm64, macOS 27.0.0, Python 3.12.13,
numpy 2.5.1, scipy 1.18.0, mujoco 3.11.0, no `RIGBY_*` flags set.

```
uv run python -m evals.corpus verify   ->  1 matched, 46 moved, 0 unblessed here   (exit 1)
uv run pytest                          ->  2626 collected, 2493 passed,
                                           119 failed, 3 errored                   (exit 1)
```

> **Correction, 2026-09-13.** This document and PR 38 first reported "22 failed, 3 errored".
> That was wrong: the figure was counted from a run captured with `tail -25`, so it counted only
> the `FAILED` lines that survived truncation. The corpus figures above were measured directly
> and are unaffected.

The single match is `knownbad-eigenvalues-unsupported`, which compiles zero frames.

`evals/probes/corpus_snapshot.py` exists to separate the two causes, because
`evals.corpus verify` reports both as "moved" and the remedies are opposite.

| | cases |
|---|---|
| motion digest differs from committed | 46 of 47 |
| — identical to the committed clip at slim precision (below rounding) | 43 |
| — **genuinely different motion** | **3** |

---

## 1. The three real changes are a regression, and PR 24 caused it

| case | what changed |
|---|---|
| `object-place-gently` | max deviation **0.7837** at `.frames[70].bones.rightLowerArm.rotation.y` |
| `object-throw-far` | frames **132 → 131** |
| `knownbad-sequence-throw-then-catch` | frames **229 → 228** |

`object-place-gently` compiles today with **`discontinuities=4`** and still reports
`structural_valid=True`. The largest single-frame bone step is **76.54°** on `rightLowerArm`
against a **5.00°** median for the clip, in a burst across four consecutive frames (69-72,
t≈2.36-2.46 s), which is the release → recover region.

Attributed by checking out `humanoid/src/rigby_poc/compiler.py` from `8b99785^1` into an
otherwise-unchanged `main` worktree:

| `object-place-gently` | before PR 24 | at `main` |
|---|---|---|
| frames | 93 | 93 |
| `discontinuities` | **0** | **4** |
| worst single-frame bone step | **18.35°** | **76.54°** |

`object-throw-far` is unchanged in smoothness (0 discontinuities, worst step 13.66° both sides)
but its frame count moved 132 → 131.

`8b99785` / PR 24 "seat the hand for throw and place like grab" changed `compiler.py` by
**+79/−3 and touched no corpus file**. Its `seated` branch covers only
`_INTERACTION_PATH_PHASES` (reach, preshape, contact, close); `lift`, `move`, `release` and
`recover` still run the old pivot path, and `recover` targets `base.copy()` for `PLACE`. The arm
arrives at recover from the new seat solve and is sent to the old neutral target, which is where
the step appears. Diagnosis only — the fix belongs to whoever wrote the seat.

By `evals/corpus/README.md:62` — "A hash that moved without an intended compiler change is a
regression, not a re-bless" — this has been on `main` since 2026-09-11.

## 2. The other 43 are environmental, and blessing them here would be wrong

Their deviation from the committed slim clip is **exactly 0.000** at slim precision; the
full-precision quaternion deltas are ~8.6e-7. No motion source for gestures, strikes or
composites changed since the bless, so this cannot be code.

It is also not this tree. Checking out `8e777f8` — the last bless, whose own commit message
records "verify 47 matched" — into a clean worktree and running verify **on this machine** gives
**1 matched, 46 moved**. Same commit, same result.

Eliminated:

- **Dependency drift.** `uv.lock` and every `pyproject.toml` are untouched since `8e777f8`
  (`git log 8e777f8..main -- uv.lock` is empty). Installed versions match the pins exactly.
- **BLAS threading.** `gesture-fist-right` produces a byte-identical digest under
  `OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1` and under the default.
- **Nondeterminism here.** Two full `corpus_snapshot` passes in separate processes compare as
  "47 cases, none moved".

What is left is a difference below the resolution of the platform key. `darwin-arm64` does not
distinguish two Apple Silicon machines, and `|mujoco-<version>` is appended only for cases that
enter the solver. So two macOS checkouts with identical pinned dependencies cannot both be green,
and whichever blesses last makes the other red.

**Therefore this branch does not re-bless.** A bless here would overwrite the 2026-09-07
`darwin-arm64` column with this machine's digests and move the failure rather than remove it.
The options, none of which is this PR's to choose:

1. Re-bless on a machine that reproduces the 2026-09-07 column, after the §1 regression is fixed.
2. Widen the platform key so a mismatch means something. Two machines that genuinely differ would
   then read `UNBLESSED_PLATFORM`, and `compare_slim_clip`'s 1e-3 tolerance would pass 43 of these
   47 at deviation 0.000 while still failing the three in §1 — which is the answer we want.
3. Accept that `test_corpus_determinism` is a single-machine gate and say so in `CONTRIBUTING.md`.

## 3. The remaining suite failures

Of the 119 failures:

| count | test | cause |
|---|---|---|
| 67 | `test_analysis_equivalence` | committed fixtures derived from compiler metrics that have moved |
| 48 | `test_corpus_determinism` | the 46 moved digests |
| 2 | `test_corpus_cli` | the same |
| 1 | `test_prompt_family_matrix[placing object-…]` | the §1 regression, `assert 3 == 0` on discontinuities |
| 1 | `test_ci_workflow` | unrelated, present on `main` |
| 3 (errors) | `test_corpus_loads_offline` | fixture assertion at `:45` |

`test_published_rates` fails on a checkout whose gitignored `results/10f2-20260829/SUMMARY.md`
is present; it is not collected on a clean tree.

`CONTRIBUTING.md` §6 listed `object-throw-far`, `knownbad-sequence-throw-then-catch` and
`placing object` in `test_prompt_family_matrix` as "Known Windows-only differences … not
regressions". All three fail on macOS, and the third is a real defect. That note is corrected in
this PR.

## Reproducing

```bash
cd humanoid
uv run python -m evals.probes.corpus_snapshot before.json   # the table in §1 and §2
uv run python -m evals.probes.standing                      # runtime, balance, actuator groups
uv run python -m evals.probes.retarget_residual             # per-bone visual -> physical feasibility
uv run python -m evals.probes.frame_alignment               # axis constancy and the conjugation spike
```

`corpus_snapshot --compare before.json after.json` is the regression gate to use on this tree
while `verify` cannot be green: it asks whether anything moved against a snapshot taken on the
same machine, which is a question this tree can currently answer.
