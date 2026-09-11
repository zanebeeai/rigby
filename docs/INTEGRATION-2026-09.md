# Integration state, 10-11 September 2026

What every branch and pull request was when the repository was consolidated,
what was done about each, and what each person has to do next. Written so
that nobody needs to redo the archaeology; `CONTRIBUTING.md` has the rules
that stop it happening again.

## The three things that had pulled everyone apart

1. **The 27 Aug history rewrite.** Commit trailers were stripped from `main`,
   so every commit from mid-August has two hashes: the original and the
   rewritten one. Their trees are identical. Anyone who branched before the
   rewrite carries the originals, and git sees their branch as 20 to 30
   commits "ahead" that are already on `main`.
2. **The 27 Aug tier split (commit 2d39970).** `rigby-poc/` became
   `humanoid/`, `rigby-poc-general/` became `any-robot/`, and the
   morphology-neutral part of `rigby_v2` became `core/`. Files modified on a
   pre-split branch merge cleanly (git follows the rename); files *added* on
   such a branch land in the old directory unless the merge runs with
   `merge.directoryRenames=true`.
3. **A local `main` that was not `main`.** The maintainer's checkout had
   `main` tracking the deleted branch `feat/environments-and-contact`, six
   commits from 28 Aug that never reached GitHub, and a further 1,700 lines
   uncommitted.

The pair of commits that undoes (1) is `7a570d2` (the original "Merge PR 10",
on every old branch) and `5579c9f` (its rewrite, on `main`). They have the
same tree, so `git rebase --onto 5579c9f 7a570d2` replays only the real work
and drops the duplicates without a single conflict.

## What was done

| item | before | after |
|---|---|---|
| local `main` | 6 unpushed commits + uncommitted work, tracking a deleted branch | those 7 commits merged as **PR 21**; `main` tracks `origin/main` |
| `feat/grasp-solver` (Tony) | merged as PR 20, remote branch left behind | remote branch deleted |
| `grasp/auto-lift` (Angelo, PR 19) | 120 commits ahead / 104 behind, 682 files "changed", GitHub says conflicting | rebased copy pushed as **`grasp/auto-lift-on-main`**: 99 commits, 0 duplicates, tree byte-identical to the original tip (`f00861d`, 10 Sep). Original branch untouched. |
| `fridge/wip`, `eval/joint-rate-limits-and-thumb-opposition` (Angelo) | look like separate branches | both are ancestors of `grasp/auto-lift`; nothing on them is not in PR 19 |
| `eval/embodiment-truthfulness` (Angelo, PR 11) | open, conflicting | its three commits are patch-identical to three commits inside PR 19; superseded |
| `feat/grasp-throw-place` (Tony) | 1 commit, no PR | merges cleanly onto `main`; needs a PR |
| local branches `backup/pre-trailer-strip`, `codex/*`, `feat/irl-robots` | 0 commits not already on `main` | deleted |
| PR 3 (`patch-1`, not a collaborator) | open | left open, not merged, per the maintainer |
| CI | every run since 7 Sep refused by GitHub billing | unchanged; needs the owner's Billing page, not code |
| results | scattered across three untracked directories and one branch | catalogued in `docs/results/RESULTS-MANIFEST.md`, media copied under `docs/results/media/` |

## Angelo: landing PR 19

The rebased branch is `origin/grasp/auto-lift-on-main`. Its tree is exactly
your tip, so nothing of yours is lost; only the 22 already-merged commits are
gone. From there:

```bash
git fetch origin
git switch -c grasp/auto-lift-on-main origin/grasp/auto-lift-on-main
git -c merge.directoryRenames=true merge origin/main
```

That merge was run as a trial and aborted. It leaves these conflicts and no
others:

| file (new path) | what collided | hunks |
|---|---|---|
| `humanoid/src/rigby_poc/compiler.py` | your grasp and joint-rate commits (d7766f5, 1db2b80, 3c96c6d, 12af4f5) against Tony's grasp solver line (45f093b, 4a1c340, 8e777f8, d50010f) | 6 |
| `humanoid/src/rigby_poc/physics.py` | your embodiment and contact commits (56638b1, 12af4f5, 844c30d, 756e7f7) against Tony's 45f093b (table as manifest object, contact audit) | 1 |
| `humanoid/frontend/src/scene.ts` | your scene wiring (56638b1, 5ac7385, 1db2b80, 11cbeee) against Tony's 45f093b | 1 |
| `humanoid/config/thresholds.v1.json` | your render-contact threshold (060ba02) against the root-drift, arm-calibration and grasp thresholds (66f0c85, 3d8c99b, 45f093b) | 1 |
| `humanoid/src/rigby_poc/primitives.py` | git could not decide which directory your edit belongs in; `git mv rigby-poc/src/rigby_poc/primitives.py humanoid/src/rigby_poc/primitives.py` then merge by hand | 1 |
| `humanoid/evals/corpus/cases/*/expected.json` (about 45) and `clip.slim.json.gz` (47, deleted on `main`) | you re-blessed the corpus on 26 Aug; Tony re-blessed it with legs-in-strikes and the dense evidence format on 28 Aug and dropped the slim clips | take `main`'s corpus, then re-bless the cases your physics change alters, with the corpus CLI |

Everything else in the 298 files you touched merges on its own. When it is
green locally (`uv run pytest` in `humanoid` and `any-robot`), push, and
either retarget PR 19 to the new branch or push the merged branch over
`grasp/auto-lift` (both are yours). Close PR 11 as contained in 19. Delete
`fridge/wip` and `eval/joint-rate-limits-and-thumb-opposition` once 19 lands.

If you have local commits on `grasp/auto-lift` newer than `f00861d`, cherry
pick them onto the new branch; they will apply, the trees match.

## Tony: `feat/grasp-throw-place`

One commit, `8b99785`, touches only `humanoid/src/rigby_poc/compiler.py` and
merges onto `main` without conflict. Open the PR. Note that Angelo's
`compiler.py` conflict above is partly against this commit, so landing yours
first gives him one conflict to resolve instead of two rounds.

## Zane: the checkout

Three untracked directories in the checkout are not stale source. Git moved
every source file at the split; what remains under `rigby-poc/`,
`rigby-generalized-urdf/` and `rigby-mjco-sim/` is ignored residue, and that
residue is where the results are:

| directory | holds | in git? |
|---|---|---|
| `rigby-poc/results` | 6,449 humanoid runs of 7-12 Aug, the acceptance runs, the goal audit | no |
| `rigby-poc/artifacts-v2` | the v2 release evidence and live-model canaries | no |
| `rigby-generalized-urdf/results`, `docs/media` | the 24-27 Aug zoo trials and the 13 demo GIFs | no |
| `rigby-mjco-sim` | the 12-18 Aug production workspace; 1,770 of its 1,903 Python files exist in no commit | no |

The manifest records all of it. Nothing was deleted. When you want the disk
back, archive each directory (zip, or a `results` release on GitHub) and
delete it; the manifest carries enough to know what the archive is.
`rigby-humanoid/`, `src/`, `scripts/` and `native/` at the root are empty
shells (only `__pycache__`) and can go now. The five `_tpr*` directories are
unreadable from this account; nothing here knows what they are.

PR 3 is a one-line README change from someone outside the project, with a
note in its body asking that it not be merged. It was left open and untouched;
closing it is one click.
