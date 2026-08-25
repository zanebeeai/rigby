# Golden corpus

Committed reference cases that every check runs against. Loading and compiling the
corpus needs **no API key, no server, and no browser** — that is the point of it.

Specification: [`docs/plans/03-golden-corpus.md`](../../docs/plans/03-golden-corpus.md).

## What a case is

```
cases/<case-id>/
  scene.json          # SceneManifest
  program.json        # MotionProgram, seed pinned to a literal
  overrides.json      # ParameterOverrides, only when non-empty
  expected.json       # what compiling the above produced when it was blessed
  clip.slim.json.gz   # the motion itself, for the unblessed-platform fallback
```

The corpus stores the **program** and recompiles at test time. The hash always comes
from that recompile, never from the stored clip, so the clip can never become a second
source of truth. That makes every corpus run a determinism regression test of the
compiler, and it makes a deliberate compiler change produce a reviewable diff instead
of a silent re-recording.

`manifest.json` carries the schema version, the compiler the corpus was blessed
against, one row per case, and a coverage table naming every member of five enums —
`Intent`, `BodyAction`, `ObjectAction`, `StrikeType`, `HandShape` — as either
**covered** by a case or **deferred** with a reason. A member in neither list fails
`tests/test_corpus_coverage.py`, so adding an enum member without a case is a build
failure, not an oversight. All five are currently fully covered with **zero
deferrals**.

## Known-bad cases

Six cases exist to **fail**. A corpus of only-valid clips cannot detect a check that
has stopped firing: every case passes before and after the check is deleted.

Each names the gate it must fail, in `must_fail`, using the `StructuralGate`
vocabulary in [`gates.py`](gates.py) — so the assertion survives a reworded failure
message, and a case that trips a *different* gate is a failure rather than a pass.
`gates.py` scores gates from the numeric metrics and is deliberately not a second
definition of validity: everything it calls `FAILED` must also appear in the
compiler's own `structural_failures`.

Three of the six need no overrides at all. They fail on the plain prompt.

## Commands

```bash
python -m evals.corpus list                     # cases and coverage
python -m evals.corpus verify                   # recompile and compare; non-zero on a move
python -m evals.corpus bless                    # same, but prints a diff. Writes nothing.
python -m evals.corpus bless --case ID --write  # apply the diff
python -m evals.corpus freeze --from-seed ID    # rebuild a case from its recorded prompt
python -m evals.corpus freeze --id ID --family gesture --prompt "..."
python -m evals.corpus freeze --id ID --family grasp --result results/000042-slug
```

`bless` is a dry run until `--write` is passed. A corpus that silently re-records its
own expectations tests nothing.

**When a hash moves, say why in the PR description.** A hash that moved without an
intended compiler change is a regression, not a re-bless.

## Cross-platform hashes

**The compiler is not bit-reproducible across instruction sets.** 03a assumed MuJoCo
was the only source of cross-platform nondeterminism and classified everything else
`portable`; Windows CI disproved it, failing 12 cases — none of them MuJoCo cases — on
last-few-ulp floating point. arm64 and x86-64 differ in libm `sin`/`cos`/`acos` and in
FMA contraction, and numpy dispatches to different SIMD kernels. `max_angular_jerk_rad_s3`
differed by 56 ulps, amplified because a third derivative divides by `dt` three times.

So **every case is `platform_dependent`**, and all three digests — `motion_sha256`,
`metrics_sha256`, `observables_sha256` — are platform-keyed maps. A platform is blessed
for all three or for none.

The key is `<sys.platform>-<machine>`, plus `|mujoco-<version>` **only** when compiling
the case actually enters the solver. Whether it does is observed by counting
`simulate_grasp` calls rather than inferred from the intent: four cases reach it without
being named `grab`. Keying non-solver cases on the MuJoCo version would invalidate every
hash on an unrelated dependency bump, and a routine re-bless is indistinguishable from a
real drift at review time.

On a platform with no entry, the committed `clip.slim.json.gz` is compared against a
fresh compile instead — every numeric leaf, with the maximum deviation and its JSON path
reported. **Within tolerance is a pass; over tolerance is a failure.** The tolerance is
`1e-3` and PROVISIONAL: it bounds cross-ISA drift, which cannot be measured on one
machine, so the comparison reports the number that would replace it.

`bless --write` **merges** into the existing maps, so whoever blesses second contributes
their platform without deleting the first developer's hashes.

## Where the cases came from

[`seed_cases.py`](seed_cases.py) records the prompt each case was planned from, and
`freeze --from-seed` rebuilds any case from it. `tests/test_corpus_cli.py` asserts
the rebuild is byte-identical, so that table is provenance rather than a comment.
