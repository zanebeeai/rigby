# Golden corpus

Committed reference cases that every check runs against. Loading and compiling the
corpus needs **no API key, no server, and no browser** — that is the point of it.

Specification: [`docs/plans/03-golden-corpus.md`](../../docs/plans/03-golden-corpus.md).

## What a case is

```
cases/<case-id>/
  scene.json       # SceneManifest
  program.json     # MotionProgram, seed pinned to a literal
  overrides.json   # ParameterOverrides, only when non-empty
  expected.json    # what compiling the above produced when it was blessed
```

The corpus stores the **program** and recompiles at test time. It does not store the
clip. That makes every corpus run a determinism regression test of the compiler, and
it makes a deliberate compiler change produce a reviewable diff instead of a silent
re-recording.

`manifest.json` carries the schema version, the compiler the corpus was blessed
against, one row per case, and a coverage table naming every `Intent` and
`BodyAction` as either **covered** by a case or **deferred** with a reason. A member
in neither list fails `tests/test_corpus_coverage.py` — so adding an enum member
without a case is a build failure, not an oversight.

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

MuJoCo is deterministic per platform and version, not across them. `expected.json`
therefore keys `motion_sha256` by platform:

- `determinism_class: "portable"` → exactly one hash under the reserved key `"any"`.
  Every 03a case is portable; none of them touch the physics solver.
- `determinism_class: "platform_dependent"` → one hash per
  `"<sys.platform>-<machine>|mujoco-<version>"` key. A platform with no entry is
  **skipped with a message**, never failed, and `bless --write` contributes this
  platform's hash without deleting anybody else's.

## Where the cases came from

[`seed_cases.py`](seed_cases.py) records the prompt each case was planned from, and
`freeze --from-seed` rebuilds any case from it. `tests/test_corpus_cli.py` asserts
the rebuild is byte-identical, so that table is provenance rather than a comment.
