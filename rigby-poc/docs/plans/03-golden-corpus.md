# PR 03 — Golden corpus and reproducibility

Status: proposed, not started.
Scope: a committed, versioned set of reference clips that every check runs against, plus
the seed pinning that makes them reproducible.

Depends on: [02a — analysis layer skeleton](02-analysis-layer.md).
Blocks: [06 — mutation library](06-mutation-library.md),
[09 — CI and tiering](09-ci-and-tiering.md), [10 — eval redesign](10-eval-redesign.md).

---

## 1. Problem

Every eval today requires running the live pipeline: an API key, a server, a browser, and
minutes of wall clock. There is no committed clip data, so a check cannot be developed or
regression-tested without generating motion first. That makes eval iteration slow, costly,
and non-reproducible — and it means **the checks themselves have no regression tests**.

It is also why the existing release audit cannot pass on a clean clone:
`evals/autonomous_goal_audit.py:79` reads
`results/judge-calibration/005673-v4-pronation/run-02-truthful-joints`, a path from the
original author's machine. That data never shipped and cannot be reconstructed.

### 1.1 The good news: compilation is already deterministic

Verified empirically across three intents, in-process and in separate processes with
`PYTHONHASHSEED` set to two different values:

```
Throw up a hang-ten sign.        MATCH  a12f22143ae55a4d  (59 frames)
Pick up the block on the table.  MATCH  951e305d29ef539a  (98 frames)   ← MuJoCo path
Do a push-up.                    MATCH  80700e194557ff82  (125 frames)
```

GLB export is byte-identical too. `compile(scene, program)` is a pure function:
`program.seed` reaches the compiler but is **never consumed** — it is pure provenance
(`compiler.py:495`). There is no `random`, no `hash()`, no `os.urandom`, no time-dependent
branching, and no threading anywhere in `src/rigby_poc/`. The two sets that exist
(`physics.py:110, 184`) are used for membership tests only, never iterated into output.

**So no determinism work is needed in the compiler.** This PR is much smaller than it
would otherwise be.

### 1.2 The one nondeterminism, and it is upstream

`planner.py:4246` derives the program seed from the LLM response id:

```python
seed = sha256(response.id)[:4]      # new API call ⇒ new id ⇒ new seed ⇒ different motion
```

with `secrets.randbelow(2**31-1)` as fallback when the response has no id
(`planner.py:4252`). Downstream, `planner.py:1190` uses `random.Random(variation_seed)`
for gesture accents — deterministic given the seed.

So two runs of the same prompt produce different `motion_sha256` and different
`result_id`. The **decision procedure** is reproducible; the **motion** is not.

For a corpus this does not matter, because a corpus stores programs rather than prompts.
It matters a great deal for [10](10-eval-redesign.md), which needs to compare pipeline
behaviour across runs.

### 1.3 Sizes, measured

| Artifact | Raw | Gzipped |
| --- | --- | --- |
| `clip.json` (95 frames × 52 bones) | 1140 KB | 50 KB |
| slim clip, quaternions rounded to 1e-6 | 213 KB | **17 KB** |
| slim clip, rounded to 1e-4 | 196 KB | 13 KB |
| `program.json` | 6.8 KB | — |
| `scene.json` | 7.8 KB | — |

A 40-case corpus is therefore ~600 KB as programs, or ~700 KB as gzipped slim clips.
Both are comfortably committable.

---

## 2. Goals and non-goals

**Goals**

- G1. A committed corpus spanning every intent family, loadable with no server, browser,
  or key.
- G2. Recompilation from the corpus is byte-identical, and a compiler change that alters
  motion **fails loudly** rather than silently re-blessing itself.
- G3. Corpus cases carry pinned seeds, so they never depend on an LLM response id.
- G4. `autonomous_goal_audit` reads committed evidence, not a machine-local path.

**Non-goals**

- Making renders reproducible across GPUs. That is [05](05-capture-integrity.md), and
  the answer there is that pixel hashes are machine-local by design.
- Curating a *quality*-labelled set. The corpus is reference input, not ground truth about
  what looks good. Labels come from mutation in [06](06-mutation-library.md).

---

## 3. Design

### 3.1 Corpus case format

```
evals/corpus/
  manifest.json
  cases/
    gesture-hangten-right/
      scene.json
      program.json          # seed pinned to a literal
      expected.json         # motion_sha256, frame_count, duration_s, key metric digest
      clip.slim.json.gz     # optional, for slow paths — see §3.3
```

`manifest.json` carries `schema_version`, a `compiler_version` the corpus was blessed
against, and per-case `{id, intent, family, tags, expected_structural_valid}`.

### 3.2 Recompile-and-verify, not store-and-trust

The default is to **store the program and recompile at test time**, asserting
`motion_sha256` matches `expected.json`. This is strictly better than storing clips:

- it is 15 KB per case rather than 17 KB, and more importantly
- it makes every corpus run a **determinism regression test of the compiler**, and
- a deliberate compiler change produces a clean, reviewable diff of which cases moved,
  via `python -m evals.corpus bless --case <id>`.

Blessing must be explicit. A corpus that silently re-records its own expectations tests
nothing.

### 3.3 Where stored clips are still needed

Two cases warrant `clip.slim.json.gz`:

- **MuJoCo paths.** `Intent.GRAB` runs `simulate_grasp` inside compilation with an
  8-attempt retry ladder. It is deterministic per platform and version — `version("mujoco")`
  is already recorded at `physics.py:275` — but **cross-platform bit-equality is not
  guaranteed.** See §6.1; this is the cross-OS risk in this PR.
- **Speed.** Any case slower than ~200 ms to compile gets a stored clip so the fast tier
  stays fast.

### 3.4 Corpus coverage

Target ~40 cases, chosen for check coverage rather than prompt variety:

| Family | Cases | Purpose |
| --- | --- | --- |
| Gesture | 6 | hand shapes, shake, both hands |
| Strike | 4 | hook / jab / cross / uppercut path geometry |
| Composite | 4 | paired forearms, cycles |
| Grasp | 3 | MuJoCo contact, one per block size |
| Object interaction | 6 | throw, catch, push, roll, place, handoff |
| Full body | 12 | one per `BodyAction` with distinct support topology |
| Sequence | 3 | multi-step continuity |
| **Known-bad** | 4 | cases that *must* fail specific gates |

The known-bad cases are the important ones. A corpus of only-valid clips cannot detect a
check that has stopped firing.

### 3.5 Seed pinning

Corpus programs carry a literal `seed`, never one derived from a response id. Add
`evals/corpus/freeze.py` to take a live result, strip volatile provenance, pin the seed,
and emit a case directory.

Separately, and for [10](10-eval-redesign.md)'s benefit, add an optional
`RIGBY_PLANNER_SEED` override so a pipeline run can be made reproducible end to end when
the planner is stubbed or replayed.

### 3.6 Repointing the audit

`evals/autonomous_goal_audit.py:79` moves to `docs/evidence/`, which is committed.
Combined with the corpus this makes the audit runnable on a clean clone — currently
impossible.

---

## 4. Sequencing — two PRs

| PR | Contents | Effort |
| --- | --- | --- |
| **03a** | Case format, loader, `bless` CLI, `freeze.py`, 12 cases covering gesture/strike/composite/full-body | ~2 days |
| **03b** | Remaining ~28 cases incl. known-bad and MuJoCo; repoint `goal_audit`; per-platform hash handling | ~2 days |

---

## 5. Test plan

- `test_corpus_determinism.py` — every case recompiles to its recorded `motion_sha256`.
  This is the whole point of the corpus and should run in the fast tier.
- `test_corpus_coverage.py` — every `Intent` and every `BodyAction` appears in at least
  one case; fails when a new enum member is added without a case.
- `test_corpus_known_bad.py` — each known-bad case fails **the specific check it was
  built to fail**, and passes the others. Guards against a check that silently stops
  firing.
- `test_corpus_loads_offline.py` — loading and analysing the whole corpus performs no
  network I/O and starts no browser.

---

## 6. Risks and open decisions

### 6.1 Risk — MuJoCo hashes may differ on Windows

This is the cross-platform hazard, and it matters because development is split between
macOS and Windows. MuJoCo determinism is guaranteed per platform and version, not across
them. If the grab-path hash differs between the two machines, `test_corpus_determinism`
fails for one developer and passes for the other — the worst possible failure mode.

Options, in preference order:

1. **Store `expected.json` hashes per `(platform, mujoco_version)` key**, and assert only
   against the current platform's entry, skipping with a clear message when absent. Honest
   and non-blocking.
2. Store the clip for MuJoCo cases and compare metrics with a tolerance rather than a hash.
3. Exclude MuJoCo cases from the determinism gate entirely — loses real coverage.

**Recommendation: option 1, and verify on both machines before 03b lands.** Whoever runs
it second contributes their platform's hashes in the same PR.

### 6.2 Risk — corpus rot

A corpus that is expensive to re-bless gets stale, and a stale corpus gets ignored.
Mitigation: `bless` must be a single command with a readable diff, and the PR description
template should require stating why any hash moved.

### 6.3 Open — commit slim clips for everything?

Storing every case's clip (~700 KB total) would make the corpus usable even if the
compiler is mid-refactor and cannot compile. It costs repo size and creates two sources of
truth. **Recommendation: programs only, plus clips for MuJoCo and slow cases.** Revisit if
[02](02-analysis-layer.md) turns out to break compilation for long stretches.
