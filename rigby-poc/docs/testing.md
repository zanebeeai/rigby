# Running the tests

The suite is hermetic. It needs no server, no browser, no API key and no network.
That is a property worth protecting, so it is stated here and enforced by
`tests/test_invocations.py`.

## Read the exit code. There is no summary line.

**This suite prints no `=== N passed ===` line.** Not on a full run, not on one
file, not on `--collect-only`. The entire output of a green run is a row of dots
and a `[100%]`. So a green run and a run whose tail you did not capture look
identical, and `tail` on the log tells you nothing.

**Copy the command, not the paragraph.**

```bash
LOG=/tmp/rigby-$(whoami)-$$-$(date +%H%M%S).log
python3 -c "
import os, subprocess, sys
if os.fork() == 0:
    os.setsid()
    with open(sys.argv[1], 'w') as f:
        subprocess.run(['sh','-c','uv run pytest -q --tb=line; printf \"\nEXIT=%s\n\" \"\$?\"'],
                       stdout=f, stderr=f)
    os._exit(0)" "$LOG"
```

Checking it needs **two** lines, not one:

```bash
grep -o 'EXIT=[0-9]*' "$LOG" | tail -1        # the result, if the run finished
pgrep -f "rigby-wt/$LANE/rigby-poc/.venv/bin/pytest" >/dev/null && echo RUNNING || echo NOT-RUNNING
```

**Scope the second line to your own worktree, and note which direction the
unscoped form fails in.** `pgrep -f 'bin/pytest'` matches every lane, so while
anyone is verifying it answers RUNNING for everyone — and "no marker AND not
running = killed" becomes unreachable. It never reports "killed"; it reports
"still running", which is indistinguishable from a slow suite and invites
waiting rather than re-running. That is a guard present, documented, and unable
to fire, sitting inside the check written to catch that class.

Scoping is necessary and not sufficient: **track the pid you launched.** A run
started by a session that has since exited keeps matching its worktree's path
for hours. And do not infer whose a pid is from whose session is alive — that
guess was made here and was wrong, and acting on it would have killed another
lane's verification. `ps -eo pid,ppid,lstart,command` gives the parent chain and
the start time; the wrapper you launched is at the top of yours.

The rule that follows, and it is the one to keep: **a scoped hit that is not
your pid is somebody else's run, not yours.** The scoped pattern answers "is
anything running in this worktree", never "is my run still going". Only the
`EXIT=` marker in a unique log is run-scoped; the `pgrep` is a tiebreaker for
the marker-absent case.

**And a run started before an edit is testing a tree that no longer exists.** One
lane had a full suite in flight when `ruff` rewrote three files it was
collecting; the mix of old and new is unknowable, so the honest answer is to
discard the run rather than read its exit code. Note the launch time and check
`git status` plus `find -newermt <launch time>` over the tracked tree before you
believe a green. A one-commit delta is exactly the size where "too small to
matter" is tempting.

**No marker AND not running = killed. No result. Re-run.** The marker only
appears if pytest died and the wrapper survived; if the whole process group goes,
the `printf` never runs and a one-line check waits forever.

Every piece is load-bearing. Simpler forms of this were tried and each was wrong:

- **`python3 -c` with `os.fork()`/`os.setsid()`, not the `setsid` binary.**
  The `setsid` **binary does not exist on macOS** and every lane here is on
  darwin-arm64. That form exits 127, writes `command not found` into the log,
  runs zero tests, and then the finished-check reports "still in flight" forever
  — indistinguishable from a long suite under contention, which is the exact
  state the check exists to disambiguate. Two lanes reproduced it, one with a
  31-byte log containing only the shell error. `os.setsid` is in the stdlib and
  works; `nohup … & disown` is also available.
- **A unique log path.** A shared `/tmp/t.log` is what actually caused this
  section's first draft to be wrong: two concurrent runs interleave into a file
  that describes neither, and an `EXIT=` line on a shared path may be another
  run's, so "has mine finished?" answers yes early.
- **`printf "\nEXIT=%s\n"`, not `echo "EXIT=$?"`.** pytest's final progress line
  is unterminated, so `echo` appends onto it: the file ends `........EXIT=143`
  and an anchored `grep '^EXIT='` returns nothing. It fails that way on killed
  runs specifically, which is the case it exists for. Hence the leading newline
  *and* the unanchored `grep -o` above.
- **Detachment is recommended, not established as necessary.** One lane reports
  kills surviving tool-level backgrounding; another cannot reproduce that and has
  backgrounded suites completing past 25 minutes. Detaching costs nothing.

## Reading the result

| Exit code | Meaning |
| --- | --- |
| `0` | green |
| `1` | real failures — read the log |
| `5` | **no tests collected** — e.g. a `-m` expression matching nothing |
| `>= 128` | killed by a signal (143 = SIGTERM). **No result**, neither pass nor fail |

`>= 128` is the one worth knowing. It is not "some failure": there is no result
at all, and reading the log sends you hunting through output with zero `FAILED`
lines in it.

**A wrong exit code does not merely mislead one run — it gets written down.** A
comment inside `test_corpus_compile_budget.py` asserted that pytest exits `0`
when `-m` deselects everything. It exits `5`. The claim came from a shell
pipeline reading `tail`'s status rather than pytest's, and someone had to
re-measure to find the real answer. A comment inside a guard is the last place
anyone re-derives, because it reads as the output of exactly the rigour it is
missing. Reported by lane `groundtruth`.

**Re-run.** If it recurs, split the suite into disjoint halves and require
`EXIT=0` from each — same tests, same markers, one extra process boundary, so it
does not lower the bar. Report it as "verified in two halves, N+M files, zero
overlap, both `EXIT=0`", not as a full green.

**Cause not established.** Ruled out: memory pressure (ample free memory, no
swap in use, no jetsam events); the 600 s foreground tool ceiling alone (a run
killed at about three minutes); `pipeline.py`'s `killpg` (scoped to a child
spawned with `start_new_session=True`, and it sends SIGKILL/137, not
SIGTERM/143). Full detachment does not fix it either.

**Do not attach an explanation to this line without a command you can name that
produced it.** Four mechanisms were proposed for this in one evening, by four
different people, and every one was retracted. The next editor deserves the
warning more than they deserve a theory.

## Two more rules

**Do not judge by the wrapper's reported status. This is not a pytest rule — it
is true of any command you pipe or chain.** In `a; b; c` the overall status is
`c`'s, and in `a | tail` it is `tail`'s. Two instances, twenty minutes apart:
`pytest > log; echo "EXIT=$?"; tail log` had a task runner report "completed
(exit code 0)" over a log saying `EXIT=143`; and `git rebase … | tail` masked a
rebase that failed on unstaged changes, after which the `&&` chain started a full
suite **on an unrebased tree**.

So it masks a signal death identically to a clean pass, and it masks a failed
rebase identically to a successful one. Redirect, then read the status out of the
file — for rebases and merges as much as for pytest.

**Do not grep the log for `FAILED`, `ERROR`, `F` or `E` — and do not use that as
licence to dismiss one.** The reason is the shared-log-path collision above: a
log can contain output from more than one run. Judge on the exit code.

That licenses ignoring a `FAILED` line only when it is **traced to another run**.
It licenses nothing else. `test_semantic_orientation.py` spawns no subprocess and
shares no log, so a `FAILED` line from it is real and must be investigated. An
untraced failure is a failure.

For general awareness rather than as a log hazard: eight test files spawn
`sys.executable` subprocesses — `test_corpus_determinism`,
`test_corpus_loads_offline`, `test_capture_timeout`, `test_analysis_imports`,
`test_camera_config`, `test_corpus_compile_budget`, `test_run_transcript`,
`test_mutation_determinism`. Their child output is captured
(`capture_output=True`) and does not reach the parent log.

## Run the repo-hygiene guards before the full suite

`test_markers_complete`, `test_invocations`, `test_module_reachability`,
`test_no_hardcoded_thresholds`, `test_no_orphan_thresholds`,
`test_published_rates` and `test_ci_workflow` take about 75 s together and fail
on exactly what a new PR breaks: an unmarked file, an undeclared
corpus-touching file, a new uncited threshold. The full suite is minutes to tens
of minutes under contention. Lane `anatomy` caught
`test_every_corpus_touching_file_is_declared` this way and saved a full run.
They are a filter, not a substitute — the full suite on a rebased tree is still
the gate.

## Windows-only path failures, and why there is no guard for them

Two of these have now reached CI, both invisible to every macOS run:
`str(Path)` returning backslashes defeated a path-keyed waiver in
`test_published_rates`, and `tmp_path / str({"judgeable": False})` raised
`NotADirectoryError` in `test_compacted_run_is_not_judgeable` because `:` is
reserved on Windows and accepted on POSIX.

A guard rejecting the Windows-reserved characters `<>:"/\|?*` in a `tmp_path`
subdirectory name was proposed for the second. **Measured before building it, on
`b958e41`: 15 sites build a `tmp_path` child from a non-literal and 14 are an
`int`, an f-string of an index or a view name, or an already-slugged id.** After
the one fix the guard's blast radius is zero. A guard for a class with one
known, already-closed member only ever fires on false positives, acquires an
allowlist, and then looks like coverage — so the shape is documented here
instead:

> **Never build a path component out of a `repr`.** `str()` of a dict, a tuple
> or a dataclass embeds `:`, `'` and spaces. Write the label out; it is also
> what you want to read in the failure.

The general form is worth keeping separate from the fix, because the two halves
have different prices. A second platform catches assumptions that are true where
you develop and false elsewhere, and it costs a CI matrix. A second dataset
catches assumptions that were true when you wrote them and are false now, and it
is nearly free. Neither substitutes for the other: the backslash bug was
behaviour, not data, and no amount of locally rewritten `platform_key` would
have surfaced it.

## Full suites serialise on the verify slot

Three lanes running full suites at once took the load average to 25 and a
79-second suite to 25 minutes. So take the slot before a full run, and release it
the instant the run ends -- green, red or killed:

```bash
cd "/Users/tonypan/Developer/03 - Startups/rigby-wt"
./lockq.sh enqueue slot <lane>                    # when READY, not when you start preparing
./lockq.sh acquire slot <lane> && <your run>      # && -- see below
./lockq.sh release slot <lane>                    # prints who is next; message them by name
```

**Chain with `&&`, or check the status.** `acquire` refuses if you are not the
head of the queue and **exits 1**, and a `;`-separated next command runs anyway,
without the lock. That is the general rule two sections up -- *in `a; b; c` the
overall status is `c`'s* -- and it is worth restating here because writing it
down did not prevent it: the author of that rule chained past a `NOT-ACQUIRED`
with `;` and wrote to shared state hours after landing it, and the conductor
skipped the acquire entirely the same afternoon. Prose in the interface did not
protect the person who wrote the prose. In a script, `set -e` and an explicit
`if` are the mechanism:

```bash
set -e
if ./lockq.sh acquire slot <lane>; then <your run>; else echo "not head"; exit 1; fi
```

Three lanes wrote to an unlocked 161 KB `TRACKING.md` within minutes of each
other that day and lost nothing. That is luck, not design, and the absence of a
lost update is the reason the next one will be a surprise.

If `acquire` refuses, another lane is verifying or is ahead of you: wait and
retry rather than start.
Break a slot whose `owner` stamp is more than 45 minutes old, and say so in your
broadcast. Order among waiters: infra, analysis, groundtruth, judge -- that is a
queue discipline, not a licence to preempt a held slot. **It is separate from the
merge lock and you hold at most one, never both.** A targeted run of a few files
is not a full suite and does not need the slot.

## No lane can `git checkout main`, and the obvious merge reports success anyway

The main checkout holds `main`, so every worktree is refused it:

```
$ git checkout main
fatal: 'main' is already used by worktree at '.../rigby'     # exit 128
```

That refusal is fine on its own. What is not fine is the next line. Written as
two statements rather than an `&&` chain, the checkout fails, you stay on your
own branch, and the merge merges that branch **into itself**:

```
$ git merge --no-ff eval/<pr>
Already up to date.                                          # exit 0
```

A plausible message and a zero exit for an operation that did nothing at all.
This is structural rather than anyone's slip -- **the obvious sequence silently
no-ops for every lane, every time, and reports success.** Same family as
`git rebase ... | tail` masking a failed rebase: a status that describes the
wrong command, arriving through `worktree` instead of through a pipe, and
failing toward green like every other member of that set.

Merge through a temp branch in your own worktree, and check the tree afterwards:

```bash
git checkout -q -B merge-<pr> origin/main     # -B, not -b: -b fails on your second merge
git merge --no-ff eval/<pr> -m "Merge PR <pr>: ..."

# Three checks. They catch different things and none of them is redundant.
git rev-parse -q --verify HEAD^2 >/dev/null || echo "NOT A MERGE COMMIT -- the no-op"
[ "$(git rev-parse HEAD^1)" = "$(git rev-parse origin/main)" ] || echo "NOT ON CURRENT MAIN"
[ "$(git rev-parse HEAD^{tree})" = "$(git rev-parse <verified-sha>^{tree})" ] || echo "NOT THE TREE YOU VERIFIED"

git push origin merge-<pr>:main
```

**Two weaker forms were proposed first and both have the same hole.** Measured on
a scratch repository against four scenarios, because reasoning about it produced
two confident wrong answers:

| check | no-op self-merge | merged onto stale main | merged wrong branch |
| --- | --- | --- | --- |
| `git diff --stat <verified> HEAD` empty | **passes** | **passes** | catches |
| `HEAD^{tree}` equals verified tree | **passes** | **passes** | catches |
| `HEAD^2` resolves | catches | passes | catches |
| `HEAD^1` equals `origin/main` | catches | catches | catches |

Both tree-comparisons pass the no-op **for the same reason the no-op is invisible
in the first place**: it leaves `HEAD` at your branch tip, so the tree you are
comparing against is the tree you are standing on. A comparison whose two sides
become the same object under the failure cannot detect that failure. And a merge
onto stale `main` also matches the verified tree whenever your branch was rebased
first, since the branch already carries main's content.

So the parent checks do the work and the tree check earns its place separately:
it is the only one that catches merging *something other than what you tested*.
`HEAD^2` for "did I merge at all", `HEAD^1` for "onto the right base",
`HEAD^{tree}` for "the thing I verified". Found by lane `judge`, refined by lane
`analysis`, and both of our proposed single checks were wrong.

## `git stash` is shared across every worktree -- do not use it here

`refs/stash` lives in the one `.git` all six worktrees share, so a stash from any
lane renumbers everyone else's `stash@{n}`. **`git stash pop` applies whatever is
at `{0}` right now, which may be another lane's WIP, and drops their entry.** One
lane read `stash@{0}` twice twenty minutes apart and got two different lanes'
work; another watched its own entry move `{1}` -> `{0}` inside an hour.

This matters because the anti-tautology step tells you to set your change aside
and put it back. Use a patch file instead -- no shared ref, no index ambiguity:

```bash
git diff > /tmp/rigby-<lane>-antitaut.patch
git checkout -- <the source files>          # run the new test, show it RED
git apply /tmp/rigby-<lane>-antitaut.patch
diff <(git diff) /tmp/rigby-<lane>-antitaut.patch    # verify the restore
```

**Verify the restore rather than assuming it.** `git apply` can succeed
partially, and a restored-but-unchecked tree is how a mutation reaches a commit.
Two lanes had independently arrived at the equivalent of this (copy aside,
`git checkout`, copy back) before it was written down.

**The git form silently does nothing for a file that is not tracked yet, and its
verification step passes.** Measured: `git diff` excludes untracked files, so a
new test file yields a **0-byte patch**; `git checkout -- <newfile>` errors with
"did not match any file(s) known to git"; and
`diff <(git diff) /tmp/...patch` then compares empty against empty and
**passes**. Nothing was set aside, nothing was restored, and the check said so
was fine -- which is this file's own fourth direction, a comparison whose two
sides become the same object under the failure it guards. Found in this recipe
within an hour of it landing, by an `infra` restore that quietly left a scan
pointed at a directory that did not exist.

So **copy by path and verify per file with `cmp`**, which works whether or not
git has heard of the file:

```bash
mkdir -p /tmp/rigby-<lane>-antitaut
cp <files> /tmp/rigby-<lane>-antitaut/            # tracked or not
[ -s /tmp/rigby-<lane>-antitaut/<basename> ] || { echo "backup empty"; exit 1; }
# ... break it, run the new test, show it RED ...
cp /tmp/rigby-<lane>-antitaut/<basenames> <paths>
cmp <path> /tmp/rigby-<lane>-antitaut/<basename>  # per file; silence is the pass
```

**Assert the backup is non-empty BEFORE mutating, or `cmp` certifies the
damage.** If the `cp` produced nothing, the "restore" copies an empty file over
your source and `cmp` then compares empty against empty and passes. Measured: a
20-byte source became 0 bytes and the check said the restore was correct. That
is the empty-versus-empty collapse one layer below the `git diff` one, inside
the fix for it -- and it is the same rule this file already gives for evidence
collections, *assert non-emptiness for a collection you did not construct*,
pointed at the backup instead of at a scan. Found by lane `judge`.

`cmp` per file still beats a diff-of-diffs, because there is no state in which
both sides are trivially equal *for a reason you did not create* -- but it is
only sound once the backup is known to exist. And if you use the git form,
`git status --porcelain` first: a `??` line means that file is not in your patch.

**Nothing tests the recipes in this file.** `test_invocations` guards what it
says about commands and `test_markers_complete` guards what it says about tiers,
but the procedures are prose that no test executes -- so a documented procedure
is a guard nobody runs. Both collapses above were found by *using* the recipe,
within an hour of it landing, in the document that describes that exact failure
family. A scratch-directory test that extracts these blocks and executes them
against a throwaway file is the missing guard; `test_invocations` already parses
this file, so the extraction half exists. Named by lane `judge`.

If you must recover from an existing stash: find it by sha with
`git stash list --format='%H %gs'` and `git stash apply <sha>`. **`drop` cannot
take a sha** -- it must name an index, so re-check the index in the same command:

```bash
[ "$(git rev-parse stash@{0})" = "$MINE" ] && git stash drop stash@{0}
```

And a stash entry is a **merge** commit, so `git show <sha>` emits a combined
`diff --cc` that `git apply` refuses. Extract with `git diff <sha>^1 <sha>` and
check it with `git apply --check` before relying on it.

## The invocations

Every command below is written with its exit-code check, deliberately.

| What you want | Command |
| --- | --- |
| Everything (the default) | `uv run pytest -q > /tmp/t.log 2>&1; echo "EXIT=$?" >> /tmp/t.log` |
| One file | `uv run pytest -q tests/test_full_body_motion.py > /tmp/t.log 2>&1; echo "EXIT=$?" >> /tmp/t.log` |
| One test | `uv run pytest -q tests/test_x.py::test_y > /tmp/t.log 2>&1; echo "EXIT=$?" >> /tmp/t.log` |
| The fast tier | `uv run pytest -q -m fast > /tmp/t.log 2>&1; echo "EXIT=$?" >> /tmp/t.log` |
| The slow tier (opt-in) | `uv run pytest -q -m slow > /tmp/t.log 2>&1; echo "EXIT=$?" >> /tmp/t.log` |
| Coverage (rarely — see below) | `COVERAGE_CORE=sysmon uv run pytest -q --cov=rigby_poc --cov=evals > /tmp/t.log 2>&1; echo "EXIT=$?" >> /tmp/t.log` |
| Frontend | `npm --prefix frontend test` |
| Frontend type-check | `npm --prefix frontend run build` |

**`fast or medium` is the bar.** That is what `addopts` selects, and it is what
CI runs. `slow` is excluded by design — it needs a real browser, real model calls
or calibration. A green `uv run pytest` **is a pass**; do not reach for `-m ''`
to chase a fuller suite, because that is not the suite this project runs.

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

**If your test compiles the corpus, declare it too.** Add it to
`tests/test_corpus_compile_budget.py` -- `COMPILE_BUDGET` with a measured
ceiling, or `UNBUDGETED` with the reason. Measure it with that file's own
counter rather than guessing; a guessed ceiling is the next defect, where
`groundtruth`'s 55 and `judge`'s `COMPILES=4 RAN=5` were exact.
`test_every_corpus_touching_file_is_declared` fails the suite otherwise, and it
has caught three lanes this way -- each costing a full cycle on a loaded
machine.

That requirement was already written down, in the tracker, correctly, before two
of those three hit it. **A trap recorded in the record and not in the interface
will keep firing.** TRACKING is the record; this file is what a lane actually
opens while writing a test. Put the operational half here.

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

## Guards fail in more directions than they are written for

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

**A guard structurally blind to its own failure case.** The check is live, it
compares two real things, and the failure it exists to catch is precisely the
condition that makes those two things equal. It cannot fire, not because it is
unreachable, but because its inputs collapse into agreement exactly when the
thing has gone wrong.

The instance that cost the most: a merge-verification step asserting the merged
tree was byte-identical to the tree that had been verified. The failure it was
meant to catch -- a merge that silently did nothing -- leaves `HEAD` at the
branch tip, so both sides of the comparison become the same object and the check
passes. It printed `IDENTICAL` on the no-op it was written to detect. Two lanes
then proposed variants of the same comparison, and a third believed both; four
confident wrong answers about a three-line check, settled in two minutes by a
scratch repository with four scenarios in it.

**The question to ask of any comparison: under the failure you are guarding
against, do the two sides stay different?** If the failure makes them the same,
you have written a tautology with extra steps. That is the same family as
`all(...)` over an empty collection and a scan pointed at a missing directory --
the guard agrees with itself when its subject has vanished -- but it is worth
naming separately, because those two are about *absent* inputs and this one has
two perfectly real inputs that happen to coincide. Found by lane `judge`, whose
own run printed the passing line.

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

## A number that moves is not evidence the thing under it moved

*Found by lane `groundtruth` scoring mutations, and independently by lane
`analysis` in a composite oracle, within one hour. Two lanes, two call sites, one
door — which is why it is here and not in either plan.*

**Ask of a derived value: is this source still tracking its subject, or did it
stop when the subject was transformed?** A stale source does not raise. It returns
a well-formed number of the right type, and the number is right about a clip that
no longer exists.

The instance: `MutationSpec.apply` deep-copies and transforms `frames` and nothing
in `evals/mutations` touches `metrics`, so a **mutated clip carries the compiler's
pre-mutation metrics**. Measured on four full-body corpus cases, a 40° injection
trips up to six structural gates in `analyze(...)` output while
`clip.metrics["structural_failures"]` stays byte-identical to the unmutated clip's
— empty, because that namespace's base rate is 0 on all 10 corpus cases that
evaluate it. So the rule is: read `clip.metrics` for compiler-written keys, and
`analyze()` output for anything scored against a transformed clip.

**The general form is worse than "read the fresh one", and it is what makes this
hard to see.** `validate()` splits its sources: `frames` for the ROM layer,
`metrics` for everything else. A consumer handed a transformed clip therefore gets
a live majority and a stale remainder: of the 176 ids `validate()` emits over the
47-case corpus, **156 are ROM and move with `frames`, while the other 20 are read
from the frozen `metrics`** — and the stale 20 are the ones carrying the
discriminating signal, since the ROM verdicts are near-uniform across cases. The
output moves, monotonically, with severity. It looks like it is working.
`validate_clip(clip, program)`, the convenience form, is exactly this shape.

Both failure directions land on green rather than on an error:

| what you read | what you get | how it reads |
| --- | --- | --- |
| stale metrics, whole surface | "undetected" at every severity | a namespace of dead gates |
| stale metrics, 4% of surface | a curve that rises with severity | a working detector |

Neither is an exception, an empty result, or a missing key — which is what
separates this from the not-measured/measured-negative family next door. There the
absence is *visible* once you look for it. Here the value is present, correct,
and about the wrong object.

**The cheap check is a differential, not a review.** Score the same transformed
input through both sources and assert they disagree. If they agree, either the
transform did nothing or you are reading one source twice — and both of those are
findings. A test that only asserts the fresh source is right will pass against the
stale one whenever the transform happens to change nothing, which is every case
the transform was inapplicable to.

**And assert the differential ran.** The corpus test pinning this skips any case
whose injection trips no gate; against the stale source that is *every* case, so
the loop body executes zero times and the test passes vacuously against the exact
defect it exists to catch. It is only red because it asserts its own sample is
non-empty first. Same rule as the `all(...)`-over-nothing family above, reached
from the other end: there the collection was empty because a scan found nothing,
here because the thing being measured stopped moving.

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
