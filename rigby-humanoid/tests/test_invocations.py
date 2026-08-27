"""Guards on how the suite is invoked.

Plan 09 §3.1. Coverage is not on the hot path and must not drift onto it; the
suite is hermetic and must not acquire a secret. Both are cheap to protect and
expensive to notice the loss of.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _prose(path: Path) -> str:
    """Document text with whitespace collapsed, lowercased.

    Every substring check below has to run against this rather than the raw
    file. Markdown wraps, so a phrase that reads as one sentence is split by a
    newline in the source and a naive `in` check misses it -- which has now cost
    three separate guards in this repository a false failure. Collapse first,
    then match.
    """

    return " ".join(path.read_text(encoding="utf-8").split()).lower()
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
TESTING_DOC = PROJECT_ROOT / "docs" / "testing.md"


def _pytest_config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]


def _addopts() -> str:
    return _pytest_config().get("addopts", "")


@pytest.mark.parametrize("flag", ["--cov", "--cov-report", "--cov-append", "-p no:cacheprovider"])
def test_addopts_does_not_enable_coverage(flag: str) -> None:
    """Coverage costs 3.95x on this suite. It belongs to the nightly job.

    See docs/testing.md. If you need coverage locally, pass it on the command
    line with COVERAGE_CORE=sysmon rather than making everyone pay for it.
    """

    assert flag not in _addopts(), (
        f"`{flag}` was added to pyproject.toml addopts. Coverage must stay off the "
        f"default invocation -- see docs/testing.md."
    )


def test_the_default_invocation_selects_fast_and_medium_but_never_slow() -> None:
    """`slow` starts a real browser; a bare `pytest` must not."""

    addopts = _addopts()
    assert "-q" in addopts
    assert "fast or medium" in addopts, (
        "the default invocation must select the fast and medium tiers explicitly"
    )
    assert "slow" not in addopts.replace("fast or medium", ""), (
        "`slow` must be opt-in via -m slow, never implied by the default"
    )


def test_the_default_invocation_carries_nothing_else() -> None:
    """Anything added here is paid by every local run and every CI job."""

    allowed = {"-q", "-m", "'fast", "or", "medium'"}
    assert set(_addopts().split()) <= allowed, (
        f"addopts grew beyond the tier selection: {_addopts()!r}. Document the "
        "invocation in docs/testing.md instead of making everyone pay for it."
    )


def test_the_doc_states_that_the_exit_code_is_the_only_green_signal() -> None:
    """This suite prints no summary line, so `$?` is the only result.

    A green run's entire output is dots and `[100%]`. That makes a completed
    green run indistinguishable from a truncated one to anything reading the
    text, and the reflex on seeing no `=== N passed ===` is to re-run a suite
    that already passed. The rule has to be in the file, not in anyone's head.
    """

    text = TESTING_DOC.read_text(encoding="utf-8")
    assert "no summary line" in text.lower(), (
        "docs/testing.md must state that this pytest prints no summary line"
    )
    block = text.split("```bash", 1)[1].split("```", 1)[0]
    # The marker is nested inside a shell string inside a Python string, so the
    # backslash depth varies with the wrapper. Match the format specifier, and
    # assert the leading-newline REASON separately in prose -- that is the part a
    # later editor would drop.
    assert "printf" in block and "EXIT=%s" in block, (
        "the command must use printf, not echo: pytest's final progress line is "
        "unterminated, so echo appends onto it and the marker lands mid-line"
    )
    assert "leading newline" in _prose(TESTING_DOC), (
        "the doc must say why printf carries a leading newline"
    )
    # `setsid nohup` appears in prose explaining why NOT to use it, so scope the
    # negative check to the block a lane would copy.
    assert "os.setsid()" in block and "setsid nohup" not in block, (
        "the command must detach via Python's os.setsid, not the setsid BINARY, "
        "which does not exist on macOS -- the binary form runs zero tests and "
        "then reports 'still running' forever"
    )
    checks = text.split("Checking it needs", 1)[1].split("```", 2)[1]
    assert "grep -o 'EXIT=[0-9]*'" in checks, (
        "the completion check must be UNANCHORED: echo-appended markers land "
        "mid-line and an anchored grep misses them"
    )
    assert "pgrep -f" in checks and "bin/pytest" in checks, (
        "a marker check alone waits forever when the whole group is killed and "
        "the printf never runs; the liveness check is the second half"
    )
    assert "rigby-wt/" in checks and ".venv/bin/pytest" in checks, (
        "the liveness check must be scoped to one worktree. The bare "
        "`pgrep -f 'bin/pytest'` matches every lane, so while anyone is verifying "
        "it answers RUNNING for all of them and 'no marker AND not running = "
        "killed' can never be reached -- and it fails in the hiding direction, "
        "reporting 'still running' rather than 'killed'"
    )
    assert "track the pid you launched" in _prose(TESTING_DOC), (
        "scoping is necessary and not sufficient: a run from an exited session "
        "keeps matching its worktree's path for hours"
    )
    assert "/tmp/rigby-$(whoami)-$$-" in block, (
        "the log path must be unique per run: a shared path lets two concurrent "
        "runs interleave into a file that describes neither"
    )


def test_the_doc_carries_the_exit_code_taxonomy() -> None:
    """143 is not a failure. Three lanes hunted a log that had no result in it."""

    prose = _prose(TESTING_DOC)
    assert "no tests collected" in prose, "EXIT=5 must be documented"
    assert "killed by a signal" in prose, "EXIT>=128 must be documented"
    assert "neither pass nor fail" in prose, (
        "the doc must say a signal death is NO RESULT, not a failure"
    )


def test_the_doc_does_not_assert_an_underived_cause() -> None:
    """The retracted draft blamed nested pytest.main(); that was wrong.

    The real cause was two concurrent runs sharing a log path. Asserting an
    underived mechanism in this file would make the correction another instance
    of the pattern it describes.
    """

    prose = _prose(TESTING_DOC)
    assert "pytest.main()" not in prose, (
        "the nested-pytest log-contamination mechanism was retracted; the child "
        "output is captured and never reaches the parent log"
    )
    assert "cause not established" in prose, (
        "the 143 clause must state that no cause is established"
    )
    assert "without a command you can name that produced it" in prose, (
        "the warning against attaching an underived cause must stay -- four "
        "mechanisms were proposed for this and all four were retracted"
    )


def test_the_doc_does_not_tell_readers_to_grep_the_log_for_failures() -> None:
    """Exit code is the ONLY green signal; a text check makes it unsound.

    A log can contain output from more than one run, because two concurrent runs
    sharing a path interleave. So a `FAILED` line may belong to a process other
    than the one being judged. A later editor will be tempted to add
    belt-and-braces back; this is the brace on the brace.

    The scoping matters as much as the rule: it licenses ignoring a line traced
    to another run, and nothing else.
    """

    text = TESTING_DOC.read_text(encoding="utf-8")
    lowered = _prose(TESTING_DOC)
    assert "do not grep the log for" in lowered, (
        "docs/testing.md must state that the exit code is the only signal"
    )
    assert "an untraced failure is a failure" in lowered, (
        "the rule must be scoped: it licenses ignoring a FAILED line traced to a "
        "known-nesting file, not any FAILED line anywhere"
    )
    assert "test_semantic_orientation" in text, (
        "the doc must name a counter-example whose FAILED lines are always real"
    )
    assert "more than one run" in lowered, (
        "the doc must give the REAL reason -- a log can contain output from more "
        "than one run -- not the retracted nested-pytest mechanism"
    )


def test_the_documented_invocations_all_capture_the_exit_code() -> None:
    """Every pytest command in the invocation table shows `EXIT=$?`.

    A table row without it teaches the habit the section above warns against.
    """

    text = TESTING_DOC.read_text(encoding="utf-8")
    table = text.split("## The invocations", 1)[1].split("##", 1)[0]
    rows = [
        line
        for line in table.splitlines()
        if line.startswith("|") and "uv run pytest" in line
    ]
    assert rows, "the invocation table must contain pytest commands"
    missing = [row for row in rows if 'echo "EXIT=$?" >> ' not in row]
    assert not missing, (
        "these documented pytest invocations do not capture the exit code:\n"
        + "\n".join(f"  {row.strip()}" for row in missing)
    )


def test_the_doc_warns_that_a_pipe_masks_pytests_exit_code() -> None:
    """`pytest ... | tail` reports tail's status. Two lanes hit this today."""

    text = TESTING_DOC.read_text(encoding="utf-8")
    prose = _prose(TESTING_DOC)
    assert "exits with `tail`'s status" in prose or "it is `tail`'s" in prose, (
        "the doc must name the pipeline hazard concretely"
    )
    assert "not a pytest rule" in prose, (
        "the rule must be stated for ANY piped or chained command -- written as a "
        "pytest idiom it reads as being about pytest, and `git rebase | tail` "
        "masked a failed rebase that then ran a suite on an unrebased tree"
    )
    assert "git rebase" in prose, "the doc must carry the non-pytest instance"


def test_the_doc_says_fast_or_medium_is_the_bar() -> None:
    """`slow` is excluded by design; a green default run is a pass."""

    text = TESTING_DOC.read_text(encoding="utf-8")
    assert "is the bar" in text
    assert "-m ''" in text, "the doc must warn against widening the selection"


def test_testing_doc_exists_and_states_the_coverage_rule() -> None:
    assert TESTING_DOC.exists(), "docs/testing.md is the documented-invocations contract"
    text = TESTING_DOC.read_text(encoding="utf-8")
    assert "COVERAGE_CORE=sysmon" in text
    assert "Never add `--cov` to `addopts`" in text


def _tests_tree() -> list[tuple[Path, ast.AST]]:
    """Every test module, parsed.

    The hermeticity guards below assert that *nothing* in this list reads a
    provider key or launches a browser. An empty list satisfies that vacuously,
    so a broken scan would report a hermetic suite rather than a broken scan.
    The collection is the evidence. Rule from lane `judge`.
    """

    parsed = [
        (path, ast.parse(path.read_text(encoding="utf-8"), str(path)))
        for path in sorted((PROJECT_ROOT / "tests").rglob("*.py"))
    ]
    assert len(parsed) > 50, (
        f"only {len(parsed)} test modules parsed; the scan is broken, not the "
        f"suite. `no test reads an API key` is trivially true of no tests."
    )
    return parsed


def _is_dotted(node: ast.AST, *names: str) -> bool:
    """True if ``node`` is an attribute/name access spelling one of ``names``."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts)) in names


def test_the_measurement_axes_rule_is_documented() -> None:
    """The rule that produced this PR's own retraction must stay in the file.

    Two lanes shipped a timing assertion that passed for a reason unrelated to
    what it checked, an hour apart. The rule is short, it is the thing you need
    *before* writing the assertion, and a plan document is where it would go to
    die.
    """

    text = TESTING_DOC.read_text(encoding="utf-8")
    assert "which axis are you crossing" in text.lower()
    for form in (
        "Ratios cancel load but not platform",
        "Absolutes survive platform but not load",
        "Counts survive both",
    ):
        assert form in text, f"the {form!r} rule was dropped from docs/testing.md"


def test_no_test_reads_an_api_key() -> None:
    """The suite must stay runnable with no secrets, so CI needs none.

    Detects *reads* only. Several tests deliberately delete these variables to
    make an accidental provider call fail loudly; that is the opposite offence
    and must not trip this guard, so the check is over the parsed tree rather
    than over the text.
    """

    secrets = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    offenders: list[str] = []
    for path, tree in _tests_tree():
        for node in ast.walk(tree):
            read = False
            if isinstance(node, ast.Call) and _is_dotted(
                node.func, "os.getenv", "os.environ.get", "environ.get", "getenv"
            ):
                read = any(
                    isinstance(arg, ast.Constant) and arg.value in secrets
                    for arg in node.args
                )
            elif isinstance(node, ast.Subscript) and _is_dotted(
                node.value, "os.environ", "environ"
            ):
                read = isinstance(node.slice, ast.Constant) and node.slice.value in secrets
            if read:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"tests read a provider API key at {offenders}; the suite must stay hermetic "
        f"so CI needs no secrets -- see docs/testing.md"
    )


def test_no_test_launches_a_real_browser() -> None:
    """Every capture path is injected. A real browser launch in CI is a hang."""

    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
        for path, tree in _tests_tree()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_dotted(node.func, "sync_playwright")
    ]
    assert not offenders, (
        f"tests launch a real browser at {offenders}; inject capture_fn instead"
    )
