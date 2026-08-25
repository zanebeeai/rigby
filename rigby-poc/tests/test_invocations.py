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
