"""Guards on the CI workflows.

Plan 09 §3.4 and §6.2. Two properties are worth failing the suite over.

The first is that CI carries no secrets. The suite is hermetic today -- it passes
with ``OPENAI_API_KEY`` unset and every internet socket raising -- and a workflow
that holds a key is how that quietly stops being true.

The second is that Windows stays in the matrix. It is the only mechanism that
catches the platform divergence this repository is exposed to: filesystem paths
in capture output directories, and the ``PermissionError`` retry ladder in
``io_utils.py`` that exists precisely because Windows behaves differently.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
NIGHTLY = WORKFLOWS / "nightly.yml"


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _all_workflows() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _jobs(path: Path) -> dict[str, Any]:
    return _load(path)["jobs"]


def _steps(path: Path, job: str) -> list[dict[str, Any]]:
    return _jobs(path)[job].get("steps", [])


def test_workflows_exist() -> None:
    assert CI.exists(), "the repository must have CI"
    assert NIGHTLY.exists(), "coverage must have a scheduled home (§3.1)"


@pytest.mark.parametrize("name", ["ci.yml", "nightly.yml"])
def test_workflow_parses(name: str) -> None:
    document = _load(WORKFLOWS / name)
    assert isinstance(document, dict) and document.get("jobs")


def test_no_workflow_uses_a_secret() -> None:
    """The hermeticity contract, enforced.

    ``GITHUB_TOKEN`` is exempt: it is minted per run, not configured, and this
    repository's workflows request only ``contents: read``.
    """

    offenders: list[str] = []
    for path in _all_workflows():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "secrets." in line and "secrets.GITHUB_TOKEN" not in line:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, (
        f"a workflow references a secret at {offenders}. The suite is hermetic and CI "
        f"must stay that way -- see humanoid/docs/testing.md."
    )


def test_workflows_request_read_only_permissions() -> None:
    for path in _all_workflows():
        permissions = _load(path).get("permissions")
        assert permissions == {"contents": "read"}, (
            f"{path.name} must declare `permissions: contents: read` explicitly"
        )


def test_windows_is_in_the_python_matrix() -> None:
    """§6.2: Windows on main and on any PR touching code."""

    text = CI.read_text(encoding="utf-8")
    assert "windows-latest" in text, "Windows must be in the matrix (§3.4)"
    assert "macos-latest" in text, "macOS must be in the matrix"


def test_docs_only_prs_still_run_macos() -> None:
    """Skipping Windows on a docs PR must not skip CI altogether."""

    text = CI.read_text(encoding="utf-8")
    assert 'os=["macos-latest"]' in text, (
        "the docs-only branch of the matrix must still run macOS"
    )
    assert 'os=["macos-latest","windows-latest"]' in text


def test_ci_does_not_run_coverage() -> None:
    """Coverage is 3.95x. It belongs to the nightly job, not to every PR."""

    for job in _jobs(CI):
        for step in _steps(CI, job):
            command = step.get("run", "")
            assert "--cov" not in command, (
                f"ci.yml job `{job}` runs coverage; move it to nightly.yml (§3.1)"
            )


def test_nightly_runs_coverage_with_sysmon() -> None:
    """Without sysmon the nightly job pays the full settrace multiplier."""

    job = _jobs(NIGHTLY)["coverage"]
    assert job.get("env", {}).get("COVERAGE_CORE") == "sysmon", (
        "the nightly coverage job must set COVERAGE_CORE=sysmon (§3.1)"
    )
    assert any("--cov" in step.get("run", "") for step in job["steps"]), (
        "the nightly job exists to measure coverage; it must actually pass --cov"
    )


def _triggers(path: Path) -> dict[str, Any]:
    """The `on:` block. YAML 1.1 parses the bare key `on` as the boolean True."""

    document = _load(path)
    return document.get("on", document.get(True))


def test_nightly_is_scheduled_and_ci_is_not() -> None:
    assert "schedule" in _triggers(NIGHTLY), "nightly must be on a cron schedule"
    assert "schedule" not in _triggers(CI), "CI runs on push and PR, not on a timer"
    assert "pull_request" in _triggers(CI), "CI must run on pull requests"
    assert "workflow_dispatch" in _triggers(CI), "CI must be manually runnable"


def test_main_pushes_are_never_cancelled_by_a_later_merge() -> None:
    """Found by the first run: 6c154e3 was cancelled by 5e8353a four minutes later.

    Six lanes merge every few minutes and the Windows job takes over ten, so a
    concurrency group keyed by ref with unconditional cancel-in-progress means
    `main` almost never completes a verification -- the exact opposite of what CI
    is for. A push must get its own group; only a PR may cancel its predecessor.
    """

    concurrency = _load(CI)["concurrency"]
    group = concurrency["group"]
    assert "github.sha" in group, (
        "a push to main must group by sha so a later merge cannot cancel it"
    )
    cancel = str(concurrency["cancel-in-progress"])
    assert cancel != "True", (
        "cancel-in-progress must be conditional on the event, not unconditional"
    )
    assert "pull_request" in cancel, (
        "only pull_request runs may cancel an in-progress run"
    )


def test_the_python_job_runs_pytest_exactly_once() -> None:
    """The duplication this replaced, pinned so it cannot come back.

    `addopts` is `-m 'fast or medium'`, so `pytest -m fast` is a strict subset of
    a bare `pytest`. CI ran both, which re-ran all 703 fast tests on every macOS
    job to produce one duration. Two invocations here is that mistake returning.
    """

    invocations = [
        step for step in _steps(CI, "python")
        if re.search(r"\bpytest\b", step.get("run", ""))
    ]
    assert len(invocations) == 1, (
        f"the python job runs pytest {len(invocations)} times: "
        f"{[step.get('name') or step['run'] for step in invocations]}. The tier "
        f"duration comes from the conftest hook now, not from a second run."
    )


def test_ci_measures_the_fast_tier_from_that_one_run() -> None:
    """The rot ceiling survives the de-duplication, and reports every run."""

    steps = _steps(CI, "python")
    producer = [step for step in steps if "RIGBY_TIER_REPORT" in str(step.get("env", {}))]
    assert producer, (
        "the pytest step must set RIGBY_TIER_REPORT so tests/conftest.py writes "
        "the tier timing; without it the ceiling below has nothing to read"
    )

    checks = [step for step in steps if "fast_tier_seconds" in step.get("run", "")]
    assert checks, "CI must still enforce a ceiling on the fast tier"
    command = checks[0]["run"]

    # Match the number, not a prefix of it: `"-gt 30" in "-gt 300"` is true, and
    # that substring check passed a budget loosened tenfold.
    budgets = [int(value) for value in re.findall(r"-gt\s+(\d+)", command)]
    assert budgets == [90], (
        f"the fast tier ceiling must be the recorded 90s rot ceiling, found {budgets}. "
        f"It is deliberately loose: the 30s budget in plan 09 §5 has never been "
        f"measured on a controlled machine. Tighten it from CI data, not from the plan."
    )
    assert "::notice::" in command, (
        "the tier duration must be reported every run, not only on failure -- the "
        "trend is what shows rot"
    )
    assert "exit 1" in command, "exceeding the budget must fail the job"
    assert checks[0].get("if"), (
        "the budget is a claim about a developer machine; it must not run on the "
        "Windows runner, which is roughly 4x slower for unrelated reasons"
    )


def test_the_tier_measurement_includes_collection() -> None:
    """The regression that actually fired must stay inside the measurement.

    A module-level corpus compile is paid at import, so `-m fast` pays it and
    then deselects every test in the file. That took the tier to 131s against
    this ceiling. A metric summing only test durations would have read green
    through it, so `fast_tier_seconds` is test time *plus* collection and the
    collection number is printed on its own.
    """

    conftest = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    assert "def pytest_collection(" in conftest, (
        "conftest.py must time collection; the tier metric is built on it"
    )
    assert '"fast_tier_seconds"' in conftest
    assert "collect_seconds" in conftest

    command = next(
        step["run"]
        for step in _steps(CI, "python")
        if "fast_tier_seconds" in step.get("run", "")
    )
    assert "collect" in command, "CI must surface the collection number separately"


def test_the_bless_job_is_manual_and_still_compares_by_value() -> None:
    """It asks to be run; it does not run itself. And it never gated.

    The previous version of this test asserted the `bless_diff` step was not
    `continue-on-error` so that it "must be able to fail". That guarantee did not
    exist: `continue-on-error: true` sits on the JOB, so the job's conclusion
    cannot affect the workflow's however the step exits -- ci.yml's own comment
    says "it never gates". A guard for a property nothing provides is worse than
    no guard, because it reads as coverage.

    What is worth pinning is the shape that survives: the job is opt-in, and when
    someone does run it, the comparison is by value rather than by diff stat.
    Adding a second platform's digest rewrites the first entry's line, so the
    first Windows bless rendered as 423 insertions and 282 deletions with zero
    digests changed.
    """

    job = _jobs(CI)["windows-corpus-hashes"]
    assert job.get("if") == "github.event_name == 'workflow_dispatch'", (
        "the bless job must be workflow_dispatch only. It asserts nothing and its "
        "output is an artifact a human merges by hand; a 30-minute Windows runner "
        "on every code PR buys a file nobody reads on that PR."
    )
    assert job.get("continue-on-error") is True, (
        "if this job can ever fail the workflow, the comment saying it never "
        "gates is wrong and the artifact-only contract needs revisiting"
    )

    comparisons = [step for step in job["steps"] if "bless_diff" in step.get("run", "")]
    assert comparisons, "the bless job must compare digests by value, not by diff stat"


def test_nightly_runs_the_slow_tier() -> None:
    """The only place it runs. `-m slow` is never implied by a bare `pytest`.

    It went unrun for months: nightly's bare `pytest` selects `fast or medium`
    from addopts, so the job was a duplicate of every PR run plus coverage, and
    the slow tier executed in no automated context at all.
    """

    commands = [step.get("run", "") for step in _steps(NIGHTLY, "coverage")]
    assert any(re.search(r"pytest\b.*-m slow", command) for command in commands), (
        "nightly must run `pytest -m slow`; nothing else does, and a tier that "
        "runs nowhere is not a tier"
    )


def test_every_env_gated_test_has_a_workflow_that_opens_the_gate() -> None:
    """A skipif on an env var nothing sets is a test that does not exist.

    Four of these accumulated -- roughly 440 lines covering the postgres job and
    library stores, real production surface, gated on RIGBY_TEST_POSTGRES and
    friends that NO workflow set. They read as covered and ran nowhere.

    Only names inside a `skipif` count. Tests set plenty of other RIGBY_ vars on
    themselves via monkeypatch to drive configuration, and those are inputs, not
    gates -- an earlier version of this guard flagged all fourteen of them and
    would have been silenced rather than read.

    The one exception is declared by name rather than by silence: the library
    canary asserts row counts from one populated database and cannot run against
    a fresh one, so it is opt-in on purpose.
    """

    unrunnable_by_design = {"RIGBY_V2_CANARY_DATABASE_URL"}

    gates: dict[str, str] = {}
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (isinstance(function, ast.Attribute) and function.attr == "skipif"):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    if re.fullmatch(r"RIGBY_[A-Z0-9_]+", inner.value):
                        gates.setdefault(inner.value, path.name)

    assert gates, "no env-gated tests found at all; the scan is broken, not the suite"

    workflows = "\n".join(path.read_text(encoding="utf-8") for path in _all_workflows())
    unopened = sorted(
        f"{name} (gates {source})"
        for name, source in gates.items()
        if name not in unrunnable_by_design and name not in workflows
    )
    assert not unopened, (
        f"these env gates are set by no workflow, so the tests behind them run "
        f"nowhere: {unopened}. Wire one up in nightly.yml, delete the test, or add "
        f"it to `unrunnable_by_design` with the reason."
    )


def test_no_job_duplicates_a_check_the_suite_already_makes() -> None:
    """`generated-sources` was a whole job for one assertion pytest already made.

    A dedicated runner plus a full `uv sync` (mujoco, playwright, openai) to run
    `evals.generate_camera_ts --check`, which tests/test_camera_config.py asserts
    in-process on every run of the suite.
    """

    for job in _jobs(CI):
        for step in _steps(CI, job):
            assert "generate_camera_ts" not in step.get("run", ""), (
                f"ci.yml job `{job}` re-runs the camera codegen check; "
                f"tests/test_camera_config.py already covers it"
            )


def test_ci_runs_the_frontend_checks() -> None:
    commands = [step.get("run", "") for step in _steps(CI, "frontend")]
    assert any("npm ci" in command for command in commands)
    assert any(command.strip() == "npm test" for command in commands)
    assert any("npm run build" in command for command in commands)


def test_every_job_has_a_timeout() -> None:
    """An untimed job that hangs burns the whole runner budget."""

    for path in _all_workflows():
        for name, job in _jobs(path).items():
            if name == "select-platforms":
                continue  # trivial, ubuntu-only, seconds long
            assert job.get("timeout-minutes"), f"{path.name}:{name} has no timeout"
