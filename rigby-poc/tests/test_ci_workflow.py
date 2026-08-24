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

from pathlib import Path
from typing import Any

import pytest
import yaml


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
        f"must stay that way -- see rigby-poc/docs/testing.md."
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
