"""Plan 05 sections 1.4 and 1.6 — a timed-out capture must not leak a browser.

`capture_in_subprocess` allowed 3 attempts of 45 s navigation + 45 s selector + 45 s
screenshot per snapshot, roughly 405 s, against a hardcoded 300 s subprocess budget. The
budget therefore fired during normal work, and `subprocess.TimeoutExpired` escaped a
layer that did not catch it, leaving the orphaned Chrome process unreaped. Capture output
was sent to the null device besides, so a failure surfaced only as an exit code.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evals.capture import capture_deadline_s, subprocess_timeout_s
from rigby_poc.pipeline import _capture_budget_s, _run_capture, capture_in_subprocess


def _alive(pid: int) -> bool:
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False
    )
    return str(pid) in completed.stdout


def _wait_until_gone(pid: int, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


def test_the_subprocess_budget_is_derived_from_captures_own_deadline() -> None:
    """The ordering that made the old failure possible is now true by construction."""
    for result_count in (1, 5, 11):
        assert subprocess_timeout_s(result_count=result_count) > capture_deadline_s(
            result_count=result_count
        )
        assert _capture_budget_s(result_count) == subprocess_timeout_s(result_count=result_count)
    # The old constant was 300 s for a single result; the derived budget must clear
    # capture's own worst case for that same single result.
    assert _capture_budget_s(1) >= capture_deadline_s(result_count=1)


def test_a_hanging_capture_is_killed_together_with_its_children(tmp_path: Path) -> None:
    """The direct guard against the orphaned browser.

    `subprocess.run(timeout=...)` kills only the direct child, which is exactly why a
    timed-out capture used to leave Chrome running. The grandchild here stands in for
    the browser.
    """
    marker = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)']); "
        f"open({str(marker)!r}, 'w').write(str(child.pid)); "
        "sys.stdout.flush(); "
        "time.sleep(300)"
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        _run_capture(
            [sys.executable, "-c", script],
            tmp_path,
            budget_s=5.0,
            description="hanging capture",
        )
    assert marker.is_file(), "the stand-in child never started"
    grandchild = int(marker.read_text())
    assert _wait_until_gone(grandchild), (
        f"pid {grandchild} survived the timeout; a real capture would have leaked Chrome"
    )


def test_a_timeout_names_the_log_it_wrote(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="capture.log"):
        _run_capture(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            budget_s=2.0,
            description="hanging capture",
        )
    assert (tmp_path / "capture.log").is_file()


def test_capture_output_is_persisted_rather_than_discarded(tmp_path: Path) -> None:
    """Section 1.6: stdout and stderr went to the null device, so failures were mute."""
    script = (
        "import sys; "
        "print('capture said something'); "
        "print('and something on stderr', file=sys.stderr); "
        "sys.exit(3)"
    )
    with pytest.raises(RuntimeError) as failure:
        _run_capture(
            [sys.executable, "-c", script], tmp_path, budget_s=60.0, description="failing capture"
        )
    message = str(failure.value)
    assert "exit code 3" in message
    assert "and something on stderr" in message
    log = (tmp_path / "capture.log").read_text(encoding="utf-8")
    assert "capture said something" in log
    assert "and something on stderr" in log


def test_a_successful_capture_leaves_its_log_and_returns_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "evidence-manifest.json"
    recorded: dict[str, object] = {}

    def fake_run(command: list[str], log_dir: Path, *, budget_s: float, description: str) -> str:
        recorded["budget_s"] = budget_s
        recorded["command"] = command
        log_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        return ""

    monkeypatch.setattr("rigby_poc.pipeline._run_capture", fake_run)
    assert capture_in_subprocess("000001-test", tmp_path, base_url="http://127.0.0.1:8000") == manifest
    assert recorded["budget_s"] == subprocess_timeout_s(result_count=1)


def test_a_capture_that_writes_no_manifest_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rigby_poc.pipeline._run_capture",
        lambda command, log_dir, *, budget_s, description: "",
    )
    with pytest.raises(RuntimeError, match="without a manifest"):
        capture_in_subprocess("000001-test", tmp_path, base_url="http://127.0.0.1:8000")
