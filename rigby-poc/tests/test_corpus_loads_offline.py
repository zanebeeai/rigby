"""The corpus must be usable on a clean clone with no key, no server, no browser.

That property is the reason the corpus exists (plan 03 section 1), so it is asserted
rather than assumed. The check runs in a fresh subprocess with network sockets,
hostname resolution and process launching all disabled -- see
``tests/corpus_offline_probe.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from evals.corpus import load_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).resolve().parent / "corpus_offline_probe.py"


@pytest.fixture(scope="module")
def probe_result() -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    )
    # A key in the environment would let an accidental provider call succeed
    # quietly. Removing it means such a call fails loudly instead.
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        environment.pop(name, None)
    completed = subprocess.run(
        [sys.executable, str(PROBE)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_the_whole_corpus_loads_and_compiles_with_no_network(
    probe_result: dict[str, object],
) -> None:
    assert probe_result["case_count"] == len(load_corpus())
    # `tolerance_match` is a check that ran and passed: no hash is blessed for this
    # platform, so the committed clip was compared instead.  Treating it as a
    # failure would make every unblessed platform red for having less evidence
    # rather than for disagreeing.
    assert set(probe_result["verdicts"].values()) <= {"match", "tolerance_match"}  # type: ignore[union-attr]
    assert probe_result["verdicts"], "the probe compared nothing"


def test_no_browser_server_or_model_client_is_imported(
    probe_result: dict[str, object],
) -> None:
    assert probe_result["forbidden_imports"] == []


def test_no_browser_is_started(probe_result: dict[str, object]) -> None:
    """Any subprocess the corpus run touches is reported, not just browsers.

    Importing ``mujoco`` on macOS runs ``sysctl -n sysctl.proc_translated`` to detect
    Rosetta. That is a local CPU probe, so it is recorded rather than blocked -- but
    it is the only launch a corpus run is expected to make, and anything browser-like
    fails inside the probe before it gets here.
    """
    launches = probe_result["launches"]
    assert isinstance(launches, list)
    for argv in launches:
        assert argv[0] in {"sysctl"}, f"unexpected subprocess during a corpus run: {argv}"


GUARD_FIRES_SCRIPT = """
import socket, subprocess, sys
sys.path.insert(0, {probe_dir!r})
from corpus_offline_probe import (
    BrowserLaunchError,
    NetworkAccessError,
    _block_the_outside_world,
)

_block_the_outside_world()
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except NetworkAccessError:
    print("socket blocked")
try:
    socket.getaddrinfo("example.com", 443)
except NetworkAccessError:
    print("dns blocked")
try:
    subprocess.Popen(["/opt/chromium/headless_shell", "--remote-debugging-port=0"])
except BrowserLaunchError:
    print("browser blocked")
"""


def test_the_offline_probe_would_notice_a_network_call_or_a_browser() -> None:
    """The guard is only worth having if it actually fires."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    )
    completed = subprocess.run(
        [sys.executable, "-c", GUARD_FIRES_SCRIPT.format(probe_dir=str(PROBE.parent))],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == [
        "socket",
        "blocked",
        "dns",
        "blocked",
        "browser",
        "blocked",
    ]
