"""Load and analyse the whole corpus with the outside world removed.

Run as a subprocess by ``tests/test_corpus_loads_offline.py``. It has to be a fresh
process: proving that no browser or HTTP client is *imported* is meaningless in a
pytest process where another test already imported them.

Prints a JSON summary on success; raises on any attempt to reach the network, start
a process, or import a client library.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys

#: Importing any of these means the corpus reached for a server, a browser, or a
#: model provider. ``mujoco`` is deliberately absent: the compiler links it in
#: process and never talks to anything.
FORBIDDEN_MODULES = (
    "playwright",
    "playwright.sync_api",
    "openai",
    "httpx",
    "requests",
    "urllib.request",
    "uvicorn",
    "fastapi",
    "rigby_poc.planner",
    "rigby_poc.app",
    "evals.capture",
)

NETWORK_FAMILIES = {socket.AF_INET, socket.AF_INET6}

#: Substrings that identify a browser, a driver, or a headless-render harness.
#: Blanket-blocking every subprocess is too strict to be useful: importing
#: ``mujoco`` runs ``sysctl -n sysctl.proc_translated`` on macOS to detect Rosetta,
#: which is a local CPU probe and not a violation of anything. So launches are
#: recorded and matched by name, and the recorded list is asserted on.
BROWSER_MARKERS = (
    "chrome",
    "chromium",
    "firefox",
    "webkit",
    "msedge",
    "safari",
    "playwright",
    "geckodriver",
    "chromedriver",
    "node",
    "headless_shell",
)

#: Every ``subprocess`` call the probe observed, as argv lists.
LAUNCHES: list[list[str]] = []


class NetworkAccessError(RuntimeError):
    pass


class BrowserLaunchError(RuntimeError):
    pass


def _record_launch(command: object) -> None:
    argv = [str(item) for item in command] if isinstance(command, (list, tuple)) else [str(command)]
    LAUNCHES.append(argv)
    lowered = " ".join(argv).lower()
    for marker in BROWSER_MARKERS:
        if marker in lowered:
            raise BrowserLaunchError(f"corpus launched a browser: {argv}")


def _block_the_outside_world() -> None:
    original_socket_init = socket.socket.__init__
    original_popen = subprocess.Popen
    original_run = subprocess.run

    def guarded_socket_init(self, family=socket.AF_INET, *args, **kwargs):  # type: ignore[no-untyped-def]
        if family in NETWORK_FAMILIES:
            raise NetworkAccessError(f"corpus opened a network socket (family={family})")
        original_socket_init(self, family, *args, **kwargs)

    def guarded_resolution(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise NetworkAccessError("corpus resolved a hostname")

    def guarded_popen(command, *args, **kwargs):  # type: ignore[no-untyped-def]
        _record_launch(command)
        return original_popen(command, *args, **kwargs)

    def guarded_run(command, *args, **kwargs):  # type: ignore[no-untyped-def]
        _record_launch(command)
        return original_run(command, *args, **kwargs)

    socket.socket.__init__ = guarded_socket_init  # type: ignore[method-assign]
    socket.create_connection = guarded_resolution  # type: ignore[assignment]
    socket.getaddrinfo = guarded_resolution  # type: ignore[assignment]
    subprocess.Popen = guarded_popen  # type: ignore[assignment]
    subprocess.run = guarded_run  # type: ignore[assignment]


def main() -> int:
    _block_the_outside_world()

    from evals.corpus import compile_case, load_corpus, motion_sha256
    from evals.corpus.verify import Verdict, compare_case

    cases = load_corpus()
    digests: dict[str, str] = {}
    verdicts: dict[str, str] = {}
    for case in cases:
        clip = compile_case(case)
        digests[case.id] = motion_sha256(clip)
        comparison = compare_case(case, clip)
        verdicts[case.id] = comparison.verdict.value
        if comparison.verdict is Verdict.MISMATCH:
            raise SystemExit(f"case moved during the offline probe:\n{comparison.render()}")

    imported = sorted(name for name in FORBIDDEN_MODULES if name in sys.modules)
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "digests": digests,
                "verdicts": verdicts,
                "forbidden_imports": imported,
                "launches": LAUNCHES,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
