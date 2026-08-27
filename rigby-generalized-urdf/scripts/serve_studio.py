"""Serve the trace studio and print the URL.

Opening ``results/studio.html`` straight off disk mostly works -- the page has no
fetch or XHR and every payload is inlined -- but it is not what the studio is
tested against, and some viewers refuse a file that size. A server costs one
command and removes the question.

It has to be rooted at ``results/`` rather than the repository: contact-probe
clips are still GIFs referenced relatively from each trace's own directory, and
serving from anywhere else turns them into broken images.
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import socket
import socketserver
import webbrowser
from functools import partial
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_PORT = 8712


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """The request log is noise; a 3 MB page is one line of it per reload."""

    def log_message(self, fmt, *args):  # noqa: A003 - base class signature
        return


def free_port(preferred: int) -> int:
    """Use the preferred port, or the next one nothing is already holding."""

    for candidate in range(preferred, preferred + 20):
        with contextlib.closing(socket.socket()) as probe:
            if probe.connect_ex(("127.0.0.1", candidate)) != 0:
                return candidate
    return preferred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true")
    arguments = parser.parse_args()

    page = RESULTS / "studio.html"
    if not page.is_file():
        print(f"No studio at {page}.\nBuild it: uv run python scripts/build_studio.py")
        return 1

    port = free_port(arguments.port)
    handler = partial(QuietHandler, directory=str(RESULTS))
    socketserver.TCPServer.allow_reuse_address = True

    url = f"http://127.0.0.1:{port}/studio.html"
    with socketserver.TCPServer(("127.0.0.1", port), handler) as server:
        size = page.stat().st_size / 1024 / 1024
        print(f"Trace studio ({size:.1f} MB)  ->  {url}")
        print("Ctrl-C to stop.")
        if not arguments.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
