"""A minimal stand-in for the Rigby app, sufficient to drive a real capture.

The full FastAPI app needs a persisted `ResultStore`, a results root, and a built
frontend mounted at `/`. Capture only reads three things from it: the built
`capture.html` bundle, the humanoid GLB under `/assets`, and the result payload at
`/api/v1/results/<id>`. Serving exactly those keeps the browser-backed tests fast and
lets a test corrupt one of them — the point of `tests/test_capture_asset_integrity.py`
is what capture does when the GLB is missing or is the wrong body.
"""

from __future__ import annotations

import json
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"
DIST = FRONTEND / "dist"
REAL_GLB = PROJECT_ROOT / "assets" / "models" / "human-male.glb"

_BUILD_LOCK = threading.Lock()
_MEDIA_TYPES = {
    ".css": "text/css",
    ".glb": "model/gltf-binary",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".json": "application/json",
    ".map": "application/json",
}


def frontend_prerequisites() -> str | None:
    """Return why the browser-backed tests cannot run, or None when they can."""
    if not (FRONTEND / "node_modules").is_dir() and _dist_is_stale():
        return "frontend/node_modules is absent and frontend/dist is missing or stale"
    if not REAL_GLB.is_file():
        return f"{REAL_GLB} is missing"
    return None


def _dist_is_stale() -> bool:
    """Whether any frontend source is newer than the built bundle.

    A stale `dist` is the quiet way a browser-backed test lies: it exercises the
    previous bundle and reports on code that is no longer there.
    """
    built = DIST / "capture.html"
    if not built.is_file():
        return True
    newest = max(
        (path.stat().st_mtime for path in FRONTEND.glob("src/**/*") if path.is_file()),
        default=0.0,
    )
    newest = max(newest, *(path.stat().st_mtime for path in FRONTEND.glob("*.html")))
    return newest > built.stat().st_mtime


def build_frontend() -> Path:
    """Build `frontend/dist` when it is missing or stale. Vite takes under a second."""
    with _BUILD_LOCK:
        if not _dist_is_stale():
            return DIST
        if not (FRONTEND / "node_modules").is_dir():
            raise RuntimeError(
                "frontend/dist is stale and frontend/node_modules is absent; run `npm ci` in frontend/"
            )
        completed = subprocess.run(
            ["npm", "run", "build"],
            cwd=FRONTEND,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if completed.returncode != 0 or not (DIST / "capture.html").is_file():
            raise RuntimeError(f"frontend build failed:\n{completed.stdout}\n{completed.stderr}")
    return DIST


class CaptureAppStub:
    """Routing state for one served app. Mutate the fields to break one dependency."""

    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        self.results = results
        # None serves the real asset; b"" is served as a 404; any other value is
        # served verbatim, which is how a *successful load of the wrong body* is
        # reproduced without shipping a second GLB.
        self.glb_override: bytes | None = None
        self.glb_status = 200
        self.request_log: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    stub: CaptureAppStub
    dist: Path

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base signature
        return  # keep the test output readable

    def _send(self, status: int, body: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, path.read_bytes(), _MEDIA_TYPES.get(path.suffix, "application/octet-stream"))

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        route = unquote(urlparse(self.path).path)
        self.stub.request_log.append(route)
        if route == "/assets/models/human-male.glb":
            self._send_glb()
        elif route == "/assets/manifest.json":
            self._send_file(PROJECT_ROOT / "assets" / "manifest.json")
        elif route == "/api/v1/results":
            self._send(200, json.dumps({"results": []}).encode("utf-8"), "application/json")
        elif route.startswith("/api/v1/results/"):
            result_id = route[len("/api/v1/results/"):]
            payload = self.stub.results.get(result_id)
            if payload is None:
                self._send(404, b'{"detail":"result not found"}', "application/json")
            else:
                self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
        elif route.startswith("/static/"):
            self._send_file(_within(self.dist, route[len("/static/"):]))
        else:
            self._send_file(_within(self.dist, route.lstrip("/") or "index.html"))

    def _send_glb(self) -> None:
        if self.stub.glb_status != 200:
            self._send(self.stub.glb_status, b"unavailable", "text/plain")
            return
        body = REAL_GLB.read_bytes() if self.stub.glb_override is None else self.stub.glb_override
        self._send(200, body, "model/gltf-binary")


def _within(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root`, refusing to escape it."""
    candidate = (root / relative).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        return root / "__forbidden__"
    return candidate


@contextmanager
def serve_capture_app(results: dict[str, dict[str, Any]]) -> Iterator[tuple[str, CaptureAppStub]]:
    """Serve a built frontend plus `results` and yield `(base_url, stub)`."""
    dist = build_frontend()
    stub = CaptureAppStub(results)
    handler = type("_BoundHandler", (_Handler,), {"stub": stub, "dist": dist})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="capture-e2e-app")
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", stub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
