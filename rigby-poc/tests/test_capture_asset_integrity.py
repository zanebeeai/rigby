"""Plan 05 section 1.1 — a failed asset load must never produce evidence.

The old capture path swapped in a procedurally generated capsule avatar when the
humanoid GLB failed to load, still reported `data-capture-status="ready"`, and wrote a
complete, well-formed manifest with valid per-snapshot hashes. Nothing recorded that the
real asset never loaded, so a grader scored a different body and a calibration run
poisoned its own ground truth.

These tests exercise the Python half of the guard at the `CaptureSession` seam, with no
browser. The browser-backed proof of the same failure lives in
`test_capture_asset_integrity_e2e.py`; the frontend half is `frontend/src/avatar.test.ts`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.capture import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    CaptureAssetError,
    CaptureDeadlineExceeded,
    CaptureSession,
    MAX_SNAPSHOTS_PER_VIEW,
    capture_deadline_s,
    humanoid_asset_sha256,
    subprocess_timeout_s,
)


ASSET_SHA = "c7" + "0" * 62
OTHER_SHA = "ff" + "0" * 62


def _png(width: int = CAPTURE_WIDTH, height: int = CAPTURE_HEIGHT) -> bytes:
    """A byte string whose PNG header declares `width` x `height`."""
    import struct

    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", width, height) + b"body"


def _provenance(**overrides: Any) -> dict[str, Any]:
    return {
        "three_revision": "179",
        "webgl_vendor": "Google Inc. (Apple)",
        "webgl_renderer": "ANGLE (Apple, Apple M-series)",
        "webgl_version": "WebGL 2.0",
        "shading_language_version": "WebGL GLSL ES 3.00",
        "max_texture_size": 16384,
        "device_pixel_ratio": 1,
        "antialias": True,
        "shadow_map": "PCFSoft/2048",
        "render_mode": "judged",
        "asset_url": "/assets/models/human-male.glb",
        "asset_sha256": ASSET_SHA,
        "asset_bytes": 534004,
        "asset_status": "loaded",
        "asset_error": None,
        "user_agent": "Mozilla/5.0 test",
        **overrides,
    }


def _rendered_time_s(time_s: float) -> float:
    """What a real page reports: the time of the frame it snapped to, not the request."""
    best = CLIP_FRAMES[0]["time_s"]
    for frame in CLIP_FRAMES:
        if frame["time_s"] > min(time_s, 1.6):
            break
        best = frame["time_s"]
    return float(best)


def _state(time_s: float, view: str, **overrides: Any) -> dict[str, Any]:
    state = {
        "status": "ready",
        "result_id": "000001-test",
        "prompt": "test prompt",
        "requested_time_s": time_s,
        "rendered_time_s": _rendered_time_s(time_s),
        "camera": {"view": view, "vertical_fov_deg": 94 if view == "ego" else 46},
        "render_provenance": _provenance(),
    }
    state.update(overrides)
    return state


class FakePage:
    """A capture page whose responses a test dictates."""

    def __init__(self, states: Any = None) -> None:
        self.states = states
        self.opened: list[str] = []
        self.seeks: list[tuple[float, str]] = []
        self.screenshots = 0
        self.recycles = 0
        self.png = _png()

    def _state(self, time_s: float, view: str) -> dict[str, Any]:
        if callable(self.states):
            return self.states(time_s, view)
        return _state(time_s, view)

    def open(self, url: str) -> dict[str, Any]:
        self.opened.append(url)
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(url).query)
        return self._state(float(query["time"][0]), query["view"][0])

    def seek(self, time_s: float, view: str) -> dict[str, Any]:
        self.seeks.append((time_s, view))
        return self._state(time_s, view)

    def screenshot(self) -> bytes:
        self.screenshots += 1
        return self.png

    def recycle(self) -> None:
        self.recycles += 1

    def supports_seek(self) -> bool:
        return True


CLIP_FRAMES = [
    {"time_s": 0.0, "bones": {"hips": {"rotation": [0.0, 0.0, 0.0, 1.0]}}},
    {"time_s": 0.4, "bones": {"hips": {"rotation": [0.0, 0.1, 0.0, 0.99]}}},
    {"time_s": 0.8, "bones": {"hips": {"rotation": [0.0, 0.2, 0.0, 0.98]}}},
    {"time_s": 1.2, "bones": {"hips": {"rotation": [0.0, 0.3, 0.0, 0.95]}}},
    {"time_s": 1.6, "bones": {"hips": {"rotation": [0.0, 0.4, 0.0, 0.91]}}},
]

PAYLOAD = {
    "id": "000001-test",
    "program": {"source_text": "test prompt", "intent": "gesture"},
    "scene": {"objects": []},
    "clip": {
        "duration_s": 1.6,
        "frames": CLIP_FRAMES,
        "metrics": {
            "phase_ranges_s": [
                {"kind": "present", "start_s": 0.0, "end_s": 0.8},
                {"kind": "hold", "start_s": 0.8, "end_s": 1.6},
            ]
        },
    },
}


@pytest.fixture(autouse=True)
def _stub_result_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.capture._result_payload", lambda base_url, result_id: PAYLOAD)


def _session(page: FakePage, **kwargs: Any) -> CaptureSession:
    return CaptureSession(
        page,
        base_url="http://127.0.0.1:8000",
        expected_asset_sha256=ASSET_SHA,
        **kwargs,
    )


def test_capture_writes_a_manifest_when_the_declared_humanoid_loaded(tmp_path: Path) -> None:
    page = FakePage()
    manifest_path = _session(page).capture("000001-test", tmp_path, views=("orbit",))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["snapshots"]
    assert manifest["render_provenance"]["asset_sha256"] == ASSET_SHA
    assert manifest["render_provenance"]["asset_status"] == "loaded"


def test_a_failed_asset_load_fails_capture_and_writes_no_manifest(tmp_path: Path) -> None:
    """The direct guard against section 1.1.

    Against the old behaviour the page reported `ready` with a substituted body and this
    wrote a full manifest. It must now raise, and leave nothing behind that a grader
    could pick up.
    """
    page = FakePage(
        lambda time_s, view: _state(
            time_s,
            view,
            status="error",
            error="humanoid asset /assets/models/human-male.glb could not be fetched: HTTP 404",
            render_provenance=None,
        )
    )
    with pytest.raises(CaptureAssetError, match="404"):
        _session(page).capture("000001-test", tmp_path, views=("orbit",))
    assert not (tmp_path / "evidence-manifest.json").exists()
    assert not list(tmp_path.glob("*.png"))


def test_a_substituted_body_fails_even_when_the_page_reports_ready(tmp_path: Path) -> None:
    """The exact section 1.1 shape: status ready, but the body is the procedural stand-in."""
    page = FakePage(
        lambda time_s, view: _state(
            time_s,
            view,
            render_provenance=_provenance(
                asset_status="substituted",
                asset_sha256=None,
                asset_error="humanoid asset could not be fetched: HTTP 404",
            ),
        )
    )
    with pytest.raises(CaptureAssetError, match="substituted"):
        _session(page).capture("000001-test", tmp_path, views=("orbit",))
    assert not (tmp_path / "evidence-manifest.json").exists()


def test_a_successfully_loaded_wrong_body_fails_on_the_asset_hash(tmp_path: Path) -> None:
    """A load that succeeds against the wrong GLB is the failure a status check misses."""
    page = FakePage(
        lambda time_s, view: _state(
            time_s, view, render_provenance=_provenance(asset_sha256=OTHER_SHA)
        )
    )
    with pytest.raises(CaptureAssetError, match="different body"):
        _session(page).capture("000001-test", tmp_path, views=("orbit",))
    assert not (tmp_path / "evidence-manifest.json").exists()


def test_a_manifest_without_render_provenance_is_refused(tmp_path: Path) -> None:
    """A stale frontend bundle cannot attest what it rendered, so it cannot be trusted."""
    page = FakePage(lambda time_s, view: _state(time_s, view, render_provenance=None))
    with pytest.raises(CaptureAssetError, match="predates plan 05"):
        _session(page).capture("000001-test", tmp_path, views=("orbit",))


def test_an_asset_failure_is_never_retried(tmp_path: Path) -> None:
    """Retrying a missing humanoid only produces the same wrong body three times."""
    page = FakePage(lambda time_s, view: _state(time_s, view, status="error", error="gone"))
    with pytest.raises(CaptureAssetError):
        _session(page).capture("000001-test", tmp_path, views=("orbit",))
    assert page.recycles == 0
    assert len(page.opened) == 1


def test_the_expected_asset_hash_comes_from_the_shipped_asset_manifest() -> None:
    """The hash capture compares against is the repository's declared humanoid."""
    import hashlib

    declared = humanoid_asset_sha256()
    glb = Path(__file__).resolve().parents[1] / "assets" / "models" / "human-male.glb"
    assert hashlib.sha256(glb.read_bytes()).hexdigest() == declared


def test_a_run_past_its_own_budget_fails_before_the_subprocess_timeout(tmp_path: Path) -> None:
    """Capture's deadline sits strictly below the subprocess budget derived from it."""
    ticks = iter([0.0] + [10_000.0] * 50)
    page = FakePage()
    session = _session(page, clock=lambda: next(ticks))
    with pytest.raises(CaptureDeadlineExceeded, match="exceeded its budget"):
        session.capture("000001-test", tmp_path, views=("orbit",))
    assert subprocess_timeout_s() > capture_deadline_s()


def test_the_snapshot_bound_the_budget_assumes_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An intent that outgrows the sampling bound fails here, not as a mystery timeout."""
    monkeypatch.setattr(
        "evals.capture.phase_sampling_points",
        lambda payload: [
            {"time_s": 0.0, "label": f"p{index}", "phase": "present"}
            for index in range(MAX_SNAPSHOTS_PER_VIEW + 1)
        ],
    )
    with pytest.raises(RuntimeError, match="phase points per view"):
        _session(FakePage()).capture("000001-test", tmp_path, views=("orbit",))
