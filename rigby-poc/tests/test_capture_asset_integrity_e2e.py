"""Browser-backed proof of plan 05 sections 1.1, 1.2 and 3.4.

`test_capture_asset_integrity.py` covers the same guards at the `CaptureSession` seam
with no browser, and is the version that runs everywhere. This module drives a real
Chrome against a real built frontend, because the failure it exists to prove -- a 404 on
the humanoid GLB producing a complete manifest of a procedurally generated capsule --
lives in the interaction between the two halves and can only be demonstrated by running
both.

Skipped when the frontend cannot be built or Chrome is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from evals.capture import (
    capture_result_frames,
    frame_at,
    humanoid_asset_sha256,
    pose_sha256,
)
from tests.capture_e2e_server import frontend_prerequisites, serve_capture_app


_MISSING = frontend_prerequisites()
#: needs a real browser -- see docs/testing.md. `slow` is never implied by the
#: default invocation, and the skipif stays so a machine without npm/Chrome
#: skips cleanly rather than failing.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(_MISSING is not None, reason=str(_MISSING)),
]

RESULT_ID = "000001-capture-integrity"
PROMPT = "Throw up a hang-ten sign with your right hand."


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    from rigby_poc.compiler import compile_motion
    from rigby_poc.models import CompileRequest, PlanRequest, default_scene
    from rigby_poc.planner import OfflinePlanner

    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=PROMPT, scene=scene, provider="offline")
    ).program
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert clip.success, clip.failure
    return {
        "id": RESULT_ID,
        "scene": scene.model_dump(mode="json"),
        "program": program.model_dump(mode="json"),
        "clip": clip.model_dump(mode="json"),
        "animation_url": None,
    }


@pytest.fixture
def app(payload: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    with serve_capture_app({RESULT_ID: payload}) as served:
        yield served


def _capture(base_url: str, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("views", ("orbit",))
    path = capture_result_frames(RESULT_ID, output_dir, base_url=base_url, **kwargs)
    return json.loads(path.read_text(encoding="utf-8"))


def test_capture_records_the_hash_of_the_humanoid_it_actually_rendered(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    base_url, _ = app
    manifest = _capture(base_url, tmp_path)
    provenance = manifest["render_provenance"]
    assert provenance["asset_status"] == "loaded"
    assert provenance["asset_sha256"] == humanoid_asset_sha256()
    assert provenance["asset_sha256"] == provenance["expected_asset_sha256"]
    # The hash is attested by the browser over the bytes it parsed, not recomputed
    # Python-side from the same manifest it is checked against -- which would prove
    # nothing at all.
    assert provenance["three_revision"]
    assert provenance["webgl_renderer"]
    assert provenance["render_mode"] == "judged"
    assert provenance["antialias"] is True
    assert manifest["snapshots"]


def test_a_missing_humanoid_glb_fails_capture_and_writes_no_manifest(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """Plan 05 section 1.1, end to end.

    Against the pre-05 tree this produced 13 snapshots of a capsule mannequin with a
    valid manifest and valid per-snapshot hashes, while the GLB 404'd 13 times.
    """
    base_url, stub = app
    stub.glb_status = 404
    with pytest.raises(RuntimeError) as failure:
        _capture(base_url, tmp_path)
    assert "404" in str(failure.value) or "could not be fetched" in str(failure.value)
    assert not (tmp_path / "evidence-manifest.json").exists()
    assert not list(tmp_path.glob("*.png"))


def test_a_body_that_is_not_the_declared_humanoid_fails_capture(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """A *successful* load of the wrong bytes is what a status check alone cannot catch."""
    base_url, stub = app
    original = (Path(__file__).resolve().parents[1] / "assets" / "models" / "human-male.glb").read_bytes()
    # Same container, different contents: a valid-looking GLB that is not the one the
    # clip was compiled against.
    stub.glb_override = original[:-64]
    with pytest.raises(RuntimeError) as failure:
        _capture(base_url, tmp_path)
    message = str(failure.value)
    assert "different body" in message or "could not be parsed" in message
    assert not (tmp_path / "evidence-manifest.json").exists()


def test_pose_hashes_are_identical_across_two_captures_of_the_same_result(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """Section 3.3: pose determinism is the property the suite may assert."""
    base_url, _ = app
    first = _capture(base_url, tmp_path / "first")
    second = _capture(base_url, tmp_path / "second")
    assert [snapshot["pose_sha256"] for snapshot in first["snapshots"]] == [
        snapshot["pose_sha256"] for snapshot in second["snapshots"]
    ]
    assert [snapshot["rendered_time_s"] for snapshot in first["snapshots"]] == [
        snapshot["rendered_time_s"] for snapshot in second["snapshots"]
    ]


def test_the_pose_hash_is_the_hash_of_the_frame_the_page_rendered(
    app: tuple[str, Any], tmp_path: Path, payload: dict[str, Any]
) -> None:
    """The hash means nothing unless Python selected the frame the renderer selected."""
    base_url, _ = app
    manifest = _capture(base_url, tmp_path)
    for snapshot in manifest["snapshots"]:
        expected = frame_at(payload["clip"], float(snapshot["requested_time_s"]))
        assert expected is not None
        assert float(expected["time_s"]) == pytest.approx(snapshot["rendered_time_s"], abs=1e-9)
        assert snapshot["pose_sha256"] == pose_sha256(expected)
    # Distinct sample points must not collapse to one pose, or the assertion above would
    # hold trivially for a hash of nothing.
    assert len({snapshot["pose_sha256"] for snapshot in manifest["snapshots"]}) > 1


def test_pixel_hashes_are_reproducible_on_one_machine(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """Section 1.2 claims pixels differ *across* machines; this pins what holds on one.

    Plan 06 needs this: if a render carried same-machine noise, a pixel-diff detection
    threshold across a mutation severity sweep would be measuring the noise floor.
    """
    base_url, _ = app
    first = _capture(base_url, tmp_path / "a")
    second = _capture(base_url, tmp_path / "b")
    first_pixels = [snapshot["pixel_sha256"] for snapshot in first["snapshots"]]
    second_pixels = [snapshot["pixel_sha256"] for snapshot in second["snapshots"]]
    identical = sum(one == two for one, two in zip(first_pixels, second_pixels))
    assert identical == len(first_pixels), (
        f"only {identical}/{len(first_pixels)} snapshots were pixel-identical across two "
        "captures on this machine; a pixel-diff detection threshold would be measuring "
        "render noise"
    )
    assert [snapshot["sha256"] for snapshot in first["snapshots"]] == first_pixels


def test_the_seek_path_and_the_reload_path_produce_the_same_poses(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """Section 6.3: the seek hook must not diverge from the page reload it replaces."""
    base_url, _ = app
    seeked = _capture(base_url, tmp_path / "seek", strategy="seek")
    reloaded = _capture(base_url, tmp_path / "reload", strategy="reload")
    assert [snapshot["pose_sha256"] for snapshot in seeked["snapshots"]] == [
        snapshot["pose_sha256"] for snapshot in reloaded["snapshots"]
    ]
    assert [snapshot["rendered_time_s"] for snapshot in seeked["snapshots"]] == [
        snapshot["rendered_time_s"] for snapshot in reloaded["snapshots"]
    ]


def test_the_seek_path_loads_the_humanoid_once_per_result(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """Section 3.4, and the reason a load failure is now one loud error, not 13."""
    base_url, stub = app
    _capture(base_url, tmp_path, strategy="seek")
    assert stub.request_log.count("/assets/models/human-male.glb") == 1


def test_deterministic_render_mode_is_recorded_and_turns_antialiasing_off(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """Plan 05 section 6.2, resolved 2026-08-24: a second mode, never a silent default."""
    base_url, _ = app
    manifest = _capture(base_url, tmp_path, deterministic_render=True)
    provenance = manifest["render_provenance"]
    assert provenance["render_mode"] == "deterministic"
    assert provenance["antialias"] is False
    assert provenance["shadow_map"].startswith("Basic/")


def test_capture_screenshot_sources_agree(app: tuple[str, Any], tmp_path: Path) -> None:
    """Reading the drawing buffer must be the same image the compositor would hand back.

    `canvas.toDataURL` is 5.3x faster than a Playwright element screenshot, which is
    what makes plan 05's speed goal reachable -- but only if it is the same picture.
    The PNG *bytes* differ (different encoders); the pixels must not.
    """
    from PIL import Image

    base_url, _ = app
    canvas = _capture(base_url, tmp_path / "canvas", screenshot_source="canvas")
    compositor = _capture(base_url, tmp_path / "compositor", screenshot_source="compositor")
    assert len(canvas["snapshots"]) == len(compositor["snapshots"])
    for left, right in zip(canvas["snapshots"], compositor["snapshots"]):
        one = Image.open(tmp_path / "canvas" / left["path"]).convert("RGB")
        two = Image.open(tmp_path / "compositor" / right["path"]).convert("RGB")
        assert one.size == two.size == (1600, 900)
        differing = sum(
            1 for a, b in zip(one.get_flattened_data(), two.get_flattened_data()) if a != b
        )
        assert differing == 0, f"{left['id']}: {differing} pixels differ between sources"
    assert canvas["render_provenance"]["screenshot_source"] == "canvas"
    assert compositor["render_provenance"]["screenshot_source"] == "compositor"


def test_the_seek_path_is_materially_faster_than_the_page_reload(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """Plan 05 goal G4, as a relative assertion.

    An absolute per-candidate ceiling is a flaky assertion on a shared machine. The
    ratio is not: both halves render the same clip in the same process moments apart.
    Measured on this machine at 4329 ms against 8561 ms, a ratio of 0.51.
    """
    import time

    base_url, _ = app
    reload_start = time.perf_counter()
    _capture(base_url, tmp_path / "timed-reload", strategy="reload")
    reload_s = time.perf_counter() - reload_start

    seek_start = time.perf_counter()
    _capture(base_url, tmp_path / "timed-seek", strategy="seek")
    seek_s = time.perf_counter() - seek_start

    assert seek_s < reload_s * 0.8, (
        f"seek took {seek_s:.2f}s against reload's {reload_s:.2f}s "
        f"(ratio {seek_s / reload_s:.2f}); the seek hook is not paying for itself"
    )


def test_the_rendered_ego_fov_is_the_one_config_declares(
    app: tuple[str, Any], tmp_path: Path
) -> None:
    """Close the last leg of the camera-constant loop: config -> TS -> rendered pixels.

    `evals/generate_camera_ts.py --check` proves `frontend/src/generated/camera.ts` is not
    stale against `config/camera.v1.json`, and `tests/test_camera_config.py` proves the
    Python constants agree with the same file. Neither proves the *browser* used the
    value: before 08c the frontend carried its own literals, and after it the import could
    still be bypassed by re-hardcoding one.

    This renders an ego frame and reads the FOV back out of the manifest the page
    reported, so the assertion runs the whole boundary. Verified to fail by setting the
    generated constant to 80.0.
    """
    import json as _json

    base_url, _ = app
    declared = _json.loads(
        (Path(__file__).resolve().parents[1] / "config" / "camera.v1.json").read_text(
            encoding="utf-8"
        )
    )
    camera = declared["camera"]
    expected_fov = float(camera["ego_vertical_fov_deg"]["value"])
    expected_width = int(camera["capture_width_px"]["value"])
    expected_height = int(camera["capture_height_px"]["value"])

    manifest = _capture(base_url, tmp_path, views=("ego",))
    ego = [s for s in manifest["snapshots"] if s["view"] == "ego"]
    assert ego, "no ego snapshot was captured"
    for snapshot in ego:
        assert float(snapshot["camera"]["vertical_fov_deg"]) == expected_fov
        assert snapshot["width_px"] == expected_width
        assert snapshot["height_px"] == expected_height
