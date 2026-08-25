"""The judge's half of the evidence-manifest contract with lane `capture`.

PR 05 split the snapshot hash in two (`pixel_sha256` from the render,
`pose_sha256` from the compiled clip), added top-level `render_provenance`, and
moved `schema_version` to 1.1 — while keeping `sha256` as an alias because
`judge._manifest` verifies against it.

Lane `capture` has a test that runs `judge._manifest()` against a freshly
produced manifest. This is the mirror, asserted from the consuming side. Two
independent assertions of one cross-lane contract is the right amount for the
part that breaks on Windows, and a contract verified only by its producer is
verified by the party with the least incentive to find a mismatch.

The boundary is worth stating: this file pins the shape the judge requires and
accepts. It does not prove capture emits it — that is capture's e2e test, which
uses a real render.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from rigby_poc.judge import _manifest

pytestmark = pytest.mark.medium


CAPTURE_CONTRACT_KEYS = {
    "raw_canvas_only",
    "width_px",
    "height_px",
    "egocentric_vertical_fov_deg",
    "device_pixel_ratio",
    "ui_overlay_included",
}


def _write(tmp_path: Path, *, snapshot_overrides: dict | None = None, **manifest_overrides) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for index, view in enumerate(("ego", "orbit"), start=1):
        buffer = io.BytesIO()
        Image.new("RGB", (1600, 900), color=(10 * index, 30, 60)).save(buffer, format="PNG")
        image = buffer.getvalue()
        (tmp_path / f"{view}.png").write_bytes(image)
        digest = hashlib.sha256(image).hexdigest()
        snapshot = {
            "id": f"01-{view}",
            "phase": "hold",
            "label": "presented_pose",
            "view": view,
            "requested_time_s": 0.5,
            "rendered_time_s": 0.5,
            "path": f"{view}.png",
            "sha256": digest,
            "pixel_sha256": digest,
            "pose_sha256": "0" * 64,
            "width_px": 1600,
            "height_px": 900,
            "camera": {
                "view": view,
                "width_px": 1600,
                "height_px": 900,
                "vertical_fov_deg": 94 if view == "ego" else 46,
            },
        }
        snapshot.update(snapshot_overrides or {})
        snapshots.append(snapshot)
    payload = {
        "schema_version": "1.1",
        "result_id": "contract-result",
        "intent": "gesture",
        "prompt": "Throw up a hang-ten sign.",
        "render_provenance": {"renderer": "test", "asset_sha256": "a" * 64},
        "capture_contract": {
            "raw_canvas_only": True,
            "width_px": 1600,
            "height_px": 900,
            "egocentric_vertical_fov_deg": 94.0,
            "device_pixel_ratio": 1,
            "ui_overlay_included": False,
        },
        "snapshots": snapshots,
    }
    payload.update(manifest_overrides)
    path = tmp_path / "evidence-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_05_era_manifest_parses(tmp_path: Path) -> None:
    manifest, snapshots = _manifest(_write(tmp_path))
    assert manifest["schema_version"] == "1.1"
    assert len(snapshots) == 2


def test_the_capture_contract_allowlist_is_unchanged(tmp_path: Path) -> None:
    manifest, _ = _manifest(_write(tmp_path))
    assert set(manifest["capture_contract"]) == CAPTURE_CONTRACT_KEYS


def test_render_provenance_is_top_level_not_inside_the_camera_dict(tmp_path: Path) -> None:
    # `_manifest` validates the shape of `snapshot["camera"]`, so provenance
    # living there would have been rejected.
    manifest, snapshots = _manifest(_write(tmp_path))
    assert "render_provenance" in manifest
    assert "render_provenance" not in snapshots[0]["camera"]


def test_the_sha256_alias_equals_the_pixel_hash_while_it_exists(tmp_path: Path) -> None:
    """Written conditionally on purpose.

    `sha256` is the alias and `pixel_sha256` is the survivor. Asserting the pair
    unconditionally would turn this red on the PR in this lane that eventually
    drops the alias, which is the opposite of what a contract test should do.
    """
    _, snapshots = _manifest(_write(tmp_path))
    assert snapshots
    for snapshot in snapshots:
        if "sha256" in snapshot:
            assert snapshot["sha256"] == snapshot["pixel_sha256"]


def test_the_pose_hash_is_carried_and_is_not_the_pixel_hash(tmp_path: Path) -> None:
    # They answer different questions: `pose_sha256` is machine-independent and
    # survives a re-render; `pixel_sha256` does not.
    _, snapshots = _manifest(_write(tmp_path))
    assert snapshots
    for snapshot in snapshots:
        assert len(snapshot["pose_sha256"]) == 64
        assert snapshot["pose_sha256"] != snapshot["pixel_sha256"]


@pytest.mark.parametrize(
    "path_value",
    [
        "sub/ego.png",
        "sub\\ego.png",
        "../ego.png",
        "/tmp/ego.png",
        "C:\\evidence\\ego.png",
    ],
)
def test_a_snapshot_path_that_is_not_a_bare_filename_is_refused(
    tmp_path: Path, path_value: str
) -> None:
    """The one contract term that breaks on exactly one platform.

    A Windows drive letter or backslash in `path` resolves differently on POSIX
    than on Windows, so this is asserted judge-side as well as capture-side
    rather than relying on the resolve check happening to reject it.
    """
    with pytest.raises(ValueError):
        _manifest(_write(tmp_path, snapshot_overrides={"path": path_value}))


def test_a_pixel_hash_mismatch_is_still_refused(tmp_path: Path) -> None:
    # The alias split must not have weakened the check it exists to feed.
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        _manifest(_write(tmp_path, snapshot_overrides={"sha256": "b" * 64}))
