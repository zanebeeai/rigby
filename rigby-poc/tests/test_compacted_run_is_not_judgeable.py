"""A compacted run fails as archival, not as a missing file.

PR 01d transcodes a run's evidence PNGs to WebP and deletes the originals, so
every snapshot path in a compacted manifest is genuinely missing. `judge.py`
would have said "snapshot path escapes or is missing from the evidence
directory" — true, and in the vocabulary of a capture bug, which is where it
would have sent the reader. The run is archival by decision.

This is the class this lane catalogued from the other side: not a guard that
cannot fire, but **damage reporting in the wrong subsystem's language**. The tell
is the same — the report names a subsystem nobody changed.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from evals.compact_run import is_compacted
from rigby_poc.judge import CompactedEvidenceError, _manifest

pytestmark = pytest.mark.medium


def _manifest_file(tmp_path: Path, *, compaction: dict | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for index, view in enumerate(("ego", "orbit"), start=1):
        buffer = io.BytesIO()
        Image.new("RGB", (1600, 900), color=(5 * index, 11, 23)).save(buffer, format="PNG")
        image = buffer.getvalue()
        if compaction is None:
            (tmp_path / f"{view}.png").write_bytes(image)
        snapshots.append(
            {
                "id": f"01-{view}",
                "phase": "hold",
                "label": "presented_pose",
                "view": view,
                "requested_time_s": 0.5,
                "rendered_time_s": 0.5,
                "path": f"{view}.png",
                "sha256": hashlib.sha256(image).hexdigest(),
                "width_px": 1600,
                "height_px": 900,
                "camera": {
                    "view": view,
                    "width_px": 1600,
                    "height_px": 900,
                    "vertical_fov_deg": 94 if view == "ego" else 46,
                },
            }
        )
    payload: dict = {
        "schema_version": "1.1",
        "result_id": "compaction-result",
        "intent": "gesture",
        "prompt": "Throw up a hang-ten sign.",
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
    if compaction is not None:
        payload["compaction"] = compaction
    path = tmp_path / "evidence-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_compacted_run_fails_as_archival(tmp_path: Path) -> None:
    path = _manifest_file(tmp_path, compaction={"judgeable": False, "format": "webp"})
    with pytest.raises(CompactedEvidenceError, match="archival and not judgeable"):
        _manifest(path)


def test_it_does_not_fail_as_a_missing_snapshot_path(tmp_path: Path) -> None:
    """The whole point: the PNGs really are gone, so the path-layer message is
    true. A true statement in the wrong vocabulary sends the reader to look for
    a capture bug in a run nobody broke."""
    path = _manifest_file(tmp_path, compaction={"judgeable": False})
    with pytest.raises(ValueError) as raised:
        _manifest(path)
    assert "escapes or is missing" not in str(raised.value)


def test_an_uncompacted_run_still_parses(tmp_path: Path) -> None:
    manifest, snapshots = _manifest(_manifest_file(tmp_path))
    assert len(snapshots) == 2
    assert "compaction" not in manifest


def test_the_error_is_catchable_as_a_value_error(tmp_path: Path) -> None:
    # Existing handlers keep working; the subclass only lets a caller that wants
    # to skip archived runs do so without matching on a message string.
    path = _manifest_file(tmp_path, compaction={"judgeable": False})
    with pytest.raises(ValueError):
        _manifest(path)


def test_the_judge_predicate_agrees_with_is_compacted(tmp_path: Path) -> None:
    """`judge.py` keys off the manifest rather than importing `evals.compact_run`,
    because `src/` should not depend on `evals/` and the artifact-level key is
    what travels with the evidence. This pins the two definitions together so
    they cannot drift apart while both look correct.
    """
    for compaction in ({"judgeable": False}, {"judgeable": False, "format": "webp"}, None):
        path = _manifest_file(tmp_path / str(compaction), compaction=compaction)
        payload = json.loads(path.read_text(encoding="utf-8"))
        judge_says_compacted = True
        try:
            _manifest(path)
            judge_says_compacted = False
        except CompactedEvidenceError:
            pass
        assert judge_says_compacted == is_compacted(payload)


def test_a_non_dict_compaction_key_is_not_treated_as_compacted(tmp_path: Path) -> None:
    # `is_compacted` requires a dict; matching that exactly is what keeps the two
    # predicates in step. A truthy string here must not short-circuit into "archival".
    path = _manifest_file(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["compaction"] = "yes"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert is_compacted(payload) is False
    _manifest(path)  # parses normally rather than raising
