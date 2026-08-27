from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import numpy as np

import pytest

from rigby_core.hashing import canonical_json_bytes, hash_file
from rigby_v2.library import SIGLIP2_SPEC
from scripts.v2.run_siglip2_canary import _load_verified_frame, _write_new

pytestmark = pytest.mark.fast


PNG = b"\x89PNG\r\n\x1a\nverified-canary-frame"
ROOT = Path(__file__).resolve().parents[1]


def test_canary_frame_is_content_bound_and_confined(tmp_path) -> None:
    archive_path = tmp_path / "frames.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("frames/000150.png", PNG)

    value, digest = _load_verified_frame(
        archive_path,
        hash_file(archive_path),
        "frames/000150.png",
    )

    assert value == PNG
    assert len(digest) == 64
    with pytest.raises(ValueError, match="content-addressed"):
        _load_verified_frame(archive_path, "0" * 64, "frames/000150.png")
    with pytest.raises(ValueError, match="canonical"):
        _load_verified_frame(archive_path, hash_file(archive_path), "../escape.png")


def test_canary_evidence_is_non_overwriting_and_sealed(tmp_path) -> None:
    output = tmp_path / "canary.json"
    digest = _write_new(output, {"status": "passed", "vector": [0.0, 1.0]})

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert output.with_suffix(".json.sha256").read_text(encoding="ascii").strip() == digest
    with pytest.raises(FileExistsError):
        _write_new(output, {"status": "changed"})


def test_sealed_real_siglip2_canary_is_repeat_identical_and_normalized() -> None:
    path = ROOT / "assets" / "v2" / "library" / "siglip2_keyframe_canary.v2.json"
    seal = path.with_suffix(path.suffix + ".sha256")
    document = json.loads(path.read_text(encoding="utf-8"))
    vector = tuple(float(value) for value in document["inference"]["vector"])

    assert hash_file(path) == seal.read_text(encoding="ascii").strip()
    assert document["status"] == "passed"
    assert document["model"]["revision"] == SIGLIP2_SPEC.revision
    assert document["model"]["snapshot_tree_sha256"] == (
        "6ea3a0eb94d36c6968e9ae10dcfadc76e88b11c937e15134c69cdc1419bd9bc8"
    )
    assert document["source"]["kind"] == "mujoco_task_closeup_raw_frame"
    assert len(vector) == SIGLIP2_SPEC.dimensions
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)
    assert document["inference"]["maximum_repeat_delta"] == 0.0
    assert document["inference"]["vector_sha256"] == hashlib.sha256(
        canonical_json_bytes(vector)
    ).hexdigest()
