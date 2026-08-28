"""`config/camera.v1.json` is the source; the frontend copy is generated.

Plan 08 §3.3, G5. FOV 94.0 had six hand-maintained copies spanning Python and
TypeScript, 1600x900 six, and the neutral gaze three. Nothing stopped one of
them drifting, and a divergence there means evidence captured at one FOV judged
against the contract for another.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.generate_camera_ts import CAMERA_CONFIG, GENERATED_TS, main, render

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _camera() -> dict:
    return json.loads(CAMERA_CONFIG.read_text(encoding="utf-8"))["camera"]


def _value(name: str):
    return _camera()[name]["value"]


def test_the_generated_frontend_file_is_not_stale() -> None:
    """The gate. A hand-edit or an un-regenerated JSON change fails here."""

    assert GENERATED_TS.exists(), "run: uv run python -m evals.generate_camera_ts"
    assert GENERATED_TS.read_text(encoding="utf-8") == render(), (
        f"{GENERATED_TS.relative_to(PROJECT_ROOT)} is stale. "
        "Run: uv run python -m evals.generate_camera_ts"
    )


def test_the_check_flag_agrees_with_the_test() -> None:
    """The ``--check`` flag must not be able to disagree with the suite.

    In-process, not a subprocess. This file is `fast`, and `fast` is defined as
    "no compile, no pipeline, no corpus, no subprocess" -- a spawn here was the
    one violation of that contract in the tier, and `test_markers_complete.py`
    could not see it because it checks that a marker exists, not that it is
    honest. ``main`` returns the exit code, so nothing is lost by calling it.
    """

    assert main(["--check"]) == 0


def test_the_generated_file_says_it_is_generated() -> None:
    text = GENERATED_TS.read_text(encoding="utf-8")
    assert "DO NOT EDIT" in text
    assert "config/camera.v1.json" in text
    assert "evals.generate_camera_ts" in text


@pytest.mark.parametrize("name", sorted(json.loads(CAMERA_CONFIG.read_text())["camera"]))
def test_every_camera_entry_carries_a_unit_and_a_source(name: str) -> None:
    entry = _camera()[name]
    assert entry.get("unit"), f"{name} has no unit"
    source = entry.get("source")
    assert source, f"{name} has no source"
    assert source.get("kind") in {"measured", "invariant", "external", "provisional"}
    assert source.get("cites"), f"{name} cites nothing"


def test_the_python_capture_constants_match_the_config() -> None:
    """The Python half of the boundary, pinned to the same source."""

    from evals.capture import CAPTURE_FOV_DEG, CAPTURE_HEIGHT, CAPTURE_WIDTH

    assert CAPTURE_FOV_DEG == _value("ego_vertical_fov_deg")
    assert CAPTURE_WIDTH == _value("capture_width_px")
    assert CAPTURE_HEIGHT == _value("capture_height_px")


def test_the_analysis_neutral_gaze_matches_the_config() -> None:
    """`analysis/rig.py` normalises at use, so compare the normalised direction."""

    import numpy as np

    from rigby_poc.analysis.rig import EGO_NEUTRAL_GAZE

    authored = np.asarray(_value("ego_neutral_gaze"), dtype=float)
    authored = authored / np.linalg.norm(authored)
    assert np.allclose(EGO_NEUTRAL_GAZE, authored, atol=0.0, rtol=0.0)


def test_the_quality_reference_camera_contract_matches_the_config() -> None:
    """`motion_quality_reference.json` still carries a copy; it must agree."""

    contract = json.loads(
        (PROJECT_ROOT / "config" / "motion_quality_reference.json").read_text(encoding="utf-8")
    )["camera_contract"]
    assert contract["vertical_fov_deg"] == _value("ego_vertical_fov_deg")
    assert contract["width_px"] == _value("capture_width_px")
    assert contract["height_px"] == _value("capture_height_px")
    assert contract["hand_visibility_radius_m"] == _value("hand_visibility_radius_m")


def test_the_acceptance_criteria_fov_matches_the_config() -> None:
    criteria = json.loads(
        (PROJECT_ROOT / "acceptance_criteria.yaml").read_text(encoding="utf-8")
    )
    assert criteria["autonomous_pipeline"]["egocentric_full_fov_deg"] == _value(
        "ego_vertical_fov_deg"
    )


def test_the_aspect_ratio_is_exactly_sixteen_by_nine() -> None:
    """A drift here silently letterboxes evidence rather than failing."""

    width = _value("capture_width_px")
    height = _value("capture_height_px")
    assert width * 9 == height * 16, f"{width}x{height} is not 16:9"
