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


def test_the_analysis_eye_offset_matches_the_config() -> None:
    """The fifth camera value, and the last one that was free to drift.

    The other four reached the frontend by generation while this one stayed
    hand-typed on BOTH sides, and the copies diverged: the analysis side had
    folded it into a rest head position rounded to four decimals, putting its
    implied offset 0.024 mm from the renderer's. That is 0.028 px of a 1600x900
    capture, and it decided ``active_hand_visibility_fraction`` at frame 0 of
    every strike and gesture clip, because the compiler places the hand at the
    visibility limit by construction.
    """

    import numpy as np

    from rigby_poc.analysis.rig import EGO_EYE_OFFSET_M

    authored = np.asarray(_value("ego_eye_offset_m"), dtype=float)
    assert np.allclose(EGO_EYE_OFFSET_M, authored, atol=0.0, rtol=0.0)


def test_the_frontend_eye_offset_is_the_generated_one_not_a_retyped_copy() -> None:
    """`camera.ts` must import the offset, not hand-type it again.

    Pinning the two values equal is not enough on its own: a hand-typed literal
    that happens to match today is exactly what the last four values looked like
    before they drifted. The import is the mechanism; this asserts it exists.
    """

    source = (PROJECT_ROOT / "frontend" / "src" / "camera.ts").read_text(encoding="utf-8")

    assert "egoEyeOffsetM" in source, (
        "frontend/src/camera.ts no longer reads the generated eye offset. It must "
        "import egoEyeOffsetM from ./generated/camera rather than retyping it."
    )
    assert "new THREE.Vector3(0, 0.04, 0.11)" not in source, (
        "frontend/src/camera.ts has a retyped eye-offset literal again."
    )


def test_the_rest_ego_camera_is_derived_from_the_rig_not_a_rounded_literal() -> None:
    """The derived rest camera must sit where the renderer puts it.

    ``rest_ego_camera_position`` composes the config offset onto the rig's
    measured rest head. The literal it replaced carried the same sum with the
    head rounded to 4 dp; this asserts the derivation reproduces the renderer's
    construction rather than that older rounding.
    """

    import numpy as np

    from rigby_poc.analysis.gesture import rest_ego_camera_position
    from rigby_poc.analysis.rig import EGO_EYE_OFFSET_M, identity_bones
    from rigby_poc.kinematics import rig_kinematics

    kinematics = rig_kinematics()
    head = kinematics.world_matrices(identity_bones())[
        kinematics.node_by_canonical["head"]
    ]
    # The renderer rotates the offset by the head delta, identity at rest.
    expected = head[:3, 3] + EGO_EYE_OFFSET_M

    assert np.allclose(rest_ego_camera_position(), expected, atol=0.0, rtol=0.0)

    superseded = np.asarray([0.0, 1.5685 + 0.04, 0.0114 + 0.11], dtype=float)
    assert not np.allclose(rest_ego_camera_position(), superseded, atol=1e-9, rtol=0.0), (
        "the derived rest camera equals the rounded literal it superseded, which "
        "means the derivation is not reading the rig's measured rest head"
    )


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
