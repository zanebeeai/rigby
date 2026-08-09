from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evals.criteria import load_criteria
from evals.models import Status
from evals.orientation import camera_orientation_gate, inspect_egocentric_camera_source, semantic_forward_gate
from evals.runner import DEFAULT_SCENE


ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / "config" / "rig_profiles" / "mesh2motion-human-vrm1.json").read_text(encoding="utf-8"))
CRITERIA = load_criteria()


def _gesture(z: float) -> dict:
    observables = {"wrist_lateral_m": -0.12, "wrist_height_m": 1.36, "wrist_depth_m": z}
    return {
        "id": "s01", "program": {"intent": "gesture"}, "scene": DEFAULT_SCENE,
        "clip": {"slider_observables": observables},
        "response": SimpleNamespace(body={"clip": {"slider_observables": observables}}),
    }


def _grasp(z: float) -> dict:
    scene = json.loads(json.dumps(DEFAULT_SCENE))
    scene["objects"][0]["transform"]["translation"]["z"] = z
    transform = {"translation": {"x": 0.0, "y": 1.05, "z": z}, "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}}
    return {
        "id": "g01", "program": {"intent": "grab", "primitives": [{"object_id": "block"}]}, "scene": scene,
        "clip": {
            "frames": [{"objects": {"block": transform}}],
            "slider_observables": {"wrist_lateral_m": 0.0, "wrist_height_m": 1.05, "wrist_depth_m": z},
        },
    }


def test_semantic_forward_gate_accepts_positive_rig_forward_half_space() -> None:
    result = semantic_forward_gate([_gesture(0.33), _grasp(0.29)], PROFILE, CRITERIA["semantic_forward_space"])
    assert result.status is Status.PASS


def test_semantic_forward_gate_rejects_the_previous_negative_z_scene_and_motion() -> None:
    result = semantic_forward_gate([_gesture(-0.35), _grasp(-0.42)], PROFILE, CRITERIA["semantic_forward_space"])
    assert result.status is Status.FAIL
    assert len(result.failures) == 2
    assert all(value < 0 for value in result.measured["projections_m"].values())


def test_legacy_negative_z_acceptance_scene_is_detected_as_behind_the_rig() -> None:
    result = semantic_forward_gate([_grasp(-0.42)], PROFILE, CRITERIA["semantic_forward_space"])
    assert result.status is Status.FAIL


def test_default_acceptance_scene_is_in_front_of_the_rig() -> None:
    result = semantic_forward_gate([_grasp(DEFAULT_SCENE["objects"][0]["transform"]["translation"]["z"])], PROFILE, CRITERIA["semantic_forward_space"])
    assert result.status is Status.PASS


def test_positive_z_target_beyond_calibrated_arm_length_is_rejected() -> None:
    result = semantic_forward_gate([_grasp(0.48)], PROFILE, CRITERIA["semantic_forward_space"])
    assert result.status is Status.FAIL
    assert result.measured["reach_distances_m"]["g01"] > result.measured["max_arm_reach_m"]
    assert any("arm reach" in failure for failure in result.failures)


def test_camera_gate_accepts_aligned_view_and_rejects_reversed_view() -> None:
    config = CRITERIA["egocentric_camera_orientation"]
    assert camera_orientation_gate({"vectors": [[0, -0.2, 1], [0, -0.1, 1]], "source_sha256": "test"}, PROFILE, config).status is Status.PASS
    assert camera_orientation_gate({"vectors": [[0, -0.2, -1], [0, -0.1, -1]], "source_sha256": "test"}, PROFILE, config).status is Status.FAIL


def test_current_egocentric_camera_source_satisfies_forward_orientation_after_fix() -> None:
    evidence = inspect_egocentric_camera_source(ROOT / "frontend" / "src" / "scene.ts")
    result = camera_orientation_gate(evidence, PROFILE, CRITERIA["egocentric_camera_orientation"])
    assert result.status is Status.PASS
    assert len(result.measured["forward_dots"]) >= 2
    assert all(dot > 0 for dot in result.measured["forward_dots"])
