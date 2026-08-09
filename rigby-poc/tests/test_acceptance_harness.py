from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from evals.archive import AcceptanceLedger, ResultArchive
from evals.criteria import fixture_errors, grasp_trials, load_criteria, review_template, supported_cases, unsupported_cases
from evals.evidence import (
    hangten_assertion,
    gesture_diversity_gate,
    parametric_gate,
    physical_trial,
    planner_gate,
    safety_trial,
    structural_physics_gate,
)
from evals.models import AcceptanceReport, GateResult, Status
from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, PlanRequest, default_scene
from rigby_poc.planner import OfflinePlanner


def test_acceptance_fixture_cardinality_is_exact() -> None:
    criteria = load_criteria()
    assert fixture_errors(criteria) == []
    assert len(supported_cases()) == 60
    assert len(unsupported_cases()) == 20
    assert len(review_template()["records"]) == 20
    assert len(list(grasp_trials(criteria))) == 5 * 5 * 2 * 3 == 150


def test_review_prompts_compile_to_pairwise_distinct_motion_outcomes() -> None:
    scene = default_scene()
    planner = OfflinePlanner()
    items = []
    for case in supported_cases()[:20]:
        program = planner.plan(PlanRequest(text=case["prompt"], scene=scene, provider="offline")).program
        assert program.motion_profile is not None
        assert program.motion_profile.model_dump(mode="json") == case["motion_profile"]
        clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
        assert clip.success, f"{case['id']}: {clip.failure}"
        items.append({"id": case["id"], "program": program.model_dump(mode="json"), "clip": clip.model_dump(mode="json")})

    criteria = load_criteria()["gesture_diversity"]
    result = gesture_diversity_gate(items, criteria)
    assert result.status is Status.PASS, result.failures
    assert result.measured["unique_motion_hashes"] == 20
    assert result.measured["unique_motion_descriptors"] == 20

    duplicate = deepcopy(items)
    duplicate[-1]["program"] = deepcopy(duplicate[0]["program"])
    duplicate[-1]["clip"] = deepcopy(duplicate[0]["clip"])
    assert gesture_diversity_gate(duplicate, criteria).status is Status.FAIL


def test_grasp_matrix_uses_canonical_scene_contract_and_calibrated_workspace() -> None:
    trials = list(grasp_trials(load_criteria()))
    for trial in trials:
        scene = trial["scene"]
        assert scene["rig"]["fixed_root"] is True
        block = scene["objects"][0]
        assert block["kind"] == "block"
        assert set(block["dimensions_m"]) == {"x", "y", "z"}
        assert block["sockets"][0]["id"] == "front_center"
        point = block["transform"]["translation"]
        assert -0.05 <= point["x"] <= 0.05
        assert 1.00 <= point["y"] <= 1.13
        assert 0.23 <= point["z"] <= 0.35
        hand = trial["expected"]["hand"]
        shoulder = (0.1737 if hand == "left" else -0.1737, 1.4457, -0.0652)
        # The grasp compiler targets the near face and lifts 10 cm for its final
        # visual wrist target; verify that target rather than the block center.
        wrist_target = (point["x"], point["y"] + 0.10, point["z"] - 0.035)
        distance = math.sqrt(sum((wrist_target[index] - shoulder[index]) ** 2 for index in range(3)))
        assert distance <= 0.2966 + 0.2798


def _planner_program(case: dict) -> dict:
    primitive = case["primitive"]
    if primitive == "grab":
        return {
            "intent": "grab", "hand": case["hand"],
            "primitives": [{"kind": "reach", "object_id": "block"}],
        }
    program = {
        "intent": "gesture", "hand": case["hand"],
        "primitives": [{"kind": "present", "hand_shape": primitive}],
    }
    if "motion_profile" in case:
        program["motion_profile"] = case["motion_profile"]
    return program


def test_planner_gate_requires_full_threshold_and_openai_provider() -> None:
    criteria = load_criteria()["planner"]
    supported = []
    for case in supported_cases():
        expected = {key: case[key] for key in ("intent", "hand", "primitive", "object_id")}
        if "motion_profile" in case:
            expected["motion_profile"] = case["motion_profile"]
        supported.append({"id": case["id"], "expected": expected, "body": {"program": _planner_program(case), "provider": "openai", "model": "test-model", "model_calls": 1}})
    unsupported = [{"id": case["id"], "status": 200, "body": {"unsupported_reason": case["category"], "provider": "openai", "model": "test-model", "model_calls": 1}} for case in unsupported_cases()]
    assert planner_gate(supported, unsupported, criteria, "openai").status is Status.PASS
    assert planner_gate(supported, unsupported, criteria, "offline").status is Status.UNVERIFIED
    for item in supported[:4]:
        item["body"]["program"]["hand"] = "left" if item["expected"]["hand"] == "right" else "right"
    assert planner_gate(supported, unsupported, criteria, "openai").status is Status.FAIL


def test_hangten_assertion_uses_normalized_finger_state() -> None:
    valid = {"finger_state": {"thumb": 0.1, "index": 0.8, "middle": 0.9, "ring": 0.8, "little": 0.1}}
    invalid = {"finger_state": {"thumb": 0.9, "index": 0.8, "middle": 0.9, "ring": 0.8, "little": 0.1}}
    assert hangten_assertion(valid)[0] is True
    assert hangten_assertion(invalid)[0] is False
    assert hangten_assertion({})[0] is None


def test_hangten_assertion_can_verify_raw_clip_bone_rotations() -> None:
    import math
    def rotation(angle: float) -> dict:
        return {"rotation": {"x": math.sin(angle / 2), "y": 0.0, "z": 0.0, "w": math.cos(angle / 2)}}
    clip = {"frames": [{"bones": {
        "rightThumbMetacarpal": rotation(0.2), "rightIndexProximal": rotation(1.0),
        "rightMiddleProximal": rotation(1.1), "rightRingProximal": rotation(1.0),
        "rightLittleProximal": rotation(0.25),
    }}]}
    assert hangten_assertion(clip, "right")[0] is True


def _physical_metrics() -> dict:
    return {
        "lift_height_m": 0.101,
        "hold_duration_s": 1.01,
        "vertical_drift_m": 0.014,
        "palm_relative_slip_m": 0.019,
        "lost_table_contact": True,
        "opposing_contacts": True,
        "weld_used": False,
    }


def test_physical_proof_forbids_weld_and_treats_missing_as_unverified() -> None:
    criteria = load_criteria()["physical_proof"]
    assert physical_trial(_physical_metrics(), criteria)[0] is True
    welded = _physical_metrics()
    welded["weld_used"] = True
    assert physical_trial(welded, criteria)[0] is False
    missing = _physical_metrics()
    del missing["weld_used"]
    assert physical_trial(missing, criteria)[0] is None


def test_safety_thresholds_include_all_invariants() -> None:
    criteria = load_criteria()["safety"]
    values = {
        "joint_limit_violations": 0,
        "root_drift_m": 0.001,
        "foot_drift_m": 0.001,
        "unresolved_non_hand_collisions": 0,
        "nan_count": 0,
        "discontinuities": 0,
        "max_penetration_m": 0.004,
    }
    assert safety_trial(values, criteria)[0] is True
    values["max_penetration_m"] = 0.0041
    assert safety_trial(values, criteria)[0] is False


def test_structural_gate_rejects_parallel_gripper_and_requires_axis_conversion() -> None:
    metadata = {
        "physics_model": {
            "digits": [
                {"name": name, "independently_actuated": True, "contactable": True}
                for name in ("thumb", "index", "middle", "ring", "little")
            ],
            "parallel_gripper_proxy": False,
        },
        "coordinate_frames": {
            "app_up_axis": "Y", "simulation_up_axis": "Z",
            "explicit_trajectory_conversion": True, "trajectory_roundtrip_error_m": 0.0,
        },
    }
    class Response:
        body = metadata
    items = [{"id": "s01", "response": Response()}, {"id": "g-cube", "response": Response()}]
    criteria = load_criteria()["structural_physics"]
    assert structural_physics_gate(items, criteria).status is Status.PASS
    metadata["physics_model"]["parallel_gripper_proxy"] = True
    assert structural_physics_gate(items, criteria).status is Status.FAIL


def test_all_ui_controls_are_declared_with_expected_behavior() -> None:
    config = load_criteria()["parametric_control"]
    controls = config["controls"]
    names = {control["name"] for control in controls}
    assert len(controls) == 28
    assert {"thumb_curl", "index_curl", "middle_curl", "ring_curl", "little_curl"} <= names
    assert {"block_width_m", "block_height_m", "block_depth_m", "block_mass_kg", "block_friction"} <= names
    assert {"block_x_m", "block_y_m", "block_z_m", "handedness"} <= names
    cases = []
    for control in controls:
        observed = [-0.2, 0.0, 0.2] if control["behavior"] == "increasing" else [-0.15, 0.15]
        cases.append({
            "control": control["name"], "observed": observed,
            "compile_ms": [100.0] * len(observed), "planner_calls": [0] * len(observed),
            "successful_recompiles": [True] * len(observed),
        })
    assert parametric_gate(cases, config).status is Status.PASS

    cases[0]["successful_recompiles"][-1] = False
    assert parametric_gate(cases, config).status is Status.FAIL


def test_archive_creates_exact_sequential_contract(tmp_path: Path) -> None:
    archive = ResultArchive(tmp_path / "results")
    first = archive.add(
        label="Hang ten", request={"prompt": "hang ten"}, scene={}, program={}, clip={},
        metrics={}, provenance={},
    )
    second = archive.add(
        label="Grab block", request={"prompt": "grab"}, scene={}, program={}, clip={},
        metrics={}, provenance={}, glb=b"glTF-placeholder",
    )
    assert first == "000001-hang-ten"
    assert second == "000002-grab-block"
    expected = {"request.json", "scene.json", "program.json", "clip.json", "metrics.json", "provenance.json"}
    assert expected <= {path.name for path in (tmp_path / "results" / first).iterdir()}
    assert (tmp_path / "results" / second / "animation.glb").read_bytes() == b"glTF-placeholder"


def test_certification_requires_two_consecutive_real_passes_and_excludes_synthetic(tmp_path: Path) -> None:
    ledger = AcceptanceLedger(tmp_path / "acceptance-runs")
    base = {
        "run_id": "pending", "started_at": "2026-01-01T00:00:00Z", "completed_at": "2026-01-01T00:00:01Z",
        "source": "unit-test", "passed": True, "gates": [], "artifacts": [],
    }
    synthetic = ledger.record({**base, "synthetic": True}, "synthetic", 2)
    assert synthetic["consecutive_passes"] == 0 and not synthetic["certified"]
    first = ledger.record({**base, "synthetic": False}, "first", 2)
    assert first["consecutive_passes"] == 1 and not first["certified"]
    second = ledger.record({**base, "synthetic": False}, "second", 2)
    assert second["consecutive_passes"] == 2 and second["certified"]
    failed = ledger.record({**base, "synthetic": False, "passed": False}, "failed", 2)
    assert failed["consecutive_passes"] == 0 and not failed["certified"]


def test_synthetic_acceptance_report_can_never_pass() -> None:
    report = AcceptanceReport(
        run_id="test", started_at="now", completed_at="now", source="unit-test", synthetic=True,
        gates=[GateResult("all", Status.PASS, "synthetic")],
    )
    assert report.passed is False
