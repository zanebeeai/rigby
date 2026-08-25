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
    safety_gate,
    safety_trial,
    structural_physics_gate,
)
from evals.models import AcceptanceReport, GateResult, Status
from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, PlanRequest, default_scene
from rigby_poc.planner import OfflinePlanner

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _safety_metrics() -> dict:
    """A clip whose every safety invariant was actually measured.

    The two witness keys are what say so: ``physics_engine`` is emitted only by
    the MuJoCo grasp trial (``physics.py:273``) and ``self_collision_frames``
    only by the structure analysis ``analysis/hand.py:207`` reads. Without them
    the corresponding values are ``compiler._base_metrics`` seeds -- see
    ``evidence.SEEDED_SAFETY_FIELDS``.
    """

    return {
        "joint_limit_violations": 0,
        "root_drift_m": 0.001,
        "unresolved_non_hand_collisions": 0,
        "self_collision_frames": 0,
        "nan_count": 0,
        "discontinuities": 0,
        "max_penetration_m": 0.004,
        "physics_engine": "MuJoCo",
    }


def test_safety_thresholds_include_all_invariants() -> None:
    """The six measurable invariants still gate, and still gate together.

    The verdict is UNVERIFIED rather than PASS because `foot_drift_m` is never
    measured on any path -- see
    `test_the_safety_gate_cannot_certify_a_clip_while_foot_drift_is_unmeasured`.
    A breach must still be *reported* under that verdict, or removing the
    guaranteed pass would have bought silence at a different address.
    """

    criteria = load_criteria()["safety"]
    clean_result, clean_reasons = safety_trial(_safety_metrics(), criteria)
    assert clean_result is None
    assert not [reason for reason in clean_reasons if "exceeds" in reason], clean_reasons

    for field, breaching in (
        ("max_penetration_m", 0.0041),
        ("nan_count", 1),
        ("discontinuities", 1),
        ("joint_limit_violations", 1),
        ("root_drift_m", 10.0),
        ("unresolved_non_hand_collisions", 1),
    ):
        breached = _safety_metrics()
        breached[field] = breaching
        _, reasons = safety_trial(breached, criteria)
        assert any(reason.startswith(f"{field}=") for reason in reasons), (field, reasons)


def test_the_safety_gate_cannot_certify_a_clip_while_foot_drift_is_unmeasured() -> None:
    """The consequence, stated as a test rather than left to be discovered.

    `safety_and_quality` published "N/M clips passed every safety invariant"
    while one of those invariants was the literal `0.0`. It cannot honestly say
    PASS again until forward kinematics measures foot drift, so the gate reports
    UNVERIFIED for every clip and this test is what makes that deliberate.
    Deleting it means deciding to publish a narrower claim -- a product
    decision, filed as a CONDUCTOR line, not a threshold to tune.
    """

    criteria = load_criteria()["safety"]
    result = safety_gate(
        [{"id": "clean", "metrics": _safety_metrics()}], criteria
    )
    assert result.status is Status.UNVERIFIED
    assert result.measured["missing_evidence"] == 1
    assert any("foot_drift_m" in failure for failure in result.failures), result.failures


def test_a_seeded_safety_value_is_unverified_rather_than_a_guaranteed_pass() -> None:
    """Plan 08 §1.1 item 3. Present, finite, and not a measurement.

    Each of these three reached the comparison as a seeded constant, so the
    dimension read as PASS on every clip while nothing had looked. The gate must
    now say UNVERIFIED, and it must say which field and why -- a bare ``None``
    is the same silence in a different place.
    """

    criteria = load_criteria()["safety"]
    for field, witness in (
        ("max_penetration_m", "physics_engine"),
        ("unresolved_non_hand_collisions", "self_collision_frames"),
    ):
        seeded = _safety_metrics()
        del seeded[witness]
        result, reasons = safety_trial(seeded, criteria)
        assert result is None, f"{field} gated on a seeded constant"
        assert any(field in reason and "not measured" in reason for reason in reasons), reasons

    drifting = _safety_metrics()
    drifting["foot_drift_m"] = 0.0
    result, reasons = safety_trial(drifting, criteria)
    assert result is None
    assert any("foot_drift_m" in reason and "not measured" in reason for reason in reasons), reasons


def test_a_measured_breach_is_still_reported_under_an_unverified_verdict() -> None:
    """Two independent problems must not collapse into one message.

    A clip can both fail an invariant that WAS measured and carry one that was
    not. Reporting only the second is how a real failure disappears behind an
    UNVERIFIED badge.
    """

    criteria = load_criteria()["safety"]
    both = _safety_metrics()
    del both["physics_engine"]
    both["nan_count"] = 3
    result, reasons = safety_trial(both, criteria)
    assert result is None
    assert any("max_penetration_m" in reason and "not measured" in reason for reason in reasons), reasons
    assert any(reason.startswith("nan_count=3") for reason in reasons), reasons


def test_foot_drift_is_not_compared_at_all_because_no_path_measures_it() -> None:
    """The comparison is gone, not merely unreachable.

    A value that breaches ``max_foot_drift_m`` by three orders of magnitude must
    still come back UNVERIFIED. If it came back ``False`` the gate would be
    reporting a failure it did not measure, which is the mirror of the defect.
    """

    criteria = load_criteria()["safety"]
    breaching = _safety_metrics()
    breaching["foot_drift_m"] = criteria["max_foot_drift_m"] * 1000.0
    assert safety_trial(breaching, criteria)[0] is None


def test_seeded_safety_fields_are_still_seeded_in_committed_compiler_output() -> None:
    """The ledger must not go stale into the opposite error.

    ``SEEDED_SAFETY_FIELDS`` claims a producer never writes these. If one starts
    computing a real value, that claim silently becomes the new defect -- a
    measurement discarded as a constant. The evidence is 35 committed
    ``compiler_metrics`` blocks covering walk, run, burpee, squat and push-up,
    which move the feet as much as anything this compiler emits.

    Asserted non-empty first: an ``all(...)`` over a moved or renamed fixture
    directory is True, and greenness would then mean the scan found nothing.
    """

    fixtures = sorted((PROJECT_ROOT / "tests" / "fixtures" / "analysis_equivalence").glob("*.json"))
    assert len(fixtures) >= 30, f"fixture corpus is missing or moved: {len(fixtures)} cases"

    measured_foot_drift = {}
    measured_penetration = {}
    for path in fixtures:
        metrics = json.loads(path.read_text(encoding="utf-8")).get("compiler_metrics") or {}
        if "foot_drift_m" in metrics and metrics["foot_drift_m"] != 0.0:
            measured_foot_drift[path.name] = metrics["foot_drift_m"]
        if "physics_engine" not in metrics and metrics.get("max_penetration_m", 0.0) != 0.0:
            measured_penetration[path.name] = metrics["max_penetration_m"]

    assert not measured_foot_drift, (
        "foot_drift_m is no longer a seeded constant -- something measures it now. "
        f"Remove it from evidence.SEEDED_SAFETY_FIELDS: {measured_foot_drift}"
    )
    assert not measured_penetration, (
        "max_penetration_m is non-zero on a clip that ran no physics trial: "
        f"{measured_penetration}"
    )


def test_no_live_producer_measures_foot_drift() -> None:
    """The producer-side half, and the fixture scan is not a substitute for it.

    `test_seeded_safety_fields_are_still_seeded_in_committed_compiler_output`
    reads committed fixtures, which lag the producers until someone re-blesses
    them -- so it cannot fail *at the moment a producer changes*, which is the
    moment that matters. Caught by lane `analysis`, who pointed out that reading
    a fixture-scan's green as evidence about a producer is the same shape both
    of us keep finding in other people's guards.

    This compiles through the real path instead. If forward kinematics ever
    lands and starts computing foot drift, this goes red and
    `evidence.SEEDED_SAFETY_FIELDS` must drop its `foot_drift_m` entry -- which
    is the opposite error, a real measurement discarded as a constant.
    """

    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text="Walk forward four steps.", scene=scene, provider="offline")
    ).program
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert clip.success, clip.failure
    assert clip.frames, "a clip with no frames cannot evidence anything about foot drift"
    assert "foot_drift_m" in clip.metrics, (
        "foot_drift_m vanished from the metrics dict; SEEDED_SAFETY_FIELDS keys on it"
    )
    assert clip.metrics["foot_drift_m"] == 0.0, (
        f"a producer now measures foot drift ({clip.metrics['foot_drift_m']}). "
        "Remove it from evidence.SEEDED_SAFETY_FIELDS -- the safety gate should "
        "start certifying against it again rather than reporting UNVERIFIED."
    )


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
