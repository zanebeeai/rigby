from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from evals.glb import check_glb

from rigby_poc.app import app
from rigby_poc.compiler import PROJECT_ROOT, RIG_PROFILE, compile_motion
from rigby_poc.exporter import export_glb
from rigby_poc.models import (
    CompileRequest,
    Hand,
    HandShape,
    Intent,
    ParameterOverrides,
    PlanRequest,
    PrimitiveParameters,
    PrimitiveKind,
    default_scene,
)
from rigby_poc.planner import OfflinePlanner, OpenAIPlanner, PlannerSelection, provider_status
from rigby_poc.physics import app_to_mj_position, mj_to_app_position
import rigby_poc.primitives as primitive_library
from rigby_poc.store import ResultStore

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _plan(text: str):
    scene = default_scene()
    return scene, OfflinePlanner().plan(PlanRequest(text=text, scene=scene, provider="offline")).program


def test_contracts_are_strict_and_provider_status_never_contains_key() -> None:
    with pytest.raises(ValidationError):
        PlanRequest.model_validate(
            {"text": "shaka", "scene": default_scene().model_dump(), "provider": "offline", "secret": "no"}
        )
    status = provider_status()
    assert isinstance(status["openai_available"], bool)
    assert "key" not in json.dumps(status).lower()


@pytest.mark.parametrize(
    ("prompt", "intent", "hand"),
    [
        ("Throw up a hang-ten sign.", "gesture", "right"),
        ("Give me a left-handed hang loose gesture.", "gesture", "left"),
        ("Right hand: thumb and pinky out, middle fingers curled.", "gesture", "right"),
        ("Touch the left thumb to the left index fingertip.", "gesture", "left"),
        ("Grab the block in front of you.", "grab", "right"),
        ("Collect the block in front of you with the left hand.", "grab", "left"),
    ],
)
def test_offline_planner_aliases(prompt: str, intent: str, hand: str) -> None:
    _, program = _plan(prompt)
    assert program.intent.value == intent
    assert program.hand.value == hand


@pytest.mark.parametrize(
    "prompt",
    [
        "Pick up the coffee mug.",
        "Use only your left hand and only your right hand to grab the block.",
        "Do not touch the block, but pick it up.",
        "Keep both hands closed and show an open right palm.",
    ],
)
def test_offline_planner_rejects_unsupported(prompt: str) -> None:
    _, program = _plan(prompt)
    assert program.intent.value == "unsupported"
    assert program.unsupported_reason


def test_offline_planner_supports_weight_bearing_ladder_affordance() -> None:
    scene, program = _plan("Climb the ladder for three rungs.")
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))

    assert program.intent.value == "full_body"
    assert program.primitives[0].body.action.value == "climb"
    assert program.primitives[0].body.support_object_id == "ladder"
    assert clip.success, clip.failure
    assert clip.metrics["climb_vertical_completion_fraction"] == pytest.approx(1.0)
    assert clip.metrics["climb_three_point_support_fraction"] >= 0.70


def test_hang_ten_is_computed_from_all_fifteen_clip_bones() -> None:
    scene, program = _plan("Throw up a hang-ten sign.")
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert clip.success
    assert clip.metrics["finger_assertions_computed_from_clip"] is True
    assert all(clip.metrics["finger_assertions"].values())
    assert clip.metrics["normalized_finger_curls"]["thumb"] < 0.35
    assert clip.metrics["normalized_finger_curls"]["little"] < 0.35
    assert clip.metrics["normalized_finger_curls"]["middle"] > 0.65
    assert len([key for key in clip.frames[0].bones if key.startswith("right") and any(x in key for x in ("Thumb", "Index", "Middle", "Ring", "Little"))]) == 15


def test_short_gesture_phases_are_retimed_without_discontinuities() -> None:
    scene, program = _plan("Throw up a hang-ten sign.")
    program.primitives[0].parameters.duration_s = 0.10
    program.primitives[2].parameters.duration_s = 0.10
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert clip.success
    assert clip.metrics["discontinuities"] == 0
    assert clip.metrics["max_frame_rotation_delta_rad"] <= 0.35


def test_free_body_grasp_satisfies_contact_and_safety_gates() -> None:
    scene, program = _plan("Grab the block in front of you.")
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    metrics = clip.metrics
    assert clip.success
    assert metrics["lift_height_m"] >= 0.10
    assert metrics["lost_table_contact"] is True
    assert metrics["opposing_contacts"] is True
    assert metrics["vertical_drift_m"] < 0.015
    assert metrics["palm_relative_slip_m"] < 0.020
    assert metrics["weld_used"] is False
    assert metrics["max_penetration_m"] < 0.004
    assert metrics["physics_model"]["parallel_gripper_proxy"] is False
    assert set(metrics["physics_model"]["digits"]) == {"thumb", "index", "middle", "ring", "little"}
    assert all(value == 0 for value in (metrics["joint_limit_violations"], metrics["nan_count"], metrics["discontinuities"]))


def test_grasp_uses_lift_and_hold_phase_parameters() -> None:
    scene, program = _plan("Grab the block in front of you.")
    phases = {item.kind: item for item in program.primitives}
    phases[PrimitiveKind.CLOSE].parameters.lift_height_m = 0.0
    phases[PrimitiveKind.CLOSE].parameters.hold_duration_s = 0.0
    phases[PrimitiveKind.LIFT].parameters.lift_height_m = 0.12
    phases[PrimitiveKind.HOLD].parameters.hold_duration_s = 1.1
    phases[PrimitiveKind.HOLD].parameters.duration_s = 1.1
    OpenAIPlanner._validate_semantics(program, scene)

    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))

    assert clip.success
    assert clip.metrics["lift_height_m"] >= 0.10
    assert clip.metrics["hold_duration_s"] == pytest.approx(1.1)


def test_compact_model_selection_expands_to_audited_smart_primitives() -> None:
    scene = default_scene()
    gesture_request = PlanRequest(text="Make a left shaka.", scene=scene, provider="openai")
    gesture = OpenAIPlanner._expand(
        PlannerSelection(intent=Intent.GESTURE, hand=Hand.LEFT, hand_shape=HandShape.HANG_TEN),
        gesture_request,
    )
    assert [item.kind.value for item in gesture.primitives] == ["present", "hold", "recover"]
    assert gesture.primitives[1].parameters.duration_s >= 1.0
    assert gesture.primitives[0].parameters.thumb_opposition == 0.0

    grab_request = PlanRequest(text="Grab the block.", scene=scene, provider="openai")
    grab = OpenAIPlanner._expand(
        PlannerSelection(intent=Intent.GRAB, hand=Hand.RIGHT, object_id="block"),
        grab_request,
    )
    OpenAIPlanner._validate_semantics(grab, scene)
    assert [item.kind.value for item in grab.primitives] == [
        "reach", "preshape", "contact", "close", "lift", "hold", "recover"
    ]


def test_openai_response_seed_adds_bounded_reproducible_gesture_variation() -> None:
    scene = default_scene()
    request = PlanRequest(text="Make a precise right shaka.", scene=scene, provider="openai")
    selection = PlannerSelection(intent=Intent.GESTURE, hand=Hand.RIGHT, hand_shape=HandShape.HANG_TEN)
    first = OpenAIPlanner._expand(selection, request, variation_seed=1234)
    replay = OpenAIPlanner._expand(selection, request, variation_seed=1234)
    second = OpenAIPlanner._expand(selection, request, variation_seed=5678)

    assert first.model_dump(mode="json") == replay.model_dump(mode="json")
    assert first.seed == 1234 and second.seed == 5678
    assert first.motion_profile == second.motion_profile
    assert first.primitives[0].parameters != second.primitives[0].parameters
    for program in (first, second):
        parameters = program.primitives[0].parameters
        assert -1.0 <= parameters.arm_height <= 1.0
        assert -1.0 <= parameters.arm_depth <= 1.0
        assert -1.0 <= parameters.wrist_roll <= 1.0
        assert 0.0 <= parameters.torso_participation <= 1.0


def test_model_cannot_reject_a_deterministically_supported_multiphase_gesture() -> None:
    scene = default_scene()
    request = PlanRequest(
        text=(
            'Throw up a "hang-ten" sign with your right hand, there should be a swift motion '
            "up to the main position wherein the middle three fingers are as contracted as possible, "
            "the wrist should then shake rapidly back and forth a few times, before returning to default"
        ),
        scene=scene,
        provider="openai",
    )
    program = OpenAIPlanner._expand(
        PlannerSelection(intent=Intent.UNSUPPORTED, unsupported_reason="mistaken sequence rejection"),
        request,
        variation_seed=42,
    )
    assert program.intent is Intent.GESTURE
    assert [item.kind for item in program.primitives] == [
        PrimitiveKind.PRESENT,
        PrimitiveKind.HOLD,
        PrimitiveKind.SHAKE,
        PrimitiveKind.RECOVER,
    ]
    assert program.seed == 42
    assert program.motion_profile is not None
    assert program.motion_profile.height.value == "chest"
    assert program.motion_profile.wrist_pitch.value == "neutral"

    invented = OpenAIPlanner._expand(
        PlannerSelection.model_validate(
            {
                "intent": "gesture",
                "hand": "right",
                "hand_shape": "hang_ten",
                "motion_profile": {"style": "energetic", "height": "high", "wrist_pitch": "up"},
            }
        ),
        request,
        variation_seed=43,
    )
    assert invented.motion_profile is not None
    assert invented.motion_profile.style.value == "neutral"
    assert invented.motion_profile.height.value == "chest"
    assert invented.motion_profile.wrist_pitch.value == "neutral"


def test_model_cannot_reinterpret_a_left_hook_as_a_stationary_fist() -> None:
    scene = default_scene()
    request = PlanRequest(text="throw a left hook", scene=scene, provider="openai")
    program = OpenAIPlanner._expand(
        PlannerSelection(intent=Intent.GESTURE, hand=Hand.LEFT, hand_shape=HandShape.FIST),
        request,
        variation_seed=42,
    )

    assert program.intent is Intent.STRIKE
    assert program.hand is Hand.LEFT
    assert program.strike_type.value == "hook"
    assert [item.kind.value for item in program.primitives] == [
        "guard",
        "load",
        "strike",
        "follow_through",
        "recover",
    ]
    assert all(
        item.hand_shape is HandShape.FIST
        for item in program.primitives
        if item.kind is not PrimitiveKind.RECOVER
    )


def test_parameter_only_recompile_is_monotonic_and_makes_no_model_calls() -> None:
    scene, program = _plan("Throw up a hang-ten sign.")
    low = compile_motion(
        CompileRequest(
            scene=scene,
            program=program,
            parameter_overrides=ParameterOverrides(arm_height=-0.5, index_curl=-0.5),
            persist=False,
        )
    )
    high = compile_motion(
        CompileRequest(
            scene=scene,
            program=program,
            parameter_overrides=ParameterOverrides(arm_height=0.5, index_curl=0.5),
            persist=False,
        )
    )
    assert low.slider_observables["wrist_height_m"] < high.slider_observables["wrist_height_m"]
    assert low.slider_observables["index_curl"] < high.slider_observables["index_curl"]
    assert low.provenance.model_calls == high.provenance.model_calls == 0


def test_handedness_mirrors_gesture_trajectory() -> None:
    scene, program = _plan("Throw up a hang-ten sign.")
    left = compile_motion(
        CompileRequest(scene=scene, program=program, parameter_overrides=ParameterOverrides(hand=Hand.LEFT), persist=False)
    )
    right = compile_motion(
        CompileRequest(scene=scene, program=program, parameter_overrides=ParameterOverrides(hand=Hand.RIGHT), persist=False)
    )
    assert left.slider_observables["wrist_lateral_m"] == pytest.approx(-right.slider_observables["wrist_lateral_m"])


def test_handedness_override_preserves_outward_profile_semantics() -> None:
    scene, program = _plan(
        "With your right hand, make a quick energetic shaka high and extended, held outward from the body. "
        "Pitch the wrist up, yaw it outward, and roll it clockwise."
    )
    right = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    left = compile_motion(
        CompileRequest(scene=scene, program=program, parameter_overrides=ParameterOverrides(hand=Hand.LEFT), persist=False)
    )
    assert left.slider_observables["wrist_lateral_m"] == pytest.approx(-right.slider_observables["wrist_lateral_m"])
    assert left.slider_observables["wrist_yaw_rad"] == pytest.approx(-right.slider_observables["wrist_yaw_rad"])


@pytest.mark.parametrize("hand", [Hand.LEFT, Hand.RIGHT])
def test_targets_objects_and_arm_segments_stay_in_avatar_forward_half_space(hand: Hand) -> None:
    scene = default_scene()
    shoulder = primitive_library.shoulder_position(hand)
    parameters = PrimitiveParameters(arm_height=0.5, arm_depth=0.3, elbow_swivel=0.2)
    target = primitive_library.gesture_target(hand, parameters)

    assert scene.objects[0].transform.translation.z > 0.0
    assert target.z > shoulder.z

    pose, target_distance = primitive_library.arm_pose_from_target(
        hand,
        shoulder,
        target,
        parameters,
        present_hand=True,
    )
    assert target_distance < 0.2966 + 0.2798
    rest_upper = Rotation.from_quat(primitive_library._UPPER_ARM_REST_WORLD_XYZW[hand])
    upper_world = rest_upper * Rotation.from_quat(pose[f"{hand.value}UpperArm"].as_list())
    lower_base = upper_world * Rotation.from_quat(primitive_library._LOWER_ARM_REST_LOCAL_XYZW[hand])
    lower_world = lower_base * Rotation.from_quat(pose[f"{hand.value}LowerArm"].as_list())
    assert upper_world.apply([0.0, 1.0, 0.0])[2] > 0.0
    assert lower_world.apply([0.0, 1.0, 0.0])[2] > 0.0

    program = OfflinePlanner().plan(
        PlanRequest(text=f"Grab the block with your {hand.value} hand.", scene=scene, provider="offline")
    ).program
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert clip.success
    assert clip.slider_observables["wrist_depth_m"] > shoulder.z
    compiled_target = np.asarray(
        [
            clip.slider_observables["wrist_lateral_m"],
            clip.slider_observables["wrist_height_m"],
            clip.slider_observables["wrist_depth_m"],
        ]
    )
    assert np.linalg.norm(compiled_target - np.asarray(shoulder.as_list())) < 0.2966 + 0.2798
    assert all(frame.objects["block"].translation.z > 0.0 for frame in clip.frames)
    assert clip.provenance.coordinate_frames["application"] == "glTF Y-up, forward +Z, meters"


def test_forward_coordinate_conversion_round_trips_and_behind_target_fails_cleanly() -> None:
    point = [0.17, 1.08, 0.43]
    assert mj_to_app_position(app_to_mj_position(point)) == pytest.approx(point)
    assert app_to_mj_position(point)[1] < 0.0

    scene, program = _plan("Grab the block in front of you.")
    scene.objects[0].transform.translation.z = -0.42
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert not clip.success
    assert clip.failure is not None
    assert clip.failure.code.value == "unreachable_target"
    assert clip.failure.details["forward_axis"] == "+Z"


def test_forward_target_beyond_two_link_arm_reach_fails_before_ik_clamping() -> None:
    scene, program = _plan("Grab the block in front of you.")
    scene.objects[0].transform.translation.z = 0.50
    assert scene.reachable_radius_m == pytest.approx(0.72)

    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))

    assert not clip.success
    assert clip.failure is not None
    assert clip.failure.code.value == "unreachable_target"
    assert clip.failure.details["distance_m"] < clip.failure.details["declared_scene_limit_m"]
    assert clip.failure.details["distance_m"] > clip.failure.details["effective_limit_m"]
    assert clip.failure.details["analytic_arm_reach_m"] == pytest.approx(0.2966 + 0.2798)
    assert clip.failure.details["effective_limit_m"] == pytest.approx(0.2966 + 0.2798 - 1e-4)


def test_store_contract_and_glb_reimport(tmp_path: Path) -> None:
    scene, program = _plan("Throw up a hang-ten sign.")
    request = CompileRequest(scene=scene, program=program, persist=True)
    clip = compile_motion(request)
    store = ResultStore(tmp_path / "results")
    result_id = store.persist(request, clip)
    folder = tmp_path / "results" / result_id
    assert {path.name for path in folder.iterdir()} == {
        "request.json", "scene.json", "program.json", "clip.json", "metrics.json", "provenance.json", "animation.glb"
    }
    index = json.loads((tmp_path / "results" / "index.json").read_text())
    assert index["schema_version"] == "1.0"
    assert index["next_sequence"] == 2
    raw = (folder / "animation.glb").read_bytes()
    magic, version, declared = struct.unpack_from("<4sII", raw)
    assert (magic, version, declared) == (b"glTF", 2, len(raw))
    profile = json.loads(RIG_PROFILE.read_text(encoding="utf-8"))
    aliases = {source: canonical for canonical, source in profile["bone_map"].items()}
    roundtrip = check_glb(
        raw,
        clip.model_dump(mode="json"),
        translation_tolerance=0.0001,
        rotation_tolerance=0.0001,
        node_aliases=aliases,
    )
    assert roundtrip.valid and roundtrip.compared_samples > 0
    assert roundtrip.max_translation_error_m == pytest.approx(0.0, abs=1e-7)
    assert all(frame.objects["block"].translation.z > 0.0 for frame in clip.frames)


def test_full_body_root_translation_roundtrips_through_glb_axes(tmp_path: Path) -> None:
    scene, program = _plan("Step forward with your right foot.")
    request = CompileRequest(scene=scene, program=program, persist=True)
    clip = compile_motion(request)
    assert clip.success
    assert any(
        frame.bones["hips"].position is not None
        for frame in clip.frames
    )

    store = ResultStore(tmp_path / "results")
    result_id = store.persist(request, clip)
    profile = json.loads(RIG_PROFILE.read_text(encoding="utf-8"))
    aliases = {source: canonical for canonical, source in profile["bone_map"].items()}
    roundtrip = check_glb(
        (tmp_path / "results" / result_id / "animation.glb").read_bytes(),
        clip.model_dump(mode="json"),
        translation_tolerance=0.0001,
        rotation_tolerance=0.0001,
        node_aliases=aliases,
    )

    assert roundtrip.valid and roundtrip.compared_samples > 0
    assert roundtrip.max_translation_error_m == pytest.approx(0.0, abs=1e-6)
    assert roundtrip.max_rotation_error == pytest.approx(0.0, abs=1e-6)


def test_typed_failure_persists_without_attempting_empty_glb_export(tmp_path: Path) -> None:
    scene, program = _plan("Grab the block in front of you.")
    scene.objects[0].transform.translation.z = -0.42
    request = CompileRequest(scene=scene, program=program, persist=True)
    clip = compile_motion(request)
    assert not clip.success and not clip.frames

    store = ResultStore(tmp_path / "results")
    result_id = store.persist(request, clip)
    folder = tmp_path / "results" / result_id

    assert (folder / "clip.json").is_file()
    assert not (folder / "animation.glb").exists()
    assert store.detail(result_id)["animation_url"] is None
    assert store.list()[0].success is False


def test_api_health_assets_and_offline_plan() -> None:
    client = TestClient(app)
    assert client.get("/api/v1/health").status_code == 200
    profile = client.get("/api/v1/assets/rig-profile")
    assert profile.status_code == 200
    assert len(profile.json()["finger_bones"]) == 30
    response = client.post(
        "/api/v1/plan",
        json={"text": "Throw up a hang-ten sign.", "scene": default_scene().model_dump(mode="json"), "provider": "offline"},
    )
    assert response.status_code == 200
    assert response.json()["program"]["intent"] == "gesture"
