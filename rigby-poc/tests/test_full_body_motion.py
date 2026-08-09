from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evals.capture import phase_sampling_points
from evals.flywheel import apply_recipe, apply_repair, candidate_recipes
from rigby_poc.compiler import PROJECT_ROOT, compile_motion
from rigby_poc.exporter import export_glb
from rigby_poc.judge import RepairPatch
from rigby_poc.kinematics import rig_kinematics
from rigby_poc.models import (
    BodyAction,
    BodyObstacleMode,
    BodyPoseTarget,
    BodyRotationAxis,
    BodyRotationMode,
    CompileRequest,
    Hand,
    Intent,
    ParameterOverrides,
    PlanRequest,
    Transform,
    Vec3,
    default_scene,
)
from rigby_poc.planner import (
    GenericBodySegmentSelection,
    OfflinePlanner,
    OpenAIPlanner,
    PlannerSelection,
)


def _program(prompt: str):
    scene = default_scene()
    outcome = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    )
    return scene, outcome.program


def _glb_json(path):
    return _glb_parts(path)[0]


def _glb_parts(path):
    raw = path.read_bytes()
    offset = 12
    document = None
    binary = b""
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \x00"))
        elif kind == 0x004E4942:
            binary = payload
        offset += 8 + length
    if document is None:
        raise AssertionError("GLB JSON chunk missing")
    return document, binary


def _float_accessor(document, binary, accessor_index):
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    components = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    values = np.frombuffer(
        binary,
        dtype=np.float32,
        count=int(accessor["count"]) * components,
        offset=offset,
    )
    return values.reshape(int(accessor["count"]), components)


def test_walk_four_steps_authors_mobile_root_and_leg_cycles() -> None:
    scene, program = _program("walk forward four steps")
    clip = compile_motion(CompileRequest(scene=scene, program=program))

    assert program.intent == Intent.FULL_BODY
    assert program.primitives[0].body.cycles == 4.0
    assert clip.success, clip.failure
    assert clip.metrics["root_displacement_m"] > 1.8
    assert clip.metrics["joint_limit_violations"] == 0
    assert clip.metrics["discontinuities"] == 0
    assert clip.metrics["final_support_center_offset_m"] < 0.05
    assert clip.metrics["final_foot_ground_error_m"] < 0.001
    assert clip.metrics["final_root_vertical_speed_m_s"] < 0.001
    assert clip.metrics["ground_penetration_m"] == 0.0
    assert clip.metrics["max_support_foot_target_error_m"] < 0.001
    assert clip.metrics["max_support_foot_slide_per_frame_m"] < 0.001
    assert clip.metrics["support_contact_fraction"] == 1.0
    assert clip.metrics["airborne_frame_count"] == 0
    assert clip.frames[-1].bones["hips"].position.z > 1.8
    assert len(
        {
            frame.bones["leftUpperLeg"].rotation.model_dump_json()
            for frame in clip.frames
        }
    ) > 8


def test_turn_crouch_jump_kick_and_step_sequences_compile() -> None:
    prompts = (
        "run forward three steps",
        "turn left 90 degrees",
        "crouch down then jump twice",
        "kick forward with your left leg",
        "step backward with your right foot",
    )
    for prompt in prompts:
        scene, program = _program(prompt)
        clip = compile_motion(CompileRequest(scene=scene, program=program))
        assert program.intent == Intent.FULL_BODY, prompt
        assert clip.success, (prompt, clip.failure)
    turn_scene, turn_program = _program("turn left 90 degrees")
    turn_clip = compile_motion(CompileRequest(scene=turn_scene, program=turn_program))
    assert 89.0 <= turn_clip.metrics["final_root_yaw_deg"] <= 91.0
    run_scene, run_program = _program("run forward three steps")
    run_clip = compile_motion(CompileRequest(scene=run_scene, program=run_program))
    assert run_clip.metrics["root_displacement_m"] > 1.6
    assert run_clip.metrics["discontinuities"] == 0
    assert run_clip.metrics["airborne_frame_count"] > 0
    kick_scene, kick_program = _program("kick forward with your left leg")
    kick_clip = compile_motion(CompileRequest(scene=kick_scene, program=kick_program))
    assert kick_program.primitives[0].body.lead_side.value == "left"
    assert kick_program.primitives[0].body.direction_x == 0.0
    assert kick_program.primitives[0].body.direction_z == 1.0
    assert kick_clip.metrics["maximum_foot_clearance_m"] > 0.40
    assert kick_clip.metrics["max_support_foot_slide_per_frame_m"] < 0.001


def test_full_body_candidates_preserve_semantics_and_change_timing() -> None:
    scene, program = _program("walk forward three steps")
    recipes = candidate_recipes(program.primitives[0].parameters, Intent.FULL_BODY)
    clips = [
        compile_motion(
            CompileRequest(
                scene=scene,
                program=apply_recipe(program, recipe, seed_offset=index),
            )
        )
        for index, recipe in enumerate(recipes, start=1)
    ]

    assert len(recipes) == 5
    assert all(clip.success for clip in clips)
    assert len({round(clip.duration_s, 3) for clip in clips}) == 5
    assert len(
        {
            round(
                apply_recipe(program, recipe, seed_offset=index)
                .primitives[0]
                .body.intensity,
                3,
            )
            for index, recipe in enumerate(recipes, start=1)
        }
    ) == 5
    assert all(abs(clip.metrics["root_displacement_m"] - 1.44) < 1e-6 for clip in clips)


def test_walk_can_wave_one_hand_concurrently() -> None:
    scene, program = _program(
        "walk forward three steps while waving your right hand three times"
    )
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    body = program.primitives[0]

    assert program.intent == Intent.FULL_BODY
    assert body.body.cycles == 3.0
    assert body.trajectory.value == "oscillate"
    assert body.parameters.trajectory_cycles == 3.0
    assert [target.hand.value for target in body.effectors] == ["right"]
    assert clip.success, clip.failure
    assert clip.metrics["root_displacement_m"] > 1.3
    assert clip.metrics["discontinuities"] == 0
    kinematics = rig_kinematics()
    wrist_x = np.asarray(
        [
            kinematics.canonical_positions(frame.bones)["rightHand"][0]
            for frame in clip.frames
        ],
        dtype=float,
    )
    assert float(np.ptp(wrist_x)) > 0.12


def test_grounded_pose_primitive_covers_bend_lean_kneel_and_sit() -> None:
    cases = {
        "bend forward at the waist": {"drop": 0.0, "foot_span": 0.0},
        "lean to the left": {"drop": 0.0, "foot_span": 0.0},
        "kneel on your left knee": {"drop": 0.45, "foot_span": 0.35},
        "sit down": {"drop": 0.50, "foot_span": 0.0},
    }
    for prompt, expected in cases.items():
        scene, program = _program(prompt)
        clip = compile_motion(CompileRequest(scene=scene, program=program))
        pose_primitive = program.primitives[0]
        body_end = float(clip.metrics["phase_ranges_s"][0]["end_s"])
        authored_frame = min(clip.frames, key=lambda frame: abs(frame.time_s - body_end))
        positions = rig_kinematics().canonical_positions(authored_frame.bones)

        assert program.intent == Intent.FULL_BODY, prompt
        assert pose_primitive.body.action == BodyAction.POSE, prompt
        assert pose_primitive.body.pose.has_authored_change(), prompt
        assert clip.success, (prompt, clip.failure)
        assert clip.metrics["ground_penetration_m"] == 0.0
        assert clip.metrics["airborne_frame_count"] == 0
        assert clip.metrics["discontinuities"] == 0
        standing_y = clip.frames[0].bones["hips"].position.y
        assert standing_y - authored_frame.bones["hips"].position.y >= expected["drop"]
        if expected["foot_span"]:
            assert abs(float(positions["leftFoot"][2] - positions["rightFoot"][2])) > expected["foot_span"]


def test_structured_planner_can_author_unfamiliar_grounded_joint_pose() -> None:
    request = PlanRequest(
        text="shift your hips left, rotate at the waist, and soften both knees",
        scene=default_scene(),
        provider="openai",
    )
    selection = PlannerSelection(
        intent=Intent.FULL_BODY,
        body_segments=[
            GenericBodySegmentSelection(
                label="shift_twist_bend",
                duration_s=1.4,
                action=BodyAction.POSE,
                pose=BodyPoseTarget(
                    root_drop_m=0.16,
                    root_shift_x_m=0.10,
                    pelvis_yaw_deg=18.0,
                    torso_yaw_deg=32.0,
                    torso_roll_deg=8.0,
                    left_knee_flexion_deg=38.0,
                    right_knee_flexion_deg=38.0,
                ),
            )
        ],
    )

    program = OpenAIPlanner._expand(selection, request, variation_seed=71)
    OpenAIPlanner._validate_semantics(program, request.scene)
    clip = compile_motion(CompileRequest(scene=request.scene, program=program))

    assert clip.success, clip.failure
    assert program.primitives[0].body.pose.torso_yaw_deg == 32.0
    assert clip.metrics["root_vertical_min_m"] < -0.20
    assert clip.metrics["max_support_foot_slide_per_frame_m"] < 0.001


def test_redundant_zero_pose_from_model_is_absorbed_by_recovery() -> None:
    request = PlanRequest(
        text="shift your hips left and rotate your waist, then return to neutral",
        scene=default_scene(),
        provider="openai",
    )
    selection = PlannerSelection(
        intent=Intent.FULL_BODY,
        body_segments=[
            GenericBodySegmentSelection(
                label="shift_and_twist",
                duration_s=1.2,
                action=BodyAction.POSE,
                pose=BodyPoseTarget(root_shift_x_m=0.1, torso_yaw_deg=-30.0),
            ),
            GenericBodySegmentSelection(
                label="return_to_neutral",
                duration_s=0.8,
                action=BodyAction.POSE,
                pose=BodyPoseTarget(),
            ),
        ],
    )

    program = OpenAIPlanner._expand(selection, request, variation_seed=72)
    clip = compile_motion(CompileRequest(scene=request.scene, program=program))

    assert len(program.primitives) == 2
    assert program.primitives[0].body.action == BodyAction.POSE
    assert program.primitives[-1].body.action == BodyAction.HOLD
    assert clip.success, clip.failure


def test_locked_foot_knee_flexion_produces_visible_grounded_drop() -> None:
    request = PlanRequest(
        text="soften both knees",
        scene=default_scene(),
        provider="openai",
    )
    selection = PlannerSelection(
        intent=Intent.FULL_BODY,
        body_segments=[
            GenericBodySegmentSelection(
                label="soft_knees",
                duration_s=1.2,
                action=BodyAction.POSE,
                pose=BodyPoseTarget(
                    left_knee_flexion_deg=20.0,
                    right_knee_flexion_deg=20.0,
                    lock_feet=True,
                ),
            )
        ],
    )
    program = OpenAIPlanner._expand(selection, request, variation_seed=73)
    clip = compile_motion(CompileRequest(scene=request.scene, program=program))
    end = float(clip.metrics["phase_ranges_s"][0]["end_s"])
    frame = min(clip.frames, key=lambda item: abs(item.time_s - end))

    assert clip.success, clip.failure
    assert clip.frames[0].bones["hips"].position.y - frame.bones["hips"].position.y > 0.06
    assert clip.metrics["max_support_foot_slide_per_frame_m"] < 0.001


def test_sequential_grounded_poses_recover_original_root_and_stance() -> None:
    scene, program = _program("sit down then lean forward")
    clip = compile_motion(CompileRequest(scene=scene, program=program))

    assert [primitive.body.action for primitive in program.primitives[:-1]] == [
        BodyAction.POSE,
        BodyAction.POSE,
    ]
    assert clip.success, clip.failure
    assert clip.metrics["root_vertical_min_m"] < -0.55
    assert clip.metrics["final_support_center_offset_m"] < 0.06
    assert clip.metrics["final_foot_ground_error_m"] < 0.001


def test_pose_candidate_set_varies_spatial_amplitude_not_only_timing() -> None:
    scene, program = _program("sit down")
    recipes = candidate_recipes(program.primitives[0].parameters, Intent.FULL_BODY)
    candidates = [
        apply_recipe(program, recipe, seed_offset=index)
        for index, recipe in enumerate(recipes, start=1)
    ]
    clips = [
        compile_motion(CompileRequest(scene=scene, program=candidate))
        for candidate in candidates
    ]

    assert all(clip.success for clip in clips)
    assert len(
        {
            round(candidate.primitives[0].body.pose.root_drop_m, 3)
            for candidate in candidates
        }
    ) == 5
    assert len({round(clip.metrics["root_vertical_min_m"], 3) for clip in clips}) == 5


def test_visual_repair_can_strengthen_pose_without_changing_direction() -> None:
    scene, program = _program("lean to the left")
    original = program.primitives[0].body.pose
    repair = RepairPatch(
        arm_height_delta=0.0,
        arm_depth_delta=0.0,
        lateral_offset_delta=0.0,
        wrist_pitch_delta=0.0,
        wrist_yaw_delta=0.0,
        wrist_roll_delta=0.0,
        elbow_swivel_delta=0.0,
        finger_splay_delta=0.0,
        thumb_curl_delta=0.0,
        little_curl_delta=0.0,
        wrist_shake_amplitude_delta=0.0,
        present_duration_scale=1.0,
        hold_duration_scale=1.0,
        shake_duration_scale=1.0,
        recover_duration_scale=1.0,
        easing_delta=0.0,
        pose_root_scale=1.0,
        pose_directional_scale=1.3,
        rationale="Make the requested left lean visibly stronger without changing its side.",
    )

    repaired = apply_repair(program, repair)
    strengthened = repaired.primitives[0].body.pose
    clip = compile_motion(CompileRequest(scene=scene, program=repaired))

    assert strengthened.root_shift_x_m > original.root_shift_x_m > 0.0
    assert strengthened.torso_roll_deg > original.torso_roll_deg > 0.0
    assert clip.success, clip.failure


def test_full_body_parameters_are_directly_adjustable() -> None:
    scene, program = _program("walk forward two steps")
    clip = compile_motion(
        CompileRequest(
            scene=scene,
            program=program,
            parameter_overrides=ParameterOverrides(
                body_distance_m=1.25,
                body_cycles=3.0,
                body_intensity=0.8,
            ),
        )
    )

    assert clip.success, clip.failure
    assert abs(clip.metrics["root_displacement_m"] - 1.25) < 1e-6
    assert clip.slider_observables["body_cycles"] == 3.0
    assert clip.slider_observables["body_intensity"] == 0.8


def test_full_body_capture_samples_action_and_recovery() -> None:
    scene, program = _program("crouch then jump")
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    points = phase_sampling_points(
        {
            "program": program.model_dump(mode="json"),
            "clip": clip.model_dump(mode="json"),
        }
    )

    # Crouch is represented by entry/peak/exit; jump retains five samples for
    # takeoff, apex, and landing.  Full-body recovery uses three decisive poses.
    assert len([point for point in points if point["phase"] == "body"]) == 8
    assert len([point for point in points if point["phase"] == "recover"]) == 3


def test_crouch_height_is_a_physical_drop_not_double_scaled_by_intensity() -> None:
    scene, program = _program("crouch down")
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    body_end = clip.metrics["phase_ranges_s"][0]["end_s"]
    crouch_frame = min(clip.frames, key=lambda frame: abs(frame.time_s - body_end))
    standing_y = clip.frames[0].bones["hips"].position.y
    crouch_y = crouch_frame.bones["hips"].position.y

    assert clip.success, clip.failure
    assert standing_y - crouch_y > 0.22


def test_jump_after_crouch_reaches_requested_height_above_standing() -> None:
    scene, program = _program("crouch down then jump")
    clip = compile_motion(CompileRequest(scene=scene, program=program))

    assert clip.success, clip.failure
    assert clip.metrics["root_vertical_min_m"] < -0.30
    assert clip.metrics["root_vertical_max_m"] > 0.22
    assert clip.metrics["airborne_frame_count"] > 0
    assert 0.20 < clip.metrics["maximum_foot_clearance_m"] < 0.50


def test_burpees_preserve_support_order_and_exact_repetition_counts() -> None:
    for repetitions in (1, 3):
        scene, program = _program(f"do {repetitions} burpees")
        clip = compile_motion(CompileRequest(scene=scene, program=program))

        assert program.intent is Intent.FULL_BODY
        assert sum(
            primitive.label == f"burpee_{index}_push_up"
            for index in range(1, repetitions + 1)
            for primitive in program.primitives
        ) == repetitions
        assert clip.success, clip.failure
        assert clip.metrics["requested_burpee_cycles"] == repetitions
        assert clip.metrics["measured_burpee_push_up_cycles"] == repetitions
        assert clip.metrics["measured_burpee_jump_cycles"] == repetitions
        assert clip.metrics["burpee_floor_support_phase_count"] == repetitions
        assert clip.metrics["burpee_airborne_phase_count"] == repetitions
        assert clip.metrics["burpee_root_vertical_excursion_m"] > 0.55
        assert clip.metrics["burpee_max_wrist_height_m"] > 1.65
        assert clip.metrics["discontinuities"] == 0
        assert clip.metrics["ground_penetration_m"] == 0.0
        assert clip.metrics["final_support_center_offset_m"] < 0.14


def test_maximum_burpee_request_fits_the_typed_phase_program() -> None:
    _, program = _program("do eight burpees")

    assert program.intent is Intent.FULL_BODY
    # Five substantive phases per repetition, seven inter-cycle land/reset
    # transitions, plus the final balanced recovery.
    assert len(program.primitives) == 48
    assert program.primitives[-1].label == "settle_to_balanced_stance"


def test_auto_provider_uses_calibrated_burpee_graph_without_model_spend() -> None:
    class UnexpectedClient:
        @property
        def responses(self):
            raise AssertionError("calibrated burpees must not call the planner model")

    outcome = OpenAIPlanner(client=UnexpectedClient()).plan(
        PlanRequest(
            text="do three burpees",
            scene=default_scene(),
            provider="openai",
        )
    )

    assert outcome.program.intent is Intent.FULL_BODY
    assert outcome.model_calls == 0
    assert outcome.provider == "offline"


@pytest.mark.parametrize(
    ("prompt", "repetitions", "minimum_depth_m"),
    (
        ("do one squat", 1, 0.18),
        ("do three squats", 3, 0.18),
        ("do eight deep squats", 8, 0.24),
        ("do two shallow squats slowly", 2, 0.12),
    ),
)
def test_squats_preserve_exact_down_and_stance_return_cycles(
    prompt: str,
    repetitions: int,
    minimum_depth_m: float,
) -> None:
    scene, program = _program(prompt)
    clip = compile_motion(CompileRequest(scene=scene, program=program))

    assert program.intent is Intent.FULL_BODY
    assert sum(
        str(primitive.label or "").endswith("_descent")
        for primitive in program.primitives
    ) == repetitions
    assert clip.success, clip.failure
    assert clip.metrics["requested_squat_cycles"] == repetitions
    assert clip.metrics["measured_squat_cycles"] == repetitions
    assert clip.metrics["squat_minimum_depth_m"] > minimum_depth_m
    assert clip.metrics["squat_max_stance_return_error_m"] < 0.01
    assert clip.metrics["discontinuities"] == 0
    assert clip.metrics["ground_penetration_m"] == 0.0


def test_auto_provider_uses_calibrated_squat_cycles_without_model_spend() -> None:
    class UnexpectedClient:
        @property
        def responses(self):
            raise AssertionError("calibrated squats must not call the planner model")

    outcome = OpenAIPlanner(client=UnexpectedClient()).plan(
        PlanRequest(
            text="do three deep squats slowly",
            scene=default_scene(),
            provider="openai",
        )
    )

    assert outcome.program.intent is Intent.FULL_BODY
    assert outcome.model_calls == 0
    assert outcome.provider == "offline"
    assert sum(
        str(primitive.label or "").endswith("_descent")
        for primitive in outcome.program.primitives
    ) == 3


@pytest.mark.parametrize(
    ("prompt", "repetitions"),
    (
        ("lunge forward with your right leg", 1),
        ("do two lunges", 2),
        ("do three backward lunges", 3),
        ("do four side lunges", 4),
        ("do two shallow lunges slowly", 2),
    ),
)
def test_lunges_preserve_exact_stagger_and_stance_return_cycles(
    prompt: str,
    repetitions: int,
) -> None:
    scene, program = _program(prompt)
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    lunge_phases = [
        primitive
        for primitive in program.primitives
        if str(primitive.label or "").startswith("lunge_")
        and not str(primitive.label or "").endswith("_stance_reset")
    ]

    assert program.intent is Intent.FULL_BODY
    assert len(lunge_phases) == repetitions
    if repetitions > 1 and "right leg" not in prompt and "left leg" not in prompt:
        assert [phase.body.lead_side for phase in lunge_phases[:2]] == [
            Hand.RIGHT,
            Hand.LEFT,
        ]
    assert clip.success, clip.failure
    assert clip.metrics["requested_lunge_cycles"] == repetitions
    assert clip.metrics["measured_lunge_cycles"] == repetitions
    assert clip.metrics["lunge_minimum_root_drop_m"] > 0.10
    assert clip.metrics["lunge_minimum_foot_stagger_m"] > 0.12
    assert clip.metrics["lunge_max_stance_return_error_m"] < 0.01
    assert clip.metrics["discontinuities"] == 0
    assert clip.metrics["ground_penetration_m"] == 0.0


def test_auto_provider_uses_calibrated_lunge_cycles_without_model_spend() -> None:
    class UnexpectedClient:
        @property
        def responses(self):
            raise AssertionError("calibrated lunges must not call the planner model")

    outcome = OpenAIPlanner(client=UnexpectedClient()).plan(
        PlanRequest(
            text="do three alternating backward lunges",
            scene=default_scene(),
            provider="openai",
        )
    )

    assert outcome.program.intent is Intent.FULL_BODY
    assert outcome.model_calls == 0
    assert outcome.provider == "offline"
    assert sum(
        str(primitive.label or "").startswith("lunge_")
        and not str(primitive.label or "").endswith("_stance_reset")
        for primitive in outcome.program.primitives
    ) == 3


@pytest.mark.parametrize(
    ("prompt", "raised_side"),
    (
        ("stand on your left leg", Hand.RIGHT),
        ("stand on your right foot", Hand.LEFT),
        ("balance on one leg", Hand.RIGHT),
        ("lift your right knee", Hand.RIGHT),
        ("lift your left foot high", Hand.LEFT),
    ),
)
def test_single_leg_balance_lifts_only_the_free_foot(
    prompt: str,
    raised_side: Hand,
) -> None:
    scene, program = _program(prompt)
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    pose = program.primitives[0].body.pose

    assert program.intent is Intent.FULL_BODY
    assert program.primitives[0].label.startswith("single_leg_balance_")
    assert (
        pose.left_foot_lift_m if raised_side is Hand.LEFT else pose.right_foot_lift_m
    ) > 0.25
    assert (
        pose.right_foot_lift_m if raised_side is Hand.LEFT else pose.left_foot_lift_m
    ) == 0.0
    assert clip.success, clip.failure
    assert clip.metrics["single_leg_balance_phase_count"] == 1
    assert clip.metrics["single_leg_min_raised_foot_clearance_m"] > 0.25
    assert clip.metrics["single_leg_max_support_foot_slide_m"] < 0.001
    assert clip.metrics["single_leg_support_contact_fraction"] == 1.0
    assert clip.metrics["airborne_frame_count"] == 0
    assert clip.metrics["discontinuities"] == 0


def test_auto_provider_uses_calibrated_single_leg_balance_without_model_spend() -> None:
    class UnexpectedClient:
        @property
        def responses(self):
            raise AssertionError("calibrated balance must not call the planner model")

    outcome = OpenAIPlanner(client=UnexpectedClient()).plan(
        PlanRequest(
            text="stand on your left leg",
            scene=default_scene(),
            provider="openai",
        )
    )

    assert outcome.program.intent is Intent.FULL_BODY
    assert outcome.model_calls == 0
    assert outcome.provider == "offline"
    assert outcome.program.primitives[0].label == "single_leg_balance_left_support"


@pytest.mark.parametrize(
    ("prompt", "repetitions"),
    (
        ("do one sit-up", 1),
        ("do three sit ups", 3),
        ("do eight sit-ups", 8),
        ("do two slow controlled sit-ups", 2),
    ),
)
def test_sit_ups_preserve_exact_supine_curl_and_return_cycles(
    prompt: str,
    repetitions: int,
) -> None:
    scene, program = _program(prompt)
    clip = compile_motion(CompileRequest(scene=scene, program=program))

    assert program.intent is Intent.FULL_BODY
    assert sum(
        str(primitive.label or "").endswith("_curl")
        for primitive in program.primitives
    ) == repetitions
    assert sum(
        str(primitive.label or "").endswith("_return")
        for primitive in program.primitives
    ) == repetitions
    assert clip.success, clip.failure
    assert clip.metrics["requested_sit_up_cycles"] == repetitions
    assert clip.metrics["measured_sit_up_cycles"] == repetitions
    assert clip.metrics["sit_up_minimum_head_lift_m"] > 0.50
    assert clip.metrics["sit_up_max_supine_return_error_m"] < 0.01
    assert clip.metrics["horizontal_pose_contact_point_count"] >= 2
    assert clip.metrics["ground_penetration_m"] == 0.0
    assert clip.metrics["discontinuities"] == 0


def test_auto_provider_uses_calibrated_sit_up_cycles_without_model_spend() -> None:
    class UnexpectedClient:
        @property
        def responses(self):
            raise AssertionError("calibrated sit-ups must not call the planner model")

    outcome = OpenAIPlanner(client=UnexpectedClient()).plan(
        PlanRequest(
            text="do three controlled sit-ups",
            scene=default_scene(),
            provider="openai",
        )
    )

    assert outcome.program.intent is Intent.FULL_BODY
    assert outcome.model_calls == 0
    assert outcome.provider == "offline"
    assert sum(
        str(primitive.label or "").endswith("_curl")
        for primitive in outcome.program.primitives
    ) == 3


@pytest.mark.parametrize(
    ("prompt", "mode", "axis", "travel_axis", "travel_sign"),
    (
        ("do a forward roll", BodyRotationMode.FLOOR, BodyRotationAxis.PITCH, 2, 1.0),
        ("do a backward roll", BodyRotationMode.FLOOR, BodyRotationAxis.PITCH, 2, -1.0),
        ("do a cartwheel to the left", BodyRotationMode.CARTWHEEL, BodyRotationAxis.ROLL, 0, 1.0),
        ("do a cartwheel to the right", BodyRotationMode.CARTWHEEL, BodyRotationAxis.ROLL, 0, -1.0),
    ),
)
def test_grounded_rotations_complete_angle_travel_and_support_transfer(
    prompt: str,
    mode: BodyRotationMode,
    axis: BodyRotationAxis,
    travel_axis: int,
    travel_sign: float,
) -> None:
    scene, program = _program(prompt)
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    rotation = program.primitives[0].body

    assert program.intent is Intent.FULL_BODY
    assert rotation is not None
    assert rotation.action is BodyAction.ROTATE
    assert rotation.rotation_mode is mode
    assert rotation.rotation_axis is axis
    assert abs(rotation.rotation_degrees) == 360.0
    assert clip.success, clip.failure
    assert clip.metrics["requested_body_rotation_degrees"] == 360.0
    assert clip.metrics["measured_body_rotation_degrees"] == pytest.approx(
        360.0,
        abs=1e-5,
    )
    assert clip.metrics["minimum_body_rotation_completion_fraction"] == pytest.approx(
        1.0,
        abs=1e-5,
    )
    assert clip.metrics["minimum_rotation_travel_completion_fraction"] == pytest.approx(
        1.0,
        abs=1e-5,
    )
    final_root = clip.frames[-1].bones["hips"].position
    assert final_root is not None
    assert final_root.as_list()[travel_axis] * travel_sign > 0.70
    assert clip.metrics["ground_penetration_m"] == 0.0
    assert clip.metrics["discontinuities"] == 0
    if mode is BodyRotationMode.FLOOR:
        assert clip.metrics["floor_roll_nonfoot_contact_fraction"] > 0.30
    else:
        assert clip.metrics["cartwheel_hand_contact_frame_count"] > 10
        assert clip.metrics["cartwheel_inverted_frame_count"] > 8
        assert clip.metrics["cartwheel_max_foot_clearance_m"] > 1.2
        assert clip.metrics["cartwheel_minimum_head_clearance_m"] > 0.08


@pytest.mark.parametrize(
    ("prompt", "axis", "degrees"),
    (
        ("do a front flip", BodyRotationAxis.PITCH, 360.0),
        ("do a backflip", BodyRotationAxis.PITCH, 360.0),
        ("jump and spin", BodyRotationAxis.YAW, 360.0),
        ("perform a 720 clockwise", BodyRotationAxis.YAW, 720.0),
    ),
)
def test_airborne_rotations_complete_requested_axis_and_leave_support(
    prompt: str,
    axis: BodyRotationAxis,
    degrees: float,
) -> None:
    scene, program = _program(prompt)
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    rotation = program.primitives[0].body

    assert program.intent is Intent.FULL_BODY
    assert rotation is not None
    assert rotation.action is BodyAction.ROTATE
    assert rotation.rotation_mode is BodyRotationMode.AIRBORNE
    assert rotation.rotation_axis is axis
    assert abs(rotation.rotation_degrees) == degrees
    assert clip.success, clip.failure
    assert clip.metrics["requested_body_rotation_degrees"] == degrees
    assert clip.metrics["measured_body_rotation_degrees"] == pytest.approx(
        degrees,
        abs=1e-5,
    )
    assert clip.metrics["airborne_rotation_airborne_frame_count"] > 50
    assert clip.metrics["airborne_rotation_airborne_fraction"] > 0.90
    assert clip.metrics["ground_penetration_m"] == 0.0
    assert clip.metrics["discontinuities"] == 0


@pytest.mark.parametrize(
    "prompt",
    (
        "do a forward roll",
        "do a cartwheel to the right",
        "do a front flip",
        "jump and spin",
    ),
)
def test_auto_provider_uses_calibrated_rotation_graphs_without_model_spend(
    prompt: str,
) -> None:
    class UnexpectedClient:
        @property
        def responses(self):
            raise AssertionError("calibrated rotations must not call the planner model")

    outcome = OpenAIPlanner(client=UnexpectedClient()).plan(
        PlanRequest(
            text=prompt,
            scene=default_scene(),
            provider="openai",
        )
    )

    assert outcome.program.intent is Intent.FULL_BODY
    assert outcome.program.primitives[0].body.action is BodyAction.ROTATE
    assert outcome.model_calls == 0
    assert outcome.provider == "offline"


def _ground_obstacle_scene():
    scene = default_scene()
    obstacle = scene.objects[0].model_copy(
        update={
            "id": "hurdle",
            "transform": Transform(
                translation=Vec3(x=0.0, y=0.07, z=0.55)
            ),
            "dimensions_m": Vec3(x=0.12, y=0.10, z=0.10),
        }
    )
    return scene.model_copy(update={"objects": [obstacle]})


def test_step_over_scene_obstacle_clears_the_full_top_surface() -> None:
    scene = _ground_obstacle_scene()
    outcome = OfflinePlanner().plan(
        PlanRequest(
            text="step over the hurdle with your right foot",
            scene=scene,
            provider="offline",
        )
    )
    program = outcome.program
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    step = program.primitives[0].body

    assert step is not None
    assert step.action is BodyAction.STEP
    assert step.obstacle_mode is BodyObstacleMode.OVER
    assert step.obstacle_object_id == "hurdle"
    assert step.lead_side is Hand.RIGHT
    assert clip.success, clip.failure
    assert clip.metrics["obstacle_missing_target_count"] == 0
    assert clip.metrics["minimum_obstacle_step_foot_clearance_m"] > 0.09
    assert clip.metrics["maximum_obstacle_step_crossing_error_m"] < 0.10
    crossing_time = clip.metrics["obstacle_traversal_phase_metrics"][0][
        "closest_approach_time_s"
    ]
    points = phase_sampling_points(
        {
            "program": program.model_dump(mode="json"),
            "clip": clip.model_dump(mode="json"),
        }
    )
    assert any(
        point["label"] == "obstacle_closest_approach"
        and point["time_s"] == pytest.approx(crossing_time)
        for point in points
    )
    assert clip.metrics["max_support_foot_target_error_m"] < 0.001
    assert clip.metrics["max_support_foot_slide_per_frame_m"] < 0.001
    assert clip.metrics["discontinuities"] == 0


@pytest.mark.parametrize(
    ("prompt", "detour_sign"),
    (
        ("walk around the hurdle", 1.0),
        ("run around the hurdle on the right", -1.0),
    ),
)
def test_locomotion_detours_around_scene_obstacle(
    prompt: str,
    detour_sign: float,
) -> None:
    scene = _ground_obstacle_scene()
    outcome = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    )
    program = outcome.program
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    traversal = program.primitives[0].body
    root_x = np.asarray(
        [frame.bones["hips"].position.x for frame in clip.frames],
        dtype=float,
    )

    assert traversal is not None
    assert traversal.obstacle_mode is BodyObstacleMode.AROUND
    assert traversal.path_lateral_offset_m * detour_sign > 0.30
    assert clip.success, clip.failure
    assert float(np.max(root_x) if detour_sign > 0 else -np.min(root_x)) > 0.35
    record = clip.metrics["obstacle_traversal_phase_metrics"][0]
    assert record["minimum_root_clearance_m"] > record["required_root_clearance_m"]
    assert clip.metrics["ground_penetration_m"] == 0.0
    assert clip.metrics["discontinuities"] == 0


def test_elevated_task_object_is_rejected_as_a_floor_obstacle() -> None:
    scene = default_scene()
    outcome = OfflinePlanner().plan(
        PlanRequest(
            text="step over the block",
            scene=scene,
            provider="offline",
        )
    )

    assert outcome.program.intent is Intent.UNSUPPORTED
    assert outcome.program.unsupported_reason == (
        "requested object 'block' is not a traversable floor-level obstacle"
    )


@pytest.mark.parametrize(
    ("prompt", "mode"),
    (
        ("step over the hurdle with your right foot", BodyObstacleMode.OVER),
        ("walk around the hurdle", BodyObstacleMode.AROUND),
    ),
)
def test_default_scene_floor_hurdle_compiles(
    prompt: str,
    mode: BodyObstacleMode,
) -> None:
    scene = default_scene()
    outcome = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    )
    clip = compile_motion(CompileRequest(scene=scene, program=outcome.program))
    traversals = [
        primitive.body
        for primitive in outcome.program.primitives
        if primitive.body is not None
        and primitive.body.obstacle_mode is not BodyObstacleMode.NONE
    ]

    assert len(traversals) == 1
    assert traversals[0].obstacle_object_id == "hurdle"
    assert traversals[0].obstacle_mode is mode
    assert clip.success, clip.failure
    assert clip.metrics["obstacle_missing_target_count"] == 0
    assert clip.metrics["discontinuities"] == 0


@pytest.mark.parametrize(
    "prompt",
    (
        "step over the hurdle",
        "walk around the hurdle",
    ),
)
def test_auto_provider_uses_scene_aware_obstacle_graph_without_model_spend(
    prompt: str,
) -> None:
    class UnexpectedClient:
        @property
        def responses(self):
            raise AssertionError(
                "calibrated obstacle traversal must not call the planner model"
            )

    outcome = OpenAIPlanner(client=UnexpectedClient()).plan(
        PlanRequest(
            text=prompt,
            scene=_ground_obstacle_scene(),
            provider="openai",
        )
    )

    assert outcome.program.intent is Intent.FULL_BODY
    assert any(
        primitive.body is not None
        and primitive.body.obstacle_mode is not BodyObstacleMode.NONE
        for primitive in outcome.program.primitives
    )
    assert outcome.model_calls == 0
    assert outcome.provider == "offline"


def test_walk_capture_avoids_step_boundaries() -> None:
    scene, program = _program("walk forward four steps")
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    points = phase_sampling_points(
        {
            "program": program.model_dump(mode="json"),
            "clip": clip.model_dump(mode="json"),
        }
    )
    body_points = [point for point in points if point["phase"] == "body"]
    body_end = clip.metrics["phase_ranges_s"][0]["end_s"]

    assert len(body_points) == 5
    for point in body_points:
        step_progress = (point["time_s"] / body_end) * 4.0
        assert abs(step_progress - round(step_progress)) > 0.1


def test_glb_export_contains_animated_hips_translation(tmp_path) -> None:
    scene, program = _program("walk forward two steps")
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    profile = json.loads(
        (PROJECT_ROOT / "config/rig_profiles/mesh2motion-human-vrm1.json").read_text(
            encoding="utf-8"
        )
    )
    output = tmp_path / "walk.glb"
    export_glb(
        clip,
        scene,
        output,
        PROJECT_ROOT / scene.rig.asset_uri,
        profile,
    )
    document, binary = _glb_parts(output)
    pelvis = next(
        index for index, node in enumerate(document["nodes"])
        if node.get("name") == "pelvis"
    )

    assert any(
        channel["target"] == {"node": pelvis, "path": "translation"}
        for animation in document["animations"]
        for channel in animation["channels"]
    )
    generated = next(
        animation
        for animation in document["animations"]
        if animation.get("name") == "RigbyGeneratedMotion"
    )
    translation_channel = next(
        channel
        for channel in generated["channels"]
        if channel["target"] == {"node": pelvis, "path": "translation"}
    )
    values = _float_accessor(
        document,
        binary,
        generated["samplers"][translation_channel["sampler"]]["output"],
    )
    local_delta = values[-1] - values[0]
    root = next(node for node in document["nodes"] if node.get("name") == "root")
    world_delta = Rotation.from_quat(root["rotation"]).apply(local_delta)

    assert abs(float(world_delta[0])) < 1e-5
    # A balanced split stance may settle the pelvis slightly lower than the
    # initial parallel stance, but export must not introduce vertical travel.
    assert abs(float(world_delta[1])) < 0.02
    assert float(world_delta[2]) > 0.9
