from __future__ import annotations

import pytest

from evals.capture import phase_sampling_points
from rigby_poc.compiler import compile_motion
from rigby_poc.models import (
    BodyAction,
    BodyClimbDirection,
    BodySupportMode,
    CompileRequest,
    HandShape,
    Intent,
    ObjectAction,
    PlanRequest,
    default_scene,
)
from rigby_poc.planner import OpenAIPlanner, plan_motion

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


@pytest.mark.parametrize(
    ("prompt", "action"),
    (
        ("sidestep left twice", BodyAction.STEP),
        ("shuffle to the left", BodyAction.WALK),
        ("lunge forward with your right leg", BodyAction.POSE),
        ("look over your left shoulder", BodyAction.POSE),
        ("nod your head twice", BodyAction.POSE),
        ("shake your head no", BodyAction.POSE),
        ("shrug both shoulders", BodyAction.POSE),
    ),
)
def test_extended_body_vocabulary_compiles_without_keyword_drops(
    prompt: str,
    action: BodyAction,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.FULL_BODY
    assert program.primitives[0].body is not None
    assert program.primitives[0].body.action is action
    assert clip.success, clip.failure
    assert clip.metrics["structural_valid"] is True
    assert clip.metrics["discontinuities"] == 0


@pytest.mark.parametrize(
    ("prompt", "shape"),
    (
        ("give a thumbs up with your left hand", HandShape.THUMBS_UP),
        ("make a peace sign with your right hand", HandShape.PEACE),
    ),
)
def test_common_finger_signs_have_explicit_shapes(
    prompt: str,
    shape: HandShape,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.GESTURE
    assert {item.hand_shape for item in program.primitives[:-1]} == {shape}
    assert clip.success, clip.failure


@pytest.mark.parametrize(
    "prompt",
    (
        "put both hands on your hips",
        "reach forward with both hands",
        "salute with your right hand",
        "beckon someone closer with your right hand",
        "snap the fingers of your left hand",
        "swing both arms in circles",
    ),
)
def test_common_spatial_arm_actions_compile_as_visible_compositions(
    prompt: str,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.COMPOSITE
    assert clip.success, clip.failure
    assert clip.metrics["structural_valid"] is True
    assert clip.metrics["active_hand_visibility_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("prompt", "beats"),
    (
        ("do a little dance", 4),
        ("dance for 6 beats", 6),
        ("do a quick energetic two-step for 8 counts", 8),
    ),
)
def test_dance_uses_bounded_alternating_choreography(
    prompt: str,
    beats: int,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )
    dance = next(
        primitive
        for primitive in program.primitives
        if primitive.label == "dance_rhythmic_two_step"
    )
    hips_x = [frame.bones["hips"].position.x for frame in clip.frames]

    assert program.intent is Intent.FULL_BODY
    assert dance.body is not None
    assert dance.body.action is BodyAction.DANCE
    assert dance.body.cycles == pytest.approx(beats)
    assert len(dance.effectors) == 2
    assert max(hips_x) - min(hips_x) >= 0.075
    assert abs(hips_x[-1]) <= 0.02
    assert clip.success, clip.failure
    assert clip.metrics["structural_valid"] is True
    assert clip.metrics["discontinuities"] == 0
    assert clip.metrics["requested_dance_beats"] == beats
    assert clip.metrics["measured_dance_beats"] == beats
    assert clip.metrics["dance_alternating_lift_count"] == beats
    assert clip.metrics["dance_lateral_root_range_m"] >= 0.075
    assert clip.metrics["dance_left_foot_peak_clearance_m"] >= 0.055
    assert clip.metrics["dance_right_foot_peak_clearance_m"] >= 0.055
    assert clip.metrics["max_support_foot_slide_per_frame_m"] < 0.001
    sampling_labels = {
        point["label"]
        for point in phase_sampling_points(
            {
                "program": program.model_dump(mode="json"),
                "clip": clip.model_dump(mode="json"),
            }
        )
    }
    assert {f"dance_beat_{index}" for index in range(1, beats + 1)} <= sampling_labels


def test_auto_provider_uses_calibrated_dance_graph_without_model_spend() -> None:
    class UnexpectedClient:
        @property
        def responses(self):
            raise AssertionError("calibrated dance must not call the planner model")

    outcome = OpenAIPlanner(client=UnexpectedClient()).plan(
        PlanRequest(
            text="dance for four beats",
            scene=default_scene(),
            provider="openai",
        )
    )

    assert outcome.program.intent is Intent.FULL_BODY
    assert outcome.model_calls == 0
    assert outcome.provider == "offline"


@pytest.mark.parametrize(
    "prompt",
    (
        "walk forward two steps",
        "crawl forward two steps",
        "carry the block forward two steps",
    ),
)
def test_step_counts_are_not_misclassified_as_the_two_step_dance(
    prompt: str,
) -> None:
    program = plan_motion(
        PlanRequest(text=prompt, scene=default_scene(), provider="offline")
    ).program
    programs = program.steps if program.intent is Intent.SEQUENCE else [program]
    body_actions = [
        primitive.body.action
        for child in programs
        for primitive in child.primitives
        if primitive.body is not None
    ]

    assert body_actions
    assert BodyAction.DANCE not in body_actions


@pytest.mark.parametrize(
    ("prompt", "reason_fragment"),
    (
        ("kick the block forward", "persistent contact lifecycle"),
        ("turn the doorknob with your right hand", "scene does not contain"),
        ("sit down on the chair", "scene does not contain"),
    ),
)
def test_scene_dependent_actions_never_silently_drop_the_object_or_affordance(
    prompt: str,
    reason_fragment: str,
) -> None:
    program = plan_motion(
        PlanRequest(text=prompt, scene=default_scene(), provider="offline")
    ).program

    assert program.intent is Intent.UNSUPPORTED
    assert reason_fragment in (program.unsupported_reason or "")


@pytest.mark.parametrize(
    ("prompt", "direction", "height_m", "cycles"),
    (
        ("climb the ladder for three rungs", BodyClimbDirection.UP, 0.615, 3.0),
        ("climb down the ladder for three rungs", BodyClimbDirection.DOWN, 0.615, 3.0),
        ("scale the ladder for five rungs", BodyClimbDirection.UP, 1.025, 5.0),
    ),
)
def test_ladder_climb_uses_weight_bearing_four_limb_contacts(
    prompt: str,
    direction: BodyClimbDirection,
    height_m: float,
    cycles: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    climb = program.primitives[0]
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.FULL_BODY
    assert climb.body is not None
    assert climb.body.action is BodyAction.CLIMB
    assert climb.body.support_object_id == "ladder"
    assert climb.body.climb_direction is direction
    assert climb.body.height_m == pytest.approx(height_m)
    assert climb.body.cycles == pytest.approx(cycles)
    assert len(climb.effectors) == 2
    assert clip.success, clip.failure
    assert clip.metrics["climb_vertical_completion_fraction"] == pytest.approx(1.0)
    assert clip.metrics["climb_three_point_support_fraction"] >= 0.70
    assert clip.metrics["climb_final_supported_limb_count"] >= 3
    assert clip.metrics["climb_support_target_max_error_m"] <= 0.16
    labels = {
        point["label"]
        for point in phase_sampling_points(
            {
                "program": program.model_dump(mode="json"),
                "clip": clip.model_dump(mode="json"),
            }
        )
    }
    assert {f"climb_cycle_{index}" for index in range(1, int(cycles) + 1)} <= labels


def test_ladder_climb_rejects_a_scene_without_climb_affordances() -> None:
    default = default_scene()
    block_only = default.model_copy(update={"objects": [default.objects[0]]})
    program = plan_motion(
        PlanRequest(
            text="climb the ladder",
            scene=block_only,
            provider="offline",
        )
    ).program

    assert program.intent is Intent.UNSUPPORTED
    assert "scene does not contain requested ladder affordance" in (
        program.unsupported_reason or ""
    )


def test_carry_composes_pickup_and_root_motion_without_object_reset() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="carry the block forward two steps",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.SEQUENCE
    assert [step.intent for step in program.steps] == [Intent.GRAB, Intent.FULL_BODY]
    assert clip.success, clip.failure
    assert clip.metrics["root_displacement_m"] > 0.80
    assert clip.metrics["carried_object_ids"] == ["block"]
    assert clip.metrics["carried_object_max_step_m"] < 0.08
    first = clip.frames[0].objects["block"].translation
    last = clip.frames[-1].objects["block"].translation
    assert last.z - first.z > 0.80


@pytest.mark.parametrize(
    ("prompt", "action", "minimum_forward_displacement_m"),
    (
        (
            "grab the block with your right hand then carry it forward two steps then drop it",
            ObjectAction.DROP,
            0.80,
        ),
        (
            "carry the block forward two steps then throw it forward",
            ObjectAction.THROW,
            1.50,
        ),
    ),
)
def test_carried_object_releases_from_its_accumulated_world_position(
    prompt: str,
    action: ObjectAction,
    minimum_forward_displacement_m: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program

    assert program.intent is Intent.SEQUENCE
    assert [step.intent for step in program.steps] == [
        Intent.GRAB,
        Intent.FULL_BODY,
        Intent.OBJECT_INTERACTION,
    ]
    assert sum(step.intent is Intent.GRAB for step in program.steps) == 1
    assert program.steps[-1].object_action is action

    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert clip.success, clip.failure
    assert clip.metrics["root_displacement_m"] > 0.80
    assert clip.metrics["stateful_object_transition_count"] == 1
    transition = clip.metrics["stateful_object_transitions"][0]
    assert transition["action"] == action.value
    assert transition["release_time_s"] > clip.metrics["sequence_step_ranges_s"][1]["end_s"]
    assert transition["landing_time_s"] > transition["release_time_s"]
    assert (
        clip.metrics["carried_object_max_step_m"]
        < clip.metrics["carried_object_max_step_reference_m"]
    )
    first = clip.frames[0].objects["block"].translation
    last = clip.frames[-1].objects["block"].translation
    assert last.z - first.z > minimum_forward_displacement_m
    assert last.y == pytest.approx(
        program.steps[-1].object_motion.landing_height_m,
        abs=1e-8,
    )


@pytest.mark.parametrize(
    ("prompt", "action", "expected_z_sign"),
    (
        ("push the block away from you", ObjectAction.PUSH, 1.0),
        ("pull the block toward you", ObjectAction.PULL, -1.0),
    ),
)
def test_guided_object_motion_preserves_contact_and_support_plane(
    prompt: str,
    action: ObjectAction,
    expected_z_sign: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.OBJECT_INTERACTION
    assert program.object_action is action
    assert clip.success, clip.failure
    first = clip.frames[0].objects["block"].translation
    last = clip.frames[-1].objects["block"].translation
    assert (last.z - first.z) * expected_z_sign > 0.10
    assert last.y == pytest.approx(first.y, abs=1e-8)
    assert clip.metrics["palm_relative_object_slip_m"] < 0.005
    assert clip.metrics["object_guided_support_height_error_m"] < 0.01
    states = [item["state"] for item in clip.metrics["object_lifecycle"]]
    assert "guided" in states
    assert states[-1] == "released"


@pytest.mark.parametrize(
    ("prompt", "action", "axis", "sign"),
    (
        ("slide the block to the left", ObjectAction.PUSH, "x", 1.0),
        ("drag the block toward you", ObjectAction.PULL, "z", -1.0),
    ),
)
def test_guided_object_motion_understands_slide_and_drag_aliases(
    prompt: str,
    action: ObjectAction,
    axis: str,
    sign: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.object_action is action
    assert clip.success, clip.failure
    first = clip.frames[0].objects["block"].translation
    last = clip.frames[-1].objects["block"].translation
    assert (getattr(last, axis) - getattr(first, axis)) * sign > 0.10


def test_roll_couples_support_plane_translation_to_physical_rotation() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="roll the block forward",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.OBJECT_INTERACTION
    assert program.object_action is ObjectAction.ROLL
    assert clip.success, clip.failure
    first = clip.frames[0].objects["block"].translation
    last = clip.frames[-1].objects["block"].translation
    assert last.z - first.z > 0.10
    assert abs(last.y - first.y) < 0.02
    assert clip.metrics["object_measured_roll_turns"] >= (
        clip.metrics["object_expected_roll_turns"] * 0.75
    )
    assert clip.metrics["object_guided_lateral_error_m"] < 0.06
    assert clip.metrics["object_guided_support_height_error_m"] < 0.01
    states = [item["state"] for item in clip.metrics["object_lifecycle"]]
    assert "guided" in states
    assert states[-1] == "released"


@pytest.mark.parametrize(
    ("prompt", "sign"),
    (
        ("spin the block clockwise", 1.0),
        ("spin the block counterclockwise", -1.0),
    ),
)
def test_support_plane_spin_keeps_center_fixed_and_completes_yaw(
    prompt: str,
    sign: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.OBJECT_INTERACTION
    assert program.object_action is ObjectAction.SPIN
    assert program.object_motion is not None
    assert program.object_motion.spin_turns * sign > 0.0
    assert clip.success, clip.failure
    first = clip.frames[0].objects["block"].translation
    last = clip.frames[-1].objects["block"].translation
    assert last.x == pytest.approx(first.x, abs=1e-8)
    assert last.z == pytest.approx(first.z, abs=1e-8)
    assert clip.metrics["object_measured_support_spin_turns"] >= 0.90
    assert clip.metrics["object_guided_support_height_error_m"] < 0.01
    assert clip.metrics["palm_relative_object_slip_m"] < 0.005
    states = [item["state"] for item in clip.metrics["object_lifecycle"]]
    assert "guided" in states
    assert states[-1] == "released"


def test_place_lifts_translates_lowers_and_releases_on_support() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="place the block to the left on the table",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.OBJECT_INTERACTION
    assert program.object_action is ObjectAction.PLACE
    assert clip.success, clip.failure
    first = clip.frames[0].objects["block"].translation
    last = clip.frames[-1].objects["block"].translation
    assert last.x - first.x > 0.10
    assert last.y == pytest.approx(first.y, abs=0.01)
    assert clip.metrics["object_placement_horizontal_distance_m"] > 0.10
    assert clip.metrics["landing_height_error_m"] < 0.01
    assert clip.metrics["palm_relative_object_slip_m"] < 0.005
    states = [item["state"] for item in clip.metrics["object_lifecycle"]]
    assert "attached" in states
    assert states[-1] == "placed"


def test_ground_place_is_typed_until_body_reach_is_coupled() -> None:
    program = plan_motion(
        PlanRequest(
            text="place the block on the ground",
            scene=default_scene(),
            provider="offline",
        )
    ).program

    assert program.intent is Intent.UNSUPPORTED
    assert "coupled body reach" in (program.unsupported_reason or "")


def test_drop_releases_before_gravity_flight_and_lands_on_support() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="drop the block on the ground",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.object_action is ObjectAction.DROP
    assert clip.success, clip.failure
    states = [item["state"] for item in clip.metrics["object_lifecycle"]]
    assert states.index("released") < states.index("ballistic") < states.index("landed")
    assert clip.metrics["landing_height_error_m"] < 0.01
    first = clip.frames[0].objects["block"].translation
    last = clip.frames[-1].objects["block"].translation
    assert last.y < first.y - 0.50


def test_supported_grab_with_explicit_hand_is_not_blocked_by_object_guard() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="grab the block with your right hand",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.GRAB
    assert clip.success, clip.failure


def test_overhead_lift_uses_extended_physical_grasp_workspace() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="lift the block over your head",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.GRAB
    assert clip.success, clip.failure
    assert clip.metrics["lift_height_m"] > 0.65
    assert clip.frames[-1].objects["block"].translation.y > 1.70


def test_all_fours_transfers_support_to_both_palms_and_knees() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="get down on all fours",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    pose = program.primitives[0]
    assert program.intent is Intent.FULL_BODY
    assert pose.body is not None
    assert pose.body.action is BodyAction.POSE
    assert pose.body.pose.support_mode is BodySupportMode.QUADRUPED
    assert {target.hand.value for target in pose.effectors} == {"left", "right"}
    assert clip.success, clip.failure
    assert clip.metrics["structural_valid"] is True
    assert clip.metrics["horizontal_pose_variant"] == "quadruped"
    assert clip.metrics["horizontal_pose_contact_point_count"] >= 4
    assert clip.metrics["horizontal_pose_minimum_clearance_m"] >= -0.012
    assert clip.metrics["horizontal_body_axis_vertical_fraction"] <= 0.35
    assert clip.metrics["discontinuities"] == 0


@pytest.mark.parametrize(
    ("prompt", "variant", "roll_sign"),
    (
        ("roll onto your left side", "side_left", 1.0),
        ("roll onto your right side", "side_right", -1.0),
    ),
)
def test_side_lying_pose_uses_lateral_floor_support(
    prompt: str,
    variant: str,
    roll_sign: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    pose = program.primitives[0]
    assert program.intent is Intent.FULL_BODY
    assert pose.body is not None
    assert pose.body.pose.pelvis_roll_deg * roll_sign >= 60.0
    assert pose.body.pose.support_mode is BodySupportMode.BROAD_FLOOR
    assert clip.success, clip.failure
    assert clip.metrics["horizontal_pose_variant"] == variant
    assert clip.metrics["horizontal_pose_contact_point_count"] >= 2
    assert clip.metrics["horizontal_body_axis_vertical_fraction"] <= 0.35


def test_forward_roll_is_not_misclassified_as_side_lying() -> None:
    program = plan_motion(
        PlanRequest(
            text="do a forward roll",
            scene=default_scene(),
            provider="offline",
        )
    ).program

    assert program.intent is Intent.FULL_BODY
    assert program.primitives[0].body is not None
    assert program.primitives[0].body.action is BodyAction.ROTATE


@pytest.mark.parametrize(
    ("prompt", "cycles"),
    (
        ("do one push-up", 1.0),
        ("do three push-ups", 3.0),
        ("do 5 push-ups", 5.0),
    ),
)
def test_push_ups_use_palm_and_toe_support_with_requested_cycles(
    prompt: str,
    cycles: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text=prompt,
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    pose = program.primitives[0]
    assert program.intent is Intent.FULL_BODY
    assert pose.body is not None
    assert pose.body.pose.support_mode is BodySupportMode.PLANK
    assert pose.body.cycles == pytest.approx(cycles)
    assert {target.hand.value for target in pose.effectors} == {"left", "right"}
    assert clip.success, clip.failure
    assert clip.metrics["horizontal_pose_variant"] == "plank"
    assert clip.metrics["requested_push_up_cycles"] == pytest.approx(cycles)
    assert clip.metrics["push_up_vertical_excursion_m"] >= 0.06
    assert clip.metrics["measured_push_up_cycles"] == int(cycles)
    assert clip.metrics["push_up_palm_height_range_m"] <= 0.045
    assert clip.metrics["push_up_toe_support_clearance_m"] <= 0.035
    assert clip.metrics["push_up_toe_height_range_m"] <= 0.015
    assert clip.metrics["push_up_toe_position_range_m"] <= 0.02
    assert clip.metrics["push_up_min_knee_extension_deg"] >= 150.0
    assert clip.metrics["horizontal_pose_contact_point_count"] >= 2


def test_static_plank_holds_straight_palm_and_toe_support_without_cycles() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="hold a plank for 4 seconds",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    plank = program.primitives[0]
    assert plank.body is not None
    assert plank.body.pose.support_mode is BodySupportMode.PLANK
    assert plank.body.cycles == pytest.approx(0.0)
    assert plank.parameters.duration_s == pytest.approx(4.0)
    assert clip.success, clip.failure
    assert clip.metrics["measured_push_up_cycles"] == 0
    assert clip.metrics["plank_hold_vertical_range_m"] <= 0.025
    assert clip.metrics["push_up_palm_height_range_m"] <= 0.045
    assert clip.metrics["push_up_toe_position_range_m"] <= 0.02
    assert clip.metrics["push_up_min_knee_extension_deg"] >= 150.0


@pytest.mark.parametrize(
    ("prompt", "cycles"),
    (
        ("do a jumping jack", 1),
        ("do three jumping jacks", 3),
    ),
)
def test_jumping_jacks_synchronize_overhead_arms_and_lateral_feet(
    prompt: str,
    cycles: int,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    jack = program.primitives[0]
    assert jack.body is not None
    assert jack.body.action is BodyAction.JUMP
    assert jack.body.cycles == pytest.approx(float(cycles))
    assert jack.body.pose.left_foot_shift_x_m > 0.0
    assert jack.body.pose.right_foot_shift_x_m < 0.0
    assert clip.success, clip.failure
    assert clip.metrics["requested_jumping_jack_cycles"] == pytest.approx(cycles)
    assert clip.metrics["measured_jumping_jack_cycles"] == cycles
    assert clip.metrics["jumping_jack_foot_spread_excursion_m"] >= 0.30
    assert clip.metrics["jumping_jack_max_wrist_height_m"] >= 1.65


@pytest.mark.parametrize(
    ("prompt", "cycles", "axis", "sign"),
    (
        ("crawl forward two steps", 2.0, "z", 1.0),
        ("crawl backward three steps", 3.0, "z", -1.0),
        ("crawl to the left two steps", 2.0, "x", 1.0),
    ),
)
def test_crawl_uses_alternating_quadruped_support_and_retains_travel(
    prompt: str,
    cycles: float,
    axis: str,
    sign: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    crawl = program.primitives[0]
    assert program.intent is Intent.FULL_BODY
    assert crawl.body is not None
    assert crawl.body.pose.support_mode is BodySupportMode.QUADRUPED
    assert crawl.body.cycles == pytest.approx(cycles)
    assert crawl.trajectory.value == "oscillate"
    assert clip.success, clip.failure
    assert clip.metrics["horizontal_pose_variant"] == "quadruped"
    assert clip.metrics["horizontal_pose_contact_point_count"] >= 4
    assert clip.metrics["requested_crawl_cycles"] == pytest.approx(cycles)
    assert clip.metrics["crawl_hand_alternation_range_m"] >= 0.10
    first = clip.frames[0].bones["hips"].position
    last = clip.frames[-1].bones["hips"].position
    assert first is not None and last is not None
    displacement = getattr(last, axis) - getattr(first, axis)
    assert displacement * sign > 0.20
    assert clip.metrics["crawl_root_displacement_m"] > 0.20
