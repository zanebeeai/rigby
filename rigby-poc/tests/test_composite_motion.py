from __future__ import annotations

from evals.capture import phase_sampling_points
from evals.flywheel import (
    apply_recipe,
    candidate_recipes,
    compare_perceptual_descriptors,
    motion_perceptual_descriptor,
)
from rigby_poc.compiler import compile_motion
from rigby_poc.models import (
    CompileRequest,
    Hand,
    HandShape,
    Intent,
    PlanRequest,
    PrimitiveKind,
    TrajectoryKind,
    TrajectoryPlane,
    default_scene,
)
from rigby_poc.planner import (
    GenericEffectorSelection,
    GenericSegmentSelection,
    OfflinePlanner,
    OpenAIPlanner,
    PlannerSelection,
)


TRAVEL_PROMPT = (
    'roll your forearms around eachother repeatedly, as if you are indicating '
    'the "travel" foul in basketball'
)


def _offline_program(prompt: str = TRAVEL_PROMPT):
    scene = default_scene()
    outcome = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    )
    return scene, outcome.program


def test_travel_foul_decomposes_into_concurrent_phase_opposed_forearms() -> None:
    _, program = _offline_program()

    assert program.intent == Intent.COMPOSITE
    assert set(program.hands) == {Hand.LEFT, Hand.RIGHT}
    assert [primitive.kind for primitive in program.primitives] == [
        PrimitiveKind.MOVE,
        PrimitiveKind.CYCLE,
        PrimitiveKind.RECOVER,
    ]
    assert program.primitives[0].label == "parallel_forearm_travel_setup"
    cycle = program.primitives[1]
    assert cycle.label == "parallel_forearm_travel_cycle"
    assert cycle.trajectory == TrajectoryKind.CIRCLE
    assert cycle.trajectory_plane == TrajectoryPlane.FRONTAL
    assert cycle.parameters.trajectory_cycles == 3.0
    assert cycle.parameters.trajectory_amplitude_m == 0.09
    assert cycle.parameters.axial_rotation_amplitude > 0.5
    assert {target.phase_offset_cycles for target in cycle.effectors} == {0.0, 0.5}
    assert all(target.hand_shape == HandShape.FIST for target in cycle.effectors)


def test_travel_foul_compiles_both_hands_with_full_fov_and_recovery() -> None:
    scene, program = _offline_program()
    clip = compile_motion(CompileRequest(scene=scene, program=program))

    assert clip.success, clip.failure
    assert clip.metrics["structural_valid"] is True
    assert clip.metrics["active_hands"] == ["left", "right"]
    assert clip.metrics["active_hand_visibility_fraction"] == 1.0
    assert clip.metrics["self_collision_frames"] == 0
    assert clip.metrics["trajectory_cycles"] == 3.0
    assert clip.metrics["forearm_rotation_cycles"] >= 2.5
    assert clip.metrics["forearm_rotation_amplitude_rad"] >= 0.35
    assert clip.metrics["wrist_flexion_cycles"] == 0.0
    assert clip.metrics["wrist_deviation_cycles"] == 0.0
    assert clip.metrics["parallel_forearm_max_axis_error_deg"] <= 20.0
    assert clip.metrics["parallel_forearm_max_frontal_axis_error_deg"] <= 8.0
    assert clip.metrics["parallel_forearm_p95_frontal_axis_error_deg"] <= 8.0
    assert clip.metrics["parallel_forearm_minimum_separation_m"] >= 0.025
    assert clip.metrics["parallel_forearm_minimum_hand_separation_m"] >= 0.28
    assert clip.metrics["travel_wheel_cross_body_fraction"] >= 0.95
    assert clip.metrics["travel_wheel_maximum_opposite_elbow_distance_m"] <= 0.20
    assert clip.metrics["travel_wheel_vertical_order_range_m"] >= 0.30
    assert clip.metrics["travel_wheel_depth_order_range_m"] >= 0.16
    assert clip.frames[0].bones["leftHand"].rotation == clip.frames[-1].bones["leftHand"].rotation
    assert clip.frames[0].bones["rightHand"].rotation == clip.frames[-1].bones["rightHand"].rotation


def test_explicit_parallel_travel_foul_language_uses_the_coupled_constraint() -> None:
    prompt = (
        "roll your forearms around eachother repeatedly as they remain parallel "
        "to one another, as if you're indicating the travel foul in basketball"
    )
    scene, program = _offline_program(prompt)
    clip = compile_motion(CompileRequest(scene=scene, program=program))

    assert program.primitives[1].label == "parallel_forearm_travel_cycle"
    assert clip.success, clip.failure
    assert clip.metrics["parallel_forearm_max_frontal_axis_error_deg"] <= 8.0
    assert clip.metrics["parallel_forearm_minimum_separation_m"] >= 0.025
    assert clip.metrics["travel_wheel_cross_body_fraction"] == 1.0
    assert clip.metrics["travel_wheel_vertical_order_range_m"] >= 0.30


def test_unseen_bilateral_language_compiles_through_the_same_schema() -> None:
    prompts = (
        "raise both open hands above your head then lower them",
        "move both fists side to side in opposite directions three times",
        "circle both hands forward twice",
    )
    for prompt in prompts:
        scene, program = _offline_program(prompt)
        clip = compile_motion(CompileRequest(scene=scene, program=program))
        assert program.intent == Intent.COMPOSITE, prompt
        assert clip.success, (prompt, clip.failure)


def test_structured_planner_can_author_an_unfamiliar_composite_sequence() -> None:
    request = PlanRequest(
        text="sweep both open hands outward, pause, then bring them inward",
        scene=default_scene(),
        provider="openai",
    )
    selection = PlannerSelection(
        intent=Intent.COMPOSITE,
        composition_segments=[
            GenericSegmentSelection(
                label="outward_sweep",
                duration_s=0.8,
                trajectory=TrajectoryKind.ARC,
                trajectory_plane=TrajectoryPlane.FRONTAL,
                trajectory_amplitude_m=0.04,
                effectors=[
                    GenericEffectorSelection(
                        hand=Hand.LEFT,
                        target_x=0.75,
                        target_y=0.1,
                        target_z=0.5,
                        hand_shape=HandShape.OPEN,
                    ),
                    GenericEffectorSelection(
                        hand=Hand.RIGHT,
                        target_x=-0.75,
                        target_y=0.1,
                        target_z=0.5,
                        hand_shape=HandShape.OPEN,
                    ),
                ],
            ),
            GenericSegmentSelection(
                label="inward_return",
                duration_s=0.9,
                trajectory=TrajectoryKind.LINEAR,
                effectors=[
                    GenericEffectorSelection(hand=Hand.LEFT, target_x=0.2, target_y=0.1, target_z=0.45),
                    GenericEffectorSelection(hand=Hand.RIGHT, target_x=-0.2, target_y=0.1, target_z=0.45),
                ],
            ),
        ],
    )

    program = OpenAIPlanner._expand(selection, request, variation_seed=42)
    OpenAIPlanner._validate_semantics(program, request.scene)
    clip = compile_motion(CompileRequest(scene=request.scene, program=program))

    assert clip.success, clip.failure
    assert program.primitives[-1].kind == PrimitiveKind.RECOVER
    assert program.seed == 42


def test_composite_candidate_set_changes_visible_paths_without_changing_semantics() -> None:
    scene, program = _offline_program()
    recipes = candidate_recipes(program.primitives[0].parameters, Intent.COMPOSITE)
    descriptors = []
    hashes = set()
    for index, recipe in enumerate(recipes, start=1):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        clip = compile_motion(CompileRequest(scene=scene, program=candidate))
        assert clip.success, (recipe.name, clip.failure)
        assert [item.parameters.trajectory_cycles for item in candidate.primitives] == [
            item.parameters.trajectory_cycles for item in program.primitives
        ]
        descriptor = motion_perceptual_descriptor(clip, candidate.hands)
        assert {sample["hand"] for sample in descriptor["samples"]} == {"left", "right"}
        descriptors.append(descriptor)
        hashes.add(tuple(frame.model_dump_json() for frame in clip.frames))

    assert len(recipes) == 5
    assert len(hashes) == 5
    assert sum(
        compare_perceptual_descriptors(descriptors[0], other)["perceptually_distinct"]
        for other in descriptors[1:]
    ) >= 3


def test_composite_capture_samples_every_cycle_extreme_in_full_sequence() -> None:
    scene, program = _offline_program()
    clip = compile_motion(CompileRequest(scene=scene, program=program))
    payload = {
        "program": program.model_dump(mode="json"),
        "clip": clip.model_dump(mode="json"),
    }

    points = phase_sampling_points(payload)
    cycle_points = [point for point in points if point["phase"] == "cycle"]

    assert len(cycle_points) == 8
    assert points[0]["phase"] == "move"
    assert points[-1]["phase"] == "recover"
