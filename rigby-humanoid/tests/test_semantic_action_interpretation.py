from __future__ import annotations

import pytest

from evals.flywheel import apply_recipe, candidate_recipes
from rigby_poc.compiler import compile_motion
from rigby_poc.models import (
    CompileRequest,
    Hand,
    HandShape,
    Intent,
    MotionPrimitive,
    MotionProgram,
    PlanRequest,
    PrimitiveKind,
    PrimitiveParameters,
    TrajectoryKind,
    TrajectoryPlane,
    default_scene,
)
from rigby_poc.planner import OfflinePlanner, OpenAIPlanner, PlannerSelection

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


@pytest.mark.parametrize(
    ("prompt", "hands", "cycles", "amplitude"),
    (
        ("wave hello", (Hand.RIGHT,), 3.0, 0.10),
        ("Wave goodbye with your left hand twice", (Hand.LEFT,), 2.0, 0.10),
        (
            "give a small wave with both hands four times",
            (Hand.LEFT, Hand.RIGHT),
            4.0,
            0.065,
        ),
        ("slowly wave to the crowd", (Hand.RIGHT,), 3.0, 0.10),
        (
            "quickly wave your right hand side to side five times",
            (Hand.RIGHT,),
            5.0,
            0.10,
        ),
    ),
)
def test_greeting_language_expands_to_observable_cyclic_semantics(
    prompt: str,
    hands: tuple[Hand, ...],
    cycles: float,
    amplitude: float,
) -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program

    assert program.intent == Intent.COMPOSITE
    assert tuple(program.hands) == hands
    assert [item.kind for item in program.primitives] == [
        PrimitiveKind.MOVE,
        PrimitiveKind.CYCLE,
        PrimitiveKind.RECOVER,
    ]
    assert program.primitives[0].label == "hello_wave_setup"
    cycle = program.primitives[1]
    assert cycle.label == "hello_wave_cycles"
    assert cycle.trajectory == TrajectoryKind.OSCILLATE
    assert cycle.trajectory_plane == TrajectoryPlane.FRONTAL
    assert cycle.parameters.trajectory_cycles == cycles
    assert cycle.parameters.trajectory_amplitude_m == pytest.approx(amplitude)
    assert {target.hand for target in cycle.effectors} == set(hands)
    assert all(target.hand_shape == HandShape.OPEN for target in cycle.effectors)
    assert any(
        assertion.name == "hello_wave_trajectory_reversals"
        for assertion in program.assertions
    )


@pytest.mark.parametrize(
    "prompt",
    (
        "wave hello",
        "Wave goodbye with your left hand twice",
        "give a small wave with both hands four times",
    ),
)
def test_greeting_wave_compiles_with_measured_reversals(prompt: str) -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    required = next(
        assertion.threshold
        for assertion in program.assertions
        if assertion.name == "hello_wave_trajectory_reversals"
    )
    assert clip.success, clip.failure
    assert clip.metrics["semantic_cycle_action"] == "hello_wave"
    assert clip.metrics["semantic_cycle_min_reversal_count"] >= required
    assert clip.metrics["semantic_cycle_min_excursion_m"] >= 0.09
    assert clip.metrics["active_hand_visibility_fraction"] == pytest.approx(1.0)


def test_wave_candidate_flywheel_preserves_the_motion_obligation() -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text="wave hello", scene=scene, provider="offline")
    ).program
    recipes = candidate_recipes(
        program.primitives[0].parameters,
        Intent.COMPOSITE,
    )

    for index, recipe in enumerate(recipes, start=1):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        clip = compile_motion(
            CompileRequest(scene=scene, program=candidate, persist=False)
        )
        assert clip.success, (recipe.name, clip.failure)
        assert candidate.primitives[1].kind == PrimitiveKind.CYCLE
        assert candidate.primitives[1].parameters.trajectory_cycles == 3.0
        assert clip.metrics["semantic_cycle_min_reversal_count"] >= 4

    assert len(recipes) == 5


def test_model_cannot_collapse_a_motion_verb_to_a_static_open_palm() -> None:
    scene = default_scene()
    request = PlanRequest(text="wave hello", scene=scene, provider="openai")
    static_selection = PlannerSelection(
        intent=Intent.GESTURE,
        hand=Hand.RIGHT,
        hand_shape=HandShape.OPEN,
    )

    reconciled = OpenAIPlanner._expand(
        static_selection,
        request,
        variation_seed=17,
    )

    assert reconciled.intent == Intent.COMPOSITE
    assert reconciled.primitives[1].label == "hello_wave_cycles"
    OpenAIPlanner._validate_semantics(reconciled, scene)

    static_program = MotionProgram(
        source_text="wave hello",
        intent=Intent.GESTURE,
        hand=Hand.RIGHT,
        primitives=[
            MotionPrimitive(
                kind=PrimitiveKind.PRESENT,
                hand_shape=HandShape.OPEN,
                parameters=PrimitiveParameters(duration_s=0.5),
            ),
            MotionPrimitive(
                kind=PrimitiveKind.HOLD,
                hand_shape=HandShape.OPEN,
                parameters=PrimitiveParameters(duration_s=0.8),
            ),
            MotionPrimitive(
                kind=PrimitiveKind.RECOVER,
                hand_shape=HandShape.OPEN,
                parameters=PrimitiveParameters(duration_s=0.5),
            ),
        ],
    )
    with pytest.raises(ValueError, match="requires 3 observable frontal oscillation cycles"):
        OpenAIPlanner._validate_semantics(static_program, scene)


def test_object_and_static_shape_language_are_not_misclassified_as_greeting_waves() -> None:
    scene = default_scene()
    object_program = OfflinePlanner().plan(
        PlanRequest(text="wave a flag", scene=scene, provider="offline")
    ).program
    open_palm = OfflinePlanner().plan(
        PlanRequest(text="show an open palm", scene=scene, provider="offline")
    ).program

    assert object_program.intent == Intent.UNSUPPORTED
    assert open_palm.intent == Intent.GESTURE
    assert all(
        primitive.kind != PrimitiveKind.CYCLE
        for primitive in open_palm.primitives
    )


def test_concurrent_walk_and_wave_uses_the_same_semantic_contract() -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(
            text="walk forward three steps while waving your right hand three times",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent == Intent.FULL_BODY
    assert any(
        assertion.name == "hello_wave_trajectory_reversals"
        for assertion in program.assertions
    )
    assert clip.success, clip.failure
    assert clip.metrics["semantic_cycle_min_reversal_count"] >= 4
    assert clip.metrics["semantic_cycle_min_excursion_m"] >= 0.09
