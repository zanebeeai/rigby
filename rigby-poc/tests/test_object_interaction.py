from __future__ import annotations

import numpy as np

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
    Intent,
    ObjectAction,
    ObjectInteractionStyle,
    ParameterOverrides,
    PlanRequest,
    PrimitiveKind,
    default_scene,
)
from rigby_poc.planner import OpenAIPlanner, plan_motion

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _program(text: str):
    scene = default_scene()
    outcome = plan_motion(PlanRequest(text=text, scene=scene, provider="offline"))
    OpenAIPlanner._validate_semantics(outcome.program, scene)
    return scene, outcome.program


def test_physical_throw_is_disambiguated_from_gesture_and_strike_idioms() -> None:
    scene = default_scene()
    throw = plan_motion(
        PlanRequest(text="toss the block gently to the left with your left hand", scene=scene, provider="offline")
    ).program
    shaka = plan_motion(
        PlanRequest(text="throw up a hang-ten sign", scene=scene, provider="offline")
    ).program
    hook = plan_motion(
        PlanRequest(text="throw a left hook", scene=scene, provider="offline")
    ).program

    assert throw.intent == Intent.OBJECT_INTERACTION
    assert throw.object_action == ObjectAction.THROW
    assert throw.hand == Hand.LEFT
    assert throw.object_motion is not None
    assert throw.object_motion.style == ObjectInteractionStyle.TOSS
    assert throw.object_motion.direction_x > 0.0
    assert shaka.intent == Intent.GESTURE
    assert hook.intent == Intent.STRIKE


def test_throw_has_contact_attachment_release_and_ballistic_flight() -> None:
    scene, program = _program("throw the block forward")
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))

    assert clip.success, clip.failure
    assert [item["state"] for item in clip.metrics["object_lifecycle"]] == [
        "free",
        "attached",
        "released",
        "ballistic",
    ]
    assert clip.metrics["palm_relative_object_slip_m"] < 0.005
    assert clip.metrics["object_max_step_m"] <= clip.metrics["object_max_step_reference_m"]
    assert clip.metrics["landing_height_error_m"] < 0.01

    ranges = {item["kind"]: item for item in clip.metrics["phase_ranges_s"]}
    flight = ranges[PrimitiveKind.FLIGHT.value]
    flight_frames = [
        frame
        for frame in clip.frames
        if flight["start_s"] - 1e-8 <= frame.time_s <= flight["end_s"] + 1e-8
    ]
    positions = np.asarray(
        [frame.objects["block"].translation.as_list() for frame in flight_frames],
        dtype=float,
    )
    assert positions[-1, 2] - positions[0, 2] > 0.75
    assert positions[:, 1].max() - positions[0, 1] > 0.30
    assert abs(positions[-1, 1] - program.object_motion.landing_height_m) < 0.01


def test_catch_intercepts_then_absorbs_and_retains_object() -> None:
    scene, program = _program("catch the block high with your left hand")
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))

    assert clip.success, clip.failure
    assert program.object_action == ObjectAction.CATCH
    assert clip.metrics["catch_intercept_error_m"] < 0.015
    assert clip.metrics["palm_relative_object_slip_m"] < 0.005
    assert clip.metrics["catch_time_s"] is not None
    states = [item["state"] for item in clip.metrics["object_lifecycle"]]
    assert states == ["free", "ballistic", "attached"]
    assert [item.kind for item in program.primitives][-3:] == [
        PrimitiveKind.ABSORB,
        PrimitiveKind.HOLD,
        PrimitiveKind.RECOVER,
    ]


def test_object_controls_are_parametric_compile_overrides() -> None:
    scene, program = _program("throw the block forward")
    clip = compile_motion(
        CompileRequest(
            scene=scene,
            program=program,
            parameter_overrides=ParameterOverrides(
                object_distance_m=1.20,
                object_apex_height_m=0.55,
                object_spin_turns=1.25,
            ),
            persist=False,
        )
    )

    assert clip.success, clip.failure
    assert clip.parametric_observables["object_flight_distance_m"] == 1.20
    assert clip.parametric_observables["object_apex_height_m"] == 0.55
    assert clip.parametric_observables["object_spin_turns"] == 1.25


def test_cross_hand_handoff_transfers_ownership_after_dual_contact() -> None:
    scene, program = _program(
        "pass the block from your right hand to your left hand"
    )
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.OBJECT_INTERACTION
    assert program.object_action is ObjectAction.HANDOFF
    assert program.hand is Hand.RIGHT
    assert program.hands == [Hand.RIGHT, Hand.LEFT]
    assert clip.success, clip.failure
    assert clip.metrics["handoff_source_hand"] == "right"
    assert clip.metrics["handoff_receiver_hand"] == "left"
    assert clip.metrics["handoff_receiver_retained"] is True
    assert clip.metrics["handoff_dual_contact_duration_s"] >= 0.08
    assert clip.metrics["handoff_attachment_slip_m"] <= 0.005
    assert clip.metrics["object_max_step_m"] <= 0.08
    assert clip.metrics["discontinuities"] == 0
    assert [item["state"] for item in clip.metrics["object_lifecycle"]] == [
        "free",
        "source_attached",
        "dual_contact",
        "receiver_attached",
    ]
    assert {contact.hand for contact in clip.contacts} == {Hand.RIGHT, Hand.LEFT}


def test_handoff_target_only_uses_the_opposite_hand_as_source() -> None:
    scene, program = _program("pass the block to your right hand")
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.hand is Hand.LEFT
    assert program.hands == [Hand.LEFT, Hand.RIGHT]
    assert clip.success, clip.failure
    assert clip.metrics["handoff_receiver_retained"] is True


def test_best_of_five_varies_the_visible_object_path() -> None:
    scene, program = _program("throw the block forward")
    recipes = candidate_recipes(program.primitives[0].parameters, program.intent)
    assert len(recipes) == 5
    descriptors = []
    for index, recipe in enumerate(recipes):
        candidate = apply_recipe(program, recipe, seed_offset=index + 1)
        clip = compile_motion(CompileRequest(scene=scene, program=candidate, persist=False))
        assert clip.success, (recipe.name, clip.failure)
        descriptors.append(motion_perceptual_descriptor(clip, candidate.hand))

    comparisons = [
        compare_perceptual_descriptors(descriptors[0], descriptor)
        for descriptor in descriptors[1:]
    ]
    assert any(item["maximum_object_separation_m"] >= 0.08 for item in comparisons)
    assert len(
        {
            round(float(item["object_samples"][2]["position_world_m"][1]), 3)
            for item in descriptors
        }
    ) >= 3
