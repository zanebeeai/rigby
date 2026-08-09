from __future__ import annotations

import pytest

from evals.capture import motion_diagnostics, phase_sampling_points
from evals.flywheel import apply_recipe, candidate_recipes
from rigby_poc import compiler
from rigby_poc.compiler import compile_motion
from rigby_poc.models import (
    CompileRequest,
    Digit,
    Hand,
    HandShape,
    Intent,
    PlanRequest,
    PrimitiveKind,
    Quat,
    default_scene,
)
from rigby_poc.planner import OfflinePlanner, OpenAIPlanner, PlannerSelection
from rigby_poc.primitives import hand_pose


PROMPT = (
    "use your right thumb to one by one count each of the fingers on your "
    "right hand (while looking at it)"
)


@pytest.mark.parametrize(
    ("prompt", "hand", "digits", "gaze"),
    (
        (
            PROMPT,
            Hand.RIGHT,
            (Digit.INDEX, Digit.MIDDLE, Digit.RING, Digit.LITTLE),
            True,
        ),
        (
            "carefully count every finger on your left hand with your left "
            "thumb while watching your hand",
            Hand.LEFT,
            (Digit.INDEX, Digit.MIDDLE, Digit.RING, Digit.LITTLE),
            True,
        ),
        (
            "one by one tap your right thumb against your pinky, ring, "
            "middle, and index fingertips",
            Hand.RIGHT,
            (Digit.LITTLE, Digit.RING, Digit.MIDDLE, Digit.INDEX),
            False,
        ),
        (
            "touch each of the index and middle fingertips with your left "
            "thumb, one at a time",
            Hand.LEFT,
            (Digit.INDEX, Digit.MIDDLE),
            False,
        ),
    ),
)
def test_ordered_digit_language_expands_to_typed_contacts(
    prompt: str,
    hand: Hand,
    digits: tuple[Digit, ...],
    gaze: bool,
) -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program

    contacts = [
        primitive.intra_hand_contact
        for primitive in program.primitives
        if primitive.intra_hand_contact is not None
    ]
    assert program.intent == Intent.COMPOSITE
    assert program.hands == [hand]
    assert tuple(contact.target_digit for contact in contacts) == digits
    assert all(contact.hand == hand for contact in contacts)
    assert all(contact.driver_digit == Digit.THUMB for contact in contacts)
    assert len(program.primitives) == 2 * len(digits) + 2
    assert program.primitives[0].label in {
        "finger_count_prepare",
        "finger_taps_prepare",
    }
    assert program.primitives[-1].kind == PrimitiveKind.RECOVER
    assert any(
        assertion.name == "ordered_intra_hand_contacts"
        and assertion.threshold == len(digits)
        for assertion in program.assertions
    )
    gaze_primitives = [
        primitive
        for primitive in program.primitives[:-1]
        if primitive.gaze_target is not None
    ]
    assert bool(gaze_primitives) is gaze
    if gaze:
        assert all(primitive.gaze_target.hand == hand for primitive in gaze_primitives)


@pytest.mark.parametrize(
    "prompt",
    (
        PROMPT,
        "count every finger on your left hand with your left thumb while "
        "looking at your hand",
        "one by one tap your right thumb against your little, ring, and "
        "middle fingertips",
    ),
)
def test_dexterous_program_compiles_and_measures_real_contacts(prompt: str) -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    expected = [
        primitive.intra_hand_contact.target_digit.value
        for primitive in program.primitives
        if primitive.intra_hand_contact is not None
    ]
    assert clip.success, clip.failure
    assert clip.metrics["intra_hand_contact_expected_order"] == expected
    assert clip.metrics["intra_hand_contact_observed_order"] == expected
    assert clip.metrics["intra_hand_contact_count"] == len(expected)
    assert clip.metrics["intra_hand_minimum_release_separation_m"] >= 0.025
    assert all(
        record["minimum_distance_m"] <= record["maximum_distance_m"]
        and record["passed"]
        for record in clip.metrics["intra_hand_contact_records"]
    )
    if any(primitive.gaze_target for primitive in program.primitives):
        assert clip.metrics["gaze_max_endpoint_angle_deg"] <= 12.0
    assert clip.metrics["active_hand_visibility_fraction"] == pytest.approx(1.0)


def test_incidental_looking_cannot_replace_the_dexterous_action() -> None:
    scene = default_scene()
    request = PlanRequest(text=PROMPT, scene=scene, provider="openai")
    collapsed_selection = PlannerSelection(
        intent=Intent.GESTURE,
        hand=Hand.RIGHT,
        hand_shape=HandShape.PINCH,
    )

    reconciled = OpenAIPlanner._expand(
        collapsed_selection,
        request,
        variation_seed=19,
    )
    assert reconciled.intent == Intent.COMPOSITE
    assert len(
        [primitive for primitive in reconciled.primitives if primitive.intra_hand_contact]
    ) == 4
    OpenAIPlanner._validate_semantics(reconciled, scene)

    invalid = reconciled.model_copy(deep=True)
    for primitive in invalid.primitives:
        primitive.intra_hand_contact = None
    with pytest.raises(ValueError, match="ordered dexterous action requires"):
        OpenAIPlanner._validate_semantics(invalid, scene)


def test_compiler_rejects_a_static_open_hand_in_place_of_fingertip_contacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=PROMPT, scene=scene, provider="offline")
    ).program

    monkeypatch.setattr(
        compiler,
        "thumb_to_fingertip_pose",
        lambda hand, _target: hand_pose(
            hand,
            HandShape.OPEN,
            program.primitives[0].parameters,
        ),
    )
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert not clip.success
    assert clip.metrics["intra_hand_contact_count"] == 0
    assert "ordered fingertip contacts" in clip.failure.message


def test_compiler_rejects_head_motion_that_does_not_track_the_hand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=PROMPT, scene=scene, provider="offline")
    ).program
    monkeypatch.setattr(compiler, "_head_gaze_rotation", lambda _target: Quat())

    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert not clip.success
    assert clip.metrics["gaze_max_endpoint_angle_deg"] > 12.0
    assert "gaze does not track" in clip.failure.message


def test_all_five_candidates_preserve_digit_order_and_gaze() -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=PROMPT, scene=scene, provider="offline")
    ).program
    expected = ["index", "middle", "ring", "little"]
    recipes = candidate_recipes(program.primitives[0].parameters, program.intent)

    for index, recipe in enumerate(recipes, start=1):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        clip = compile_motion(
            CompileRequest(scene=scene, program=candidate, persist=False)
        )
        assert clip.success, (recipe.name, clip.failure)
        assert clip.metrics["intra_hand_contact_observed_order"] == expected
        assert clip.metrics["gaze_max_endpoint_angle_deg"] <= 12.0
    assert len(recipes) == 5


def test_dexterous_capture_contract_prioritizes_each_contact_without_payload_bloat() -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=PROMPT, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )
    payload = {
        "program": program.model_dump(mode="json"),
        "clip": clip.model_dump(mode="json"),
    }

    points = phase_sampling_points(payload)
    labels = [str(point["label"]) for point in points]
    diagnostics = motion_diagnostics(payload)["measurements"]
    assert labels == [
        "dexterous_hand_presented",
        "thumb_to_index",
        "thumb_to_middle",
        "thumb_to_ring",
        "thumb_to_little",
        "dexterous_recovered",
    ]
    assert diagnostics["ordered_intra_hand_contact_sequence"] == {
        "value": ["index", "middle", "ring", "little"],
        "required_sequence": ["index", "middle", "ring", "little"],
    }
