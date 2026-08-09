from __future__ import annotations

from evals.flywheel import (
    apply_recipe,
    apply_repair,
    candidate_recipes,
    compare_perceptual_descriptors,
    motion_perceptual_descriptor,
    single_sample_baseline_index,
)
from evals.heldout import heldout_hangten_cases
from rigby_poc.compiler import compile_motion
from rigby_poc.judge import RepairPatch
from rigby_poc.models import CompileRequest, Intent, PlanRequest, PrimitiveKind, default_scene
from rigby_poc.planner import OfflinePlanner


def _program():
    scene = default_scene()
    return OfflinePlanner().plan(
        PlanRequest(
            text=(
                "With your right hand, make a quick energetic shaka high and extended, held outward "
                "from the body. Pitch the wrist up, yaw it outward, and roll it clockwise."
            ),
            scene=scene,
            provider="offline",
        )
    ).program


def test_best_of_five_recipes_are_distinct_and_preserve_direction_semantics() -> None:
    program = _program()
    recipes = candidate_recipes(program.primitives[0].parameters)
    assert len(recipes) == 5
    assert len({recipe.name for recipe in recipes}) == 5
    candidates = [apply_recipe(program, recipe, seed_offset=index) for index, recipe in enumerate(recipes, 1)]
    parameter_sets = {
        candidate.primitives[0].parameters.model_dump_json() for candidate in candidates
    }
    assert len(parameter_sets) == 5
    original = program.primitives[0].parameters
    for candidate in candidates:
        present = next(item for item in candidate.primitives if item.kind == PrimitiveKind.PRESENT).parameters
        for field in ("arm_height", "arm_depth", "lateral_offset", "wrist_pitch", "wrist_yaw", "wrist_roll"):
            before = float(getattr(original, field))
            after = float(getattr(present, field))
            if abs(before) >= 0.20:
                assert before * after > 0.0, (field, before, after)


def test_positive_finger_splay_recipe_expands_both_outer_digit_parameters() -> None:
    program = _program()
    wide = apply_recipe(program, candidate_recipes(program.primitives[0].parameters)[1], seed_offset=2)
    parameters = wide.primitives[0].parameters
    assert parameters.finger_splay > program.primitives[0].parameters.finger_splay
    assert parameters.thumb_curl < program.primitives[0].parameters.thumb_curl
    assert parameters.little_curl < program.primitives[0].parameters.little_curl


def test_single_sample_baseline_is_precommitted_and_not_always_canonical() -> None:
    prompts = [case["prompt"] for case in heldout_hangten_cases()]
    first = [single_sample_baseline_index(prompt) for prompt in prompts]
    second = [single_sample_baseline_index(prompt) for prompt in prompts]
    assert first == second
    assert all(1 <= index <= 5 for index in first)
    assert len(set(first)) == 5


def test_repair_patch_applies_to_object_pickup_phases() -> None:
    program = OfflinePlanner().plan(
        PlanRequest(
            text="Grab the block in front of you with your right hand.",
            scene=default_scene(),
            provider="offline",
        )
    ).program
    repair = RepairPatch(
        arm_height_delta=0.1,
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
        present_duration_scale=1.1,
        hold_duration_scale=1.2,
        shake_duration_scale=0.9,
        recover_duration_scale=1.05,
        easing_delta=0.05,
        rationale="Raise the pickup and make contact easier to read.",
    )
    repaired = apply_repair(program, repair)
    before = {item.kind: item.parameters for item in program.primitives}
    after = {item.kind: item.parameters for item in repaired.primitives}

    assert after[PrimitiveKind.REACH].arm_height > before[PrimitiveKind.REACH].arm_height
    assert after[PrimitiveKind.REACH].duration_s > before[PrimitiveKind.REACH].duration_s
    assert after[PrimitiveKind.CONTACT].duration_s > before[PrimitiveKind.CONTACT].duration_s
    assert after[PrimitiveKind.LIFT].duration_s < before[PrimitiveKind.LIFT].duration_s
    assert after[PrimitiveKind.RECOVER].arm_height == before[PrimitiveKind.RECOVER].arm_height


def test_pickup_recipes_supply_five_rankable_structurally_valid_candidates() -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(
            text="Grab the block in front of you with your right hand.",
            scene=scene,
            provider="offline",
        )
    ).program
    recipes = candidate_recipes(program.primitives[0].parameters, Intent.GRAB)
    descriptors = []

    for index, recipe in enumerate(recipes, 1):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        clip = compile_motion(CompileRequest(scene=scene, program=candidate, persist=False))
        assert clip.success
        assert clip.metrics["structural_valid"] is True
        descriptor = motion_perceptual_descriptor(clip, candidate.hand)
        assert all(
            compare_perceptual_descriptors(prior, descriptor)["perceptually_distinct"]
            for prior in descriptors
        )
        descriptors.append(descriptor)

    assert len(descriptors) == 5
