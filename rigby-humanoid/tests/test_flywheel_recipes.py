from __future__ import annotations

from evals.flywheel import (
    adaptive_timing_recipes,
    apply_recipe,
    candidate_recipe_pool,
    candidate_recipes,
    compare_perceptual_descriptors,
    describe_program_delta,
    motion_perceptual_descriptor,
    select_diverse_candidate_batch,
    select_five_way_winner,
)
from evals.heldout import heldout_complex_hangten_cases
from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, Intent, PlanRequest, PrimitiveParameters, default_scene
from rigby_poc.planner import plan_motion

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _candidate(result_id: str, *, duration_s: float, wrist_x: float = 0.0) -> dict:
    return {
        "candidate_index": int(result_id.removeprefix("candidate-")),
        "recipe": {"name": f"recipe-{result_id}"},
        "result_id": result_id,
        "compile_success": True,
        "structural_valid": True,
        "motion_sha256": f"hash-{result_id}",
        "perceptual_descriptor": {
            "duration_s": duration_s,
            "phase_durations_s": {"recover": 0.5},
            "samples": [
                {
                    "wrist_world_m": [wrist_x, 1.0, 0.2],
                    "elbow_world_m": [wrist_x, 1.2, 0.1],
                    "hand_world_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
        },
    }


def test_diversity_is_a_batch_objective_not_a_fatal_candidate_gate() -> None:
    candidates = [
        _candidate(f"candidate-{index}", duration_s=2.0 + index * 0.01)
        for index in range(1, 6)
    ]
    selected, audit = select_diverse_candidate_batch(
        candidates,
        baseline_result_id="candidate-3",
    )
    assert len(selected) == 5
    assert audit["status"] == "complete"
    assert audit["relaxed_diversity"] is True
    assert audit["hard_pair_count"] < audit["total_pair_count"]
    assert audit["baseline_included"] is True


def test_batch_selector_prefers_a_fully_separated_set_when_available() -> None:
    candidates = [
        _candidate(f"candidate-{index}", duration_s=2.0, wrist_x=index * 0.05)
        for index in range(1, 7)
    ]
    selected, audit = select_diverse_candidate_batch(
        candidates,
        baseline_result_id="candidate-2",
    )
    assert len(selected) == 5
    assert audit["all_pairs_above_threshold"] is True
    assert audit["hard_pair_count"] == audit["total_pair_count"] == 10
    assert audit["baseline_included"] is True


def test_every_supported_motion_family_has_global_safe_replenishment() -> None:
    recipes = adaptive_timing_recipes()
    assert len(recipes) >= 5
    assert len({recipe.name for recipe in recipes}) == len(recipes)
    assert all(recipe.deltas == {} and recipe.scales == {} for recipe in recipes)
    assert all(min(recipe.duration_scales.values()) > 1.0 for recipe in recipes)
    adaptive_names = {recipe.name for recipe in recipes}
    for intent in (Intent.GESTURE, Intent.GRAB, Intent.STRIKE):
        pool_names = {
            recipe.name for recipe in candidate_recipe_pool(PrimitiveParameters(), intent)
        }
        assert adaptive_names <= pool_names


def test_five_way_no_accepted_candidate_requests_repair_instead_of_crashing() -> None:
    candidates = [
        {"result_id": f"candidate-{index}", "accepted": False, "judgment": {"overall": 2}}
        for index in range(5)
    ]
    winner, reason = select_five_way_winner(
        candidates,
        [],
        winner_index=None,
        baseline_result_id="candidate-2",
    )
    assert winner is None
    assert reason == "no_candidate_passed_visual_threshold"


def test_five_way_visual_tie_uses_precommitted_accepted_baseline() -> None:
    candidates = [
        {"result_id": f"candidate-{index}", "accepted": True, "judgment": {"overall": 4}}
        for index in range(5)
    ]
    winner, reason = select_five_way_winner(
        candidates,
        candidates,
        winner_index=None,
        baseline_result_id="candidate-2",
    )
    assert winner is candidates[2]
    assert reason == "no_clear_winner_fallback_baseline"


def test_clearance_recipes_preserve_inward_and_extended_directions() -> None:
    prompt = (
        "With your right hand, make a quick playful shaka low and fully extended, drawn inward. "
        "Keep wrist pitch neutral, yaw it outward, and roll it counterclockwise."
    )
    program = plan_motion(PlanRequest(text=prompt, scene=default_scene(), provider="offline")).program
    original = program.primitives[0].parameters
    recipes = {recipe.name: recipe for recipe in candidate_recipe_pool(original)}
    for name in ("torso_clearance", "inward_clearance"):
        candidate = apply_recipe(program, recipes[name], seed_offset=1)
        parameters = candidate.primitives[0].parameters
        assert parameters.arm_depth * original.arm_depth > 0
        assert parameters.lateral_offset * original.lateral_offset > 0
        assert abs(parameters.arm_depth) >= 0.5
        assert parameters.elbow_swivel < original.elbow_swivel


def test_primary_recipes_are_visibly_differentiated_and_preserve_semantics() -> None:
    prompt = (
        "With your right hand, make a quick playful shaka high and fully extended, held outward. "
        "Pitch the wrist up, yaw it outward, and roll it clockwise."
    )
    program = plan_motion(PlanRequest(text=prompt, scene=default_scene(), provider="offline")).program
    original = program.primitives[0].parameters
    recipes = candidate_recipes(original)
    assert [recipe.name for recipe in recipes] == [
        "canonical",
        "semantic_emphasis",
        "grounded_natural",
        "expressive_arc",
        "camera_silhouette",
    ]
    for index, recipe in enumerate(recipes[1:], start=2):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        delta = describe_program_delta(program, candidate)
        assert delta["materially_changed_dimension_count"] >= 4
        parameters = candidate.primitives[0].parameters
        for name in (
            "arm_height",
            "arm_depth",
            "lateral_offset",
            "wrist_pitch",
            "wrist_yaw",
            "wrist_roll",
        ):
            assert getattr(parameters, name) * getattr(original, name) > 0


def test_primary_recipes_clear_pairwise_motion_diversity_gate() -> None:
    prompt = (
        "With your left hand, make a balanced playful shaka at chest height and fully extended, "
        "centered on the arm. Pitch the wrist up, keep yaw neutral, and roll it counterclockwise."
    )
    scene = default_scene()
    program = plan_motion(PlanRequest(text=prompt, scene=scene, provider="offline")).program
    descriptors = []
    for index, recipe in enumerate(candidate_recipes(program.primitives[0].parameters), start=1):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        clip = compile_motion(CompileRequest(scene=scene, program=candidate))
        assert clip.success and clip.metrics["structural_valid"]
        descriptors.append(motion_perceptual_descriptor(clip, candidate.hand))

    for first_index, first in enumerate(descriptors):
        comparison_to_self = compare_perceptual_descriptors(first, first)
        assert not comparison_to_self["perceptually_distinct"]
        for second in descriptors[first_index + 1 :]:
            comparison = compare_perceptual_descriptors(first, second)
            assert comparison["perceptually_distinct"]
            assert comparison["passing_dimensions"]


def test_shake_candidates_record_reversals_recovery_and_phase_diversity() -> None:
    prompt = (
        'Throw up a "hang-ten" sign with your right hand with a swift motion; keep the middle '
        "three fingers as contracted as possible, shake the wrist rapidly back and forth three "
        "times, then return to default."
    )
    scene = default_scene()
    program = plan_motion(PlanRequest(text=prompt, scene=scene, provider="offline")).program
    recipes = candidate_recipes(program.primitives[0].parameters)
    descriptors = {}
    authored = {}
    for index, recipe in enumerate(recipes, start=1):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        shake = next(item for item in candidate.primitives if item.kind.value == "shake")
        clip = compile_motion(CompileRequest(scene=scene, program=candidate))
        assert clip.success and clip.metrics["structural_valid"]
        descriptor = motion_perceptual_descriptor(clip, candidate.hand)
        assert descriptor["shake_cycles"] == 3.0
        assert descriptor["shake_reversal_count"] >= 5
        assert descriptor["recovery_endpoint_wrist_error_m"] < 1e-6
        assert descriptor["recovery_endpoint_hand_error_rad"] < 1e-6
        assert len([sample for sample in descriptor["samples"] if sample["phase"] == "shake"]) == 6
        descriptors[recipe.name] = descriptor
        authored[recipe.name] = (
            shake.parameters.wrist_shake_amplitude,
            shake.parameters.duration_s,
            shake.parameters.wrist_shake_cycles,
        )
    assert len(set(authored.values())) == 5
    grounded_difference = compare_perceptual_descriptors(
        descriptors["canonical"], descriptors["grounded_natural"]
    )
    assert grounded_difference["perceptually_distinct"]
    assert any(name.startswith("shake_") for name in grounded_difference["passing_dimensions"])


def test_primary_recipes_reserve_forearm_twist_and_shake_kinematic_headroom() -> None:
    scene = default_scene()
    # These two cases previously exposed independent unsafe edges: the
    # two-cycle quick shake exceeded acceleration, while the high/outward
    # profile exhausted forearm twist before the oscillation was added.
    cases = {
        case["id"]: case
        for case in heldout_complex_hangten_cases()
        if case["id"] in {"cx02", "cx03"}
    }
    for case in cases.values():
        program = plan_motion(
            PlanRequest(text=case["prompt"], scene=scene, provider="offline")
        ).program
        for index, recipe in enumerate(candidate_recipes(program.primitives[0].parameters), start=1):
            candidate = apply_recipe(program, recipe, seed_offset=index)
            clip = compile_motion(CompileRequest(scene=scene, program=candidate))
            assert clip.success, (case["id"], recipe.name, clip.failure)
            assert clip.metrics["structural_valid"]
            assert clip.metrics["max_forearm_twist_rad"] <= 1.30 + 1e-8
