from __future__ import annotations

import math

import numpy as np
import pytest

from evals.capture import phase_sampling_points
from evals.flywheel import (
    _motion_hash,
    apply_recipe,
    candidate_recipe_pool,
    candidate_recipes,
    compare_perceptual_descriptors,
    motion_perceptual_descriptor,
    select_diverse_candidate_batch,
    single_sample_baseline_index,
)
from rigby_poc.compiler import compile_motion
from rigby_poc.kinematics import rig_kinematics
from rigby_poc.models import CompileRequest, Hand, Intent, PlanRequest, PrimitiveKind, default_scene
from rigby_poc.planner import plan_motion

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _left_hook():
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text="throw a left hook", scene=scene, provider="offline")
    ).program
    return scene, program


def test_left_hook_expands_to_dynamic_strike_primitives() -> None:
    _, program = _left_hook()
    assert program.intent is Intent.STRIKE
    assert program.hand is Hand.LEFT
    assert program.strike_type and program.strike_type.value == "hook"
    assert [item.kind for item in program.primitives] == [
        PrimitiveKind.GUARD,
        PrimitiveKind.LOAD,
        PrimitiveKind.STRIKE,
        PrimitiveKind.FOLLOW_THROUGH,
        PrimitiveKind.RECOVER,
    ]


def test_supported_strike_family_compiles_for_both_hands() -> None:
    scene = default_scene()
    for name in ("hook", "jab", "cross", "uppercut"):
        for hand in ("left", "right"):
            prompt = f"throw a {hand} {name}"
            program = plan_motion(
                PlanRequest(text=prompt, scene=scene, provider="offline")
            ).program
            clip = compile_motion(
                CompileRequest(scene=scene, program=program, persist=False)
            )
            assert program.intent is Intent.STRIKE
            assert program.strike_type and program.strike_type.value == name
            assert clip.success, (prompt, clip.failure)
            assert clip.metrics["structural_valid"] is True
            assert clip.metrics["active_hand_visibility_fraction"] == 1.0


def test_left_hook_compiles_as_visible_curved_bent_elbow_motion() -> None:
    scene, program = _left_hook()
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert clip.success, clip.failure
    assert clip.metrics["structural_valid"] is True
    assert clip.metrics["active_hand_visibility_fraction"] == 1.0
    assert clip.metrics["self_collision_frames"] == 0
    # The `strike_*_excursion` metrics are computed from `arm_landmarks`,
    # which reconstructs the arm from a fixed rest shoulder and cannot see the
    # trunk, so since the trunk began carrying part of the swing they describe
    # the arm's motion relative to the chest, not through the world. The
    # world-path claim in this test's name is asserted below via full FK; the
    # trunk-frame arc is still pinned as non-degenerate.
    assert clip.metrics["strike_wrist_path_length_m"] > 0.20
    assert clip.metrics["strike_lateral_excursion_m"] > 0.08
    assert 70.0 <= clip.metrics["impact_elbow_angle_deg"] <= 120.0
    assert min(clip.metrics["normalized_finger_curls"].values()) > 0.65

    strike_range = next(
        (float(item["start_s"]), float(item["end_s"]))
        for item in clip.metrics["phase_ranges_s"]
        if item["kind"] == "strike"
    )
    kinematics = rig_kinematics()
    node_of = {name: index for index, name in kinematics.canonical_by_node.items()}
    wrist_index = node_of["leftHand"]
    chest_index = node_of["chest"]
    wrists = []
    chest_yaws = []
    for frame in clip.frames:
        matrices = kinematics.world_matrices(frame.bones)
        forward = matrices[chest_index][:3, :3] @ np.array([0.0, 0.0, 1.0])
        chest_yaws.append(math.degrees(math.atan2(forward[0], forward[2])))
        if strike_range[0] - 1e-8 <= frame.time_s <= strike_range[1] + 1e-8:
            wrists.append(matrices[wrist_index][:3, 3].copy())
    world_path_m = float(
        sum(np.linalg.norm(second - first) for first, second in zip(wrists, wrists[1:]))
    )
    world_lateral_m = float(max(w[0] for w in wrists) - min(w[0] for w in wrists))
    assert world_path_m > 0.25
    assert world_lateral_m > 0.20
    # The trunk throws the hook with the arm: the chest visibly rotates
    # through the swing. This is the vocabulary this change adds; losing it
    # regresses the hook back to an arm-only punch.
    assert max(chest_yaws) - min(chest_yaws) > 20.0

    descriptor = motion_perceptual_descriptor(clip, Hand.LEFT)
    assert descriptor["recovery_endpoint_wrist_error_m"] < 1e-6
    assert descriptor["recovery_endpoint_hand_error_rad"] < 1e-6
    assert len([item for item in descriptor["samples"] if item["phase"] == "strike"]) == 3


def test_five_hook_candidates_are_valid_and_perceptually_distinct() -> None:
    scene, program = _left_hook()
    descriptors = []
    recipes = candidate_recipes(program.primitives[0].parameters, program.intent)
    assert len(recipes) == 5
    for index, recipe in enumerate(recipes, start=1):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        clip = compile_motion(CompileRequest(scene=scene, program=candidate, persist=False))
        assert clip.success, (recipe.name, clip.failure)
        assert clip.metrics["structural_valid"] is True
        descriptors.append(motion_perceptual_descriptor(clip, candidate.hand))
    for index, first in enumerate(descriptors):
        for second in descriptors[index + 1 :]:
            assert compare_perceptual_descriptors(first, second)["perceptually_distinct"]


@pytest.mark.parametrize(
    "prompt",
    (
        "throw a right jab",
        "throw a left cross",
        "throw a right uppercut",
    ),
)
def test_generalized_adaptive_pool_supplies_five_way_strike_batch(prompt: str) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    candidates = []
    baseline_index = single_sample_baseline_index(prompt)
    baseline_result_id = f"candidate-{baseline_index}"
    audit = {}
    selected = []
    for index, recipe in enumerate(
        candidate_recipe_pool(program.primitives[0].parameters, program.intent),
        start=1,
    ):
        candidate_program = apply_recipe(program, recipe, seed_offset=index)
        clip = compile_motion(
            CompileRequest(scene=scene, program=candidate_program, persist=False)
        )
        candidates.append(
            {
                "candidate_index": index,
                "recipe": {"name": recipe.name},
                "result_id": f"candidate-{index}",
                "compile_success": clip.success,
                "structural_valid": clip.metrics.get("structural_valid"),
                "motion_sha256": _motion_hash(clip),
                "perceptual_descriptor": motion_perceptual_descriptor(
                    clip, candidate_program.hand
                ),
            }
        )
        selected, audit = select_diverse_candidate_batch(
            candidates,
            baseline_result_id=baseline_result_id,
        )
        if audit.get("all_pairs_above_threshold"):
            break

    assert len(selected) == 5
    assert audit["status"] == "complete"
    assert audit["all_pairs_above_threshold"] is True
    assert len({item["motion_sha256"] for item in selected}) == 5
    assert all(item["compile_success"] and item["structural_valid"] for item in selected)


def test_strike_evidence_samples_guard_arc_impact_follow_through_and_recovery() -> None:
    scene, program = _left_hook()
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    points = phase_sampling_points({"clip": clip.model_dump(mode="json")})
    labels = {str(item["label"]) for item in points}
    assert {
        "guard_ready",
        "load_ready",
        "strike_early",
        "strike_midpoint",
        "impact_pose",
        "follow_through_late",
        "recover_end",
    } <= labels
