from __future__ import annotations

import pytest

from evals.capture import phase_sampling_points
from evals.flywheel import (
    _motion_hash,
    apply_recipe,
    candidate_recipes,
    motion_perceptual_descriptor,
    select_diverse_candidate_batch,
)
from rigby_poc.compiler import compile_motion
from rigby_poc.models import (
    CompileRequest,
    Intent,
    PlanRequest,
    default_scene,
)
from rigby_poc.planner import plan_motion

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


@pytest.mark.parametrize(
    ("prompt", "expected"),
    (
        ("turn left then throw a right jab", (Intent.FULL_BODY, Intent.STRIKE)),
        (
            "wave with your right hand then kick forward with your left leg",
            (Intent.COMPOSITE, Intent.FULL_BODY),
        ),
        (
            "walk forward two steps then point forward with your left hand",
            (Intent.FULL_BODY, Intent.GESTURE),
        ),
        (
            "catch the block then turn around",
            (Intent.OBJECT_INTERACTION, Intent.FULL_BODY),
        ),
    ),
)
def test_explicit_temporal_connectors_preserve_every_action(
    prompt: str,
    expected: tuple[Intent, Intent],
) -> None:
    program = plan_motion(
        PlanRequest(text=prompt, scene=default_scene(), provider="offline")
    ).program

    assert program.intent is Intent.SEQUENCE
    assert tuple(step.intent for step in program.steps) == expected
    assert not program.primitives


def test_sequence_compiler_preserves_order_root_turn_and_continuity() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="turn left then throw a right jab",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert clip.success, clip.failure
    assert clip.metrics["structural_valid"] is True
    assert clip.metrics["discontinuities"] == 0
    assert clip.metrics["sequence_intents"] == ["full_body", "strike"]
    assert clip.metrics["final_root_yaw_deg"] == pytest.approx(90.0)
    ranges = clip.metrics["sequence_step_ranges_s"]
    assert ranges[0]["end_s"] < ranges[1]["start_s"]
    impact = next(
        item
        for item in clip.metrics["phase_ranges_s"]
        if item["kind"] == "strike"
    )
    assert impact["step_index"] == 2
    assert impact["start_s"] >= ranges[1]["start_s"]


def test_sequence_capture_samples_each_child_lifecycle() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="walk forward two steps then point forward with your left hand",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )
    points = phase_sampling_points(
        {
            "program": program.model_dump(mode="json"),
            "clip": clip.model_dump(mode="json"),
        }
    )
    labels = {str(item["label"]) for item in points}

    assert "body_midpoint" in labels
    assert "presented_pose" in labels
    assert "recover_end_2" in labels


def test_unrecognized_sequence_clause_is_rejected_instead_of_dropped() -> None:
    program = plan_motion(
        PlanRequest(
            text="turn left then teleport through the wall",
            scene=default_scene(),
            provider="offline",
        )
    ).program

    assert program.intent is Intent.UNSUPPORTED
    assert "sequence step 2" in (program.unsupported_reason or "")


def test_grab_then_throw_uses_single_stateful_throw_lifecycle() -> None:
    program = plan_motion(
        PlanRequest(
            text="grab the block then throw it forward",
            scene=default_scene(),
            provider="offline",
        )
    ).program

    assert program.intent is Intent.OBJECT_INTERACTION
    assert program.object_action and program.object_action.value == "throw"
    assert program.source_text == "grab the block then throw it forward"


def test_sequence_candidate_recipes_change_timing_without_losing_steps() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="turn left then throw a right jab",
            scene=scene,
            provider="offline",
        )
    ).program
    hashes: set[str] = set()
    candidates: list[dict[str, object]] = []
    for index, recipe in enumerate(
        candidate_recipes(program.steps[0].primitives[0].parameters, Intent.SEQUENCE),
        start=1,
    ):
        candidate = apply_recipe(program, recipe, seed_offset=index)
        assert [step.intent for step in candidate.steps] == [
            Intent.FULL_BODY,
            Intent.STRIKE,
        ]
        clip = compile_motion(
            CompileRequest(scene=scene, program=candidate, persist=False)
        )
        assert clip.success, (recipe.name, clip.failure)
        hashes.add(_motion_hash(clip))
        candidates.append(
            {
                "candidate_index": index,
                "recipe": {"name": recipe.name},
                "result_id": f"candidate-{index}",
                "compile_success": clip.success,
                "structural_valid": clip.metrics["structural_valid"],
                "motion_sha256": _motion_hash(clip),
                "perceptual_descriptor": motion_perceptual_descriptor(
                    clip, candidate.hands or candidate.hand
                ),
            }
        )

    assert len(hashes) == 5
    selected, audit = select_diverse_candidate_batch(
        candidates,
        baseline_result_id="candidate-1",
    )
    assert len(selected) == 5
    assert audit["status"] == "complete"


@pytest.mark.parametrize(
    ("prompt", "action", "hands", "shape"),
    (
        (
            "crouch down and point forward with your left hand",
            "crouch",
            {"left"},
            "point",
        ),
        (
            "step to the right while throwing a left hook",
            "step",
            {"left"},
            "fist",
        ),
        (
            "turn left while raising both open hands high",
            "turn",
            {"left", "right"},
            "open",
        ),
    ),
)
def test_concurrent_body_and_arm_actions_are_not_dropped(
    prompt: str,
    action: str,
    hands: set[str],
    shape: str,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    body = next(item for item in program.primitives if item.kind.value == "body")

    assert program.intent is Intent.FULL_BODY
    assert body.body and body.body.action.value == action
    assert {target.hand.value for target in body.effectors} == hands
    assert {target.hand_shape.value for target in body.effectors} == {shape}
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )
    assert clip.success, clip.failure
    assert clip.metrics["structural_valid"] is True
    assert clip.metrics["discontinuities"] == 0


@pytest.mark.parametrize(
    ("prompt", "variant", "pitch_sign"),
    (
        ("lie down on your back", "supine", -1.0),
        ("lie face down on your stomach", "prone", 1.0),
        ("go prone", "prone", 1.0),
    ),
)
def test_horizontal_grounded_poses_transfer_support_and_recover(
    prompt: str,
    variant: str,
    pitch_sign: float,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    pose_primitive = next(
        item
        for item in program.primitives
        if item.body is not None and item.body.action.value == "pose"
    )

    assert program.intent is Intent.FULL_BODY
    assert pose_primitive.body is not None
    assert pose_primitive.body.pose.lock_feet is False
    assert pose_primitive.body.pose.pelvis_pitch_deg * pitch_sign >= 80.0
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )
    assert clip.success, clip.failure
    assert clip.metrics["horizontal_pose_variant"] == variant
    assert clip.metrics["horizontal_pose_contact_point_count"] >= 2
    assert clip.metrics["horizontal_pose_minimum_clearance_m"] >= -0.012
    assert clip.metrics["horizontal_body_axis_vertical_fraction"] <= 0.35
    assert clip.metrics["discontinuities"] == 0


def test_horizontal_pose_can_precede_another_grounded_action() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text="recline into a supine position then sit up",
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is Intent.SEQUENCE
    assert [step.intent for step in program.steps] == [
        Intent.FULL_BODY,
        Intent.FULL_BODY,
    ]
    assert clip.success, clip.failure
    assert clip.metrics["horizontal_pose_requested"] is True
    assert clip.metrics["horizontal_pose_variants"] == ["supine"]
    assert clip.metrics["discontinuities"] == 0
