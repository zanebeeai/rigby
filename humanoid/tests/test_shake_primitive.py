from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from evals.capture import phase_sampling_points
from evals.heldout import heldout_complex_hangten_cases
from rigby_poc.compiler import compile_motion
from rigby_poc.models import (
    CompileRequest,
    GestureTiming,
    Hand,
    PlanRequest,
    PrimitiveKind,
    default_scene,
)
from rigby_poc.planner import OfflinePlanner

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


PROMPT = (
    'Throw up a "hang-ten" sign with your right hand, there should be a swift motion up to '
    'the main position wherein the middle three fingers are as contracted as possible, the '
    'wrist should then shake rapidly back and forth a few times, before returning to default'
)


def _planned_and_compiled():
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=PROMPT, scene=scene, provider="offline")
    ).program
    clip = compile_motion(CompileRequest(program=program, scene=scene, persist=False))
    return program, clip


def test_complex_hang_ten_plans_typed_shake_sequence() -> None:
    program, _ = _planned_and_compiled()
    assert [item.kind for item in program.primitives] == [
        PrimitiveKind.PRESENT,
        PrimitiveKind.HOLD,
        PrimitiveKind.SHAKE,
        PrimitiveKind.RECOVER,
    ]
    present, hold, shake, _ = program.primitives
    assert present.parameters.duration_s == 0.32
    for primitive in (present, hold, shake):
        assert primitive.parameters.index_curl == 1.0
        assert primitive.parameters.middle_curl == 1.0
        assert primitive.parameters.ring_curl == 1.0
    assert shake.parameters.wrist_shake_amplitude == 0.90
    assert shake.parameters.wrist_shake_cycles == 3.0


def test_shake_reverses_repeatedly_and_returns_to_default_within_gates() -> None:
    _, clip = _planned_and_compiled()
    assert clip.success, clip.failure
    phases = {item["kind"]: item for item in clip.metrics["phase_ranges_s"]}
    shake_range = phases["shake"]
    baseline_frame = min(
        clip.frames,
        key=lambda frame: abs(frame.time_s - float(shake_range["start_s"])),
    )
    baseline_rotation = Rotation.from_quat(
        baseline_frame.bones["rightLowerArm"].rotation.as_list()
    )
    baseline_hand_rotation = Rotation.from_quat(
        baseline_frame.bones["rightHand"].rotation.as_list()
    )
    deviations: list[float] = []
    hand_joint_deviations: list[float] = []
    for frame in clip.frames:
        if float(shake_range["start_s"]) <= frame.time_s <= float(shake_range["end_s"]):
            current = Rotation.from_quat(frame.bones["rightLowerArm"].rotation.as_list())
            deviations.append(float((baseline_rotation.inv() * current).as_rotvec()[1]))
            hand_current = Rotation.from_quat(frame.bones["rightHand"].rotation.as_list())
            hand_joint_deviations.append(
                float((baseline_hand_rotation.inv() * hand_current).magnitude())
            )
    signs = [int(np.sign(value)) for value in deviations if abs(value) > 0.015]
    reversals = sum(first != second for first, second in zip(signs, signs[1:]))
    assert reversals >= 5
    assert max(abs(value) for value in deviations) > 0.10
    # Regression for the first pilot: the oscillation must not be authored on
    # the hand joint, where it reads as wrist flexion/deviation.
    assert max(hand_joint_deviations) < 1e-6

    first, last = clip.frames[0], clip.frames[-1]
    for bone in first.bones:
        assert abs(float(np.dot(first.bones[bone].rotation.as_list(), last.bones[bone].rotation.as_list()))) > 0.999999
    assert clip.metrics["normalized_finger_curls"]["index"] == 1.0
    assert clip.metrics["normalized_finger_curls"]["middle"] == 1.0
    assert clip.metrics["normalized_finger_curls"]["ring"] == 1.0
    assert clip.metrics["wrist_swing_twist_limit_violations"] == 0
    assert clip.metrics["max_wrist_swing_rad"] < 0.01
    assert clip.metrics["forearm_rotation_cycles"] == 3.0
    assert clip.metrics["forearm_rotation_amplitude_rad"] > 0.20
    assert clip.metrics["self_collision_frames"] == 0
    assert clip.metrics["active_hand_visibility_fraction"] == 1.0
    assert clip.metrics["structural_failures"] == []


def test_shake_is_sampled_across_full_judge_timeline() -> None:
    _, clip = _planned_and_compiled()
    points = phase_sampling_points({"clip": clip.model_dump(mode="json")})
    shake_points = [point for point in points if point["phase"] == "shake"]
    assert [point["label"] for point in shake_points] == [
        "shake_start",
        "shake_extreme_1",
        "shake_extreme_2",
        "shake_extreme_3",
        "shake_extreme_4",
        "shake_extreme_5",
        "shake_extreme_6",
        "shake_end",
    ]


def test_complex_heldout_prompts_preserve_hand_cycles_and_maximum_curl() -> None:
    scene = default_scene()
    for case in heldout_complex_hangten_cases():
        program = OfflinePlanner().plan(
            PlanRequest(text=case["prompt"], scene=scene, provider="offline")
        ).program
        assert program.hand == Hand(case["hand"])
        shake = next(item for item in program.primitives if item.kind == PrimitiveKind.SHAKE)
        assert shake.parameters.wrist_shake_cycles == case["expected_cycles"]
        assert shake.parameters.index_curl == 1.0
        assert shake.parameters.middle_curl == 1.0
        assert shake.parameters.ring_curl == 1.0
        clip = compile_motion(CompileRequest(program=program, scene=scene, persist=False))
        assert clip.success, clip.failure
        assert clip.metrics["forearm_rotation_cycles"] == case["expected_cycles"]
        assert clip.metrics["max_wrist_swing_rad"] < 0.01
        assert clip.metrics["max_wrist_twist_rad"] < 0.01
    slow_program = OfflinePlanner().plan(
        PlanRequest(
            text=heldout_complex_hangten_cases()[1]["prompt"],
            scene=scene,
            provider="offline",
        )
    ).program
    assert slow_program.motion_profile is not None
    assert slow_program.motion_profile.timing == GestureTiming.SLOW
