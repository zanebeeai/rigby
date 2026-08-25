from __future__ import annotations

import json
import math
from copy import deepcopy

from scipy.spatial.transform import Rotation

from evals.criteria import supported_cases
from rigby_poc.compiler import RIG_PROFILE, compile_motion
from rigby_poc.models import BonePose, CompileRequest, PlanRequest, Quat, default_scene
from rigby_poc.planner import OfflinePlanner
from rigby_poc.quality import evaluate_gesture_structure, quality_reference, swing_twist_angles

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _gesture(prompt: str = "Throw up a hang-ten sign."):
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    return program, clip


def test_swing_twist_decomposition_separates_pure_rotations() -> None:
    pure_swing = Rotation.from_rotvec([0.32, 0.0, 0.0]).as_quat()
    swing, twist = swing_twist_angles(pure_swing, [0.0, 1.0, 0.0])
    assert math.isclose(swing, 0.32, abs_tol=1e-8)
    assert math.isclose(twist, 0.0, abs_tol=1e-8)

    pure_twist = Rotation.from_rotvec([0.0, -0.14, 0.0]).as_quat()
    swing, twist = swing_twist_angles(pure_twist, [0.0, 1.0, 0.0])
    assert math.isclose(swing, 0.0, abs_tol=1e-8)
    assert math.isclose(twist, -0.14, abs_tol=1e-8)


def test_quality_limits_are_part_of_the_calibrated_rig_contract() -> None:
    profile = json.loads(RIG_PROFILE.read_text(encoding="utf-8"))
    limits = quality_reference()["hard_limits"]
    assert profile["anatomical_limits"]["wrist_swing_rad"] == limits["wrist_swing_rad"]
    assert profile["anatomical_limits"]["wrist_twist_rad"] == limits["wrist_twist_rad"]
    assert profile["anatomical_limits"]["forearm_twist_rad"] == limits["forearm_twist_rad"]


def test_all_twenty_review_profiles_pass_structural_and_temporal_gates() -> None:
    scene = default_scene()
    planner = OfflinePlanner()
    for case in supported_cases()[:20]:
        program = planner.plan(
            PlanRequest(text=case["prompt"], scene=scene, provider="offline")
        ).program
        clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
        assert clip.success, f"{case['id']}: {clip.failure}"
        assert clip.metrics["structural_valid"] is True
        assert clip.metrics["wrist_swing_twist_limit_violations"] == 0
        assert clip.metrics["self_collision_frames"] == 0
        assert clip.metrics["active_hand_visibility_fraction"] == 1.0
        limits = clip.metrics["quality_limits"]
        assert clip.metrics["max_angular_velocity_rad_s"] <= limits["angular_velocity_rad_s"]
        assert clip.metrics["max_angular_acceleration_rad_s2"] <= limits["angular_acceleration_rad_s2"]
        assert clip.metrics["max_angular_jerk_rad_s3"] <= limits["angular_jerk_rad_s3"]


def test_structural_gate_rejects_deliberate_wrist_and_timing_corruptions() -> None:
    program, clip = _gesture()
    wrist_corruption = deepcopy(clip.frames)
    bad_quat = Quat.model_validate(
        dict(zip(("x", "y", "z", "w"), Rotation.from_rotvec([0.0, 0.5, 0.0]).as_quat()))
    )
    for frame in wrist_corruption:
        frame.bones[f"{program.hand.value}Hand"] = BonePose(rotation=bad_quat)
    wrist_result = evaluate_gesture_structure(
        wrist_corruption,
        program.hand,
        [(wrist_corruption[0].time_s, wrist_corruption[-1].time_s)],
    )
    assert wrist_result["structural_valid"] is False
    assert wrist_result["wrist_swing_twist_limit_violations"] == len(wrist_corruption)

    timing_corruption = deepcopy(clip.frames)
    for index, frame in enumerate(timing_corruption):
        frame.time_s = index / 300.0
    timing_result = evaluate_gesture_structure(
        timing_corruption,
        program.hand,
        [(timing_corruption[0].time_s, timing_corruption[-1].time_s)],
    )
    assert timing_result["structural_valid"] is False
    assert any("angular" in failure for failure in timing_result["structural_failures"])
