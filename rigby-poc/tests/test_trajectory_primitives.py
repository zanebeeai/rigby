from __future__ import annotations

import math

import numpy as np

from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, PlanRequest, default_scene
from rigby_poc.planner import plan_motion

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _angular_delta(first: list[float], second: list[float]) -> float:
    return 2.0 * math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))


def _frame_near(clip, time_s: float):
    return min(clip.frames, key=lambda frame: abs(frame.time_s - time_s))


def test_path_arc_and_wrist_flourish_change_mid_trajectory_not_hold_pose() -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(
            text=(
                "With your right hand, make a balanced playful shaka high and fully extended, "
                "held outward. Pitch the wrist up, yaw it outward, and roll it clockwise."
            ),
            scene=scene,
            provider="offline",
        )
    ).program
    authored = program.model_copy(deep=True)
    for primitive in authored.primitives[:2]:
        primitive.parameters = primitive.parameters.model_copy(
            update={"path_arc": 0.70, "wrist_flourish": 0.35}
        )

    baseline = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    curved = compile_motion(CompileRequest(scene=scene, program=authored, persist=False))
    assert baseline.success is True
    assert curved.success is True
    assert curved.metrics["structural_valid"] is True

    present = curved.metrics["phase_ranges_s"][0]
    midpoint_s = (float(present["start_s"]) + float(present["end_s"])) / 2.0
    end_s = float(present["end_s"])
    baseline_mid = _frame_near(baseline, midpoint_s)
    curved_mid = _frame_near(curved, midpoint_s)
    baseline_end = _frame_near(baseline, end_s)
    curved_end = _frame_near(curved, end_s)

    assert _angular_delta(
        baseline_mid.bones["rightUpperArm"].rotation.as_list(),
        curved_mid.bones["rightUpperArm"].rotation.as_list(),
    ) > 0.03
    assert _angular_delta(
        baseline_mid.bones["rightLowerArm"].rotation.as_list(),
        curved_mid.bones["rightLowerArm"].rotation.as_list(),
    ) > 0.03
    assert _angular_delta(
        baseline_end.bones["rightUpperArm"].rotation.as_list(),
        curved_end.bones["rightUpperArm"].rotation.as_list(),
    ) < 1e-6
    assert _angular_delta(
        baseline_end.bones["rightLowerArm"].rotation.as_list(),
        curved_end.bones["rightLowerArm"].rotation.as_list(),
    ) < 1e-6
