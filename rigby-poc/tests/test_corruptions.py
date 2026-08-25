from __future__ import annotations

from evals.corruptions import CorruptionSpec, corrupt_clip, corruption_specs
from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, PlanRequest, default_scene
from rigby_poc.planner import OfflinePlanner

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def _base():
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(text="Throw up a hang-ten sign.", scene=scene, provider="offline")
    ).program
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    assert clip.success
    return program, clip


def test_calibration_suite_has_wrist_finger_and_timing_corruptions() -> None:
    specs = corruption_specs()
    assert len(specs) == 28
    assert len({spec.id for spec in specs}) == 28
    assert {spec.kind for spec in specs} == {
        "wrist_rotation",
        "wrong_joint_shake",
        "fist_shape",
        "open_middle_fingers",
        "timing",
    }
    assert sum(spec.kind == "timing" for spec in specs) == 4
    assert sum(spec.kind == "wrong_joint_shake" for spec in specs) == 4


def test_wrist_corruption_is_rejected_by_the_structural_gate() -> None:
    program, clip = _base()
    corrupted = corrupt_clip(
        clip,
        program.hand,
        CorruptionSpec(
            id="test-wrist",
            kind="wrist_rotation",
            axis=(1.0, 0.0, 0.0),
            angle_rad=1.1,
        ),
    )
    assert corrupted.success is False
    assert corrupted.metrics["structural_valid"] is False
    assert corrupted.metrics["wrist_swing_twist_limit_violations"] > 0
    assert corrupted.metrics["deliberate_corruption"]["id"] == "test-wrist"


def test_snap_timing_corruption_is_rejected_by_kinematic_gates() -> None:
    program, clip = _base()
    corrupted = corrupt_clip(
        clip,
        program.hand,
        CorruptionSpec(
            id="test-snap",
            kind="timing",
            mode="snap_present",
        ),
    )
    assert corrupted.success is False
    assert corrupted.metrics["structural_valid"] is False
    assert any(
        "angular" in failure
        for failure in corrupted.metrics["structural_failures"]
    )
    assert corrupted.metrics["deliberate_corruption"]["mode"] == "snap_present"


def test_remove_hold_timing_corruption_changes_the_hold_pose() -> None:
    program, clip = _base()
    corrupted = corrupt_clip(
        clip,
        program.hand,
        CorruptionSpec(
            id="test-remove-hold",
            kind="timing",
            mode="remove_hold",
        ),
    )
    hold = next(
        item
        for item in clip.metrics["phase_ranges_s"]
        if item["kind"] == "hold"
    )
    held_frames = [
        frame
        for frame in corrupted.frames
        if float(hold["start_s"]) <= frame.time_s <= float(hold["end_s"])
    ]
    forearm_bone = f"{program.hand.value}LowerArm"
    assert held_frames[0].bones[forearm_bone] != held_frames[-1].bones[forearm_bone]
    assert corrupted.metrics["deliberate_corruption"]["mode"] == "remove_hold"


def test_wrong_joint_shake_recreates_visual_anatomy_failure_inside_broad_limits() -> None:
    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(
            text=(
                "Throw up a hang-ten with the middle three fingers fully curled. "
                "Shake it back and forth three times, then return to default."
            ),
            scene=scene,
            provider="offline",
        )
    ).program
    clip = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    corrupted = corrupt_clip(
        clip,
        program.hand,
        CorruptionSpec(
            id="test-wrong-joint",
            kind="wrong_joint_shake",
            axis=(1.0, 0.0, 0.0),
            angle_rad=1.0,
            mode="move_forearm_oscillation_to_hand_joint",
        ),
    )
    assert corrupted.metrics["structural_valid"] is True
    assert 0.10 < corrupted.metrics["max_wrist_swing_rad"] < 0.48
    assert corrupted.metrics["wrist_swing_twist_limit_violations"] == 0
    assert corrupted.metrics["forearm_rotation_cycles"] == 0.0
    assert corrupted.metrics["forearm_rotation_amplitude_rad"] < 1e-4
    assert corrupted.metrics["wrist_flexion_cycles"] == 3.0
    assert corrupted.metrics["wrist_flexion_amplitude_rad"] > 0.10
    assert corrupted.metrics["wrist_deviation_cycles"] == 0.0
    assert corrupted.metrics["deliberate_corruption"]["kind"] == "wrong_joint_shake"
