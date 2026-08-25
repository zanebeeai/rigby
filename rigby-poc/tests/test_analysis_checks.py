"""Unit tests for the moved checks, driven by hand-built synthetic clips.

Every clip here is constructed frame by frame with one deliberate violation, so
a failure names the defect rather than "some number moved". This is the thing
that was impossible while the checks lived inside an 8446-line compile path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rigby_poc.analysis import (
    evaluate_gesture_structure,
    gesture_structure_checks,
    safety_checks,
    safety_metrics,
    semantic_cycle_metrics,
    shake_joint_oscillation_metrics,
    validate,
)
from rigby_poc.analysis.geometry import line_segment_distance
from rigby_poc.analysis.rig import identity_pose
from rigby_poc.models import (
    AssertionSpec,
    BonePose,
    ClipFrame,
    EffectorTarget,
    Hand,
    HandShape,
    Intent,
    MotionPrimitive,
    MotionProgram,
    PrimitiveKind,
    PrimitiveParameters,
    Quat,
    TrajectoryKind,
    TrajectoryPlane,
    Vec3,
)

# Every analysis test compiles a clip or reads one (plan 09 §3.3 tiering).
pytestmark = pytest.mark.medium



FPS = 30.0


def _quat(rotation: Rotation) -> Quat:
    x, y, z, w = (float(value) for value in rotation.as_quat())
    return Quat(x=x, y=y, z=z, w=w)


def _frame(index: int, overrides: dict[str, Quat] | None = None, *, hips: Vec3 | None = None) -> ClipFrame:
    """A rest-pose frame with named bones overridden."""

    bones = {name: BonePose(rotation=value) for name, value in identity_pose().items()}
    for name, rotation in (overrides or {}).items():
        bones[name] = BonePose(rotation=rotation)
    if hips is not None:
        bones["hips"] = BonePose(rotation=bones["hips"].rotation, position=hips)
    return ClipFrame(time_s=index / FPS, bones=bones)


def _rest_clip(count: int) -> list[ClipFrame]:
    return [_frame(index) for index in range(count)]


# --------------------------------------------------------------------------
# safety_metrics
# --------------------------------------------------------------------------


def test_safety_metrics_on_a_still_clip_reports_nothing_wrong() -> None:
    metrics = safety_metrics(_rest_clip(6))

    assert metrics["nan_count"] == 0
    assert metrics["discontinuities"] == 0
    assert metrics["joint_limit_violations"] == 0
    assert metrics["root_drift_m"] == 0.0
    assert metrics["max_frame_rotation_delta_rad"] == pytest.approx(0.0)
    assert metrics["quaternion_norm_max_error"] == pytest.approx(0.0, abs=1e-12)


def test_a_single_frame_rotation_jump_counts_as_one_discontinuity() -> None:
    """The threshold is 0.35 rad of per-frame local delta on any bone."""

    jump = _quat(Rotation.from_rotvec([0.0, 0.5, 0.0]))
    frames = [_frame(0), _frame(1, {"leftHand": jump}), _frame(2, {"leftHand": jump})]

    metrics = safety_metrics(frames)

    assert metrics["discontinuities"] == 1
    assert metrics["max_frame_rotation_delta_rad"] == pytest.approx(0.5, abs=1e-9)


def test_a_rotation_step_just_under_the_threshold_is_not_a_discontinuity() -> None:
    nudge = _quat(Rotation.from_rotvec([0.0, 0.34, 0.0]))
    frames = [_frame(0), _frame(1, {"leftHand": nudge})]

    assert safety_metrics(frames)["discontinuities"] == 0


def test_root_translation_is_a_joint_limit_violation_unless_it_is_allowed() -> None:
    frames = [
        _frame(0, hips=Vec3(x=0.0, y=0.0, z=0.0)),
        _frame(1, hips=Vec3(x=0.0, y=0.0, z=0.4)),
    ]

    fixed_root = safety_metrics(frames)
    moving_root = safety_metrics(frames, allow_root_motion=True)

    assert fixed_root["root_drift_m"] == pytest.approx(0.4)
    assert fixed_root["joint_limit_violations"] == 1
    assert moving_root["root_drift_m"] == pytest.approx(0.4)
    assert moving_root["joint_limit_violations"] == 0
    assert "must remain fixed" in fixed_root["safety_derivation"]
    assert "explicitly enabled" in moving_root["safety_derivation"]


def test_an_elbow_past_its_profile_limit_is_counted_once_per_frame() -> None:
    """``leftLowerArm`` is bounded at 2.75 rad by the rig profile."""

    over = _quat(Rotation.from_rotvec([2.9, 0.0, 0.0]))
    frames = [_frame(index, {"leftLowerArm": over}) for index in range(3)]

    assert safety_metrics(frames)["joint_limit_violations"] == 3


def test_a_non_finite_rotation_is_counted_component_wise() -> None:
    frames = _rest_clip(2)
    frames[1].bones["leftHand"] = BonePose(
        rotation=Quat(x=float("nan"), y=0.0, z=0.0, w=1.0)
    )

    metrics = safety_metrics(frames)

    assert metrics["nan_count"] == 1


def test_safety_checks_report_each_violation_separately() -> None:
    over = _quat(Rotation.from_rotvec([2.9, 0.0, 0.0]))
    frames = [_frame(index, {"leftLowerArm": over}) for index in range(3)]

    checks = {check.id: check for check in safety_checks(safety_metrics(frames))}

    assert checks["contract.clip.joint_limit_violations"].status == "fail"
    assert checks["contract.clip.joint_limit_violations"].measured == 3.0
    assert 0.0 < checks["contract.clip.joint_limit_violations"].severity <= 1.0
    assert checks["contract.clip.non_finite_transforms"].status == "pass"
    assert checks["contract.clip.non_finite_transforms"].severity == 0.0


def test_safety_metrics_on_an_empty_clip_returns_zeros_not_an_error() -> None:
    metrics = safety_metrics([])

    assert metrics["safety_derivation"] == "no frames"
    assert metrics["nan_count"] == 0


# --------------------------------------------------------------------------
# evaluate_gesture_structure
# --------------------------------------------------------------------------


def test_a_forearm_twisted_past_the_calibrated_limit_fails_that_check() -> None:
    """The limit is 1.35 rad of twist about the forearm's own Y axis."""

    twist = _quat(Rotation.from_rotvec([0.0, 1.6, 0.0]))
    frames = [_frame(index, {"rightLowerArm": twist}) for index in range(6)]

    structure = evaluate_gesture_structure(frames, Hand.RIGHT, [])
    checks = {check.id: check for check in gesture_structure_checks(structure)}

    assert structure["max_forearm_twist_rad"] == pytest.approx(1.6, abs=1e-9)
    assert not structure["structural_valid"]
    assert any(
        "forearm twist" in failure for failure in structure["structural_failures"]
    )
    assert checks["anatomy.forearm.twist"].status == "fail"
    assert checks["anatomy.forearm.twist"].measured == pytest.approx(1.6, abs=1e-9)
    assert checks["anatomy.forearm.twist"].threshold == 1.35
    assert checks["anatomy.forearm.twist"].severity == pytest.approx(
        (1.6 - 1.35) / 1.35, abs=1e-9
    )


def test_a_forearm_inside_the_limit_passes_with_zero_severity() -> None:
    twist = _quat(Rotation.from_rotvec([0.0, 1.0, 0.0]))
    frames = [_frame(index, {"rightLowerArm": twist}) for index in range(6)]

    structure = evaluate_gesture_structure(frames, Hand.RIGHT, [])
    checks = {check.id: check for check in gesture_structure_checks(structure)}

    assert checks["anatomy.forearm.twist"].status == "pass"
    assert checks["anatomy.forearm.twist"].severity == 0.0


def test_a_wrist_past_its_swing_limit_is_counted_on_every_offending_frame() -> None:
    """``wrist_swing_rad`` is 0.48; swing is measured about the hand's Y axis."""

    swing = _quat(Rotation.from_rotvec([0.9, 0.0, 0.0]))
    frames = [_frame(index, {"rightHand": swing}) for index in range(4)]

    structure = evaluate_gesture_structure(frames, Hand.RIGHT, [])
    checks = {check.id: check for check in gesture_structure_checks(structure)}

    assert structure["max_wrist_swing_rad"] == pytest.approx(0.9, abs=1e-9)
    assert structure["wrist_swing_twist_limit_violations"] == 4
    assert checks["anatomy.wrist.swing_twist_limit"].status == "fail"
    assert checks["anatomy.wrist.swing_twist_limit"].measured == 4.0


def test_hand_visibility_is_only_sampled_inside_the_presentation_window() -> None:
    frames = _rest_clip(10)

    windowed = evaluate_gesture_structure(
        frames, Hand.RIGHT, [(frames[2].time_s, frames[5].time_s)]
    )
    unwindowed = evaluate_gesture_structure(frames, Hand.RIGHT, [])

    assert windowed["active_hand_visibility_samples"] == 4
    assert unwindowed["active_hand_visibility_samples"] == 0
    assert unwindowed["active_hand_visibility_fraction"] == 0.0


# --------------------------------------------------------------------------
# shake_joint_oscillation_metrics
# --------------------------------------------------------------------------


def test_a_forearm_oscillation_is_measured_from_frames_not_parameters() -> None:
    """Two full cycles of forearm twist at 0.4 rad, built by hand."""

    cycles = 2.0
    count = 61
    frames = []
    for index in range(count):
        phase = 2.0 * math.pi * cycles * index / (count - 1)
        angle = 0.4 * math.sin(phase)
        frames.append(
            _frame(index, {"rightLowerArm": _quat(Rotation.from_rotvec([0.0, angle, 0.0]))})
        )
    window = [(frames[0].time_s, frames[-1].time_s)]

    metrics = shake_joint_oscillation_metrics(frames, Hand.RIGHT, window)

    assert metrics["forearm_rotation_amplitude_rad"] == pytest.approx(0.4, abs=0.01)
    assert metrics["forearm_rotation_cycles"] == pytest.approx(cycles, abs=0.5)
    assert metrics["wrist_flexion_amplitude_rad"] == pytest.approx(0.0, abs=1e-9)


def test_an_oscillation_moved_to_the_wrist_is_attributed_to_the_wrist() -> None:
    """The adversarial case the metric exists for: same motion, wrong joint."""

    count = 61
    frames = []
    for index in range(count):
        angle = 0.4 * math.sin(2.0 * math.pi * 2.0 * index / (count - 1))
        frames.append(
            _frame(index, {"rightHand": _quat(Rotation.from_rotvec([angle, 0.0, 0.0]))})
        )
    window = [(frames[0].time_s, frames[-1].time_s)]

    metrics = shake_joint_oscillation_metrics(frames, Hand.RIGHT, window)

    assert metrics["forearm_rotation_amplitude_rad"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["forearm_rotation_cycles"] == 0.0
    assert metrics["wrist_flexion_amplitude_rad"] == pytest.approx(0.4, abs=0.01)


def test_too_few_frames_in_the_window_reports_zeros() -> None:
    frames = _rest_clip(10)

    metrics = shake_joint_oscillation_metrics(frames, Hand.RIGHT, [(0.0, 0.02)])

    assert set(metrics.values()) == {0.0}


# --------------------------------------------------------------------------
# semantic_cycle_metrics
# --------------------------------------------------------------------------


def _wave_program(required_reversals: float) -> MotionProgram:
    return MotionProgram(
        source_text="wave your right hand",
        intent=Intent.COMPOSITE,
        hand=Hand.RIGHT,
        hands=[Hand.RIGHT],
        primitives=[
            MotionPrimitive(
                kind=PrimitiveKind.CYCLE,
                label="hello_wave_cycle",
                trajectory=TrajectoryKind.OSCILLATE,
                trajectory_plane=TrajectoryPlane.FRONTAL,
                effectors=[
                    EffectorTarget(
                        hand=Hand.RIGHT,
                        target_x=-0.72,
                        target_y=0.68,
                        target_z=0.52,
                        hand_shape=HandShape.OPEN,
                    )
                ],
                parameters=PrimitiveParameters(
                    duration_s=1.6,
                    trajectory_cycles=3.0,
                    trajectory_amplitude_m=0.10,
                ),
            )
        ],
        assertions=[
            AssertionSpec(
                name="hello_wave_trajectory_reversals", threshold=required_reversals
            )
        ],
    )


def _lateral_wave_frames(cycles: float, amplitude_rad: float, count: int) -> list[ClipFrame]:
    """Swing the right upper arm about Z so the wrist sweeps laterally."""

    frames = []
    for index in range(count):
        phase = 2.0 * math.pi * cycles * index / (count - 1)
        angle = amplitude_rad * math.sin(phase)
        frames.append(
            _frame(
                index,
                {"rightUpperArm": _quat(Rotation.from_rotvec([0.0, 0.0, angle]))},
            )
        )
    return frames


def test_a_lateral_wave_reports_its_reversals_and_excursion() -> None:
    program = _wave_program(required_reversals=4.0)
    frames = _lateral_wave_frames(cycles=3.0, amplitude_rad=0.8, count=91)
    phase_ranges = [
        {
            "kind": PrimitiveKind.CYCLE.value,
            "label": "hello_wave_cycle",
            "start_s": frames[0].time_s,
            "end_s": frames[-1].time_s,
        }
    ]

    metrics = semantic_cycle_metrics(frames, phase_ranges, program)

    checks = {check.id: check for check in validate(metrics, program, frames, fps=30.0)}

    assert metrics["semantic_cycle_action"] == "hello_wave"
    assert metrics["semantic_cycle_requested_cycles"] == 3.0
    # Three authored cycles must yield at least the four reversals the
    # assertion demands. The exact count is not pinned: the metric reads the
    # wrist path in the shoulder frame, so it also sees the small secondary
    # excursions a real shoulder rotation produces.
    assert metrics["semantic_cycle_min_reversal_count"] >= 5
    # The threshold is max(0.045, requested amplitude * 0.75) = 0.075 m here.
    assert metrics["semantic_cycle_min_excursion_m"] > 0.075
    assert checks["signal.semantic_cycle.reversals"].status == "pass"
    assert checks["signal.semantic_cycle.excursion"].status == "pass"


def test_a_wave_that_never_reverses_fails_the_reversal_check() -> None:
    program = _wave_program(required_reversals=4.0)
    frames = _rest_clip(91)
    phase_ranges = [
        {
            "kind": PrimitiveKind.CYCLE.value,
            "label": "hello_wave_cycle",
            "start_s": frames[0].time_s,
            "end_s": frames[-1].time_s,
        }
    ]

    metrics = semantic_cycle_metrics(frames, phase_ranges, program)
    checks = {check.id: check for check in validate(metrics, program, frames, fps=30.0)}

    assert metrics["semantic_cycle_min_reversal_count"] == 0
    assert checks["signal.semantic_cycle.reversals"].status == "fail"
    assert checks["signal.semantic_cycle.reversals"].severity == 1.0
    assert checks["signal.semantic_cycle.excursion"].status == "fail"


def test_a_program_with_no_reversal_assertion_produces_no_cycle_metrics() -> None:
    program = _wave_program(required_reversals=4.0).model_copy(
        update={"assertions": []}
    )
    frames = _lateral_wave_frames(cycles=3.0, amplitude_rad=0.45, count=31)

    assert semantic_cycle_metrics(frames, [], program) == {}


# --------------------------------------------------------------------------
# line_segment_distance
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        # Parallel, offset along Y.
        (((0, 0, 0), (1, 0, 0)), ((0, 0.5, 0), (1, 0.5, 0)), 0.5),
        # Crossing in projection, separated in Z.
        (((-1, 0, 0), (1, 0, 0)), ((0, -1, 0.25), (0, 1, 0.25)), 0.25),
        # Touching.
        (((0, 0, 0), (1, 0, 0)), ((0.5, 0, 0), (0.5, 1, 0)), 0.0),
    ],
)
def test_segment_distance_matches_hand_computed_geometry(
    first: tuple, second: tuple, expected: float
) -> None:
    result = line_segment_distance(
        np.asarray(first[0], dtype=float),
        np.asarray(first[1], dtype=float),
        np.asarray(second[0], dtype=float),
        np.asarray(second[1], dtype=float),
    )

    assert result == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("second", "true_distance", "reported"),
    [
        # Collinear, disjoint: the true gap is 2.0.
        ((( 3.0, 0.0, 0.0), (4.0, 0.0, 0.0)), 2.0, 3.0),
        # Parallel and offset, second segment starting ahead of the first:
        # the true perpendicular distance is 0.3.
        (((0.2, 0.3, 0.0), (1.2, 0.3, 0.0)), 0.3, math.hypot(0.2, 0.3)),
    ],
)
def test_exactly_parallel_segments_are_over_estimated(
    second: tuple, true_distance: float, reported: float
) -> None:
    """A known limitation of the moved helper, pinned so it cannot drift silently.

    When ``aa * bb - ab**2`` collapses — which is exactly when the two segments
    are parallel — the solver stops searching along the first segment and
    clamps it to its start point. It then reports the distance from that start
    point rather than the true closest approach.

    ``parallel_forearm_metrics`` is the only caller. Its forearms are *near*
    parallel, not exactly parallel: the check itself tolerates up to 20 degrees
    of axis error, and at segment lengths around 0.25 m the determinant only
    falls below the 1e-10 epsilon within about 1e-4 of true parallelism. So the
    branch is effectively unreachable in production. Recorded rather than fixed
    because 02a is a move, not a redesign; see plan 02 §1.2.
    """

    result = line_segment_distance(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray(second[0], dtype=float),
        np.asarray(second[1], dtype=float),
    )

    assert result == pytest.approx(reported, abs=1e-9)
    assert result > true_distance


def test_segment_distance_is_symmetric() -> None:
    a, b = np.asarray([0.0, 0.0, 0.0]), np.asarray([1.0, 0.3, -0.2])
    c, d = np.asarray([0.4, 0.9, 0.6]), np.asarray([-0.2, 0.5, 1.1])

    assert line_segment_distance(a, b, c, d) == pytest.approx(
        line_segment_distance(c, d, a, b), abs=1e-9
    )
