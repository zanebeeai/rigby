"""Continuous polynomial checks, independent of the tangent limiter's formula."""

import json
from pathlib import Path

import mujoco
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from rigby_core.contracts import (
    ArtifactRefV1, DofSpecV1, InterpolationKind, MotionKeyframeV2,
    MotionPhaseV2, MotionProgramV2, MotionTrackV2, PhaseKind,
    RigAssetManifestV1, TrackOwnership, Vec3,
)
from rigby_core.motion import (
    MotionCompilationError, MotionFailureReason, TimingProfile, compile_motion_program,
)
from rigby_core.motion.ownership import JointTrackSeries


FIXTURE = Path(__file__).parent / "fixtures/g05_offending_keyframes.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def track(times, values, mode=InterpolationKind.BOUNDED_QUINTIC, **kwargs):
    return MotionTrackV2(
        track_id="motion", target="axis", owner="test", interpolation=mode,
        keyframes=tuple(MotionKeyframeV2(time_s=float(t), joint_values={"axis": float(q)})
                        for t, q in zip(times, values)), **kwargs,
    )


def polynomial(segment):
    """Recover the normalized polynomial by six boundary equations.

    This neither reads internal coefficients nor restates Bernstein limiting.
    Root checks below inspect the full interval, including sub-sample peaks.
    """
    h = segment.duration_s
    start, end = segment.sample(0), segment.sample(h)
    matrix = np.array([
        [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
        [1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 2, 6, 12, 20],
    ], dtype=float)
    rhs = [start.position, start.velocity*h, start.acceleration*h*h,
           end.position, end.velocity*h, end.acceleration*h*h]
    return Polynomial(np.linalg.solve(matrix, rhs))


def extrema(poly):
    roots = poly.deriv().roots()
    points = [0., 1.] + [float(r.real) for r in roots
                        if abs(r.imag) < 1e-7 and 0 < r.real < 1]
    return np.array(points), poly(points)


def check_continuous_bounds(series):
    for i, segment in enumerate(series._segments):
        poly = polynomial(segment)
        _, positions = extrema(poly)
        q0, q1 = series.values[i:i+2]
        assert positions.min() >= min(q0, q1) - 2e-10
        assert positions.max() <= max(q0, q1) + 2e-10
        _, derivatives = extrema(poly.deriv())
        if q0 == q1:
            np.testing.assert_allclose(positions, q0, atol=2e-10, rtol=0)
            np.testing.assert_allclose(derivatives, 0, atol=2e-10, rtol=0)
        else:
            assert (np.sign(q1-q0)*derivatives).min() >= -2e-10
        # Check the independent polynomial against the runtime evaluator too.
        for u in (.07, .39, .81):
            actual = segment.sample(u*segment.duration_s)
            np.testing.assert_allclose(
                [actual.position, actual.velocity*segment.duration_s,
                 actual.acceleration*segment.duration_s**2],
                [poly(u), poly.deriv()(u), poly.deriv(2)(u)], atol=2e-10, rtol=1e-9,
            )
    for i, (left, right) in enumerate(zip(series._segments, series._segments[1:])):
        a, b = left.sample(left.duration_s), right.sample(0)
        np.testing.assert_allclose([a.position, a.velocity, a.acceleration],
                                   [b.position, b.velocity, b.acceleration],
                                   atol=1e-8, rtol=1e-9)
        assert b.position == series.values[i+1]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_frozen_dual_arm_overshoot_fixed_without_retiming_or_clipping(case):
    keys = case["neighboring_keyframes"]
    times = [k["authored_time_s"] for k in keys]
    values = [k["position_rad"] for k in keys]
    legacy = JointTrackSeries(track(times, values, InterpolationKind.QUINTIC), "axis", times[-1])
    xs, ys = extrema(polynomial(legacy._segments[1]))
    peak = np.argmax(ys)
    assert ys[peak] == pytest.approx(case["expected_maximum_rad"], abs=1e-12)
    assert times[1] + xs[peak]*(times[2]-times[1]) == pytest.approx(
        case["expected_maximum_authored_time_s"], abs=1e-10)
    assert ys[peak] > case["joint_limits_rad"][1]
    bounded = JointTrackSeries(track(times, values), "axis", times[-1])
    np.testing.assert_array_equal(bounded.times, legacy.times)
    np.testing.assert_array_equal(bounded.values, legacy.values)
    check_continuous_bounds(bounded)


@pytest.mark.parametrize("seed", range(20))
def test_continuous_extrema_with_uneven_keys_plateaus_and_reversals(seed):
    rng = np.random.default_rng(seed)
    for _ in range(10):
        times = np.r_[0., np.cumsum(10.**rng.uniform(-1, 1, 11))]
        values = rng.uniform(-2.85, 2.85, len(times))
        values[3] = values[2]  # exact hold
        values[8] = values[7] + 1e-12  # nearly flat interval
        series = JointTrackSeries(track(times, values), "axis", times[-1])
        check_continuous_bounds(series)


@pytest.mark.parametrize("times,values", [([0.], [.3]), ([0., 1.], [.3, .3]),
                                         ([0., 1.], [-.3, .3])])
def test_short_series_and_holds(times, values):
    series = JointTrackSeries(track(times, values), "axis", 1.)
    check_continuous_bounds(series)
    if len(set(values)) == 1:
        for t in np.linspace(0, 1, 11):
            s = series.sample(t)
            assert (s.position, s.velocity, s.acceleration) == (values[0], 0, 0)


def test_linear_interior_keeps_nonzero_velocity_and_duration_scaling():
    base = JointTrackSeries(track([0, 1, 2, 3], [0, 1, 2, 3]), "axis", 3)
    assert base.sample(1.5).velocity == pytest.approx(1)
    assert base.sample(1.5).acceleration == pytest.approx(0)
    slower = JointTrackSeries(track([0, 7, 14, 21], [0, 1, 2, 3]), "axis", 21)
    for t in np.linspace(0, 3, 31):
        a, b = base.sample(t), slower.sample(7*t)
        np.testing.assert_allclose([a.position, a.velocity, a.acceleration],
                                   [b.position, b.velocity*7, b.acceleration*49], atol=1e-12)


def model_and_rig():
    xml = '''<mujoco><compiler angle="radian"/><worldbody><body>
      <joint name="axis" range="-2.85 2.85"/>
      <geom type="sphere" size=".1" mass="1"/><site name="tip" pos=".1 0 0"/>
    </body></worldbody><actuator><motor joint="axis"/></actuator></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    rig = RigAssetManifestV1(
        rig_id="test-rig", mjcf=ArtifactRefV1(sha256="a"*64, size_bytes=len(xml)),
        dofs=(DofSpecV1(name="axis", joint="axis", minimum=-2.85, maximum=2.85,
                        velocity_limit=100, effort_limit=40),),
        actuator_order=("axis",), rest_qpos=(0.,),
    )
    return model, rig


def program(tracks):
    duration = max(k.time_s for tr in tracks for k in tr.keyframes)
    return MotionProgramV2(
        program_id="test", source_text="move", duration_s=duration,
        rig_id="test-rig", scene_id="test-scene", seed=0,
        phases=(MotionPhaseV2(phase_id="move", kind=PhaseKind.ACTION,
                              start_s=0, end_s=duration),), tracks=tracks,
    )


def test_compiler_persistence_identity_and_unchanged_key_limit_refusal():
    model, rig = model_and_rig()
    keys = CASES[0]["neighboring_keyframes"]
    times = np.array([k["authored_time_s"] for k in keys])
    times -= times[0]
    values = [k["position_rad"] for k in keys]
    bounded = program((track(times, values),))
    restored = MotionProgramV2.model_validate_json(bounded.model_dump_json())
    assert restored == bounded
    candidate = compile_motion_program(restored, model, rig, timing_profile=TimingProfile.NEUTRAL)
    assert candidate.qpos.max() <= 2.85
    assert candidate.times_s[-1] == bounded.duration_s
    legacy = program((track(times, values, InterpolationKind.QUINTIC),))
    assert legacy.content_hash() != bounded.content_hash()
    with pytest.raises(MotionCompilationError) as failure:
        compile_motion_program(legacy, model, rig, timing_profile=TimingProfile.NEUTRAL)
    assert failure.value.reason is MotionFailureReason.JOINT_LIMIT_VIOLATION
    bad = program((track([0, 1], [0, 2.851]),))
    with pytest.raises(MotionCompilationError) as failure:
        compile_motion_program(bad, model, rig)
    assert failure.value.reason is MotionFailureReason.JOINT_LIMIT_VIOLATION


def test_task_space_bounded_quintic_is_typed_unsupported():
    tr = MotionTrackV2(track_id="task", target="tip", owner="test",
                       interpolation=InterpolationKind.BOUNDED_QUINTIC,
                       keyframes=(MotionKeyframeV2(time_s=0, position=Vec3(x=0, y=0, z=0)),
                                  MotionKeyframeV2(time_s=1, position=Vec3(x=.1, y=0, z=0))))
    with pytest.raises(MotionCompilationError) as failure:
        compile_motion_program(program((tr,)), *model_and_rig())
    assert failure.value.reason is MotionFailureReason.UNSUPPORTED_TRACK


def test_additive_composition_still_obeys_compiler_limit_gate():
    base = track([0, 1], [2, 2])
    addition = track([0, 1], [0, 1], ownership=TrackOwnership.ADDITIVE).model_copy(
        update={"owner": "other", "track_id": "offset"})
    with pytest.raises(MotionCompilationError) as failure:
        compile_motion_program(program((base, addition)), *model_and_rig())
    assert failure.value.reason is MotionFailureReason.JOINT_LIMIT_VIOLATION
