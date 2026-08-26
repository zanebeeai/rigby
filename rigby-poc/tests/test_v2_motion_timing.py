from __future__ import annotations

import numpy as np
import pytest

from rigby_v2.contracts import (
    CoordinateFrame,
    MotionPhaseV2,
    PhaseKind,
    Quaternion,
    QuaternionConvention,
    QuaternionMeaning,
    QuaternionOrder,
)
from rigby_v2.motion import (
    MonotoneTimeLaw,
    MotionCompilationError,
    MotionFailureReason,
    PhaseRetimer,
    QuinticSegment,
    TimingProfile,
    slerp,
    squad,
)
from rigby_v2.motion.timing import validate_phase_schedule


def _quaternion(
    values: tuple[float, float, float, float],
    *,
    order: QuaternionOrder = QuaternionOrder.WXYZ,
) -> Quaternion:
    return Quaternion(
        values=values,
        convention=QuaternionConvention(
            order=order,
            frame=CoordinateFrame.LOCAL,
            meaning=QuaternionMeaning.REST_DELTA,
        ),
    )


def test_phase_order_is_explicit_and_backward_transitions_are_typed() -> None:
    phases = (
        MotionPhaseV2(phase_id="action", kind=PhaseKind.ACTION, start_s=0, end_s=0.5),
        MotionPhaseV2(phase_id="setup", kind=PhaseKind.SETUP, start_s=0.5, end_s=1),
    )
    with pytest.raises(MotionCompilationError) as failure:
        validate_phase_schedule(phases, 1.0)
    assert failure.value.reason is MotionFailureReason.INVALID_PHASE_LAYOUT


def test_energetic_and_relaxed_laws_are_monotone_and_endpoint_flat() -> None:
    energetic = MonotoneTimeLaw(TimingProfile.ENERGETIC)
    relaxed = MonotoneTimeLaw(TimingProfile.RELAXED)
    grid = np.linspace(0.0, 1.0, 101)
    energetic_progress = np.asarray([energetic.sample(value).progress for value in grid])
    relaxed_progress = np.asarray([relaxed.sample(value).progress for value in grid])
    assert np.all(np.diff(energetic_progress) >= 0.0)
    assert np.all(np.diff(relaxed_progress) >= 0.0)
    assert energetic.sample(0.5).progress > 0.5
    assert relaxed.sample(0.5).progress < 0.5
    for law in (energetic, relaxed):
        assert law.sample(0).progress == pytest.approx(0.0)
        assert law.sample(1).progress == pytest.approx(1.0)
        assert law.sample(0).first_derivative == pytest.approx(0.0)
        assert law.sample(1).first_derivative == pytest.approx(0.0)


def test_phase_retimer_keeps_hard_and_contact_boundaries_fixed() -> None:
    phase = MotionPhaseV2(
        phase_id="attack", kind=PhaseKind.ATTACK, start_s=0, end_s=1, energy=1
    )
    retimer = PhaseRetimer(
        (phase,),
        1.0,
        protected_times_s=(0.4, 0.6),
        profile=TimingProfile.ENERGETIC,
    )
    assert retimer.sample(0.4).authored_time_s == pytest.approx(0.4)
    assert retimer.sample(0.6).authored_time_s == pytest.approx(0.6)
    assert retimer.sample(0.5).authored_time_s > 0.5


def test_quintic_satisfies_all_six_boundary_conditions() -> None:
    segment = QuinticSegment(
        np.asarray([0.0, 1.0]),
        np.asarray([2.0, -1.0]),
        1.5,
        start_velocity=np.asarray([0.2, -0.1]),
        end_velocity=np.asarray([-0.3, 0.4]),
        start_acceleration=np.asarray([0.5, 0.0]),
        end_acceleration=np.asarray([0.0, -0.2]),
    )
    start = segment.sample(0)
    end = segment.sample(1.5)
    assert start.position == pytest.approx([0.0, 1.0])
    assert end.position == pytest.approx([2.0, -1.0])
    assert start.velocity == pytest.approx([0.2, -0.1])
    assert end.velocity == pytest.approx([-0.3, 0.4])
    assert start.acceleration == pytest.approx([0.5, 0.0])
    assert end.acceleration == pytest.approx([0.0, -0.2])


def test_slerp_and_squad_preserve_convention_unit_norm_and_endpoints() -> None:
    identity = _quaternion((1.0, 0.0, 0.0, 0.0))
    quarter_turn = _quaternion((2**-0.5, 0.0, 0.0, 2**-0.5))
    half_turn = _quaternion((0.0, 0.0, 0.0, 1.0))
    midpoint = slerp(identity, half_turn, 0.5)
    assert midpoint.values == pytest.approx(quarter_turn.values)
    cubic = squad(identity, identity, half_turn, half_turn, 0.35)
    assert cubic.convention == identity.convention
    assert np.linalg.norm(cubic.values) == pytest.approx(1.0)
    assert squad(identity, identity, half_turn, half_turn, 0).values == pytest.approx(
        identity.values
    )
    assert squad(identity, identity, half_turn, half_turn, 1).values == pytest.approx(
        half_turn.values
    )

    xyzw = _quaternion((0.0, 0.0, 0.0, 1.0), order=QuaternionOrder.XYZW)
    with pytest.raises(ValueError, match="convention"):
        slerp(identity, xyzw, 0.5)
