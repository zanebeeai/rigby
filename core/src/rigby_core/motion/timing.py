from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

import numpy as np

from rigby_core.contracts import MotionPhaseV2, PhaseKind

from .errors import MotionCompilationError, MotionFailureReason


class TimingProfile(StrEnum):
    RELAXED = "relaxed"
    NEUTRAL = "neutral"
    ENERGETIC = "energetic"


_PHASE_ORDER = {
    PhaseKind.ANTICIPATION: 0,
    PhaseKind.SETUP: 0,
    PhaseKind.ATTACK: 1,
    PhaseKind.ACTION: 1,
    PhaseKind.HOLD: 2,
    PhaseKind.RELEASE: 3,
    PhaseKind.FOLLOW_THROUGH: 3,
    PhaseKind.RECOVERY: 4,
}


def validate_phase_schedule(
    phases: Iterable[MotionPhaseV2], duration_s: float
) -> tuple[MotionPhaseV2, ...]:
    ordered = tuple(sorted(phases, key=lambda phase: (phase.start_s, phase.end_s)))
    if not ordered:
        raise MotionCompilationError(
            MotionFailureReason.INVALID_PHASE_LAYOUT,
            "At least one explicit motion phase is required",
        )
    previous_end = -np.inf
    previous_order = -1
    for phase in ordered:
        if phase.end_s > duration_s + 1e-12:
            raise MotionCompilationError(
                MotionFailureReason.INVALID_PHASE_LAYOUT,
                f"Phase {phase.phase_id!r} extends beyond motion duration",
            )
        if phase.start_s < previous_end - 1e-12:
            raise MotionCompilationError(
                MotionFailureReason.INVALID_PHASE_LAYOUT,
                f"Phase {phase.phase_id!r} overlaps the preceding phase",
            )
        order = _PHASE_ORDER[phase.kind]
        if order < previous_order:
            raise MotionCompilationError(
                MotionFailureReason.INVALID_PHASE_LAYOUT,
                f"Phase {phase.phase_id!r} moves backward in the canonical phase order",
            )
        previous_end = phase.end_s
        previous_order = order
    return ordered


@dataclass(frozen=True)
class TimeLawSample:
    progress: float
    first_derivative: float
    second_derivative: float


@dataclass(frozen=True)
class MonotoneTimeLaw:
    """Endpoint-flat, monotone time law with controllable temporal bias."""

    profile: TimingProfile = TimingProfile.NEUTRAL
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.strength) or not 0.0 <= self.strength <= 1.0:
            raise ValueError("time-law strength must lie in [0, 1]")

    @property
    def bias(self) -> float:
        if self.profile is TimingProfile.ENERGETIC:
            return 1.0 - 0.48 * self.strength
        if self.profile is TimingProfile.RELAXED:
            return 1.0 + 0.48 * self.strength
        return 1.0

    def sample(self, normalized_time: float) -> TimeLawSample:
        u = float(np.clip(normalized_time, 0.0, 1.0))
        bias = self.bias
        denominator = bias + (1.0 - bias) * u
        warped = u / denominator
        warp_first = bias / denominator**2
        warp_second = -2.0 * bias * (1.0 - bias) / denominator**3

        progress = 6.0 * warped**5 - 15.0 * warped**4 + 10.0 * warped**3
        smooth_first = 30.0 * warped**2 * (warped - 1.0) ** 2
        smooth_second = 120.0 * warped**3 - 180.0 * warped**2 + 60.0 * warped
        first = smooth_first * warp_first
        second = smooth_second * warp_first**2 + smooth_first * warp_second
        return TimeLawSample(
            progress=float(progress),
            first_derivative=float(first),
            second_derivative=float(second),
        )


def profile_for_phase(phase: MotionPhaseV2) -> TimingProfile:
    if phase.kind in {PhaseKind.ATTACK, PhaseKind.ACTION} and phase.energy >= 0.55:
        return TimingProfile.ENERGETIC
    if phase.kind in {
        PhaseKind.ANTICIPATION,
        PhaseKind.SETUP,
        PhaseKind.HOLD,
        PhaseKind.RECOVERY,
    }:
        return TimingProfile.RELAXED
    return TimingProfile.NEUTRAL


@dataclass(frozen=True)
class RetimedSample:
    authored_time_s: float
    first_derivative: float
    second_derivative: float


@dataclass(frozen=True)
class _RetimingInterval:
    start_s: float
    end_s: float
    law: MonotoneTimeLaw


class PhaseRetimer:
    """Apply a local law inside each phase without moving protected anchors."""

    def __init__(
        self,
        phases: Iterable[MotionPhaseV2],
        duration_s: float,
        *,
        protected_times_s: Iterable[float] = (),
        profile: TimingProfile | None = None,
    ) -> None:
        self.duration_s = float(duration_s)
        ordered = validate_phase_schedule(phases, self.duration_s)
        protected = {
            float(value)
            for value in protected_times_s
            if 0.0 <= float(value) <= self.duration_s
        }
        intervals: list[_RetimingInterval] = []
        for phase in ordered:
            boundaries = sorted(
                {phase.start_s, phase.end_s}
                | {
                    value
                    for value in protected
                    if phase.start_s < value < phase.end_s
                }
            )
            selected_profile = profile or profile_for_phase(phase)
            strength = max(0.0, min(1.0, phase.energy))
            law = MonotoneTimeLaw(selected_profile, strength=strength)
            intervals.extend(
                _RetimingInterval(left, right, law)
                for left, right in zip(boundaries, boundaries[1:])
            )
        self._intervals = tuple(intervals)

    def sample(self, time_s: float) -> RetimedSample:
        time_s = float(np.clip(time_s, 0.0, self.duration_s))
        for interval in self._intervals:
            if interval.start_s <= time_s <= interval.end_s:
                duration = interval.end_s - interval.start_s
                normalized = (time_s - interval.start_s) / duration
                law = interval.law.sample(normalized)
                return RetimedSample(
                    authored_time_s=interval.start_s + duration * law.progress,
                    first_derivative=law.first_derivative,
                    second_derivative=law.second_derivative / duration,
                )
        return RetimedSample(time_s, 1.0, 0.0)


@dataclass(frozen=True)
class QuinticSample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class QuinticSegment:
    """Quintic polynomial satisfying position, velocity and acceleration ends."""

    def __init__(
        self,
        start_position: np.ndarray | float,
        end_position: np.ndarray | float,
        duration_s: float,
        *,
        start_velocity: np.ndarray | float = 0.0,
        end_velocity: np.ndarray | float = 0.0,
        start_acceleration: np.ndarray | float = 0.0,
        end_acceleration: np.ndarray | float = 0.0,
    ) -> None:
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("quintic duration must be finite and positive")
        arrays = np.broadcast_arrays(
            np.asarray(start_position, dtype=np.float64),
            np.asarray(end_position, dtype=np.float64),
            np.asarray(start_velocity, dtype=np.float64),
            np.asarray(end_velocity, dtype=np.float64),
            np.asarray(start_acceleration, dtype=np.float64),
            np.asarray(end_acceleration, dtype=np.float64),
        )
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("quintic boundary conditions must be finite")
        p0, p1, v0, v1, a0, a1 = (array.copy() for array in arrays)
        duration = float(duration_s)
        c0 = p0
        c1 = v0
        c2 = 0.5 * a0
        position_residual = p1 - (c0 + c1 * duration + c2 * duration**2)
        velocity_residual = v1 - (c1 + 2.0 * c2 * duration)
        acceleration_residual = a1 - 2.0 * c2
        c3 = (
            10.0 * position_residual
            - 4.0 * velocity_residual * duration
            + 0.5 * acceleration_residual * duration**2
        ) / duration**3
        c4 = (
            -15.0 * position_residual
            + 7.0 * velocity_residual * duration
            - acceleration_residual * duration**2
        ) / duration**4
        c5 = (
            6.0 * position_residual
            - 3.0 * velocity_residual * duration
            + 0.5 * acceleration_residual * duration**2
        ) / duration**5
        self.duration_s = duration
        self._coefficients = (c0, c1, c2, c3, c4, c5)

    def sample(self, time_s: float) -> QuinticSample:
        time_s = float(np.clip(time_s, 0.0, self.duration_s))
        c0, c1, c2, c3, c4, c5 = self._coefficients
        position = c0 + c1 * time_s + c2 * time_s**2 + c3 * time_s**3 + c4 * time_s**4 + c5 * time_s**5
        velocity = c1 + 2 * c2 * time_s + 3 * c3 * time_s**2 + 4 * c4 * time_s**3 + 5 * c5 * time_s**4
        acceleration = 2 * c2 + 6 * c3 * time_s + 12 * c4 * time_s**2 + 20 * c5 * time_s**3
        return QuinticSample(position.copy(), velocity.copy(), acceleration.copy())
