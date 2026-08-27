from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from rigby_core.contracts import InterpolationKind, MotionTrackV2, TrackOwnership

from .errors import MotionCompilationError, MotionFailureReason
from .timing import QuinticSegment


@dataclass(frozen=True)
class JointTrackSample:
    position: float
    velocity: float
    acceleration: float


class JointTrackSeries:
    def __init__(self, track: MotionTrackV2, joint_name: str, duration_s: float) -> None:
        entries = [
            (keyframe.time_s, float(keyframe.joint_values[joint_name]))
            for keyframe in track.keyframes
            if joint_name in keyframe.joint_values
        ]
        if not entries:
            raise ValueError(f"track does not animate joint {joint_name!r}")
        self.track = track
        self.joint_name = joint_name
        self.times = np.asarray([entry[0] for entry in entries], dtype=np.float64)
        self.values = np.asarray([entry[1] for entry in entries], dtype=np.float64)
        if len(entries) == 1:
            self.start_s = 0.0
            self.end_s = duration_s
            self._velocities = np.zeros(1, dtype=np.float64)
            self._segments: tuple[QuinticSegment, ...] = ()
            return
        self.start_s = float(self.times[0])
        self.end_s = float(self.times[-1])
        velocities = np.zeros(len(entries), dtype=np.float64)
        for index in range(1, len(entries) - 1):
            velocities[index] = (
                self.values[index + 1] - self.values[index - 1]
            ) / (self.times[index + 1] - self.times[index - 1])
        # Identical endpoints are an authored hold/contact plateau.  Killing
        # both tangents makes the entire quintic span exactly constant.
        for index in range(len(entries) - 1):
            if self.values[index] == self.values[index + 1]:
                velocities[index] = 0.0
                velocities[index + 1] = 0.0
        self._velocities = velocities
        self._segments = tuple(
            QuinticSegment(
                self.values[index],
                self.values[index + 1],
                self.times[index + 1] - self.times[index],
                start_velocity=velocities[index],
                end_velocity=velocities[index + 1],
            )
            for index in range(len(entries) - 1)
        )

    @property
    def ownership(self) -> TrackOwnership:
        return self.track.ownership

    @property
    def priority(self) -> int:
        return self.track.priority

    @property
    def owner(self) -> str:
        return self.track.owner

    def active(self, time_s: float) -> bool:
        return self.start_s - 1e-12 <= time_s <= self.end_s + 1e-12

    def sample(self, time_s: float) -> JointTrackSample:
        if len(self.times) == 1:
            return JointTrackSample(float(self.values[0]), 0.0, 0.0)
        right = int(np.searchsorted(self.times, time_s, side="right"))
        right = min(max(right, 1), len(self.times) - 1)
        left = right - 1
        if self.track.interpolation is InterpolationKind.STEP:
            value = self.values[-1] if time_s >= self.times[-1] else self.values[left]
            return JointTrackSample(float(value), 0.0, 0.0)
        sample = self._segments[left].sample(float(time_s - self.times[left]))
        return JointTrackSample(
            float(sample.position), float(sample.velocity), float(sample.acceleration)
        )


def build_joint_series(
    tracks: tuple[MotionTrackV2, ...], duration_s: float
) -> dict[str, tuple[JointTrackSeries, ...]]:
    by_joint: dict[str, list[JointTrackSeries]] = {}
    for track in tracks:
        if track.interpolation in {InterpolationKind.SLERP, InterpolationKind.SQUAD}:
            if any(keyframe.joint_values for keyframe in track.keyframes):
                raise MotionCompilationError(
                    MotionFailureReason.UNSUPPORTED_TRACK,
                    f"Joint track {track.track_id!r} cannot use quaternion interpolation",
                )
            continue
        joint_names = {
            joint_name
            for keyframe in track.keyframes
            for joint_name in keyframe.joint_values
        }
        if not joint_names:
            raise MotionCompilationError(
                MotionFailureReason.UNSUPPORTED_TRACK,
                f"Track {track.track_id!r} has no joint-value keyframes",
            )
        for joint_name in sorted(joint_names):
            by_joint.setdefault(joint_name, []).append(
                JointTrackSeries(track, joint_name, duration_s)
            )
    result = {joint: tuple(series) for joint, series in by_joint.items()}
    validate_track_ownership(result)
    return result


def _overlap(left: JointTrackSeries, right: JointTrackSeries) -> tuple[float, float] | None:
    start = max(left.start_s, right.start_s)
    end = min(left.end_s, right.end_s)
    return (start, end) if end - start > 1e-12 else None


def validate_track_ownership(by_joint: dict[str, tuple[JointTrackSeries, ...]]) -> None:
    for joint_name, all_series in by_joint.items():
        additive = [series for series in all_series if series.ownership is TrackOwnership.ADDITIVE]
        for left, right in combinations(additive, 2):
            overlap = _overlap(left, right)
            if overlap is not None and left.owner == right.owner:
                raise MotionCompilationError(
                    MotionFailureReason.UNRESOLVED_TRACK_OWNERSHIP,
                    f"Owner {left.owner!r} adds to joint {joint_name!r} twice",
                    details={"joint": joint_name, "interval_s": overlap},
                )

        exclusive = [series for series in all_series if series.ownership is TrackOwnership.EXCLUSIVE]
        boundaries = sorted(
            {value for series in exclusive for value in (series.start_s, series.end_s)}
        )
        probes = [
            0.5 * (left + right)
            for left, right in zip(boundaries, boundaries[1:])
            if right - left > 1e-12
        ]
        for probe in probes:
            active = [series for series in exclusive if series.active(probe)]
            if len(active) < 2:
                continue
            highest = max(series.priority for series in active)
            winners = [series for series in active if series.priority == highest]
            if len(winners) > 1:
                raise MotionCompilationError(
                    MotionFailureReason.UNRESOLVED_TRACK_OWNERSHIP,
                    f"Exclusive tracks tie for joint {joint_name!r}",
                    details={
                        "joint": joint_name,
                        "track_ids": [series.track.track_id for series in winners],
                        "priority": highest,
                    },
                )


def resolve_joint_sample(
    series: tuple[JointTrackSeries, ...],
    time_s: float,
    neutral_position: float,
) -> JointTrackSample:
    active = [item for item in series if item.active(time_s)]
    exclusive = [item for item in active if item.ownership is TrackOwnership.EXCLUSIVE]
    additive = [item for item in active if item.ownership is TrackOwnership.ADDITIVE]
    if exclusive:
        highest = max(item.priority for item in exclusive)
        winners = [item for item in exclusive if item.priority == highest]
        if len(winners) != 1:
            boundary_samples = [winner.sample(time_s) for winner in winners]
            if not all(
                np.allclose(
                    (sample.position, sample.velocity, sample.acceleration),
                    (
                        boundary_samples[0].position,
                        boundary_samples[0].velocity,
                        boundary_samples[0].acceleration,
                    ),
                    atol=1e-10,
                    rtol=1e-10,
                )
                for sample in boundary_samples[1:]
            ):
                raise MotionCompilationError(
                    MotionFailureReason.UNRESOLVED_TRACK_OWNERSHIP,
                    f"Exclusive ownership remains unresolved for joint {series[0].joint_name!r}",
                )
            base = boundary_samples[0]
        else:
            base = winners[0].sample(time_s)
    else:
        base = JointTrackSample(neutral_position, 0.0, 0.0)
    additions = [item.sample(time_s) for item in additive]
    return JointTrackSample(
        base.position + sum(item.position for item in additions),
        base.velocity + sum(item.velocity for item in additions),
        base.acceleration + sum(item.acceleration for item in additions),
    )
