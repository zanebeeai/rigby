"""Measure how far a chain reaches, direction by direction.

A reachable set is not a ball. An arm typically extends nearly its full length
along one axis and only a fraction of it along another, and the difference is not
a detail: a target placed at 84% of the *maximum* reach in a direction where the
arm only manages 60% is simply unreachable. The IK solver then spends its entire
iteration budget failing to get there, and the failure surfaces as a slow compile
rather than as the impossible request it actually was.

So the envelope is measured as a coarse spherical histogram in the chain's own
workspace frame, and every magnitude-neutral radius resolves against the bin it
points into. That is what makes "distal" mean "as far as this arm goes *that
way*" instead of "as far as this arm goes at all".
"""

from __future__ import annotations

import math

import numpy as np


AZIMUTH_BINS = 12
ELEVATION_BINS = 7


def _bin_indices(
    relative: np.ndarray,
    radius: np.ndarray,
    out: np.ndarray,
    side: np.ndarray,
    up: np.ndarray,
    azimuth_bins: int,
    elevation_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Which direction bin each sampled point falls into."""

    azimuth = np.arctan2(relative @ side, relative @ out)
    elevation = np.arcsin(np.clip((relative @ up) / radius, -1.0, 1.0))
    return (
        np.clip(
            ((azimuth + math.pi) / (2.0 * math.pi) * azimuth_bins).astype(int),
            0,
            azimuth_bins - 1,
        ),
        np.clip(
            ((elevation + math.pi / 2.0) / math.pi * elevation_bins).astype(int),
            0,
            elevation_bins - 1,
        ),
    )


def _bearing_bin(
    shape: tuple[int, int], azimuth_deg: float, elevation_deg: float
) -> tuple[int, int]:
    azimuth_bins, elevation_bins = shape
    azimuth = math.radians(((azimuth_deg + 180.0) % 360.0) - 180.0)
    elevation = math.radians(max(-89.9, min(89.9, elevation_deg)))
    return (
        min(
            azimuth_bins - 1,
            max(0, int((azimuth + math.pi) / (2.0 * math.pi) * azimuth_bins)),
        ),
        min(
            elevation_bins - 1,
            max(0, int((elevation + math.pi / 2.0) / math.pi * elevation_bins)),
        ),
    )


def build_envelope(
    points: np.ndarray,
    origin: np.ndarray,
    out: np.ndarray,
    side: np.ndarray,
    up: np.ndarray,
    *,
    azimuth_bins: int = AZIMUTH_BINS,
    elevation_bins: int = ELEVATION_BINS,
) -> np.ndarray:
    """Furthest sampled radius per direction bin, as an ``(az, el)`` grid."""

    relative = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    radius = np.linalg.norm(relative, axis=1)
    keep = radius > 1e-9
    relative, radius = relative[keep], radius[keep]
    if radius.size == 0:  # pragma: no cover - a chain always reaches somewhere
        return np.zeros((azimuth_bins, elevation_bins), dtype=float)

    azimuth_index, elevation_index = _bin_indices(
        relative, radius, out, side, up, azimuth_bins, elevation_bins
    )

    grid = np.zeros((azimuth_bins, elevation_bins), dtype=float)
    np.maximum.at(grid, (azimuth_index, elevation_index), radius)
    return _fill_empty_bins(grid)


def _fill_empty_bins(grid: np.ndarray) -> np.ndarray:
    """Give unsampled directions a conservative radius rather than zero.

    An empty bin means the sampler never landed there, which usually means the
    arm cannot go there. Filling with the smallest *observed* radius keeps the
    envelope honest -- it never promises reach that was not measured -- while
    still leaving a usable number so grounding fails with a clear reason instead
    of dividing by nothing.
    """

    filled = grid.copy()
    observed = filled[filled > 0.0]
    if observed.size == 0:  # pragma: no cover - defensive
        return filled
    floor = float(observed.min())

    azimuth_bins, elevation_bins = filled.shape
    for azimuth in range(azimuth_bins):
        for elevation in range(elevation_bins):
            if filled[azimuth, elevation] > 0.0:
                continue
            neighbours = [
                grid[(azimuth + da) % azimuth_bins, elevation + de]
                for da in (-1, 0, 1)
                for de in (-1, 0, 1)
                if 0 <= elevation + de < elevation_bins
            ]
            positive = [value for value in neighbours if value > 0.0]
            filled[azimuth, elevation] = min(positive) if positive else floor
    return filled


def directional_reach(
    envelope: np.ndarray, azimuth_deg: float, elevation_deg: float
) -> float:
    """How far the chain reaches along one bearing."""

    azimuth_index, elevation_index = _bearing_bin(
        envelope.shape, azimuth_deg, elevation_deg
    )
    return float(envelope[azimuth_index, elevation_index])


def flatten(envelope: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in envelope.reshape(-1))


def unflatten(
    values: tuple[float, ...], azimuth_bins: int, elevation_bins: int
) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(azimuth_bins, elevation_bins)


def working_frame(
    centroid: np.ndarray, origin: np.ndarray, up: np.ndarray, fallback: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The chain's ``out`` and ``side`` axes, from its own reachable centroid."""

    bearing = np.asarray(centroid, dtype=float) - np.asarray(origin, dtype=float)
    horizontal = bearing - up * float(bearing @ up)
    norm = float(np.linalg.norm(horizontal))
    out = horizontal / norm if norm > 1e-9 else np.asarray(fallback, dtype=float)
    out_norm = float(np.linalg.norm(out))
    out = out / out_norm if out_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(up, out)
    side_norm = float(np.linalg.norm(side))
    side = side / side_norm if side_norm > 1e-9 else np.array([0.0, 1.0, 0.0])
    return out, side


def build_inner_envelope(
    points: np.ndarray,
    origin: np.ndarray,
    out: np.ndarray,
    side: np.ndarray,
    up: np.ndarray,
    *,
    azimuth_bins: int = AZIMUTH_BINS,
    elevation_bins: int = ELEVATION_BINS,
) -> np.ndarray:
    """Closest sampled radius per direction bin, as an ``(az, el)`` grid.

    The other half of the shell. A serial arm cannot fold its tool onto its own
    shoulder: there is a hole in the middle of the reachable set, and its radius
    is a property of the linkage, not a rounding error. On a real industrial arm
    it is 15-25% of the maximum reach, which is precisely the range the closest
    degrees of remove were asking for.
    """

    relative = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    radius = np.linalg.norm(relative, axis=1)
    keep = radius > 1e-9
    relative, radius = relative[keep], radius[keep]
    if radius.size == 0:  # pragma: no cover - a chain always reaches somewhere
        return np.zeros((azimuth_bins, elevation_bins), dtype=float)

    azimuth_index, elevation_index = _bin_indices(
        relative, radius, out, side, up, azimuth_bins, elevation_bins
    )

    grid = np.full((azimuth_bins, elevation_bins), np.inf, dtype=float)
    np.minimum.at(grid, (azimuth_index, elevation_index), radius)
    return _fill_empty_inner_bins(grid)


def _fill_empty_inner_bins(grid: np.ndarray) -> np.ndarray:
    """Unsampled directions inherit the *largest* neighbouring inner radius.

    Conservative in the same sense as :func:`_fill_empty_bins` and in the
    opposite direction: never claim the arm can get closer than was measured.
    """

    observed = grid[np.isfinite(grid)]
    if observed.size == 0:  # pragma: no cover - defensive
        return np.zeros_like(grid)
    ceiling = float(observed.max())

    filled = grid.copy()
    azimuth_bins, elevation_bins = filled.shape
    for azimuth in range(azimuth_bins):
        for elevation in range(elevation_bins):
            if np.isfinite(filled[azimuth, elevation]):
                continue
            neighbours = [
                grid[(azimuth + da) % azimuth_bins, elevation + de]
                for da in (-1, 0, 1)
                for de in (-1, 0, 1)
                if 0 <= elevation + de < elevation_bins
            ]
            finite = [value for value in neighbours if np.isfinite(value)]
            filled[azimuth, elevation] = max(finite) if finite else ceiling
    return filled


def inner_reach(
    envelope: np.ndarray, azimuth_deg: float, elevation_deg: float
) -> float:
    """How close to the origin the chain can bring its effector on one bearing."""

    if envelope.size == 0:
        return 0.0
    azimuth_index, elevation_index = _bearing_bin(
        envelope.shape, azimuth_deg, elevation_deg
    )
    value = float(envelope[azimuth_index, elevation_index])
    return value if math.isfinite(value) else 0.0
