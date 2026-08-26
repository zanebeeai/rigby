"""Explicit, bounded trajectory accents for authored motion tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionAccentSpec:
    """Add an attack overshoot followed by a smaller authored rebound."""

    track_id: str
    overshoot_fraction: float = 0.0
    rebound_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("accent track_id cannot be empty")
        if not 0.0 <= self.overshoot_fraction <= 0.15:
            raise ValueError("overshoot_fraction must be in [0, 0.15]")
        if not 0.0 <= self.rebound_fraction <= 0.10:
            raise ValueError("rebound_fraction must be in [0, 0.10]")
        if self.overshoot_fraction == 0.0 and self.rebound_fraction == 0.0:
            raise ValueError("an accent must author overshoot or rebound")


def accent_fraction(progress: float, accent: MotionAccentSpec) -> float:
    """Return a C1 endpoint-zero displacement fraction of segment travel."""

    u = float(np.clip(progress, 0.0, 1.0))
    envelope = 16.0 * u * u * (1.0 - u) * (1.0 - u)
    overshoot = np.exp(-0.5 * ((u - 0.72) / 0.10) ** 2)
    rebound = np.exp(-0.5 * ((u - 0.90) / 0.055) ** 2)
    return float(
        envelope
        * (
            accent.overshoot_fraction * overshoot
            - accent.rebound_fraction * rebound
        )
    )
