"""Resolve magnitude-neutral terms against one robot's measurements.

Every function here takes an ordinal and a measured scalar and returns a number
in real units. Nothing above this module has ever seen a metre, and nothing below
it has ever seen an ordinal. That is the whole seam.

The scaling is geometric rather than additive throughout: ``speed = +1`` means
"about forty percent quicker than this robot's comfortable pace", not "plus
0.4 m/s". An additive step would mean something entirely different on a desktop
arm than on a long-reach one, which is exactly the body-dependence the design is
built to avoid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schema.program import MannerV1


# One ordinal step is this much more or less. Chosen so the full range of an axis
# spans roughly a factor of four end to end -- wide enough to be visible, narrow
# enough that the extremes stay inside a real robot's limits.
SPEED_STEP = 1.4
AMPLITUDE_STEP = 1.25
EFFORT_STEP = 1.3

MIN_SEGMENT_DURATION_S = 0.25
MAX_SEGMENT_DURATION_S = 12.0

# Oscillation cycles per repetition ordinal, when no exact count was stated.
_REPETITION_CYCLES = {-2: 1, -1: 2, 0: 3, 1: 5, 2: 8}


@dataclass(frozen=True, slots=True)
class ResolvedManner:
    """A manner co-event expressed in this robot's units."""

    speed_mps: float
    amplitude_scale: float
    effort_scale: float
    cycles: int
    speed_ordinal: int
    """The requested pace as an ordinal, kept alongside the resolved speed.

    Segment timing works from joint travel against velocity limits rather than
    from a Cartesian speed, so it needs the ordinal itself: the metres-per-second
    figure answers a different question."""

    hard_endpoints: bool
    """Precision at or above +1 pins the endpoints so retiming cannot drift them."""

    smoothness: int
    rhythm: int


def resolve_manner(manner: MannerV1, *, neutral_speed_mps: float) -> ResolvedManner:
    speed = neutral_speed_mps * (SPEED_STEP**manner.speed)
    return ResolvedManner(
        speed_mps=max(speed, 1e-4),
        amplitude_scale=AMPLITUDE_STEP**manner.amplitude,
        effort_scale=EFFORT_STEP**manner.effort,
        cycles=resolve_cycles(manner),
        speed_ordinal=manner.speed,
        hard_endpoints=manner.precision >= 1,
        smoothness=manner.smoothness,
        rhythm=manner.rhythm,
    )


def resolve_cycles(manner: MannerV1) -> int:
    """How many times over.

    An exact count wins outright. Cardinality is the one thing language *does*
    commit to exactly -- "three times" is three on any body -- so a stated count
    is never reinterpreted as an ordinal.
    """

    if manner.repetition_count is not None:
        return int(manner.repetition_count)
    return _REPETITION_CYCLES[manner.repetition]


def duration_for_path(
    path_length_m: float,
    *,
    speed_mps: float,
    minimum_s: float = MIN_SEGMENT_DURATION_S,
) -> float:
    """How long a path of this length takes at this pace.

    Clamped at both ends. Too short and the trajectory violates velocity limits
    the moment it compiles; too long and a demo becomes unwatchable. Both bounds
    are stated rather than emergent so a clamped duration can be recognised.
    """

    if speed_mps <= 0.0:  # pragma: no cover - defensive
        raise ValueError("speed must be positive")
    raw = path_length_m / speed_mps
    return float(min(MAX_SEGMENT_DURATION_S, max(minimum_s, raw)))


def path_length(points: list) -> float:
    total = 0.0
    for first, second in zip(points, points[1:]):
        total += float(
            math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(first, second)))
        )
    return total


def resolve_radius_fraction(
    base_fraction: float, *, amplitude_scale: float, ceiling: float = 0.95
) -> float:
    """Scale a remove fraction by manner amplitude, without leaving the workspace.

    The ceiling is a real limit, not a taste: past it the target is outside what
    the arm was measured to reach, and asking for it produces an IK failure
    rather than a bigger gesture.
    """

    scaled = base_fraction * amplitude_scale
    return float(min(ceiling, max(0.0, scaled)))
