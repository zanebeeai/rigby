"""Severity sweeps: a family is a parameterized generator, not a fixed list.

Plan 06 section 3.2.  The point of a sweep is to locate a **detection threshold**,
which you cannot do from above it -- so the bottom of every sweep must be plausibly
sub-perceptual, and a sweep whose mildest level is already egregious measures nothing
beyond "the grader has eyes".  The existing library's mildest wrist spec is 0.85 rad,
about 49 degrees; that is the *top* of the sweep here, not the bottom.

Uniform, not adaptive.  Plan 06 section 6.3 is resolved to a uniform sweep because
bisection returns a point while plan 10 section 5.2 needs a threshold **with a
confidence interval** and section 6.2 needs the curve -- the two produce different
objects, and only one is the object those plans require.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .family import MutationFamily, Tier
from .spec import MutationSpec, Severity

#: Default sweep levels as a fraction of the family's top magnitude.  Seven points,
#: geometric rather than linear, because a detection threshold is far more likely to
#: sit near the bottom than the middle -- a linear sweep spends most of its samples
#: above the interesting region.
DEFAULT_LEVELS: tuple[Severity, ...] = (0.04, 0.08, 0.16, 0.28, 0.48, 0.68, 1.00)

#: Which tier a normalised severity falls in.  Boundaries are declared rather than
#: derived: the tiers exist so a report that covers only part of the range says so.
TIER_BOUNDS: tuple[tuple[float, Tier], ...] = (
    (0.10, Tier.SUBPERCEPTUAL),
    (0.30, Tier.MILD),
    (0.70, Tier.MODERATE),
    (1.01, Tier.SEVERE),
)


def tier_for(severity: Severity) -> Tier:
    for bound, tier in TIER_BOUNDS:
        if severity < bound:
            return tier
    return Tier.SEVERE


def sweep(
    template: MutationSpec,
    *,
    top_magnitude: float,
    unit: str,
    magnitude_key: str = "magnitude_rad",
    levels: Sequence[Severity] = DEFAULT_LEVELS,
) -> list[MutationSpec]:
    """Expand one template into a graded series.

    ``template.params[magnitude_key]`` is replaced at each level with
    ``level * top_magnitude``, and ``severity`` is set to the level.

    **``unit`` is required, and that is the point.**  ``severity`` is a *share* of
    the family's top magnitude, so "the detection threshold is 0.16" means nothing
    outside this sweep and changes meaning if the top is ever retuned.  The physical
    magnitude and its unit travel.  Lane `capture` measured the general form of this
    the hard way: a stage-duration share that was 99.8% on macOS was 93.5% on
    Windows for identical code, because the denominator moved.  **A share is not
    portable; an absolute is.**  Plan 10 section 5.2 publishes detection thresholds
    with confidence intervals, so any threshold expressed as a fraction of something
    that itself moves carries the same exposure.

    Requiring the unit here means a spec cannot reach the report without one.
    """
    if not unit.strip():
        raise ValueError(
            f"{template.id}: a sweep must declare the unit its magnitude is in; a "
            f"detection threshold reported as a bare severity is a share of a top "
            f"magnitude that may be retuned, and does not travel"
        )
    if not levels:
        raise ValueError(f"{template.id}: a sweep needs at least one level")
    if any(not 0.0 < level <= 1.0 for level in levels):
        raise ValueError(f"{template.id}: levels must lie in (0, 1]")
    if list(levels) != sorted(levels):
        raise ValueError(f"{template.id}: levels must be ascending")
    return [
        MutationSpec(
            id=f"{template.id}-{index:02d}",
            family=template.family,
            targets=template.targets,
            severity=level,
            tier=tier_for(level),
            params={
                **template.params,
                magnitude_key: level * top_magnitude,
                "magnitude_unit": unit,
                "sweep_top_magnitude": top_magnitude,
            },
            seed=template.seed,
            transform=template.transform,
            guard=template.guard,
            asserts=template.asserts,
        )
        for index, level in enumerate(levels, start=1)
    ]


def degrees_sweep(
    template: MutationSpec,
    *,
    top_magnitude_deg: float,
    levels: Sequence[Severity] = DEFAULT_LEVELS,
) -> list[MutationSpec]:
    """:func:`sweep` for families whose natural unit is degrees.

    The stored magnitude is still radians, because that is what the injector takes;
    the *declared* unit is degrees because that is what a human reasons about, and
    a sweep whose levels are quietly in the wrong unit is exactly the class of defect
    plan 06 section 5 is about.
    """
    specs = sweep(
        template,
        top_magnitude=math.radians(top_magnitude_deg),
        unit="rad",
        levels=levels,
    )
    return [
        MutationSpec(
            **{
                **{
                    field: getattr(spec, field)
                    for field in (
                        "id",
                        "family",
                        "targets",
                        "severity",
                        "tier",
                        "seed",
                        "transform",
                        "guard",
                        "asserts",
                    )
                },
                "params": {
                    **spec.params,
                    "magnitude_deg": math.degrees(spec.params["magnitude_rad"]),
                    "reported_unit": "deg",
                },
            }
        )
        for spec in specs
    ]


def subperceptual_floor(specs: Sequence[MutationSpec]) -> MutationSpec:
    """The mildest spec in a sweep.

    Named rather than indexed, because "the bottom of the sweep" is a load-bearing
    concept: plan 06 section 6.2 says a sweep floor below every grader's threshold
    is the expected and correct outcome, not a failure, and the report must present
    the curve rather than a pass/fail at one severity.
    """
    if not specs:
        raise ValueError("an empty sweep has no floor")
    return min(specs, key=lambda spec: spec.severity)


def families_covered(specs: Sequence[MutationSpec]) -> set[MutationFamily]:
    return {spec.family for spec in specs}
