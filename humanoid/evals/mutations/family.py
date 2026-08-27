"""Mutation families and the check axes they target.

A family is the axis a mutation perturbs; ``targets`` is what it should trip.  Both
are declared rather than inferred, because per-axis scoring depends on crediting a
grader only for detecting what a mutation actually broke (plan 06 section 3.1).
"""

from __future__ import annotations

from enum import StrEnum


class MutationFamily(StrEnum):
    """The seven families of plan 06 section 3.3."""

    ANATOMY = "anatomy"
    TIMING = "timing"
    SIGNAL = "signal"
    CONTACT = "contact"
    BALANCE = "balance"
    SEMANTIC = "semantic"
    CLIPPING = "clipping"


class Tier(StrEnum):
    """Where a spec sits in its family's severity ordering.

    ``SEVERE`` is where the 28 specs ported from ``evals/corruptions.py`` land: the
    mildest of them is 0.85 rad, about 49 degrees, which plan 06 section 1.3 calls
    out as far past anything a grader should be credited for catching.  The tiers
    exist so that a suite reporting only severe results is visibly reporting only
    severe results.
    """

    SUBPERCEPTUAL = "subperceptual"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
