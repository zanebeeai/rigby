"""Sweep and composition mechanics, including the two ways a sweep lies quietly."""

from __future__ import annotations

import math

import pytest

from evals.corpus import load_case
from evals.corpus.loader import compile_case
from evals.mutations.compose import compose, directional_pair, strata_b_pairs
from evals.mutations.family import MutationFamily, Tier
from evals.mutations.spec import Applicability, MutationSpec
from evals.mutations.sweep import (

    DEFAULT_LEVELS,
    degrees_sweep,
    subperceptual_floor,
    sweep,
    tier_for,
)

#: Compiles corpus cases, so `medium` by input rather than by duration.
pytestmark = pytest.mark.medium


def _template(family: MutationFamily = MutationFamily.ANATOMY) -> MutationSpec:
    return MutationSpec(
        id="elbow-abduction",
        family=family,
        targets=("anatomy.elbow.off_axis",),
        severity=1.0,
        tier=Tier.SEVERE,
        params={"bone": "rightLowerArm", "dof": "abduction"},
    )


def test_a_sweep_must_declare_the_unit_of_its_magnitude() -> None:
    """A bare severity is a share of a top magnitude that may be retuned.

    Lane `capture` measured the general form: a stage-duration share of 99.8% on
    macOS was 93.5% on Windows for identical code, because the denominator moved.
    A share is not portable; an absolute is. Plan 10 section 5.2 publishes
    thresholds with intervals, so the unit has to reach the report.
    """
    with pytest.raises(ValueError, match="declare the unit"):
        sweep(_template(), top_magnitude=math.radians(45.0), unit="")


def test_every_sweep_level_carries_its_physical_magnitude_and_unit() -> None:
    specs = degrees_sweep(_template(), top_magnitude_deg=45.0)
    for spec in specs:
        assert spec.params["magnitude_unit"] == "rad"
        assert spec.params["reported_unit"] == "deg"
        assert spec.params["magnitude_deg"] == pytest.approx(
            math.degrees(spec.params["magnitude_rad"])
        )
        assert spec.params["magnitude_deg"] == pytest.approx(spec.severity * 45.0)


def test_the_bottom_of_a_sweep_is_plausibly_subperceptual() -> None:
    """Plan 06 section 3.2: you cannot locate a threshold from above it.

    The existing library's *mildest* wrist spec is 0.85 rad, about 49 degrees. Here
    that is the top of the sweep, and the floor is under two degrees.
    """
    specs = degrees_sweep(_template(), top_magnitude_deg=45.0)
    floor = subperceptual_floor(specs)
    assert floor.tier is Tier.SUBPERCEPTUAL
    assert floor.params["magnitude_deg"] < 2.0
    assert floor is min(specs, key=lambda spec: spec.severity)


def test_sweep_levels_must_be_ascending_and_within_range() -> None:
    """A sweep whose levels are unsorted still produces a curve, just a wrong one."""
    with pytest.raises(ValueError, match="ascending"):
        sweep(_template(), top_magnitude=1.0, unit="rad", levels=(0.5, 0.2))
    with pytest.raises(ValueError, match="lie in"):
        sweep(_template(), top_magnitude=1.0, unit="rad", levels=(0.0, 0.5))
    with pytest.raises(ValueError, match="lie in"):
        sweep(_template(), top_magnitude=1.0, unit="rad", levels=(0.5, 1.5))
    with pytest.raises(ValueError, match="at least one level"):
        sweep(_template(), top_magnitude=1.0, unit="rad", levels=())


def test_severity_is_monotonic_in_magnitude() -> None:
    specs = degrees_sweep(_template(), top_magnitude_deg=45.0)
    severities = [spec.severity for spec in specs]
    magnitudes = [spec.params["magnitude_deg"] for spec in specs]
    assert severities == sorted(severities)
    assert magnitudes == sorted(magnitudes)


def test_the_default_levels_span_every_tier() -> None:
    """A sweep confined to one tier cannot locate a threshold that is in another."""
    tiers = {tier_for(level) for level in DEFAULT_LEVELS}
    assert tiers == set(Tier)


def test_composition_takes_the_maximum_severity_not_the_mean() -> None:
    """A clip carrying one severe defect is at least as bad as its worst component.

    Mean would report severe-plus-subperceptual as moderate, which understates it;
    sum would leave the normalised axis.
    """
    specs = degrees_sweep(_template(), top_magnitude_deg=45.0)
    composed = compose(specs[0], specs[-1])
    assert composed.severity == specs[-1].severity
    assert composed.tier is Tier.SEVERE
    assert composed.params["components"] == (specs[0].id, specs[-1].id)


def test_composition_unions_targets_and_conjoins_applicability() -> None:
    """A composite that silently drops a component is not the mutation it names."""
    clip = compile_case(load_case("gesture-hangten-shake-right"))
    left = _template()
    right = MutationSpec(
        id="wrist-twist",
        family=MutationFamily.ANATOMY,
        targets=("anatomy.wrist.twist",),
        severity=0.5,
        tier=Tier.MODERATE,
        guard=lambda _clip, _spec: Applicability(False, "component declined"),
    )
    composed = compose(left, right)
    assert set(composed.targets) == {"anatomy.elbow.off_axis", "anatomy.wrist.twist"}

    verdict = composed.applies_to(clip)
    assert not verdict
    assert "wrist-twist" in verdict.reason, verdict.reason
    assert "component declined" in verdict.reason, verdict.reason
    # The applicable component alone still applies, so the composite's refusal is
    # the conjunction rather than a broken component.
    assert left.applies_to(clip)


def test_a_directional_pair_needs_a_genuinely_milder_member() -> None:
    """Stratum B's correct answer must actually be better.

    A pair whose 'correct' answer is not milder is an unlabelled item, not a hard
    one, and it would score as grader error rather than as an authoring mistake.
    """
    specs = degrees_sweep(_template(), top_magnitude_deg=45.0)
    with pytest.raises(ValueError, match="not milder"):
        directional_pair(specs[-1], specs[0])
    with pytest.raises(ValueError, match="not milder"):
        directional_pair(specs[0], specs[0])


def test_a_directional_pair_compares_one_axis() -> None:
    anatomy = degrees_sweep(_template(), top_magnitude_deg=45.0)
    timing = degrees_sweep(_template(MutationFamily.TIMING), top_magnitude_deg=45.0)
    with pytest.raises(ValueError, match="one axis"):
        directional_pair(anatomy[0], timing[-1])


def test_stratum_b_pairs_contain_no_base_clip() -> None:
    """The whole point of stratum B: 'always pick base' has nothing to pick.

    Plan 10 section 5.4 -- that predictor is correct on all of stratum A and at
    chance on B, so pooling them gives 0.75 with a 95% lower bound of 0.645 at
    n = 64, comfortably past a 0.50 pooled baseline.
    """
    specs = degrees_sweep(_template(), top_magnitude_deg=45.0)
    pairs = strata_b_pairs(specs)
    assert pairs
    for mild, severe in pairs:
        assert mild.severity < severe.severity
        assert mild.severity > 0.0, "a base clip has severity 0 and must not appear"
        assert mild.tier in {Tier.SUBPERCEPTUAL, Tier.MILD}
        assert severe.tier is Tier.SEVERE
