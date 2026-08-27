"""Composing mutations, and why plan 10's stratum B needs it.

Plan 06 section 3.4.  Compounding gives a severe tier without pushing any single axis
to absurdity, and -- the load-bearing use -- it lets a directional pair be built from
**two mutated clips** rather than from base-versus-mutated.

That is what removes the degenerate strategy that invalidated the previous
calibration.  Plan 10 section 5.4: "always pick the base clip" is correct on every
item of stratum A and at chance on stratum B, which pools to 0.75 with a 95% lower
bound of 0.645 at n = 64 -- comfortably past a 0.50 pooled baseline.  Stratum B has
no base clip in it, so the strategy has nothing to pick, and its accuracy there is
0.5 by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

from rigby_poc.models import ClipResult

from .family import MutationFamily, Tier
from .spec import Applicability, MutationSpec, Severity


def compose(*specs: MutationSpec, id: str | None = None) -> MutationSpec:
    """One spec applying every component in order.

    Severity is the **maximum** of the components rather than the sum or the mean.
    Sum would exceed 1.0 and break the normalised axis; mean would report a clip
    carrying one severe defect and one sub-perceptual one as moderate, which is
    wrong in the direction that matters -- the clip is at least as bad as its worst
    component.

    ``targets`` is the union: a composite should be credited for any check it trips.
    Applicability is the conjunction, with the first failing reason reported, since a
    composite that silently drops a component is a different mutation than the one
    the report names.
    """
    if not specs:
        raise ValueError("compose() needs at least one spec")
    if len(specs) == 1:
        return specs[0]
    families = {spec.family for spec in specs}
    severity: Severity = max(spec.severity for spec in specs)
    targets: tuple[str, ...] = tuple(
        dict.fromkeys(target for spec in specs for target in spec.targets)
    )
    asserts: tuple[str, ...] = tuple(
        dict.fromkeys(item for spec in specs for item in spec.asserts)
    )
    composed_id = id or "+".join(spec.id for spec in specs)

    def guard(clip: ClipResult, _spec: MutationSpec) -> Applicability:
        for component in specs:
            verdict = component.applies_to(clip)
            if not verdict:
                return Applicability(
                    False, f"component {component.id} does not apply: {verdict.reason}"
                )
        return Applicability(True)

    def transform(clip: ClipResult, _spec: MutationSpec) -> ClipResult:
        for component in specs:
            if component.transform is None:
                raise ValueError(f"{component.id} has no transform")
            clip = component.transform(clip, component)
        return clip

    return MutationSpec(
        id=composed_id,
        # A composite spanning families is reported under the family of its most
        # severe component, and `components` records the rest.  Inventing a
        # "composite" family would put it in a row of the detection matrix that no
        # grader is responsible for.
        family=max(specs, key=lambda spec: spec.severity).family,
        targets=targets,
        severity=severity,
        tier=max((spec.tier for spec in specs), key=_TIER_ORDER.index),
        params={
            "components": tuple(spec.id for spec in specs),
            "families": tuple(sorted(family.value for family in families)),
        },
        seed=specs[0].seed,
        transform=transform,
        guard=guard,
        asserts=asserts,
    )


_TIER_ORDER = [Tier.SUBPERCEPTUAL, Tier.MILD, Tier.MODERATE, Tier.SEVERE]


def directional_pair(mild: MutationSpec, severe: MutationSpec) -> tuple[MutationSpec, MutationSpec]:
    """A stratum B pair: two mutated clips, no base, mild is the correct answer.

    Returned as an ordered tuple ``(better, worse)``.  The caller is responsible for
    alternating presentation position, which is what makes "always pick first" score
    0.5 -- plan 10 section 5.4 requires both the absent base *and* the position
    balance, and only one of them is this module's job.
    """
    if mild.severity >= severe.severity:
        raise ValueError(
            f"{mild.id} at {mild.severity} is not milder than {severe.id} at "
            f"{severe.severity}; a pair whose 'correct' answer is not actually better "
            f"is an unlabelled item, not a hard one"
        )
    if mild.family is not severe.family:
        raise ValueError(
            f"a directional pair compares severity on one axis; {mild.id} is "
            f"{mild.family} and {severe.id} is {severe.family}"
        )
    return mild, severe


def strata_b_pairs(
    specs: Sequence[MutationSpec], *, family: MutationFamily | None = None
) -> list[tuple[MutationSpec, MutationSpec]]:
    """Every mild/severe pairing available within a sweep.

    Pairs are drawn from one family so that "worse" means worse along one axis.
    Across families the comparison has no ground truth -- which is stratum C's job,
    not stratum B's.
    """
    pool = [spec for spec in specs if family is None or spec.family is family]
    by_family: dict[MutationFamily, list[MutationSpec]] = {}
    for spec in pool:
        by_family.setdefault(spec.family, []).append(spec)
    pairs: list[tuple[MutationSpec, MutationSpec]] = []
    for members in by_family.values():
        ordered = sorted(members, key=lambda spec: spec.severity)
        mild = [spec for spec in ordered if spec.tier in {Tier.SUBPERCEPTUAL, Tier.MILD}]
        severe = [spec for spec in ordered if spec.tier is Tier.SEVERE]
        pairs.extend(
            directional_pair(one, other) for one in mild for other in severe
        )
    return pairs
