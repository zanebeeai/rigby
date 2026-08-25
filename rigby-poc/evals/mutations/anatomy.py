"""The anatomy family: graded per-DOF range-of-motion violations.

Plan 06 section 3.3 gives this family three mutations -- per-DOF ROM violation on
any of 52 bones at graded magnitude, hinge off-axis injection, and finger
hyperextension -- all targeting ``anatomy.*``.  The injector is 06a's
(:mod:`evals.mutations.inject`), so the four contract rules in
:mod:`evals.mutations.spec` already hold; what this module adds is the sweep, the
target ids, and the applicability rules that decide whether a given
``(spec, case)`` pair can be scored at all.

**All three of section 3.3's anatomy mutations are covered, two of them by the same
generator.**  Per-DOF ROM violation is :func:`rom_sweep`.  Hinge off-axis injection is
:func:`hinge_off_axis_sweep`, which is ``rom_sweep`` restricted to a hinge's abduction
DOF and refusing a bone that is not a hinge.  *Finger hyperextension* needs no separate
mutation: a finger bone is a ``(bone, dof)`` pair like any other, and
``rom_sweep("rightIndexProximal", "flexion")`` builds a seven-level sweep to a 110
degree top against a target the registry resolves, on a base that is in band and moving
in ``gesture-fist-right``.  Stated here rather than left implicit, so the coverage
cannot be read as a silently dropped third mutation.

**The top of a sweep is derived, not authored.**  ``top_magnitude`` is the width of
the DOF's own typical band in ``config/rom.v1.json``.  A delta equal to the whole
typical width, applied from anywhere inside that band, necessarily lands outside it,
so the top of every sweep is a violation by construction rather than by a number
someone guessed.  It is also per ``(bone, dof)``, which is what makes the published
magnitude comparable: the knee's width is 135 degrees and the elbow's abduction
width is 10, and a single authored top would be absurd for one of them.

**Three applicability states, and the third is new.**

*Bone absent* -- not applicable.  06a's rule.

*Static target* -- applicable and tagged.  The base holds one pose and the mutated
clip holds another, so this measures capability, not a threshold.  06a's rule.

*The base clip is already out of band* -- **not applicable, and this is the state
06a did not have.**  Measured on ``fullbody-dance``: ``rightLowerArm.abduction``
peaks at -42.23 degrees against an authored max band of (-5, 5), so
``anatomy.rom.rightLowerArm.abduction`` reports ``beyond_max`` on the *unmutated*
clip and reports it identically at every level of the sweep, from +0.4 degrees to
+10.  A detection matrix scoring that pair would record a perfect detection at every
severity including the sub-perceptual floor, and the number would be a property of
the bound rather than of the mutation.  Lanes ``anatomy`` and ``judge`` reached the
same bound from the enforcement side -- 46 of 46 corpus cases are violations on it
and none are clean.  This module concurs with that measurement rather than
re-deriving it; the observation added here is the consequence for plan 06's
instrument, which is that such a pair is not a usable mutation target at all, at any
severity.

**The scope of that refusal is small, and saying so matters as much as the rule.**
The 46-of-46 rejection rate invites the conclusion that the corpus is unusable for
this family; the band partition says the opposite.  Lane ``anatomy`` generated it per
case at this module's request and the numbers were re-derived here from their file:
over 46 cases with frames, **139 of the 156 (bone, dof) pairs are within band in
every single case**, median 154 clean per case, minimum 148.  Exactly **two are never
clean anywhere, and both are elbows** -- ``leftLowerArm.abduction`` and
``rightLowerArm.abduction``.  Fifteen more are dirty in some cases only, so they are
usable per case rather than corpus-wide, the largest being
``rightLittleProximal.abduction`` at 8 of 46.  The elbow is catastrophic and almost
nothing else is.  Sizing this family off the rejection rate rather than off the band
partition would have been a decision to abandon a family that is 89% intact.

``hips.*`` is excluded from targets independently of band, on lane ``anatomy``'s
advice: it is the skeleton root rather than a joint, so a mutation there perturbs
whole-body orientation, which is a semantic question and not an anatomical one.

**Detection is read from the band, and ``status`` is a different question.**  Before
04c, :func:`~rigby_poc.analysis.anatomy.rom.rom_checks` returns ``status="pass"`` on
every result, so a matrix keyed on status would score this whole family as undetected
-- a manufactured false negative.  After 04c, ``status`` becomes meaningful but is
**not** equivalent to ``band != "within_typical"``: 82 of the 156 DOFs are enforced
and 74 are not, so an out-of-band excursion on an unenforced DOF stays
``status="pass"`` with its band recorded.  ``band`` is therefore the measurement and
``status`` is the gating answer, and this module reads the measurement.  A pair that
is out of band on an unenforced DOF must be rendered as a third state rather than as
a zero, because "this bound is not enforced" and "no excursion occurred" are
different facts.  :func:`evals.mutations.checks.rom_detected` reads ``band`` and
raises on a bone the clip never posed, rather than folding "not measured" into
"measured negative".
"""

from __future__ import annotations

import math

from rigby_poc.analysis.anatomy.rom import DofLimit, rom_limit, rom_limits
from rigby_poc.models import ClipResult

from .family import MutationFamily, Tier
from .inject import add_dof, bone_dof_series, is_static
from .spec import Applicability, MutationSpec
from .sweep import DEFAULT_LEVELS, degrees_sweep

#: A hinge's *off-axis* direction: sideways deviation, varus/valgus.  Driving a hinge
#: there is the second mutation in plan 06 section 3.3, and it is a different defect
#: from over-flexing it -- the joint goes somewhere it has no degree of freedom for.
#:
#: **Twist is deliberately not here.**  It reads like an off-axis DOF for a hinge and
#: is not one: the authored ROM gives the elbow +/-80 degrees of twist and the knee
#: -30/+10, because forearm pronation and tibial rotation are real.  Including twist
#: put ``max(off_axis)`` at 160 degrees for the elbow and 40 for the knee, and
#: :func:`is_hinge` then classified neither joint as a hinge -- a rule that answers
#: "no hinges exist" is not a rule.  Abduction alone gives 14.5:1 for the elbow and
#: 13.5:1 for the knee, and still correctly refuses the wrist (3.0:1, condyloid) and
#: the shoulder (1.04:1, ball).
HINGE_OFF_AXIS: tuple[str, ...] = ("abduction",)

#: How much wider a hinge's primary band is than its off-axis band.  Both hinges
#: clear it by roughly 3x and the two non-hinges miss it, so the boundary is not
#: sitting on top of any joint in the rig.
HINGE_RATIO = 5.0


def typical_width_deg(bone: str, dof: str) -> float:
    """The width of a DOF's typical band, which is the top of its sweep."""
    limit = rom_limit(bone, dof)
    return float(limit.typical_deg[1] - limit.typical_deg[0])


def is_hinge(bone: str, dof: str = "flexion") -> bool:
    """Whether ``bone`` carries ``dof`` far more freely than its off-axis DOFs."""
    try:
        primary = typical_width_deg(bone, dof)
    except LookupError:
        return False
    off = [
        typical_width_deg(bone, other)
        for other in HINGE_OFF_AXIS
        if other != dof and (bone, other) in rom_limits()
    ]
    if not off or max(off) <= 0.0:
        return False
    return primary / max(off) >= HINGE_RATIO


def base_band(clip: ClipResult, bone: str, dof: str) -> str | None:
    """The band the *unmutated* clip already sits in, or ``None`` if unposed."""
    series = bone_dof_series(clip, bone, dof)
    if not series:
        return None
    limit: DofLimit = rom_limit(bone, dof)
    return limit.band_of(math.degrees(max(series, key=abs)))


def rom_guard(bone: str, dof: str):
    """Applicability for a ROM mutation on one ``(bone, dof)`` pair."""

    def guard(clip: ClipResult, _spec: MutationSpec) -> Applicability:
        band = base_band(clip, bone, dof)
        if band is None:
            return Applicability(
                False, f"{bone} carries no pose in this clip, so {dof} was not measured"
            )
        if band != "within_typical":
            return Applicability(
                False,
                f"{bone}.{dof} is already {band} on the unmutated clip, so "
                f"anatomy.rom.{bone}.{dof} fires before the mutation and every "
                f"severity would score as a perfect detection of the bound rather "
                f"than of the mutation",
            )
        return Applicability(True, static_target=is_static(clip, bone))

    return guard


def _rom_transform(bone: str, dof: str):
    def transform(clip: ClipResult, spec: MutationSpec) -> ClipResult:
        return add_dof(clip, bone, dof, float(spec.params["magnitude_rad"]))

    return transform


def rom_sweep(
    bone: str,
    dof: str,
    *,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    prefix: str = "anatomy.rom",
) -> list[MutationSpec]:
    """A graded ROM-violation sweep for one ``(bone, dof)``.

    The top is :func:`typical_width_deg` for that pair, so the published magnitude
    is in degrees and the normalised severity is only the ordering (section 3.2).
    """
    template = MutationSpec(
        id=f"{prefix}.{bone}.{dof}",
        family=MutationFamily.ANATOMY,
        targets=(f"anatomy.rom.{bone}.{dof}",),
        severity=1.0,
        tier=Tier.SEVERE,
        params={"bone": bone, "dof": dof},
        transform=_rom_transform(bone, dof),
        guard=rom_guard(bone, dof),
    )
    return degrees_sweep(
        template, top_magnitude_deg=typical_width_deg(bone, dof), levels=levels
    )


def hinge_off_axis_sweep(
    bone: str, *, levels: tuple[float, ...] = DEFAULT_LEVELS
) -> list[MutationSpec]:
    """Drive a hinge sideways, on whichever off-axis DOF has a limit.

    Raises rather than returning an empty list for a bone that is not a hinge: an
    empty family is indistinguishable from a family whose checks never fire, which
    is the section 6.1 confusion this module exists to keep out of the matrix.
    """
    if not is_hinge(bone):
        raise ValueError(
            f"{bone} is not a hinge by the authored ROM document: its flexion band is "
            f"less than {HINGE_RATIO}x its widest off-axis band, so an off-axis "
            f"injection there is an ordinary ROM violation, not a hinge defect"
        )
    limits = rom_limits()
    specs: list[MutationSpec] = []
    for dof in HINGE_OFF_AXIS:
        if (bone, dof) in limits:
            specs.extend(rom_sweep(bone, dof, levels=levels, prefix="anatomy.hinge"))
    if not specs:
        raise ValueError(f"{bone} has no off-axis limit to violate")
    return specs


def generation_bones() -> tuple[str, ...]:
    """Bones ``compiler.py`` can actually drive, so their limits can fire.

    A ``mutation_only`` bone still makes a valid mutation target -- that is what the
    marking is *for* -- but it can never produce a positive from generated motion, so
    a family built for threshold measurement is built from these.
    """
    from rigby_poc.analysis.anatomy.rom import enforceability

    seen = {bone for bone, _ in rom_limits()}
    return tuple(sorted(bone for bone in seen if enforceability(bone) == "generation"))


__all__ = [
    "HINGE_OFF_AXIS",
    "HINGE_RATIO",
    "base_band",
    "generation_bones",
    "hinge_off_axis_sweep",
    "is_hinge",
    "rom_guard",
    "rom_sweep",
    "typical_width_deg",
]
