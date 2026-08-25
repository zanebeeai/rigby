"""The clipping family: drive a limb through the torso.

Plan 06 section 3.3 lists two mutations here, limb-through-torso and
finger-through-palm, both targeting ``anatomy.self_collision``.  **This module ships
limb-through-torso.**  Finger-through-palm is deferred with a reason:
:func:`~rigby_poc.analysis.gesture._arm_self_collision` samples the upper arm and
forearm segments only and has no finger geometry at all, so a finger driven through
the palm has **no detector** -- the section 6.1 outcome that must be reported as a
gap rather than shipped as a mutation with a zero detection rate.

**This family answers an open question about a check, and the answer is good news.**
``anatomy.arm.self_collision`` is one of the three gates TRACKING records as
untrippable: zero self-collisions over 517 swept compiles, and zero over lane
``capture``'s 20 ordinary ones.  A gate that never fires is either a dead gate or a
live gate that generated motion never reaches, and those need different responses --
the first should be deleted, the second is a real gate guarding a region the
compiler happens not to visit.  **Measured here: it is the second.**  A graded
flexion injection into ``rightUpperArm`` trips it, going 0 -> 44 -> 59 collision
frames across the sweep on ``gesture-fist-right``.  So the check is live, and the
517-compile zero is a statement about the compiler's reachable set rather than about
the check.

**It does not trip on every gesture case, and that is the honest result.**  Over five
cases at the same sweep: ``gesture-fist-right``, ``gesture-point-right`` and
``gesture-thumbsup-right`` reach 44 frames at 61.2 degrees and 59 at 90;
``strike-jab-left`` and ``gesture-peace-left`` stay at 0 through the whole sweep,
because their arm never passes near enough to the torso ellipse for the injection to
push it inside.  Monotone on all five.  Per-``(spec, case)`` applicability is plan 06
section 6.1's rule and this is a clean instance of it: the pairs that stay at zero
are a real "this mutation cannot reach this check on this case", not a detection
failure, and the matrix must render them apart.

**The axis is upper-arm flexion, and the two rejected alternatives are recorded.**
``rightUpperArm.abduction`` trips it too but as a step -- 0 at -30 degrees and 59 at
-45 and everywhere beyond -- so it has no resolution to locate a threshold inside.
``rightLowerArm.flexion`` is worse than useless: 40 frames at +60, 27 at +90, **0 at
+120**, because the forearm swings through the torso and out the far side.  A sweep
on that axis would report detection falling as severity rises, which is the shape of
a real finding rather than of a bad axis, and ``test_mutation_severity_monotonic``
is the only thing that separates them.
"""

from __future__ import annotations

from rigby_poc.models import ClipResult

from .family import MutationFamily, Tier
from .inject import add_dof, bone_dof_series
from .spec import Applicability, MutationSpec
from .sweep import DEFAULT_LEVELS, degrees_sweep

#: The bone and DOF driven into the torso.  See the module docstring for the two
#: alternatives measured and rejected.
CLIP_BONE = "rightUpperArm"
CLIP_DOF = "flexion"

#: Top of the sweep in degrees.  At 90 the collision count saturates at the number of
#: frames in the presenting phase, and the first non-zero level sits at 61.2, so the
#: sweep has the transition inside it rather than at an endpoint.
CLIP_TOP_DEG = 90.0


def _clip_transform(clip: ClipResult, spec: MutationSpec) -> ClipResult:
    # `sign_from_clip=False`: the sign here is not a matter of increasing an
    # excursion, it is the direction of the torso. Letting `signed_magnitude` pick
    # would flip the injection on any case whose arm already swings negative, and
    # the mutation would then drive the limb *away* from the body -- monotone,
    # plausible, and measuring nothing.
    return add_dof(
        clip,
        CLIP_BONE,
        CLIP_DOF,
        float(spec.params["magnitude_rad"]),
        sign_from_clip=False,
    )


def _clip_guard(clip: ClipResult, _spec: MutationSpec) -> Applicability:
    if not bone_dof_series(clip, CLIP_BONE, CLIP_DOF):
        return Applicability(
            False, f"{CLIP_BONE} carries no pose in this clip"
        )
    return Applicability(True)


def limb_through_torso_sweep(
    *, levels: tuple[float, ...] = DEFAULT_LEVELS
) -> list[MutationSpec]:
    """A graded shoulder-flexion injection that drives the arm into the torso.

    Emitted only on the 14 corpus cases that reach the gesture path, and reaching a
    non-zero collision count on a subset of those.  Both facts belong in the matrix
    as distinct states: no detector on 33 cases, detector present and unreached on
    some of the remaining 14.
    """
    template = MutationSpec(
        id="clipping.limb_through_torso",
        family=MutationFamily.CLIPPING,
        targets=("anatomy.arm.self_collision",),
        severity=1.0,
        tier=Tier.SEVERE,
        params={"bone": CLIP_BONE, "dof": CLIP_DOF},
        transform=_clip_transform,
        guard=_clip_guard,
    )
    return degrees_sweep(template, top_magnitude_deg=CLIP_TOP_DEG, levels=levels)


__all__ = [
    "CLIP_BONE",
    "CLIP_DOF",
    "CLIP_TOP_DEG",
    "limb_through_torso_sweep",
]
