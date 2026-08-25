"""The signal family: graded noise on an otherwise clean trajectory.

Plan 06 section 3.3 lists additive joint jitter, velocity-profile flattening toward
constant velocity, and dead-limb.  **This module ships jitter.**  Flattening is
deferred with a stated reason rather than attempted: it lowers every derivative, and
the angular-kinematics checks are one-sided ceilings, so a flattened clip is *more*
compliant on every axis that could see it -- it would be a mutation that no
deterministic check can detect by construction, which plan 06 section 6.1 says must
be reported as "no detector exists" rather than shipped as a measurement.  Dead-limb
is the freeze in :mod:`evals.mutations.timing` restricted to one limb and belongs
with 06c's per-limb attribution work.

**Jitter is the one mutation here whose seed does real work**, so determinism is a
property to prove rather than to assume.  06a's contract says ``mutate(clip, spec)``
is byte-identical across runs for a given seed.  A ``numpy`` default generator seeded
per ``(spec, bone)`` gives that, and ``test_mutation_determinism`` covers it; what it
does *not* give is stability across numpy versions, so the seed is drawn through
``np.random.Generator(np.random.PCG64(seed))`` explicitly rather than through the
module-level singleton, whose bit stream numpy reserves the right to change.

**The magnitude is the per-frame standard deviation in degrees, not the peak.**  A
peak-parameterised jitter is not comparable across clip lengths: the more frames a
clip has, the more chances the peak has to be large, so the same nominal magnitude
would be a stronger mutation on a longer clip.  A standard deviation is a property of
the distribution the samples are drawn from and does not move with n.
"""

from __future__ import annotations

import math

import numpy as np

from rigby_poc.models import ClipResult

from .family import MutationFamily, Tier
from .inject import DOFS, add_dof, bone_dof_series, is_static
from .spec import Applicability, MutationSpec
from .sweep import DEFAULT_LEVELS, degrees_sweep

#: Top of the jitter sweep, as a per-frame standard deviation in degrees.  Set so the
#: sweep straddles ``signal.discontinuity_rad`` (20.05 degrees): a jitter of standard
#: deviation ``s`` produces frame-to-frame differences of standard deviation
#: ``s * sqrt(2)``, so the crossing is expected near 14 degrees of jitter, which sits
#: between the 0.28 and 0.48 levels of a 30-degree top.
JITTER_TOP_DEG = 30.0

#: Bones jitter is applied to.  The distal arm chain: it is the chain every corpus
#: family moves, so the mutation is applicable beyond the 14 gesture-path cases.
JITTER_BONES: tuple[str, ...] = ("rightUpperArm", "rightLowerArm", "rightHand")


def jitter_bone(
    clip: ClipResult, bone: str, dof: str, sigma_rad: float, *, seed: int
) -> ClipResult:
    """Add zero-mean Gaussian noise of ``sigma_rad`` to one DOF, per frame.

    Applied frame by frame through :func:`~evals.mutations.inject.add_dof`, which
    decomposes and recomposes in DOF coordinates -- the 06a contract rule that a
    post-multiplied delta violates.

    The noise is **not** signed against the excursion the way a ROM injection is.
    ``signed_magnitude`` exists so a single delta increases a peak rather than
    cancelling it; zero-mean noise has no single direction to sign, and forcing every
    sample outward would turn jitter into a drift, which is a different mutation with
    a different detector.
    """
    generator = np.random.Generator(np.random.PCG64(seed))
    for index in range(len(clip.frames)):
        add_dof(
            clip,
            bone,
            dof,
            float(generator.normal(0.0, sigma_rad)),
            frame_indices=(index,),
            sign_from_clip=False,
        )
    return clip


def _jitter_transform(clip: ClipResult, spec: MutationSpec) -> ClipResult:
    sigma = float(spec.params["magnitude_rad"])
    for offset, bone in enumerate(spec.params["bones"]):
        jitter_bone(clip, bone, spec.params["dof"], sigma, seed=spec.seed + offset)
    return clip


def _jitter_guard(clip: ClipResult, spec: MutationSpec) -> Applicability:
    bones = [bone for bone in spec.params["bones"] if bone_dof_series(clip, bone, spec.params["dof"])]
    if not bones:
        return Applicability(
            False,
            f"none of {', '.join(spec.params['bones'])} carries a pose in this clip, "
            f"so there is nothing to add noise to",
        )
    return Applicability(True, static_target=all(is_static(clip, bone) for bone in bones))


def jitter_sweep(
    *,
    dof: str = "flexion",
    bones: tuple[str, ...] = JITTER_BONES,
    seed: int = 20260824,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
) -> list[MutationSpec]:
    """A graded additive-jitter sweep on one DOF of the distal arm chain."""
    if dof not in DOFS:
        raise ValueError(f"unknown DOF {dof!r}; expected one of {DOFS}")
    template = MutationSpec(
        id=f"signal.jitter.{dof}",
        family=MutationFamily.SIGNAL,
        targets=("contract.clip.rotational_discontinuities",),
        severity=1.0,
        tier=Tier.SEVERE,
        params={"bones": tuple(bones), "dof": dof, "statistic": "per_frame_stddev"},
        seed=seed,
        transform=_jitter_transform,
        guard=_jitter_guard,
    )
    return degrees_sweep(template, top_magnitude_deg=JITTER_TOP_DEG, levels=levels)


def expected_frame_delta_sigma_deg(jitter_sigma_deg: float) -> float:
    """The standard deviation of the frame-to-frame difference a jitter produces.

    Two independent draws of standard deviation ``s`` differ with standard deviation
    ``s * sqrt(2)``.  Stated as a function rather than as a comment because the
    instrument test asserts against it: a jitter whose delivered frame deltas do not
    scale this way is not adding independent per-frame noise, whatever the curve
    looks like.
    """
    return jitter_sigma_deg * math.sqrt(2.0)


__all__ = [
    "JITTER_BONES",
    "JITTER_TOP_DEG",
    "expected_frame_delta_sigma_deg",
    "jitter_bone",
    "jitter_sweep",
]
