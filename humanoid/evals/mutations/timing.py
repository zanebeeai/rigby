"""The timing family: graded discontinuities in when the motion happens.

Plan 06 section 3.3 lists seven timing mutations: snap, stutter, freeze, reverse,
hold removal, ease inversion and global scaling.  **This module ships two of them,
snap and freeze**, and the reason is the same for all five that are missing: a
mutation is only usable here if it is **monotone in its own magnitude**, which plan
06 section 5's ``test_mutation_severity_monotonic`` requires and which is not free.
Stutter, reverse, hold removal and ease inversion have no graded magnitude at all --
each is a single edit that either happened or did not, so each is one point rather
than a sweep, and plan 06 section 6.3 resolved that a threshold needs a curve.
Global scaling has a magnitude and fails monotonicity for the reason below.  They are
recorded here as deferred rather than dropped: they remain valid *severe-tier* single
points, and 06c can carry them as such once the matrix can express a one-point
family.

**Global scaling by decimation was the obvious design and it is wrong.**  Dropping evenly spaced
frames compresses the motion in time and does raise the per-frame rotation delta, so
it looks like a clean severity axis.  Measured on ``fullbody-walk-forward`` (116
frames), the resulting ``contract.clip.rotational_discontinuities`` status across a
rising sweep was: fail at 23 frames dropped, **pass at 40**, fail at 57, fail at 74,
fail at 91.  The outcome depends on *which* frames land next to each other after the
drop, so it beats against the gait cycle rather than rising with severity.  A sweep
like that reports a detection threshold that is an artifact of frame phase, and
because the underlying ``max_frame_rotation_delta_rad`` also wobbles (21.31, 18.73,
21.22 degrees) the monotonicity test would have caught it only intermittently.  The
measurement is kept here rather than in a deleted branch because "we tried the
obvious axis and it is not an axis" is the finding.

**Snap is bounded, and that is what makes it usable.**  ``safety_metrics`` derives
the per-frame delta as ``2*acos(|dot(a, b)|)`` over the local delta quaternion, which
for a rigid rotation of magnitude theta about any axis is theta.  The delivered
``max_frame_rotation_delta_rad`` is therefore **not** exactly theta and the module
does not claim it is: the boundary the snap lands on also carries the clip's own
motion, so the injected rotation composes with it, and below theta the clip's
untouched peak dominates instead.  What holds exactly is the two-sided bound

    theta  <=  delivered  <=  theta + base_max_delta

which is enough for an instrument test to assert *delivered against requested* rather
than merely "something moved", and it is what catches a mis-scaled injector -- the
04b defect that delivered 43.2 degrees for a 30-degree request would breach the upper
bound at every level.  Measured: on ``fullbody-dance`` (base peak 5.64 degrees) a
45-degree request delivers 46.13; on ``fullbody-walk-forward`` (base peak 11.69) it
delivers 45.70.  Both inside the bound, and the residual shrinks as theta grows
because the injected term comes to dominate.

Unlike decimation, the axis is monotone on every case tried and the flip point does
not move with the case: both cases above cross from pass to fail between the 12.60
and 21.60 degree levels, bracketing the stated threshold.

The detection threshold is therefore known in advance rather than discovered:
``signal.discontinuity_rad`` is 0.35 rad, 20.05 degrees, so a snap fires above that
and not below, and the measured flip lands in the one sweep interval containing it.
A sweep that straddles a threshold the deterministic layer already states is a
*calibration* of the grader against it, which is exactly what plan 10 section 5.2
compares model graders to.  :data:`SNAP_TOP_DEG` is set so the sweep has levels on
both sides.
"""

from __future__ import annotations

import math

import numpy as np

from rigby_poc.models import BonePose, ClipResult, Quat

from .family import MutationFamily, Tier
from .spec import Applicability, MutationSpec
from .sweep import DEFAULT_LEVELS, degrees_sweep

#: Top of the snap sweep, in degrees.  Chosen so ``DEFAULT_LEVELS`` puts four levels
#: below ``signal.discontinuity_rad`` (20.05 degrees) and three above: 1.8, 3.6, 7.2,
#: 12.6 | 21.6, 30.6, 45.0.  A sweep entirely on one side of a known threshold
#: measures nothing about where the threshold is.
SNAP_TOP_DEG = 45.0

#: A clip needs enough frames for a boundary to exist with motion on both sides.
MIN_FRAMES = 4


def _rotate_quat(quat: list[float], axis: np.ndarray, angle_rad: float) -> list[float]:
    """Pre-multiply ``quat`` (x, y, z, w) by a rotation of ``angle_rad`` about ``axis``."""
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    dx, dy, dz = axis * math.sin(half)
    dw = math.cos(half)
    x, y, z, w = quat
    return [
        dw * x + dx * w + dy * z - dz * y,
        dw * y - dx * z + dy * w + dz * x,
        dw * z + dx * y - dy * x + dz * w,
        dw * w - dx * x - dy * y - dz * z,
    ]


#: The snap axis.  Fixed rather than random: the delivered delta is ``theta`` about
#: *any* axis, so a random axis would add a degree of freedom that changes nothing
#: measurable while making the mutation non-reproducible across numpy versions.
SNAP_AXIS = np.array([0.0, 1.0, 0.0])


def snap_at(clip: ClipResult, boundary: int, angle_rad: float) -> ClipResult:
    """Rotate every bone by ``angle_rad`` from ``boundary`` onward.

    The pose jumps once and then carries the offset, so exactly one frame boundary
    holds the injected delta.  Offsetting a single frame instead would inject the
    delta *twice*, once in and once out, and the sweep would measure a spike rather
    than a snap.
    """
    for frame in clip.frames[boundary:]:
        for name, pose in frame.bones.items():
            x, y, z, w = _rotate_quat(pose.rotation.as_list(), SNAP_AXIS, angle_rad)
            frame.bones[name] = BonePose(
                rotation=Quat(x=float(x), y=float(y), z=float(z), w=float(w)),
                position=pose.position,
            )
    return clip


def _snap_guard(clip: ClipResult, _spec: MutationSpec | None = None) -> Applicability:
    if len(clip.frames) < MIN_FRAMES:
        return Applicability(
            False,
            f"a snap needs an interior frame boundary; this clip has "
            f"{len(clip.frames)} frames and the minimum is {MIN_FRAMES}",
        )
    return Applicability(True)


def _snap_transform(clip: ClipResult, spec: MutationSpec) -> ClipResult:
    boundary = max(1, len(clip.frames) // 2)
    return snap_at(clip, boundary, float(spec.params["magnitude_rad"]))


def snap_sweep(*, levels: tuple[float, ...] = DEFAULT_LEVELS) -> list[MutationSpec]:
    """A graded pose discontinuity at the clip's midpoint.

    Applicable to every corpus case, not only the 14 that emit the gesture-path
    checks: ``contract.clip.rotational_discontinuities`` is one of the three checks
    :func:`rigby_poc.analysis.validate` emits for all 47.
    """
    template = MutationSpec(
        id="timing.snap.midpoint",
        family=MutationFamily.TIMING,
        targets=("contract.clip.rotational_discontinuities",),
        severity=1.0,
        tier=Tier.SEVERE,
        params={"boundary": "midpoint", "axis": "y"},
        transform=_snap_transform,
        guard=_snap_guard,
    )
    return degrees_sweep(template, top_magnitude_deg=SNAP_TOP_DEG, levels=levels)


def _freeze_transform(clip: ClipResult, spec: MutationSpec) -> ClipResult:
    """Hold the midpoint pose for a share of the clip, then resume.

    Frames are replaced, never removed: the clip keeps its duration and frame count,
    so a downstream consumer cannot mistake the mutation for a shorter clip and the
    only thing that changed is that motion stopped.
    """
    total = len(clip.frames)
    held = int(round(float(spec.params["frames"])))
    if held <= 0:
        return clip
    start = max(1, (total - held) // 2)
    frozen = clip.frames[start].bones
    for frame in clip.frames[start : start + held]:
        for name, pose in frozen.items():
            x, y, z, w = pose.rotation.as_list()
            frame.bones[name] = BonePose(
                rotation=Quat(x=x, y=y, z=z, w=w), position=pose.position
            )
    return clip


def _freeze_guard(clip: ClipResult, spec: MutationSpec) -> Applicability:
    total = len(clip.frames)
    held = int(round(float(spec.params["frames"])))
    if total < MIN_FRAMES:
        return Applicability(False, f"a freeze needs at least {MIN_FRAMES} frames")
    if held >= total - 1:
        return Applicability(
            False,
            f"the freeze is {held} frames and the clip is {total}; freezing the whole "
            f"clip removes the motion rather than interrupting it",
        )
    return Applicability(True)


def freeze_sweep(
    *, top_frames: int = 24, levels: tuple[float, ...] = DEFAULT_LEVELS
) -> list[MutationSpec]:
    """A graded dead hold in the middle of the motion.

    **The magnitude is a count of frames, not a share of the clip.**  Corpus clips run
    from 103 to 116 frames and a share would mean a different number of held frames
    per case, so a threshold reported in shares could not be compared across cases.
    The cross-cutting rule measured across four lanes is that a count is invariant to
    load, platform and CPU where a ratio is not; the same reasoning applies to a
    denominator that varies per case.

    Freezing raises no discontinuity -- it lowers every delta -- so this targets the
    angular-kinematics axis instead, which is emitted on the 14 gesture-path cases.
    On the other 33 it has no detector, and the detection matrix must render that as
    "no detector exists" rather than as a zero.
    """
    template = MutationSpec(
        id="timing.freeze.midpoint",
        family=MutationFamily.TIMING,
        targets=("signal.angular.velocity", "signal.angular.acceleration"),
        severity=1.0,
        tier=Tier.SEVERE,
        params={"placement": "midpoint"},
        transform=_freeze_transform,
        guard=_freeze_guard,
    )
    from dataclasses import replace as _replace

    from .sweep import sweep

    specs = sweep(
        template,
        top_magnitude=float(top_frames),
        unit="frames",
        magnitude_key="frames",
        levels=levels,
    )
    # A count is published as a count.  `sweep` multiplies the top by the level and
    # would otherwise put 1.9 frames in the report, which is not a thing a clip can
    # hold; the transform would round it and the published magnitude would then
    # disagree with the mutation actually applied.
    return [
        _replace(spec, params={**spec.params, "frames": int(round(spec.params["frames"]))})
        for spec in specs
    ]


__all__ = [
    "MIN_FRAMES",
    "SNAP_AXIS",
    "SNAP_TOP_DEG",
    "freeze_sweep",
    "snap_at",
    "snap_sweep",
]
