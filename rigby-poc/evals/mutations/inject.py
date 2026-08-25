"""Injecting an anatomical delta into a clip, correctly.

Two rules, both learned from defects that produce plausible monotonic curves.

**Decompose, add, recompose.**  ``existing * delta`` is exact only when the bone
carries nothing but the target DOF.  Lane ``anatomy`` measured the failure: on
``fullbody-dance`` frame 0, where the elbow carries flexion as well as abduction, a
30-degree abduction injection delivered a **43.2-degree** change.  Adding in DOF
coordinates is exact -- ``decompose``/``compose`` round-trip to 1e-9 -- and it is also
the right *meaning*: "this DOF, plus X degrees", not "times this rotation".

**Sign the delta against the excursion.**  Injecting +30 degrees into a bone already
at -41.8 *reduces* the peak by 30.  In a severity sweep that reads as a detection
failure at high severity, which is the shape of a real finding rather than of a bug.
:func:`signed_magnitude` resolves the sign from the clip.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from rigby_poc.analysis.anatomy import (
    DofAngles,
    bone_anatomical_frame,
    compose,
    decompose,
)
from rigby_poc.analysis.anatomy.frame import decompose_series
from rigby_poc.models import BonePose, ClipResult, Quat

#: A bone whose rotation varies by less than this across a clip is treated as
#: static: mutating it perturbs an authored constant rather than motion.
STATIC_EPSILON_RAD = 1e-6

#: The three DOF names, in ``DofAngles`` order.
DOFS = ("flexion", "abduction", "twist")


def bone_dof_series(clip: ClipResult, bone: str, dof: str) -> list[float]:
    """The per-frame value of one DOF, in radians, relative to the rig's rest pose.

    Rest-relative, **not** anatomical: ``decompose`` measures from the T-pose, which
    is not the anatomical neutral.  A caller that needs an anatomical angle must add
    the rest offset -- shoulder abduction alone differs by -89.7 degrees, because a
    T-pose *is* 90 degrees of abduction.
    """
    quaternions = [
        clip_frame.bones[bone].rotation.as_list()
        for clip_frame in clip.frames
        if bone in clip_frame.bones
    ]
    if not quaternions:
        return []
    # Batched: lane `anatomy` measured the scalar loop at 364 ms per 125 frames,
    # over plan 02 section 5's ceiling from a single report-only check.
    flexion, abduction, twist = decompose_series(
        np.asarray(quaternions, dtype=float), bone_anatomical_frame(bone)
    )
    return [float(value) for value in {"flexion": flexion, "abduction": abduction, "twist": twist}[dof]]


def is_static(clip: ClipResult, bone: str) -> bool:
    """Whether ``bone`` holds one rotation for the whole clip."""
    seen: list[list[float]] = []
    for clip_frame in clip.frames:
        pose = clip_frame.bones.get(bone)
        if pose is None:
            continue
        seen.append(pose.rotation.as_list())
    if len(seen) < 2:
        return True
    first = seen[0]
    return all(
        all(abs(value - reference) <= STATIC_EPSILON_RAD for value, reference in zip(item, first))
        for item in seen[1:]
    )


def signed_magnitude(values: Iterable[float], magnitude_rad: float) -> float:
    """Give ``magnitude_rad`` the sign that *increases* the peak excursion.

    A delta applied against the existing excursion cancels it, which measures the
    injector rather than the check.  The sign is taken from the extremum of largest
    absolute value, so a bone already swinging negative is pushed further negative.
    """
    series = list(values)
    if not series:
        return magnitude_rad
    extremum = max(series, key=abs)
    magnitude = abs(magnitude_rad)
    return -magnitude if extremum < 0.0 else magnitude


def add_dof(
    clip: ClipResult,
    bone: str,
    dof: str,
    magnitude_rad: float,
    *,
    frame_indices: Iterable[int] | None = None,
    sign_from_clip: bool = True,
) -> ClipResult:
    """Add ``magnitude_rad`` to one DOF of ``bone``, in place, on a copy's frames.

    ``clip`` is mutated: callers hold a deep copy, made by ``MutationSpec.apply``.
    """
    if dof not in DOFS:
        raise ValueError(f"unknown DOF {dof!r}; expected one of {DOFS}")
    anatomical = bone_anatomical_frame(bone)
    delta = (
        signed_magnitude(bone_dof_series(clip, bone, dof), magnitude_rad)
        if sign_from_clip
        else magnitude_rad
    )
    wanted = set(frame_indices) if frame_indices is not None else None
    for index, clip_frame in enumerate(clip.frames):
        if wanted is not None and index not in wanted:
            continue
        pose = clip_frame.bones.get(bone)
        if pose is None:
            continue
        angles = decompose(pose.rotation.as_list(), anatomical)
        shifted = DofAngles(
            **{
                f"{name}_rad": getattr(angles, f"{name}_rad") + (delta if name == dof else 0.0)
                for name in DOFS
            }
        )
        x, y, z, w = compose(shifted, anatomical)
        clip_frame.bones[bone] = BonePose(
            rotation=Quat(x=float(x), y=float(y), z=float(z), w=float(w)),
            position=pose.position,
        )
    return clip


def peak_dof(clip: ClipResult, bone: str, dof: str) -> float:
    """The largest absolute value of one DOF across the clip, in radians."""
    series = bone_dof_series(clip, bone, dof)
    return max((abs(value) for value in series), default=0.0)


def degrees(radians: float) -> float:
    return math.degrees(radians)
