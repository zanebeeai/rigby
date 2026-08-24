"""Anatomical frames and range-of-motion checks.

Plan 04. ``frame`` derives a per-bone flexion / abduction / twist frame from the
rig's rest data and resolves local delta rotations onto it; ``conventions``
holds the one thing that is an assumption rather than geometry.
"""

from __future__ import annotations

from .conventions import CONVENTIONS, Reference, bone_side, convention, joint_class
from .frame import (
    AnatomicalFrame,
    DofAngles,
    all_frames,
    bone_anatomical_frame,
    compose,
    decompose,
    digit_bones,
    palmar_direction,
    radial_direction,
    reference_direction,
    swing_twist,
    world_axes,
)

__all__ = [
    "CONVENTIONS",
    "AnatomicalFrame",
    "DofAngles",
    "Reference",
    "all_frames",
    "bone_anatomical_frame",
    "bone_side",
    "compose",
    "convention",
    "decompose",
    "digit_bones",
    "joint_class",
    "palmar_direction",
    "radial_direction",
    "reference_direction",
    "swing_twist",
    "world_axes",
]
