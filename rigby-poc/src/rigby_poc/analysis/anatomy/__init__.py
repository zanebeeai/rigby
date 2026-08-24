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
from .neutral import (
    NO_NEUTRAL,
    REST_IS_NEUTRAL,
    rest_offset,
    rest_offsets,
    rest_relative,
    to_rest_relative,
)
from .rom import (
    DofLimit,
    RomError,
    RomViolation,
    enforceability,
    rom_checks,
    rom_limit,
    rom_limits,
    rom_violations,
)

__all__ = [
    "CONVENTIONS",
    "NO_NEUTRAL",
    "REST_IS_NEUTRAL",
    "AnatomicalFrame",
    "DofAngles",
    "DofLimit",
    "Reference",
    "RomError",
    "RomViolation",
    "all_frames",
    "bone_anatomical_frame",
    "bone_side",
    "compose",
    "convention",
    "decompose",
    "digit_bones",
    "enforceability",
    "joint_class",
    "palmar_direction",
    "radial_direction",
    "reference_direction",
    "rest_offset",
    "rest_offsets",
    "rest_relative",
    "rom_checks",
    "rom_limit",
    "rom_limits",
    "rom_violations",
    "swing_twist",
    "to_rest_relative",
    "world_axes",
]
