"""Which anatomical motion each bone's positive flexion and abduction mean.

This module is the one place where anatomy enters as a *convention* rather than
as geometry. It does **not** name axes. It names, per bone, the direction the
bone's distal tip travels under positive flexion and under positive abduction,
expressed as a world-space reference derived from the rig profile's declared
``up_axis`` and ``forward_axis``.

:mod:`rigby_poc.analysis.anatomy.frame` then *derives* which local axis realises
that motion. The split matters: plan 04 §6.1 warns that a frame whose axes are
labelled by assumption produces confident garbage. Here the assumption is a
sentence about a joint ("the knee flexes the heel toward the buttock"), which a
reader can check against an anatomy text, rather than an axis index, which they
cannot.
"""

from __future__ import annotations

from enum import Enum


class Reference(Enum):
    """A world-space direction the distal tip moves in, at the rest pose."""

    ANTERIOR = "anterior"
    """Forward, along the profile's ``forward_axis``."""

    POSTERIOR = "posterior"
    SUPERIOR = "superior"
    """Up, along the profile's ``up_axis``."""

    INFERIOR = "inferior"
    LATERAL = "lateral"
    """Away from the body midline: +X for a left bone, -X for a right one."""

    MEDIAL = "medial"
    LEFTWARD = "leftward"
    """Toward the character's left. Used for midline bones, which have no
    lateral side of their own."""

    PALMAR = "palmar"
    """Toward the palm. Derived from the digits, not declared -- see
    :func:`~rigby_poc.analysis.anatomy.frame.palmar_direction`."""

    RADIAL = "radial"
    """Toward the thumb side of the hand. Derived from the knuckle line rather
    than declared: in this rig's T-pose the radial side points anteriorly, not
    laterally, so a fixed world direction would be wrong. See
    :func:`~rigby_poc.analysis.anatomy.frame.radial_direction`."""


#: The joint classes the 52 canonical bones fall into, and what positive
#: flexion and abduction mean for each. Every entry is a claim about anatomy
#: and carries the sentence that justifies it.
CONVENTIONS: dict[str, tuple[Reference, Reference, str]] = {
    "spine": (
        Reference.ANTERIOR,
        Reference.LEFTWARD,
        (
            "trunk flexion bends forward; positive lateral flexion bends to the "
            "character's left"
        ),
    ),
    "clavicle": (
        Reference.SUPERIOR,
        Reference.ANTERIOR,
        (
            "scapular elevation raises the shoulder; positive abduction is "
            "protraction, drawing the shoulder forward"
        ),
    ),
    "shoulder": (
        Reference.ANTERIOR,
        Reference.SUPERIOR,
        (
            "shoulder flexion carries the humerus forward; abduction raises it in "
            "the frontal plane. Both are measured from this rig's T-pose rest, not "
            "from the anatomical position, so they are horizontal flexion and "
            "elevation. Published shoulder ROM columns are quoted from arms-at-side "
            "and do not transfer to these numbers unchanged"
        ),
    ),
    "elbow": (
        Reference.ANTERIOR,
        Reference.SUPERIOR,
        (
            "in this rig's palm-down T-pose the elbow hinge axis is vertical, so "
            "flexion sweeps the forearm forward across the body; the elbow has no "
            "anatomical abduction, and off-axis motion is signed superior"
        ),
    ),
    "wrist": (
        Reference.PALMAR,
        Reference.RADIAL,
        (
            "wrist flexion is palmar flexion; positive deviation is radial, toward "
            "the thumb"
        ),
    ),
    "hip": (
        Reference.ANTERIOR,
        Reference.LATERAL,
        (
            "hip flexion carries the thigh forward; abduction swings it away from "
            "the midline"
        ),
    ),
    "knee": (
        Reference.POSTERIOR,
        Reference.LATERAL,
        (
            "knee flexion draws the heel toward the buttock; the knee has no "
            "anatomical abduction, and off-axis motion is signed lateral (valgus)"
        ),
    ),
    "ankle": (
        Reference.INFERIOR,
        Reference.LATERAL,
        (
            "positive ankle flexion is plantarflexion, pointing the toes down; "
            "negative is dorsiflexion. Positive abduction is eversion"
        ),
    ),
    "toes": (
        Reference.INFERIOR,
        Reference.LATERAL,
        "toe flexion curls the toes toward the sole",
    ),
    "digit": (
        Reference.PALMAR,
        Reference.RADIAL,
        (
            "digit flexion curls the segment toward the palm; positive abduction "
            "is splay toward the thumb side, the hand-ROM radial/ulnar convention"
        ),
    ),
    "thumb": (
        Reference.PALMAR,
        Reference.RADIAL,
        (
            "thumb flexion is palmar, as for the other digits. Note that the rig's "
            "own crate_grip preset closes the thumb by adduction across the palm "
            "rather than by flexion, so the thumb is the one chain where the "
            "authored asset and the anatomical frame disagree; see "
            "tests.test_anatomical_frame. Its limits are the least trustworthy in "
            "the table"
        ),
    ),
}


_MIDLINE = {
    "hips": "spine",
    "spine": "spine",
    "chest": "spine",
    "upperChest": "spine",
    "neck": "spine",
    "head": "spine",
}

_LIMB = {
    "Shoulder": "clavicle",
    "UpperArm": "shoulder",
    "LowerArm": "elbow",
    "Hand": "wrist",
    "UpperLeg": "hip",
    "LowerLeg": "knee",
    "Foot": "ankle",
    "Toes": "toes",
}

DIGITS = ("Thumb", "Index", "Middle", "Ring", "Little")


def bone_side(canonical: str) -> str | None:
    """``"left"``, ``"right"``, or ``None`` for a midline bone."""

    if canonical.startswith("left"):
        return "left"
    if canonical.startswith("right"):
        return "right"
    return None


def joint_class(canonical: str) -> str:
    """The joint class a canonical bone belongs to.

    Raises for a bone this module has no convention for, which is what stops a
    new rig silently getting a default frame.
    """

    if canonical in _MIDLINE:
        return _MIDLINE[canonical]
    side = bone_side(canonical)
    if side is not None:
        stem = canonical[len(side) :]
        if stem in _LIMB:
            return _LIMB[stem]
        if stem.startswith("Thumb"):
            return "thumb"
        if any(stem.startswith(digit) for digit in DIGITS):
            return "digit"
    raise KeyError(f"no anatomical convention for canonical bone {canonical!r}")


def convention(canonical: str) -> tuple[Reference, Reference, str]:
    """The (flexion, abduction, rationale) convention for one bone."""

    return CONVENTIONS[joint_class(canonical)]


__all__ = [
    "CONVENTIONS",
    "DIGITS",
    "Reference",
    "bone_side",
    "convention",
    "joint_class",
]
