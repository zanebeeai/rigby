"""How far the rig's rest pose sits from the anatomical neutral, per bone.

Plan 04 §6.4. ``decompose`` returns a delta from the rig's **rest** pose. The
rest pose is a T-pose. Every published range-of-motion column is measured from
the **anatomical position** -- arms at the sides, palms forward, legs straight,
feet flat at 90 degrees to the shin. Those are different zeros, and the gap is
large: 89.7 degrees at the shoulder, 40.9 at the neck, 27.1 at the ankle.

So a limit taken from a published column is applied as
``rest_relative = published - offset``. Getting this wrong does not produce a
slightly wrong limit; it produces a limit on the wrong band, which is how an
earlier draft of plan 04 came to report 60 degrees of elbow hyperextension in
motion whose elbow never passes straight.

The offset is measured **relative to a parent that is also at neutral**. An
earlier attempt rotated each bone's world direction onto an absolute neutral
direction, which double-counts the parent's own correction and disagreed with
clinical goniometry by the parent's rest deflection. This module's values are
cross-checked against goniometry in ``tests/test_rest_offsets.py``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from ...kinematics import rig_kinematics
from ..rig import canonical_bone_names
from .conventions import joint_class
from .frame import DofAngles, bone_anatomical_frame, decompose

_UP = np.asarray([0.0, 1.0, 0.0])
_FORWARD = np.asarray([0.0, 0.0, 1.0])

#: Joint classes whose anatomical neutral is not "in line with the parent".
#: Everything else is a straight chain at neutral: spine erect, elbow and knee
#: extended, fingers straight.
_ABSOLUTE_NEUTRAL = {
    "shoulder": -_UP,  # arms at the sides, not along the clavicle
    "hip": -_UP,  # thigh straight down; the pelvis bone points up
    "ankle": _FORWARD,  # foot flat, 90 degrees to a vertical shin
    "toes": _FORWARD,
}

#: The thumb has no non-arbitrary anatomical-position direction, so no offset
#: can be derived for it and no published thumb column transfers. This is a
#: well-formed answer, not a gap to be filled -- see plan 04 §6.1.
NO_NEUTRAL = frozenset({"thumb"})

#: Joint classes whose rest pose **is** the anatomical neutral, so the offset is
#: zero by definition rather than by measurement.
#:
#: The spine is the whole of this set, and the reason is anatomy rather than
#: convenience: the vertebral column is not straight in the anatomical position.
#: It carries a thoracic kyphosis and a cervical lordosis, and this rig's bind
#: pose reproduces them -- measured from vertical, `hips` tilts 14.5 deg back,
#: `upperChest` 12.5 deg back and `neck` 28.4 deg forward. Treating the chain as
#: "straight at neutral" charged the neck a 40.9 deg flexion offset it does not
#: have, which then put the rig's own rest pose outside its own typical band.
#: No published column measures one vertebral segment against its neighbour
#: anyway; the whole-trunk figure is the only validated quantity, which is why
#: every per-segment spine limit here is `provisional`.
REST_IS_NEUTRAL = frozenset({"spine"})


#: Bones whose first child is not along the bone. `leftHand`/`rightHand` list
#: the index metacarpal first, which sits 13.8 degrees off the palm's long axis;
#: the middle metacarpal is the anatomical axis. Plan 04 §1.5 records the same
#: fact as a 0.8984 cosine.
_LONG_AXIS_CHILD = {"leftHand": "leftMiddleProximal", "rightHand": "rightMiddleProximal"}


def _long_axis(canonical: str) -> np.ndarray | None:
    """The bone's anatomical long axis, which is not always its first child."""

    kinematics = rig_kinematics()
    node = kinematics.node_by_canonical[canonical]
    override = _LONG_AXIS_CHILD.get(canonical)
    if override is not None:
        target = kinematics.node_by_canonical[override]
        offset = kinematics.rest_world[target][:3, 3] - kinematics.rest_world[node][:3, 3]
        return offset / max(float(np.linalg.norm(offset)), 1e-12)
    return _tip_direction(node)


def _tip_direction(node_index: int) -> np.ndarray | None:
    kinematics = rig_kinematics()
    children = kinematics.nodes[node_index].get("children")
    if not children:
        return None
    offset = (
        kinematics.rest_world[children[0]][:3, 3] - kinematics.rest_world[node_index][:3, 3]
    )
    return offset / max(float(np.linalg.norm(offset)), 1e-12)


def _neutral_tip_direction(canonical: str) -> np.ndarray | None:
    kinematics = rig_kinematics()
    node = kinematics.node_by_canonical[canonical]
    cls = joint_class(canonical)
    if cls in NO_NEUTRAL:
        return None
    if cls in REST_IS_NEUTRAL:
        return _long_axis(canonical)
    if canonical == "hips":
        # The root's parent is the GLB root node, which carries the -90 degree X
        # correction and no anatomy. Its neutral is simply upright.
        return _UP
    if cls == "clavicle":
        # The clavicle's rest orientation is its anatomical position.
        return _tip_direction(node)
    absolute = _ABSOLUTE_NEUTRAL.get(cls)
    if absolute is not None:
        return absolute
    parent = kinematics.canonical_by_node.get(kinematics.parents[node])
    if parent is not None:
        return _long_axis(parent)
    return _tip_direction(kinematics.parents[node])


@lru_cache(maxsize=128)
def rest_offset(canonical: str) -> DofAngles | None:
    """Delta from the rig rest pose to anatomical neutral, on the bone's frame.

    ``None`` when the bone has no defined anatomical neutral. The sign
    convention is **neutral = rest + offset**, so a published limit becomes a
    rest-relative bound by *subtracting* this.
    """

    kinematics = rig_kinematics()
    node = kinematics.node_by_canonical[canonical]
    rest = _long_axis(canonical)
    target = _neutral_tip_direction(canonical)
    if rest is None or target is None:
        return None
    axis = np.cross(rest, target)
    norm = float(np.linalg.norm(axis))
    dot = float(np.clip(np.dot(rest, target), -1.0, 1.0))
    world_delta = (
        np.eye(3)
        if norm < 1e-9
        else Rotation.from_rotvec(axis / norm * float(np.arccos(dot))).as_matrix()
    )
    rest_world = kinematics.rest_world[node][:3, :3]
    local = Rotation.from_matrix(rest_world.T @ world_delta @ rest_world).as_quat()
    return decompose(local, bone_anatomical_frame(canonical))


@lru_cache(maxsize=1)
def rest_offsets() -> dict[str, DofAngles | None]:
    """Every canonical bone's offset. ``None`` marks "no neutral exists"."""

    return {name: rest_offset(name) for name in canonical_bone_names()}


def rest_relative(anatomical_deg: float, offset_deg: float | None) -> float:
    """The one definition of the rest/anatomical conversion in this codebase.

    The offset is the delta taking **rest to neutral**, so a rest-relative angle
    ``r`` has anatomical value ``a = r - offset`` and the inverse is
    ``r = a + offset``. A second implementation of this three-line function
    disagreed with the first on the sign, which is how ``rom.v1.json`` briefly
    put the knee's own rest pose outside the knee's own band -- so both callers
    now go through here.

    ``offset_deg`` of ``None`` means no anatomical neutral exists, in which case
    the bound was authored rest-relatively already and the frames coincide by
    fiat.
    """

    return anatomical_deg + (offset_deg or 0.0)


def to_rest_relative(canonical: str, dof: str, published_deg: float) -> float:
    """Convert a published anatomical limit into a rest-relative bound.

    Raises for a bone with no anatomical neutral, rather than silently
    returning the published value as though the two zeros coincided.
    """

    offset = rest_offset(canonical)
    if offset is None:
        raise ValueError(
            f"{canonical} has no anatomical neutral, so a published {dof} limit "
            "cannot be transferred to it; author the bound rest-relatively and "
            'mark the source kind "provisional"'
        )
    return rest_relative(published_deg, float(np.degrees(getattr(offset, f"{dof}_rad"))))


__all__ = [
    "NO_NEUTRAL",
    "REST_IS_NEUTRAL",
    "rest_offset",
    "rest_offsets",
    "rest_relative",
    "to_rest_relative",
]
