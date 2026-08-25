"""Per-bone anatomical frames derived from the rig's own rest data.

Plan 04 §1.4: everything this needs is already parsed at startup.
:class:`~rigby_poc.kinematics.RigKinematics` exposes the rest local transform,
the rest world transform and the parent chain for all 52 canonical bones, so an
anatomical frame is a derivation, not new asset data.

The derivation has two halves, deliberately separated:

*Geometry*, which is exact. The longitudinal (twist) axis is local +Y --
verified against the child offset for all 52 bones, cosine 1.0 for 50 of them
and 0.8984 for the two hands, whose first child is the index metacarpal rather
than a continuation of the palm. The two cross-axes are the rest rotation's X
and Z columns, orthonormal by construction.

*Convention*, which is an assumption, and lives in
:mod:`rigby_poc.analysis.anatomy.conventions`. It says what positive flexion
means as a motion, never as an axis. This module then asks which signed cross
axis actually produces that motion, and refuses to answer when the two
candidates are too close to call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from ...kinematics import RigKinematics, rig_kinematics
from ..rig import canonical_bone_names, rig_profile
from .conventions import DIGITS, Reference, bone_side, convention, joint_class

#: Minimum |cos| between the winning candidate's tip motion and the declared
#: anatomical reference. Below this the convention is not realised by either
#: cross axis and the derivation refuses.
MIN_ALIGNMENT = 0.55

#: Minimum gap between the winning and runner-up candidate alignments. Below
#: this the two cross axes are equally good answers, which means the frame
#: would be a coin flip rather than a derivation.
MIN_MARGIN = 0.20

#: Minimum cosine between local +Y and the bone's child offset for +Y to be
#: accepted as the longitudinal axis. The two hands sit at 0.8984.
MIN_LONGITUDINAL_COSINE = 0.85


@dataclass(frozen=True)
class AnatomicalFrame:
    """One bone's flexion / abduction / twist axes, in its local rest frame.

    All three axes are unit, mutually orthogonal, and *signed*: rotating the
    bone by ``+theta`` about :attr:`flexion_axis` produces ``+theta`` flexion.
    The plan's §3.1 sketch carried a separate ``flexion_sign``; folding the sign
    into the axis is the same information with one source of truth instead of
    two, so the sketch was corrected rather than followed.
    """

    bone: str
    twist_axis: np.ndarray
    flexion_axis: np.ndarray
    abduction_axis: np.ndarray
    joint_class: str
    rationale: str
    flexion_world: np.ndarray
    """World direction the distal tip travels under positive flexion, at rest."""

    abduction_world: np.ndarray
    longitudinal_cosine: float
    """Cosine between local +Y and the child offset. 1.0 is a perfect bone."""

    flexion_margin: float
    """How much better the chosen flexion axis matched the convention than the
    runner-up. Small values mean the label is weakly determined."""

    abduction_margin: float

    def matrix(self) -> np.ndarray:
        """Columns ``(flexion, abduction, twist)``, for projecting a rotvec."""

        return np.column_stack((self.flexion_axis, self.abduction_axis, self.twist_axis))


@dataclass(frozen=True)
class DofAngles:
    """A local delta rotation resolved onto one bone's anatomical frame."""

    flexion_rad: float
    abduction_rad: float
    twist_rad: float

    @property
    def swing_rad(self) -> float:
        return math.hypot(self.flexion_rad, self.abduction_rad)

    def as_degrees(self) -> tuple[float, float, float]:
        return (
            math.degrees(self.flexion_rad),
            math.degrees(self.abduction_rad),
            math.degrees(self.twist_rad),
        )


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _child_index(kinematics: RigKinematics, canonical: str) -> int:
    node = kinematics.node_by_canonical[canonical]
    children = kinematics.nodes[node].get("children", [])
    if not children:
        raise ValueError(f"{canonical} has no child, so it has no bone direction")
    return int(children[0])


def _tip_offset_world(kinematics: RigKinematics, canonical: str) -> np.ndarray:
    """Rest world vector from the bone's own pivot to its distal tip."""

    node = kinematics.node_by_canonical[canonical]
    child = _child_index(kinematics, canonical)
    return kinematics.rest_world[child][:3, 3] - kinematics.rest_world[node][:3, 3]


@lru_cache(maxsize=1)
def world_axes() -> dict[str, np.ndarray]:
    """The rig profile's declared world axes, read rather than assumed."""

    profile = rig_profile()
    up = profile["up_axis"]
    forward = profile["forward_axis"]
    if up != "Y" or forward != "+Z":
        raise ValueError(
            f"anatomical frames assume a Y-up, +Z-forward profile; got up={up!r} "
            f"forward={forward!r}"
        )
    return {
        "superior": np.asarray([0.0, 1.0, 0.0]),
        "anterior": np.asarray([0.0, 0.0, 1.0]),
        "leftward": np.asarray([1.0, 0.0, 0.0]),
    }


@lru_cache(maxsize=2)
def palmar_direction(side: str) -> np.ndarray:
    """World direction the fingertips travel under digit flexion, at rest.

    Not declared. Derived from ``grip_presets.crate_grip.curl_axis`` in the rig
    profile -- an authored, cited config value whose sign is independently
    checked by :mod:`tests.test_anatomical_frame`, which asserts that applying
    the preset closes the hand. Plan 04 §6.2 flags that the *magnitudes* in
    ``grip_presets`` are authored rather than measured; the direction is not in
    question, because a grip that opened the hand would be visibly wrong.
    """

    kinematics = rig_kinematics()
    profile = rig_profile()
    curl = _unit(
        np.asarray(profile["grip_presets"]["crate_grip"]["curl_axis"], dtype=float)
    )
    bone = f"{side}IndexProximal"
    node = kinematics.node_by_canonical[bone]
    world_curl = kinematics.rest_world[node][:3, :3] @ curl
    return _unit(np.cross(world_curl, _tip_offset_world(kinematics, bone)))


@lru_cache(maxsize=2)
def radial_direction(side: str) -> np.ndarray:
    """World direction toward the thumb side of the hand, at rest.

    Derived from the knuckle line -- the rest positions of the index and little
    metacarpophalangeal pivots -- with the hand's longitudinal component removed.
    Landmark positions are independent of every rotation-axis convention in this
    module, which is what makes this usable as a reference rather than an
    assumption. It is also mirror-correct by construction: the left and right
    knuckle lines are reflections, so the two radial directions are too.
    """

    kinematics = rig_kinematics()
    positions = {
        name: kinematics.rest_world[kinematics.node_by_canonical[name]][:3, 3]
        for name in (f"{side}IndexProximal", f"{side}LittleProximal", f"{side}Hand")
    }
    radial = positions[f"{side}IndexProximal"] - positions[f"{side}LittleProximal"]
    longitudinal = _unit(_tip_offset_world(kinematics, f"{side}Hand"))
    return _unit(radial - longitudinal * float(np.dot(radial, longitudinal)))


def reference_direction(reference: Reference, side: str | None) -> np.ndarray:
    """Resolve an anatomical :class:`Reference` to a world unit vector."""

    axes = world_axes()
    if reference is Reference.SUPERIOR:
        return axes["superior"]
    if reference is Reference.INFERIOR:
        return -axes["superior"]
    if reference is Reference.ANTERIOR:
        return axes["anterior"]
    if reference is Reference.POSTERIOR:
        return -axes["anterior"]
    if reference is Reference.LEFTWARD:
        return axes["leftward"]
    if reference is Reference.LATERAL:
        if side is None:
            raise ValueError("a midline bone has no lateral direction")
        return axes["leftward"] if side == "left" else -axes["leftward"]
    if reference is Reference.MEDIAL:
        return -reference_direction(Reference.LATERAL, side)
    if reference is Reference.PALMAR:
        if side is None:
            raise ValueError("a midline bone has no palmar direction")
        return palmar_direction(side)
    if reference is Reference.RADIAL:
        if side is None:
            raise ValueError("a midline bone has no radial direction")
        return radial_direction(side)
    raise ValueError(f"unhandled reference {reference!r}")


def _tip_motion(rest_rotation: np.ndarray, axis: np.ndarray, lever: np.ndarray) -> np.ndarray:
    """Unit world direction the distal tip moves under +rotation about ``axis``.

    Exact for an infinitesimal rotation, which is all the sign of a convention
    needs: ``omega x r``.
    """

    return _unit(np.cross(rest_rotation @ axis, lever))


def _select_axis(
    rest_rotation: np.ndarray,
    lever: np.ndarray,
    candidates: tuple[np.ndarray, ...],
    target: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Pick the signed local axis whose tip motion best matches ``target``."""

    scored = [
        (float(np.dot(_tip_motion(rest_rotation, axis, lever), target)), axis)
        for axis in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best, axis = scored[0]
    # The runner-up is the best *other axis line*: an axis and its negation are
    # one choice, not two, and the sign is settled by ``best`` already. When the
    # caller offers only one line the choice is unambiguous by construction, so
    # the margin is the alignment itself.
    others = [
        abs(score)
        for score, other in scored
        if not np.allclose(other, axis) and not np.allclose(other, -axis)
    ]
    return axis, best, best - max(others, default=0.0)


@lru_cache(maxsize=128)
def bone_anatomical_frame(canonical: str) -> AnatomicalFrame:
    """Derive one bone's anatomical frame. Pure, cached, no new asset data."""

    kinematics = rig_kinematics()
    if canonical not in kinematics.node_by_canonical:
        raise KeyError(f"{canonical!r} is not a canonical bone of this rig")
    node = kinematics.node_by_canonical[canonical]
    rest_world = kinematics.rest_world[node][:3, :3]

    child_offset = np.asarray(kinematics.rest[_child_index(kinematics, canonical)].translation)
    longitudinal_cosine = float(np.dot(_unit(child_offset), np.asarray([0.0, 1.0, 0.0])))
    if longitudinal_cosine < MIN_LONGITUDINAL_COSINE:
        raise ValueError(
            f"{canonical}: local +Y is not the longitudinal axis "
            f"(cos={longitudinal_cosine:.4f} < {MIN_LONGITUDINAL_COSINE})"
        )

    side = bone_side(canonical)
    flexion_reference, abduction_reference, rationale = convention(canonical)
    lever = _tip_offset_world(kinematics, canonical)

    x_axis = np.asarray([1.0, 0.0, 0.0])
    z_axis = np.asarray([0.0, 0.0, 1.0])
    candidates = (x_axis, -x_axis, z_axis, -z_axis)

    flexion_axis, flexion_score, flexion_margin = _select_axis(
        rest_world, lever, candidates, reference_direction(flexion_reference, side)
    )
    if flexion_score < MIN_ALIGNMENT or flexion_margin < MIN_MARGIN:
        raise ValueError(
            f"{canonical}: no cross axis realises {flexion_reference.value} flexion "
            f"(best={flexion_score:.3f}, margin={flexion_margin:.3f}). The rest pose "
            "and the declared convention disagree; fix one before trusting a limit."
        )

    # Abduction is chosen from the two axes orthogonal to the flexion axis, so
    # the frame stays orthonormal by construction rather than by hope.
    orthogonal = tuple(
        axis for axis in candidates if abs(float(np.dot(axis, flexion_axis))) < 0.5
    )
    abduction_axis, abduction_score, abduction_margin = _select_axis(
        rest_world, lever, orthogonal, reference_direction(abduction_reference, side)
    )
    if abduction_score < MIN_ALIGNMENT or abduction_margin < MIN_MARGIN:
        raise ValueError(
            f"{canonical}: no cross axis realises {abduction_reference.value} abduction "
            f"(best={abduction_score:.3f}, margin={abduction_margin:.3f})"
        )

    twist_axis = np.asarray([0.0, 1.0, 0.0])
    frame_matrix = np.column_stack((flexion_axis, abduction_axis, twist_axis))
    if abs(abs(float(np.linalg.det(frame_matrix))) - 1.0) > 1e-9:
        raise ValueError(f"{canonical}: derived frame is not orthonormal")

    return AnatomicalFrame(
        bone=canonical,
        twist_axis=twist_axis,
        flexion_axis=flexion_axis,
        abduction_axis=abduction_axis,
        joint_class=joint_class(canonical),
        rationale=rationale,
        flexion_world=_tip_motion(rest_world, flexion_axis, lever),
        abduction_world=_tip_motion(rest_world, abduction_axis, lever),
        longitudinal_cosine=longitudinal_cosine,
        flexion_margin=flexion_margin,
        abduction_margin=abduction_margin,
    )


def swing_twist(
    quaternion_xyzw: np.ndarray, axis: np.ndarray
) -> tuple[Rotation, float]:
    """Split a rotation into its swing and its signed twist about ``axis``.

    The generalisation of ``analysis.gesture.swing_twist_angles``, which
    collapses the swing to a magnitude. Here the swing rotation itself survives,
    which is what makes a two-axis projection possible at all.
    """

    value = np.asarray(quaternion_xyzw, dtype=float)
    value = value / max(float(np.linalg.norm(value)), 1e-12)
    axis = _unit(np.asarray(axis, dtype=float))
    projected = axis * float(np.dot(value[:3], axis))
    twist_value = np.asarray([*projected, value[3]], dtype=float)
    twist_norm = float(np.linalg.norm(twist_value))
    twist = (
        Rotation.identity() if twist_norm < 1e-12 else Rotation.from_quat(twist_value / twist_norm)
    )
    swing = Rotation.from_quat(value) * twist.inv()
    twist_angle = float(twist.magnitude())
    if float(np.dot(twist.as_rotvec(), axis)) < 0.0:
        twist_angle = -twist_angle
    return swing, twist_angle


def decompose(quaternion_xyzw, frame: AnatomicalFrame) -> DofAngles:
    """Resolve a local delta rotation onto one bone's anatomical frame.

    Twist is taken about the longitudinal axis exactly as the existing
    swing-twist decomposition does. The remaining swing is a rotation whose axis
    is perpendicular to the longitudinal axis by construction, so projecting its
    rotation vector onto the flexion and abduction axes is exact rather than an
    approximation.

    The swing rotation vector is bounded by pi, so a swing of exactly 180
    degrees is a sign singularity: the magnitude is right and the sign is
    arbitrary. Callers comparing against a limit should use the magnitude there.
    """

    swing, twist_angle = swing_twist(np.asarray(quaternion_xyzw, dtype=float), frame.twist_axis)
    rotation_vector = swing.as_rotvec()
    return DofAngles(
        flexion_rad=float(np.dot(rotation_vector, frame.flexion_axis)),
        abduction_rad=float(np.dot(rotation_vector, frame.abduction_axis)),
        twist_rad=twist_angle,
    )


def decompose_series(
    quaternions_xyzw: np.ndarray, frame: AnatomicalFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`decompose` over a whole clip at once, as three ``(n,)`` arrays.

    Identical maths, vectorised. The scalar form builds four ``Rotation``
    objects per call, which is fine for a test and not for a check: measured at
    **2.9 ms per frame** over all 52 bones, a 589-frame clip cost 1.75 s, well
    past plan 02 §5's 300 ms ceiling for the whole analysis layer. Batching per
    bone turns 52 x n scalar constructions into 52 vectorised ones.
    """

    values = np.asarray(quaternions_xyzw, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("expected an (n, 4) array of xyzw quaternions")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-12)

    axis = frame.twist_axis
    projected = np.outer(values[:, :3] @ axis, axis)
    twist_quat = np.concatenate([projected, values[:, 3:4]], axis=1)
    twist_norm = np.linalg.norm(twist_quat, axis=1, keepdims=True)
    degenerate = twist_norm[:, 0] < 1e-12
    safe = np.where(degenerate[:, None], np.asarray([0.0, 0.0, 0.0, 1.0]), twist_quat)
    safe = safe / np.maximum(np.linalg.norm(safe, axis=1, keepdims=True), 1e-12)

    twist = Rotation.from_quat(safe)
    swing_rotvec = (Rotation.from_quat(values) * twist.inv()).as_rotvec()
    twist_rotvec = twist.as_rotvec()

    twist_angle = np.linalg.norm(twist_rotvec, axis=1)
    twist_angle = np.where(twist_rotvec @ axis < 0.0, -twist_angle, twist_angle)

    return (
        swing_rotvec @ frame.flexion_axis,
        swing_rotvec @ frame.abduction_axis,
        twist_angle,
    )


def compose(angles: DofAngles, frame: AnatomicalFrame) -> np.ndarray:
    """Inverse of :func:`decompose`, as an xyzw quaternion.

    Exists so tests can pose the rig at a stated anatomical angle rather than at
    a quaternion someone believed corresponded to one.
    """

    swing = Rotation.from_rotvec(
        angles.flexion_rad * frame.flexion_axis + angles.abduction_rad * frame.abduction_axis
    )
    twist = Rotation.from_rotvec(angles.twist_rad * frame.twist_axis)
    return (swing * twist).as_quat()


@lru_cache(maxsize=1)
def all_frames() -> dict[str, AnatomicalFrame]:
    """Every canonical bone's frame. Deriving all 52 is the coverage test."""

    return {name: bone_anatomical_frame(name) for name in canonical_bone_names()}


def digit_bones(side: str) -> tuple[str, ...]:
    """Every finger and thumb bone on one side, in profile order."""

    return tuple(
        name
        for name in canonical_bone_names()
        if bone_side(name) == side and any(name[len(side) :].startswith(d) for d in DIGITS)
    )


__all__ = [
    "MIN_ALIGNMENT",
    "MIN_LONGITUDINAL_COSINE",
    "MIN_MARGIN",
    "AnatomicalFrame",
    "DofAngles",
    "all_frames",
    "bone_anatomical_frame",
    "compose",
    "decompose",
    "decompose_series",
    "digit_bones",
    "palmar_direction",
    "radial_direction",
    "reference_direction",
    "swing_twist",
    "world_axes",
]
