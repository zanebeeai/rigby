"""Bend-plane math shared by the two arm solvers.

Both arm solvers place the elbow geometrically on a plane chosen by a pole
vector or a hint, then author the upper arm as a shortest-arc aim.  A
shortest-arc rotation carries zero twist about its source direction, so the
humerus was never rotated about its own long axis and the whole orientation of
the bend plane landed in the LowerArm joint as out-of-hinge-plane rotation
(elbow abduction, measured up to 130.6 deg on the corpus).  The helpers here
compute the roll about the humerus long axis that presents the elbow hinge
inside the chosen bend plane, so the downstream shortest-arc forearm solve
decomposes to near-pure flexion.  Pure numpy/scipy on purpose: this module
sits below both :mod:`rigby_poc.primitives` and :mod:`rigby_poc.kinematics`
and must not import either.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

#: The elbow hinge (flexion) axis in the LowerArm's local rest frame.  Kept as
#: a constant rather than derived at authoring time so the authoring path
#: loads no rig asset; ``tests/test_arm_plane.py`` pins it against the derived
#: ``analysis.anatomy.frame.bone_anatomical_frame(...).flexion_axis`` for both
#: sides, which is the guard that this constant tracks the rig if the GLB is
#: ever re-exported.
ELBOW_FLEXION_AXIS_LOCAL = np.array([1.0, 0.0, 0.0])

#: The humeral long axis in the UpperArm's local rest frame: both solvers aim
#: the carried child pivot, which sits on local +Y.
UPPER_ARM_LONG_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])

#: The enforced shoulder-twist band, ``config/rom.v1.json``
#: ``leftUpperArm/rightUpperArm.twist.max_deg`` = (-95, 95).  A constant here
#: for the same reason as :data:`ELBOW_FLEXION_AXIS_LOCAL` — the authoring path
#: must not import the analysis layer — and pinned against
#: ``analysis.anatomy.rom.rom_limit`` (rest-offset conversion included) by
#: ``tests/test_arm_plane.py``.
UPPER_ARM_TWIST_BAND_RAD = math.radians(95.0)


def signed_angle_about_axis(source: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    """Return the signed projected angle from source to target about axis."""
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    source = source - axis * float(np.dot(source, axis))
    target = target - axis * float(np.dot(target, axis))
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < 1e-8 or target_norm < 1e-8:
        return 0.0
    source /= source_norm
    target /= target_norm
    sine = float(np.dot(axis, np.cross(source, target)))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return math.atan2(sine, cosine)


def bend_plane_normal(bend: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Unit normal of the two-link triangle's plane, oriented for +flexion.

    With elbow = shoulder + direction*along + bend*height, the identity
    ``cross(along*d + h*b, (dist-along)*d - h*b) = h*dist*cross(b, d)`` makes
    this equal to ``unit(cross(upper_dir, lower_dir))`` with a positive
    coefficient, while staying defined and continuous through a straight arm.
    Aligning the elbow hinge axis onto this normal therefore makes the
    decomposed elbow flexion the non-negative interior bend for every target.
    """
    normal = np.cross(bend, direction)
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def humeral_roll(
    hinge_world: np.ndarray, plane_normal: np.ndarray, long_axis_world: np.ndarray
) -> float:
    """Signed roll about the humerus long axis taking the hinge into the plane.

    ``hinge_world`` is the elbow flexion axis the un-rolled humerus presents;
    rolling the upper arm by this angle about ``long_axis_world`` aligns it
    with ``plane_normal``.  Because the aim maps the rest long axis exactly
    onto the shoulder->elbow direction and the child pivot lies on that axis,
    the roll moves neither the elbow nor the hand pivot.
    """
    return signed_angle_about_axis(hinge_world, plane_normal, long_axis_world)


def roll_rotation(long_axis_world: np.ndarray, roll_rad: float) -> Rotation:
    """The humeral roll as a Rotation about the world-space long axis."""
    return Rotation.from_rotvec(np.asarray(long_axis_world, dtype=float) * roll_rad)


def twist_about_local_y(rotation: Rotation) -> float:
    """Signed swing-twist twist of a local delta about the bone's local +Y.

    Exact for the swing-then-twist convention the analysis layer uses: the
    twist component of a unit quaternion about +Y is ``(0, y, 0, w)``, whose
    angle is ``2*atan2(y, w)``.  The sign flip keeps the answer in (-pi, pi]
    when scipy hands back the antipodal quaternion.
    """
    xyzw = rotation.as_quat()
    if float(xyzw[3]) < 0.0:
        xyzw = -xyzw
    return 2.0 * math.atan2(float(xyzw[1]), float(xyzw[3]))


def forearm_roll_compensation(
    lower_world: Rotation,
    lower_world_preroll: Rotation,
    lower_base_world: Rotation,
    budget_rad: float,
) -> tuple[Rotation, float, float]:
    """Restore the pre-roll hand orientation as forearm pronation.

    The rolled and pre-roll forearm solutions both map the forearm's local +Y
    onto the elbow->wrist direction, so they differ by a pure twist about that
    axis: restoring the pre-roll branch's world orientation is exactly forearm
    pronation, a declared DOF.  The pronation is budgeted so the *total*
    longitudinal twist in the LowerArm joint (its existing twist plus this
    compensation) stays inside ``budget_rad``; whatever the budget cannot
    absorb is left for the hand bone to carry.

    Returns the pronated forearm world rotation, the pronation applied, and
    the un-absorbed overflow (0.0 when the compensation fit entirely).
    """
    twist_to_preroll = lower_world.inv() * lower_world_preroll
    required = float(twist_to_preroll.as_rotvec()[1])
    pre_twist = twist_about_local_y(lower_base_world.inv() * lower_world)
    total = pre_twist + required
    clamped = float(np.clip(total, -budget_rad, budget_rad))
    pronation = clamped - pre_twist
    overflow = abs(total) - abs(clamped)
    return (
        lower_world * Rotation.from_rotvec([0.0, pronation, 0.0]),
        pronation,
        overflow,
    )


__all__ = [
    "ELBOW_FLEXION_AXIS_LOCAL",
    "UPPER_ARM_LONG_AXIS_LOCAL",
    "UPPER_ARM_TWIST_BAND_RAD",
    "bend_plane_normal",
    "forearm_roll_compensation",
    "humeral_roll",
    "roll_rotation",
    "signed_angle_about_axis",
    "twist_about_local_y",
]
