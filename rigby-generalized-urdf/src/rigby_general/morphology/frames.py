"""Derive the robot's own up, front, and lateral axes.

``up`` is free: it is the opposite of gravity, and every robot agrees.

``front`` is the hard one, and it is worth being honest about why. A bare arm
bolted to a table has no intrinsic front. Nothing in its geometry distinguishes
one horizontal direction from another, so no amount of analysis can *derive* the
answer with certainty -- it can only find evidence. Two sources actually carry
signal:

*Workspace asymmetry.* Joint limits rarely permit a full turn, so the reachable
set has a gap. The direction the reachable volume leans is the direction the
robot was built to work in. The stronger the lean, the more the measurement is
worth.

*A mirror plane.* On a two-armed robot, the arms straddle a plane. Its normal is
the lateral axis, which fixes front up to a sign -- and the sign follows from the
workspace lean.

Both are proposals. The result is stamped ``FrameSource.DERIVED`` with a
confidence, and a person confirms it once at upload. Deixis (``HITHER`` and
``THITHER``) and every ``INTRINSIC``-frame schema depend on front being right, so
the grounder refuses to use them until someone has said so. Guessing silently and
then rendering a confidently backwards motion is the worse failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..contracts import DirectionV1, FrameSource, IntrinsicFrameV1, SymmetryV1


# Below this, the reachable set is effectively symmetric about the base and its
# lean says nothing about which way the robot faces.
_MIN_LEAN_RATIO = 0.04
_MIRROR_RESIDUAL_CEILING_FRACTION = 0.12


@dataclass(frozen=True, slots=True)
class FrameProposal:
    front: np.ndarray
    confidence: float
    evidence: tuple[str, ...]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _project_horizontal(vector: np.ndarray, up: np.ndarray) -> np.ndarray:
    return vector - up * float(vector @ up)


def gravity_up(gravity: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(gravity))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return -gravity / norm


def propose_from_workspace(
    tip_samples: np.ndarray, base_position: np.ndarray, up: np.ndarray
) -> FrameProposal | None:
    """Front from the direction the reachable set leans."""

    if tip_samples.size == 0:
        return None
    offsets = tip_samples - base_position
    horizontal = offsets - np.outer(offsets @ up, up)
    radii = np.linalg.norm(horizontal, axis=1)
    mean_radius = float(radii.mean())
    if mean_radius < 1e-9:
        return None

    centroid = horizontal.mean(axis=0)
    lean = float(np.linalg.norm(centroid))
    ratio = lean / mean_radius
    if ratio < _MIN_LEAN_RATIO:
        return None

    # Saturates around a quarter of the mean radius: beyond that the direction is
    # already unambiguous and more lean adds no information.
    confidence = float(min(0.85, ratio / 0.25 * 0.85))
    return FrameProposal(
        front=_unit(centroid),
        confidence=confidence,
        evidence=("workspace_lean",),
    )


def propose_from_mirror_plane(
    plane_normal: np.ndarray | None, up: np.ndarray
) -> FrameProposal | None:
    """Front from the plane two arms straddle.

    The mirror normal *is* the lateral axis, which pins front to a line -- but
    only to a line. Nothing about a symmetry plane says which of its two
    perpendiculars is forward, so the confidence stops at 0.5 and the caller
    resolves the sign from the workspace lean if there is one.
    """

    if plane_normal is None:
        return None
    lateral = _project_horizontal(np.asarray(plane_normal, dtype=float), up)
    if float(np.linalg.norm(lateral)) < 1e-6:
        return None
    lateral = _unit(lateral)
    return FrameProposal(
        front=_unit(np.cross(lateral, up)),
        confidence=0.5,
        evidence=("mirror_plane",),
    )


def build_intrinsic_frame(
    *,
    gravity: np.ndarray,
    tip_samples: np.ndarray,
    base_position: np.ndarray,
    mirror_normal: np.ndarray | None = None,
) -> IntrinsicFrameV1:
    up = gravity_up(np.asarray(gravity, dtype=float))

    workspace = propose_from_workspace(tip_samples, base_position, up)
    mirror = propose_from_mirror_plane(mirror_normal, up)

    evidence: list[str] = []
    if workspace is not None and mirror is not None:
        # The mirror fixes the axis; the lean fixes which end of it is forward.
        sign = 1.0 if float(mirror.front @ workspace.front) >= 0.0 else -1.0
        front = _unit(mirror.front * sign)
        agreement = abs(float(mirror.front @ workspace.front))
        confidence = min(0.9, 0.5 + 0.4 * agreement)
        evidence = ["mirror_plane", "workspace_lean"]
    elif workspace is not None:
        front = workspace.front
        confidence = workspace.confidence
        evidence = list(workspace.evidence)
    elif mirror is not None:
        front = mirror.front
        confidence = mirror.confidence
        evidence = list(mirror.evidence)
    else:
        # Nothing to go on. Pick a horizontal axis so the frame is well formed and
        # say plainly that it carries no information.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(float(fallback @ up)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        front = _unit(_project_horizontal(fallback, up))
        confidence = 0.0
        evidence = ["no_evidence_default_axis"]

    front = _unit(_project_horizontal(front, up))
    lateral = _unit(np.cross(up, front))
    front = _unit(np.cross(lateral, up))

    return IntrinsicFrameV1(
        up=DirectionV1(x=float(up[0]), y=float(up[1]), z=float(up[2])),
        front=DirectionV1(x=float(front[0]), y=float(front[1]), z=float(front[2])),
        lateral=DirectionV1(
            x=float(lateral[0]), y=float(lateral[1]), z=float(lateral[2])
        ),
        source=FrameSource.DERIVED,
        confidence=round(confidence, 4),
        evidence=tuple(evidence),
    )


def confirm_frame(
    frame: IntrinsicFrameV1, *, front: DirectionV1 | None = None
) -> IntrinsicFrameV1:
    """Record that a person accepted or corrected the proposed front direction."""

    up = np.array([frame.up.x, frame.up.y, frame.up.z], dtype=float)
    chosen = (
        np.array([front.x, front.y, front.z], dtype=float)
        if front is not None
        else np.array([frame.front.x, frame.front.y, frame.front.z], dtype=float)
    )
    horizontal = _project_horizontal(chosen, up)
    if float(np.linalg.norm(horizontal)) < 1e-9:
        raise ValueError("A front direction cannot be parallel to up")
    forward = _unit(horizontal)
    lateral = _unit(np.cross(up, forward))

    return IntrinsicFrameV1(
        up=frame.up,
        front=DirectionV1(
            x=float(forward[0]), y=float(forward[1]), z=float(forward[2])
        ),
        lateral=DirectionV1(
            x=float(lateral[0]), y=float(lateral[1]), z=float(lateral[2])
        ),
        source=FrameSource.OPERATOR_CONFIRMED,
        confidence=1.0,
        evidence=(*frame.evidence, "operator_confirmed"),
    )


def measure_symmetry(
    *,
    left_chain_id: str,
    right_chain_id: str,
    left_positions: np.ndarray,
    right_positions: np.ndarray,
    up: np.ndarray,
    scale_m: float,
) -> SymmetryV1 | None:
    """Test whether two chains are genuine mirror images of one another.

    Reflects the left chain's body origins through the candidate plane and
    measures how far they land from the right chain's. Symmetry is evidence about
    handedness, never a requirement -- many real dual-arm rigs are not mirrored,
    and left and right can still be told apart by the lateral axis.
    """

    if left_positions.shape != right_positions.shape or left_positions.size == 0:
        return None

    midpoint = (left_positions.mean(axis=0) + right_positions.mean(axis=0)) / 2.0
    separation = _project_horizontal(
        right_positions.mean(axis=0) - left_positions.mean(axis=0), np.asarray(up)
    )
    norm = float(np.linalg.norm(separation))
    if norm < 1e-9:
        return None
    normal = separation / norm

    reflected = left_positions - 2.0 * np.outer(
        (left_positions - midpoint) @ normal, normal
    )
    residual = float(np.abs(np.linalg.norm(reflected - right_positions, axis=1)).max())
    if residual > _MIRROR_RESIDUAL_CEILING_FRACTION * max(scale_m, 1e-6):
        return None

    from ..contracts import Vec3

    return SymmetryV1(
        plane_normal=DirectionV1(
            x=float(normal[0]), y=float(normal[1]), z=float(normal[2])
        ),
        plane_point_m=Vec3(
            x=float(midpoint[0]), y=float(midpoint[1]), z=float(midpoint[2])
        ),
        left_chain_id=left_chain_id,
        right_chain_id=right_chain_id,
        residual_m=round(residual, 6),
    )


def horizontal_angle(vector: DirectionV1) -> float:
    """Bearing in the world XY plane, for reporting and tests."""

    return math.degrees(math.atan2(vector.y, vector.x))
