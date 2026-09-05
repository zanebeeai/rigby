"""Keep the hand's skin out of the bodies it must not pass through.

The arm solver places a wrist pivot. The thing a viewer sees is the skinned
hand around that pivot, whose palm surface sits about 38 mm palmar of the
knuckle plane and whose fingertips reach a further 100 mm. A wrist placed at
an object's centre buries the fingers in whatever the object rests on.

This module closes that gap at the keyframe level: solve the arm for a wrist
target, skin the hand, find the deepest vertex inside each forbidden body, and
move the target out along that vertex's shortest escape until every vertex
clears by a margin. The solve is repeated on the moved target, so the answer
is measured on the surface that gets rendered, never on a pivot.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .hand_mesh import box_escape, hand_mesh
from .kinematics import rig_kinematics
from .models import BonePose, Hand, PrimitiveParameters, Quat, SceneObject, Transform, Vec3
from .primitives import arm_pose_from_target, shoulder_position
from .thresholds import threshold

#: Bounded fixed-point: each pass removes the deepest overlap it can see, and a
#: pass that moves the target may uncover a shallower one behind it.
MAX_PASSES = 12


@dataclass(frozen=True)
class ForbiddenBody:
    """An oriented box the hand must stay out of."""

    name: str
    centre: np.ndarray
    half_extents: np.ndarray
    rotation: np.ndarray
    #: World direction the hand leaves this body along, or ``None`` for the
    #: nearest face. A support surface is left through its top.
    escape_along: np.ndarray | None = None
    #: Clearance to keep from this body, or ``None`` for the configured
    #: default. Zero for an object the fingers are meant to be touching.
    margin_m: float | None = None

    @classmethod
    def from_scene_object(
        cls,
        item: SceneObject,
        *,
        transform: Transform | None = None,
        escape_along: Vec3 | None = None,
        margin_m: float | None = None,
    ) -> "ForbiddenBody":
        """The box of ``item``, where ``transform`` (default: as authored) puts it."""

        placed = item.transform if transform is None else transform
        rotation = Rotation.from_quat(placed.rotation.as_list()).as_matrix()
        along = (
            None
            if escape_along is None
            else rotation @ np.asarray(escape_along.as_list(), dtype=float)
        )
        return cls(
            name=item.id,
            centre=np.asarray(placed.translation.as_list(), dtype=float),
            half_extents=np.asarray(item.dimensions_m.as_list(), dtype=float) * 0.5,
            rotation=rotation,
            escape_along=along,
            margin_m=margin_m,
        )


@dataclass(frozen=True)
class Clearance:
    """One keyframe's cleared wrist target and the arm that reaches it."""

    target: Vec3
    arm: dict[str, Quat]
    #: Metres the target moved from where the phase authored it.
    displacement_m: float
    #: Deepest remaining overlap per body after the last pass; all ≤ margin
    #: when ``converged``.
    residual_m: dict[str, float]
    passes: int
    converged: bool
    #: World escape direction used per body that was entered, so a caller
    #: stepping along a path can hold it for the following frames.
    escapes: dict[str, np.ndarray]


def hand_clearance_m() -> float:
    return float(threshold("contact.hand_clearance_m").value)


def clear_wrist_target(
    hand: Hand,
    target: Vec3,
    solve_arm: Callable[[Vec3], dict[str, Quat]],
    body_pose: Mapping[str, Quat],
    fingers: Mapping[str, Quat],
    bodies: list[ForbiddenBody],
    *,
    margin_m: float | None = None,
) -> Clearance:
    """Move ``target`` until the skinned hand clears every body by ``margin_m``.

    ``solve_arm`` is the caller's own arm solve for a wrist target, so the
    clearance is measured with exactly the pose the caller will render.
    ``body_pose`` is everything but the arm and fingers.
    """

    margin = hand_clearance_m() if margin_m is None else float(margin_m)
    mesh = hand_mesh(hand.value)
    origin = np.asarray(target.as_list(), dtype=float)
    current = origin.copy()
    arm = solve_arm(target)
    residual: dict[str, float] = {body.name: 0.0 for body in bodies}
    escapes: dict[str, np.ndarray] = {}
    converged = not bodies
    passes = 0
    for passes in range(1, MAX_PASSES + 1):
        pose = dict(body_pose)
        pose.update(arm)
        pose.update(fingers)
        points = mesh.world({name: BonePose(rotation=rotation) for name, rotation in pose.items()})
        worst_depth, worst_step = 0.0, None
        for body in bodies:
            depth, direction = box_escape(
                points, body.centre, body.rotation, body.half_extents, body.escape_along
            )
            deepest = int(np.argmax(depth))
            residual[body.name] = float(depth[deepest])
            if depth[deepest] > 0.0 and body.name not in escapes:
                escapes[body.name] = np.asarray(direction[deepest], dtype=float)
            if depth[deepest] > worst_depth:
                worst_depth = float(depth[deepest])
                body_margin = margin if body.margin_m is None else body.margin_m
                worst_step = direction[deepest] * (depth[deepest] + body_margin)
        if worst_step is None or worst_depth <= 0.0:
            converged = True
            break
        current = current + worst_step
        arm = solve_arm(Vec3(x=float(current[0]), y=float(current[1]), z=float(current[2])))
    else:
        converged = all(value <= 0.0 for value in residual.values())
    cleared = Vec3(x=float(current[0]), y=float(current[1]), z=float(current[2]))
    return Clearance(
        target=cleared,
        arm=arm,
        displacement_m=float(np.linalg.norm(current - origin)),
        residual_m=residual,
        passes=passes,
        converged=converged,
        escapes=escapes,
    )


def hand_clearance_blend_m() -> float:
    return float(threshold("contact.hand_clearance_blend_m").value)


def _nlerp(first: Quat, second: Quat, alpha: float) -> Quat:
    a = np.asarray(first.as_list(), dtype=float)
    b = np.asarray(second.as_list(), dtype=float)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    value = a * (1.0 - alpha) + b * alpha
    value /= max(float(np.linalg.norm(value)), 1e-12)
    return Quat(x=float(value[0]), y=float(value[1]), z=float(value[2]), w=float(value[3]))


def correct_frame_arm(
    hand: Hand,
    frame_pose: Mapping[str, Quat],
    wrist: Vec3,
    solve_arm: Callable[[Vec3], dict[str, Quat]],
    bodies: list[ForbiddenBody],
) -> dict[str, Quat] | None:
    """Lift an already-interpolated frame's arm out of the bodies it entered.

    Returns replacement arm rotations, or ``None`` when the frame is clear.
    The re-solved arm is mixed with the interpolated one in proportion to
    the depth found, saturating at :func:`hand_clearance_blend_m`, so the
    correction is continuous in time: a frame that is barely inside is
    barely changed.
    """

    if not bodies:
        return None
    mesh = hand_mesh(hand.value)
    points = mesh.world({name: BonePose(rotation=rotation) for name, rotation in frame_pose.items()})
    worst = 0.0
    for body in bodies:
        depth, _ = box_escape(points, body.centre, body.rotation, body.half_extents, body.escape_along)
        worst = max(worst, float(depth.max()))
    if worst <= 0.0:
        return None
    cleared = clear_wrist_target(hand, wrist, solve_arm, frame_pose, {}, bodies)
    weight = min(1.0, worst / hand_clearance_blend_m())
    return {
        name: _nlerp(frame_pose[name], rotation, weight) if name in frame_pose else rotation
        for name, rotation in cleared.arm.items()
    }


def bone_world_position(pose: Mapping[str, Quat], canonical: str) -> Vec3:
    """Where one bone's origin is, for a rest-relative pose, by the rig's own FK."""

    kinematics = rig_kinematics()
    matrices = kinematics.world_matrices(
        {name: BonePose(rotation=rotation) for name, rotation in pose.items()}
    )
    origin = matrices[kinematics.node_by_canonical[canonical]][:3, 3]
    return Vec3(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))


#: Distance of the hinted elbow from the shoulder-wrist line below which the
#: hint stops steering the bend plane. On a straight arm the true elbow lies on
#: that line and its offset is noise; steering by noise flips the plane.
ELBOW_HINT_FADE_M = 0.03


def hinted_arm_solve(
    hand: Hand,
    point: Vec3,
    parameters: PrimitiveParameters,
    hint: Vec3 | None,
    pole: Vec3 | None = None,
) -> dict[str, Quat]:
    """One frame's arm along a path: primary branch, elbow steered by last frame's.

    ``pole`` is the bend plane to fall back on as the hint fades near a
    straight arm -- the plane the phase started in, so the fade lands where
    the arm already is rather than on the solver's default side.
    """

    weight = 1.0
    if hint is not None:
        shoulder = np.asarray(shoulder_position(hand).as_list(), dtype=float)
        axis = np.asarray(point.as_list(), dtype=float) - shoulder
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        offset = np.asarray(hint.as_list(), dtype=float) - shoulder
        offset -= axis * float(np.dot(offset, axis))
        weight = min(1.0, float(np.linalg.norm(offset)) / ELBOW_HINT_FADE_M)
    return arm_pose_from_target(
        hand,
        shoulder_position(hand),
        point,
        parameters,
        present_hand=False,
        elbow_hint=hint,
        elbow_hint_weight=weight,
        elbow_pole=pole,
        allow_mirror=False,
    )[0]


def pushing_front_m(
    skin: np.ndarray,
    object_position: np.ndarray,
    rotation_matrix: np.ndarray,
    half_extents: np.ndarray,
    direction: np.ndarray,
) -> float | None:
    """How far along ``direction`` the hand's block-facing skin reaches.

    Only skin within the block's own height and width counts: fingers pointing
    forward over its top are not touching it. ``None`` when nothing is.
    """

    vertical_extent = float(np.dot(np.abs(rotation_matrix[1]), half_extents))
    lateral = np.cross(np.asarray([0.0, 1.0, 0.0]), direction)
    lateral_extent = float(np.dot(np.abs(lateral @ rotation_matrix), half_extents))
    offsets = skin - object_position
    in_height = np.abs(offsets[:, 1]) <= vertical_extent
    in_width = np.abs(offsets @ lateral) <= lateral_extent
    pushing = skin[in_height & in_width]
    if not len(pushing):
        return None
    return float((pushing @ direction).max())


def basis_with_first_axis(axis: np.ndarray) -> np.ndarray:
    """An orthonormal frame (columns) whose first axis is ``axis``."""

    first = np.asarray(axis, dtype=float)
    first /= max(float(np.linalg.norm(first)), 1e-8)
    helper = np.asarray([0.0, 1.0, 0.0]) if abs(first[1]) < 0.9 else np.asarray([1.0, 0.0, 0.0])
    second = np.cross(helper, first)
    second /= max(float(np.linalg.norm(second)), 1e-8)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def bend_pole(hand: Hand, pose: Mapping[str, Quat], parameters: PrimitiveParameters) -> Vec3:
    """The bend plane a pose's arm is in, as a pole the solver would reproduce."""

    shoulder = np.asarray(shoulder_position(hand).as_list(), dtype=float)
    wrist = np.asarray(bone_world_position(pose, f"{hand.value}Hand").as_list(), dtype=float)
    elbow = np.asarray(bone_world_position(pose, f"{hand.value}LowerArm").as_list(), dtype=float)
    axis = wrist - shoulder
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    bend = elbow - shoulder
    bend -= axis * float(np.dot(bend, axis))
    if np.linalg.norm(bend) < 1e-6:
        bend = np.cross(axis, np.asarray([0.0, 1.0, 0.0]))
    bend /= max(float(np.linalg.norm(bend)), 1e-8)
    # The solver swivels its pole about the reach axis; hand it the pre-swivel plane.
    bend = Rotation.from_rotvec(-axis * parameters.elbow_swivel * 0.75).apply(bend)
    # On a straight arm the measured plane is noise, but the humerus roll the
    # keyframe chose is not: of the measured plane, the solver's default and
    # its mirror, keep whichever reproduces the pose's upper arm.
    side = 1.0 if hand == Hand.LEFT else -1.0
    default = np.asarray([side, -0.55, 0.25], dtype=float)
    wrist_target = Vec3(x=float(wrist[0]), y=float(wrist[1]), z=float(wrist[2]))
    actual = np.asarray(pose[f"{hand.value}UpperArm"].as_list(), dtype=float)
    best, best_error = bend, float("inf")
    for candidate in (bend, default, -default):
        solved = arm_pose_from_target(
            hand,
            shoulder_position(hand),
            wrist_target,
            parameters,
            present_hand=False,
            elbow_pole=Vec3(x=float(candidate[0]), y=float(candidate[1]), z=float(candidate[2])),
            allow_mirror=False,
        )[0]
        error = 1.0 - abs(float(np.dot(actual, np.asarray(solved[f"{hand.value}UpperArm"].as_list()))))
        if error < best_error:
            best, best_error = candidate, error
    return Vec3(x=float(best[0]), y=float(best[1]), z=float(best[2]))


__all__ = [
    "ELBOW_HINT_FADE_M",
    "Clearance",
    "ForbiddenBody",
    "basis_with_first_axis",
    "bend_pole",
    "bone_world_position",
    "hinted_arm_solve",
    "pushing_front_m",
    "clear_wrist_target",
    "correct_frame_arm",
    "hand_clearance_blend_m",
    "hand_clearance_m",
]
