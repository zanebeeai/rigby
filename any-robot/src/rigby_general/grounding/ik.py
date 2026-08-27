"""Damped least-squares inverse kinematics at schema waypoints.

Why the grounder solves its own IK rather than handing Cartesian tracks to the v2
compiler: that compiler builds a site objective at *every* sampled frame and then
solves one enormous least-squares problem over all of them at once. It is exact,
and for a single certified humanoid clip it is worth the cost -- but it scales
badly, and baking a library means compiling hundreds of primitives. Measured on
one small arm: 22 s at 30 Hz, 70 s at 60 Hz, 291 s at 120 Hz, for one two-segment
motion. A five-hundred-binding bake at those rates takes days.

Solving instead at the handful of points where the path actually turns, and
handing the compiler a joint-space track it need only interpolate, moves the work
to where the meaning is. It also follows Tversky and Lee directly: the semantics
of a route live at its reorientations, and the straight stretches between them
carry no information worth spending a solver on.

Waypoints are densified along the intended contour before solving, so joint-space
interpolation still traces the Cartesian shape the schema asked for. Without that,
a straight path and an arced one would arrive at the same joint endpoints and
become indistinguishable -- which would quietly erase half the Path vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


DEFAULT_TOLERANCE_M = 0.004
MAX_ITERATIONS = 140
DAMPING = 0.06
MAX_STEP_RAD = 0.25
NULL_SPACE_GAIN = 0.25


@dataclass(frozen=True, slots=True)
class IkSolution:
    """Joint values per waypoint, plus how well each one was actually hit."""

    qpos: np.ndarray
    residuals_m: tuple[float, ...]
    joints: tuple[str, ...]

    @property
    def worst_residual_m(self) -> float:
        return max(self.residuals_m) if self.residuals_m else 0.0


class IkFailure(RuntimeError):
    def __init__(self, message: str, *, index: int, residual_m: float) -> None:
        super().__init__(message)
        self.index = index
        self.residual_m = residual_m


def solve_site_path(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: tuple[str, ...],
    targets: np.ndarray,
    *,
    seed_qpos: np.ndarray,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> IkSolution:
    """Walk a site through a sequence of world targets.

    Each solve warm-starts from the previous one. That is not only faster: it is
    what keeps the arm from flipping to a different elbow configuration halfway
    along a path and tearing the trajectory in two.
    """

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise KeyError(f"site {site_name!r} is not in the model")

    joint_ids = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"joint {name!r} is not in the model")
        joint_ids.append(joint_id)
    if not joint_ids:
        raise ValueError("inverse kinematics needs at least one joint")

    qpos_adr = np.array([int(model.jnt_qposadr[j]) for j in joint_ids])
    dof_adr = np.array([int(model.jnt_dofadr[j]) for j in joint_ids])
    lower = np.array(
        [
            float(model.jnt_range[j][0]) if model.jnt_limited[j] else -np.inf
            for j in joint_ids
        ]
    )
    upper = np.array(
        [
            float(model.jnt_range[j][1]) if model.jnt_limited[j] else np.inf
            for j in joint_ids
        ]
    )

    data = mujoco.MjData(model)
    current = np.array(seed_qpos, dtype=float)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    solutions = np.zeros((len(targets), model.nq), dtype=float)
    residuals: list[float] = []

    reference = np.array(seed_qpos, dtype=float)
    for index, target in enumerate(np.asarray(targets, dtype=float)):
        residual = np.inf
        for _ in range(MAX_ITERATIONS):
            data.qpos[:] = current
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            error = target - np.array(data.site_xpos[site_id], dtype=float)
            residual = float(np.linalg.norm(error))
            if residual <= tolerance_m:
                break

            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jacobian = jacp[:, dof_adr]

            # Damped least squares. The damping is what keeps the step finite
            # near a singularity, where an undamped pseudo-inverse would demand
            # an enormous joint velocity to produce a tiny Cartesian one.
            gram = jacobian @ jacobian.T + (DAMPING**2) * np.eye(3)
            pseudo_inverse = jacobian.T @ np.linalg.inv(gram)
            step = pseudo_inverse @ error

            # Null-space term, pulling toward the previous waypoint's solution.
            #
            # Without it the solver is free to wander anywhere in the null space
            # while still hitting its target, and on a redundant arm it does: two
            # neighbouring waypoints a centimetre apart come back as
            # configurations most of a radian apart, and the trajectory between
            # them is a wrist flip the arm cannot physically perform. Measured on
            # a five-axis arm, wrist joints drifted 0.6 to 0.8 rad from plan and
            # peaked at 3.5 rad/s against a 2.4 rad/s limit. Preferring the
            # nearest solution costs nothing and removes the whole failure mode.
            # The projector is built from a lightly-damped pseudo-inverse, not
            # from the heavily-damped one used for the primary step. Damping is
            # what makes the primary step safe near a singularity, but it also
            # makes ``I - Jdag J`` a poor null-space projector -- enough of the
            # secondary objective leaks back into the task that the solver stops
            # converging and reports a target it can plainly reach as
            # unreachable.
            projector = np.eye(len(joint_ids)) - np.linalg.pinv(
                jacobian, rcond=1e-3
            ) @ jacobian
            step = step + projector @ (
                NULL_SPACE_GAIN * (reference[qpos_adr] - current[qpos_adr])
            )
            step = np.clip(step, -MAX_STEP_RAD, MAX_STEP_RAD)

            proposed = np.clip(current[qpos_adr] + step, lower, upper)
            if float(np.max(np.abs(proposed - current[qpos_adr]))) < 1e-10:
                break  # Pinned against limits; no further progress is possible.
            current = current.copy()
            current[qpos_adr] = proposed

        if residual > tolerance_m:
            raise IkFailure(
                f"waypoint {index} is {residual:.4f} m from where it was asked to "
                f"be, past a tolerance of {tolerance_m:.4f} m",
                index=index,
                residual_m=residual,
            )
        solutions[index] = current
        residuals.append(residual)
        reference = current.copy()

    return IkSolution(
        qpos=solutions, residuals_m=tuple(residuals), joints=tuple(joint_names)
    )


def densify(points: list[np.ndarray], *, per_span: int) -> list[np.ndarray]:
    """Subdivide a waypoint list so joint interpolation follows the shape.

    Straight-line subdivision is right here even for an arced contour: the arc
    was already expressed as an apex waypoint, so the polyline through it is the
    arc. What subdivision buys is that the joint-space interpolation between
    consecutive solutions stays close to that polyline.
    """

    if per_span <= 1 or len(points) < 2:
        return list(points)
    dense: list[np.ndarray] = [points[0]]
    for start, end in zip(points, points[1:]):
        for step in range(1, per_span + 1):
            dense.append(start + (end - start) * (step / per_span))
    return dense


def chain_joint_names(
    model: mujoco.MjModel, site_name: str, *, exclude: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    """Actuated joints between a site and the world, proximal first.

    ``exclude`` keeps closure joints out of a reach. Without it the solver is free
    to open and close the gripper to help hit a target, and the jaw flaps its way
    across the workspace.
    """

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise KeyError(f"site {site_name!r} is not in the model")

    names: list[str] = []
    body = int(model.site_bodyid[site_id])
    while body > 0:
        start = int(model.body_jntadr[body])
        count = int(model.body_jntnum[body])
        for joint in range(start, start + count):
            if model.jnt_type[joint] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if name and name not in exclude:
                names.append(name)
        body = int(model.body_parentid[body])
    names.reverse()
    return tuple(names)
