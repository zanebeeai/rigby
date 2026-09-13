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

A solution that hits its waypoint is not yet a solution the body can adopt. The
solver clips into joint limits, but limits are not the only thing a body cannot
pass through: its own links are. Handed a :class:`CollisionGuard`, the solver
keeps guarded link pairs apart while it converges and refuses, with a typed
failure, any waypoint or span it cannot keep clear. Measured on a two-arm rig,
the unguarded solver answered "come back from the far edge" by folding the wrist
164 degrees onto its own forearm: every key was inside its joint range, and the
palm was 28 mm inside the forearm at every one of them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np


DEFAULT_TOLERANCE_M = 0.004
MAX_ITERATIONS = 140
DAMPING = 0.02
"""Damped-least-squares damping.

Swept against the contact suites rather than chosen: 0.06 holds three of the
authored worlds, 0.02 holds four, and 0.035, 0.01 and 0.005 all come out behind.
Damping is what keeps the step finite near a singularity, and too much of it
turns a reachable waypoint into a residual the solver never closes -- which is
what `unreachable_object` was on arms that can plainly reach the block."""
MAX_STEP_RAD = 0.25
NULL_SPACE_GAIN = 0.25

SEPARATION_DAMPING = 1e-3
"""Damping of the separation task's least-squares solve.

Small on purpose: a guarded pair that is *penetrating* has to be pushed out by
the full measured depth, and heavy damping would leave it inside for many
iterations. The step is clipped with everything else, so a small damping cannot
make it explode."""
PATH_SAMPLES = 4
"""Interior joint-space samples checked for penetration between consecutive
solved waypoints. Waypoints are already densified along the contour, so the
spans are short; this catches a link swinging through another between two keys
that are individually clear."""
PENETRATION_TOLERANCE_M = 1e-6
"""Numerical slack on a zero-penetration requirement, not a permitted depth."""


@dataclass(frozen=True, slots=True)
class CollisionGuard:
    """Body pairs the solver keeps apart, and how far apart it prefers them.

    ``pairs`` are model body ids. Any penetration between a guarded pair is
    refused. A separation below ``clearance_m`` is opened as the solver's
    first-priority task, with the waypoint solved in the freedom that remains;
    once every pair is at least the clearance apart the waypoint has the
    solver to itself again.
    """

    pairs: tuple[tuple[int, int], ...]
    clearance_m: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.clearance_m) or self.clearance_m < 0.0:
            raise ValueError("clearance must be a finite, non-negative distance")
        for first, second in self.pairs:
            if first == second:
                raise ValueError("a body cannot be guarded against itself")


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
    def __init__(
        self,
        message: str,
        *,
        index: int,
        residual_m: float,
        collision: tuple[str, str] | None = None,
        penetration_m: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.index = index
        self.residual_m = residual_m
        self.collision = collision
        self.penetration_m = penetration_m


@dataclass(frozen=True, slots=True)
class Proximity:
    """One guarded geom pair closer than the clearance, at one configuration."""

    geom1: int
    geom2: int
    distance_m: float
    point: np.ndarray
    normal: np.ndarray
    """Unit direction from ``geom1`` towards ``geom2``; the direction along
    which their separation grows."""


class _GuardEvaluator:
    """The contact solver itself, asked to look ``clearance`` further than usual.

    A private copy of the model carries a contact margin on every guarded geom,
    so the engine reports each guarded pair that is inside the clearance --
    separated or penetrating -- with the signed distance, the point and the
    normal it would use in physics. That is the same pipeline the certification
    gate reads, so what the solver avoids and what the gate reports agree by
    construction. Closed-form distance queries were tried first and are not
    reliable for every geom pair the engine handles (two boxes, for one), and
    the margin is the engine's own way of asking the question.

    The copy's margins change only which contacts are *reported*; kinematics,
    Jacobians and site positions are identical to the caller's model, so the
    solver runs on the copy throughout.
    """

    def __init__(self, model: mujoco.MjModel, guard: CollisionGuard) -> None:
        self.model = copy.copy(model)
        self.guard = guard
        self.data = mujoco.MjData(self.model)
        collidable: dict[int, list[int]] = {}
        for geom in range(model.ngeom):
            if int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]):
                collidable.setdefault(int(model.geom_bodyid[geom]), []).append(geom)
        self.geom_pairs: set[tuple[int, int]] = set()
        for body_a, body_b in guard.pairs:
            for geom_a in collidable.get(body_a, ()):
                for geom_b in collidable.get(body_b, ()):
                    self.geom_pairs.add((min(geom_a, geom_b), max(geom_a, geom_b)))
        involved = {geom for pair in self.geom_pairs for geom in pair}
        # A pair's reporting margin is the sum of its two geoms' margins.
        for geom in involved:
            self.model.geom_margin[geom] = 0.5 * guard.clearance_m

    def observe(self, qpos: np.ndarray) -> list[Proximity]:
        """Run the position stage at ``qpos`` on the margin copy; afterwards
        ``self.data`` also holds the kinematics for Jacobians."""

        self.data.qpos[:] = qpos
        mujoco.mj_fwdPosition(self.model, self.data)
        found: list[Proximity] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            key = (min(geom1, geom2), max(geom1, geom2))
            if key not in self.geom_pairs:
                continue
            distance = float(contact.dist)
            if distance >= self.guard.clearance_m:
                continue
            normal = np.array(contact.frame[:3], dtype=float)
            if geom1 > geom2:
                normal = -normal
            found.append(
                Proximity(key[0], key[1], distance, np.array(contact.pos, dtype=float), normal)
            )
        return found


def _pair_names(model: mujoco.MjModel, item: Proximity) -> tuple[str, str]:
    return tuple(
        sorted(
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[item.geom1])) or "?",
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[item.geom2])) or "?",
            )
        )
    )


def _deepest(
    model: mujoco.MjModel, items: list[Proximity]
) -> tuple[tuple[str, str] | None, float]:
    depth = 0.0
    pair: tuple[str, str] | None = None
    for item in items:
        if -item.distance_m > depth:
            depth = -item.distance_m
            pair = _pair_names(model, item)
    return pair, depth


def penetrations(
    model: mujoco.MjModel, guard: CollisionGuard, qpos: np.ndarray
) -> tuple[tuple[str, str, float], ...]:
    """Guarded body pairs inside one another at ``qpos``: names and depth in
    metres, deepest first. Empty means the configuration is clear."""

    evaluator = _GuardEvaluator(model, guard)
    found: dict[tuple[str, str], float] = {}
    for item in evaluator.observe(np.asarray(qpos, dtype=float)):
        if item.distance_m >= -PENETRATION_TOLERANCE_M:
            continue
        names = _pair_names(model, item)
        found[names] = max(found.get(names, 0.0), -item.distance_m)
    return tuple(
        (first, second, depth)
        for (first, second), depth in sorted(found.items(), key=lambda entry: -entry[1])
    )


def _separation_task(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    items: list[Proximity],
    dof_adr: np.ndarray,
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """The linear task that opens every listed pair to the clearance at once.

    Each pair contributes one row: the rate at which its separation changes per
    unit joint motion, from the relative Jacobian of its contact point on the
    two bodies projected on the separating normal; the right-hand side is how
    far short of the clearance it is. ``None`` when no listed pair can be moved
    by these joints at all."""

    jac1 = np.zeros((3, model.nv), dtype=float)
    jac2 = np.zeros((3, model.nv), dtype=float)
    rows: list[np.ndarray] = []
    wanted: list[float] = []
    for item in items:
        body1 = int(model.geom_bodyid[item.geom1])
        body2 = int(model.geom_bodyid[item.geom2])
        mujoco.mj_jac(model, data, jac1, None, item.point, body1)
        mujoco.mj_jac(model, data, jac2, None, item.point, body2)
        row = item.normal @ (jac2 - jac1)[:, dof_adr]
        if float(np.linalg.norm(row)) < 1e-12:
            continue  # These joints cannot change this separation at all.
        rows.append(row)
        wanted.append(clearance_m - item.distance_m)
    if not rows:
        return None
    return np.asarray(rows, dtype=float), np.asarray(wanted, dtype=float)


def _damped_pseudo_inverse(matrix: np.ndarray, damping: float) -> np.ndarray:
    gram = matrix @ matrix.T + damping * np.eye(matrix.shape[0])
    return matrix.T @ np.linalg.inv(gram)


def solve_site_path(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: tuple[str, ...],
    targets: np.ndarray,
    *,
    seed_qpos: np.ndarray,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    guard: CollisionGuard | None = None,
) -> IkSolution:
    """Walk a site through a sequence of world targets.

    Each solve warm-starts from the previous one. That is not only faster: it is
    what keeps the arm from flipping to a different elbow configuration halfway
    along a path and tearing the trajectory in two.

    With a ``guard``, every solved waypoint and every span between consecutive
    waypoints is also required to be free of penetration between the guarded
    pairs; a failure to keep them clear is reported as an :class:`IkFailure`
    carrying the pair and the depth, distinct from an unreachable target.
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

    evaluator = _GuardEvaluator(model, guard) if guard is not None else None
    # With a guard the solve runs on the evaluator's margin copy: identical
    # kinematics, and the contacts it reports are the ones being avoided.
    kinematic_model = evaluator.model if evaluator is not None else model
    data = evaluator.data if evaluator is not None else mujoco.MjData(model)
    current = np.array(seed_qpos, dtype=float)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    solutions = np.zeros((len(targets), model.nq), dtype=float)
    residuals: list[float] = []

    def observe(qpos: np.ndarray) -> tuple[np.ndarray, float, list[Proximity]]:
        if evaluator is None:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(kinematic_model, data)
            mujoco.mj_comPos(kinematic_model, data)
            near: list[Proximity] = []
        else:
            near = evaluator.observe(qpos)
        error = target - np.array(data.site_xpos[site_id], dtype=float)
        return error, float(np.linalg.norm(error)), near

    def step_toward(step: np.ndarray) -> bool:
        """Apply a clipped step inside the limits; False when nothing moved."""

        nonlocal current
        step = np.clip(step, -MAX_STEP_RAD, MAX_STEP_RAD)
        proposed = np.clip(current[qpos_adr] + step, lower, upper)
        if float(np.max(np.abs(proposed - current[qpos_adr]))) < 1e-10:
            return False  # Pinned against limits; no further progress is possible.
        current = current.copy()
        current[qpos_adr] = proposed
        return True

    reference = np.array(seed_qpos, dtype=float)
    previous: np.ndarray | None = None
    for index, target in enumerate(np.asarray(targets, dtype=float)):
        residual = np.inf
        blocking: tuple[str, str] | None = None
        for _ in range(MAX_ITERATIONS):
            error, residual, near = observe(current)
            blocking = _pair_names(kinematic_model, near[0]) if near else None
            if residual <= tolerance_m and not near:
                break

            mujoco.mj_jacSite(kinematic_model, data, jacp, jacr, site_id)
            jacobian = jacp[:, dof_adr]
            toward_reference = NULL_SPACE_GAIN * (reference[qpos_adr] - current[qpos_adr])

            separation = (
                _separation_task(kinematic_model, data, near, dof_adr, guard.clearance_m)
                if near
                else None
            )
            if separation is not None:
                # Prioritised: keeping the guarded pairs apart is the first
                # task and the waypoint is solved in what freedom remains. An
                # additive push was tried first and stalled 13 mm short of the
                # waypoint on the two-arm rig -- the push and the target fought
                # over the same wrist joint every iteration. Ordering them
                # lets the target route round the obstacle through the other
                # joints instead, which is what a body actually does.
                rows, wanted = separation
                opening = _damped_pseudo_inverse(rows, SEPARATION_DAMPING) @ wanted
                remaining = np.eye(len(joint_ids)) - np.linalg.pinv(rows, rcond=1e-3) @ rows
                restricted = jacobian @ remaining
                step = opening + _damped_pseudo_inverse(restricted, DAMPING**2) @ (
                    error - jacobian @ opening
                )
                projector = remaining - np.linalg.pinv(restricted, rcond=1e-3) @ restricted
                if not step_toward(step + projector @ toward_reference):
                    break
                continue

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
            if not step_toward(step + projector @ toward_reference):
                break

        _, residual, near = observe(current)
        pair, penetration = _deepest(kinematic_model, near)
        if residual > tolerance_m:
            if blocking is not None:
                # The waypoint was within reach of the unguarded solver; what
                # stopped this one was a guarded pair standing between. That
                # is a collision refusal, not an unreachable target.
                raise IkFailure(
                    f"waypoint {index} is {residual:.4f} m from where it was asked "
                    f"to be, held off by {blocking[0]} against {blocking[1]}: no "
                    "configuration reaching it kept them apart",
                    index=index,
                    residual_m=residual,
                    collision=blocking,
                    penetration_m=max(0.0, penetration),
                )
            raise IkFailure(
                f"waypoint {index} is {residual:.4f} m from where it was asked to "
                f"be, past a tolerance of {tolerance_m:.4f} m",
                index=index,
                residual_m=residual,
            )
        if pair is not None and penetration > PENETRATION_TOLERANCE_M:
            raise IkFailure(
                f"waypoint {index} puts {pair[0]} {penetration * 1000:.1f} mm inside "
                f"{pair[1]}; no configuration reaching it kept them apart",
                index=index,
                residual_m=residual,
                collision=pair,
                penetration_m=penetration,
            )
        if evaluator is not None and previous is not None:
            for sample in range(1, PATH_SAMPLES + 1):
                fraction = sample / (PATH_SAMPLES + 1)
                between = previous + (current - previous) * fraction
                pair, penetration = _deepest(kinematic_model, evaluator.observe(between))
                if pair is not None and penetration > PENETRATION_TOLERANCE_M:
                    raise IkFailure(
                        f"the span into waypoint {index} passes {pair[0]} "
                        f"{penetration * 1000:.1f} mm through {pair[1]}, although "
                        "both of its ends are clear",
                        index=index,
                        residual_m=residual,
                        collision=pair,
                        penetration_m=penetration,
                    )
        solutions[index] = current
        residuals.append(residual)
        reference = current.copy()
        previous = current.copy()

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
