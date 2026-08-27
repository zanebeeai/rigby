"""Measurements taken by moving the robot and watching what happens.

Nothing here reads a name to decide anything. A gripper is a set of parts whose
surfaces approach one another when a joint is driven; a wrist joint is one that
turns the tip without carrying it anywhere; a redundant joint is one that adds no
rank to the tip Jacobian. Naming is at most a hint about *where to look*, and
where a hint is used it is recorded as such.

That matters more than it might seem. The link called ``gripper`` on one robot is
a fixed shroud, and the one called ``link_7`` on another is the moving jaw. Only
the behaviour is reliable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .graph import KinematicGraph


SWEEP_STEPS = 9
CLOSURE_DISTANCE_CEILING = 1.0
CLOSURE_MIN_TRAVEL_M = 5e-4
CLOSURE_MONOTONE_TOLERANCE = 1.02

# A joint counts as carrying the tip somewhere if it moves it at least this
# fraction of the robot's own reach. Relative, not absolute, so a 0.30 m desktop
# arm and a 1.75 m long-reach arm are judged on the same terms.
MAJOR_TRANSLATION_FRACTION = 0.12
ORIENT_MIN_ROTATION_RAD = 0.5


@dataclass(frozen=True, slots=True)
class JointMotion:
    """What driving one joint alone does to a chain tip."""

    joint: int
    translation_m: float
    rotation_rad: float


@dataclass(frozen=True, slots=True)
class ClosureEvidence:
    """The result of driving a cluster's interior joints through their range."""

    closes: bool
    open_distance_m: float
    closed_distance_m: float
    monotone: bool
    drive_to_upper: bool
    """True when the upper joint limit is the closed end, false when it is open."""

    member_directions: tuple[tuple[float, float, float], ...]
    opposition_groups: tuple[tuple[int, ...], ...]
    """Member indices, split into the two sides that face each other."""


REST_RELAXATION_STEPS = 50
"""Interpolation samples between the clamped zero pose and the range midpoint."""

REST_PENETRATION_TOLERANCE_M = -1e-4

REST_DESCENT_PASSES = 3
REST_DESCENT_SAMPLES = 9


def _clamped_qpos(model: mujoco.MjModel) -> np.ndarray:
    """The zero pose, pushed inside every joint limit."""

    qpos = np.array(model.qpos0, dtype=float)
    for joint in range(model.njnt):
        if model.jnt_type[joint] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        address = int(model.jnt_qposadr[joint])
        if model.jnt_limited[joint]:
            low, high = (float(value) for value in model.jnt_range[joint])
            qpos[address] = min(max(qpos[address], low), high)
    return qpos


def _midpoint_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Every hinge and slide halfway between its limits, the rest left alone.

    The midpoint is the configuration furthest from every limit at once, which
    makes it the natural direction to move when the zero pose is unusable.
    """

    qpos = _clamped_qpos(model)
    for joint in range(model.njnt):
        if model.jnt_type[joint] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        low, high = joint_range(model, joint)
        qpos[int(model.jnt_qposadr[joint])] = 0.5 * (low + high)
    return qpos


def _penetration_depth(model: mujoco.MjModel, qpos: np.ndarray) -> float:
    """How deeply two links that are not parent and child overlap, in metres.

    Zero means clear. Kept as a depth rather than a boolean so a pose can be
    improved rather than only judged.
    """

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    worst = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        distance = float(contact.dist)
        if distance > REST_PENETRATION_TOLERANCE_M:
            continue
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if first == second:
            continue
        if int(model.body_parentid[first]) == second:
            continue
        if int(model.body_parentid[second]) == first:
            continue
        worst = max(worst, -distance)
    return worst


def _movable_joints(model: mujoco.MjModel) -> tuple[int, ...]:
    return tuple(
        joint
        for joint in range(model.njnt)
        if model.jnt_type[joint]
        in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    )


def _descend(
    model: mujoco.MjModel, qpos: np.ndarray, depth: float
) -> tuple[np.ndarray, float]:
    """Coordinate descent on penetration depth, one joint at a time.

    Sweeping every joint together along one line is a single path through
    configuration space, and for a folded arm it can stay pinched the whole way.
    Moving one joint at a time is what actually unfolds it: the joint between the
    two overlapping links is usually the one that has to give, and scanning each
    joint's range in turn finds it without being told which one.

    Deterministic -- fixed joint order, fixed sample grid, no randomness -- so
    the rest pose is reproducible, which everything downstream assumes.
    """

    joints = _movable_joints(model)
    best, best_depth = qpos.copy(), depth
    for _ in range(REST_DESCENT_PASSES):
        improved = False
        for joint in joints:
            low, high = joint_range(model, joint)
            address = int(model.jnt_qposadr[joint])
            original = float(best[address])
            for step in range(REST_DESCENT_SAMPLES):
                value = low + (high - low) * step / (REST_DESCENT_SAMPLES - 1)
                if abs(value - original) < 1e-12:
                    continue
                trial = best.copy()
                trial[address] = value
                trial_depth = _penetration_depth(model, trial)
                if trial_depth < best_depth - 1e-9:
                    best, best_depth, improved = trial, trial_depth, True
                    if best_depth <= 0.0:
                        return best, best_depth
        if not improved:
            break
    return best, best_depth


def neutral_qpos(model: mujoco.MjModel) -> np.ndarray:
    """A rest pose that is inside every joint limit and not self-intersecting.

    Clamping zero into the joint limits is the obvious rest pose and it is wrong
    for a real arm. The Franka Panda's fourth joint has range ``[-3.1416, 0]``,
    so zero *is* its limit, and the arm's zero configuration is folded back on
    itself with two links overlapping by 5.7 mm; the KUKA LWR's zero pose buries
    one link 21 mm inside another. Every measurement taken from such a pose --
    reach, manipulability, the direction the tool points -- comes from a
    configuration the robot cannot hold, and a simulation starting there begins
    by exploding out of a penetration.

    So when the clamped zero self-intersects, unfold it. First walk toward the
    joint-range midpoint, the configuration furthest from every limit, and stop
    at the first clear pose; that keeps the displacement small. If the whole line
    stays pinched, fall back to coordinate descent on penetration depth.

    If nothing found is clear, the least-penetrating pose is returned and
    :func:`~rigby_general.ingest.integrity.check_rest_contacts` reports it, which
    is the honest outcome for a model with no pose it can rest in.
    """

    clamped = _clamped_qpos(model)
    depth = _penetration_depth(model, clamped)
    if depth <= 0.0:
        return clamped

    midpoint = _midpoint_qpos(model)
    best, best_depth = clamped, depth
    for step in range(1, REST_RELAXATION_STEPS + 1):
        blend = step / REST_RELAXATION_STEPS
        candidate = (1.0 - blend) * clamped + blend * midpoint
        candidate_depth = _penetration_depth(model, candidate)
        if candidate_depth <= 0.0:
            return candidate
        if candidate_depth < best_depth:
            best, best_depth = candidate, candidate_depth

    relaxed, _ = _descend(model, best, best_depth)
    return relaxed


def joint_range(model: mujoco.MjModel, joint: int) -> tuple[float, float]:
    """Limits, with an explicit span substituted for an unlimited joint.

    A ``continuous`` URDF joint arrives unlimited. Treating that as an infinite
    range would make every sweep meaningless, so it is measured over one full
    turn, which is all a revolute joint can distinguish anyway.
    """

    if model.jnt_limited[joint]:
        low, high = (float(value) for value in model.jnt_range[joint])
        if high > low:
            return low, high
    if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
        return -math.pi, math.pi
    return -0.1, 0.1


def _body_pose(data: mujoco.MjData, body: int) -> tuple[np.ndarray, np.ndarray]:
    return np.array(data.xpos[body], dtype=float), np.array(
        data.xmat[body], dtype=float
    ).reshape(3, 3)


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def measure_joint_motion(
    graph: KinematicGraph,
    joint: int,
    tip_body: int,
    *,
    base_qpos: np.ndarray,
    context_joints: tuple[int, ...] = (),
    probes: int = 6,
    seed: int = 0,
) -> JointMotion:
    """Measure how much one joint can move the tip, over the whole workspace.

    Two things this has to get right, both of which a naive version gets wrong:

    *Take the diameter of the swept set, not the endpoint separation.* A base yaw
    with a range of plus or minus 163 degrees returns the tip almost exactly
    where it started, so comparing endpoints scores the most important
    positioning joint on the robot at a couple of centimetres.

    *Probe from several configurations, not just from rest.* Most arms rest with
    the links folded along one axis, and a joint turning about that same axis
    moves nothing at all from that pose -- the tip is sitting on the rotation
    axis. Measured at rest alone, the base yaw of a vertically parked arm scores
    zero. What a joint is *for* is what it can do somewhere in the workspace, so
    the figure taken is the best over sampled configurations of the others.
    """

    model = graph.model
    data = mujoco.MjData(model)
    address = int(model.jnt_qposadr[joint])
    low, high = joint_range(model, joint)
    generator = np.random.default_rng(seed + joint)

    others = tuple(other for other in context_joints if other != joint)
    contexts: list[np.ndarray] = [np.asarray(base_qpos, dtype=float)]
    for _ in range(max(0, probes - 1) if others else 0):
        candidate = np.array(base_qpos, dtype=float)
        for other in others:
            other_low, other_high = joint_range(model, other)
            candidate[int(model.jnt_qposadr[other])] = generator.uniform(
                other_low, other_high
            )
        contexts.append(candidate)

    translation = 0.0
    rotation = 0.0
    for context in contexts:
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        for step in range(SWEEP_STEPS):
            data.qpos[:] = context
            data.qpos[address] = low + (high - low) * step / (SWEEP_STEPS - 1)
            mujoco.mj_kinematics(model, data)
            poses.append(_body_pose(data, tip_body))

        for index, (position, orientation) in enumerate(poses):
            for other_position, other_orientation in poses[index + 1 :]:
                translation = max(
                    translation, float(np.linalg.norm(other_position - position))
                )
                rotation = max(
                    rotation, _rotation_angle(orientation, other_orientation)
                )

    return JointMotion(joint=joint, translation_m=translation, rotation_rad=rotation)


def _min_surface_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first_geoms: tuple[int, ...],
    second_geoms: tuple[int, ...],
) -> float:
    best = CLOSURE_DISTANCE_CEILING
    scratch = np.zeros(6, dtype=float)
    for first in first_geoms:
        for second in second_geoms:
            distance = mujoco.mj_geomDistance(
                model, data, first, second, CLOSURE_DISTANCE_CEILING, scratch
            )
            best = min(best, float(distance))
    return best


def measure_closure(
    graph: KinematicGraph,
    member_bodies: tuple[int, ...],
    interior_joints: tuple[int, ...],
    *,
    base_qpos: np.ndarray,
) -> ClosureEvidence:
    """Drive the cluster's joints and watch whether its parts converge.

    This is the whole gripper test. Either the surfaces come together or they do
    not; the answer does not depend on what anything is called.
    """

    model = graph.model
    empty = ClosureEvidence(
        closes=False,
        open_distance_m=0.0,
        closed_distance_m=0.0,
        monotone=False,
        drive_to_upper=True,
        member_directions=(),
        opposition_groups=(),
    )
    if len(member_bodies) < 2 or not interior_joints:
        return empty

    geoms = [graph.collidable_geoms_of_body(body) for body in member_bodies]
    # Members with no collidable geometry are frames, not parts. Real URDFs are
    # full of them -- a massless link marking the tool centre point sits right
    # between the fingers, and the Franka hand has exactly that. Demanding that
    # every member have a surface meant one such marker abandoned the whole
    # test, and a real parallel jaw was reported as a rigid tool tip.
    #
    # They stay in ``member_bodies`` (a grasp centre is genuinely part of the
    # effector, and the index order is what ``member_directions`` and the
    # opposition groups are keyed by) and are simply not asked how far their
    # surfaces are apart.
    solid = [index for index, group in enumerate(geoms) if group]
    if len(solid) < 2:
        return empty

    data = mujoco.MjData(model)
    ranges = {joint: joint_range(model, joint) for joint in interior_joints}

    def evaluate(fraction: float) -> tuple[float, list[np.ndarray]]:
        data.qpos[:] = base_qpos
        for joint in interior_joints:
            low, high = ranges[joint]
            data.qpos[int(model.jnt_qposadr[joint])] = low + fraction * (high - low)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        distances = [
            _min_surface_distance(model, data, geoms[first], geoms[second])
            for position, first in enumerate(solid)
            for second in solid[position + 1 :]
        ]
        positions = [np.array(data.xpos[body], dtype=float) for body in member_bodies]
        return distances, positions

    # Closure is measured on the pair that actually converges, and finding that
    # pair took three attempts because two plausible reductions are both wrong.
    #
    # The mean over every pair hides the grip: on a thumb-opposed hand the index
    # and little finger curl in parallel and never approach each other, so
    # averaging them in turned a real 44 mm to 2 mm closure into a mean that
    # barely moved, and the hand read as not closing at all.
    #
    # The minimum is worse. Two adjacent fingers sit 4 mm apart and stay there
    # for the whole sweep, so the minimum locks onto a pair that never moves and
    # reports zero travel.
    #
    # What defines a grip is the surfaces that come *together*. So every pair is
    # tracked across the sweep and the one that converges most is the one the
    # measurement is taken from. A two-jaw gripper has exactly one pair, so this
    # is the number it always was.
    pairs = [
        (first, second)
        for position, first in enumerate(solid)
        for second in solid[position + 1 :]
    ]
    raw = [evaluate(step / (SWEEP_STEPS - 1)) for step in range(SWEEP_STEPS)]
    series = np.array([values for values, _ in raw], dtype=float)
    convergence = series[0] - series[-1]
    pair = int(np.argmax(np.abs(convergence))) if convergence.size else 0
    samples = [(float(values[pair]), positions) for values, positions in raw]
    distances = [value for value, _ in samples]

    at_lower, at_upper = distances[0], distances[-1]
    drive_to_upper = at_upper < at_lower
    ordered = distances if drive_to_upper else list(reversed(distances))

    # Closure is judged up to the point of closest approach, not to the end of
    # the sweep. A digit driven to its limit can travel *past* the one opposing
    # it -- the uHand's thumb reaches the index finger at -1 mm and is 6 mm the
    # other side of it one sample later -- and requiring monotonicity over the
    # whole range calls that "does not close". It plainly does; it then keeps
    # going. What happens after two surfaces meet is not evidence about whether
    # they met.
    meeting = int(np.argmin(ordered))
    approach = ordered[: meeting + 1] if meeting > 0 else ordered
    open_distance, closed_distance = approach[0], approach[-1]

    travel = open_distance - closed_distance
    monotone = all(
        later <= earlier * CLOSURE_MONOTONE_TOLERANCE + 1e-9
        for earlier, later in zip(approach, approach[1:])
    )
    closes = travel > CLOSURE_MIN_TRAVEL_M and monotone

    # Travel is read over the same span the closure was judged on: from open to
    # the point of closest approach. Reading it to the end of the sweep instead
    # measures a digit that has already passed its partner and is on its way out
    # again, which points the wrong way and puts every digit in one group.
    ordered_positions = (
        [positions for _, positions in samples]
        if drive_to_upper
        else [positions for _, positions in reversed(samples)]
    )
    open_positions = ordered_positions[0]
    closed_positions = ordered_positions[meeting if meeting > 0 else -1]

    directions: list[tuple[float, float, float]] = []
    for opened, closed in zip(open_positions, closed_positions):
        delta = closed - opened
        norm = float(np.linalg.norm(delta))
        directions.append(
            (0.0, 0.0, 0.0) if norm < 1e-9 else tuple(float(v) for v in delta / norm)
        )

    return ClosureEvidence(
        closes=closes,
        open_distance_m=open_distance,
        closed_distance_m=closed_distance,
        monotone=monotone,
        drive_to_upper=drive_to_upper,
        member_directions=tuple(directions),
        opposition_groups=(
            _opposition_groups(tuple(directions), open_positions, pairs[pair])
            if closes
            else ()
        ),
    )


def _opposition_groups(
    directions: tuple[tuple[float, float, float], ...],
    positions: "list[np.ndarray] | None" = None,
    axis_pair: "tuple[int, int] | None" = None,
) -> tuple[tuple[int, ...], ...]:
    """Split members into the two sides that face one another.

    Opposition is moving *toward each other*, which is not the same as moving
    differently. Comparing raw travel directions gets a two-jaw gripper right and
    a hand wrong: the uHand's thumb and index finger both rise as they curl, so
    their travel vectors sit only 79 degrees apart and the dot product comes out
    positive -- every digit lands in one group and a hand that visibly closes is
    reported as having nothing to close against.

    What separates the sides is the axis between the two digits that actually
    meet. Project each member's travel onto that axis and the sign says which
    side it closes from, whatever else its motion is doing.

    Falls back to the direction comparison when the caller has no positions,
    which keeps the older callers working.
    """

    if len(directions) < 2:
        return ()

    if positions is not None and axis_pair is not None:
        first, second = axis_pair
        axis = np.asarray(positions[second], dtype=float) - np.asarray(
            positions[first], dtype=float
        )
        norm = float(np.linalg.norm(axis))
        if norm > 1e-9:
            axis = axis / norm
            toward: list[int] = []
            away: list[int] = []
            for index, direction in enumerate(directions):
                projection = float(np.asarray(direction, dtype=float) @ axis)
                (toward if projection >= 0.0 else away).append(index)
            if toward and away:
                return (tuple(toward), tuple(away))

    reference = np.array(directions[0], dtype=float)
    same: list[int] = [0]
    against: list[int] = []
    for index in range(1, len(directions)):
        candidate = np.array(directions[index], dtype=float)
        (same if float(reference @ candidate) > 0.0 else against).append(index)
    if not against:
        return ()
    return (tuple(same), tuple(against))


def sample_reach(
    graph: KinematicGraph,
    probes: dict[str, tuple[int, np.ndarray]],
    chain_joints: dict[str, tuple[int, ...]],
    *,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample where each chain can put the point it is actually steered by.

    ``probes`` maps a chain to ``(body, local offset)`` rather than to a body
    alone, because the steered point is a site on that body and not its origin.
    On a gripper the two differ by the whole length of the palm, and measuring
    the envelope at one while aiming at the other makes a robot's own home
    position read as out of reach.

    Deterministically seeded: a robot's measured reach must not drift between two
    ingests of the same file, or every primitive baked against it would have been
    certified on a different body.
    """

    model = graph.model
    data = mujoco.MjData(model)
    generator = np.random.default_rng(seed)
    base_qpos = neutral_qpos(model)

    collected: dict[str, list[np.ndarray]] = {key: [] for key in probes}
    limits = {
        joint: joint_range(model, joint)
        for joints in chain_joints.values()
        for joint in joints
    }

    def record() -> None:
        mujoco.mj_kinematics(model, data)
        for key, (body, local) in probes.items():
            position = np.array(data.xpos[body], dtype=float)
            rotation = np.array(data.xmat[body], dtype=float).reshape(3, 3)
            collected[key].append(position + rotation @ local)

    # The rest pose belongs in the envelope, and random sampling will not find
    # it: with eight joints, the chance of a uniform draw landing near the middle
    # of every range is negligible. Yet rest is exactly where the arm parks and
    # where every primitive begins and ends, so an envelope that under-reports
    # reach in that direction rejects the robot's own home position.
    data.qpos[:] = base_qpos
    record()

    for _ in range(max(0, samples - 1)):
        data.qpos[:] = base_qpos
        for joint, (low, high) in limits.items():
            data.qpos[int(model.jnt_qposadr[joint])] = generator.uniform(low, high)
        record()

    return {key: np.asarray(values, dtype=float) for key, values in collected.items()}


def pose_jacobian(
    graph: KinematicGraph,
    tip_body: int,
    joints: tuple[int, ...],
    qpos: np.ndarray,
    *,
    length_scale: float,
) -> np.ndarray:
    """The 6 x N tip pose Jacobian, with the position rows made dimensionless.

    Position rows are metres per radian and orientation rows are radians per
    radian, so stacking them raw makes the rank of the result depend on whether
    the robot is measured in metres or millimetres. Dividing the position block
    by a characteristic length puts both blocks in the same units and makes the
    rank a statement about the mechanism instead of about the unit system.
    """

    model = graph.model
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacBody(model, data, jacp, jacr, tip_body)

    columns = [int(model.jnt_dofadr[joint]) for joint in joints]
    scale = max(float(length_scale), 1e-6)
    return np.vstack([jacp[:, columns] / scale, jacr[:, columns]])


def redundant_joints(
    graph: KinematicGraph,
    tip_body: int,
    chain_joints: tuple[int, ...],
    *,
    length_scale: float,
    seed: int,
    probes: int = 12,
) -> frozenset[int]:
    """Joints that add no rank to the full tip pose Jacobian, anywhere.

    This is the classical definition of kinematic redundancy, and it has to be
    taken against the full six-row pose Jacobian. Against the three position rows
    alone, every joint past the third scores as redundant -- which would file a
    six-axis arm's entire wrist as surplus, when in fact it is what lets the arm
    reach a pose rather than merely a point.

    A joint is only reported once it has failed to add rank at *every* sampled
    configuration, because any single pose may be singular and would condemn a
    perfectly useful joint on the strength of one unlucky draw.
    """

    if len(chain_joints) <= 6:
        return frozenset()

    model = graph.model
    generator = np.random.default_rng(seed)
    base_qpos = neutral_qpos(model)
    limits = {joint: joint_range(model, joint) for joint in chain_joints}

    never_added_rank = set(chain_joints)
    for _ in range(probes):
        qpos = base_qpos.copy()
        for joint, (low, high) in limits.items():
            qpos[int(model.jnt_qposadr[joint])] = generator.uniform(low, high)
        jacobian = pose_jacobian(
            graph, tip_body, chain_joints, qpos, length_scale=length_scale
        )

        kept: list[int] = []
        rank = 0
        for column, joint in enumerate(chain_joints):
            trial = kept + [column]
            trial_rank = int(np.linalg.matrix_rank(jacobian[:, trial], tol=1e-6))
            if trial_rank > rank:
                rank = trial_rank
                kept.append(column)
                never_added_rank.discard(joint)
    return frozenset(never_added_rank)


def farthest_geom_point(
    graph: KinematicGraph, body: int, direction: np.ndarray, qpos: np.ndarray
) -> np.ndarray:
    """The extreme point of a body's geometry along ``direction``, in body frame.

    Used to put a tip site at the actual end of the last link rather than at its
    origin, which on most robots sits at the joint rather than at the fingertip.
    """

    model = graph.model
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)

    body_pos = np.array(data.xpos[body], dtype=float)
    body_mat = np.array(data.xmat[body], dtype=float).reshape(3, 3)

    best_local = np.zeros(3, dtype=float)
    best_score = -math.inf
    for geom in graph.geoms_of_body(body):
        centre = np.array(data.geom_xpos[geom], dtype=float)
        rotation = np.array(data.geom_xmat[geom], dtype=float).reshape(3, 3)
        extent = _geom_extent(model, geom)
        for sign in (-1.0, 1.0):
            for axis in range(3):
                offset = np.zeros(3, dtype=float)
                offset[axis] = sign * extent[axis]
                world = centre + rotation @ offset
                score = float(world @ direction)
                if score > best_score:
                    best_score = score
                    best_local = body_mat.T @ (world - body_pos)
    return best_local


def _geom_extent(model: mujoco.MjModel, geom: int) -> np.ndarray:
    size = np.array(model.geom_size[geom], dtype=float)
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.array([size[0], size[0], size[0]])
    if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        return np.array([size[0], size[0], size[1]])
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return size
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return size
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        if mesh >= 0:
            return np.array(model.mesh_extent[mesh] if hasattr(model, "mesh_extent")
                            else [size[0] or 0.05] * 3, dtype=float)
    return np.array([max(float(size[0]), 0.01)] * 3)
