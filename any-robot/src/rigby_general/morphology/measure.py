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
SIGHTED_CLOSURE_MIN_FRACTION = 0.25

WITNESS_SEARCH_M = 10.0
"""How far apart two surfaces may be and still be asked where they meet."""
"""How much of its own opening a sighted gap must give up to count as a grip.

An absolute threshold is meaningless for this measure. Any two bodies with a
joint between them show *some* change in the clear span across a sweep, so half a
millimetre admitted the KUKA iiwa's bare wrist (4.5 mm out of 158) and a cart on
a rail as grippers. A gripper gives up most of its opening; a wrist that happens
to swing gives up a few percent of a gap that was never a grasp.
"""

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

    grasp_point_world: tuple[float, float, float] | None = None
    """Where the aperture was measured, in world coordinates at the open pose.

    An object sized from the aperture belongs where that aperture is."""

    grasp_aperture_m: float = 0.0
    """How much room there is between the opposing sides, where an object sits.

    Distinct from ``open_distance_m``, which is the narrowest gap *anywhere*
    between two member bodies. Those are the same number on parallel plates and
    are not the same number on anything hinged: a jaw pivoted at its base keeps
    its closest surfaces at the pivot, where opening moves nothing, so the global
    minimum reports the hinge gap rather than the opening. The EEZYbotARM
    measures 10 mm that way with its tips 49 mm apart, and is then refused every
    object in every world -- including a 24 mm cube in the one bench authored so
    that every gripper here could attempt it.

    Zero when it could not be measured, in which case the caller falls back.
    """


REST_RELAXATION_STEPS = 50
"""Interpolation samples between the clamped zero pose and the range midpoint."""

REST_PENETRATION_TOLERANCE_M = -1e-4

REST_DESCENT_PASSES = 3
REST_DESCENT_SAMPLES = 9


LIMIT_STANDOFF_FRACTION = 0.02
"""How far inside its range a resting joint is held, as a fraction of that range.

A joint parked exactly on a limit is pushed past it by the first disturbance,
and the position gate then reports a violation for a robot that never moved. The
KUKA's gripper rests fully open, which is its lower limit, and 32 of its 72
bake attempts failed on that alone -- not because anything went wrong, but
because "open" and "as far open as it goes" were the same number.
"""


def _clamped_qpos(model: mujoco.MjModel) -> np.ndarray:
    """The zero pose, pushed inside every joint limit and off the limits."""

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
            standoff = LIMIT_STANDOFF_FRACTION * (high - low)
            qpos[address] = min(
                max(qpos[address], low + standoff), high - standoff
            )
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



def _first_hit_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: np.ndarray,
    direction: np.ndarray,
    expected: frozenset[int],
) -> float | None:
    """Distance to the first geometry along ``direction``, if it is ``expected``.

    Requiring the hit to land on the side being sighted is what keeps the
    measurement honest. Without it a ray that slips past the edge of a jaw flies
    on and strikes the forearm, or the floor, and reports that gap as the
    opening -- the Panda measured 201 mm that way, against a hand that opens 80.
    A sightline that misses the jaw has not measured the jaw.
    """

    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    geomid = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model,
        data,
        np.ascontiguousarray(origin, dtype=float),
        np.ascontiguousarray(direction / norm, dtype=float),
        None,
        1,
        -1,
        geomid,
    )
    if distance < 0 or int(geomid[0]) not in expected:
        return None
    return float(distance)


def grasp_aperture(
    graph: "KinematicGraph",
    data: mujoco.MjData,
    member_bodies: tuple[int, ...],
    opposition_groups: tuple[tuple[int, ...], ...],
) -> tuple[float, "np.ndarray | None"]:
    """Measure the opening the way the number is used: what fits between them.

    Stand at the midpoint between the two opposing sides and look at each of
    them. The room between the surfaces those two sightlines land on is the
    widest thing that can sit there. That is a measurement rather than a rule,
    so it needs to know nothing about whether the members slide, pivot, or curl,
    and it reduces to the plate separation when the members *are* plates.

    ``data`` must already be at the open pose.
    """

    if len(opposition_groups) != 2:
        return 0.0, None

    # Sight from the middle of the *geometry*, not the body origins. A body
    # origin sits on its joint, so on a hinged jaw the origins are the two
    # pivots and the midpoint between them lands back at the hinge -- measuring
    # the very gap this exists to stop measuring. Geom centroids sit on the jaws
    # themselves, where an object is held.
    centres = []
    side_geoms: list[frozenset[int]] = []
    for group in opposition_groups:
        side_geoms.append(
            frozenset(
                geom
                for index in group
                if 0 <= index < len(member_bodies)
                for geom in graph.collidable_geoms_of_body(member_bodies[index])
            )
        )
        points = [
            np.array(data.geom_xpos[geom], dtype=float)
            for index in group
            if 0 <= index < len(member_bodies)
            for geom in graph.collidable_geoms_of_body(member_bodies[index])
        ]
        if not points:
            return 0.0, None
        centres.append(np.mean(points, axis=0))

    axis = centres[1] - centres[0]
    span = float(np.linalg.norm(axis))
    if span < 1e-9:
        return 0.0, None
    axis = axis / span
    middle = (centres[0] + centres[1]) / 2.0

    # One sightline through the middle is enough for plates, which are parallel
    # everywhere, and wrong for anything that curls. A hand's opposing groups
    # average out to somewhere near the palm, and the single line from there
    # crosses a finger edge-on rather than the opening: the uHand measured 31 mm
    # against a 73 mm sweep, which would have refused it the 50 mm ball its own
    # bench was authored to present. So sample a patch across the gap and take
    # the widest clear span, which is what "what fits" means. Plates give the
    # same answer everywhere on the patch and are unaffected.
    first = np.cross(axis, (0.0, 0.0, 1.0))
    if float(np.linalg.norm(first)) < 1e-6:
        first = np.cross(axis, (0.0, 1.0, 0.0))
    first = first / float(np.linalg.norm(first))
    second = np.cross(axis, first)

    radius = 0.5 * span
    widest = 0.0
    widest_at: np.ndarray | None = None
    for u in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
            origin = middle + radius * (u * first + v * second)
            forward = _first_hit_distance(
                graph.model, data, origin, axis, side_geoms[1]
            )
            backward = _first_hit_distance(
                graph.model, data, origin, -axis, side_geoms[0]
            )
            if forward is None or backward is None:
                continue
            clear = float(forward + backward)
            if clear > widest:
                widest = clear
                # Midway between the two surfaces the sightline landed on. This
                # is the point the aperture is an aperture *at*, so it is also
                # where an object of that width has to be put.
                widest_at = origin + axis * (forward - backward) / 2.0
    return widest, widest_at


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

    # A joint with no limits has no closed configuration, so it cannot be a grip
    # joint: there is no pose at which it holds rather than passes through.
    # Without this a cart on a rail measures as a 900 mm parallel jaw, its pole
    # swinging freely through the cart and duly "converging" on it. Every real
    # grip joint in the fleet is limited; a free hinge is a mechanism, not a
    # hand. One bounded joint is enough to be worth measuring.
    if not any(bool(model.jnt_limited[joint]) for joint in interior_joints):
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

    # Candidates are *surface* pairs, not body pairs. Taking the minimum over a
    # body's whole surface finds wherever two members are closest, and on real
    # hardware that is the mount rather than the grip: the SO-ARM101's moving
    # jaw is seated 20 mm inside the hull of the palm that carries its servo, so
    # the body-level minimum is a constant negative number, never converges, and
    # the gripper reads as not closing. The pair that converges has to be chosen
    # at the resolution the geometry is actually authored at.
    candidates = [
        (first, second, geom_first, geom_second)
        for position, first in enumerate(solid)
        for second in solid[position + 1 :]
        for geom_first in geoms[first]
        for geom_second in geoms[second]
    ]
    if not candidates:
        return empty

    scratch = np.zeros(6, dtype=float)

    def evaluate(fraction: float) -> tuple[list[float], list[np.ndarray]]:
        data.qpos[:] = base_qpos
        for joint in interior_joints:
            low, high = ranges[joint]
            data.qpos[int(model.jnt_qposadr[joint])] = low + fraction * (high - low)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        distances = [
            float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    geom_first,
                    geom_second,
                    CLOSURE_DISTANCE_CEILING,
                    scratch,
                )
            )
            for _, _, geom_first, geom_second in candidates
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
    pairs = [(first, second) for first, second, _, _ in candidates]
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

    groups = (
        _opposition_groups(tuple(directions), open_positions, pairs[pair])
        if closes
        else ()
    )

    # Last resort, and only for two surfaces: watch the *gap* instead of the
    # bodies. Convex collision hulls can overlap for the whole sweep on hardware
    # that grips perfectly well -- the SO-ARM101's moving jaw is seated inside
    # the hull of the palm carrying its servo, so every surface pair between
    # them is a constant negative number and convergence is unmeasurable. The
    # opening between them is not: sight across it at each step and watch it
    # shrink. Restricted to the two-surface case because that is where the sides
    # are unambiguous; a hand's grouping is what the convergence test is for.
    sighted_open: float | None = None
    # The permissive path asks for more: every joint bounded, not merely one.
    bounded = interior_joints and all(
        bool(model.jnt_limited[joint]) for joint in interior_joints
    )
    if not closes and len(solid) == 2 and bounded:
        candidate_groups = ((solid[0],), (solid[1],))
        spans = []
        for step in range(SWEEP_STEPS):
            evaluate(step / (SWEEP_STEPS - 1))
            spans.append(
                grasp_aperture(graph, data, member_bodies, candidate_groups)[0]
            )
        if all(span > 0.0 for span in spans):
            # Judge from the widest point outward, not end to end. A sightline
            # tracks the opening while the jaw swings through it and then catches
            # the far face once it swings past, so the series rises and falls:
            # the SO-ARM101 reads 26 mm closed, opens to 94, and reports 29 at
            # the very last sample with the jaw right past the sightline. What
            # is being asked is whether the gap closes, and the answer is read
            # from the widest configuration down to the narrow end.
            peak = int(np.argmax(spans))
            best: tuple[float, bool, float, float] | None = None
            for end, forward in ((0, False), (len(spans) - 1, True)):
                if end == peak:
                    continue
                step = 1 if end > peak else -1
                walk = spans[peak : end + step : step] if step > 0 else spans[peak::step]
                shrinks = all(
                    later <= earlier * CLOSURE_MONOTONE_TOLERANCE + 1e-9
                    for earlier, later in zip(walk, walk[1:])
                )
                travel = walk[0] - walk[-1]
                # Both directions off the peak can shrink; the grip is the one
                # that shrinks *most*. Taking whichever was tested first gave
                # the Beetlebot a 4 mm stroke when its jaws travel 46.
                enough = travel > CLOSURE_MIN_TRAVEL_M and (
                    travel >= SIGHTED_CLOSURE_MIN_FRACTION * walk[0]
                )
                if shrinks and enough:
                    if best is None or travel > best[0]:
                        best = (travel, forward, walk[0], walk[-1])
            if best is not None:
                closes = True
                monotone = True
                _, drive_to_upper, open_distance, closed_distance = best
                groups = candidate_groups
                sighted_open = open_distance

    # Re-pose at the open end before sighting across the gap: `evaluate` left
    # `data` wherever the sweep last put it, which is the closed end half the
    # time, and an aperture measured with the jaws shut is zero.
    aperture = 0.0
    grasp_point: np.ndarray | None = None
    if closes and sighted_open is not None:
        # The sighted path already measured the opening, at the widest pose
        # rather than at a limit. Re-posing to an end would re-measure the very
        # artifact the peak anchoring exists to step around.
        aperture = sighted_open
        # ...but it still owes a point. This branch set the aperture and left
        # `grasp_point` as None, while its twin below set both, so every
        # effector measured through the sighted path came out with an opening
        # and nowhere to hold: the SO-ARM101 reported a 94 mm aperture and no
        # grasp point at all, fell back to the tool centre its URDF declares,
        # and drove that into the block -- 28 mm of housing inside the object
        # before the jaws had closed.
        evaluate(0.0 if drive_to_upper else 1.0)
        _, grasp_point = grasp_aperture(graph, data, member_bodies, groups)
        if grasp_point is None:
            grasp_point = _witness_midpoint(
                graph, data, groups, evaluate, drive_to_upper
            )
    elif closes:
        evaluate(0.0 if drive_to_upper else 1.0)
        aperture, grasp_point = grasp_aperture(graph, data, member_bodies, groups)
        if grasp_point is None:
            grasp_point = _witness_midpoint(
                graph, data, groups, evaluate, drive_to_upper
            )

    return ClosureEvidence(
        closes=closes,
        open_distance_m=open_distance,
        closed_distance_m=closed_distance,
        monotone=monotone,
        drive_to_upper=drive_to_upper,
        member_directions=tuple(directions),
        opposition_groups=groups,
        grasp_aperture_m=aperture,
        grasp_point_world=(
            tuple(float(v) for v in grasp_point) if grasp_point is not None else None
        ),
    )


def _witness_midpoint(
    graph: "KinematicGraph",
    data: mujoco.MjData,
    groups: tuple,
    evaluate,
    drive_to_upper: bool,
) -> "np.ndarray | None":
    """Halfway between the two surfaces that actually converge.

    Where sighting finds no clear line -- a housing whose hull swallows its own
    jaw, so no ray leaves one member and lands on the other -- the closest
    approach between the opposing groups still says where they meet.

    The *closest* pair is not the right pair, though. A hinged jaw is nearest
    its housing at the pivot, and on the SO-ARM101 those two surfaces overlap by
    22 mm at every jaw angle: taking the global minimum put the grasp point
    inside the mechanism, and the arm pressed 1 kN through a 10 g block trying
    to reach it. The pair that grips is the pair that *closes* -- open the hand,
    shut it, and keep whichever surfaces gave up the most distance.
    """

    if len(groups) < 2:
        return None
    model = graph.model
    pairs = [
        (one, other)
        for first in groups[0]
        for second in groups[1]
        for one in graph.collidable_geoms_of_body(first)
        for other in graph.collidable_geoms_of_body(second)
    ]
    if not pairs:
        return None

    open_fraction = 0.0 if drive_to_upper else 1.0
    segment = np.zeros(6, dtype=float)

    evaluate(open_fraction)
    opened = [
        float(mujoco.mj_geomDistance(model, data, a, b, WITNESS_SEARCH_M, None))
        for a, b in pairs
    ]
    evaluate(1.0 - open_fraction)
    shut = [
        float(mujoco.mj_geomDistance(model, data, a, b, WITNESS_SEARCH_M, None))
        for a, b in pairs
    ]

    best_index, best_travel = None, 0.0
    for index, (start, end) in enumerate(zip(opened, shut)):
        if start <= 0.0 or start >= WITNESS_SEARCH_M:
            continue
        travel = start - end
        if travel > best_travel:
            best_index, best_travel = index, travel
    if best_index is None:
        return None

    evaluate(open_fraction)
    one, other = pairs[best_index]
    mujoco.mj_geomDistance(model, data, one, other, WITNESS_SEARCH_M, segment)
    return (segment[:3] + segment[3:]) / 2.0


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
                return _ordered_groups(toward, away, positions, axis)

    reference = np.array(directions[0], dtype=float)
    same: list[int] = [0]
    against: list[int] = []
    for index in range(1, len(directions)):
        candidate = np.array(directions[index], dtype=float)
        (same if float(reference @ candidate) > 0.0 else against).append(index)
    if not against:
        return ()
    if len(against) > len(same):
        same, against = against, same
    return (tuple(same), tuple(against))



def _ordered_groups(
    toward: "list[int]",
    away: "list[int]",
    positions: "list[np.ndarray]",
    axis: "np.ndarray",
) -> tuple[tuple[int, ...], ...]:
    """The two sides, larger first, each ordered across the effector.

    Two orderings, both measured, both needed by a posture:

    The *groups* are ordered by size, so the larger side comes first. On a hand
    that is the fingers and then the thumb; on a two-jaw gripper the sides tie and
    the order is settled by position, which is arbitrary but stable. A posture
    naming ``opposed`` therefore means the same kind of thing on both.

    The *members within a group* are ordered along the axis across the effector --
    perpendicular to the direction they close along. That is what makes "two
    fingers" mean two neighbours rather than two arbitrary ones: on the uHand the
    members arrive index, little, middle, ring in alphabetical order and index,
    middle, ring, little in this one, and only the second is a hand.
    """

    lateral = _lateral_axis(positions, axis)

    def across(index: int) -> float:
        return float(np.asarray(positions[index], dtype=float) @ lateral)

    first, second = (toward, away) if len(toward) >= len(away) else (away, toward)
    if len(first) == len(second):
        # A tie has no larger side to pick, so settle it on position rather than
        # on whichever way the closure sweep happened to run.
        first, second = sorted((first, second), key=lambda group: across(group[0]))
    return (
        tuple(sorted(first, key=across)),
        tuple(sorted(second, key=across)),
    )


def _lateral_axis(positions: "list[np.ndarray]", closing: "np.ndarray") -> "np.ndarray":
    """The direction across the effector, perpendicular to how it closes.

    Taken from the spread of the members themselves rather than from any body
    frame: the widest direction they occupy, with the closing direction removed.
    A gripper whose members are collinear with its closing direction has no such
    spread, and any consistent axis will do -- there is nothing to order.
    """

    cloud = np.asarray(positions, dtype=float)
    centred = cloud - cloud.mean(axis=0)
    # Remove the closing direction so members that merely close at different
    # times do not read as being at different places across the hand.
    centred = centred - np.outer(centred @ closing, closing)

    if centred.shape[0] < 2 or float(np.abs(centred).max()) < 1e-9:
        fallback = np.array([1.0, 0.0, 0.0])
        fallback = fallback - float(fallback @ closing) * closing
        norm = float(np.linalg.norm(fallback))
        return fallback / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])

    # The widest remaining direction, which for any ordinary hand is the line the
    # digits are planted along.
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    axis = np.asarray(right[0], dtype=float)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])


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


EFFORT_RAMP_SECONDS = 0.25
"""How long a joint is allowed to take to reach its own top speed from rest.

Turns a declared speed limit into the acceleration a limit has to support. A
quarter second is brisk without being a step change, and it is stated here
rather than derived because nothing in a URDF says how hard a robot is meant to
be driven -- only how fast it may end up going.
"""

DEFAULT_JOINT_SPEED_RAD_S = 1.0
"""Reference speed for a joint whose source declares none.

Used only to size an *undeclared* torque limit, so it never overrides anything
the installer actually said.
"""


EFFORT_SAMPLE_POSES = 192
"""How many configurations the gravity-torque sweep visits.

Enough to catch the coupling between joints -- an elbow's torque about the
shoulder depends on where the elbow is -- without making ingest slow. The
sequence is deterministic, so the number a robot is admitted with does not
change between runs.
"""

EFFORT_DYNAMIC_MARGIN = 2.5
"""Headroom over the worst static hold, for the accelerations of actually moving.

A joint sized exactly to hold its own weight can hold it and do nothing else:
every newton-metre is spent on gravity and none is left to accelerate. The
margin is what separates a limit that can support the arm from one that can move
it. It is a stated engineering choice rather than a measurement, which is why it
is a named constant and not a literal buried in a formula.
"""


def _halton(index: int, base: int) -> float:
    """One term of a Halton sequence: deterministic, and spread better than a grid.

    A grid over eight joints is either too coarse to see the coupling or far too
    large to evaluate. A low-discrepancy sequence covers the box evenly at any
    sample count, and unlike a random sample it gives the same answer every run.
    """

    fraction = 1.0
    value = 0.0
    while index > 0:
        fraction /= base
        value += fraction * (index % base)
        index //= base
    return value


_HALTON_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def effort_floor(
    model: mujoco.MjModel,
    velocity_limits: "dict[int, float] | None" = None,
    moment_arms: "dict[int, float] | None" = None,
    position_tolerance_m: float = 0.0,
) -> np.ndarray:
    """The worst torque each joint must produce, over its own workspace.

    Two terms, because a joint that can only hold is a joint that cannot move:

        tau = |g(q)|  +  M(q)[i,i] * qacc_ref

    Gravity alone is not enough, and the KUKA shows exactly why. Its first axis
    is a base yaw about the vertical, so gravity exerts *no* torque about it in
    any pose whatsoever -- sizing from gravity gives it a limit of zero and an
    arm that cannot turn. What that axis actually has to do is accelerate
    everything above it, which is the mass matrix, not gravity.

    A URDF that declares ``effort="0"`` has not given a small limit; it has
    declined to give one. Substituting a fixed number for that absence is the
    same mistake a single controller gain would be: the KUKA's second axis needs
    145 N*m to hold its own arm up, a default of 100 said it could not, and every
    one of its 72 primitives failed tracking because the simulated arm sagged
    under gravity exactly as the model said it must.

    So measure it instead. Both terms are properties of the body and the pose,
    and both are known -- ``mj_rne`` with acceleration zeroed reports gravity
    directly, and ``mj_fullM`` reports the inertia the joint sees right now.
    Sweeping the joint box and taking the worst case per joint gives a floor that
    is a statement about *this* robot rather than about robots in general.

    ``qacc_ref`` comes from the joint's own declared speed, reached from rest
    within ``EFFORT_RAMP_SECONDS``.

    What this is *not* is a substitute for a limit the source declined to give.
    It is a lower bound -- below it the joint cannot hold its own arm up, let
    alone move it -- and a lower bound is not a limit. Sizing an actual limit
    would mean sizing it to what the controller demands, and computed torque
    commands ``omega**2 * e`` with omega at 88 rad/s: for the KUKA's base yaw,
    at the tolerance this system certifies to, that is 23 kN*m. A number that
    large is evidence the question has no sound answer, not an answer. So the
    floor is reported for what it is worth, and the limit is left unknown --
    see ``RobotJointV1.effort_declared``.

    Returns one value per degree of freedom, in the model's dof order.
    """

    data = mujoco.MjData(model)
    rest = neutral_qpos(model)
    movable = _movable_joints(model)
    worst = np.zeros(model.nv, dtype=float)
    torque = np.zeros(model.nv, dtype=float)
    inertia = np.zeros((model.nv, model.nv), dtype=float)

    reference_acceleration = np.full(
        model.nv, DEFAULT_JOINT_SPEED_RAD_S / EFFORT_RAMP_SECONDS, dtype=float
    )
    for joint in movable:
        dof = int(model.jnt_dofadr[joint])
        speed = abs((velocity_limits or {}).get(int(joint), DEFAULT_JOINT_SPEED_RAD_S))
        reference_acceleration[dof] = speed / EFFORT_RAMP_SECONDS

    for sample in range(EFFORT_SAMPLE_POSES):
        qpos = np.array(rest, dtype=float)
        for slot, joint in enumerate(movable):
            low, high = joint_range(model, joint)
            base = _HALTON_BASES[slot % len(_HALTON_BASES)]
            qpos[int(model.jnt_qposadr[joint])] = low + (high - low) * _halton(
                sample + 1, base
            )
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_forward(model, data)
        mujoco.mj_rne(model, data, 0, torque)
        mujoco.mj_fullM(model, data, inertia)
        demand = np.abs(torque) + np.abs(np.diag(inertia)) * reference_acceleration
        np.maximum(worst, demand, out=worst)

    return worst


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
