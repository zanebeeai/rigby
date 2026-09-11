"""Size an authored world so a particular arm can actually be asked for it.

An environment is authored once, in absolute metres, knowing nothing about who
will be dropped into it -- that is what makes `object_out_of_reach` a real
finding rather than a tuning artefact. But a world authored for a metre-class
arm asks nothing at all of a 240 mm one: every object refuses on distance and
the arm is never tested.

So the world is offered at its authored size and, where that cannot be asked,
at the largest size that can. The check the fit closes against is admission
itself, because scaling by the reach radius is the right idea and the wrong
number -- admission measures from the workspace frame against the usable
fraction of directional reach, and those differ enough to leave scaled worlds
still refusing.

This lives here rather than in the trial script because two callers need it and
they drifted: the console loaded authored worlds while the runner fitted them,
so the same robot in the same world got two different answers.
"""

from __future__ import annotations

import numpy as np

import mujoco

from ..contact.closure import ClosureConfig
from ..morphology.graph import KinematicGraph
from ..morphology.measure import neutral_qpos
from .environment import MIN_BLOCK_DENSITY


DYNAMIC_HOLD_FACTOR = 5.0
"""How much more than the object's own weight a grip must supply to carry it --
the acceleration of the lift, not a safety factor."""

APPROACH_CLEARANCE = 0.8
"""How much of the measured opening is usable once the jaws need room to get
around an object rather than onto it."""

CLOSED_SPAN_MARGIN = 1.1
"""How much wider than the shut jaws the smallest object must be.

An object has to clear the shut jaws by enough to be *squeezed* rather than
merely cleared. It also has to stay under `object_too_wide`, and on a hand whose
jaws shut wide those two bounds nearly meet: the Beetlebot admits objects
between 91 mm and 125 mm, and at 1.4 this sized its blocks to 127.5 -- three
millimetres outside its own window, which left it one feasible pairing in the
whole suite. Holds are thirty-seven at 1.0 through 1.6 alike, so the value is
chosen for coverage: 1.1 asks eighty-five pairings against eighty-one at 1.4,
and gives the Beetlebot five attempts instead of one.

This changed no holds and was kept anyway: it stops a class of task being set
at all. A gripper whose surfaces still stand 91 mm apart when shut cannot pinch
a 44 mm block, and asking it to reports `grasp_not_achieved` -- a grasp that
failed -- for a grasp that was never geometrically available."""


def _closed_span(robot, effector) -> float:
    """How far apart this hand's gripping surfaces still are when shut."""

    from rigby_general.morphology.analyze import _measure_chains
    from rigby_general.morphology.graph import KinematicGraph
    from rigby_general.morphology.measure import neutral_qpos

    model = robot.finalized.model
    graph = KinematicGraph(model)
    for chain in _measure_chains(graph, neutral_qpos(model)):
        if chain.closure.closes and chain.chain_id == effector.chain_id:
            return float(chain.closure.closed_distance_m)
    return 0.0


def _grippable_aperture(robot, effector, authored, aperture: float) -> float:
    """Size the world's objects to what the jaws can *pinch*, not merely admit.

    An opening has two ends and only one of them was being used. The Beetlebot's
    claws shut at 91 mm -- that is as close as its gripping surfaces ever come --
    and every block its worlds handed it measured 44 to 69 mm, so the jaws
    bottomed out around an object they never touched, on all eight pairings. The
    block was sized as a fraction of the *opening*, which says what the hand can
    admit and nothing about what it can hold.

    So the objects are scaled up, if need be, until the smallest of them clears
    the closed span with a margin.
    """

    closed = _closed_span(robot, effector)
    if closed <= 0.0 or not authored.authored_for_aperture_m:
        return aperture
    smallest = min(
        (item.span_m for item in authored.objects), default=0.0
    )
    if smallest <= 0.0:
        return aperture
    needed = (
        closed
        * CLOSED_SPAN_MARGIN
        * authored.authored_for_aperture_m
        / smallest
    )
    return max(aperture, needed)


def _holdable_mass(robot, effector) -> float:
    """What this hand can actually hold, from its own grip actuators.

    `scale.payload_kg` is derived from the arm's declared joint efforts, and
    those are frequently fiction -- the SO-ARM101 declares 10 N*m on every axis,
    the 5-DOF SG90 arm declares 1000. What decides whether a block stays in the
    hand is narrower and measurable: the squeeze the closure controller is
    allowed to command, which is half the weakest grip actuator, held against
    gravity through friction with a safety margin.

    Taking the smaller of the two lets a world be fitted to the hand as well as
    to the arm.
    """

    import mujoco

    from rigby_general.contact.closure import ClosureConfig

    model = robot.finalized.model
    limits = []
    for name in effector.grip_joints:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0 and model.jnt_actfrclimited[joint]:
            limits.append(abs(float(model.jnt_actfrcrange[joint][1])))
    if not limits:
        return float(robot.morphology.scale.payload_kg)

    config = ClosureConfig()
    squeeze = config.actuator_ceiling_fraction * min(limits)
    opposing = max(len(effector.opposition_groups), 2)
    # Twice the controller's own safety factor again, swept: 2.0 and 3.0 both
    # hold fourteen of the authored worlds, 1.5 and 2.5 thirteen. The closure
    # controller sizes its squeeze for a static hold; an object being carried is
    # also being accelerated, and the margin is what covers the difference.
    grip_mass = (squeeze * 1.4 * opposing) / (
        # Twenty times the controller's own safety factor. The closure
        # controller sizes a squeeze for a static hold; an object being carried
        # is accelerated too, and this is the headroom that covers it. It has
        # two jobs, since a capped block now shrinks rather than thins: it sets
        # the block's size as well as its mass. Re-swept once an undeclared
        # payload stopped being a flat 10 g -- which had been quietly keeping
        # the KUKA's blocks light -- 20 holds thirty-three of the authored
        # worlds, 16 and 30 thirty-two, 12 and 8 thirty and thirty-one.
        9.81 * config.grip_safety_factor * 20.0
    )
    capped = float(min(robot.morphology.scale.payload_kg, max(grip_mass, 1e-4)))

    # ...but never below the lightest thing the jaws can actually close on.
    #
    # The cap is headroom on top of the squeeze the controller demands, and on a
    # hand with a wide closed span the two collide: the SO-ARM101's jaws stop
    # 26 mm apart, so nothing narrower than that can be pinched, and the cap
    # sized its block at 31 mm and 0.89 g -- five millimetres of travel between
    # first touch and the jaws bottoming out, which the gates read as crushing.
    # Asking for a bigger block means a heavier one, and the cap forbade it.
    #
    # The physics does not. That hand commands 5 N of squeeze through friction
    # on two members against an object weighing 0.012 N. Where the mechanism can
    # plainly hold what it can plainly pinch, the heuristic gives way to the
    # measurement.
    closed = _closed_span(robot, effector)
    if closed <= 0.0:
        return capped
    smallest = MIN_BLOCK_DENSITY * (closed * CLOSED_SPAN_MARGIN) ** 3
    physical = (squeeze * 1.4 * opposing) / (9.81 * DYNAMIC_HOLD_FACTOR)
    if smallest <= physical:
        return float(min(robot.morphology.scale.payload_kg, max(capped, smallest)))
    return capped


_MISS_PENALTY = 1.0
"""What an object the fit cannot place costs, against how well it centres the
ones it can.

Swept: 0.15, 0.5 and 1.0 all hold thirty-three, and 1.0 gets the most pairings
attempted -- eighty-three against seventy-eight -- so the tie goes to the size
that leaves fewest objects unasked."""


def fit_world(authored, robot, effector, frame):
    """The same world, sized so this arm can actually be asked for it.

    Scaling by the robot's measured reach radius is the right idea and the wrong
    number: admission measures distance from the *workspace frame* -- the hand's
    own rest position, not the base -- and against the *usable* fraction of the
    directional reach on that bearing. Those differ enough that nine pairings
    were still refused `object_out_of_reach` in worlds scaled to fit them, the
    KUKA's cube sitting 919 mm out against a 980 mm reach that admission trims
    before it looks.

    So the fit is closed against the check itself: shrink the world until every
    object in it is one the arm can be asked for, and stop at the first size that
    is. A world that never fits is left at its authored size and refused
    honestly, which is the same answer as before.
    """

    from rigby_general.scenes import admit_object

    nominal = robot.morphology.scale.reach_radius_m
    holdable = _holdable_mass(robot, effector)
    aperture = (
        effector.max_aperture_m or authored.authored_for_aperture_m or 0.05
    )
    # Leave the jaws room to approach, the way the derived scene already does.
    #
    # `build_grasp_scene` sizes its block at `BLOCK_APERTURE_FRACTION` of the
    # measured opening because "a block at the full aperture cannot be
    # approached without the jaws already touching it". The authored worlds were
    # scaled against the *whole* aperture and never got that margin, so a hand
    # arrived at an object filling its opening with nowhere to put itself.
    #
    # The clearance belongs to the hand rather than to how a world was authored:
    # keying it off each world's own widest object leaves the worlds that author
    # small objects with no margin at all, and holds thirty-one. A flat fraction
    # of the measured opening holds thirty-five. Swept: 0.8 holds thirty-five,
    # 0.75 and 0.85 thirty-four each, 0.9 thirty-two, 1.0 thirty-three and 0.6
    # twenty-five.
    aperture *= APPROACH_CLEARANCE
    aperture = _grippable_aperture(robot, effector, authored, aperture)
    floor = _closed_span(robot, effector) * CLOSED_SPAN_MARGIN
    # Place the task in the middle of the workspace, not merely inside it.
    #
    # Admission asks only whether a pose is reachable at all, and a task on that
    # boundary is one the arm can touch rather than work in: the hover above it
    # and the lift after it both land outside. Taking the first size that admits
    # therefore leaves several arms working at their limit, and shrinking blindly
    # pushes others into the hole at the centre. So each candidate size is scored
    # by where it puts the object between the two -- the inner reach and the
    # usable outer reach on its own bearing -- and the one nearest the middle
    # wins. Swept: aiming at 0.5 of that span holds eight of the authored worlds
    # against five for first-fit.

    # Swept: 0.35 and 0.42 both hold nine, 0.5 holds eight, 0.28 six, 0.2 three.
    # Nearer the inner bound than the outer, which fits how these arms fail --
    # they run out of room at full extension long before they run out at the
    # centre.
    target = 0.35
    # The ladder has to grow as well as shrink, and the fit has to survive one
    # object it cannot place.
    #
    # Both limits showed up as arms that were never asked. `uhand2` reaches
    # 178 mm and cannot bring its hand closer than 174 mm to its own frame: a
    # world scaled to its reach puts every object *inside* that hole, and the
    # ladder topped out at 1.0, so nine of its ten pairings were refused
    # `object_inside_reach_hole` without a single attempt. An arm whose hole is
    # a large fraction of its reach needs the world pushed outward, which no
    # shrink can do. And requiring every object to admit let one bad object
    # poison its world: `pallet_cell` holds three at different bearings, no
    # single size took all three, so the world fell back to its authored size
    # and refused all three -- including the two that any of the candidates
    # would have placed comfortably.
    #
    # So candidates now run either side of the reach radius, and are ranked by
    # how many objects they admit first and where they put them second, with
    # the centring measured over the admitted ones so an unplaceable outlier
    # cannot drag the mean. An object that no size admits is still refused
    # honestly; it just no longer takes its neighbours with it.
    #
    # Growth is a last resort, not another rung. `scaled_to` scales the objects
    # with the world, and an object's mass goes as the cube of its span: the
    # 2.1 world hands the arm a block roughly nine times heavier. Offered as a
    # peer of the shrink rungs it wins on centring and loses on physics -- it
    # took the suite from thirty holds to thirteen. So the shrink ladder is
    # tried first and growth runs only when no size at or below the reach
    # radius placed a single object, which is exactly `uhand2`'s case and no
    # one else's.
    # A geometric sweep rather than eight hand-picked rungs.
    #
    # The rungs were a sampling of a continuous quantity, and coarse sampling
    # steps over narrow answers: `uhand2` is a hand on a single wrist joint, its
    # reachable shell runs from 174 mm to 178 mm, and the ladder jumped from
    # one side of that 4 mm band to the other without landing in it. Nine of its
    # ten pairings were refused for want of resolution, not for want of reach.
    # 4% steps resolve any band wider than about a millimetre at these scales,
    # and admission is pure geometry, so the extra candidates cost nothing that
    # matters.
    best, best_key = None, None
    ladder = tuple(1.0 * 1.04 ** -k for k in range(0, 37))
    growth = tuple(1.0 * 1.04 ** k for k in range(1, 20))
    for shrink in ladder + growth:
        if shrink > 1.0 and best is not None:
            break
        candidate = authored.scaled_to(
            nominal * shrink, aperture, holdable
        )
        verdicts = [
            admit_object(robot.manifest, effector, frame, item)
            for item in candidate.objects
        ]
        # An object narrower than the shut jaws is not a task either.
        #
        # `object_too_wide` has a twin nobody wrote. Keeping the objects from
        # growing into one another shrinks them, and shrinking can take them
        # under the closed span: the SO-ARM101 was handed a 14 mm block by jaws
        # that stop 26 mm apart, closed around it without touching, and drove
        # them together through the space where it sat -- opposition on 116 of
        # 1441 steps and excessive penetration on an object weighing under a
        # gram. The fit is already searching world sizes; it should decline the
        # ones that cannot be gripped rather than settle for them.
        admitted = [
            item
            for item, verdict in zip(candidate.objects, verdicts)
            if verdict.admitted and item.span_m >= floor
        ]
        if not admitted:
            continue
        ratios = []
        for item in admitted:
            position = np.asarray(item.position_m, dtype=float)
            azimuth, elevation = frame.bearing_of(position)
            near = frame.inner_reach(azimuth, elevation)
            far = frame.directional_reach(azimuth, elevation)
            span = max(far - near, 1e-6)
            ratios.append(
                (float(np.linalg.norm(position - frame.origin)) - near) / span
            )
        missed = (len(candidate.objects) - len(admitted)) / len(candidate.objects)
        key = abs(sum(ratios) / len(ratios) - target) + _MISS_PENALTY * missed
        if best_key is None or key < best_key:
            best, best_key = candidate, key
    if best is not None:
        return best
    return authored.scaled_to(
        nominal, aperture, holdable
    )
