"""Getting the hand somewhere, rather than getting the joints somewhere.

Everything under this asks one question: where should the joints be. That is a
POSE, and a pose says nothing about how you arrive at it. The body is driven at
the answer by interpolating joint angles, and the route that produces in space is
whatever the arm's geometry happens to sweep -- which is how a reach for a
cabinet handle came to drag the gripper across the cabinet roof, and how a reach
for a point near the base came back as yaw at its limit with the elbow folded,
approaching over the arm's own shoulder.

Neither was a bad pose. Both were fine poses reached by an impossible route, and
no amount of scoring the destination can see that, because the objection is to
the journey.

So this layer asks for the journey. It moves the hand along a STRAIGHT LINE in
space, one short step per frame, solving each step from where the hand already
is. Three things follow, and they are the whole point:

  the route is predictable, because a straight line is;

  each solve is local, so the search cannot answer with a configuration on the
  far side of the workspace -- it is never asked about anywhere far away;

  an obstacle can be tested against the SEGMENT rather than the endpoint, which
  is the only place the question "will this hit something" can honestly be
  asked.

When the line would cross something solid, it goes over the top: up, across,
down. That is a crude plan and a real one, and it is general -- it does not know
what the obstacle is, only where it is.
"""

from __future__ import annotations

import numpy as np

from ..physics.model import Body
from .solver import _reach_for

#: How far along the line to aim each frame, metres. Short enough that the solve
#: stays local and the route stays a line; long enough to make progress.
STEP_M = 0.045
#: Clearance to leave when going over an obstacle, metres.
_OVER_M = 0.07
#: HOW BIG THE HAND ITSELF IS, metres. Every obstacle is inflated by this
#: before the line is tested against it, because what has to miss the obstacle
#: is not the point being steered -- it is the gripper, and the gripper is
#: twelve centimetres of plate and fingers hanging around that point. Avoiding
#: with a point routed the grasp centre neatly past a door panel and swept the
#: fingers straight through it, closing the door the arm had just opened.
_HAND_REACH_M = 0.065
#: How many points along a segment to test against an obstacle. The gripper is
#: about 12 cm long, so sampling every few centimetres cannot step over it.
_SAMPLES = 14


def _inside(point: np.ndarray, box: tuple) -> bool:
    low, high = box
    return bool(np.all(point > low) and np.all(point < high))


def crosses(start: np.ndarray, end: np.ndarray, box: tuple) -> bool:
    """Would a straight line from start to end pass through the box?"""
    for fraction in np.linspace(0.0, 1.0, _SAMPLES):
        if _inside(start + (end - start) * fraction, box):
            return True
    return False


def detour(start: np.ndarray, end: np.ndarray, box: tuple) -> np.ndarray:
    """The next place to head for, when the direct line is blocked.

    Over the top: climb above the obstacle before crossing it, and only descend
    once past. Chosen over going around because these arms work above a bench
    and the space above furniture is the space reliably free.
    """
    _low, high = box
    above = float(high[2]) + _OVER_M
    if start[2] < above - 0.01:
        # Not high enough yet: go straight up first.
        return np.asarray([start[0], start[1], above])
    # High enough: traverse at height, over the far side of the obstacle.
    return np.asarray([end[0], end[1], max(above, float(end[2]))])


def _hits_any(start: np.ndarray, end: np.ndarray, spheres: list) -> np.ndarray | None:
    """The first sphere a straight line runs into, if any.

    A sphere the hand is ALREADY INSIDE gets a different rule: moving away from
    its centre is allowed, moving deeper is not. Skipping such spheres outright
    was the first attempt and it fails in two opposite ways at once. The arm
    finishes an opening motion standing inside its own record of where the thing
    went -- it was holding the thing -- so skipping them left it free to walk
    straight along the panel it was avoiding, pushing the door shut behind it.
    Blocking them instead is a trap with no exit: every direction is refused,
    including the one leading out, and the arm sits still for a minute.

    You may leave what you are inside. You may not go further in.
    """
    for centre, radius in spheres:
        centre = np.asarray(centre)
        if float(np.linalg.norm(start - centre)) < radius:
            if float(np.linalg.norm(end - centre)) < float(
                    np.linalg.norm(start - centre)):
                return centre
            continue
        for fraction in np.linspace(0.0, 1.0, _SAMPLES):
            if float(np.linalg.norm(
                    start + (end - start) * fraction - centre)) < radius:
                return centre
    return None


def _around(start: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """Somewhere clear of a sphere: straight up and over it.

    The same answer as for a box, and for the same reason -- these arms work
    above a bench, and the space above a thing is the space reliably free.
    """
    return np.asarray([start[0], start[1], float(centre[2]) + radius + _OVER_M])


def clear_of(point: np.ndarray, obstacles: list, margin: float = 0.0
             ) -> np.ndarray:
    """The nearest point to ``point`` that has room around it.

    A place to retreat to is a place with ROOM, not a coordinate. The standoff
    this task backs off to was worked out from where the cabinet is, which is
    correct while the cabinet is shut and wrong the moment its door is standing
    open across the front of it -- the point lands beside the panel, and then no
    obstacle radius works: big enough to keep the arm from sweeping the door
    closed is big enough to swallow the very place the arm is retreating to, and
    small enough to leave that place reachable is small enough to let the door
    be shoved shut. There is no number that satisfies both, because the fault is
    not the number.

    So the point moves. It is pushed straight out of whatever it is inside until
    it is clear of everything, which is a few iterations because pushing out of
    one thing can push it into another.
    """
    where = np.asarray(point, dtype=float)
    for _ in range(24):
        worst = None
        for centre, radius in obstacles:
            centre = np.asarray(centre)
            room = float(np.linalg.norm(where - centre)) - (radius + margin)
            if room < 0.0 and (worst is None or room < worst[0]):
                worst = (room, centre, radius)
        if worst is None:
            return where
        _room, centre, radius = worst
        away = where - centre
        size = float(np.linalg.norm(away))
        away = away / size if size > 1e-6 else np.asarray([0.0, -1.0, 0.0])
        where = centre + away * (radius + margin + 0.005)
    return where


def travel_to(body: Body, goal: np.ndarray, square_to, support: float,
              *, keep_out: tuple | None = None, avoid: list | None = None,
              step: float = STEP_M, **kwargs) -> np.ndarray | None:
    """Joint angles for the next short step of a straight run to ``goal``.

    Call it every frame. It re-reads where the hand is, so the line is
    continuously re-aimed and the arm converges rather than tracking a plan made
    once and gone stale.
    """
    here = np.asarray(body.grasp_centre())
    aim_at = np.asarray(goal, dtype=float)

    # A KEEP-OUT YOU ARE DELIBERATELY REACHING INTO IS NOT A KEEP-OUT.
    #
    # These boxes describe furniture by its bounding volume, and a container's
    # bounding volume contains its inside. So every route to something on a
    # shelf "crossed the obstacle", the arm detoured over the top, and it hovered
    # above the cabinet indefinitely trying to get into it. The box is a
    # statement about the walls; wanting to be inside is a statement that the
    # caller knows better, and the caller is the one that chose the goal.
    reaching_in = keep_out is not None and _inside(aim_at, keep_out)
    if keep_out is not None and not reaching_in and crosses(here, aim_at, keep_out):
        aim_at = detour(here, aim_at, keep_out)
    if avoid:
        avoid = [(centre, radius + _HAND_REACH_M) for centre, radius in avoid]
        # Whatever the arm has already moved is now somewhere it was not when
        # the furniture was declared. This is the only obstacle in the system
        # that the machine learned about by doing something.
        struck = _hits_any(here, aim_at, avoid)
        if struck is not None:
            aim_at = _around(here, struck, float(avoid[0][1]))

    to_go = aim_at - here
    span = float(np.linalg.norm(to_go))
    if span < 1e-6:
        return None
    waypoint = here + to_go / span * min(step, span)
    return _reach_for(body, waypoint, square_to, support,
                      keep_out=None if reaching_in else keep_out, **kwargs)
