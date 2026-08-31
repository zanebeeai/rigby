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
    """The first sphere a straight line runs into, if any."""
    for fraction in np.linspace(0.0, 1.0, _SAMPLES):
        point = start + (end - start) * fraction
        for centre, radius in spheres:
            if float(np.linalg.norm(point - centre)) < radius:
                return np.asarray(centre)
    return None


def _around(start: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """Somewhere clear of a sphere: straight up and over it.

    The same answer as for a box, and for the same reason -- these arms work
    above a bench, and the space above a thing is the space reliably free.
    """
    return np.asarray([start[0], start[1], float(centre[2]) + radius + _OVER_M])


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
