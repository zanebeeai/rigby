"""What the planner may ask for: numbers it wants to become true.

The planner does not choose motions. It names a NUMBER -- a metric the body can
read and the value it should have -- and which controls the search may touch to
get there. Deciding what should become true is a judgement about the task;
finding the joint angles that make it true is a search. Models are good at the
first and bad at the second, and a greedy probe is the reverse.

THE VOCABULARY HAS TO COVER THE WHOLE TASK, INCLUDING THE START.

The first version of this offered only object-relative numbers -- how far the
hand is from the object, whether the object is between the pads. Every one of
them is unreadable until something has been seen, and the machine begins
blind. So the planner had nothing it could ask for, the greedy search correctly
reported that no control moved the number, and the arm sat still for
twenty-six seconds with its error frozen at exactly its starting value. The
vocabulary could not express the first thing the task requires, which is to go
and look.

So there are three kinds of number here:

  WHERE THE CAMERA IS POINTED -- view_x_m, view_y_m. Readable always, because
  it is a fact about the arm and the bench, not about anything found. This is
  what makes looking directable: "aim at this patch" is a goal a blind machine
  can pursue.

  WHERE THE HAND IS -- hand_x_m, hand_y_m, hand_z_m. Also always readable.
  Lets the planner send the hand to a place it knows about -- over the bin,
  clear of the cabinet -- without an object being involved at all.

  HOW THE HAND RELATES TO WHAT IT HAS FOUND -- the rest. Readable only once
  something has been seen, and worth nothing before.

Naming ONE number is usually wrong. Every metric here is contested by controls
that improve one and wreck another, so a search told to care about exactly one
will pay any price in the rest -- it will satisfy a grip shape while drifting
out of reach, or centre the object while opening the jaws past it. Which numbers
matter TOGETHER is a judgement about the task, not something the search can read
off the geometry, so the planner names the set and weights it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics.model import Body
from ..sensing.gripper_camera import Sensed
from . import pick_and_place as task


def _eye(body: Body) -> tuple[np.ndarray, float]:
    """Where the camera is, and how steeply it looks down (0 level, 1 straight)."""
    eye, orient = body.camera_pose()
    return np.asarray(eye), float(np.clip(-orient[2, 2], -1.0, 1.0))


#: Every number the planner may ask for. Keyed by the names the manifest
#: declares, so what the model is told it can steer by and what the search can
#: actually compute are the same list.
#:
#: LOOKING IS EXPRESSED AS A POSE, NOT AS A RAY INTERSECTION. The first version
#: offered "where the camera's axis meets the bench", which is the natural way
#: to say it and a badly conditioned thing to search on: the intersection is a
#: violent function of the pose, undefined whenever the camera is level or
#: above, and its cheapest descent is to translate the whole arm backwards
#: rather than tilt it. The arm drove itself down to eight centimetres above
#: the bench with the camera still level, the number improving all the way, and
#: then jammed. Where the camera IS and how steeply it LOOKS DOWN are both
#: smooth in the joints and independently reachable, and together they say
#: everything "look at that patch" needs to say.
READABLE = {
    # Always readable: facts about the arm, not about anything found.
    #
    # There are no separate numbers for the camera. It is bolted to the plate,
    # so where the hand is IS where the eye is, and pointing the hand down is
    # what points the camera at the bench. Offering both was two names for one
    # pose and invited a plan that moved the eye somewhere the hand could not
    # follow.
    "hand_x_m": lambda body, seen: float(body.grasp_centre()[0]),
    "hand_y_m": lambda body, seen: float(body.grasp_centre()[1]),
    "hand_z_m": lambda body, seen: float(body.grasp_centre()[2]),
    "hand_pointing_down": lambda body, seen: float(
        np.clip(-body.approach()[2], -1.0, 1.0)),
    "grip_tip_spread_m": lambda body, seen: float(body.opening()),
    # THE RANGEFINDER IS ASKABLE, and it is the only always-readable number
    # that says anything about HEIGHT. A camera looking straight down cannot
    # tell how far below the bench is; that is why the jaws were opened wide
    # and closed until they stalled. "Get the hand to 5 cm off whatever is in
    # front of it" is now a thing the planner can simply say.
    "range_ahead_m": lambda body, seen: float(seen.range_ahead_m),
    # Readable only once something has been found.
    "palm_to_object_m": task.palm_to_object_m,
    "object_in_grasp_m": task.object_in_grasp_m,
    "palm_facing": task.palm_facing,
    "object_over_target_m": task.object_over_target_m,
    "object_above_rim_m": task.object_above_rim_m,
}

#: The smallest a term's scale may be, per metric. Scaling by "how far this
#: started from its goal" breaks for a term that starts already satisfied: the
#: gap is zero, the scale collapses to the clamp, and that term is then divided
#: by a thousandth and outweighs everything else by three orders of magnitude.
#: The search will not move a millimetre in a direction that is already right,
#: which freezes the whole arm -- measured, an error pinned at exactly its
#: starting value for twenty-six seconds while every probe scored worse.
#:
#: So the floor is a real quantity per metric: how much of this number one
#: cares about at all. Below it, being closer is not an improvement worth
#: paying for.
_FLOOR = {
    "hand_pointing_down": 0.15,
    "hand_x_m": 0.03, "hand_y_m": 0.03, "hand_z_m": 0.03,
    "grip_tip_spread_m": 0.01,
    "range_ahead_m": 0.02,
    "palm_to_object_m": 0.02, "object_in_grasp_m": 0.02,
    "palm_facing": 0.25,
    "object_over_target_m": 0.03, "object_above_rim_m": 0.03,
}

#: Which numbers mean nothing until the object has been seen. The planner is
#: told this explicitly rather than left to discover it by watching a metric
#: refuse to move.
NEEDS_SIGHT = ("palm_to_object_m", "object_in_grasp_m", "palm_facing",
               "object_over_target_m", "object_above_rim_m")

#: Readable but not askable: outcomes and states, not handles. Naming an outcome
#: as a target is naming the goal as its own method.
OUTCOMES = ("object_in_target", "holding", "object_seen",
            "tip_force_left_n", "tip_force_right_n")


def readable(body: Body, seen: Sensed) -> dict[str, float]:
    """Every number, now, for the planner to read."""
    out = {}
    for name, read in READABLE.items():
        if name in NEEDS_SIGHT and seen.object_at is None:
            continue
        out[name] = round(float(read(body, seen)), 4)
    # READINGS, not goals. The pads report what they feel; there is no control
    # that pursues a force, because the squeeze is a fixed hold rather than a
    # commanded number. Offering them as targets would be offering something
    # the search cannot chase.
    out["tip_force_left_n"] = round(float(seen.tip_force_left_n), 3)
    out["tip_force_right_n"] = round(float(seen.tip_force_right_n), 3)
    out["object_in_target"] = float(task.object_in_target(body))
    out["holding"] = float(seen.holding())
    out["object_seen"] = float(seen.object_seen)
    return out


@dataclass
class NumericTarget:
    """A number the planner wants, and which controls may pursue it."""

    metric: str
    value: float
    set_at_s: float = 0.0
    #: Further (metric, value, weight) pursued at the same time.
    also: tuple[tuple[str, float, float], ...] = ()
    #: Which controls the search may touch. Naming these is half the
    #: instruction: "bring the jaws to the block" is a statement about a
    #: distance AND about which part of the body is meant to close it.
    using: tuple[str, ...] = ()

    def terms(self) -> tuple[tuple[str, float, float], ...]:
        return ((self.metric, self.value, 1.0),) + tuple(self.also)

    def usable(self, seen: Sensed) -> bool:
        """Whether every number in this target can currently be read at all."""
        return not (seen.object_at is None
                    and any(m in NEEDS_SIGHT for m, _v, _w in self.terms()))

    def error(self, body: Body, seen: Sensed,
              scale: dict[str, float] | None = None) -> float:
        """Distance from the whole set, weighted and scaled.

        Each term is divided by how far that metric started from its goal, so a
        distance in metres and a dot product in -1..1 contribute comparably.
        Without it the metre-scale terms vanish beside the others and the search
        quietly optimises only the unit-scale ones -- which reads, from outside,
        as a hand that will not approach but is very keen to stay square.
        """
        total = 0.0
        for metric, wanted, weight in self.terms():
            read = READABLE.get(metric)
            if read is None:
                continue
            if metric in NEEDS_SIGHT and seen.object_at is None:
                continue
            gap = abs(float(read(body, seen)) - float(wanted))
            span = max(float((scale or {}).get(metric, 1.0)),
                       _FLOOR.get(metric, 0.05))
            total += weight * gap / span
        return total

    def spans(self, body: Body, seen: Sensed) -> dict[str, float]:
        """How far each term is from its goal right now, captured once."""
        spans = {}
        for metric, wanted, _weight in self.terms():
            read = READABLE.get(metric)
            if read is None or (metric in NEEDS_SIGHT and seen.object_at is None):
                continue
            spans[metric] = max(abs(float(read(body, seen)) - float(wanted)),
                                _FLOOR.get(metric, 0.05))
        return spans

    def reached(self, body: Body, seen: Sensed,
                spans: dict[str, float] | None = None,
                within: float = 0.12) -> bool:
        """Whether the goal is met, measured against the scales it was SET with.

        Recomputing the scales here measures every term against its own current
        gap, which is 1.0 by construction -- a yardstick that moves with the
        thing it measures, and arrival that can never be reached or, once a gap
        falls under its floor, is reached instantly. The scales are captured
        when the target is set precisely so that progress means something, and
        this is the one place that most needs them.
        """
        return self.error(body, seen, spans or self.spans(body, seen)) <= within
