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

    # EVERY JOINT, IN DEGREES, ASKABLE. Without these the planner could only
    # ask for numbers somebody had thought to define -- and when it wanted the
    # arm a little higher, or this segment folded a little further, there was
    # no way to say so and no control that could be found to do it. A joint
    # angle is the simplest metric there is: always readable, moved by exactly
    # one part, and never contested by anything else.
    #
    # This is what lets the planner act without being handed the actuators. It
    # still names a number and the search still drives it; the difference is
    # that the vocabulary no longer runs out.
    # EACH SEGMENT'S FAR END, AS A POINT. A joint angle says how much a hinge
    # has turned, which is exact and says nothing about where the arm ended up.
    # The far end of a segment is a place, it moves on a circle around that
    # segment's own pivot, and "put this end here" is a thing that can be
    # pictured. Every one of these is forward kinematics from the encoders --
    # the machine's own body, not the world.
    "segment_1_tip_x_m": lambda body, seen: float(body.body_at("link2")[0]),
    "segment_1_tip_y_m": lambda body, seen: float(body.body_at("link2")[1]),
    "segment_1_tip_z_m": lambda body, seen: float(body.body_at("link2")[2]),
    "segment_2_tip_x_m": lambda body, seen: float(body.body_at("link3")[0]),
    "segment_2_tip_y_m": lambda body, seen: float(body.body_at("link3")[1]),
    "segment_2_tip_z_m": lambda body, seen: float(body.body_at("link3")[2]),
    "segment_3_tip_x_m": lambda body, seen: float(body.body_at("plate")[0]),
    "segment_3_tip_y_m": lambda body, seen: float(body.body_at("plate")[1]),
    "segment_3_tip_z_m": lambda body, seen: float(body.body_at("plate")[2]),
    "base_deg": lambda body, seen: float(np.degrees(body.q()[0])),
    "segment_1_deg": lambda body, seen: float(np.degrees(body.q()[1])),
    "segment_2_deg": lambda body, seen: float(np.degrees(body.q()[2])),
    "segment_3_deg": lambda body, seen: float(np.degrees(body.q()[3])),
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
    # WHETHER THE JAWS FACE THE THING THEY ARE MEANT TO CLOSE ON. A hand four
    # centimetres away and pointing across the block reads well on every
    # distance in this list, and cannot grasp anything.
    # Zero once the object is closer than 5 cm -- in the hand, or as good as --
    # for the same reason the aim block reports none there: the direction to a
    # thing you are holding is noise, and a goal driven by noise chases it.
    "pointing_at_object": lambda body, seen: (
        0.0 if (seen.object_at is None or float(np.linalg.norm(
            np.asarray(seen.object_at) - body.grasp_centre())) < 0.05)
        else float(np.clip(np.dot(
            body.approach(),
            (np.asarray(seen.object_at) - body.grasp_centre())
            / max(float(np.linalg.norm(
                np.asarray(seen.object_at) - body.grasp_centre())), 1e-6)),
            -1.0, 1.0))),
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
    # Degrees, so the floor is a couple of degrees rather than centimetres.
    "segment_1_tip_x_m": 0.02, "segment_1_tip_y_m": 0.02, "segment_1_tip_z_m": 0.02,
    "segment_2_tip_x_m": 0.02, "segment_2_tip_y_m": 0.02, "segment_2_tip_z_m": 0.02,
    "segment_3_tip_x_m": 0.02, "segment_3_tip_y_m": 0.02, "segment_3_tip_z_m": 0.02,
    "base_deg": 2.0,
    "segment_1_deg": 2.0,
    "segment_2_deg": 2.0,
    "segment_3_deg": 2.0,
    "palm_to_object_m": 0.02, "object_in_grasp_m": 0.02,
    "palm_facing": 0.25,
    "pointing_at_object": 0.2,
    "object_over_target_m": 0.03, "object_above_rim_m": 0.03,
}

#: Which numbers mean nothing until the object has been seen. The planner is
#: told this explicitly rather than left to discover it by watching a metric
#: refuse to move.

#: What each number means and which way is better. The planner had the names
#: and the values and nothing that said 1.0 was the good end of
#: pointing_at_object -- so it asked for 0.7 while sitting at 1.0, the search
#: obediently made the aim worse, and the hand swung off a block that was
#: already between the jaws. A number without a direction is half a fact.
DESCRIBES: dict[str, str] = {
    "hand_x_m": "where the hand is across the bench, metres. A place, not a "
                "score: aim for the value you want.",
    "hand_y_m": "where the hand is in depth, metres. A place.",
    "hand_z_m": "how high the hand is, metres. A place.",
    "hand_pointing_down": "how squarely the palm faces the bench. 1.0 is "
                          "straight down, 0.0 is level, negative is upward. "
                          "HIGHER IS BETTER for reaching down at something.",
    "grip_tip_spread_m": "the gap between the pads, metres. 0.007 is shut, "
                         "0.086 is as wide as they go. A width, not a score.",
    "range_ahead_m": "distance to the first surface along the grasp axis, "
                     "metres. Smaller means nearer.",
    "pointing_at_object": "whether the jaws face the block. 1.0 is straight at "
                          "it, 0.0 is square across it, negative is away. "
                          "HIGHER IS BETTER, and 1.0 is perfect -- asking for "
                          "less than you already have asks the arm to aim "
                          "worse.",
    "palm_facing": "whether the palm is square to the face it would grasp. "
                   "1.0 is flat on, 0.0 is edge-on. HIGHER IS BETTER.",
    "palm_to_object_m": "distance from the palm to the block's surface, "
                        "metres. LOWER IS BETTER, 0 is touching.",
    "object_in_grasp_m": "how far the block is from the line between the pads, "
                         "metres. LOWER IS BETTER.",
    "object_over_target_m": "how far the block is from being over the bin, "
                            "metres, measured flat. LOWER IS BETTER.",
    "object_above_rim_m": "how far the block is above the bin rim, metres. "
                          "Positive is clear of the rim.",
    "base_deg": "the turntable angle, degrees. A place.",
    "segment_1_deg": "the first hinge, degrees. A place.",
    "segment_2_deg": "the second hinge, degrees. A place.",
    "segment_3_deg": "the third hinge, degrees. A place.",
}
for _seg in ("segment_1", "segment_2", "segment_3"):
    for _ax in ("x", "y", "z"):
        DESCRIBES[f"{_seg}_tip_{_ax}_m"] = (
            f"where {_seg}'s far end is, {_ax} in metres. A place. Only "
            "segment_3's end carries the gripper.")

#: Metrics where 1.0 is perfect and less is worse, rather than a distance to be
#: driven to zero. Without this the planner asked for pointing_at_object = 0.7
#: while its aim was already 1.0 -- a reasonable-looking number that is a
#: request to aim WORSE. The search obliged, the hand swung off a block that was
#: already between the jaws, and the next four decisions were spent trying to
#: get back to the position it had just given away.
BEST_AT_ONE = ("pointing_at_object", "palm_facing", "hand_pointing_down")

NEEDS_SIGHT = ("pointing_at_object", "palm_to_object_m", "object_in_grasp_m", "palm_facing",
               "object_over_target_m", "object_above_rim_m")

#: Readable but not askable: outcomes and states, not handles. Naming an outcome
#: as a target is naming the goal as its own method.
OUTCOMES = ("grip_tip_spread_m",
            "holding", "object_seen",
            "tip_force_left_n", "tip_force_right_n",
            "object_in_hand_view", "beam_finds_object",
            "object_between_jaws")


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
    # Readable, not askable: whether the HAND can see the object, as opposed to
    # whether anything can. An approach that is close but pointed elsewhere is
    # the failure this separates out.
    from .primitives import beam_finds_object, object_in_hand_view

    from .primitives import object_between_jaws

    # READABLE, NOT ASKABLE. The jaws are a state now and the search does not
    # touch the fingers, so a goal naming this metric drives nothing at all.
    # Leaving it on the askable list meant the planner reached for the lever it
    # already knew: it asked for grip_tip_spread_m = 0.007 twice, in the step
    # called "close the gripper", and the jaws stayed open for the whole run
    # while the pads brushed the block at 1 N. A control that does nothing must
    # not be offered.
    out["grip_tip_spread_m"] = round(float(body.opening()), 5)
    out["object_between_jaws"] = float(object_between_jaws(body, seen))
    out["object_in_hand_view"] = float(object_in_hand_view(body, seen))
    out["beam_finds_object"] = float(beam_finds_object(body, seen))
    out["tip_force_left_n"] = round(float(seen.tip_force_left_n), 3)
    out["tip_force_right_n"] = round(float(seen.tip_force_right_n), 3)
    # object_in_target IS NOT HERE, AND THAT IS THE POINT. It is computed from
    # body.block() -- MuJoCo's true block pose -- so it was the one piece of
    # simulator truth in a payload that is otherwise cameras, encoders and load
    # cells. Its own docstring said "NOT a control input ... keeping the scoring
    # honest means keeping it OUT of the loop", and it was in the loop anyway:
    # the grader was visible to the thing being graded.
    #
    # Whether the object is where the task wanted it is now the MODEL'S
    # judgement, made from the overhead camera, and it reports that judgement
    # as `placed`. object_in_target still exists and still reads the simulator,
    # because something has to score the run -- it is just no longer allowed to
    # tell the model the answer.
    out["holding"] = float(seen.holding())
    out["object_seen"] = float(seen.object_seen)
    return out


@dataclass
class NumericTarget:
    """A number the planner wants, and which controls may pursue it."""

    metric: str
    value: float
    set_at_s: float = 0.0
    #: HOW THE VALUE IS TO BE MET: "==", ">=" or "<=".
    #:
    #: Equality is the wrong shape for most of these. "Aim at the block" is not
    #: "aim at exactly 0.9 of straight-on" -- it is "at least this square", and
    #: once you are squarer than that there is nothing left to fix. Driving an
    #: equality target past its value makes the error rise again, which is how
    #: a goal that is going well starts reading as a goal going wrong.
    compare: str = "=="
    #: CONDITIONS THE NUMBERS CANNOT EXPRESS. A distance can be small while the
    #: block sits below the jaws, behind them, or off to one side -- close, and
    #: impossible to grasp. Named conditions from decision.primitives.CONDITIONS
    #: are checked alongside the error, so arriving at the number is not the
    #: same as arriving.
    requires: tuple[str, ...] = ()
    #: Further (metric, value, weight) pursued at the same time.
    also: tuple[tuple[str, float, float], ...] = ()
    #: Which controls the search may touch. Naming these is half the
    #: instruction: "bring the jaws to the block" is a statement about a
    #: distance AND about which part of the body is meant to close it.
    using: tuple[str, ...] = ()

    def terms(self) -> tuple[tuple[str, float, float], ...]:
        return ((self.metric, self.value, 1.0),) + tuple(
            (a[0], a[1], a[2] if len(a) > 2 else 0.5) for a in self.also)

    def compare_for(self, metric: str) -> str:
        """The comparison this target uses for one of its numbers."""
        if metric == self.metric:
            return self.compare
        for entry in self.also:
            if entry[0] == metric and len(entry) > 3:
                return str(entry[3])
        return "=="

    @staticmethod
    def _short(now: float, wanted: float, how: str) -> float:
        """How far short of the requirement, which is 0 once it is met."""
        if how == ">=":
            return max(0.0, wanted - now)
        if how == "<=":
            return max(0.0, now - wanted)
        return abs(now - wanted)

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
            gap = self._short(float(read(body, seen)), float(wanted),
                              self.compare_for(metric))
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
            # THE SAME SHORTFALL THE ERROR USES. If the scale measured a
            # difference while the error measured a shortfall, a threshold that
            # was already satisfied would be scaled by how far PAST it the
            # number sat -- which is a large number for a goal with nothing left
            # to do.
            spans[metric] = max(
                self._short(float(read(body, seen)), float(wanted),
                            self.compare_for(metric)),
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
        if not self.satisfied(body, seen):
            return False
        return self.error(body, seen, spans or self.spans(body, seen)) <= within

    def satisfied(self, body: Body, seen: Sensed) -> bool:
        """Whether every named condition holds. Vacuously true if there are none."""
        from .primitives import CONDITIONS

        for name in self.requires:
            check = CONDITIONS.get(name)
            if check is not None and not check(body, seen):
                return False
        return True

    def unmet(self, body: Body, seen: Sensed) -> list[str]:
        """Which conditions are not holding, for the planner to be told."""
        from .primitives import CONDITIONS

        return [name for name in self.requires
                if (CONDITIONS.get(name) is not None
                    and not CONDITIONS[name](body, seen))]
