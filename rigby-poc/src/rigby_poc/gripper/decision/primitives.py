"""Named acts, each carrying the know-how that makes it work.

A numeric goal says WHAT should become true. A primitive says what kind of act
is being attempted, and carries with it the conditions that separate the act
from something that merely scores like it. That distinction is the whole reason
this layer exists.

The humanoid hand learned it the expensive way. Its reach_to aimed the
fingertips at the object's CENTRE, which is inside it, so contact was the
approach succeeding rather than failing, and no control could improve the
number past about 6 cm -- the model called it fourteen more times while the
reading never moved. Its replacement, move_to, does not merely tune the number:
it carries the palm to a point off a FACE, squares the wrist first because
squaring afterwards moves the palm, and is finished only when the hand is
somewhere a grasp could actually happen from.

The gripper had the same hole. palm_to_object_m can be small while the block
sits below the jaws, behind them, or off to one side -- close, and impossible to
grasp -- and nothing in the goal said otherwise. So an approach is not finished
because a distance is small. It is finished when the hand can SEE what it is
about to close on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..physics.model import Body
from ..sensing.gripper_camera import Sensed
from .goals import NumericTarget

#: Standoff for an approach, metres. Far enough out that the jaws are not
#: already through the object, close enough that the closing move is short.
STANDOFF_M = 0.045


def object_in_hand_view(body: Body, seen: Sensed) -> bool:
    """The hand's own camera has the object, right now.

    Not "something saw it once". The room camera can hold a fix on a block the
    hand is pointed away from, which is exactly the state that lets an approach
    report success while the jaws face empty bench.
    """
    return bool(seen.object_seen and getattr(seen, "seen_by", "") == "gripper")


def beam_finds_object(body: Body, seen: Sensed, bench: float = 0.72) -> bool:
    """The rangefinder is returning something nearer than the bench.

    The machine knows where its own hand is and how high the bench is, so it
    can work out how far the beam should travel if nothing is in the way. A
    shorter return means something IS in the way. That is an inference from two
    declared numbers, not a reading of what the beam hit -- a rangefinder
    reports a distance and never an identity.
    """
    eye, _ = body.camera_pose("gripper")
    down = float(body.approach()[2])
    if down > -1e-3:                      # not pointed at the bench at all
        return False
    to_bench = (bench - float(eye[2])) / down
    return bool(seen.range_ahead_m < to_bench - 0.02)


def object_between_jaws(body: Body, seen: Sensed) -> bool:
    """The centre beam finds something INSIDE the span of the pads.

    The distinction the whole interlock exists for is "above the block" versus
    "around the block", and a distance alone does not make it: palm_to_object_m
    was 4 cm in runs where the block sat below the jaws and the gripper closed
    on air. Measured through a working approach, the beam reads the block at
    20 cm while descending and 2.3 cm once the pads straddle it -- the two
    states are nowhere near each other.

    The pads run from 12 mm to 82 mm out from the plate, both from the declared
    finger geometry, so a hit inside that span is a surface the jaws would close
    onto. A hit beyond it is something the hand is merely pointed at.
    """
    from ..body.manifest import spec

    finger = spec()["kinematics"]["finger"]
    root = 0.012
    tip = root + float(finger["length_m"])
    distance, _hit = body.range_ahead()
    from ..physics.model import _BEAM_START_M

    out = distance + _BEAM_START_M
    return bool(root <= out <= tip)


def pads_loaded(body: Body, seen: Sensed) -> bool:
    """Both pads pressing on something that is not the gripper itself."""
    return bool(seen.tip_force_left_n > 0.15 and seen.tip_force_right_n > 0.15)


#: Conditions a goal may require beyond its numbers. Named so a target can
#: carry them, and measured so none of them can be satisfied by argument.
CONDITIONS: dict[str, Callable[[Body, Sensed], bool]] = {
    "object_in_hand_view": object_in_hand_view,
    "beam_finds_object": lambda body, seen: beam_finds_object(body, seen),
    "pads_loaded": pads_loaded,
    "object_between_jaws": object_between_jaws,
}


@dataclass(frozen=True)
class Primitive:
    """One named act: the numbers it drives, and what it insists on."""

    name: str
    describes: str
    build: Callable[[Body, Sensed, float], NumericTarget]


def _open_grip(body: Body, seen: Sensed, now: float) -> NumericTarget:
    return NumericTarget(
        metric="grip_tip_spread_m", value=0.086, set_at_s=now,
        using=("gripper",))


def _close_grip(body: Body, seen: Sensed, now: float) -> NumericTarget:
    # Closing is driven to the floor and STOPPED BY THE OBJECT, not by a
    # width the planner has to estimate. The jaws cannot pass through each
    # other any more, so a close on empty air ends at 7 mm and reports no load
    # -- which is how "closed on nothing" tells itself apart from "holding".
    return NumericTarget(
        metric="grip_tip_spread_m", value=0.007, set_at_s=now,
        using=("gripper",), requires=("object_between_jaws", "pads_loaded"))


def _hold(body: Body, seen: Sensed, now: float) -> NumericTarget:
    # Keep what is already held. The width asked for is the width now, so the
    # search has nothing to do and the squeeze rule keeps the grip on.
    return NumericTarget(
        metric="grip_tip_spread_m", value=float(body.opening()), set_at_s=now,
        using=("gripper",), requires=("pads_loaded",))


def _move_to(body: Body, seen: Sensed, now: float) -> NumericTarget:
    # THE APPROACH IS NOT DONE BECAUSE THE DISTANCE IS SMALL. It is done when
    # the hand is somewhere a grasp could happen from, and the test for that is
    # that the hand's own camera can see the thing it is about to close on.
    return NumericTarget(
        metric="palm_to_object_m", value=STANDOFF_M, set_at_s=now,
        also=(("palm_facing", 1.0, 0.6),
              ("hand_pointing_down", 1.0, 0.4)),
        using=("move",),
        requires=("object_in_hand_view",))


# open_grip, close_grip and hold are GONE. Each built a grip_tip_spread_m
# target, and that metric stopped being drivable when the jaws became a state --
# so they were three more names for a thing that no longer did anything. The
# planner chose them, nothing happened, and it moved on believing it had closed.
def _move_to(body: Body, seen: Sensed, now: float) -> NumericTarget:
    # THE APPROACH IS NOT DONE BECAUSE THE DISTANCE IS SMALL. It is done when
    # the hand is somewhere a grasp could happen from, and the test for that is
    # that the hand's own camera can see the thing it is about to close on.
    return NumericTarget(
        metric="palm_to_object_m", value=STANDOFF_M, set_at_s=now,
        also=(("palm_facing", 1.0, 0.6),
              ("hand_pointing_down", 1.0, 0.4)),
        using=("move",),
        requires=("object_in_hand_view",))


# open_grip, close_grip and hold are GONE. Each built a grip_tip_spread_m
# target, and that metric stopped being drivable when the jaws became a state --
# so they were three more names for a thing that no longer did anything. The
# planner chose them, nothing happened, and it moved on believing it had closed.
def _carry_to_bin(body: Body, seen: Sensed, now: float) -> NumericTarget:
    """Over the bin AND above its rim, which are not two separate jobs.

    Measured, carrying a block the machine had just picked up: driven one at a
    time, each goal destroys the other. At t=6.0 the block was lined up with the
    bin in x and 24 cm BELOW the rim, so crossing would have struck the wall; at
    t=9.6 it was above the rim and back over the start. Both halves, never
    together.

    So the height is a THRESHOLD held while the position is driven -- ">= 0.05"
    is "keep it clear of the rim", not "put it exactly 5 cm up", and once clear
    there is nothing left to fix and the search can spend itself on getting
    across.
    """
    return NumericTarget(
        metric="object_over_target_m", value=0.03, set_at_s=now, compare="<=",
        also=(("object_above_rim_m", 0.05, 0.8, ">="),),
        using=("move",), requires=("pads_loaded",))


def _lower_into_bin(body: Body, seen: Sensed, now: float) -> NumericTarget:
    """Down into the bin, staying over it while descending."""
    return NumericTarget(
        metric="object_above_rim_m", value=0.0, set_at_s=now, compare="<=",
        also=(("object_over_target_m", 0.04, 0.9, "<="),),
        using=("move",), requires=("pads_loaded",))


# carry_to_bin and lower_into_bin are GONE, deliberately.
#
# They worked by encoding the answer: drive object_over_target_m while holding
# object_above_rim_m above the rim. That is a recipe for THIS bin, and a system
# needing one per destination is not planning, it is choosing from a menu
# somebody else wrote. The general fact -- that two numbers can be contested and
# one must be protected while the other is driven -- is already in the prompt,
# and `also` is the field for saying so.
#
# move_to survives because what it encodes is not about the bin or the task: a
# distance to an object can be small while the object is below the jaws, and no
# reading says otherwise. That is a fact about this BODY, and it holds for
# anything the hand ever reaches for.
PRIMITIVES: dict[str, Primitive] = {
    "move_to": Primitive(
        "move_to",
        "carry the hand to a standoff in front of the object, square to it; "
        "finished only when the hand camera can see the object",
        _move_to),
}


def build(name: str, body: Body, seen: Sensed, now: float
          ) -> NumericTarget | None:
    """The target a named primitive stands for, or None if it is not one."""
    primitive = PRIMITIVES.get(name)
    return None if primitive is None else primitive.build(body, seen, now)


def catalogue() -> str:
    """The primitives and what each one means, for the prompt."""
    return chr(10).join(f"  {p.name:<11} {p.describes}"
                        for p in PRIMITIVES.values())

# ---------------------------------------------------------------------------
# SUPER PRIMITIVES: a sequence that locks when it is chosen.
#
# A primitive answers "what should be true". A super primitive answers "what
# should happen next, and for how long" -- and once chosen it is COMMITTED. The
# planner is not asked again while it runs, because the moments in the middle of
# an act are not decision points: mid-pursuit the answer is always "keep going",
# and a call spent confirming that is a call not spent on the moment that
# mattered.
#
# Every stage carries a time budget, and the budget is derived from the declared
# rate limits wherever the travel is known -- opening the jaws from the floor to
# their widest is 79 mm of aperture at 70 mm/s, so it is 1.1 s and not a guess.
# Where the travel depends on where the arm happens to be, the budget is stated
# as a ceiling rather than a promise, and a stage that arrives early hands the
# rest of its time back.
#
# Stages run their terms in PARALLEL. Reaching for the block while opening the
# jaws is one stage with terms from both, not two stages in a row: the packages
# are independent, and doing them in sequence wastes the time of whichever is
# faster.
# ---------------------------------------------------------------------------


def _aperture_seconds(fromed: float, to: float) -> float:
    """How long a jaw move takes, from the rate the manifest declares."""
    from ..body.manifest import spec

    rate = float(spec().get("rate_limits", {}).get("finger_m_per_s", 0.07))
    return abs(to - fromed) / max(rate, 1e-6)


@dataclass(frozen=True)
class Stage:
    """One committed step: what to drive, and how long it may take."""

    name: str
    build: Callable[[Body, Sensed, float], NumericTarget]
    seconds: float


@dataclass(frozen=True)
class Script:
    """A named act, as an ordered list of stages."""

    name: str
    describes: str
    stages: tuple[Stage, ...]

    @property
    def seconds(self) -> float:
        return round(sum(stage.seconds for stage in self.stages), 2)


def _both(*names: str) -> Callable[[Body, Sensed, float], NumericTarget]:
    """One target pursuing several primitives AT ONCE.

    The first named primitive supplies the metric being driven; the rest fold
    in as weighted terms and their packages are unioned, so the search moves the
    arm and the jaws in the same frames instead of taking turns.
    """
    def build(body: Body, seen: Sensed, now: float) -> NumericTarget:
        first = PRIMITIVES[names[0]].build(body, seen, now)
        also = list(first.also)
        using = list(first.using)
        requires = list(first.requires)
        for other in names[1:]:
            made = PRIMITIVES[other].build(body, seen, now)
            also.append((made.metric, made.value, 0.7))
            also.extend(made.also)
            using += [u for u in made.using if u not in using]
            requires += [r for r in made.requires if r not in requires]
        return NumericTarget(
            metric=first.metric, value=first.value, set_at_s=now,
            also=tuple(also), using=tuple(using), requires=tuple(requires))
    return build


def _lift_by(rise_m: float) -> Callable[[Body, Sensed, float], NumericTarget]:
    def build(body: Body, seen: Sensed, now: float) -> NumericTarget:
        return NumericTarget(
            metric="hand_z_m", value=float(body.grasp_centre()[2]) + rise_m,
            set_at_s=now, also=(("hand_pointing_down", 1.0, 0.4),),
            using=("move",), requires=("pads_loaded",))
    return build


SCRIPTS: dict[str, Script] = {
    "open_claw": Script(
        "open_claw", "open the jaws all the way",
        (Stage("open_grip", _open_grip, round(_aperture_seconds(0.007, 0.086), 2)),)),
    "close_claw": Script(
        "close_claw", "close the jaws onto whatever is between them",
        (Stage("close_grip", _close_grip, round(_aperture_seconds(0.086, 0.007), 2)),)),
    "reach_and_open": Script(
        "reach_and_open",
        "carry the hand to the object AND open the jaws at the same time",
        (Stage("move_to+open_grip", _both("move_to", "open_grip"), 6.0),)),
    "pick_up": Script(
        "pick_up",
        "the whole grasp: reach with the jaws opening, close on the object, "
        "then lift it clear",
        (Stage("move_to+open_grip", _both("move_to", "open_grip"), 6.0),
         Stage("close_grip", _close_grip, 2.0),
         Stage("lift", _lift_by(0.12), 4.0))),
}


def script(name: str) -> Script | None:
    return SCRIPTS.get(name)


def scripts_catalogue() -> str:
    """The super primitives, with the time each one commits to."""
    return chr(10).join(
        f"  {s.name:<15} ~{s.seconds:4.1f}s  {s.describes}"
        for s in SCRIPTS.values())


# ---------------------------------------------------------------------------
# DIRECT ACTS: no search, no goal, just a pose.
#
# Greedy earns its keep when the joints that serve a number are not obvious --
# "get the palm 5 cm from the block" is a real search over four joints and a
# lattice of amplitudes. "Open the jaws all the way" is not. The answer is the
# joint limit, it is written in the manifest, and running a search to discover
# it costs a decision, several hundred kinematics probes, and the chance of
# coming back with "hold" because the commanded pose already satisfied the
# number while the body had not caught up.
#
# So these bypass the search entirely. The planner names the act, the joints go
# to the stated pose at the declared rate, and it is finished when they arrive.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Direct:
    """An act whose pose is known without searching for it."""

    name: str
    describes: str
    #: The commanded joint vector, given the body. Only the joints it names are
    #: changed; the rest are left where the controller already had them.
    pose: Callable[[Body, "np.ndarray"], "np.ndarray"]


def _jaws_to(where: str):
    def pose(body: Body, commanded):
        import numpy as np

        out = np.asarray(commanded, dtype=float).copy()
        low, high = body.model.jnt_range[body.model.joint("finger_left").id]
        out[4] = out[5] = float(high if where == "open" else low)
        return out
    return pose


DIRECT: dict[str, Direct] = {
    "open_jaws": Direct(
        "open_jaws",
        "drive the jaws straight to their widest, 86 mm. No search: the answer "
        "is the joint limit.",
        _jaws_to("open")),
    "close_jaws": Direct(
        "close_jaws",
        "drive the jaws straight shut, to the 7 mm floor. They stop on "
        "whatever is between them.",
        _jaws_to("close")),
}


def direct(name: str) -> Direct | None:
    return DIRECT.get(name)


def direct_catalogue() -> str:
    return chr(10).join(f"  {d.name:<12} {d.describes}" for d in DIRECT.values())
