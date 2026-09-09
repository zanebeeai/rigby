"""The model that decides what should become true, from two camera images.

It never names a motion. It names a NUMBER it wants and which controls may
pursue it, and the greedy search downstairs finds the joint angles. That split
is the point of the whole system: judging what ought to be true next is a
question about the task, and finding the pose that makes it true is a search.

TWO CAMERAS, and they answer different questions. The corner camera shows where
everything is -- the arm, the block, the bin, and their relation, which a camera
buried in the workspace cannot see. The gripper camera shows what the hand is
actually pointed at, in detail, and is the image the perception layer segments.
Neither is sufficient: a plan made only from the corner view cannot tell whether
the jaws are around the block, and one made only from the wrist view does not
know the bin exists.

Everything numeric it reads comes through the same Sensed the controller uses.
It is not shown the object's true pose, because the machine does not have it.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..body.manifest import spec
from ..physics.model import JOINTS, Body
from ..sensing.gripper_camera import Sensed
from ..sensing.landmarks import (CORNERS, GRIP_H, GRIP_W, PICK_H,
                                 PICK_W, World)
from . import pick_and_place as task
from .goals import (BEST_AT_ONE, DESCRIBES, _FLOOR, NEEDS_SIGHT, OUTCOMES,
                    READABLE,
                    NumericTarget, readable)
from .greedy import changers, contests
from .grip import STATES as JAW_STATES, catalogue as jaw_catalogue
from .primitives import (
    DIRECT, PRIMITIVES, build as build_primitive, catalogue,
    direct as build_direct, direct_catalogue,
)

#: Room for the reply AND for the model's own reasoning, which is billed from
#: the same budget. At 400 a reasoning model spent the lot thinking and returned
#: empty content with finish_reason "length" -- and `content or "{}"` turned
#: that into an empty decision that looked like a refusal. Twenty-six of one
#: run's forty calls went that way, invisibly, because nothing recorded the
#: finish reason. Measured with a small payload the reasoning alone was 75-121
#: tokens; with two images and the full situation it is plainly more.
_REPLY_TOKENS = 3000
#: How long before the scene may be identified again. Re-looking costs a model
#: call, so it is rationed: an item can go missing for a second without the run
#: being in trouble, and re-identifying every call would spend the whole budget
#: on perception and leave nothing for deciding.
_RELOOK_AFTER_S = 6.0
#: How many times an instruction may be expanded before the run gives up on
#: having a plan and decides from the sentence alone.
_EXPAND_TRIES = 2
#: How long to wait before asking again when there is no goal at all. Shorter
#: than stuck_after_s, because having nothing to chase is more urgent than
#: chasing something slowly -- and far longer than a frame.
_RETRY_AFTER_S = 1.5
#: Pressing harder than this with a part that should carry no load means the
#: arm is against something, not merely slow. A clean carry measured 0.00 N
#: across 887 frames; the jams ran 52-157 N.
_PINNED_N = 5.0
#: Ways of saying the task is over. The field is `done`; the rest are spellings
#: of the same claim, and refusing an obviously-meant answer on spelling has
#: cost this project whole runs before.
_DONE_KEYS = ("done", "finished", "complete", "completed", "task_done",
              "placed")


def _says_done(parsed: dict) -> bool:
    """Did the model just say the task is over?"""
    for key in _DONE_KEYS:
        value = parsed.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in (
                "true", "yes", "done", "finished", "complete"):
            return True
    return False

PLAN_PROMPT = """You are directing a robot arm with a two-finger parallel gripper.

You do not choose motions. You choose what should be TRUE next, and a greedy
search finds the joint angles. It is good at that and can do nothing else. You
are the only part of this system that decides what ought to happen.

HOW TO REPLY -- one JSON object. Every field is optional except "why":

  {"jaws":  "open" | "close" | "hold",
   "do":    "<act>",
   "target": "<number>", "value": <n>, "compare": "==" | ">=" | "<=",
   "also":  [["<number>", <n>, <weight>], ...],
   "using": ["move"],
   "step_done": true|false,
   "done":  true, "done_why": "<what you can see that says the task is over>",
   "why":   "<one sentence>"}

Set "jaws" and a goal in the same reply when you mean both -- "open the jaws and
get the hand to the block" is one decision, not two.

{jaw_table}

The jaws are a STATE. It stays as you set it until you change it, there is no
jaw width to drive, and the search never touches the fingers. To close on the
block, say so: {"jaws": "close"}. Nothing else closes it.

ACTS THAT NEED NO SEARCH, given as "do":
{direct_table}

NAMED ACTS, given as "primitive". Each carries the conditions that make it that
act rather than something merely scoring like it:
{primitive_table}

THE NUMBERS YOU MAY DRIVE, as "target" and in "also".

Always readable, facts about the arm:
  hand_x_m hand_y_m hand_z_m   where the hand is. The bench runs roughly
                      x -0.25..0.25, y 0.10..0.50, surface at z=0.72.
  hand_pointing_down  +1 straight down, 0 level. The camera is bolted to the
                      hand, so this is also where it looks.
  pointing_at_object  +1 when the jaws face the block, 0 across it, negative
                      away. A hand close to the block and pointing elsewhere
                      cannot grasp it and no distance says so.
  range_ahead_m       distance to the first surface along the grasp axis.
  base_deg segment_1_deg segment_2_deg segment_3_deg   the joint angles.
  segment_N_tip_x_m and siblings   where each segment's far end is. Only
                      segment_3's end carries the gripper.

Readable only once the block has been SEEN, and simply absent until then:
  palm_to_object_m    metres from the gripping surfaces to the block's surface
  object_in_grasp_m   metres from the block to the LINE BETWEEN THE PADS
  palm_facing         +1 square to the face approached, 0 edge-on

READINGS, NOT GOALS -- steer something else instead: grip_tip_spread_m (the jaws
are a state), holding, object_seen, object_between_jaws, tip_force_left_n,
tip_force_right_n, pushing_n.

  pushing_n   HOW HARD YOU ARE RUNNING INTO SOMETHING, in newtons, felt by a
              part of the arm that should be carrying no load at all. On a
              clean carry it is 0.00 and stays there for the whole run.
              Anything above a newton or two means you are pressing on the
              world -- a wall, the bench, the outside of the bin.

              This is not a cost to trade off. It is a fact that whatever you
              are currently trying is being physically prevented, and no amount
              of pushing harder will finish it. A previous run drove the block
              into the outside of the bin wall at 87 N, prised its own jaws
              open, dropped the block, and went on issuing goals as though the
              carry were going fine. BACK OFF AND COME AT IT DIFFERENTLY -- for
              a container that means over the opening and then down, never
              sideways at rim height.

THE SEARCH IS BLIND TO OBSTACLES, AND YOU ARE NOT. What turns your goal into
joint angles tries every named move once and takes whichever most improves the
number. It has no idea anything is in the way: a move that drives straight
through a wall scores exactly as well as one that goes around, because the wall
is not in the number.

SO ROUTING AROUND THINGS IS YOUR JOB, and you are the only part of this system
that can do it -- you can see the wall in the pictures and the search cannot.
Break the movement into goals that are each safe on their own:

  lift clear of everything first, THEN move across, THEN come down.

Not "get the block over the bin" while it sits beside the bin at rim height.
That asks for a straight line through the wall, and the search will happily
drive into it. The failure this is written from: the arm reached rim height two
millimetres outside the bin's opening -- so it was over the WALL -- and the
next goal asked to close the horizontal distance. It crushed the block against
the wall at 87 N until the jaws were prised open and the block fell out. Every
move along the way improved the number it had been given.

pushing_n IS HOW YOU KNOW. It is zero on a clean carry and stays zero for the
whole run. Anything above a newton or two means you are pressing on something
solid and whatever you are currently asking for is being physically prevented.
Do not ask for it harder and do not try a different joint: BACK OFF, get clear
in the direction that is open -- usually up -- and approach again from a place
the straight line works from. One run spent twenty-six seconds at up to 157 N
issuing eight different goals, and moved the block two centimetres.

USING. Name "move" -- the whole arm -- and let the search pick which joint and
which motion. Do not name individual joints: asked to bring the hand closer, an
earlier run named the last hinge alone, which changes where the hand points and
not how far it reaches, and the arm stayed stretched past the block for a whole
run. There is no gripper package; the jaws are a state.

THRESHOLDS. "compare": ">=" is met once the number is at least the value, "<="
once it is at most. Use one whenever "enough" is what you mean, which is most of
the time: aiming at the block is pointing_at_object >= 0.9, not == 0.9, and an
equality goal driven past its value starts reporting error again as though
something had gone wrong. Distances take "<=", alignments ">=".

driving_this_worsens IS MEASURED FOR YOU, right now, on this body. If the
number you are about to drive appears there, the numbers it lists WILL get worse
while you drive it -- so put them in "also" with a weight and a threshold, or
you will fix one and break the other and be back where you started. This is not
advice about the task; it is what the arm just did when the search tried each
move.

NAME MORE THAN ONE NUMBER. Every metric is contested: approaching improves the
distance and destroys the facing. Put what must not be lost in "also", weighted
-- the number you chase at 1.0, the ones you protect at 0.3 to 0.6.

YOU BEGIN BLIND. Nothing object-relative can be read until the block has been
found. Point the hand down over the bench until object_seen becomes 1.

PUTTING SOMETHING SOMEWHERE, WITHOUT A METRIC FOR IT. There is no number here
that measures "over the bin" or "clear of the rim", on purpose: a vocabulary
that knows what a bin is has already decided what the task is, and cannot state
a different one. Two such numbers existed and were removed, one of them after it
read NEGATIVE when the block was resting in the bin -- which is success -- and
the run lifted it back out twice and dropped it on the bench.

You do not need them, because you already know where things are. Your
imagination carries the coordinates of everything you identified, including
whatever the object is meant to end up in or on. While you are holding
something it sits at the hand, so hand_x_m, hand_y_m and hand_z_m ARE where the
held object is. Placing it is therefore:

  drive hand_x_m and hand_y_m to the coordinates of the destination you can
  see, keeping hand_z_m high enough to clear it, then lower hand_z_m, then open
  the jaws.

Read the destination's xyz out of your imagination and use those numbers. If it
is a container, its rim is above its middle, so clear the rim on the way across
and come down only once you are over the opening.

WHAT THE TASK NEEDS, IN ORDER: find the block, open the jaws, get the hand
around it, close the jaws, lift it clear, carry it across ABOVE everything,
come down over the opening, then open the jaws to let go.

YOU DECIDE WHEN THE TASK IS OVER, AND SAYING SO ENDS THE RUN. Nothing else
will stop it. When the instruction you were given has been carried out, reply

    {"done": true, "done_why": "<what you can SEE that says so>"}

and nothing further is asked. That reply needs no goal, no jaws, nothing else:
there is nothing left to steer, which is the whole point of it.

YOU HAVE TO LOOK TO KNOW. No number tells you whether the sentence was carried
out -- that judgement is yours, from the pictures. If the task was to put
something somewhere, the OVERHEAD camera is usually how: every corner camera is
stopped by the near wall of a container, so an object inside one is invisible
to all four at once, and measured, a block resting in the bin showed 496 pixels
overhead and ZERO from every corner.

DO NOT SAY IT IS DONE BECAUSE YOU ARE OUT OF IDEAS. "done" means you can see
that it happened. Being stuck is a reason to try something different, and there
is a separate way to say so: keep deciding.

LOSING SIGHT OF SOMETHING IS NOT EVIDENCE THAT IT IS GONE, and it is not
evidence you failed. When the tracker cannot see an item you are told so, and
its position is the last one measured, going stale -- it is not a fresh
reading. An earlier run released the block cleanly into the bin, lost sight of
it at that exact instant because the corners cannot see inside, and spent its
last eight decisions trying to grasp a stale position of a block that was
already placed. Not holding something is not the same as not having placed it.
"""

def _parts_table() -> str:
    """The parts and their moves, from the manifest that declares them.

    This was prose in the prompt, hand-copied from the manifest, and it went
    stale the instant the parts were renamed -- still offering shoulder, elbow
    and wrist. The model named them, they matched nothing, and the search
    correctly found that no control moved the number. A list of what the body
    can do belongs to the body.
    """
    from .greedy import moves_for, packages

    rows = []
    for group, members in packages().items():
        moves = []
        for part in members:
            moves += [m for m in moves_for(part)
                      if m not in ("hold", "neutral") and m not in moves]
        rows.append(f"  {group:<9} {' '.join(moves)}")
        rows.append(f"  {'':<9} (parts: {', '.join(members)})")
    return chr(10).join(rows)


EXPAND_PROMPT = """You are directing a robot arm with a two-finger parallel
gripper. You will be given one instruction in plain words.

Break it into the SMALLEST NUMBER OF PHYSICAL STEPS that actually have to
happen, in order, each written as a short English sentence naming what moves and
where it goes. Do not describe joint motions, and do not invent steps the
instruction does not require.

Write each step so it names the thing that moves and, where there is one, the
thing it moves relative to -- "pick up the block", "put the block into the bin".
Those two nouns and the relation between them are what the rest of the system
partitions the step into, and a step written without them cannot be partitioned.

Reply as JSON: {"steps": ["...", "..."], "why": "<one sentence>"}.
"""

STEP_PROMPT = """Same rules. Here is what the arm senses now and what it has
just been doing. Give the next number to pursue, or repeat the current one if it
is still the right one and simply has not been reached yet.

You are working through a plan, one step at a time, and you are told which step
you are on. When that step has actually happened in the world -- not when its
number has been reached, but when the thing the step describes is true -- reply
with "step_done": true and give the first number of the NEXT step. A number
being satisfied is not the same as a step being finished, and a step being
finished is not the same as the task being done.

DO NOT REPORT A STEP FINISHED THAT DID NOT HAPPEN. step_done says the step
HAPPENED, not that you are tired of it or that nothing further seems to help. A
grasp has happened when both pads read force and holding is 1 -- not when the
jaws are nearly closed, not when the hand is nearly in place. In one measured
run the plan moved from "close the gripper to grasp the block" to "lift the
block" while holding, both pad forces and object_between_jaws all read zero in
the same message, and the ten decisions after that were spent reasoning
correctly inside a premise that was false. Being stuck on a step is a reason to
try something different WITHIN it.

YOUR IMAGINATION IS EVERYTHING YOU IDENTIFIED, kept and corrected. Before this
run began you were shown the four corner cameras and asked what the task
requires you to find; you named those things and pointed at them. What you
pointed at was turned into metres by crossing the rays, and what was under your
pick was sampled so the machine can follow each thing between your decisions.

  your_imagination.items         every thing you named, with:
        xyz                      where it is, in the same metres as hand_x_m
        confidence               0 to 1, from how well the cameras agree and
                                 how many of them can see it. Not an opinion.
        rays_missed_by_m         how far the rays were from meeting. Small
                                 means they are looking at one thing.
        seen_from                which corners voted. Fewer than two is no
                                 position at all, only a direction.
        times_checked            how many measurements it rests on.

  your_imagination.checked_just_now    THE COMPARISON, made this instant:
        measured_xyz             where the cameras put it right now
        believed_xyz             where you thought it was
        measurement_disagreed_by_m   the gap between those two
        belief_moved_by_m        how far the belief was corrected toward it

  your_imagination.not_being_followed  things you named that are no longer
        tracked -- the pick clipped the background, or nothing can see it now.
        Their positions are frozen at your last look and going stale.

THE PICTURE IS CHECKED AGAINST THE WORLD, NOT KEPT. Every one of your decisions
is preceded by measuring each thing again and moving the belief toward what was
measured. A large measurement_disagreed_by_m means the world is not where you
thought: something moved, or the tracker is following the wrong thing. A large
rays_missed_by_m means the cameras are not looking at the same object. Read both
before you steer by a coordinate.

You may steer straight to those coordinates. They are not a guess handed to
you -- they are what the cameras agree on, from where you pointed.

  your_imagination.bin_as_declared_xyz     where the fixed furniture says the
        bin is. The goal numbers are measured against THIS. If
        cameras_vs_declared_bin_m is large, the number you are driving and the
        bin you can see are not the same bin, and that is worth saying.

WHERE THE HAND IS POINTING, and whether that is at the block:

  aim.pointing_xyz         the direction the jaws face, a unit vector
  aim.toward_object_xyz    the direction the block lies in from the hand
  aim.pointing_at_object   the two, compared. 1.0 is straight at it, 0.0 is
                           square across it, negative is facing away.

pointing_at_object is askable. A hand that is close to the block and pointing
the wrong way cannot grasp it, and no distance says so -- this does. If it is
low, turn the hand before going anywhere.

A GOAL CAN BE A THRESHOLD. Send "compare" alongside the value:

  "compare": ">="   met once the number is at least the value
  "compare": "<="   met once the number is at most the value
  "compare": "=="   met at the value itself (the default)

Use a threshold whenever "enough" is what you mean, which is most of the time.
"aim at the block" is pointing_at_object >= 0.9, not == 0.9: once you are
squarer than that there is nothing left to fix, and an equality goal driven past
its value starts reporting error again as though something had gone wrong.
Distances are usually "<=" and alignments usually ">=".

what_each_number_means says what every number is and which way is better. Read
it before choosing a value.

MOST NUMBERS ARE DISTANCES YOU DRIVE TO ZERO. The ones in best_at_1_not_0 are
not: pointing_at_object, palm_facing and hand_pointing_down are best at 1.0, and
asking for a value BELOW what you already have is asking the arm to get worse at
it. That has happened and it cost a grasp -- the aim was already 1.0, 0.7 was
requested, the search obediently degraded it, and the hand swung off a block
that was sitting between the jaws.

close_enough SAYS WHEN A NUMBER IS DONE. Each metric has a tolerance there --
hand_x_m is 0.03, so being 2 mm off is not a problem worth a decision, and being
0.1 mm off is not an achievement. A number inside its tolerance is finished:
move to the next thing the task needs. Decisions are the scarce resource here,
not precision.

READ your_body_right_now BEFORE ANYTHING ELSE. It says where each joint is
inside its own travel and, for the jaws, how wide they are, how wide they GO,
and how wide the block actually measured. A hand cannot close around something
wider than its opening, and the jaws start fully shut -- so reaching for a block
with the hand clenched achieves nothing however well the arm is aimed. If
jaws_wide_enough_for_it is false, open them first; must_open_to_at_least_m says
how far.

YOU ARE WOKEN WHEN THE NUMBER ARRIVES OR WHEN IT STOPS MOVING, not on a
clock. Check woken_because. "stalled" means the search drove this number as far
as it can and it is still not there -- read search_says, which distinguishes
"no move helps at all" from "still moving but converging short". Neither is
fixed by asking for the same thing again.

A LOW ERROR IS NOT A GOOD POSE. The search will happily settle a few
centimetres from the block, square to it, scoring well, and geometrically
unable to close on it. When you are told you have stalled, suspect exactly
that, and check object_between_jaws and object_in_hand_view before believing
the number.

YOU CAN ALSO PLACE THE END OF A SEGMENT. segment_1_tip_x_m and its siblings
are where each segment's far end IS, in world metres, and you can ask for them
directly. A joint angle is exact and says nothing about where the arm ended up;
the end of a segment is a place you can picture. segment_reach in
your_body_right_now gives each segment's pivot and its length, which is the
circle its end is confined to -- ask for a point on that circle, not off it.

ONLY segment_3's end carries the gripper. The other two are elbows: driving
them swings the arm without putting the hand anywhere in particular. To move the
HAND, ask for hand_x_m / hand_y_m / hand_z_m, or for segment_3's tip. Use
segment_1 and segment_2 to change the arm's shape -- to get an elbow out of the
way, or to fold the arm in when it is over-extended.

EVERY JOINT IS ALSO A NUMBER. base_deg, segment_1_deg, segment_2_deg and
segment_3_deg are the joint angles in degrees, always readable, each moved by
exactly one part and contested by nothing. When what you want is simply "this
segment folded a bit further" or "the arm a little higher", say it as a joint
angle rather than hunting for a task metric that happens to mean it. Read the
current value in sensed, add or subtract, and ask for that.

READ what_you_already_tried AND predicament BEFORE CHOOSING. They carry what a
single frame cannot: every goal you have set, how far its error actually moved
while you held it, and whether it arrived. A goal that moved by nearly nothing
did not work, and asking for it again will not make it work -- change the number
or change the package. If times_you_asked_for_this_metric is climbing, you are
in a loop, and the way out is a different goal rather than a better argument for
the same one.

READ packages_that_change_each_number BEFORE CHOOSING. It is measured on this
body, in this pose, this instant. If a number has an EMPTY list, nothing the
body can do will move it now, and asking for it wastes the whole interval until
you are asked again. That is usually a sign the number belongs to a later step:
nothing moves the block toward the bin while the jaws are empty, because moving
the arm does not move a block it is not holding. Grip it first.

Say plainly if the last target was a mistake -- a number that cannot be moved by
the controls you named, or one already satisfied while the task did not advance.
If you are handed back a target you already reached and the task has not moved,
that is the signal to advance the step, not to name it again.

Reply as JSON with ONE of {"do": "<act>"},
{"primitive": "<name>", ...}, or, optionally with "compare",
{"target": "<name>", "value": <number>,
"also": [["<name>", <number>, <weight>], ...], "using": ["<package>", ...],
"step_done": <true|false>, "why": "<one sentence>"}.
"""


IDENTIFY_PROMPT = """Look at the room and find the things this task is about.

You are shown the same moment from four cameras, one in each corner. Before
anything moves, work out WHAT THE TASK REQUIRES YOU TO FIND, and point at each
of those things in every view you can see it in.

Nothing here has been found for you. There is no list of objects, no colour to
look for, no position given. What is in this room is what you can see in these
pictures, and which of it matters is decided by the sentence you were given.

REPLY WITH ONE JSON OBJECT:

  {"items": [
     {"name":  "<short name you will keep using, e.g. block, bin>",
      "what_it_is": "<what you see, in a few words>",
      "role":  "figure" | "ground" | "obstacle" | "other",
      "picks": {"<camera name>": [x, y], ...},
      "sure":  0.0 to 1.0},
     ...],
   "why": "<one sentence on what the task needs and why these things>"}

ROLE. "figure" is the thing that moves. "ground" is what it moves to, into or
onto. "obstacle" is something that must be avoided. Exactly one figure.

PICKS are pixel coordinates in the image from that camera, x across from the
left edge and y down from the top, in an image {pick_w} wide and {pick_h} high.
Give a pick for EVERY camera that can see the thing, and leave out any camera
that cannot. Two is the minimum: one camera pointing at something says only
which direction it lies in, and a direction is not a position. Four is better
than two, and a camera you are unsure about is worse than no camera at all.

POINT AT THE MIDDLE OF WHAT YOU CAN SEE OF IT, not at where you reason its
centre must be. For an open container the middle of the opening is empty air,
and a pick there lands on whatever is behind it. Pick a spot that is plainly ON
the object, well inside its edges -- what is under that pixel is sampled and
used to follow the thing between now and the next time you are asked, so a pick
that clips the background teaches the machine to follow the background.

Your picks become metres by crossing the rays from the cameras, and how far
those rays miss each other is measured and shown back to you. Picks that
disagree produce a large miss and a low confidence, and you will be told.
"""


@lru_cache(maxsize=1)
def _load_env() -> bool:
    """Same credentials as the rest of the pipeline, from the same .env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    here = Path(__file__).resolve()
    for candidate in (here.parents[4] / ".env", here.parents[5] / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            _unquote_env()
            return True
    return False


#: Credentials that arrive by paste and are ruined by one stray character.
_PASTED = ("OPENAI_API_KEY", "RIGBY_VLM_MODEL")


def _unquote_env() -> None:
    """Strip stray quotes and whitespace from pasted credentials.

    python-dotenv strips quotes only when they MATCH. A key pasted as
    `KEY=sk-proj-...-4A"` -- one trailing quote, no leading one -- is loaded
    with the quote as part of the value, and the only symptom is a 401 that
    reaches the run as "the model did not provide a usable goal". That cost a
    whole run to diagnose: three model calls, zero of them counted, and a
    progress message blaming the model for an authentication failure.

    One character of cleanup here is cheaper than reading it in a stack trace.
    """
    import os

    for name in _PASTED:
        value = os.environ.get(name)
        if not value:
            continue
        clean = value.strip().strip('"').strip("'").strip()
        if clean != value:
            os.environ[name] = clean


def _as_png(pixels: np.ndarray, scale: int = 1) -> str:
    picture = Image.fromarray(pixels)
    if scale != 1:
        picture = picture.resize((picture.width // scale, picture.height // scale))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class Planner:
    """Asks a model what number to chase, from two views and the sensed numbers."""

    #: What it was asked to do, in words. The only instruction from outside.
    task: str = "put the block into the bin"
    model: str | None = None
    #: ASKED AGAIN WHEN THE GOAL IS REACHED, not on a clock. The model decides
    #: what should become true; the search makes it true; then, and only then,
    #: there is a new question worth paying for. A timer asks while the arm is
    #: still halfway through the last answer, which spends money to be told the
    #: same thing.
    #:
    #: The clock is only a backstop, for a goal the search cannot reach at all
    #: -- otherwise one impossible target ends the run in silence.
    stuck_after_s: float = 8.0
    #: Hard ceiling on calls per run, because a loop that repeats itself pays
    #: for every repetition.
    max_calls: int = 14
    #: A goal that stops improving is finished, whether or not it arrived.
    #:
    #: The search can converge to a pose with a low error that cannot do the
    #: job: the hand a few centimetres from the block, square to it, scoring
    #: well, and geometrically incapable of closing on it. Nothing about the
    #: number says so. What says so is that the number STOPPED MOVING -- and
    #: that is the moment worth spending a call on, because it is the moment
    #: the search has finished telling us anything.
    #:
    #: Relative improvement over the window, below which it counts as stalled.
    plateau_fraction: float = 0.05
    #: How long the number must fail to improve before it counts.
    plateau_window_s: float = 2.0
    #: Hard ceiling on calls per run, because a loop that repeats itself pays
    #: for every repetition.
    max_calls: int = 14
    #: A goal that stops improving is finished, whether or not it arrived.
    #:
    #: The search can converge to a pose with a low error that cannot do the
    #: job: the hand a few centimetres from the block, square to it, scoring
    #: well, and geometrically incapable of closing on it. Nothing about the
    #: number says so. What says so is that the number STOPPED MOVING -- and
    #: that is the moment worth spending a call on, because it is the moment
    #: the search has finished telling us anything.
    #:
    #: Relative improvement over the window, below which it counts as stalled.
    plateau_fraction: float = 0.05
    #: How long the number must fail to improve before it counts.
    plateau_window_s: float = 2.0
    #: The soonest a new decision may follow the last one, seconds.
    #:
    #: A goal is re-asked the moment it is reached, and a goal can be reached
    #: the instant it is set -- the model asked for the jaws to be at 0.00 cm
    #: when they were already at 0.00 cm, which was true on arrival, so it was
    #: re-asked on the very next frame, and again, and again: six decisions in
    #: two tenths of a second, a fifth of the run's whole budget spent before
    #: the arm had moved. Deciding is not free and a body does not change fast
    #: enough to be worth re-deciding at 30 Hz.
    client: Any | None = None

    #: The steps the instruction was expanded into, and which one is current.
    plan: list = field(default_factory=list, repr=False)
    #: Each step partitioned by Talmy: what moves, with respect to what, along
    #: which path. Structure the model does not have to re-derive every call.
    partitioned: list = field(default_factory=list, repr=False)
    step: int = field(default=0, repr=False)

    held: NumericTarget | None = field(default=None, repr=False)
    spans: dict = field(default_factory=dict, repr=False)
    calls: int = field(default=0, repr=False)
    #: THE PLANNER'S OWN PICTURE OF THE SCENE, and how well it has been paying.
    #: Its coordinates are not a cheat the way a handed-down pose would be: the
    #: model works them out from the views and keeps them, and they are checked
    #: against a reading it must predict.
    imagination: dict = field(default_factory=dict, repr=False)
    #: 0 to 1, moved by whether the picture predicts what is then measured.
    imagination_confidence: float = field(default=0.5, repr=False)
    #: Every check made, so a run can be asked whether its picture was any good.
    imagination_checks: list = field(default_factory=list, repr=False)
    #: WHAT IS IN THE ROOM, and how sure the cameras are of each. Held here
    #: rather than rebuilt per call because a belief that is thrown away and
    #: recomputed cannot be compared with anything, and comparing it with what
    #: is measured next is the entire point of keeping one.
    world: World = field(default_factory=World, repr=False)
    #: What the model said the task requires, and what it said about each.
    inventory: dict = field(default_factory=dict, repr=False)
    #: The item that moves and the item it is moved to, in the model's words.
    figure: str = field(default="", repr=False)
    ground: str = field(default="", repr=False)
    #: Every identification pass: what was asked for, found, and refused.
    identifications: list = field(default_factory=list, repr=False)
    _identified_at: float = field(default=-1e9, repr=False)
    _expand_tries: int = field(default=0, repr=False)
    #: THE MODEL'S OWN VERDICT THAT THE TASK IS OVER, and when it said so.
    #: Not "placed" -- that was a pick-and-place word in a layer that is not
    #: supposed to know what the task is, the same mistake as a carry_to_bin
    #: primitive. Whether a sentence has been carried out is a judgement about
    #: the world, and it is the model's to make for any sentence.
    #:
    #: Scored against a grader afterwards; never used to steer. The grader is
    #: task-specific and lives in the evaluation, which is where knowing what
    #: the task was is allowed.
    finished: bool = field(default=False, repr=False)
    finished_why: str = field(default="", repr=False)
    finished_at: float = field(default=0.0, repr=False)
    #: HOW OFTEN EACH METRIC HAS BEEN ASKED FOR, over the whole run.
    #:
    #: This used to be counted by scanning the last eight remembered decisions,
    #: so the count collapsed to zero exactly when the repetition was most
    #: established: measured over one run it read 0, 1, 2, 3, then 0, then 4,
    #: then 0, then 5. The one signal telling the planner it was going in
    #: circles reset itself every time the window rolled.
    asked_counts: dict = field(default_factory=dict, repr=False)
    #: Consecutive re-asks that named an already-reached target unchanged.
    repeats: int = field(default=0, repr=False)
    #: What the jaws are doing: "open", "close" or "hold". A state, set by the
    #: planner and left alone until it says otherwise.
    jaws: str = field(default="open", repr=False)
    #: What the last reply got wrong, handed straight back on the next ask.
    last_refusal: str = field(default="", repr=False)
    #: A direct act being carried out, bypassing the search entirely.
    doing: str = field(default="", repr=False)
    #: Whether the current goal has been unmet at any point since it was set.
    _was_ever_unmet: bool = field(default=False, repr=False)
    #: HOW MUCH THE RUN'S OWN BEHAVIOUR ARGUES AGAINST THE PICTURE, 0 to 1.
    #: Raised when the search converges on an object-relative goal without
    #: arriving, lowered when it converges and arrives. Geometry says how well
    #: the cameras agree; this says whether acting on them worked.
    belief_doubt: float = field(default=0.0, repr=False)
    belief_evidence: list = field(default_factory=list, repr=False)
    #: EVERY DECISION THIS RUN AND WHAT BECAME OF IT. A model answering from
    #: one frame cannot see that it has answered the same way before, and every
    #: long failure in these runs has been the same defensible choice made
    #: again: hand_pointing_down four times, hand_y_m three, palm_to_object_m
    #: three, while the reading never moved. Each was reasonable on its own
    #: frame. What made them wrong is that they had already been made.
    memory: list = field(default_factory=list, repr=False)
    #: (time, error, part/move) for the current goal, most recent last.
    trail: list = field(default_factory=list, repr=False)
    #: The most recent scoring of the picture, handed back with the next ask.
    _last_check: dict = field(default_factory=dict, repr=False)
    #: Why the model was woken last: "arrived", "stalled" or "clock".
    woken_by: str = field(default="", repr=False)
    #: Where the current goal started, so progress can be reported as a change
    #: rather than as a number with nothing to compare it to.
    _began: float = field(default=0.0, repr=False)
    transcript: list = field(default_factory=list, repr=False)
    _asked_at: float = field(default=-1e9, repr=False)

    def _openai(self):
        if self.client is None:
            _load_env()
            from openai import OpenAI

            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self.client

    def _model(self) -> str:
        return self.model or os.environ.get("RIGBY_VLM_MODEL", "gpt-4o")

    def scene(self) -> dict:
        """What the machine is entitled to know without looking: the furniture."""
        bin_doc = spec()["scene"]["bin"]
        return {
            "task": self.task,
            "bin_centre_xyz": bin_doc["centre"],
            "bin_rim_height_m": bin_doc["rim_height_m"],
            "note": "the bin is fixed furniture; the block's position is not "
                    "known and must be seen",
        }

    def _scene(self):
        """The gripper's world, in the shape Talmy expects to be handed.

        Coordinates are the viewer's Y-up, which is what SceneManifest is
        written in, so the bin's manifest entry goes in unconverted and the
        block's is stated the same way. Talmy only needs to know WHICH things
        are present and what they are called -- it partitions a sentence, it
        does not measure anything -- but the schema insists on real dimensions
        and a graspable socket, and inventing plausible ones would be inventing
        facts. These are the scene's own numbers.
        """
        from ...models import (AffordanceSocket, SceneManifest, SceneObject,
                               Transform, Vec3)

        bin_doc = spec()["scene"]["bin"]
        inner = bin_doc["inner_half_m"]
        return SceneManifest(
            objects=[
                SceneObject(
                    id="block", kind="block",
                    transform=Transform(translation=Vec3(x=0.0, y=0.76, z=0.30)),
                    dimensions_m=Vec3(x=0.06, y=0.08, z=0.06), mass_kg=0.25,
                    sockets=[AffordanceSocket(
                        id="side", transform=Transform(
                            translation=Vec3(x=0.0, y=0.76, z=0.30)),
                        approach_normal=Vec3(x=0.0, y=0.0, z=-1.0),
                        grasp_span_m=0.06)],
                ),
                SceneObject(
                    id="bin", kind="bin",
                    transform=Transform(translation=Vec3(
                        x=float(bin_doc["centre"][0]),
                        y=float(bin_doc["centre"][1]),
                        z=float(bin_doc["centre"][2]))),
                    dimensions_m=Vec3(x=float(inner[0]) * 2, y=float(inner[1]) * 2,
                                      z=float(inner[2]) * 2),
                    sockets=[AffordanceSocket(
                        id="mouth", transform=Transform(translation=Vec3(
                            x=float(bin_doc["centre"][0]),
                            y=float(bin_doc["rim_height_m"]),
                            z=float(bin_doc["centre"][2]))),
                        approach_normal=Vec3(x=0.0, y=1.0, z=0.0),
                        grasp_span_m=0.06)],
                ),
            ],
            support_height_m=0.72,
        )

    def expand(self, task: str | None = None,
               now: float = 0.0) -> list:
        """Turn one instruction into the steps it actually requires.

        Two stages rather than one, and the reason is that they are different
        questions. "What does 'put the block in the bin' consist of" is answered
        once, from the words. "What number should be true right now" is answered
        repeatedly, from what the cameras show. Folding them together makes the
        model re-derive the whole task on every call, and it drifts.

        Each step is then partitioned by Talmy into FIGURE, GROUND and PATH --
        what moves, with respect to what, and the respect in which it moves. The
        model is handed that partition rather than asked to hold the structure
        of the sentence in its head while also reading two camera images.
        """
        from ...talmy import interpret

        wanted = task or self.task
        try:
            response = self._openai().chat.completions.create(
                model=self._model(),
                messages=[
                    {"role": "system", "content": EXPAND_PROMPT},
                    {"role": "user", "content": json.dumps(
                        {"instruction": wanted, "scene": self.scene()},
                        sort_keys=True)},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=_REPLY_TOKENS,
            )
            self.calls += 1
            choice = response.choices[0]
            raw = choice.message.content
            if not raw:
                # Not a refusal by the model -- a reply that never finished.
                #
                # This block was copied from ask(), and it kept the names its
                # new scope does not have: it read `now`, which expand() has no
                # parameter for, and returned `self.held`, which is a goal and
                # not a plan. The bare except below swallowed the NameError, so
                # a truncated expansion was reported as "error: name 'now' is
                # not defined" and became an empty plan -- after which ask()
                # re-expanded on EVERY call, because an empty plan is the
                # condition it expands on. A budget overrun turned into an
                # unbounded spend, silently.
                self.transcript.append({
                    "t": round(now, 2), "refused": "",
                    "reply": "", "finish_reason": str(choice.finish_reason),
                    "completion_tokens": int(getattr(
                        getattr(response, "usage", None),
                        "completion_tokens", 0) or 0),
                    "why": "the model returned no content while expanding the "
                           "instruction; if finish_reason is 'length' the token "
                           "budget was spent before it could answer",
                })
                self.plan = []
                return self.plan
            parsed = json.loads(raw)
            self.plan = [str(s) for s in parsed.get("steps", [])]
        except Exception as error:  # noqa: BLE001
            self.transcript.append({"t": 0.0, "error": str(error)[:200]})
            self.plan = []

        scene = self._scene()
        self.partitioned = []
        for sentence in self.plan:
            situation = interpret(sentence, scene)
            self.partitioned.append(
                None if situation is None else situation.to_dict())
        self.transcript.append({
            "t": 0.0, "instruction": wanted, "steps": list(self.plan),
            "talmy": list(self.partitioned),
            "why": str(parsed.get("why", ""))[:200] if self.plan else "",
        })
        return self.plan

    def observe(self, error: float, how: dict, now: float) -> None:
        """One frame of what the search is achieving. Called every frame.

        The planner cannot see the search from the outside, and "how far did
        this actually get" is not a question a single frame answers.
        """
        # A goal only counts as arrived-at if it was ever not arrived-at.
        if self.held is not None and float(error) > 0.12:
            self._was_ever_unmet = True
        self.trail.append((float(now), float(error),
                           f"{how.get('part', '-')}/{how.get('move', 'hold')}"))
        # Two windows' worth is all that is ever read.
        cutoff = now - 2.0 * self.plateau_window_s
        self.trail = [row for row in self.trail if row[0] >= cutoff]

    def _stalled(self, now: float) -> bool:
        """Whether the current goal has stopped improving."""
        if self.held is None:
            return False
        age = now - self.held.set_at_s
        if age < self.plateau_window_s:
            return False
        window = [row for row in self.trail
                  if row[0] >= now - self.plateau_window_s]
        if len(window) < 5:
            return False
        first, last = window[0][1], window[-1][1]
        if first <= 1e-9:
            # NOTHING TO DO IS A KIND OF STALLED. A goal that has sat at zero
            # error for the whole window is finished in the only sense that
            # matters -- the search has nothing to drive and never will. This
            # used to return False, on the reasoning that a zero cannot fail to
            # improve, which left such a goal to the eight-second clock. That is
            # six seconds of a forty-five second run spent waiting to be told
            # something already true.
            return True
        return bool((first - last) / first < self.plateau_fraction)

    def stuck_because(self, now: float, seen: Sensed | None = None) -> str:
        """What KIND of stuck, in the terms a decision needs.

        A search that keeps picking `hold` has run out of moves that help, and
        no amount of patience will change that. A search still choosing moves
        while the number barely shifts is being pulled somewhere it cannot get
        to. The two want different answers.

        AND BEING PINNED IS NEITHER OF THOSE. This used to explain every stall
        in terms of the search -- "no move improves this number at all, the
        search is out of options" -- while the arm was pressed against the bin
        at 76 N. That is a search-theoretic answer to a physical question, and
        it is not merely unhelpful, it points the wrong way: "the search is out
        of options" invites naming a different joint, which is what the model
        did fifteen times over twenty-eight seconds while pushing 52 to 86 N
        and moving the block two centimetres.

        The machine had both facts and never joined them. The force is a
        reading and the stall is a reading; only together do they say "you
        cannot move because something solid is in the way".
        """
        if seen is not None and float(getattr(seen, "pushing_n", 0.0)) > _PINNED_N:
            part = str(getattr(seen, "pushing_with", "") or "the arm")
            return (f"YOU ARE PINNED. {part} is pressed against something at "
                    f"{float(seen.pushing_n):.0f} N. No move improves the "
                    f"number because the arm physically cannot go that way, so "
                    f"naming a different joint or a different metric will not "
                    f"help. Back off first -- give up ground in the direction "
                    f"that is open, usually straight up -- and only then aim "
                    f"for where you were trying to get to.")
        window = [row for row in self.trail
                  if row[0] >= now - self.plateau_window_s]
        if not window:
            return ""
        holding = sum(1 for row in window if row[2].endswith("/hold"))
        if holding >= 0.8 * len(window):
            return ("no move of the named packages improves this number at all "
                    "-- the search is out of options, not short of time")
        return ("moves are still being found but the number is barely shifting "
                "-- it is converging somewhere it cannot finish")

    def due(self, body: Body, seen: Sensed, now: float) -> bool:
        """Time for a new decision: there is none, it is done, or it is stuck."""
        # NOTHING TO DECIDE AFTER THE TASK IS OVER. Without this the model
        # answers "it is finished" and is immediately asked what to do next,
        # forever -- which is how a completed run spent 29 calls in 1.2 s.
        if self.finished:
            return False
        if self.doing:
            # No search to stall, so the only questions are whether the joints
            # got there and whether they have stopped trying.
            act = build_direct(self.doing)
            if act is None:
                self.doing = ""
                return True
            here = np.asarray(body.q())
            want = act.pose(body, here)
            if float(np.abs(want - here).max()) <= 0.002:
                self.woken_by = "the act finished"
                self.doing = ""
                return True
            return now - self._asked_at >= self.stuck_after_s
        if self.held is None:
            # A MISSING GOAL IS A REASON TO ASK, NOT A REASON TO ASK FOREVER.
            # This returned True unconditionally and never looked at the clock,
            # so any state with no goal asked once per FRAME. Measured: a
            # close_jaws act cleared the goal at t=16.2, the model then replied
            # "placed: true" with no target, that was refused for carrying no
            # number, the goal stayed None -- and 29 of the run's 43 calls went
            # in 1.2 seconds. The refusal path set _asked_at and reported
            # "asked_again_after_s: 8.0"; nothing on this branch read it.
            return self.calls == 0 or now - self._asked_at >= _RETRY_AFTER_S
        if self.calls >= self.max_calls:
            return False
        if self.held.reached(body, seen, self.spans):
            # ARRIVING AT SOMETHING YOU WERE ALREADY AT IS NOT ARRIVING.
            #
            # A clock used to guard this -- no new decision within 0.6 s -- and
            # a clock is the wrong instrument: it blocks a fast action that
            # genuinely finished, and it does not stop the thing it was for.
            # The real failure was asking for a number the body already had,
            # which is satisfied on the frame it is set, so the planner is woken
            # instantly and asks again: six decisions in two tenths of a second.
            #
            # What separates the two is whether the goal was EVER unmet. A jaw
            # command that took 0.8 s to complete deserves a decision; a goal
            # that was true before the arm moved does not.
            if self._was_ever_unmet:
                self.woken_by = "arrived"
                return True
            # A GOAL THAT WAS TRUE ALL ALONG IS NOT AN ARRIVAL -- and it is not
            # a reason to stop asking, either. Returning False here swallowed
            # the stall check and the clock below it, so a goal satisfied on the
            # frame it was set silenced the planner for the rest of the run:
            # measured, forty-three seconds and thirty-six unused calls frozen
            # on pointing_at_object == 0.9, which was already true when it was
            # asked for.
            #
            # Fall through instead. Such a goal IS stalled -- nothing is moving
            # and nothing will -- so the stall detector picks it up a couple of
            # seconds later and says so, which is the honest description.
        if self._stalled(now):
            self.woken_by = "stalled"
            # CONVERGENCE IS EVIDENCE, not just a reason to wake up.
            #
            # The search drives a number as far as the body allows and then
            # stops. If the number ARRIVED, the picture that produced it
            # predicted the world correctly and is worth more. If it converged
            # and did NOT arrive, the arm did everything it could and the goal
            # still was not met -- and for a goal defined against the object,
            # the most likely thing that is wrong is where we believe the
            # object is.
            #
            # That makes a stall a measurement of the belief rather than only a
            # report about the search, which is the difference between a signal
            # that wakes someone and a signal that teaches something.
            about_object = self.held.metric in NEEDS_SIGHT
            arrived = self.held.reached(body, seen, self.spans)
            if about_object and not arrived:
                self.belief_doubt = min(1.0, self.belief_doubt + 0.25)
                self.belief_evidence.append({
                    "t": round(now, 2), "metric": self.held.metric,
                    "converged_without_arriving": True,
                    "reads": "the arm stopped improving this and never got "
                             "there; if the body could do it, the picture is "
                             "what is wrong",
                    "doubt_now": round(self.belief_doubt, 3),
                })
            elif about_object and arrived:
                self.belief_doubt = max(0.0, self.belief_doubt - 0.35)
                self.belief_evidence.append({
                    "t": round(now, 2), "metric": self.held.metric,
                    "converged_and_arrived": True,
                    "reads": "the picture predicted a reachable goal and the "
                             "body reached it",
                    "doubt_now": round(self.belief_doubt, 3),
                })
            self.belief_evidence = self.belief_evidence[-12:]
            return True
        if now - self._asked_at >= self.stuck_after_s:
            self.woken_by = "clock"
            return True
        return False

    def identify(self, body: Body, now: float = 0.0, why: str = "first look"
                 ) -> dict:
        """Find everything the task is about, before deciding anything.

        This is the first thing that happens, and it happens again whenever the
        machine loses something. It is not a plan and it does not move the arm:
        it is the model reading its instruction, looking at four pictures, and
        saying what it has to find and where each of those things is.

        WHY THE MODEL AND NOT A SEGMENTER. The segmenter it replaces knew what
        it was looking for -- the only warm thing on a blue-grey bench -- which
        meant the scene's contents had been decided by whoever wrote the test.
        The bin was not in it, so the bin was invisible to every instrument in
        the machine and known only because a manifest declared it. A colour
        vocabulary would have been the same mistake one layer up. Reading the
        task and pointing at what it names is the only version where a sentence
        about something else works without anyone editing this file.

        WHAT COMES BACK IS CHECKED, not believed. Picks turn into metres by
        crossing rays, the miss between them is measured, and the appearance
        sampled at each pick is verified by tracking with it before it is kept.
        """
        pictures = {name: body.view(PICK_W, PICK_H, camera=name)
                    for name in CORNERS}
        content: list = [{"type": "text", "text": json.dumps(
            {"instruction": self.task, "image_size": [PICK_W, PICK_H],
             "cameras": list(pictures), "why_now": why}, sort_keys=True)}]
        for name, image in pictures.items():
            content.append({"type": "text",
                            "text": f"camera {name}, {PICK_W}x{PICK_H}:"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + _as_png(image)}})

        record: dict = {"t": round(now, 2), "why": why}
        try:
            response = self._openai().chat.completions.create(
                model=self._model(),
                messages=[
                    {"role": "system", "content": IDENTIFY_PROMPT
                     .replace("{pick_w}", str(PICK_W))
                     .replace("{pick_h}", str(PICK_H))},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=_REPLY_TOKENS,
            )
            self.calls += 1
            parsed = json.loads(response.choices[0].message.content or "{}")
        except Exception as error:  # noqa: BLE001
            record["error"] = str(error)[:200]
            self.identifications.append(record)
            self.transcript.append({"t": round(now, 2),
                                    "identify_error": record["error"]})
            return record

        record["said"] = parsed.get("why", "")
        found: list = []
        for entry in parsed.get("items") or []:
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            picks = {str(k): v for k, v in (entry.get("picks") or {}).items()}
            placed = self.world.anchor(body, name, picks, now)
            placed["role"] = str(entry.get("role", "other")).lower()
            placed["what_it_is"] = str(entry.get("what_it_is", ""))
            placed["model_was_sure"] = entry.get("sure")
            found.append(placed)
            self.inventory[name] = {
                "what_it_is": placed["what_it_is"], "role": placed["role"],
                "model_was_sure": placed["model_was_sure"],
            }
            if placed["role"] == "figure" and placed.get("seen"):
                self.figure = name
            elif placed["role"] == "ground" and placed.get("seen"):
                self.ground = name
        record["found"] = found
        self.identifications.append(record)
        self.transcript.append({"t": round(now, 2), "identified": [
            {"item": f.get("item"), "role": f.get("role"),
             "seen": f.get("seen"), "xyz": f.get("xyz"),
             "rays_missed_by_m": f.get("rays_missed_by_m"),
             "learned": bool(f.get("learned"))} for f in found]})
        return record

    def lost(self) -> list:
        """Items that were identified and are no longer being followed.

        Two ways to be lost, and they need the same repair: the appearance was
        never learned because the pick clipped the background, or it was learned
        and no camera can find it any more. Either way the only fix is to look
        again, which costs a call, so it is worth knowing exactly when.
        """
        return [name for name in self.inventory
                if name not in self.world.looks]

    def build_imagination(self, body: Body, seen: Sensed, now: float) -> dict:
        """The scene, reconstructed from the corner cameras that can see it.

        This replaces asking the planner to keep its own picture. Measured over
        one full run, its picture sat 12 cm out in depth and it never revised it
        -- sixteen decisions steering by the same three wrong numbers while the
        confidence it was shown sat at zero. The reconstruction was not the
        problem to hand to a model that cannot estimate depth from an oblique
        view; the problem was that four cameras can do it in closed form.

        None of this is the simulator's truth. It is where rays from known
        camera positions cross, which is the cameras' opinion, and the spread
        between them is how much they disagree.
        """
        # EVERY ITEM IS LOOKED FOR AGAIN AND THE BELIEF IS CORRECTED. This used
        # to rebuild one position from scratch each call and keep nothing, which
        # meant there was no belief to contradict: a picture that is discarded
        # and recomputed cannot be found to have been wrong, and cannot get
        # better either. Now each item is measured, the measurement is compared
        # with what was believed, the gap is reported, and the belief moves
        # toward the measurement by as much as the cameras' agreement earns.
        checked = self.world.refresh(body, now)
        self.imagination = {
            "items": self.world.as_dict(),
            "checked_just_now": checked,
            "roles": {name: row.get("role") for name, row
                      in self.inventory.items()},
            "not_being_followed": self.lost(),
        }

        # WHERE THE DECLARED FURNITURE SAYS THE BIN IS, beside where the cameras
        # put it. The goal metrics still measure against the manifest, so if the
        # two disagree the numbers being driven are about a different bin from
        # the one in the pictures -- which is worth seeing rather than
        # discovering later from a run that missed.
        declared = task.bin_of()["centre"]
        self.imagination["bin_as_declared_xyz"] = [round(float(v), 4)
                                                   for v in declared]
        if self.ground and self.world.at(self.ground) is not None:
            self.imagination["cameras_vs_declared_bin_m"] = round(float(
                np.linalg.norm(self.world.at(self.ground)
                               - np.asarray(declared, dtype=float))), 4)

        # The figure's position under the old name, because aim(), the prompt
        # and every transcript reader still speak of block_xyz. It is the same
        # number when the figure is the block and the right number when it is
        # not, which a rename would have broken silently.
        figure = self.figure or "block"
        held = self.world.at(figure)
        if held is None:
            self.imagination["block_xyz"] = None
            self.imagination["why_none"] = (
                f"{figure!r} has not been placed by the cameras"
                if figure in self.inventory else
                "nothing has been identified yet, so there is nothing to place")
            self.imagination_confidence = 0.0
        else:
            self.imagination["block_xyz"] = [round(float(v), 4) for v in held]
            geometric = self.world.confidence(figure)
            # Two independent things: how well the cameras agree, and whether
            # acting on what they said actually worked. A picture the cameras
            # love and the arm keeps failing to act on is not a good picture.
            self.imagination_confidence = float(
                np.clip(geometric * (1.0 - self.belief_doubt), 0.0, 1.0))
            self.imagination["cameras_agree"] = round(geometric, 3)
            self.imagination["acting_on_it_has_failed"] = round(
                self.belief_doubt, 3)
        self.imagination_checks.append({
            "t": round(now, 2), "block_xyz": self.imagination["block_xyz"],
            "checked": checked,
            "confidence": round(self.imagination_confidence, 3),
        })
        self.imagination_checks = self.imagination_checks[-20:]
        return self.imagination

    def aim(self, body: Body) -> dict:
        """Where the jaws face, and whether the block is that way.

        A hand can be four centimetres from the block and facing across it, and
        every distance in the vocabulary reads well while a grasp is impossible.
        The comparison of the two directions is the number that says so, and
        nothing else here does.
        """
        pointing = np.asarray(body.approach(), dtype=float)
        out = {"pointing_xyz": [round(float(v), 3) for v in pointing]}
        believed = (self.imagination or {}).get("block_xyz")
        if not believed:
            out["toward_object_xyz"] = None
            out["pointing_at_object"] = None
            return out
        span = np.asarray(believed, dtype=float) - body.grasp_centre()
        size = float(np.linalg.norm(span))
        # AN OBJECT IN THE HAND HAS NO DIRECTION. Once the block is gripped it
        # sits a centimetre or two from the grasp centre, and a unit vector over
        # a gap that small is dominated by the block settling rather than by
        # anything about where it is. Measured while carrying a rigidly-held
        # block: the aim wandered 0.54 -> 0.22 -> 0.59 with the block clamped
        # and never moving relative to the plate by more than 2.7 mm.
        #
        # Reporting a number that swings for no reason is worse than reporting
        # none: it invites the planner to correct an aim that is not wrong.
        if size < 0.05:
            out["toward_object_xyz"] = None
            out["pointing_at_object"] = None
            out["why_none"] = ("the object is in the hand; direction to "
                               "something you are holding is not meaningful")
            return out
        if size < 1e-6:
            out["toward_object_xyz"] = None
            out["pointing_at_object"] = 1.0
            return out
        toward = span / size
        out["toward_object_xyz"] = [round(float(v), 3) for v in toward]
        out["pointing_at_object"] = round(float(np.dot(pointing, toward)), 3)
        return out

    def machine(self, body: Body, seen: Sensed) -> dict:
        """The state of the body itself: where every joint is in its range.

        The sensed block is a flat list of readings, and a reading without its
        limits does not say what it MEANS. grip_tip_spread_m: 0.007 is the jaws
        clenched as hard as they physically go, and it was being sent as though
        it were any other small number -- so the model asked to close a hand
        that was already shut, over and over, and never once opened it before
        reaching for a block 6 cm wide that could not possibly fit between pads
        7 mm apart.

        Everything here is proprioception and manifest: joint encoders, declared
        travel, and the object width the camera measured. Nothing is new
        information; it is the same information said properly.
        """
        state: dict = {"joints": {}}
        for index, name in enumerate(JOINTS[:4]):
            low, high = body.model.jnt_range[body.model.joint(name).id]
            here = float(np.degrees(body.q()[index]))
            span = float(np.degrees(high)) - float(np.degrees(low))
            state["joints"][name] = {
                "now_deg": round(here, 1),
                "range_deg": [round(float(np.degrees(low)), 1),
                              round(float(np.degrees(high)), 1)],
                # THE SAME POSITION SAID BOTH WAYS. The absolute value is what
                # a goal is set in; the fraction is what makes it legible --
                # "0.5" says the joint is mid-travel at a glance, where "0.0
                # deg" says nothing until you have also read the range.
                "fraction_of_travel": (
                    None if span <= 1e-9
                    else round((here - float(np.degrees(low))) / span, 3)),
                "room_to_increase_deg": round(float(np.degrees(high)) - here, 1),
                "room_to_decrease_deg": round(here - float(np.degrees(low)), 1),
            }

        # WHERE EACH SEGMENT'S FAR END CAN GO. The end of a segment is on a
        # circle: centred on that segment's own pivot, of the segment's own
        # length, in the plane its hinge allows. Naming a point off that circle
        # is naming somewhere the segment cannot put it, so the centre and the
        # radius are given rather than left to be inferred from the geometry.
        reach: dict = {}
        # WHICH END IS THE HAND. Offering three tips with pivots and lengths and
        # not saying which one holds the gripper invited exactly one mistake,
        # and it was made: the planner reached for the block by driving
        # segment_1's tip, which is 36 cm from the base and carries nothing.
        # It repositioned the arm's midpoint four times while the hand stayed
        # where it was.
        carries = {
            "segment_1": "nothing -- this is the arm's first elbow, and moving "
                         "it swings everything past it",
            "segment_2": "nothing -- the second elbow",
            "segment_3": "THE GRIPPER. This tip is the plate the jaws are "
                         "mounted on, so this is the one to place when you "
                         "want the hand somewhere",
        }
        for name, pivot, tip in (("segment_1", "link1", "link2"),
                                 ("segment_2", "link2", "link3"),
                                 ("segment_3", "link3", "plate")):
            centre = body.body_at(pivot)
            end = body.body_at(tip)
            reach[name] = {
                "tip_now_xyz": [round(float(v), 4) for v in end],
                "pivots_around_xyz": [round(float(v), 4) for v in centre],
                "arm_length_m": round(float(np.linalg.norm(end - centre)), 4),
                "this_end_carries": carries[name],
            }
        state["segment_reach"] = reach

        opening = float(body.opening())
        low, high = body.model.jnt_range[body.model.joint("finger_left").id]
        widest = 2.0 * float(high) - 0.024
        narrowest = 2.0 * float(low) - 0.024
        travel = max(widest - narrowest, 1e-9)
        jaws = {
            "opening_m": round(opening, 4),
            # 0.0 is shut as hard as the pads go, 1.0 is as wide as they open.
            # Reported alongside the metres rather than instead of them: goals
            # are set in metres, and a second askable number for the same
            # physical quantity would be two names for one thing.
            "open_fraction": round(
                min(1.0, max(0.0, (opening - narrowest) / travel)), 3),
            "can_open_to_m": round(widest, 4),
            "can_close_to_m": round(narrowest, 4),
            "fully_closed": bool(opening <= narrowest + 0.002),
            "fully_open": bool(opening >= widest - 0.002),
        }
        if seen.object_size is not None:
            # The camera measured two half-extents; the widest across is what
            # has to pass between the pads.
            across = 2.0 * float(max(seen.object_size[0], seen.object_size[1]))
            jaws["object_measured_width_m"] = round(across, 4)
            jaws["width_measured_by"] = str(getattr(seen, "seen_by", "") or "-")
            # A WIDTH FROM ACROSS THE ROOM IS NOT A WIDTH TO PLAN A GRIP ON.
            # At the first frame the room camera measured this block at 9.4 cm
            # when it is 6 to 8, which would have told the model to open to
            # 10.4 cm -- wider than the jaws physically go -- and so that the
            # task was impossible. The same precedence as position applies: a
            # coarse estimate is enough to reach toward, not to size a grip.
            close = bool(getattr(seen, "fine_fix", False))
            jaws["width_is_a_close_measurement"] = close
            if close:
                jaws["jaws_wide_enough_for_it"] = bool(opening > across + 0.005)
                jaws["must_open_to_at_least_m"] = round(
                    min(across + 0.01, widest), 4)
            else:
                jaws["jaws_wide_enough_for_it"] = None
                jaws["advice"] = ("width not yet measured from close range; "
                                  "open the jaws wide and let the object stop "
                                  "them")
        state["jaws"] = jaws
        state["holding"] = bool(seen.holding())
        state["pads_n"] = [round(float(seen.tip_force_left_n), 2),
                           round(float(seen.tip_force_right_n), 2)]
        return state


    def _predicament(self, body: Body, seen: Sensed, now: float) -> dict:
        """Where this run stands, in the terms a decision needs.

        Not a log. The three questions a planner has to answer are how long it
        has been on this step, whether the last thing it asked for actually
        moved, and whether it has asked for it before -- and none of them can
        be read off a single frame of sensor values.
        """
        repeated = (0 if self.held is None
                    else int(self.asked_counts.get(self.held.metric, 0)))
        stalled = [e["asked"] for e in self.memory
                   if abs(e.get("moved", 0.0)) < 0.05]
        return {
            "run_time_s": round(float(now), 1),
            "decisions_so_far": len(self.memory),
            "model_calls_used": self.calls,
            "model_calls_left": max(0, self.max_calls - self.calls),
            "on_step": self.step,
            "of_steps": len(self.plan),
            "step_text": (self.plan[self.step]
                          if self.step < len(self.plan) else None),
            "current_goal_age_s": (None if self.held is None
                                   else round(float(now - self.held.set_at_s), 1)),
            "times_you_asked_for_this_metric": repeated,
            "times_asked_per_metric": dict(self.asked_counts),
            "goals_that_did_not_move": sorted(set(stalled)),
            "holding": bool(seen.holding()),
            "object_seen": bool(seen.object_seen),
            "seen_by": str(getattr(seen, "seen_by", "") or "nothing"),
            "current_goal_waiting_on": ([] if self.held is None
                                        else self.held.unmet(body, seen)),
            "woken_because": self.woken_by or "first decision",
            "search_says": self.stuck_because(now, seen),
            "error_now": (None if self.held is None or not self.trail
                          else round(self.trail[-1][1], 3)),
            "error_when_set": round(float(self._began), 3),
        }

    def ask(self, body: Body, seen: Sensed, now: float,
            note: str = "") -> NumericTarget | None:
        """One decision, from both cameras and every number the body senses."""
        # THE INSTRUCTION IS EXPANDED BEFORE THE FIRST DECISION. expand() was
        # written, documented and reachable, and nothing ever called it: the
        # whole run went by with an empty plan, every transcript entry reading
        # step_text: null, and the model deciding from the bare sentence and the
        # sensed numbers with no steps and no Talmy partition behind it. It
        # belongs here rather than in the run loop so that expansion happens for
        # any caller, exactly once, on the way to the first decision.
        if not self.plan and self._expand_tries < _EXPAND_TRIES:
            # BOUNDED. An expansion that comes back empty leaves self.plan
            # empty, which is exactly the condition for expanding -- so a
            # failure here used to re-ask forever, one model call per decision,
            # for the whole run. Deciding without a plan is worse than deciding
            # with one and far better than never deciding at all.
            self._expand_tries += 1
            self.expand(now=now)
            if not self.plan:
                self.transcript.append({
                    "t": round(now, 2),
                    "why": f"expansion returned no steps "
                           f"(attempt {self._expand_tries} of {_EXPAND_TRIES}); "
                           f"deciding from the instruction itself",
                })
        # NOTHING IS DECIDED BEFORE THE SCENE HAS BEEN IDENTIFIED. The task
        # names things; which things are in this room, and where, is a question
        # about the pictures and has to be answered by looking at them. Doing it
        # here rather than in the run loop means it happens for every caller,
        # on the way to the first decision, exactly once.
        if not self.inventory:
            self.identify(body, now, why="first look, nothing found yet")
            self._identified_at = now
        elif self.lost() and now - self._identified_at >= _RELOOK_AFTER_S:
            # An item stops being followed when its appearance was never
            # learned or when no camera can find it any more. Either way the
            # belief is going stale and only another look repairs it -- but a
            # look costs a call, so it is rationed rather than reflexive.
            self.identify(body, now,
                          why=f"lost track of {', '.join(self.lost())}")
            self._identified_at = now
        # SCORE THE PICTURE BEFORE ASKING FOR THE NEXT ONE, so the planner is
        # told how the last one did rather than marking its own work.
        self.build_imagination(body, seen, now)
        numbers = readable(body, seen)
        delivering = self.last_refusal
        self.last_refusal = ""
        situation = {
            "instruction": self.task,
            "plan": list(self.plan),
            "step_index": self.step,
            "step": self.plan[self.step] if self.step < len(self.plan) else None,
            "step_partitioned": (self.partitioned[self.step]
                                 if self.step < len(self.partitioned) else None),
            "sensed": numbers,
            "your_body_right_now": self.machine(body, seen),
            "jaws_are": self.jaws,
            "your_imagination": dict(self.imagination or {}),
            # WHAT YOU SAID WAS HERE, and what has become of each of those
            # things since. The planner used to be handed one position and no
            # account of where it came from or whether it had held up.
            "what_you_identified": dict(self.inventory),
            "the_thing_that_moves": self.figure or None,
            "where_it_goes": self.ground or None,
            "imagination_confidence": round(self.imagination_confidence, 3),
            "aim": self.aim(body),
            "seeing_the_object_now": bool(seen.object_seen),
            "current_target": None if self.held is None else {
                "metric": self.held.metric, "value": self.held.value,
                "also": [list(a) for a in self.held.also],
                "using": list(self.held.using),
            },
            "steps_total": len(self.plan),
            "steps_left": max(0, len(self.plan) - self.step - 1),
            "since_last_decision": note,
            "your_last_reply_was_refused": delivering,
            "what_you_already_tried": list(self.memory),
            "predicament": self._predicament(body, seen, now),
            # WHAT MOVES WHAT, measured on this body at this pose. Without it
            # the planner has only the part names to reason from, and the names
            # do not say that the last hinge cannot shorten the arm's reach.
            "packages_that_change_each_number": changers(
                body, seen, tuple(sorted(READABLE))),
            # WHICH NUMBERS FIGHT, measured on this body at this pose. The
            # planner drove "get the block over the bin", then "get it above
            # the rim", then the first again -- each undoing the last -- while
            # using `also` correctly on eleven other goals. It was not missing
            # the mechanism; it was missing the knowledge that those two are in
            # conflict, which is a fact about the mechanism and not about bins.
            "driving_this_worsens": contests(
                body, seen, tuple(sorted(READABLE))),
            # WHAT COUNTS AS CLOSE ENOUGH, per number. These floors already
            # existed -- the search scales every error by them -- and the
            # planner was never shown them. So it chased hand_x_m from 1.7 mm
            # to 0.8 to 0.5 to 0.1, four decisions spent on a number that was
            # already better than the camera can measure, with the jaws still
            # shut and nothing picked up.
            "close_enough": {name: floor for name, floor in _FLOOR.items()
                             if name in READABLE},
            "best_at_1_not_0": list(BEST_AT_ONE),
            "what_each_number_means": {k: v for k, v in DESCRIBES.items()
                                       if k in READABLE},
            "askable": sorted(READABLE),
            "not_askable": list(OUTCOMES),
        }
        content = [
            {"type": "text", "text": json.dumps(
                {"scene": self.scene(), "now": situation}, sort_keys=True)},
            # THE SAME SIZE THE PICKS WERE MADE IN, deliberately. A model
            # shown one resolution and asked to point in another is being asked
            # to do arithmetic it has no reason to get right, and it costs a GL
            # context to render the second size for no gain.
            {"type": "text",
             "text": f"CORNER camera, {PICK_W}x{PICK_H} -- the whole bench:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
             + _as_png(body.view(PICK_W, PICK_H, camera="room"))}},
            # THE ONE VIEW THAT CAN SEE INTO THE BIN. Every corner camera is
            # stopped by the near wall, so an object that has been placed
            # disappears from all of them at once -- which is precisely when
            # you most need to see it. Sent on every decision because whether
            # the task is finished is now the model's judgement to make.
            {"type": "text",
             "text": f"OVERHEAD camera, {PICK_W}x{PICK_H} -- looking down into "
                     "the bin. The only view that sees inside it:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
             + _as_png(body.view(PICK_W, PICK_H, camera="overhead"))}},
            {"type": "text",
             "text": f"GRIPPER camera, {GRIP_W}x{GRIP_H} -- what the hand is "
                     "pointed at, and what the machine segments:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
             + _as_png(body.view(GRIP_W, GRIP_H, camera="gripper"))}},
        ]
        try:
            response = self._openai().chat.completions.create(
                model=self._model(),
                messages=[
                    {"role": "system",
                     "content": (PLAN_PROMPT.replace("{parts_table}", _parts_table())
                                .replace("{primitive_table}", catalogue())
                                .replace("{direct_table}", direct_catalogue())
                                .replace("{jaw_table}", jaw_catalogue())
                                if self.held is None else STEP_PROMPT)},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=_REPLY_TOKENS,
            )
            self.calls += 1
            parsed = json.loads(response.choices[0].message.content or "{}")
        except Exception as error:  # noqa: BLE001
            self.transcript.append({"t": round(now, 2), "error": str(error)[:200]})
            return self.held

        # The jaws are set independently of the goal, because they are a state
        # and the goal is a number. Both can be given in one reply.
        asked_jaws = str(parsed.get("jaws", ""))
        if asked_jaws not in JAW_STATES:
            # "close_gripper", "close_grip", "close the jaws" -- all obviously
            # mean the same state, and refusing them on spelling helps nobody.
            spoken = " ".join(str(parsed.get(k, "")) for k in
                              ("jaws", "primitive", "do")).lower()
            for state in ("close", "open", "hold"):
                if state in spoken:
                    asked_jaws = state
                    break
        if asked_jaws in JAW_STATES:
            self.jaws = asked_jaws

        # "do" AND "primitive" ARE THE SAME QUESTION. Which of the two a named
        # thing is filed under is an implementation detail -- one skips the
        # search, one carries conditions -- and the planner has no reason to
        # track it. Measured: thirty refusals in one run, the first being
        # {"do": "move_to"} for a primitive that exists under the other name.
        # Look the name up in both, wherever it was offered.
        named_anything = (str(parsed.get("do", ""))
                          or str(parsed.get("primitive", "")))
        wanted_act = named_anything if named_anything in DIRECT else ""
        if wanted_act in DIRECT:
            self.doing = wanted_act
            self.held = None
            self._asked_at = now
            self.transcript.append({
                "t": round(now, 2), "did": wanted_act,
                "step": self.step, "step_text": (
                    self.plan[self.step] if self.step < len(self.plan) else None),
                "why": str(parsed.get("why", ""))[:200], "sensed": numbers,
            })
            return None
        self.doing = ""

        named = named_anything if named_anything in PRIMITIVES else ""
        if named in PRIMITIVES:
            target = build_primitive(named, body, seen, now)
            if target is not None:
                return self._adopt(target, body, seen, now, parsed, numbers,
                                   named, situation)

        # A REFUSAL THE MODEL NEVER HEARS IS A REFUSAL THAT REPEATS FOREVER.
        #
        # Measured: ten consecutive replies of {"primitive": "close_gripper"},
        # each with correct reasoning -- "the block is between the open jaws,
        # holding is false and both pads read zero; close the gripper" -- and
        # every one silently dropped, because close_grip had been deleted and
        # the real lever is {"jaws": "close"}. The model was right about the
        # world and wrong about one name, and nothing told it which.
        bad_name = ""
        if str(parsed.get("primitive", "")) and str(parsed["primitive"]) not in PRIMITIVES:
            bad_name = f'primitive "{parsed["primitive"]}"'
        elif str(parsed.get("do", "")) and str(parsed["do"]) not in DIRECT:
            bad_name = f'act "{parsed["do"]}"'
        if bad_name:
            wants_grip = any(w in bad_name.lower()
                             for w in ("close", "grip", "grasp", "open"))
            self.last_refusal = (
                f'Your last reply named {bad_name}, which does not exist. '
                + ('To work the jaws send {"jaws": "close"} or {"jaws": "open"} '
                   '-- there is no primitive for it. '
                   if wants_grip else '')
                + f'Valid primitives: {sorted(PRIMITIVES)}. '
                  f'Valid acts: {sorted(DIRECT)}.')
            self.transcript.append({
                "t": round(now, 2), "refused": bad_name,
                "reply": json.dumps(parsed)[:400],
                "told_the_model": self.last_refusal,
            })
            # A REFUSAL IS STILL AN ASK. Leaving the clock untouched meant the
            # wake condition was still true on the very next frame, so the
            # planner asked again 0.03 s later, and again: twenty-seven of one
            # run's forty calls went into bursts of the same rejected reply.
            # Being turned down has to cost the same wait as being answered.
            self._asked_at = now
            return self.held

        # SAYING "IT IS DONE" IS AN ANSWER, NOT A MALFORMED GOAL. Placement is
        # the model's judgement now -- object_in_target was taken out of the
        # payload precisely so it would be -- and the model made that judgement
        # correctly from the overhead camera: "the overhead camera clearly
        # shows the orange block resting inside the green bin". The reply
        # carried no `target`, because there is nothing left to steer, so it
        # was refused as "not a number this body can be steered by" and asked
        # again. Twenty-nine times. The machine had finished the task and the
        # plumbing would not let it stop.
        if _says_done(parsed):
            self.finished = True
            self.finished_why = (str(parsed.get("done_why", ""))
                                 or str(parsed.get("why", "")))[:300]
            self.finished_at = round(now, 2)
            self.transcript.append({
                "t": round(now, 2), "finished": True,
                "finished_why": self.finished_why,
                "jaws": self.jaws, "sensed": numbers,
            })
            self.held = None
            self._asked_at = now
            return None

        metric = str(parsed.get("target", ""))
        if metric not in READABLE:
            self.transcript.append(
                {"t": round(now, 2), "refused": metric, "asked_again_after_s":
                 round(self.stuck_after_s, 1),
                 # WHAT IT ACTUALLY SAID. A refusal used to record only that
                 # one happened, so eight replies in a run were invisible: I
                 # could see their effect and not their content, and spent an
                 # afternoon inferring the model's words from their side
                 # effects. The reply is small and it is the only evidence of
                 # what was asked for.
                 "reply_keys": sorted(parsed),
                 "reply": json.dumps(parsed)[:400],
                 "jaws_after": self.jaws,
                 "why": "not a number this body can be steered by"})
            self._asked_at = now
            return self.held

        also = []
        for entry in parsed.get("also") or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 \
                    and str(entry[0]) in READABLE:
                also.append((str(entry[0]), float(entry[1]),
                             float(entry[2]) if len(entry) > 2 else 0.5))
        target = NumericTarget(
            metric=metric, value=float(parsed.get("value", 0.0)), set_at_s=now,
            also=tuple(also),
            using=tuple(str(u) for u in (parsed.get("using") or [])),
            compare=(str(parsed.get("compare", "=="))
                     if str(parsed.get("compare", "==")) in ("==", ">=", "<=")
                     else "=="),
        )
        return self._adopt(target, body, seen, now, parsed, numbers, "",
                           situation)

    def _adopt(self, target, body: Body, seen: Sensed, now: float,
               parsed: dict, numbers: dict, primitive: str,
               situation: dict):
        """Take a target as the current goal, however it was named."""
        # ADVANCING THE PLAN. self.step was read in three places and written in
        # none, so the model was shown step 0 for the whole run: it could never
        # be told a step had finished, and on every re-ask it correctly named
        # step 0's goal again, spending a call to repeat itself. The plan and
        # its Talmy partition were decorative after the first decision.
        #
        # The model says when a step is done, because whether "pick up the
        # block" has happened is a judgement about the world, not a threshold on
        # one number -- a goal can be reached while the step it belongs to has
        # not occurred.
        # WHETHER THE TASK IS FINISHED IS NOW THE MODEL'S CALL. It used to be
        # handed object_in_target, computed from MuJoCo's true block pose --
        # the grader, shown to the thing being graded. That is gone, so the
        # model has to look at the overhead camera and decide, and it says so
        # here. The grader still runs; it just scores this claim instead of
        # supplying it, which turns "did it work" into two numbers that can
        # disagree: whether the block is in the bin, and whether the machine
        # knew. In the run of 2026-09-06 those two would have disagreed for
        # twenty seconds and eight decisions.
        was = self.step
        if bool(parsed.get("step_done")) and self.step < len(self.plan) - 1:
            self.step += 1
            self.repeats = 0
        else:
            # A satisfied goal handed back unchanged is not progress. If that
            # happens twice running, advance rather than pay a third time for
            # the same answer -- and say so in the transcript, because a plan
            # that moved on for a reason the model did not give is a thing the
            # reader needs to be able to see.
            same = (self.held is not None
                    and self.held.metric == target.metric
                    and abs(self.held.value - float(target.value)) < 1e-9)
            settled = self.held is not None and self.held.reached(
                body, seen, self.spans)
            self.repeats = self.repeats + 1 if (same and settled) else 0
            if self.repeats >= 2 and self.step < len(self.plan) - 1:
                self.step += 1
                self.repeats = 0
                self.transcript.append({
                    "t": round(now, 2), "step": self.step,
                    "advanced_without_being_told": True,
                    "why": "the same reached target was named twice running",
                })

        # What became of the goal now being replaced, measured rather than
        # guessed: how far its error moved, and whether it arrived.
        if self.held is not None:
            ended = self.held.error(body, seen, self.spans)
            self.memory.append({
                "at_s": round(float(self.held.set_at_s), 1),
                "asked": self.held.metric,
                "value": round(float(self.held.value), 4),
                "using": list(self.held.using),
                "error_from": round(float(self._began), 3),
                "error_to": round(float(ended), 3),
                "moved": round(float(self._began - ended), 3),
                "arrived": bool(self.held.reached(body, seen, self.spans)),
                "held_for_s": round(float(now - self.held.set_at_s), 1),
            })
            # THE WHOLE RUN, not a window. A window forgets exactly what is
            # worth remembering: the eight-entry version dropped the earliest
            # asks first, so a metric asked twelve times reported as four, and
            # the evidence of going in circles thinned out as the circling got
            # worse.

        self.asked_counts[target.metric] = (
            int(self.asked_counts.get(target.metric, 0)) + 1)
        self.held = target
        self.spans = target.spans(body, seen)
        self._began = float(target.error(body, seen, self.spans))
        self._was_ever_unmet = self._began > 0.12
        self._asked_at = now
        self.transcript.append({
            "t": round(now, 2), "target": target.metric,
            "value": target.value,
            "also": [list(a) for a in target.also], "using": list(target.using),
            "step": self.step, "step_was": was,
            "step_text": self.plan[self.step] if self.step < len(self.plan) else None,
            "step_done": bool(parsed.get("step_done")),
            "finished": bool(self.finished),
            "primitive": primitive,
            "unmet": target.unmet(body, seen),
            "why": str(parsed.get("why", ""))[:200], "sensed": numbers,
            # WHAT IT WAS TOLD, kept beside what it decided. Without this the
            # transcript cannot answer "did it know it was stuck", which is the
            # first question worth asking of a run that froze.
            "predicament": situation.get("predicament"),
            "jaws": self.jaws,
            "imagination": dict(self.imagination or {}),
            "imagination_confidence": round(self.imagination_confidence, 3),
            "aim": self.aim(body),
            "your_body_right_now": situation.get("your_body_right_now"),
        })
        return target
