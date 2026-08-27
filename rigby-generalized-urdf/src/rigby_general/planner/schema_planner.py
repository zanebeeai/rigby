"""Plain language to a body-neutral schema program.

The planner's whole job is to fill Talmy's slots. It picks a Path schema, a
Figure and a Ground, a degree of remove, and a Manner -- and it may pick nothing
else, because there is nothing else in the contract. It cannot emit a distance, a
duration, or a joint name, since ``MotionSchemaProgramV1`` has no field that
would hold one.

Two implementations sit behind one protocol.

:class:`OfflineSchemaPlanner` is a deterministic recognizer over the closed
class. It is the primary route, not a fallback. Because the target vocabulary is
closed and small, recognizing it is a parsing problem rather than a generation
one, and a parser is inspectable, instant, free, and identical on every run --
all properties a demo and a benchmark want. It also mirrors the rule the v1
system already states: local capability parsing stays authoritative when a model
contradicts a clearly implemented action.

:class:`OpenAISchemaPlanner` handles wording the recognizer does not cover. Its
output is validated against the same contract and the same sealed inventory, so
the model chooses among schemas rather than inventing them.

The surface cues below are organised by which slot they fill, because that is
what they are: English has many ways to say *far*, and every one of them means
``DISTAL`` to a robot that has no idea how long its own arm is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..errors import GeneralFailureCode, RigbyGeneralError
from ..guardrails import screen
from ..schema.inventory import SchemaEntry, SchemaInventory
from ..schema.program import (
    BoundaryCondition,
    Concurrency,
    Deixis,
    Dimensionality,
    MannerV1,
    MotionSchemaProgramV1,
    ReferenceFrame,
    RegionV1,
    Remove,
    RoleBindingV1,
    SegmentLinkV1,
    SegmentV1,
    Flexion,
    PostureV1,
    POSTURE_REMOVE,
)


# -- Path: which schema is being asked for ---------------------------------
#
# Ordered most specific first. "wave" must beat "move", or every gesture becomes
# a reach.
_PATH_CUES: tuple[tuple[str, str], ...] = (
    # Postures first. "point at" is a posture on a hand and a reach on an
    # arm, and the generic verbs below would swallow it before the hand ever
    # got the chance. Which body can actually answer is settled by the
    # affordance check downstream, not by the order of these lines.
    (r"\b(peace sign|peace|victory|v sign|two fingers up|deuce|three fingers up)\b", "configure_effector"),
    (r"\b(thumbs?[ -]?up|fist|make a fist|clench|ball up)\b", "configure_effector"),
    (r"\b(open (your |the )?(hand|fingers)|flat hand|splay|spread (your )?fingers|high five)\b", "configure_effector"),
    (r"\b(one finger up|index up|point(ing)? (with )?(one )?finger)\b", "configure_effector"),
    (r"\b(beckon|come here|call (it|them) (over|in)|draw (it )?(in|closer))\b", "draw_hither"),
    (r"\b(shoo|wave (it|them) (away|off)|push (it )?away)\b", "push_thither"),
    (r"\b(wave|waving|oscillat\w*|shake|shaking|wag|wiggle|jiggle)\b", "oscillate_about_point"),
    (r"\b(back and forth|side to side|to and fro)\b", "oscillate_along_line"),
    (r"\b(circle|circular|orbit|loop around|trace a circle|go round)\b", "circle_axis"),
    (r"\b(sweep|scan|sweeping|scanning|across|traverse)\b", "traverse_line"),
    (r"\b(arc across|swing across)\b", "traverse_line_arced"),
    (r"\b(lower|descend|go down|come down|set (it )?down|down (on)?to)\b", "descend_to_surface"),
    # Grasp verbs, before the plain lift below. "Pick it up" means take hold of
    # it and raise it, and reading it as a bare upward move is how a robot with
    # no gripper at all ends up certifying a pick: the arm rises, nothing is
    # held, and the trace says it succeeded. These map to contact schemas, which
    # are unafforded until a grasp certifies, so the answer becomes an honest
    # refusal naming the missing capability.
    (r"\b(pick (it|the \w+)? ?up|grasp|grab|take hold of|grip (it|the))\b", "transport_object"),
    (r"\b(lift|raise|go up|come up|off the (table|surface|floor))\b", "retract_from_surface"),
    (r"\b(retract|withdraw|pull back|come back|return|back away|go home)\b", "retract_from_point"),
    (r"\b(back off|move away|move back|further away)\b", "move_away"),
    (r"\b(approach|move toward|closer to|edge toward)\b", "move_toward"),
    (r"\b(swing|arc|curve|loop over|over the top)\b", "swing_to_point"),
    (r"\b(as far as (you can|possible)|full(y)? extend|maximum reach|edge of (its|your) reach)\b", "reach_to_edge"),
    (r"\b(reach into|into the|inside the)\b", "enter_volume"),
    (r"\b(hold|stay|freeze|remain|keep still|hold still)\b", "hold_still"),
    (r"\b(orient|turn (the )?(wrist|tool|gripper|hand)|rotate the (wrist|tool))\b", "orient_effector"),
    (r"\b(look at|aim at|point the camera|track)\b", "track_with_gaze"),
    (r"\b(hand (it )?(over|across)|pass (it )?(over|across|to the other))\b", "hand_across"),
    (r"\b(reach|extend|stretch|move|go|point|put|place)\b", "reach_to_point"),
)


# -- Posture: how many members, and at which end of their own travel -------
#
# Every entry here is a count and a pair of poles. None of them names a digit,
# because the phrase does not know which body it will land on: "two fingers up"
# is two of the opposed members in measured order, which is the index and middle
# of a five-digit hand, two of three on a tripod, and more members than a
# two-jaw gripper has -- an honest refusal rather than an approximation.
_POSTURE_CUES: tuple[tuple[str, "PostureV1"], ...] = (
    (
        r"\b(peace sign|peace|victory|v sign|two fingers up|deuce)\b",
        PostureV1(
            selected_count=2,
            selected=Flexion.EXTENDED,
            remainder=Flexion.FLEXED,
            opposing=Flexion.FLEXED,
        ),
    ),
    (
        r"\b(point(ing)?( with)?( one)? finger|one finger up|index up|point at)\b",
        PostureV1(
            selected_count=1,
            selected=Flexion.EXTENDED,
            remainder=Flexion.FLEXED,
            opposing=Flexion.FLEXED,
        ),
    ),
    (
        r"\b(thumbs?[ -]?up|approve|nice one)\b",
        PostureV1(
            selected_count=0,
            remainder=Flexion.FLEXED,
            opposing=Flexion.EXTENDED,
        ),
    ),
    (
        r"\b(fist|make a fist|close (your |the )?(hand|fingers)|clench|ball up)\b",
        PostureV1(
            selected_count=0,
            remainder=Flexion.FLEXED,
            opposing=Flexion.FLEXED,
        ),
    ),
    (
        r"\b(open (your |the )?(hand|fingers)|flat hand|splay|spread (your )?fingers|high five)\b",
        PostureV1(
            # Every member extended, said without knowing how many there are:
            # select none and let the remainder carry it.
            selected_count=0,
            selected=Flexion.EXTENDED,
            remainder=Flexion.EXTENDED,
            opposing=Flexion.EXTENDED,
        ),
    ),
    (
        r"\b(three fingers up|trio)\b",
        PostureV1(
            selected_count=3,
            selected=Flexion.EXTENDED,
            remainder=Flexion.FLEXED,
            opposing=Flexion.FLEXED,
        ),
    ),
)


def _match_posture(clause: str) -> "PostureV1 | None":
    """The posture this clause names, if it names one."""

    for pattern, posture in _POSTURE_CUES:
        if re.search(pattern, clause, re.IGNORECASE):
            return posture
    return None


# -- Region: how far, in terms the body resolves ---------------------------
_REGION_CUES: tuple[tuple[str, Remove], ...] = (
    (r"\b(as far as (you can|possible)|full(y)?|maximum|all the way|right out)\b", Remove.DISTAL),
    (r"\b(far|distant|way out|out there|the far)\b", Remove.DISTAL),
    (r"\b(halfway|midway|middle|part(-| )way|moderate)\b", Remove.MEDIAL),
    (r"\b(near|close|nearby|just (in front|ahead)|a little|slightly|barely)\b", Remove.PROXIMAL),
    (r"\b(right (here|there)|touching|against|adjacent|up against)\b", Remove.ADJACENT),
)

# -- Manner: the co-event, in ordinals relative to this body's own neutral --
_SPEED_CUES: tuple[tuple[str, int], ...] = (
    (r"\b(as fast as|flat out|as quick(ly)? as)\b", 2),
    (r"\b(quick(ly)?|fast|swift(ly)?|rapid(ly)?|briskly|hurry)\b", 1),
    (r"\b(slow(ly)?|gently|gradual(ly)?|carefully|take your time|easy)\b", -1),
    (r"\b(very slow(ly)?|crawl|inch)\b", -2),
)
_AMPLITUDE_CUES: tuple[tuple[str, int], ...] = (
    (r"\b(huge|enormous|as (wide|big) as|sweeping|expansive)\b", 2),
    (r"\b(wide|big|large|broad|generous)\b", 1),
    (r"\b(small|tight|narrow|slight|subtle|little)\b", -1),
    (r"\b(tiny|minute|barely)\b", -2),
)
_EFFORT_CUES: tuple[tuple[str, int], ...] = (
    (r"\b(hard|forceful(ly)?|firm(ly)?|strong(ly)?|vigorous(ly)?)\b", 1),
    (r"\b(gently|soft(ly)?|lightly|delicate(ly)?|careful(ly)?)\b", -1),
)
_SMOOTHNESS_CUES: tuple[tuple[str, int], ...] = (
    (r"\b(smooth(ly)?|fluid(ly)?|evenly|steadily)\b", 1),
    (r"\b(jerk(y|ily)|abrupt(ly)?|sharp(ly)?|stacatto|staccato)\b", -1),
)
_PRECISION_CUES: tuple[tuple[str, int], ...] = (
    (r"\b(precise(ly)?|exact(ly)?|accurate(ly)?|carefully placed)\b", 1),
    (r"\b(roughly|approximate(ly)?|about|casual(ly)?)\b", -1),
)

_COUNT_WORDS = {
    "once": 1, "twice": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple": 2, "a few": 3,
}
_COUNT_PATTERN = re.compile(
    r"\b(\d+)\s*(?:times|cycles|repetitions|reps)\b"
    r"|\b(once|twice|three|four|five|six|seven|eight|nine|ten|a couple|a few)\s*(?:times|cycles)?\b"
)

# Tversky and Lee again: a route description is a chain of segments, and the
# joints between them are marked explicitly. These are those markers.
_SEQUENCE_SPLIT = re.compile(
    r",?\s*\b(?:and then|then|after that|afterwards|next|before returning|"
    r"and finally|finally|followed by)\b\s*,?",
    re.IGNORECASE,
)


class SchemaPlanner(Protocol):
    def plan(self, prompt: str, *, afforded: tuple[SchemaEntry, ...]) -> MotionSchemaProgramV1:
        ...


@dataclass(frozen=True, slots=True)
class PlannerTrace:
    """What the recognizer saw, so a wrong reading can be traced to its cue."""

    clause: str
    entry_id: str
    remove: str
    manner: dict[str, int]


class OfflineSchemaPlanner:
    """Deterministic recognition of the closed class. No model call."""

    planner_id = "offline-recognizer-v1"

    def __init__(self, inventory: SchemaInventory) -> None:
        self.inventory = inventory
        self.last_trace: tuple[PlannerTrace, ...] = ()

    def plan(
        self, prompt: str, *, afforded: tuple[SchemaEntry, ...]
    ) -> MotionSchemaProgramV1:
        text = prompt.strip()
        if not text:
            raise RigbyGeneralError(
                GeneralFailureCode.INVALID_CONTRACT, "the prompt is empty"
            )

        # Capability screening runs before recognition, not after. A recognizer
        # keyed on spatial vocabulary will happily find a sweep in "drive across
        # the room", because the word is there -- and filtering afterwards would
        # mean the filter has to understand the recognizer's output rather than
        # the request.
        decision = screen(text)
        if not decision.allowed:
            raise RigbyGeneralError(
                GeneralFailureCode.UNSUPPORTED_MORPHOLOGY,
                decision.reason,
                details={**decision.as_details(), "prompt": text[:200]},
            )

        clauses = [part.strip() for part in _SEQUENCE_SPLIT.split(text) if part.strip()]
        available = {entry.entry_id: entry for entry in afforded}

        segments: list[SegmentV1] = []
        traces: list[PlannerTrace] = []
        for index, clause in enumerate(clauses[:8]):
            entry_id = _match_path(clause)
            if entry_id is None:
                continue
            entry = available.get(entry_id)
            if entry is None:
                # Recognized, but this body cannot do it. Naming the schema lets
                # the caller report exactly what was unavailable rather than a
                # generic failure.
                raise RigbyGeneralError(
                    GeneralFailureCode.UNAFFORDED_SCHEMA,
                    f"this robot has no certified {entry_id!r} primitive",
                    details={"entry_id": entry_id, "clause": clause},
                )
            remove = _match_region(clause)
            manner = _match_manner(clause)
            posture = _match_posture(clause) if entry.takes_posture else None
            if posture is not None:
                # A posture is shaped, not placed. Whatever the sentence implied
                # about distance is about a motion this segment is not making.
                remove = POSTURE_REMOVE
            segments.append(
                SegmentV1(
                    segment_id=f"s{index}",
                    motion_schema=entry.schema,
                    figure=RoleBindingV1(role=entry.figure_role),
                    ground=RoleBindingV1(role=entry.ground_role),
                    region=RegionV1(
                        remove=remove, dimensionality=Dimensionality.POINT
                    ),
                    frame=_frame_for(entry),
                    manner=manner,
                    posture=posture,
                    boundary=_boundary_for(entry),
                )
            )
            traces.append(
                PlannerTrace(
                    clause=clause,
                    entry_id=entry_id,
                    remove=remove.value,
                    manner={
                        axis: getattr(manner, axis)
                        for axis in ("speed", "effort", "amplitude", "repetition")
                        if getattr(manner, axis)
                    },
                )
            )

        if not segments:
            raise RigbyGeneralError(
                GeneralFailureCode.UNSUPPORTED_MORPHOLOGY
                if False
                else GeneralFailureCode.UNAFFORDED_SCHEMA,
                "nothing in that request names a motion this system knows",
                details={"prompt": text[:200]},
            )

        links = tuple(
            SegmentLinkV1(
                from_segment=first.segment_id,
                to_segment=second.segment_id,
                relation=Concurrency.SEQUENCE,
            )
            for first, second in zip(segments, segments[1:])
        )
        self.last_trace = tuple(traces)

        return MotionSchemaProgramV1(
            program_id=f"offline-{abs(hash(text)) % (10**12):012d}",
            source_text=text,
            segments=tuple(segments),
            links=links,
        )


def _match_path(clause: str) -> str | None:
    lowered = clause.lower()
    for pattern, entry_id in _PATH_CUES:
        if re.search(pattern, lowered):
            return entry_id
    return None


def _match_region(clause: str) -> Remove:
    lowered = clause.lower()
    for pattern, remove in _REGION_CUES:
        if re.search(pattern, lowered):
            return remove
    # Unmarked reach is a comfortable one. Language leaves it unstated far more
    # often than not, and the neutral reading is the middle of the range rather
    # than either extreme.
    return Remove.MEDIAL


def _ordinal(clause: str, cues: tuple[tuple[str, int], ...]) -> int:
    lowered = clause.lower()
    best = 0
    for pattern, value in cues:
        if re.search(pattern, lowered) and abs(value) > abs(best):
            best = value
    return best


def _match_manner(clause: str) -> MannerV1:
    lowered = clause.lower()
    count: int | None = None
    match = _COUNT_PATTERN.search(lowered)
    if match:
        if match.group(1):
            count = max(1, min(64, int(match.group(1))))
        elif match.group(2):
            count = _COUNT_WORDS.get(match.group(2))

    return MannerV1(
        speed=_ordinal(lowered, _SPEED_CUES),
        effort=_ordinal(lowered, _EFFORT_CUES),
        smoothness=_ordinal(lowered, _SMOOTHNESS_CUES),
        amplitude=_ordinal(lowered, _AMPLITUDE_CUES),
        precision=_ordinal(lowered, _PRECISION_CUES),
        repetition_count=count,
    )


def _frame_for(entry: SchemaEntry) -> ReferenceFrame:
    deixis = entry.schema.deixis
    if deixis is not None and deixis is not Deixis.NEUTRAL:
        return ReferenceFrame.INTRINSIC
    return ReferenceFrame.ABSOLUTE


def _boundary_for(entry: SchemaEntry) -> BoundaryCondition:
    contour = entry.schema.contour
    if contour is not None and contour.value == "oscillating":
        return BoundaryCondition.DIRECTION_REVERSED
    if entry.schema.stative is not None:
        return BoundaryCondition.DWELL
    return BoundaryCondition.TERMINUS
