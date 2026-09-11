"""Which regions a state is solving, which gates mean anything, and where to look.

The inspector without this runs every gate in every state and compiles a repair
proposal for every region that could conceivably be at fault. That is why a
candidate costs 45 to 94 seconds: it re-checks digit opposition while the arm is
still in transit, and spends a full rigid-body simulation establishing that
fingers which have not moved have still not moved.

A state controller makes both cheaper and more correct at once, and the
correctness is the more important half. Repairing the digits during ``lift``
does not merely waste a simulation -- it changes the grip that is at that moment
carrying the object, which is how a candidate that was holding the block drops
it.

The gaze half exists because the head had no part in a grab at all. Gaze is
wired in ``compiler._compile_composite`` and nowhere else, so through an entire
pickup the head stayed in its idle pose: the one moment worth watching, the
fingers closing, was never in view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "motion_states.v1.json"
)

GAZE_TARGETS = ("object", "grasp", "forward")


class MotionStateError(ValueError):
    """A state catalog that does not describe a usable controller."""


@dataclass(frozen=True)
class MotionState:
    name: str
    order: int
    goal: str
    active_regions: tuple[str, ...]
    inert_regions: tuple[str, ...]
    gates: tuple[str, ...]
    gaze: str
    rationale: str

    def solves(self, region: str) -> bool:
        """Whether a repair to ``region`` is meaningful in this state.

        ``region`` may be sided (``right_arm``); the catalog names sides
        generically because a state's structure does not depend on handedness.
        """
        stem = region.removeprefix("left_").removeprefix("right_")
        return stem in self.active_regions

    def runs(self, gate: str) -> bool:
        return gate in self.gates


@lru_cache(maxsize=1)
def state_catalog() -> dict[str, Any]:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    states = document["states"]
    if not states:
        raise MotionStateError("the state catalog is empty")
    orders = [int(s["order"]) for s in states.values()]
    if sorted(orders) != list(range(len(states))):
        raise MotionStateError("state orders must be contiguous from zero")
    for name, spec in states.items():
        if spec["gaze"] not in GAZE_TARGETS:
            raise MotionStateError(
                f"state {name!r} names unknown gaze target {spec['gaze']!r}"
            )
        overlap = set(spec["active_regions"]) & set(spec["inert_regions"])
        if overlap:
            raise MotionStateError(
                f"state {name!r} lists {sorted(overlap)} as both active and inert"
            )
        if not spec.get("rationale"):
            raise MotionStateError(
                f"state {name!r} has no rationale; a scoping decision that is not "
                "explained cannot be reviewed when a gate stops firing"
            )
    return document


@lru_cache(maxsize=1)
def states() -> dict[str, MotionState]:
    document = state_catalog()
    return {
        name: MotionState(
            name=name,
            order=int(spec["order"]),
            goal=spec["goal"],
            active_regions=tuple(spec["active_regions"]),
            inert_regions=tuple(spec["inert_regions"]),
            gates=tuple(spec["gates"]),
            gaze=spec["gaze"],
            rationale=spec["rationale"],
        )
        for name, spec in document["states"].items()
    }


def state_for(kind: str) -> MotionState | None:
    """The state a primitive kind runs in, or None if it is not controlled."""
    return states().get(kind)


def gates_for(kind: str) -> tuple[str, ...]:
    state = state_for(kind)
    return state.gates if state else ()


def every_gate() -> frozenset[str]:
    """Every gate any state runs. Used to catch a gate no state ever reaches."""
    return frozenset(g for state in states().values() for g in state.gates)


def plan(intent: str = "grab") -> tuple[MotionState, ...]:
    """The ordered state sequence for an intent.

    This is the 'move arm to box, grab box, move arm up' plan, made explicit and
    ordered rather than implied by the primitive list.
    """
    document = state_catalog()
    if intent not in document["applies_to_intents"]:
        return ()
    return tuple(sorted(states().values(), key=lambda s: s.order))


def controlled_intents() -> frozenset[str]:
    return frozenset(state_catalog()["applies_to_intents"])


def state_for_primitive(program: Any, index: int, kind: str) -> "MotionState | None":
    """The state a primitive runs in.

    The planner's choice when it made one, the kind-matched default otherwise.
    Keeping the fallback means a program authored before the field existed, or
    by a provider that does not emit it, still gets a controlled gaze and scoped
    repairs rather than silently losing both.
    """
    chosen = list(getattr(program, "motion_states", ()) or ())
    if chosen and 0 <= index < len(chosen):
        return states().get(chosen[index])
    return state_for(kind)


def default_state_sequence(kinds: list[str]) -> list[str]:
    """The kind-matched sequence, for a planner with no opinion of its own."""
    known = states()
    return [k for k in kinds if k in known] if all(k in known for k in kinds) else []
