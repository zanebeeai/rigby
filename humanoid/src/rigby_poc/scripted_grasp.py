"""A fixed controller that picks the block up, with no model in the loop.

This exists to separate two questions that were confounded for most of the
project: whether the BODY can grasp, and whether the selector chooses well. Every
failure was attributed to the selector until this was written, and none of them
were -- the hand could not have closed on anything, whatever it decided.

So this is the baseline. It makes no decisions worth the name: five phases in a
fixed order, advanced by thresholds. If it stops lifting after a change to the
primitives or the physics, that change broke the body, and no amount of better
choosing will hide it.

Measured on the authored scene: 19.99 cm of lift, held, with two opposing pairs
throughout the carry and every fingertip outside the block by at least 4.5 mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .closed_loop import (
    Sensing,
    Step,
    SuperPrimitiveSelector,
    object_in_grasp,
    palm_to_object_m,
    super_primitives,
)
from .models import Hand

#: The phases, in order: what to run and how hard.
PHASES: tuple[tuple[str, float], ...] = (
    ("move_to", 1.0),      # cross the room, arrive square to a face
    ("open_grip", 1.0),    # open to the object's size before arriving at it
    ("move_to", 1.0),      # advance until the object is INSIDE the opening
    ("close_grip", 0.4),   # arrive on the object, lightly
    ("close_grip", 1.0),   # squeeze hard enough to carry it
    ("lift", 0.05),        # raise, still gripping
)

#: Palm-to-object distance at which the approach has arrived, metres.
_ARRIVED_M = 0.10
#: How near the grasp line the object must be before this controller tries to
#: close, as a fraction of the object's own half-width. The measure runs from
#: the object's CENTRE, so a wider object sits further out at the same quality
#: of grasp and a fixed number of metres is a threshold fitted to one block.
#:
#: Deliberately TIGHTER than close_grip's own refusal, which is not a
#: duplication of it. They answer different questions: the hand refuses what is
#: impossible -- an object so far off the line that closing could only push it
#: -- while this waits for what is GOOD. Collapsing the two and closing as soon
#: as the hand would permit took the sweep from five successes to three, because
#: every early attempt that fails also shoves the block out of reach.
_ENGULFED_FRACTION = 0.8
#: How long to hold the open shape before advancing into the object, seconds.
_OPEN_DWELL_S = 1.0


@dataclass
class ScriptedGrasp(SuperPrimitiveSelector):
    """Approach, open, engulf, close, squeeze, lift."""

    phase: int = 0
    pairs_seen: int = 0
    gripped_at: float | None = None
    _by_name: dict[str, Any] = field(default_factory=dict, repr=False)

    def _primitive(self, name: str):
        if not self._by_name:
            self._by_name = {p.name: p for p in super_primitives(self.hand)}
        return self._by_name.get(name)

    def pairs(self, sensing: Sensing) -> int:
        """Loaded thumb-finger pairs: the shape a grasp has to have."""
        if sensing.contact_force_n.get("thumb", 0.0) < 0.5:
            return 0
        return sum(
            1 for digit in ("index", "middle", "ring", "little")
            if sensing.contact_force_n.get(digit, 0.0) >= 0.5
        )

    def advance(self, sensing: Sensing) -> None:
        pairs = self.pairs(sensing)
        self.pairs_seen = max(self.pairs_seen, pairs)
        if self.phase == 0 and palm_to_object_m(sensing, self.hand) <= _ARRIVED_M:
            self.phase = 1
        elif self.phase == 1 and sensing.time_s > _OPEN_DWELL_S:
            self.phase = 2
        elif self.phase == 2 and object_in_grasp(sensing, self.hand) <= (
                float(min(sensing.object_half_m)) * _ENGULFED_FRACTION):
            self.phase = 3
        elif self.phase == 3 and pairs >= 1:
            self.phase = 4
            self.gripped_at = sensing.time_s
        elif self.phase == 4 and pairs >= 1:
            self.phase = 5

    def choose(self, sensing: Sensing, step: Step) -> tuple[str, float, dict[str, Any]]:
        self.advance(sensing)
        name, amount = PHASES[self.phase]
        primitive = self._primitive(name)
        if primitive is None:
            return "hold", 0.0, {}
        try:
            rotations = primitive.solve(sensing, self.hand, amount)
        except Exception:  # noqa: BLE001
            return "hold", 0.0, {}
        return name, amount, rotations


def lift_achieved(frames) -> dict[str, float]:
    """How far the object actually rose, and whether it stayed up."""
    import numpy as np

    heights = [f.objects["block"].translation.y for f in frames]
    positions = [f.objects["block"].translation for f in frames]
    start, end = positions[0], positions[-1]
    return {
        "peak_lift_m": float(max(heights) - heights[0]),
        "final_lift_m": float(heights[-1] - heights[0]),
        "displaced_m": float(np.linalg.norm(np.asarray(
            [end.x - start.x, end.y - start.y, end.z - start.z]))),
    }
