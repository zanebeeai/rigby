"""The jaws as three states, not a rule that guesses what the plan meant.

This file used to hold `squeeze_for`, which read the current goal's terms and
worked out whether the plan wanted the jaws wider, narrower, or left alone. It
was wrong four times:

  1. clamped whenever the goal was not asking for wider, so every alignment
     move squeezed the jaws shut while the hand was near the block;
  2. rewritten to drop that clause, which meant a grip could never be released
     -- both pads read 3 N for forty seconds while the planner asked three
     times for 86 mm;
  3. and twice more in between, each fix losing a case the previous version
     handled.

Every one of those was the same mistake: deriving a persistent intention from
whatever number happened to be in front of it. OPEN, CLOSE and HOLD are states.
A state is set once and stays set, it cannot be misread from a goal that says
nothing about the jaws, and there is no expression to get wrong.

The search never touches the fingers now. It moves the arm; the jaws are told
what to be.
"""

from __future__ import annotations

import numpy as np

#: Newtons of commanded grip, when gripping at all.
GRIP_N = 12.0

#: The three states, and what each one is for.
STATES = {
    "open": "jaws driven to their widest, no grip force. Use before reaching "
            "around something.",
    "close": "jaws driven shut with grip force. They stop on whatever is "
             "between them; on nothing they reach the 7 mm floor and report no "
             "load.",
    "hold": "jaws stay exactly where they are, grip force on. Use while "
            "carrying: a carry must not need a decision every frame to remain "
            "a carry.",
}
DEFAULT = "open"


def jaw_command(state: str, body, commanded: np.ndarray) -> tuple[np.ndarray, float]:
    """The finger targets and the squeeze for a state.

    Returns a copy of `commanded` with the two finger joints set, and the grip
    force to apply. Only the fingers are touched -- the arm is the search's.
    """
    out = np.asarray(commanded, dtype=float).copy()
    low, high = body.model.jnt_range[body.model.joint("finger_left").id]
    if state == "open":
        out[4] = out[5] = float(high)
        return out, 0.0
    if state == "close":
        out[4] = out[5] = float(low)
        return out, GRIP_N
    # hold: leave the fingers where the controller already had them.
    return out, GRIP_N


def catalogue() -> str:
    return chr(10).join(f"  {name:<7} {what}" for name, what in STATES.items())
