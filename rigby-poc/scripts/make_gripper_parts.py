"""Generate the gripper's per-part movement vocabulary from its own manifest.

Same schema as config/body_parts.v1.json, which the humanoid has used all along:
a move is a SIGNED FRACTION of that joint's declared range, so a move cannot
name an out-of-range pose and the vocabulary cannot disagree with the limits.
Generated rather than written, for the same reason -- the ranges live in one
place and this is derived from them.

Each segment gets the moves that segment can actually make. The point of naming
them is that "fold the elbow" is a thing a planner can ask for and a search can
try, where "joint 2, minus 0.3 radians" is neither.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rigby_poc.gripper.body.manifest import spec  # noqa: E402

#: What each joint is for, and the moves worth naming on it. Values are signed
#: fractions of the declared range: +1.0 is the positive limit, -1.0 the
#: negative one. Nothing here goes to the limit, because a move that ends at a
#: hard stop is a move with no room to be corrected.
MOVES: dict[str, dict] = {
    "base": {
        "joint": "base", "carries": "the whole arm",
        "moves": {"hold": None, "neutral": 0.0,
                  "swing_left": 0.45, "swing_right": -0.45,
                  "nudge_left": 0.12, "nudge_right": -0.12},
    },
    "segment_1": {
        "joint": "segment_1", "carries": "the upper arm and everything past it",
        # Signs MEASURED, not assumed. Positive lift drives the hand DOWN on
        # this arm, so "raise" is negative-going here. Naming these the way they
        # read on the joint rather than the way they read on the body gives a
        # planner a vocabulary that lies to it: it asks to raise and the arm
        # descends, and the mistake looks like bad reasoning rather than a bad
        # dictionary.
        "moves": {"hold": None, "neutral": 0.0,
                  "raise": 0.5, "lower": -0.5,
                  "raise_a_little": 0.15, "lower_a_little": -0.15},
    },
    "segment_2": {
        "joint": "segment_2", "carries": "the forearm and the hand",
        "moves": {"hold": None, "neutral": 0.0,
                  "fold": 0.5, "extend": -0.5,
                  "fold_a_little": 0.15, "extend_a_little": -0.15},
    },
    "segment_3": {
        "joint": "segment_3", "carries": "the hand, and with it the camera",
        # Also measured: positive wrist points the hand UP.
        "moves": {"hold": None, "neutral": 0.0,
                  "tilt_down": -0.5, "tilt_up": 0.5,
                  "tilt_down_a_little": -0.15, "tilt_up_a_little": 0.15},
    },
    "gripper": {
        "joint": "finger_left", "also": "finger_right",
        "carries": "nothing; it is the end of the chain",
        "moves": {"hold": None, "open": 1.0, "close": 0.0,
                  "open_a_little": 0.35, "close_a_little": 0.12},
    },
}


def build() -> dict:
    document = spec()
    limits = {j["name"]: j for j in document["kinematics"]["joints"]}
    parts: dict = {}
    move_sets: dict = {}

    for part, plan in MOVES.items():
        joint = limits[plan["joint"]]
        span = joint.get("range_deg") or joint.get("range_m")
        kind = "deg" if "range_deg" in joint else "m"
        parts[part] = {
            "parent": None if part == "base" else list(MOVES)[
                list(MOVES).index(part) - 1],
            "joints": [plan["joint"]] + ([plan["also"]] if "also" in plan else []),
            "carries": plan["carries"],
            "envelope": {plan["joint"]: {f"range_{kind}": span, "enforced": True}},
        }
        move_sets[part] = {
            name: ({} if value is None else {plan["joint"]: float(value)})
            for name, value in plan["moves"].items()
        }
        if "also" in plan:
            for name, value in plan["moves"].items():
                if value is not None:
                    move_sets[part][name][plan["also"]] = float(value)

    return {
        "schema_version": "1.0",
        "$comment": "Per-part movement vocabulary for the gripper, in the same "
                    "schema as config/body_parts.v1.json. A move value is a "
                    "SIGNED FRACTION of that joint's declared range: +1.0 is the "
                    "positive limit, -1.0 the negative one. A move therefore "
                    "cannot name an out-of-range pose, and the limits and the "
                    "vocabulary cannot disagree.",
        "generated_from": "gripper.v1.json",
        "value_convention": {
            "units": "fraction of the joint's declared range",
            "positive": "value * range[1]",
            "negative": "abs(value) * range[0]",
            "range": [-1.0, 1.0],
        },
        "$why_named_moves": "A planner can ask for 'fold the elbow'. It cannot "
                            "usefully ask for 'joint 2, minus 0.3 radians', and "
                            "a search told to try every joint at every "
                            "magnitude spends its time on motions no one wants. "
                            "Naming the moves is what makes the vocabulary "
                            "something to reason in rather than a number line.",
        "parts": parts,
        "move_sets": move_sets,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / (
        "src/rigby_poc/gripper/body/gripper_parts.v1.json")
    document = build()
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"{out.name}: {len(document['parts'])} parts, "
          f"{sum(len(m) for m in document['move_sets'].values())} moves")
    for part, moves in document["move_sets"].items():
        print(f"  {part:<9} {', '.join(sorted(moves))}")
