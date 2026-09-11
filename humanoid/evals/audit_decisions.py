"""Read a directed run's transcript and ask, of every decision the model made:
was it missing something, or did it have what it needed and choose badly?

The distinction matters because the two have opposite fixes. A decision made
without the relevant number is an architecture problem -- the information never
reached the model, and no better model would have done better. A decision made
with the number in hand, contradicting it, is a reasoning problem, and the fix
is a clearer prompt or a stronger model.

Everything here is read off the recorded transcript. Nothing is re-simulated and
nothing is asked of the model again, so the audit costs nothing and cannot
disagree with what actually happened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

#: Numbers that, if present in the decision's own sensed block, mean the model
#: could have known the thing named. Read from the transcript, not recomputed.
_TELLS = {
    "object_between_jaws": "whether the block was actually in the jaws",
    "object_in_hand_view": "whether the hand camera could see the block",
    "holding": "whether anything was being gripped",
    "object_seen": "whether the block had been located at all",
    "palm_to_object_m": "how far the hand was from the block",
    "range_ahead_m": "how far the surface ahead was",
    "tip_force_left_n": "what the left pad felt",
    "tip_force_right_n": "what the right pad felt",
}


def _verdict(entry: dict, following: dict | None) -> tuple[str, str]:
    """One decision, judged against what the model was holding at the time."""
    sensed = entry.get("sensed") or {}
    asked = entry.get("target", "")
    movable = entry.get("packages_that_could_move_it")

    # 1. Did it ask for something nothing could move?
    if movable == []:
        return ("CHOSE BADLY",
                f"asked for {asked}, which the capability table in the same "
                f"message reported as movable by nothing")

    # 2. Did it try to close with nothing in the jaws, while being told so?
    closing = asked == "grip_tip_spread_m" and float(entry.get("value", 1)) < 0.02
    if closing and "object_between_jaws" in sensed:
        if not sensed["object_between_jaws"]:
            return ("CHOSE BADLY",
                    "asked to close the jaws while object_between_jaws was 0 "
                    "in the same message")
    if closing and "object_between_jaws" not in sensed:
        return ("MISSING INFORMATION",
                "asked to close the jaws and was never told whether anything "
                "was between them")

    # 3. Did it act on the object without having been told where it is?
    needs_sight = asked in ("palm_to_object_m", "object_in_grasp_m",
                            "palm_facing", "object_over_target_m",
                            "object_above_rim_m")
    if needs_sight and not sensed.get("object_seen", 0.0):
        return ("MISSING INFORMATION",
                f"asked for {asked}, which cannot be read until the block is "
                "seen, and it had not been")

    # 4. Is it working on a step that PRESUPPOSES a grasp, without one?
    #
    # This is the failure a per-decision reading misses. Every choice can be
    # locally sensible inside a premise that is false: told it is on "lift the
    # block", the model quite correctly asks for the hand to go up, and does it
    # again, and again -- while the jaws are empty and holding reads 0 in the
    # very message it is answering. Nothing is wrong with the reasoning. What is
    # wrong is that the step was declared finished when it had not happened.
    step = (entry.get("step_text") or "").lower()
    presumes_grip = any(word in step for word in
                        ("lift", "carry", "move the", "lower the", "release",
                         "place", "put"))
    if presumes_grip and "holding" in sensed and not sensed["holding"]:
        return ("FALSE PREMISE",
                f"working on {entry.get('step_text')!r} with holding=0 -- the "
                "grasp it depends on never happened")

    # 5. Did it declare a step done that the sensors contradict?
    if entry.get("step_done") and "holding" in sensed:
        if "grasp" in step or "grip" in step or "pick" in step:
            if not sensed["holding"]:
                return ("FALSE PREMISE",
                        "declared the grasp step finished while holding=0")

    # 6. Did the goal it just held actually move? Only judgeable in hindsight.
    if following is not None and "moved" in (following or {}):
        if abs(float(following.get("moved", 0.0))) < 0.02:
            return ("CHOSE BADLY",
                    f"held {asked} for "
                    f"{following.get('held_for_s', '?')}s and it moved "
                    f"{following.get('moved')}")

    return ("REASONABLE", "")


def audit(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    transcript = document.get("transcript") or []
    decisions = [e for e in transcript if "target" in e]
    memory = {}
    for entry in transcript:
        for record in entry.get("what_you_already_tried") or []:
            memory[(record.get("at_s"), record.get("asked"))] = record

    print(f"  task: {document.get('task', '?')}")
    steps = next((e.get("steps") for e in transcript if e.get("steps")), [])
    for index, line in enumerate(steps):
        print(f"    {index}. {line}")
    print()
    print(f"  {len(decisions)} decisions")
    print()

    tally: dict[str, int] = {}
    for entry in decisions:
        after = memory.get((round(float(entry.get("t", 0.0)), 1),
                            entry.get("target")))
        kind, why = _verdict(entry, after)
        tally[kind] = tally.get(kind, 0) + 1
        head = (f"t={float(entry.get('t', 0)):5.1f}  "
                f"{entry.get('primitive') or entry.get('target', '?')}"
                f" = {entry.get('value', '')}")
        print(f"  {head}")
        print(f"      step {entry.get('step')}: {entry.get('step_text')}")
        print(f"      said: {entry.get('why', '')[:110]}")
        print(f"      {kind}" + (f" -- {why}" if why else ""))
        sensed = entry.get("sensed") or {}
        knew = [k for k in _TELLS if k in sensed]
        missing = [k for k in _TELLS if k not in sensed]
        if missing:
            print(f"      was NOT told: {', '.join(missing)}")
        print()

    # CAN IT SEE? Its own report against what was actually the case. These are
    # the judgements no instrument in the system makes -- which side of the
    # block the hand is on, whether the jaws are wide enough for it -- so
    # whether the model can make them is worth knowing before anything is built
    # on top of them.
    checks = ("hand_vs_block", "hand_height_vs_block", "jaws_vs_block_width")
    scored = [(e.get("seeing") or {}, e.get("seeing_truth") or {})
              for e in decisions if e.get("seeing") and e.get("seeing_truth")]
    if scored:
        print("  " + "-" * 60)
        print("  WHAT IT SAID IT SAW, against what was the case")
        for field in checks:
            hits = [1 for said, was in scored
                    if said.get(field) and said.get(field) == was.get(field)]
            asked = [1 for said, _ in scored if said.get(field)]
            if asked:
                print(f"    {field:<24} {len(hits)}/{len(asked)} right")
        sure = [(s, w) for s, w in scored if s.get("confidence") == "sure"]
        if sure:
            right = sum(1 for s, w in sure
                        if s.get("hand_vs_block") == w.get("hand_vs_block"))
            print(f"    when it said it was SURE about the side: "
                  f"{right}/{len(sure)} right")
        print()
        for said, was in scored[:6]:
            marks = " ".join(
                f"{f.split('_')[0]}:{said.get(f)}"
                + ("=" if said.get(f) == was.get(f) else f"!={was.get(f)}")
                for f in checks if said.get(f))
            print(f"      {marks}")
        print()

    print("  " + "-" * 60)
    for kind, count in sorted(tally.items()):
        print(f"  {kind:<20} {count}")
    total = sum(tally.values()) or 1
    bad = tally.get("CHOSE BADLY", 0)
    gap = tally.get("MISSING INFORMATION", 0)
    print()
    print(f"  Of {total} decisions, {gap} could not have been made well with "
          f"what the model was given,")
    print(f"  and {bad} contradicted something it was holding at the time.")
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="frontend/public/directed-vlm.json")
    args = parser.parse_args()
    audit(Path(args.source))


if __name__ == "__main__":
    sys.exit(main())
