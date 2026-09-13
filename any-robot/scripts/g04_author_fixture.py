"""Author the G04 semantic fixture: 60 canonical cases and 120 language cases.

Every case is a prompt and the body-neutral reading a competent reader of the
closed class would give it: which inventory entries, in what order, at what
remove, with what manner, posture and stated quantities -- or which typed
refusal. The expected reading is written here by hand, case by case, and
labelled ``internal``: it was authored inside this repository by the same
engineering pass that built the planner, not by an independent reviewer. The
catalog's research-language gate stays pending on U3_independent_review until
someone outside has labelled or reviewed the language cases; the engineering
fixture does not wait for that.

The fixture is registered by hash before any scored run; the campaign refuses
a fixture that does not hash to its registration.

    python any-robot/scripts/g04_author_fixture.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "assets/general/research-protocols/g04-semantics-v1"


def seg(entry_id: str, remove: str = "medial", *, count: int | None = None, posture: dict | None = None, quantities: list | None = None, **manner: int) -> dict:
    for axis in manner:
        assert axis in ("speed", "effort", "smoothness", "rhythm", "amplitude", "repetition", "precision"), axis
    return {
        "entry_id": entry_id,
        "remove": "adjacent" if posture is not None else remove,
        "manner": {axis: value for axis, value in manner.items() if value},
        "repetition_count": count,
        "posture": posture,
        "quantities": quantities or [],
    }


def q(text: str, value_m: float) -> dict:
    return {"text": text, "value_m": value_m}


PEACE = {"group": "opposed", "selected_count": 2, "selected": "extended", "remainder": "flexed", "opposing": "flexed"}
ONE_FINGER = {"group": "opposed", "selected_count": 1, "selected": "extended", "remainder": "flexed", "opposing": "flexed"}
THUMBS_UP = {"group": "opposed", "selected_count": 0, "selected": "extended", "remainder": "flexed", "opposing": "extended"}
FIST = {"group": "opposed", "selected_count": 0, "selected": "extended", "remainder": "flexed", "opposing": "flexed"}
OPEN_HAND = {"group": "opposed", "selected_count": 0, "selected": "extended", "remainder": "extended", "opposing": "extended"}
THREE = {"group": "opposed", "selected_count": 3, "selected": "extended", "remainder": "flexed", "opposing": "flexed"}

NO_QUANTITY = {"kind": "quantity"}


def no_entry(entry_id: str) -> dict:
    return {"kind": "entry", "entry_id": entry_id}


def no_remove(remove: str) -> dict:
    return {"kind": "remove", "remove": remove}


def no_speed_above(value: int) -> dict:
    return {"kind": "speed_above", "value": value}


def no_posture(canonical: dict) -> dict:
    return {"kind": "posture", "posture": canonical}


_cases: list[dict] = []


def case(kind: str, family: str, prompt: str, segments: list[dict] | None = None, *, refusal: list[str] | None = None, prohibited: list[dict] | None = None, spans: list[str] | None = None, notes: str = "") -> None:
    assert (segments is None) != (refusal is None), prompt
    index = sum(1 for item in _cases if item["kind"] == kind) + 1
    case_id = f"{'C' if kind == 'canonical' else 'L'}{index:03d}"
    _cases.append({
        "case_id": case_id, "kind": kind, "family": family, "prompt": prompt,
        "expected": {"segments": segments} if segments is not None else {"refusal": refusal},
        "prohibited": prohibited or [], "spans": spans or [],
        "label_provenance": "internal", "notes": notes,
    })


# ==========================================================================
# Canonical cases: every entry of the inventory, every remove, every manner
# axis at both poles, repetition, sequences, object and frame roles.
# ==========================================================================

C = "canonical"
case(C, "entry", "reach out in front of you", [seg("reach_to_point")], spans=["reach_to_point", "medial"])
case(C, "extent", "reach out a little", [seg("reach_to_point", "proximal")], spans=["proximal"])
case(C, "extent", "reach far out", [seg("reach_to_point", "distal")], spans=["distal"])
case(C, "extent", "reach right there", [seg("reach_to_point", "adjacent")], spans=["adjacent"])
case(C, "entry", "reach out as far as you can", [seg("reach_to_edge", "distal")], spans=["reach_to_edge", "boundary"])
case(C, "entry", "swing over to it", [seg("swing_to_point")], spans=["swing_to_point", "arced"])
case(C, "entry", "lower down onto the table", [seg("descend_to_surface")], spans=["descend_to_surface", "world_ground"])
case(C, "entry", "come down onto the table over an arc", [seg("arc_to_surface")], spans=["arc_to_surface"], notes="no offline cue: the recognizer reads the surface descent as straight")
case(C, "entry", "reach into the box", [seg("enter_volume")], spans=["enter_volume", "volume"])
case(C, "entry", "come back", [seg("retract_from_point")], spans=["retract_from_point", "from"])
case(C, "entry", "lift off the table", [seg("retract_from_surface")], spans=["retract_from_surface"])
case(C, "entry", "move away", [seg("move_away")], spans=["move_away", "away"])
case(C, "entry", "move toward it", [seg("move_toward")], spans=["move_toward", "toward"])
case(C, "entry", "sweep across the workspace", [seg("traverse_line")], spans=["traverse_line", "via", "line"])
case(C, "entry", "arc across the workspace", [seg("traverse_line_arced")], spans=["traverse_line_arced"])
case(C, "entry", "trace a circle", [seg("circle_axis")], spans=["circle_axis", "gravity_axis", "circular"])
case(C, "entry", "wave", [seg("oscillate_about_point")], spans=["oscillate_about_point", "oscillating"])
case(C, "entry", "go back and forth", [seg("oscillate_along_line")], spans=["oscillate_along_line"])
case(C, "entry", "pass through the waypoint on the way over", [seg("pass_through_point")], spans=["pass_through_point"], notes="no offline cue")
case(C, "frame", "beckon", [seg("draw_hither")], spans=["draw_hither", "hither", "intrinsic", "front_axis"])
case(C, "frame", "shoo it away", [seg("push_thither")], spans=["push_thither", "thither", "intrinsic"])
case(C, "entry", "close in until you touch it", [seg("approach_to_contact")], spans=["approach_to_contact", "target_object", "contact_made"], notes="no offline cue; contact scene required to ground")
case(C, "entry", "pick it up", [seg("transport_object")], spans=["transport_object", "target_object"])
case(C, "entry", "hand it over", [seg("hand_across")], spans=["hand_across", "secondary_effector"])
case(C, "entry", "look at it", [seg("track_with_gaze")], spans=["track_with_gaze", "sensor"])
case(C, "entry", "turn the wrist", [seg("orient_effector")], spans=["orient_effector", "stative"])
case(C, "entry", "hold still", [seg("hold_still")], spans=["hold_still", "dwell"])
case(C, "entry", "press down on it", [seg("press")], spans=["press", "apply_force"], notes="no offline cue")
case(C, "posture", "make a peace sign", [seg("configure_effector", posture=PEACE)], spans=["configure_effector", "posture"])
case(C, "posture", "make a fist", [seg("configure_effector", posture=FIST)], spans=["posture"])
case(C, "posture", "give a thumbs up", [seg("configure_effector", posture=THUMBS_UP)], spans=["posture", "opposing"])
case(C, "posture", "open your hand", [seg("configure_effector", posture=OPEN_HAND)], spans=["posture"])
case(C, "posture", "point with one finger", [seg("configure_effector", posture=ONE_FINGER)], spans=["posture"])
case(C, "posture", "three fingers up", [seg("configure_effector", posture=THREE)], spans=["posture"])
case(C, "speed", "reach out quickly", [seg("reach_to_point", speed=1)], spans=["speed+1"])
case(C, "speed", "reach out slowly", [seg("reach_to_point", speed=-1)], spans=["speed-1"])
case(C, "speed", "reach out as fast as you can", [seg("reach_to_point", speed=2)], spans=["speed+2"])
case(C, "speed", "reach out very slowly", [seg("reach_to_point", speed=-2)], spans=["speed-2"])
case(C, "amplitude", "give a wide wave", [seg("oscillate_about_point", amplitude=1)], spans=["amplitude+1"])
case(C, "amplitude", "give a small wave", [seg("oscillate_about_point", amplitude=-1)], spans=["amplitude-1"])
case(C, "amplitude", "give a huge wave", [seg("oscillate_about_point", amplitude=2)], spans=["amplitude+2"])
case(C, "effort", "lift it firmly", [seg("retract_from_surface", effort=1)], spans=["effort+1"])
case(C, "effort", "lower it gently", [seg("descend_to_surface", speed=-1, effort=-1)], spans=["effort-1", "speed-1"])
case(C, "smoothness", "sweep across smoothly", [seg("traverse_line", smoothness=1)], spans=["smoothness+1"])
case(C, "smoothness", "wave jerkily", [seg("oscillate_about_point", smoothness=-1)], spans=["smoothness-1"])
case(C, "precision", "reach out precisely", [seg("reach_to_point", precision=1)], spans=["precision+1"])
case(C, "precision", "reach out roughly", [seg("reach_to_point", precision=-1)], spans=["precision-1"])
case(C, "repetition", "wave twice", [seg("oscillate_about_point", count=2)], spans=["repetition_count"])
case(C, "repetition", "wave three times", [seg("oscillate_about_point", count=3)], spans=["repetition_count"])
case(C, "repetition", "circle around 4 times", [seg("circle_axis", count=4)], spans=["repetition_count"])
case(C, "repetition", "go back and forth a couple of times", [seg("oscillate_along_line", count=2)], spans=["repetition_count"])
case(C, "sequence", "reach out then come back", [seg("reach_to_point"), seg("retract_from_point")], spans=["sequence"])
case(C, "sequence", "reach out as far as you can and then come back", [seg("reach_to_edge", "distal"), seg("retract_from_point")], spans=["sequence", "distal"])
case(C, "sequence", "lower onto the table, then lift off", [seg("descend_to_surface"), seg("retract_from_surface")], spans=["sequence", "world_ground"])
case(C, "sequence", "wave twice then hold still", [seg("oscillate_about_point", count=2), seg("hold_still")], spans=["sequence", "repetition_count", "dwell"])
case(C, "sequence", "pick it up then hand it over", [seg("transport_object"), seg("hand_across")], spans=["sequence", "target_object", "secondary_effector"])
case(C, "sequence", "look at it then move toward it", [seg("track_with_gaze"), seg("move_toward")], spans=["sequence", "sensor"])
case(C, "sequence", "reach far out, then come back halfway", [seg("reach_to_point", "distal"), seg("retract_from_point", "medial")], spans=["sequence", "distal", "medial"])
case(C, "sequence", "make a fist then open your hand", [seg("configure_effector", posture=FIST), seg("configure_effector", posture=OPEN_HAND)], spans=["sequence", "posture"])
case(C, "sequence", "turn the wrist then hold still", [seg("orient_effector"), seg("hold_still")], spans=["sequence", "stative"])
assert sum(1 for item in _cases if item["kind"] == C) == 60, sum(1 for item in _cases if item["kind"] == C)

# ==========================================================================
# Language cases: paraphrase, minimal contrast, negation, ambiguous deixis,
# explicit quantity, manner, sequence, unsupported.
# ==========================================================================

L = "language"
# -- paraphrases ----------------------------------------------------------------
case(L, "paraphrase", "extend your arm out in front", [seg("reach_to_point")])
case(L, "paraphrase", "stretch out ahead", [seg("reach_to_point")])
case(L, "paraphrase", "put your hand out there", [seg("reach_to_point", "distal")])
case(L, "paraphrase", "go all the way out", [seg("reach_to_edge", "distal")], notes="'all the way' is the edge of reach; the recognizer reads a distal point")
case(L, "paraphrase", "extend fully", [seg("reach_to_edge", "distal")])
case(L, "paraphrase", "stretch to your maximum reach", [seg("reach_to_edge", "distal")])
case(L, "paraphrase", "bring it back", [seg("retract_from_point")])
case(L, "paraphrase", "return to where you were", [seg("retract_from_point")])
case(L, "paraphrase", "pull back a little", [seg("retract_from_point", "proximal")])
case(L, "paraphrase", "withdraw slowly", [seg("retract_from_point", speed=-1)])
case(L, "paraphrase", "wiggle your hand", [seg("oscillate_about_point")])
case(L, "paraphrase", "shake it", [seg("oscillate_about_point")])
case(L, "paraphrase", "give a wave", [seg("oscillate_about_point")])
case(L, "paraphrase", "shuttle side to side", [seg("oscillate_along_line")])
case(L, "paraphrase", "go to and fro", [seg("oscillate_along_line")])
case(L, "paraphrase", "trace a loop around", [seg("circle_axis")])
case(L, "paraphrase", "orbit the axis", [seg("circle_axis")])
case(L, "paraphrase", "scan across the bench", [seg("traverse_line")])
case(L, "paraphrase", "traverse the workspace", [seg("traverse_line")])
case(L, "paraphrase", "descend onto the surface", [seg("descend_to_surface")])
case(L, "paraphrase", "come down to the table top", [seg("descend_to_surface")])
case(L, "paraphrase", "come up off the table", [seg("retract_from_surface")])
case(L, "paraphrase", "raise your hand", [seg("retract_from_surface")])
case(L, "paraphrase", "grab the block", [seg("transport_object")])
case(L, "paraphrase", "take hold of it", [seg("transport_object")])
case(L, "paraphrase", "pass it to the other hand", [seg("hand_across")])
case(L, "paraphrase", "aim the camera at it", [seg("track_with_gaze")])
case(L, "paraphrase", "rotate the tool", [seg("orient_effector")])
case(L, "paraphrase", "stay where you are", [seg("hold_still")])
case(L, "paraphrase", "clench your fist", [seg("configure_effector", posture=FIST)])
# -- minimal contrasts ------------------------------------------------------------
case(L, "minimal_contrast", "reach out just a little", [seg("reach_to_point", "proximal")], prohibited=[no_remove("distal"), NO_QUANTITY])
case(L, "minimal_contrast", "reach out a long way", [seg("reach_to_point", "distal")], prohibited=[no_remove("proximal"), NO_QUANTITY], notes="'a long way' is far; no recognizer cue")
case(L, "minimal_contrast", "wave slowly", [seg("oscillate_about_point", speed=-1)])
case(L, "minimal_contrast", "wave quickly", [seg("oscillate_about_point", speed=1)])
case(L, "minimal_contrast", "wave once", [seg("oscillate_about_point", count=1)])
case(L, "minimal_contrast", "wave twice over", [seg("oscillate_about_point", count=2)])
case(L, "minimal_contrast", "move toward the block", [seg("move_toward")], prohibited=[no_entry("move_away")])
case(L, "minimal_contrast", "move away from the block", [seg("move_away")], prohibited=[no_entry("move_toward")])
case(L, "minimal_contrast", "lift up off the table", [seg("retract_from_surface")], prohibited=[no_entry("descend_to_surface")])
case(L, "minimal_contrast", "lower down to the table", [seg("descend_to_surface")], prohibited=[no_entry("retract_from_surface")])
case(L, "minimal_contrast", "reach out gently", [seg("reach_to_point", speed=-1, effort=-1)])
case(L, "minimal_contrast", "reach out firmly", [seg("reach_to_point", effort=1)])
case(L, "minimal_contrast", "give a little wave", [seg("oscillate_about_point", amplitude=-1)], notes="'little' marks amplitude; the point of a wave is not a remove")
case(L, "minimal_contrast", "give a big wave", [seg("oscillate_about_point", amplitude=1)])
case(L, "minimal_contrast", "sweep across", [seg("traverse_line")], prohibited=[no_entry("traverse_line_arced")])
case(L, "minimal_contrast", "arc across", [seg("traverse_line_arced")], prohibited=[no_entry("traverse_line")])
case(L, "minimal_contrast", "reach out", [seg("reach_to_point")], prohibited=[no_remove("distal"), NO_QUANTITY])
case(L, "minimal_contrast", "reach out as far as possible", [seg("reach_to_edge", "distal")], prohibited=[no_remove("medial")])
case(L, "minimal_contrast", "close your hand", [seg("configure_effector", posture=FIST)], prohibited=[no_posture(OPEN_HAND)])
case(L, "minimal_contrast", "spread your fingers", [seg("configure_effector", posture=OPEN_HAND)], prohibited=[no_posture(FIST)])
# -- negation -------------------------------------------------------------------------
case(L, "negation", "don't wave, just reach out", [seg("reach_to_point")], prohibited=[no_entry("oscillate_about_point")])
case(L, "negation", "reach out but do not go all the way", [seg("reach_to_point")], prohibited=[no_remove("distal"), no_entry("reach_to_edge")])
case(L, "negation", "do not move", [seg("hold_still")], prohibited=[no_entry("reach_to_point")])
case(L, "negation", "wave, but not quickly", [seg("oscillate_about_point")], prohibited=[no_speed_above(0)], notes="'not quickly' removes the mark; it does not add a slow one")
case(L, "negation", "come back, not out", [seg("retract_from_point")], prohibited=[no_entry("reach_to_point")])
case(L, "negation", "no waving; hold still", [seg("hold_still")], prohibited=[no_entry("oscillate_about_point")])
case(L, "negation", "reach out, not too far", [seg("reach_to_point")], prohibited=[no_remove("distal")])
case(L, "negation", "don't pick it up, just touch it", [seg("approach_to_contact")], prohibited=[no_entry("transport_object")])
case(L, "negation", "never mind the circle, sweep across instead", [seg("traverse_line")], prohibited=[no_entry("circle_axis")])
case(L, "negation", "don't hold still, wave", [seg("oscillate_about_point")], prohibited=[no_entry("hold_still")])
case(L, "negation", "lower it, but not all the way down", [seg("descend_to_surface")], prohibited=[no_remove("distal")])
case(L, "negation", "not a fist, a thumbs up", [seg("configure_effector", posture=THUMBS_UP)], prohibited=[no_posture(FIST)])
# -- ambiguous deixis ------------------------------------------------------------------
case(L, "ambiguous_deixis", "come here", [seg("draw_hither")], notes="hither: toward the robot's own front; grounds only on a confirmed frame")
case(L, "ambiguous_deixis", "bring it in toward you", [seg("draw_hither")], notes="'toward you' is the robot's own front")
case(L, "ambiguous_deixis", "push it away from you", [seg("push_thither")])
case(L, "ambiguous_deixis", "move it away", [seg("move_away")], notes="no deictic centre named: neutral departure")
case(L, "ambiguous_deixis", "go over there", [seg("reach_to_point")], notes="'over there' names no remove the closed class distinguishes; unmarked")
case(L, "ambiguous_deixis", "back away from it", [seg("move_away")])
case(L, "ambiguous_deixis", "closer", [seg("move_toward")])
case(L, "ambiguous_deixis", "beckon them over", [seg("draw_hither")])
case(L, "ambiguous_deixis", "wave them off", [seg("push_thither")], prohibited=[no_entry("oscillate_about_point")], notes="'wave off' is a dismissal, not a wave")
case(L, "ambiguous_deixis", "reach toward me", [seg("move_toward")], notes="'me' is speaker-relative and the IR has no speaker; the neutral approach is the only honest reading")
# -- explicit quantities ------------------------------------------------------------------
case(L, "explicit_quantity", "reach out 5 cm", [seg("reach_to_point", quantities=[q("5 cm", 0.05)])])
case(L, "explicit_quantity", "reach out five centimetres", [seg("reach_to_point", quantities=[q("five centimetres", 0.05)])])
case(L, "explicit_quantity", "extend 10 cm", [seg("reach_to_point", quantities=[q("10 cm", 0.10)])])
case(L, "explicit_quantity", "move forward 0.1 m", [seg("reach_to_point", quantities=[q("0.1 m", 0.1)])])
case(L, "explicit_quantity", "reach out 3 inches", [seg("reach_to_point", quantities=[q("3 inches", 0.0762)])])
case(L, "explicit_quantity", "go out 50 mm", [seg("reach_to_point", quantities=[q("50 mm", 0.05)])])
case(L, "explicit_quantity", "reach out 5 cm then come back", [seg("reach_to_point", quantities=[q("5 cm", 0.05)]), seg("retract_from_point")])
case(L, "explicit_quantity", "come back 5 cm", [seg("retract_from_point", quantities=[q("5 cm", 0.05)])])
case(L, "explicit_quantity", "move 5 cm closer to it", [seg("move_toward", quantities=[q("5 cm", 0.05)])])
case(L, "explicit_quantity", "back off 5 cm", [seg("move_away", quantities=[q("5 cm", 0.05)])])
case(L, "explicit_quantity", "reach out 5 cm slowly", [seg("reach_to_point", speed=-1, quantities=[q("5 cm", 0.05)])])
case(L, "explicit_quantity", "reach out about 5 cm", [seg("reach_to_point", precision=-1, quantities=[q("5 cm", 0.05)])])
case(L, "explicit_quantity", "reach out a bit", [seg("reach_to_point", "proximal")], prohibited=[NO_QUANTITY], notes="a bit is a remove, never a number")
case(L, "explicit_quantity", "reach out roughly five centimetres", [seg("reach_to_point", precision=-1, quantities=[q("five centimetres", 0.05)])])
case(L, "explicit_quantity", "lift 5 cm", [seg("retract_from_surface", quantities=[q("5 cm", 0.05)])], notes="kept as stated; grounding refuses a distance on a surface departure, typed")
case(L, "explicit_quantity", "wave 5 cm", [seg("oscillate_about_point", quantities=[q("5 cm", 0.05)])], notes="kept as stated; grounding refuses a distance on an oscillation, typed")
case(L, "explicit_quantity", "reach out 5 cm, then come back 5 cm", [seg("reach_to_point", quantities=[q("5 cm", 0.05)]), seg("retract_from_point", quantities=[q("5 cm", 0.05)])])
case(L, "explicit_quantity", "reach out half a metre", [seg("reach_to_point", quantities=[q("half a metre", 0.5)])])
case(L, "explicit_quantity", "reach out 2 m", [seg("reach_to_point", quantities=[q("2 m", 2.0)])], notes="kept as stated; beyond every zoo body's reach, grounding refuses typed")
case(L, "explicit_quantity", "reach out twenty centimeters", [seg("reach_to_point", quantities=[q("twenty centimeters", 0.2)])])
# -- manner --------------------------------------------------------------------------------------
case(L, "manner", "wave as fast as you can", [seg("oscillate_about_point", speed=2)])
case(L, "manner", "reach out really slowly", [seg("reach_to_point", speed=-1)], notes="'really' intensifies without reaching the -2 pole the class reserves for crawl, inch, very slowly")
case(L, "manner", "sweep across steadily", [seg("traverse_line", smoothness=1)])
case(L, "manner", "wave abruptly", [seg("oscillate_about_point", smoothness=-1)])
case(L, "manner", "reach out exactly to the point", [seg("reach_to_point", precision=1)])
case(L, "manner", "give a tiny wave", [seg("oscillate_about_point", amplitude=-2)])
case(L, "manner", "give a sweeping wave", [seg("oscillate_about_point", amplitude=2)], prohibited=[no_entry("traverse_line")], notes="'sweeping' here is the size of a wave, not a traverse")
case(L, "manner", "lift it carefully", [seg("retract_from_surface", speed=-1, effort=-1)])
case(L, "manner", "press hard on it", [seg("press", effort=1)])
case(L, "manner", "wave three times quickly", [seg("oscillate_about_point", speed=1, count=3)])
case(L, "manner", "circle around twice slowly", [seg("circle_axis", speed=-1, count=2)])
case(L, "manner", "reach out briskly and precisely", [seg("reach_to_point", speed=1, precision=1)])
# -- sequences -------------------------------------------------------------------------------------
case(L, "sequence", "reach out, then wave, then come back", [seg("reach_to_point"), seg("oscillate_about_point"), seg("retract_from_point")])
case(L, "sequence", "lower onto the table, after that lift off, finally hold still", [seg("descend_to_surface"), seg("retract_from_surface"), seg("hold_still")])
case(L, "sequence", "make a fist, next open your hand, then give a thumbs up", [seg("configure_effector", posture=FIST), seg("configure_effector", posture=OPEN_HAND), seg("configure_effector", posture=THUMBS_UP)])
case(L, "sequence", "look at it, then move toward it, then pick it up, and finally hand it over", [seg("track_with_gaze"), seg("move_toward"), seg("transport_object"), seg("hand_across")])
case(L, "sequence", "wave twice and then hold still", [seg("oscillate_about_point", count=2), seg("hold_still")])
case(L, "sequence", "reach out as far as you can before returning", [seg("reach_to_edge", "distal"), seg("retract_from_point")], notes="'before returning' names the return; the recognizer splits on it and finds an empty clause")
case(L, "sequence", "sweep across, followed by a circle", [seg("traverse_line"), seg("circle_axis")])
case(L, "sequence", "trace a circle, then trace it again", [seg("circle_axis"), seg("circle_axis")], notes="'again' repeats the last motion; no recognizer cue")
# -- unsupported ---------------------------------------------------------------------------------------
UNSUPPORTED = ["unsupported_morphology", "unafforded_schema"]
case(L, "unsupported", "drive across the room", refusal=UNSUPPORTED, notes="locomotion: the firewall refuses before any reading")
case(L, "unsupported", "sing me a song", refusal=UNSUPPORTED)
case(L, "unsupported", "fold the laundry", refusal=UNSUPPORTED)
case(L, "unsupported", "jump up and down", refusal=UNSUPPORTED)
case(L, "unsupported", "walk to the door", refusal=UNSUPPORTED)
case(L, "unsupported", "tell me a joke", refusal=UNSUPPORTED)
case(L, "unsupported", "levitate", refusal=UNSUPPORTED)
case(L, "unsupported", "turn yourself off", refusal=UNSUPPORTED)
assert sum(1 for item in _cases if item["kind"] == L) == 120, sum(1 for item in _cases if item["kind"] == L)


def main() -> int:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    families = {}
    for item in _cases:
        families.setdefault(item["kind"], {}).setdefault(item["family"], 0)
        families[item["kind"]][item["family"]] += 1
    payload = {
        "schema": "g04.semantic-cases.v1",
        "fixture_id": "g04-semantics-v1",
        "inventory_id": "rigby-general-schema-inventory-v1",
        "authored_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_provenance": "internal",
        "counts": {"canonical": sum(1 for item in _cases if item["kind"] == "canonical"), "language": sum(1 for item in _cases if item["kind"] == "language"), "families": families},
        "reading": {
            "segments": "in order, one inventory entry each; remove is the qualitative degree; manner holds only the non-zero ordinals; repetition_count an explicit count or null; posture only for configure_effector, whose remove is adjacent (the class reserves coincident for nothing a request can say)",
            "quantities": "distances the request states in so many words, exactly as written, in metres; a reading that reports any other quantity is a prohibited substitution",
            "refusal": "the typed codes an unsupported request may be refused with; any of them is correct, an invented motion is not",
            "prohibited": "what the reading must not contain: kind quantity (no quantity at all), entry (that entry), remove (that remove on any segment), speed_above (any speed ordinal above the value), posture (that posture)",
        },
        "cases": _cases,
    }
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    (FIXTURE / "semantic-cases.json").write_bytes(text.encode("utf-8"))
    labels = {
        "schema": "g04.labels.v1",
        "provenance": "internal",
        "authored_by": "the G04 engineering pass inside this repository, 13 Sep 2026; the same pass that wrote the model planner",
        "independent_review": {"status": "pending", "external_input": "U3_independent_review", "claim_withheld": "the catalog's 'independently authored/reviewed' language gate"},
        "cases": {item["case_id"]: {"prompt": item["prompt"], "label_provenance": "internal", "family": item["family"]} for item in _cases},
        "sha256_of_cases": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    (FIXTURE / "labels.json").write_bytes((json.dumps(labels, indent=2) + "\n").encode("utf-8"))
    print(f"{len(_cases)} cases -> {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
