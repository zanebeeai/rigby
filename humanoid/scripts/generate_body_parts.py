"""Generate config/body_parts.v1.json from the committed ROM document.

The design decision worth stating: a move is authored as a signed fraction of a
DOF's own ROM range, not as an angle. ``{"flexion": 0.6}`` means "60% of the way
to this joint's typical flexion limit", so a move cannot express an out-of-range
pose -- the envelope and the vocabulary cannot disagree, because the vocabulary
is written in the envelope's units.

That is the whole point of deriving this file rather than authoring it: every
bone, DOF and range here comes from rom.v1.json, and tests/test_body_part_moves.py
fails if this file and that document drift apart.
"""

import collections
import json
import pathlib

ROM = json.loads(pathlib.Path("config/rom.v1.json").read_text(encoding="utf-8"))
LIMITS = ROM["limits"]
SIDES = ("left", "right")
DIGITS = ("Thumb", "Index", "Middle", "Ring", "Little")


def part_bones():
    """One part per articulable unit; a finger is one part, not three."""
    parts = collections.OrderedDict()
    parts["root"] = ["hips"]
    parts["spine"] = ["spine", "chest", "upperChest"]
    parts["neck"] = ["neck"]
    parts["head"] = ["head"]
    for side in SIDES:
        s = side[0].upper() + side[1:]
        parts[f"{side}_clavicle"] = [f"{side}Shoulder"]
        parts[f"{side}_shoulder"] = [f"{side}UpperArm"]
        parts[f"{side}_elbow"] = [f"{side}LowerArm"]
        parts[f"{side}_wrist"] = [f"{side}Hand"]
        for digit in DIGITS:
            stem = f"{side}{digit}"
            segs = (
                ["Metacarpal", "Proximal", "Distal"]
                if digit == "Thumb"
                else ["Proximal", "Intermediate", "Distal"]
            )
            parts[f"{side}_{digit.lower()}"] = [f"{stem}{seg}" for seg in segs]
        parts[f"{side}_hip"] = [f"{side}UpperLeg"]
        parts[f"{side}_knee"] = [f"{side}LowerLeg"]
        parts[f"{side}_ankle"] = [f"{side}Foot"]
        parts[f"{side}_toes"] = [f"{side}Toes"]
    return parts


PARENTS = {
    "root": None, "spine": "root", "neck": "spine", "head": "neck",
}
for side in SIDES:
    PARENTS[f"{side}_clavicle"] = "spine"
    PARENTS[f"{side}_shoulder"] = f"{side}_clavicle"
    PARENTS[f"{side}_elbow"] = f"{side}_shoulder"
    PARENTS[f"{side}_wrist"] = f"{side}_elbow"
    for digit in DIGITS:
        PARENTS[f"{side}_{digit.lower()}"] = f"{side}_wrist"
    PARENTS[f"{side}_hip"] = "root"
    PARENTS[f"{side}_knee"] = f"{side}_hip"
    PARENTS[f"{side}_ankle"] = f"{side}_knee"
    PARENTS[f"{side}_toes"] = f"{side}_ankle"


def d(**kw):
    return collections.OrderedDict(kw)


# ---------------------------------------------------------------- move sets
# Values are signed fractions of the bone's own ROM range: +1.0 is the typical
# positive limit, -1.0 the typical negative limit. All three DOFs of every part
# are reachable, so the vocabulary spans the rig's articulation rather than a
# convenient corner of it.

def uniform(bones, spec):
    return collections.OrderedDict((b, collections.OrderedDict(spec)) for b in bones)


def taper(bones, dof, values):
    """Different fraction per segment -- a curling finger is not a rigid rod."""
    return collections.OrderedDict(
        (b, collections.OrderedDict([(dof, v)])) for b, v in zip(bones, values)
    )


def moves_for(part, bones, joint_class):
    m = collections.OrderedDict()
    m["hold"] = collections.OrderedDict()
    if joint_class in {"digit", "thumb"}:
        m["extend"] = taper(bones, "flexion", [0.0, 0.0, 0.0])
        m["flex_light"] = taper(bones, "flexion", [0.35, 0.30, 0.25])
        m["flex"] = taper(bones, "flexion", [0.70, 0.65, 0.55])
        m["flex_full"] = taper(bones, "flexion", [1.0, 1.0, 0.95])
        m["hook"] = taper(bones, "flexion", [0.30, 1.0, 1.0])
        m["hyperextend"] = taper(bones, "flexion", [-1.0, 0.0, 0.0])
        m["abduct"] = collections.OrderedDict([(bones[0], d(abduction=0.85))])
        m["adduct"] = collections.OrderedDict([(bones[0], d(abduction=-0.85))])
        m["rotate_in"] = collections.OrderedDict([(bones[0], d(twist=0.8))])
        m["rotate_out"] = collections.OrderedDict([(bones[0], d(twist=-0.8))])
        if joint_class == "thumb":
            m["oppose"] = collections.OrderedDict([
                (bones[0], d(flexion=0.45, abduction=-0.9, twist=0.85)),
                (bones[1], d(flexion=0.35)),
            ])
            m["reposition"] = collections.OrderedDict([
                (bones[0], d(flexion=0.0, abduction=0.9, twist=-0.6))
            ])
        return m
    if joint_class == "shoulder":
        m.update({
            "flex_forward": uniform(bones, d(flexion=0.55)),
            "raise_overhead": uniform(bones, d(flexion=1.0)),
            "extend_back": uniform(bones, d(flexion=-1.0)),
            "abduct": uniform(bones, d(abduction=0.6)),
            "abduct_full": uniform(bones, d(abduction=1.0)),
            "adduct": uniform(bones, d(abduction=-1.0)),
            "rotate_in": uniform(bones, d(twist=1.0)),
            "rotate_out": uniform(bones, d(twist=-1.0)),
            "reach_forward_across": uniform(bones, d(flexion=0.6, abduction=-0.7)),
            "reach_out_and_up": uniform(bones, d(flexion=0.5, abduction=0.8)),
        })
        return m
    if joint_class == "elbow":
        m.update({
            "extend": uniform(bones, d(flexion=0.0)),
            "flex": uniform(bones, d(flexion=0.55)),
            "flex_full": uniform(bones, d(flexion=1.0)),
            "pronate": uniform(bones, d(twist=1.0)),
            "supinate": uniform(bones, d(twist=-1.0)),
            "neutral_rotation": uniform(bones, d(twist=0.0)),
            "carry": uniform(bones, d(flexion=0.62, twist=0.25)),
            # The carrying angle. ROM allows only +-5 deg here, which is the
            # point: the vocabulary must still be able to name it, or the DOF
            # exists in the envelope and nowhere else.
            "valgus": uniform(bones, d(abduction=1.0)),
            "varus": uniform(bones, d(abduction=-1.0)),
        })
        return m
    if joint_class == "wrist":
        m.update({
            "neutral": uniform(bones, d(flexion=0.0, abduction=0.0, twist=0.0)),
            "flex": uniform(bones, d(flexion=0.8)),
            "extend": uniform(bones, d(flexion=-0.8)),
            "deviate_radial": uniform(bones, d(abduction=0.9)),
            "deviate_ulnar": uniform(bones, d(abduction=-0.9)),
            "rotate_in": uniform(bones, d(twist=1.0)),
            "rotate_out": uniform(bones, d(twist=-1.0)),
            "cock_back": uniform(bones, d(flexion=-0.55, abduction=0.4)),
        })
        return m
    if joint_class == "clavicle":
        m.update({
            "neutral": uniform(bones, d(flexion=0.0, abduction=0.0)),
            "shrug": uniform(bones, d(abduction=0.9)),
            "depress": uniform(bones, d(abduction=-0.9)),
            "protract": uniform(bones, d(flexion=0.9)),
            "retract": uniform(bones, d(flexion=-0.9)),
            "rotate_up": uniform(bones, d(twist=0.9)),
            "rotate_down": uniform(bones, d(twist=-0.9)),
        })
        return m
    if joint_class == "spine":
        m.update({
            "neutral": uniform(bones, d(flexion=0.0, abduction=0.0, twist=0.0)),
            "lean_forward": uniform(bones, d(flexion=0.7)),
            "lean_back": uniform(bones, d(flexion=-0.8)),
            "lean_left": uniform(bones, d(abduction=0.8)),
            "lean_right": uniform(bones, d(abduction=-0.8)),
            "rotate_left": uniform(bones, d(twist=0.8)),
            "rotate_right": uniform(bones, d(twist=-0.8)),
            "brace": uniform(bones, d(flexion=0.15)),
        })
        if part in {"head", "neck"}:
            m.update({
                "look_down": uniform(bones, d(flexion=0.8)),
                "look_up": uniform(bones, d(flexion=-0.8)),
                "tilt_left": uniform(bones, d(abduction=0.8)),
                "tilt_right": uniform(bones, d(abduction=-0.8)),
                "look_left": uniform(bones, d(twist=0.8)),
                "look_right": uniform(bones, d(twist=-0.8)),
            })
        return m
    if joint_class == "hip":
        m.update({
            "neutral": uniform(bones, d(flexion=0.0, abduction=0.0, twist=0.0)),
            "flex": uniform(bones, d(flexion=0.5)),
            "flex_high": uniform(bones, d(flexion=1.0)),
            "extend": uniform(bones, d(flexion=-1.0)),
            "abduct": uniform(bones, d(abduction=1.0)),
            "adduct": uniform(bones, d(abduction=-1.0)),
            "rotate_in": uniform(bones, d(twist=1.0)),
            "rotate_out": uniform(bones, d(twist=-1.0)),
        })
        return m
    if joint_class == "knee":
        m.update({
            "extend": uniform(bones, d(flexion=0.0)),
            "flex": uniform(bones, d(flexion=0.5)),
            "flex_full": uniform(bones, d(flexion=1.0)),
            "rotate_in": uniform(bones, d(twist=1.0)),
            "rotate_out": uniform(bones, d(twist=-1.0)),
            "valgus": uniform(bones, d(abduction=1.0)),
            "varus": uniform(bones, d(abduction=-1.0)),
        })
        return m
    if joint_class == "ankle":
        m.update({
            "neutral": uniform(bones, d(flexion=0.0, abduction=0.0)),
            "dorsiflex": uniform(bones, d(flexion=-1.0)),
            "plantarflex": uniform(bones, d(flexion=1.0)),
            "invert": uniform(bones, d(abduction=-1.0)),
            "evert": uniform(bones, d(abduction=1.0)),
            "rotate_in": uniform(bones, d(twist=1.0)),
            "rotate_out": uniform(bones, d(twist=-1.0)),
        })
        return m
    if joint_class == "toes":
        m.update({
            "neutral": uniform(bones, d(flexion=0.0)),
            "curl": uniform(bones, d(flexion=1.0)),
            "lift": uniform(bones, d(flexion=-1.0)),
            "push_off": uniform(bones, d(flexion=0.7)),
            "splay": uniform(bones, d(abduction=1.0)),
            "rotate_in": uniform(bones, d(twist=1.0)),
            "rotate_out": uniform(bones, d(twist=-1.0)),
        })
        return m
    raise SystemExit(f"no move set authored for joint class {joint_class!r}")


# ------------------------------------------------------------- sequences
# A grouped action is a TIMED sequence over parts, not a single pose. `grip`
# is the reason the structure exists: the thumb must arrive before the fingers
# close, or the fingers push a free object out of the hand before it is caught.

def grip_sequence(side):
    f = [f"{side}_{n}" for n in ("index", "middle", "ring", "little")]
    t = f"{side}_thumb"
    return [
        collections.OrderedDict([("at", 0.00), ("label", "open"),
            ("moves", collections.OrderedDict(
                [(t, "reposition")] + [(x, "extend") for x in f]))]),
        collections.OrderedDict([("at", 0.25), ("label", "preshape"),
            ("moves", collections.OrderedDict(
                [(t, "oppose")] + [(x, "flex_light") for x in f]))]),
        collections.OrderedDict([("at", 0.45), ("label", "thumb_anchor"),
            ("moves", collections.OrderedDict([(t, "oppose")]))]),
        collections.OrderedDict([("at", 0.70), ("label", "close_fingers"),
            ("moves", collections.OrderedDict(
                [(t, "oppose")] + [(x, "flex") for x in f]))]),
        collections.OrderedDict([("at", 1.00), ("label", "secure"),
            ("moves", collections.OrderedDict(
                [(t, "oppose")] + [(x, "flex") for x in f[:2]]
                + [(x, "flex_full") for x in f[2:]]))]),
    ]


def build():
    parts = part_bones()
    out_parts = collections.OrderedDict()
    out_moves = collections.OrderedDict()
    for part, bones in parts.items():
        missing = [b for b in bones if b not in LIMITS]
        if missing:
            raise SystemExit(f"{part}: bones absent from rom.v1.json: {missing}")
        joint_class = LIMITS[bones[0]]["joint_class"]
        envelope = collections.OrderedDict()
        for bone in bones:
            envelope[bone] = collections.OrderedDict(
                (dof, collections.OrderedDict([
                    ("typical_deg", spec["typical_deg"]),
                    ("max_deg", spec["max_deg"]),
                    ("enforced", spec["enforced"]),
                ]))
                for dof, spec in LIMITS[bone]["dofs"].items()
            )
        out_parts[part] = collections.OrderedDict([
            ("parent", PARENTS[part]),
            ("joint_class", joint_class),
            ("bones", bones),
            ("dof_envelope", envelope),
        ])
        out_moves[part] = moves_for(part, bones, joint_class)

    sequences = collections.OrderedDict()
    for side in SIDES:
        sequences[f"{side}_grip_power"] = collections.OrderedDict([
            ("description",
             "Acquire a free object. The thumb reaches opposition before the "
             "fingers close, because index and middle reaching the object first "
             "push it out of the hand."),
            ("steps", grip_sequence(side)),
        ])
        sequences[f"{side}_release"] = collections.OrderedDict([
            ("description", "Open every digit together; nothing to sequence."),
            ("steps", [collections.OrderedDict([
                ("at", 1.0), ("label", "open"),
                ("moves", collections.OrderedDict(
                    [(f"{side}_thumb", "reposition")]
                    + [(f"{side}_{n}", "extend")
                       for n in ("index", "middle", "ring", "little")]))])]),
        ])
        sequences[f"{side}_point"] = collections.OrderedDict([
            ("description", "Index extended, the rest closed."),
            ("steps", [collections.OrderedDict([
                ("at", 1.0), ("label", "point"),
                ("moves", collections.OrderedDict([
                    (f"{side}_thumb", "adduct"), (f"{side}_index", "extend"),
                    (f"{side}_middle", "flex_full"), (f"{side}_ring", "flex_full"),
                    (f"{side}_little", "flex_full")]))])]),
        ])
        sequences[f"{side}_wave"] = collections.OrderedDict([
            ("description", "Arm up, forearm supinated, wrist alternating."),
            ("steps", [
                collections.OrderedDict([("at", 0.0), ("label", "raise"),
                    ("moves", collections.OrderedDict([
                        (f"{side}_shoulder", "reach_out_and_up"),
                        (f"{side}_elbow", "flex"),
                        (f"{side}_wrist", "neutral")]))]),
                collections.OrderedDict([("at", 0.5), ("label", "swing_out"),
                    ("moves", collections.OrderedDict([
                        (f"{side}_wrist", "deviate_radial")]))]),
                collections.OrderedDict([("at", 1.0), ("label", "swing_back"),
                    ("moves", collections.OrderedDict([
                        (f"{side}_wrist", "deviate_ulnar")]))]),
            ]),
        ])

    return collections.OrderedDict([
        ("schema_version", "1.0"),
        ("$comment",
         "Per-part movement vocabulary, expressed in the units of config/rom.v1.json. "
         "A move value is a SIGNED FRACTION of that DOF's typical range: +1.0 is the "
         "typical positive limit, -1.0 the typical negative one. A move therefore "
         "cannot name an out-of-range pose, and the constraint layer and the "
         "vocabulary cannot disagree. Generated from rom.v1.json; "
         "tests/test_body_part_moves.py fails if the two drift apart."),
        ("generated_from", "config/rom.v1.json"),
        ("value_convention", collections.OrderedDict([
            ("units", "fraction of the DOF's typical_deg range"),
            ("positive", "value * typical_deg[1]"),
            ("negative", "abs(value) * typical_deg[0]"),
            ("range", [-1.0, 1.0]),
        ])),
        ("parts", out_parts),
        ("move_sets", out_moves),
        ("sequences", sequences),
    ])


doc = build()
path = pathlib.Path("config/body_parts.v1.json")
path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
n_moves = sum(len(v) for v in doc["move_sets"].values())
print(f"parts     : {len(doc['parts'])}")
print(f"moves     : {n_moves}")
print(f"sequences : {len(doc['sequences'])}")
print("bones covered:", len({b for p in doc["parts"].values() for b in p["bones"]}), "of", len(LIMITS))
