"""The hand-authored range-of-motion table, and its expansion onto 52 bones.

``config/rom.v1.json`` is generated from this module and is the file the loader
reads. This module is where the numbers are *decided*, so it is where the
argument for each of them lives.

Every value below is authored by hand and carries a structured ``source`` in
the schema lane `infra` shipped for ``thresholds.v1.json`` -- the ``kind``
determines which fields are mandatory, so a value cannot be recorded as
``measured`` without an n, and a ``provisional`` one cannot omit why.

Two properties this expansion preserves deliberately:

* ``typical_deg`` and ``max_deg`` are **anatomical** angles, because that is
  how every published column is quoted. ``rest_offset_deg`` carries the
  conversion into the rig's rest-relative frame. Plan 04 §6.4.
* Only plain Python floats are written, so no numpy scalar reaches
  ``json.dump`` -- lane `infra` flagged that as a cross-platform formatting
  hazard, and `rom.v1.json` is compared byte-for-byte by a test.
"""
import json
from collections import OrderedDict

from rigby_poc.analysis.anatomy.conventions import bone_side, joint_class
from rigby_poc.analysis.anatomy.neutral import NO_NEUTRAL
from rigby_poc.analysis.rig import canonical_bone_names

AAOS = "AAOS/AMA standard joint range-of-motion tables (adult, active range)"
RIG = "config/rig_profiles/mesh2motion-human-vrm1.json"

# (class, segment_suffix or None) -> {dof: entry}
# typical/max are ANATOMICAL degrees, converted to rest-relative by the loader.
# hard_assert marks a DOF a healthy joint does not possess.
T = OrderedDict()

def row(cls, seg, dof, typical, mx, kind, cites, hard_assert=False, **extra):
    T.setdefault((cls, seg), {})[dof] = {
        "typical_deg": list(typical), "max_deg": list(mx), "hard_assert": hard_assert,
        "source": {"kind": kind, "cites": list(cites), **extra}}

# ---- trunk -------------------------------------------------------------
# Thoracolumbar totals (flex 80, ext 25, lateral 35, rot 45) split across the
# three trunk segments. The split is not in any table, hence provisional.
for seg in ("spine", "chest", "upperChest"):
    row("spine", seg, "flexion", (-8, 27), (-12, 35), "provisional", [AAOS],
        rationale="thoracolumbar flexion 80 deg / extension 25 deg divided evenly across "
                  "three trunk segments; no per-segment table exists, so the split is an "
                  "assumption and the whole-trunk sum is the only validated quantity")
    row("spine", seg, "abduction", (-12, 12), (-16, 16), "provisional", [AAOS],
        rationale="thoracolumbar lateral flexion 35 deg divided across three segments")
    row("spine", seg, "twist", (-15, 15), (-20, 20), "provisional", [AAOS],
        rationale="thoracolumbar rotation 45 deg divided across three segments")
row("spine", "hips", "flexion", (-15, 15), (-30, 30), "invariant", ["src/rigby_poc/compiler.py"],
    rationale="hips is the skeleton root, not a joint. Its rotation is whole-body "
              "orientation, so an anatomical range does not apply; the bound is a wide "
              "sanity check only and must never be read as a pelvic-tilt limit")
row("spine", "hips", "abduction", (-15, 15), (-30, 30), "invariant", ["src/rigby_poc/compiler.py"],
    rationale="root orientation, not a joint; see flexion")
row("spine", "hips", "twist", (-180, 180), (-180, 180), "invariant", ["src/rigby_poc/compiler.py"],
    rationale="root yaw is unbounded by anatomy -- a cartwheel legitimately reaches 180 deg")
row("spine", "neck", "flexion", (-30, 25), (-40, 32), "provisional", [AAOS],
    rationale="cervical flexion 50 / extension 60 split between neck and head; the split "
              "is an assumption")
row("spine", "neck", "abduction", (-22, 22), (-30, 30), "provisional", [AAOS],
    rationale="cervical lateral flexion 45 deg split between neck and head")
row("spine", "neck", "twist", (-40, 40), (-50, 50), "provisional", [AAOS],
    rationale="cervical rotation 80 deg split between neck and head")
row("spine", "head", "flexion", (-30, 25), (-40, 32), "provisional", [AAOS],
    rationale="see neck")
row("spine", "head", "abduction", (-22, 22), (-30, 30), "provisional", [AAOS], rationale="see neck")
row("spine", "head", "twist", (-40, 40), (-50, 50), "provisional", [AAOS], rationale="see neck")

# ---- shoulder girdle ---------------------------------------------------
row("clavicle", None, "flexion", (-10, 40), (-15, 50), "provisional", [AAOS],
    rationale="scapular elevation ~40 deg / depression ~10 deg; the scapula is not a "
              "conventional goniometer joint and this rig moves it only via the shrug token")
row("clavicle", None, "abduction", (-25, 25), (-35, 35), "provisional", [AAOS],
    rationale="scapular protraction/retraction, approximate")
row("clavicle", None, "twist", (-15, 15), (-25, 25), "provisional", [AAOS],
    rationale="scapular rotation, approximate")
row("shoulder", None, "flexion", (-60, 180), (-70, 185), "external", [AAOS],
    rationale="glenohumeral flexion 180 / extension 60. NOTE: measured from the T-pose rest, "
              "so this is horizontal flexion; the rest offset is applied by the loader")
row("shoulder", None, "abduction", (-50, 180), (-60, 185), "external", [AAOS],
    rationale="abduction 180 / adduction 50; rest offset is -89.7 deg, the T-pose itself")
row("shoulder", None, "twist", (-90, 90), (-95, 95), "external", [AAOS],
    rationale="glenohumeral internal/external rotation 90/90")

# ---- elbow and forearm -------------------------------------------------
row("elbow", None, "flexion", (0, 145), (-10, 150), "external", [AAOS],
    rationale="elbow flexion 145 deg; hyperextension to -10 deg is within normal variation")
row("elbow", None, "abduction", (0, 0), (-5, 5), "invariant", [AAOS], hard_assert=True,
    rationale="the elbow is a hinge and possesses no abduction; any off-axis excursion is a "
              "defect, not a range")
row("elbow", None, "twist", (-80, 80), (-85, 85), "external",
    [AAOS, RIG + " anatomical_limits.forearm_twist_rad", "src/rigby_poc/primitives.py:143"],
    rationale="pronation/supination is the radioulnar DOF carried on the forearm bone, not a "
              "hinge violation. The rig declares forearm_twist_rad 1.35 rad = 77.4 deg and "
              "MAX_FOREARM_TWIST_RAD 1.30; plan 04 3.3's sketch hard_asserted this at +-5 deg, "
              "which would have rejected every pronation the rig deliberately produces")

# ---- wrist -------------------------------------------------------------
row("wrist", None, "flexion", (-70, 80), (-80, 90), "external", [AAOS],
    rationale="wrist palmar flexion 80 / dorsiflexion 70")
row("wrist", None, "abduction", (-30, 20), (-35, 25), "external", [AAOS],
    rationale="radial deviation 20 / ulnar deviation 30")
row("wrist", None, "twist", (-10, 10), (-15, 15), "external",
    [RIG + " anatomical_limits.wrist_twist_rad"],
    rationale="the carpus barely rotates; forearm rotation carries it. The rig declares "
              "wrist_twist_rad 0.18 rad = 10.3 deg and shipped motion reaches 67 deg, so this "
              "bound is expected to fire -- see plan 04 6.4")

# ---- hip, knee, ankle --------------------------------------------------
row("hip", None, "flexion", (-30, 120), (-40, 130), "external", [AAOS],
    rationale="hip flexion 120 with knee flexed / extension 30")
row("hip", None, "abduction", (-30, 45), (-35, 50), "external", [AAOS],
    rationale="abduction 45 / adduction 30")
row("hip", None, "twist", (-45, 45), (-50, 50), "external", [AAOS],
    rationale="internal/external rotation 45/45")
row("knee", None, "flexion", (0, 135), (-5, 145), "external", [AAOS],
    rationale="knee flexion 135; hyperextension to -5 deg is within normal variation")
row("knee", None, "abduction", (0, 0), (-5, 5), "invariant", [AAOS], hard_assert=True,
    rationale="the knee is a hinge and possesses no abduction; valgus/varus excursion is a defect")
row("knee", None, "twist", (-10, 30), (-15, 35), "external", [AAOS],
    rationale="tibial axial rotation, ~10 deg internal and more external in flexion. NOT a "
              "hinge violation: lane groundtruth measured 9.0 deg on a legitimate turn, which "
              "plan 04 3.3's sketch would have rejected")
row("ankle", None, "flexion", (-20, 50), (-25, 55), "external", [AAOS],
    rationale="plantarflexion 50 positive / dorsiflexion 20 negative")
row("ankle", None, "abduction", (-15, 35), (-20, 40), "external", [AAOS],
    rationale="eversion 15 / inversion 35")
row("ankle", None, "twist", (-10, 10), (-15, 15), "provisional", [AAOS],
    rationale="the talocrural joint has little axial rotation; bound is approximate")
row("toes", None, "flexion", (-70, 40), (-80, 45), "external", [AAOS],
    rationale="metatarsophalangeal flexion 40 / extension 70")
row("toes", None, "abduction", (-10, 10), (-15, 15), "provisional", [AAOS], rationale="approximate")
row("toes", None, "twist", (-5, 5), (-10, 10), "provisional", [AAOS], rationale="approximate")

# ---- digits ------------------------------------------------------------
# The lower bound is NOT zero. Finger joints hyperextend, the MCP markedly so,
# and an anatomical zero of exactly 0 made the rig's own rest finger -- which
# sits 0.7 deg from straight -- read as beyond its typical band.
FINGER = {"Proximal": (-20, 90, -30, 100,
                       ("metacarpophalangeal flexion 90 deg; MCP hyperextension to "
                        "20-30 deg is normal and common")),
          "Intermediate": (-5, 110, -10, 120,
                           "proximal interphalangeal flexion 110 deg, slight hyperextension"),
          "Distal": (-10, 80, -15, 90,
                     "distal interphalangeal flexion 80 deg, hyperextension to 10 deg")}
for seg, (lo, hi, mlo, mhi, why) in FINGER.items():
    row("digit", seg, "flexion", (lo, hi), (mlo, mhi), "external", [AAOS], rationale=why)
    ab = (-20, 20) if seg == "Proximal" else (-5, 5)
    row("digit", seg, "abduction", ab, (ab[0] - 5, ab[1] + 5),
        "external" if seg == "Proximal" else "invariant", [AAOS],
        hard_assert=(seg != "Proximal"),
        rationale=("radial/ulnar deviation at the MCP, ~20 deg" if seg == "Proximal"
                   else "the interphalangeal joints are hinges and possess no deviation"))
    row("digit", seg, "twist", (-5, 5), (-10, 10), "provisional", [AAOS],
        rationale="finger axial rotation is passive and small; bound is approximate")

# ---- thumb: no anatomical neutral exists -------------------------------
for seg in ("Metacarpal", "Proximal", "Distal"):
    for dof in ("flexion", "abduction", "twist"):
        row("thumb", seg, dof, (-60, 60), (-75, 75), "provisional",
            [AAOS, RIG + " grip_presets.crate_grip"],
            rationale="PROVISIONAL, and not merely for want of data. The thumb has no "
                      "non-arbitrary anatomical-position direction, so no rest offset can be "
                      "derived and no published thumb column transfers. Worse, the rig's own "
                      "crate_grip preset drives all three thumb segments about an axis that "
                      "decomposes to pure abduction with a zero flexion component -- so a "
                      "thumb flexion limit would constrain a DOF the rig's authored grip "
                      "never uses. It is not established that flexion is the right DOF to "
                      "limit here at all. Bounds are wide rest-relative sanity checks.")

IMMOBILE = ("leftToes", "rightToes", "neck", "spine", "upperChest")

def key_for(bone):
    cls = joint_class(bone)
    side = bone_side(bone)
    stem = bone[len(side):] if side else bone
    if (cls, stem) in T: return (cls, stem)
    if (cls, bone) in T: return (cls, bone)
    for seg in ("Metacarpal", "Proximal", "Intermediate", "Distal"):
        if stem.endswith(seg) and (cls, seg) in T: return (cls, seg)
    return (cls, None)

limits = OrderedDict()
for bone in canonical_bone_names():
    entry = T[key_for(bone)]
    out = OrderedDict()
    side = bone_side(bone)
    for dof in ("flexion", "abduction", "twist"):
        e = dict(entry[dof])
        if dof == "twist" and side == "right":
            # Plan 04 §3.4, measured in 04a: flexion and abduction are preserved
            # under mirroring and only twist negates. So a twist range authored
            # for the left side is negated AND swapped for the right, or the two
            # sides permit different physical motions. Lane `groundtruth`
            # measured the corresponding fact in the motion: a turn twists the
            # two knees by 9.0 deg in opposite signs.
            e["typical_deg"] = [-e["typical_deg"][1], -e["typical_deg"][0]]
            e["max_deg"] = [-e["max_deg"][1], -e["max_deg"][0]]
        if e["hard_assert"]:
            # A DOF the joint does not possess has no "typical range" distinct
            # from its tolerance, so collapsing the two bands is the honest
            # shape. Leaving typical at (0, 0) made every rest pose "beyond
            # typical" on sub-degree rig noise.
            e["typical_deg"] = list(e["max_deg"])
        src = dict(e.pop("source"))
        src["reference_frame"] = (
            "none -- no anatomical neutral exists for this entity"
            # A class property, not a derived float: the thumb has no neutral by
            # anatomy, which is stable across architectures.
            if joint_class(bone) in NO_NEUTRAL else
            "anatomical neutral; rest offset from rigby_poc.analysis.anatomy.neutral.rest_offset")
        out[dof] = OrderedDict(
            typical_deg=[float(x) for x in e["typical_deg"]],
            max_deg=[float(x) for x in e["max_deg"]],
            # Required, not defaulted. False is a legitimate value here -- 145
            # of the 156 entries carry it -- so a default would be
            # indistinguishable from an authored one, and a row added without
            # going through row() would silently become non-hard-asserted.
            hard_assert=bool(e["hard_assert"]),
            # rest_offset_deg is deliberately NOT written. It is *derived* from
            # rig geometry, and several of the 52 sit within 1e-5 degrees of a
            # three-decimal rounding boundary -- leftIndexProximal.abduction is
            # 3.7e-06 away. Cross-architecture float drift is enough to flip
            # those, which turned the byte-equality test red on Windows CI while
            # passing on macOS. A derived value has no business in a
            # hand-authored config; the loader derives it at read time.
            source=src)
    limits[bone] = OrderedDict(joint_class=joint_class(bone),
                               enforceable=("mutation_only" if bone in IMMOBILE else "generation"),
                               dofs=out)

def build_document() -> OrderedDict:
    return OrderedDict(
        schema_version="1.0",
        generated_by="rigby_poc.analysis.anatomy.authored_rom -- every value there is hand-authored with a source; only the expansion onto 52 bones is mechanical. tests/test_rom_table.py fails if this file and that module disagree.",
        source_kinds={
            "measured": "Derived from data. Requires n, derived_from, date, reference_frame.",
            "invariant": "An implementation or anatomical contract, not a measurement. Requires rationale.",
            "external": "Taken from a published table, rig asset or spec. Requires derived_from or cites.",
            "provisional": "Explicitly unvalidated. Requires rationale. The value may be wrong.",
        },
        notes=OrderedDict(
            units="degrees",
            frame=("typical_deg and max_deg are ANATOMICAL angles. The rest-relative bound is "
                   "anatomical + the bone's rest offset, DERIVED at load time by "
                   "rigby_poc.analysis.anatomy.neutral.rest_offset. It is deliberately not stored "
                   "here: it is geometry rather than an authored value, and several offsets sit "
                   "within 1e-5 deg of a rounding boundary, which made this file "
                   "architecture-dependent. See plan 04 6.4."),
            mirroring="flexion and abduction are preserved across sides; only twist negates. Authored once, applied to both.",
            enforceable="'mutation_only' marks a bone compiler.py never assigns a rotation to on any path, so its limit can never fire from generated motion. Exercising it belongs to plan 06.",
            report_only="04b ships every check with status='pass' regardless of the measurement. Enforcement is 04c, per DOF, after the distribution review.",
        ),
        limits=limits)



ROM_FILE = "config/rom.v1.json"


def rom_json() -> str:
    """The exact bytes ``config/rom.v1.json`` should contain."""

    return json.dumps(build_document(), indent=2) + "\n"


__all__ = ["ROM_FILE", "build_document", "rom_json"]
