"""Prompt fragments for the split judge graders.

Plan 07 §1.2 diagnoses `judge.UNARY_SYSTEM_PROMPT` as one accreted 115-line block
that cannot be calibrated per dimension.  This module is the decomposition: five
single-purpose grader prompts (§3.1) plus per-family semantic fragments composed
only into the `semantic` grader, each carrying an explicit version and content
hash (§3.5).

The text here is a redistribution of the existing mega-prompt, not a rewrite: a
sentence that used to sit in `UNARY_SYSTEM_PROMPT` sits in exactly one grader or
one family fragment now.  No grader prompt mentions objective motion diagnostics,
which is goal G4.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Mapping


GraderName = Literal["semantic", "anatomy", "artifact", "timing", "crossview"]

#: Fixed order.  Calibration reports are keyed on it, so it is part of the contract.
GRADER_NAMES: tuple[GraderName, ...] = (
    "semantic",
    "anatomy",
    "artifact",
    "timing",
    "crossview",
)

#: Motion families are `rigby_poc.models.Intent` values minus `unsupported`, which
#: never reaches the judge because it produces no clip.
FAMILY_NAMES: tuple[str, ...] = (
    "gesture",
    "grab",
    "strike",
    "composite",
    "full_body",
    "object_interaction",
    "sequence",
)

#: Family used when a manifest carries no intent, or one this module does not know.
#: It is deliberately generic rather than a guess: mislabelling a strike as a
#: gesture would feed the semantic grader the wrong rubric silently.
FALLBACK_FAMILY = "unknown"


@dataclass(frozen=True)
class GraderSpec:
    """What one grader is allowed to see and which dimensions it owns."""

    name: GraderName
    #: Dimensions of `MotionJudgeScore` this grader alone is responsible for.
    dimensions: tuple[str, ...]
    #: Whether the user's motion request is included in the payload.
    sees_prompt: bool
    #: Which evidence the payload carries.
    evidence: Literal["key_poses", "timelines", "both"]
    #: Composes the per-family semantic fragment into its prompt.
    family_specific: bool = False

    @property
    def sees_diagnostics(self) -> bool:
        """No grader ever sees the deterministic layer's measurements (§3.3)."""
        return False


GRADER_SPECS: Mapping[GraderName, GraderSpec] = {
    "semantic": GraderSpec(
        name="semantic",
        dimensions=("semantic_match", "gesture_recognizability"),
        sees_prompt=True,
        evidence="both",
        family_specific=True,
    ),
    "anatomy": GraderSpec(
        name="anatomy",
        dimensions=("anatomical_naturalness",),
        sees_prompt=False,
        evidence="both",
    ),
    "artifact": GraderSpec(
        name="artifact",
        dimensions=("egocentric_visibility",),
        sees_prompt=False,
        evidence="both",
    ),
    "timing": GraderSpec(
        name="timing",
        dimensions=("temporal_readability",),
        sees_prompt=False,
        evidence="timelines",
    ),
    "crossview": GraderSpec(
        name="crossview",
        dimensions=("cross_view_consistency",),
        sees_prompt=True,
        evidence="key_poses",
    ),
}


# --------------------------------------------------------------------------
# Grader cores
# --------------------------------------------------------------------------

_SHARED_EVIDENCE_RULE = """Judge only the visible rendered evidence. Every snapshot and every timeline tile
preserves the complete uncropped 16:9 field of view and is labeled with its snapshot id and time. Cite exact
snapshot ids for every conclusion. Never infer quality from candidate names, parameters, seeds, filenames, or
prior scores; they are intentionally hidden. Use failure tag 'none' only when no other failure tag applies."""

SEMANTIC_CORE = f"""You are a strict semantic-fidelity judge for rendered humanoid motion. You are given the
user's motion request and the rendered evidence. Decide only whether the rendered motion is the motion that was
requested and whether the action reads immediately as itself. Do not score anatomy, rendering artifacts, timing
smoothness, or agreement between views; other graders own those.
{_SHARED_EVIDENCE_RULE}
Score semantic_match on whether every requested clause, modifier, hand, direction, count, and timing category is
present in the rendered motion. Score gesture_recognizability on whether an independent viewer who was never
told the request would name the action correctly from the evidence alone. Score 1 as clearly wrong or
unrecognizable, 3 as recognizable but incomplete or ambiguous, and 5 as exactly the requested action and
immediately readable. Judge the whole chronological sequence, including setup and recovery, not one attractive
endpoint pose."""

ANATOMY_PROMPT = f"""You are a strict anatomical-plausibility judge for rendered humanoid motion. You are NOT
told what motion was requested, and you must not guess it or reward a pose for looking deliberate. Judge only
whether the body configuration is one a human body could adopt.
{_SHARED_EVIDENCE_RULE}
Penalize unnatural wrist bending, wrist or forearm twist beyond human range, forearm and upper-arm contortion,
elbow or knee hyperextension, shoulder or hip configurations outside a human envelope, an unnaturally broken
finger configuration, and a limb intersecting the performer's own body. Judge whole-body anatomy from the orbit
evidence; the egocentric camera is mounted at the performer's head, so the torso, legs, and complete performer
being absent there is correct framing and not an anatomical fault. Score anatomical_naturalness 1 when a joint
is clearly outside human range, 3 when a configuration is strained or awkward but possible, and 5 when every
joint reads as a relaxed, physically plausible human pose."""

ARTIFACT_PROMPT = f"""You are a strict rendering-artifact judge for rendered humanoid motion. You are NOT told
what motion was requested. Judge only defects of the rendered image: geometry that clips, a body or object that
is cropped out of frame, and silhouettes that cannot be read.
{_SHARED_EVIDENCE_RULE}
Penalize a hand or the active limb leaving the frame, geometry passing through the ground plane or through a
scene object, limbs passing through each other, a support foot floating above or sunk beneath the ground, an
object penetrating a surface it should rest on, and a silhouette so occluded or self-overlapping that the pose
cannot be read. Score egocentric_visibility 1 when the acting part of the body is cropped, buried, or clipped
in a way that makes the motion unjudgeable, 3 when it is partly occluded or intermittently clipped, and 5 when
every acting part stays fully in frame and free of visible geometry defects. The absence of the performer's own
torso and legs from the head-mounted egocentric view is correct first-person framing, not a cropping defect."""

TIMING_PROMPT = f"""You are a strict temporal-readability judge for rendered humanoid motion. You are given the
chronological timeline sheets only: no single-pose views, and no statement of what was requested. Judge only how
the motion progresses from tile to tile.
{_SHARED_EVIDENCE_RULE}
Explicitly compare consecutive tiles. Penalize a pose that stays nearly frozen and then jumps, motion that
advances in visible discrete steps, a phase that reverses during presentation, a hold that never settles, and an
abrupt snap between phases. Do not infer smooth timing from the final pose. Score temporal_readability 1 when
the sequence is a series of jumps or reversals, 3 when it is followable but uneven, and 5 when every transition
is continuous, well-staged, and holds where a hold is staged."""

CROSSVIEW_PROMPT = f"""You are a strict cross-view consistency judge for rendered humanoid motion. Each phase
appears in a paired egocentric and orbit view at the same instant. Judge only whether the two views depict the
same body doing the same thing at the same moment.
{_SHARED_EVIDENCE_RULE}
Penalize a limb, hand shape, object position, or contact state that disagrees between the paired views at the
same timestamp, an object that is held in one view and free in the other, and head or camera motion in the
egocentric view that does not correspond to the head motion visible in orbit. The egocentric camera is the
performer's literal head-mounted first-person view: different framing, and the absence of the performer's torso,
legs, and complete body from it, is correct and must never be scored as a disagreement or tagged
camera_mismatch. Score cross_view_consistency 1 when the two views show materially different motion, 3 when a
detail disagrees, and 5 when both views are consistent readings of one performance."""


GRADER_CORES: Mapping[GraderName, str] = {
    "semantic": SEMANTIC_CORE,
    "anatomy": ANATOMY_PROMPT,
    "artifact": ARTIFACT_PROMPT,
    "timing": TIMING_PROMPT,
    "crossview": CROSSVIEW_PROMPT,
}


# --------------------------------------------------------------------------
# Per-family semantic fragments (§3.1) — composed only into `semantic`
# --------------------------------------------------------------------------

_GESTURE_FRAGMENT = """FAMILY: gesture. A hang-ten/shaka requires thumb and little finger extended with the
middle three fingers curled. A thumbs-up requires an extended thumb with the other four fingers curled. A
peace/victory sign requires extended, separated index and middle fingers with the ring and little fingers
curled. When the request asks for a shake or back-and-forth motion, require visible repeated direction
reversals during that phase, roughly the requested number of beats, and a clean return to the relaxed default
after it. For a hang-ten/shaka shake, the oscillation must be forearm pronation/supination about the forearm's
long axis; wrist flexion/extension or side-to-side wrist deviation used as the main oscillation is the wrong
motion. For ordered intra-hand dexterity, require the named driver digit to make visibly distinct fingertip
contacts in the requested order, with readable separation between contacts rather than one held pinch; reject a
wrong order, simultaneous collapse, skipped fingertip, or static pinch. If the request says to look at the
hand, require the head-mounted view to track that hand naturally while orbit evidence confirms the head turn;
reject a head-only substitute."""

_GRAB_FRAGMENT = """FAMILY: grab. Require a readable approach, preshape, contact, close, lift, and a stable
hold. The grasp must close on the object rather than around empty space, the object must leave its support, and
it must stay in the hand without visible slip or re-grip through the hold. Reject an open-hand carry, an object
that follows the hand without being grasped, or a lift that never separates the object from its support."""

_STRIKE_FRAGMENT = """FAMILY: strike. Require a closed fist, a readable guard and load, a decisive impact path,
controlled follow-through, and recovery. A hook must travel on a lateral curved arc with a visibly bent elbow
and torso participation; a hook that reads as a straight jab, a static crossed-arm pose, or slow arm placement
is the wrong action. A jab and a cross travel forward; an uppercut travels upward. The named hand must be the
striking hand and the other hand must stay in guard."""

_COMPOSITE_FRAGMENT = """FAMILY: composite. For coordinated or multi-part movement, verify every requested limb
moves, the relative phase and direction between limbs is correct, the path type and repetition count are
visible, and limbs do not pass through each other. For an explicit basketball travel signal, require two closed-
fist forearms held across the chest, one over the other, with each fist staying near the opposite elbow; the
parallel forearms must roll together around one shared cross-body axis, repeatedly exchanging over/under and
front/back order for about three revolutions. Reject two independent hand circles, fixed elbows with small wrist
flourishes, diverging forearm axes, or a static crossed-arm substitute."""

_OBJECT_INTERACTION_FRAGMENT = """FAMILY: object interaction. For physical throws, require secure contact, a
style-appropriate windup, hand opening at release, continuous ballistic free flight, follow-through, and
recovery; reject an object that stays glued to an open palm, teleports, or flies before release. For catches,
require the open hand to meet the incoming object, closure only at contact, a short absorbing motion, and
stable retention. For pushes and pulls, require contact before object motion, continuous hand/object coincidence
through the guided phase, travel in the requested direction, no teleport or premature release, and a clean
separation before recovery. For a roll, require that continuous guided contact and support-plane travel plus
visible rotation about the axis perpendicular to travel; reject sliding without rotation or rotation without
translation. For a support-plane spin, require continuous contact while the object center stays fixed and
completes the requested yaw turns. For controlled placement, require secure pickup, visible lift, continuous
attached transport, lowering onto the support surface before the hand opens, clean separation, and recovery
without a reset or ballistic drop. For drops, require a secured lift, visible hand opening before separation,
gravity-driven vertical fall, supported landing without tunneling, and recovery only after the object is free.
For hand-to-hand transfers, require the named source hand to secure and present the object, the receiver to make
visible contact before the source opens, a continuous ownership transfer with no teleport, and stable retention
by the named receiver. For carries, require the initial grasp to stay closed and the object to follow the
carrying hand continuously through every requested step; reject a reset to the original position, a floating
object, or an open-hand carry. When a carry is followed by a drop or throw, require the release to begin from
the transported world position with no second pickup. For an overhead lift, require the grasp to stay closed and
the object center to rise clearly above the head rather than stopping at chest, shoulder, or face height."""

_FULL_BODY_FRAGMENT = """FAMILY: full body. Use the orbit view to judge root travel, turns, leg and foot
action, support changes, ground clearance, landing, and recovery; use the egocentric view to confirm coherent
camera motion. Penalize foot sliding, missing requested steps or repetitions, wrong travel direction or
distance, and a final unbalanced stance. A run must include faster cadence and a brief flight phase with both
feet off the ground; a sped-up planted walk is not a run. A jumping jack must pair every requested jump with a
clear lateral foot spread and synchronized bilateral arm raise overhead, then close both feet and lower both
arms before the next repetition. A burpee must complete each requested crouch-to-plank transition, one palm-and-
toe push-up, a controlled return to foot support, and one airborne overhead-arm jump and landing, with no
skipped, reordered, or duplicated phase. A rhythmic dance must expose every requested beat as an alternating
foot lift and weight transfer, coordinate both arms with the beat, and return to a balanced stance. A squat must
show the exact number of distinct down-and-up cycles with a visible hip descent, both feet planted, and a full
standing-height return. A lunge must show the exact number of stagger-and-return cycles, the requested lead leg
or a clear alternating pattern, and a stance reset between cycles. A single-leg balance or knee raise must keep
the named support foot planted, lift only the free foot, and return to a balanced stance. A sit-up must begin
and return supine for every cycle and visibly curl the head and shoulders above the floor. A floor roll must
complete the requested forward/backward turn while travelling in that direction, visibly tuck, and transfer
support across the body. A cartwheel must complete one lateral turn in the named direction with extended arms, a
readable hand-support interval, feet passing above the hands, and a controlled two-foot landing. An airborne
flip or spin must complete the requested axis and angle while both feet are visibly off the ground, then land
without a snap or extra rotation. For obstacle-aware locomotion, a step-over must pass the named swing foot
above the named object's full top surface with visible margin, and an around-path must bend to the requested
side; reject walking beneath, teleporting past, or targeting the wrong object. For ladder climbing, require the
named ladder to remain visible, alternating contralateral hand and foot advances, continuous vertical root
travel in the requested direction, at least three stable limb contacts after acquisition, and a supported
terminal pose; reject a floating ascent or ground walking. For a grounded pose, verify the requested root
shift/drop and the pelvis, torso, hip, knee, ankle, and staggered-foot configuration at the decisive pose.
Distinguish a waist bend from a crouch, a lateral lean from a turn, a one-knee kneel from a symmetric squat, and
a seated posture from kneeling. A lie-down must rotate the full body to a clearly horizontal axis with broad
floor support, preserving the requested supine, prone, or side-lying orientation, and then return under control.
An all-fours request must establish a horizontal trunk supported by both open palms and both knees, with the
pelvis above rather than flattened onto the floor. A crawl additionally requires opposed hand and knee advances,
visible root travel in the requested direction, and recovery at the destination rather than a reset. A push-up
requires a straight plank, both palms planted, and the requested number of visible down/up torso cycles. A
static plank requires the same straight palm-and-toe alignment without push-up cycles."""

_SEQUENCE_FRAGMENT = """FAMILY: sequence. For an ordered mixed-action request, require every clause exactly
once and in the requested order. Reject a dropped, reordered, repeated, reset, or abruptly snapped step, and do
not accept actions connected by "then" being performed at the same time. Each step must complete before the next
begins, and each child-action boundary must stay continuous rather than resetting the body or the object to a
starting state. Judge every step against its own family's requirements as well as against the ordering."""

_UNKNOWN_FAMILY_FRAGMENT = """FAMILY: unspecified. No family-specific rubric applies to this clip, so judge the
request literally: every named action, hand, object, direction, count, and timing word in the request must be
visible in the rendered motion, and nothing that was not requested may replace it."""


FAMILY_FRAGMENTS: Mapping[str, str] = {
    "gesture": _GESTURE_FRAGMENT,
    "grab": _GRAB_FRAGMENT,
    "strike": _STRIKE_FRAGMENT,
    "composite": _COMPOSITE_FRAGMENT,
    "object_interaction": _OBJECT_INTERACTION_FRAGMENT,
    "full_body": _FULL_BODY_FRAGMENT,
    "sequence": _SEQUENCE_FRAGMENT,
    FALLBACK_FAMILY: _UNKNOWN_FAMILY_FRAGMENT,
}


#: Bumped whenever a grader core changes.  The sha256 catches edits that forget to
#: bump it; the version is what a calibration report is stated against (§3.5).
GRADER_PROMPT_VERSIONS: Mapping[GraderName, str] = {
    "semantic": "1.0",
    "anatomy": "1.0",
    "artifact": "1.0",
    "timing": "1.0",
    "crossview": "1.0",
}

FAMILY_FRAGMENT_VERSIONS: Mapping[str, str] = dict.fromkeys(FAMILY_FRAGMENTS, "1.0")


def family_for_intent(intent: str | None) -> str:
    """Map a manifest's `intent` onto a semantic-fragment family.

    An unknown or missing intent maps to `FALLBACK_FAMILY` rather than to a
    plausible-looking guess, so a family that silently stops matching shows up as
    a generic rubric in the record instead of the wrong rubric.
    """
    value = str(intent or "").strip()
    return value if value in FAMILY_FRAGMENTS and value != FALLBACK_FAMILY else FALLBACK_FAMILY


@dataclass(frozen=True)
class GraderPrompt:
    """One assembled grader prompt with the identity a calibration run cites."""

    name: GraderName
    version: str
    text: str
    family: str | None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def record(self) -> dict[str, str | None]:
        return {
            "grader": self.name,
            "version": self.version,
            "sha256": self.sha256,
            "family": self.family,
        }


def grader_prompt(name: GraderName, *, intent: str | None = None) -> GraderPrompt:
    """Assemble the prompt for one grader, composing the family fragment if it takes one."""
    if name not in GRADER_SPECS:
        raise ValueError(f"unknown grader: {name}")
    spec = GRADER_SPECS[name]
    core = GRADER_CORES[name]
    version = GRADER_PROMPT_VERSIONS[name]
    if not spec.family_specific:
        return GraderPrompt(name=name, version=version, text=core, family=None)
    family = family_for_intent(intent)
    fragment = FAMILY_FRAGMENTS[family]
    return GraderPrompt(
        name=name,
        version=f"{version}+{family}@{FAMILY_FRAGMENT_VERSIONS[family]}",
        text=f"{core}\n\n{fragment}",
        family=family,
    )
