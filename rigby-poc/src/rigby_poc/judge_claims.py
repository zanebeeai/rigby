"""Claim sets and the claims-to-score aggregation rule.

Plan 07 §1.5 diagnoses the current output: seven correlated 1-5 ratings that move
together, carry little information, and cannot be scored for accuracy against any
ground truth.  §3.2 replaces them with verifiable claims — each `yes`, `no`, or
`cannot_tell`, each citing a snapshot — and derives the dimension scores from an
explicit aggregation rule rather than asking a model for a number.

`cannot_tell` is a first-class answer.  It is what makes a low-evidence case
distinguishable from a failure, and conflating the two is how a judge ends up
reporting confident numbers about things it could not see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .judge_prompts import FALLBACK_FAMILY, GRADER_NAMES, GraderName, family_for_intent


Verdict = Literal["yes", "no", "cannot_tell"]

#: A dimension whose claims are mostly unanswerable is reported as unjudged, not
#: as a low score.  Half is the natural boundary: below it a majority of the
#: claims were actually settled.
INSUFFICIENT_EVIDENCE_FRACTION = 0.5


@dataclass(frozen=True)
class ClaimSpec:
    """One question a grader must answer about the rendered evidence."""

    id: str
    #: Written so that `yes` is always the good outcome. A rule that has to know
    #: which claims are negated is a rule that will eventually get one wrong.
    question: str
    #: A `no` here is disqualifying on its own: the dimension is capped below the
    #: acceptance threshold no matter how many other claims pass.
    critical: bool = False
    #: The `judge.FailureTag` a `no` verdict raises. Validated against that
    #: Literal by `tests/test_claim_aggregation.py`; kept as a plain string here
    #: so this module stays importable without `judge`.
    tag: str = "none"


# --------------------------------------------------------------- grader claims

# Joint-class granularity, using lane `anatomy`'s canonical vocabulary so a
# per-class VLM-versus-deterministic agreement number is computable directly.
# Per bone x DOF would be 156 questions for 52 bones and pure noise at n = 12.
ANATOMY_CLAIMS: tuple[ClaimSpec, ...] = (
    # Lane `anatomy` measured the `spine` class pooling the `hips` root bone,
    # which carries whole-body rotation: 180 deg of twist and 151 deg of flexion
    # in the cartwheel, none of it a spinal joint. Asked as one question, this
    # claim would be judging whether a cartwheel's root rotation is anatomically
    # plausible. The question excludes the root explicitly.
    ClaimSpec("anatomy.spine.flexion", "The back bends within a range a human spine could reach, ignoring whole-body rotation of the performer as a unit.", tag="arm_contortion"),
    ClaimSpec("anatomy.shoulder.abduction", "Each shoulder configuration is one a human shoulder could adopt.", tag="arm_contortion"),
    ClaimSpec("anatomy.elbow.flexion", "No elbow is bent backwards or hyperextended.", critical=True, tag="arm_contortion"),
    # The most diagnostic claim in this set. Lane `anatomy` measured the peak
    # frame carrying -96.1 deg of elbow abduction against 12.4 deg of flexion
    # while humeral twist is identically zero (max 6.5e-05 deg over 1606 frames).
    # A human must rotate the humerus to move the forearm out of the upper arm's
    # sweep plane; Rigby bends the elbow sideways instead. Present above 45 deg
    # in 251 of 1606 frames across 8 of 12 cases, so it is calibratable rather
    # than a tail event. Asked as a plane question, which is what is visible.
    ClaimSpec("anatomy.elbow.abduction", "The forearm stays in the plane the upper arm's rotation allows, rather than swinging sideways off the elbow hinge.", critical=True, tag="arm_contortion"),
    ClaimSpec("anatomy.forearm.twist", "Forearm rotation about its own long axis stays within a human range.", critical=True, tag="wrist_contortion"),
    ClaimSpec("anatomy.wrist.flexion", "No wrist is bent past what a human wrist could reach.", critical=True, tag="wrist_contortion"),
    # A distinct bone from the forearm. One question about "twist" would be
    # answering about two joints at once.
    ClaimSpec("anatomy.wrist.twist", "The hand does not rotate relative to the forearm beyond a human range.", critical=True, tag="wrist_contortion"),
    ClaimSpec("anatomy.hip.flexion", "Each hip configuration is one a human hip could adopt.", tag="arm_contortion"),
    ClaimSpec("anatomy.knee.flexion", "No knee is bent backwards or hyperextended.", critical=True, tag="arm_contortion"),
    ClaimSpec("anatomy.ankle.flexion", "No ankle is flexed past what a human ankle could reach.", tag="arm_contortion"),
    ClaimSpec("anatomy.digit.flexion", "Every finger configuration is one a human hand could form.", tag="finger_shape_error"),
    ClaimSpec("anatomy.no_self_intersection", "No limb passes through the performer's own body.", critical=True, tag="self_collision"),
)

ARTIFACT_CLAIMS: tuple[ClaimSpec, ...] = (
    ClaimSpec("acting_part_stays_in_frame", "The acting part of the body stays inside the frame.", critical=True, tag="hand_cropped"),
    ClaimSpec("no_ground_penetration", "No part of the body or object sinks through the ground plane.", critical=True, tag="ground_contact_error"),
    ClaimSpec("no_object_interpenetration", "No geometry passes through a scene object.", tag="self_collision"),
    ClaimSpec("support_feet_are_grounded", "Support feet rest on the ground rather than floating above it.", tag="balance_error"),
    ClaimSpec("silhouette_is_readable", "The silhouette is readable rather than occluded or self-overlapping.", tag="weak_readability"),
)

TIMING_CLAIMS: tuple[ClaimSpec, ...] = (
    ClaimSpec("transitions_are_continuous", "Consecutive tiles advance continuously rather than jumping.", critical=True, tag="timing_error"),
    ClaimSpec("no_frozen_then_jump", "No tile stays nearly frozen and is then followed by a large jump.", tag="timing_error"),
    ClaimSpec("no_unrequested_reversal", "The motion does not reverse direction mid-phase.", tag="timing_error"),
    ClaimSpec("holds_are_held", "Where the motion stages a hold, the hold actually settles.", tag="timing_error"),
    ClaimSpec("no_snap_between_phases", "No phase boundary snaps abruptly.", tag="timing_error"),
)

CROSSVIEW_CLAIMS: tuple[ClaimSpec, ...] = (
    ClaimSpec("views_agree_on_pose", "Paired views at the same timestamp show the same body pose.", critical=True, tag="cross_view_inconsistency"),
    ClaimSpec("views_agree_on_hand_shape", "Paired views show the same hand shape.", tag="cross_view_inconsistency"),
    ClaimSpec("views_agree_on_contact", "Paired views agree on whether an object is held or free.", tag="cross_view_inconsistency"),
    ClaimSpec("ego_head_motion_is_coherent", "Egocentric camera motion matches the head motion visible in orbit.", tag="camera_mismatch"),
)


# ------------------------------------------------- per-family semantic claims

_SEMANTIC_COMMON: tuple[ClaimSpec, ...] = (
    ClaimSpec("requested_action_is_present", "The requested action is the action performed.", critical=True, tag="wrong_gesture"),
    ClaimSpec("requested_hand_is_used", "The requested hand or side is the one used.", critical=True, tag="wrong_hand"),
    ClaimSpec("requested_counts_match", "Every requested repetition or beat count matches what is visible.", tag="missing_limb_motion"),
    ClaimSpec("requested_direction_matches", "Every requested direction matches what is visible.", tag="trajectory_error"),
    ClaimSpec("action_is_recognizable_unprompted", "A viewer who was not told the request would name the action correctly.", critical=True, tag="weak_readability"),
    ClaimSpec("no_unrequested_substitution", "Nothing that was not requested replaces a requested element.", tag="wrong_gesture"),
    ClaimSpec("setup_and_recovery_are_present", "The clip shows setup and recovery, not one endpoint pose.", tag="missing_follow_through"),
)

SEMANTIC_FAMILY_CLAIMS: Mapping[str, tuple[ClaimSpec, ...]] = {
    "gesture": (
        ClaimSpec("hand_shape_matches_request", "The hand shape is the requested one, digit for digit.", critical=True, tag="finger_shape_error"),
        ClaimSpec("shake_axis_is_correct", "A requested shake oscillates about the requested axis, not by wrist flexion or deviation.", tag="wrist_contortion"),
        ClaimSpec("returns_to_relaxed_default", "The hand returns to a relaxed default after the gesture.", tag="missing_follow_through"),
        ClaimSpec("ordered_contacts_are_distinct", "Ordered fingertip contacts happen in order with visible separation between them.", tag="coordination_error"),
    ),
    "grab": (
        ClaimSpec("grasp_closes_on_the_object", "The hand closes on the object rather than around empty space.", critical=True, tag="contact_error"),
        ClaimSpec("object_leaves_its_support", "The object visibly separates from its support.", tag="weak_lift"),
        ClaimSpec("hold_is_stable", "The object stays in the hand without slip or re-grip through the hold.", tag="object_slip"),
    ),
    "strike": (
        ClaimSpec("fist_closed_at_impact", "The striking hand is a closed fist at impact.", critical=True, tag="finger_shape_error"),
        ClaimSpec("strike_path_matches_type", "The fist travels on the path its strike type requires.", critical=True, tag="wrong_strike_path"),
        ClaimSpec("guard_is_maintained", "The non-striking hand stays in guard.", tag="missing_guard"),
        ClaimSpec("follow_through_and_recovery", "The strike follows through and recovers rather than stopping at impact.", tag="missing_follow_through"),
    ),
    "composite": (
        ClaimSpec("every_requested_limb_moves", "Every requested limb participates.", critical=True, tag="missing_limb_motion"),
        ClaimSpec("relative_phase_is_correct", "The relative phase and direction between limbs is what was requested.", tag="coordination_error"),
        ClaimSpec("path_shape_is_visible", "The requested path shape is visible rather than implied.", tag="trajectory_error"),
        ClaimSpec("limbs_stay_clear_of_each_other", "Coordinated limbs stay clear of each other.", tag="self_collision"),
    ),
    "object_interaction": (
        ClaimSpec("contact_precedes_object_motion", "The hand contacts the object before the object moves.", critical=True, tag="contact_error"),
        ClaimSpec("release_is_visible", "Where the lifecycle requires a release, the hand visibly opens at it.", tag="contact_error"),
        ClaimSpec("no_teleport_or_glue", "The object neither teleports nor stays glued to an open palm.", tag="object_slip"),
        ClaimSpec("lifecycle_order_is_complete", "The lifecycle runs in order with no phase skipped.", tag="missing_follow_through"),
    ),
    "full_body": (
        ClaimSpec("root_travel_matches_request", "Root travel direction and distance match the request.", critical=True, tag="root_motion_error"),
        ClaimSpec("support_changes_are_correct", "Support changes and foot placement are what the action requires.", tag="ground_contact_error"),
        ClaimSpec("no_foot_sliding", "Planted feet stay planted rather than sliding.", tag="foot_sliding"),
        ClaimSpec("finishes_balanced", "The performer finishes in a balanced stance.", tag="balance_error"),
    ),
    "sequence": (
        ClaimSpec("every_clause_appears_once", "Every requested clause appears exactly once.", critical=True, tag="wrong_gesture"),
        ClaimSpec("clause_order_is_correct", "The clauses appear in the requested order.", critical=True, tag="coordination_error"),
        ClaimSpec("no_overlap_between_clauses", "Clauses joined by \"then\" do not overlap.", tag="coordination_error"),
        ClaimSpec("boundaries_stay_continuous", "No boundary resets the body or the object to a starting state.", tag="timing_error"),
    ),
    FALLBACK_FAMILY: (),
}


def claim_specs(grader: GraderName, *, intent: str | None = None) -> tuple[ClaimSpec, ...]:
    """The claim set one grader must answer for one clip."""
    if grader == "anatomy":
        return ANATOMY_CLAIMS
    if grader == "artifact":
        return ARTIFACT_CLAIMS
    if grader == "timing":
        return TIMING_CLAIMS
    if grader == "crossview":
        return CROSSVIEW_CLAIMS
    if grader == "semantic":
        return _SEMANTIC_COMMON + SEMANTIC_FAMILY_CLAIMS[family_for_intent(intent)]
    raise ValueError(f"unknown grader: {grader}")


def claim_ids(grader: GraderName, *, intent: str | None = None) -> tuple[str, ...]:
    return tuple(spec.id for spec in claim_specs(grader, intent=intent))


def claim_instructions(grader: GraderName, *, intent: str | None = None) -> str:
    """The block appended to a grader prompt telling it exactly what to answer."""
    lines = [
        "Answer every claim below exactly once, in this order, using its id verbatim.",
        "Each verdict is 'yes', 'no', or 'cannot_tell'. 'yes' always means the claim holds.",
        "Answer 'cannot_tell' when the evidence genuinely cannot settle the claim. It is a",
        "correct answer and is scored differently from 'no'; do not guess to avoid it.",
        "Every claim must cite the snapshot id that decided it.",
        "",
    ]
    lines += [
        f"- {spec.id}: {spec.question}" + ("  [critical]" if spec.critical else "")
        for spec in claim_specs(grader, intent=intent)
    ]
    return "\n".join(lines)


# ------------------------------------------------------------- the aggregation

#: A `no` on a critical claim caps the dimension here — below the published
#: minimum of 4, so a disqualifying observation cannot be outvoted by volume.
CRITICAL_FAILURE_CAP = 2


@dataclass(frozen=True)
class DimensionVerdict:
    """The result of aggregating one grader's claims into one dimension."""

    dimension: str
    #: `None` means the evidence could not settle enough claims to score it.
    score: int | None
    yes: int
    no: int
    cannot_tell: int
    failed_claim_ids: tuple[str, ...]
    critical_failure_ids: tuple[str, ...]

    @property
    def insufficient_evidence(self) -> bool:
        return self.score is None

    def record(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "yes": self.yes,
            "no": self.no,
            "cannot_tell": self.cannot_tell,
            "insufficient_evidence": self.insufficient_evidence,
            "failed_claim_ids": list(self.failed_claim_ids),
            "critical_failure_ids": list(self.critical_failure_ids),
        }


def aggregate_claims(
    dimension: str,
    specs: tuple[ClaimSpec, ...],
    verdicts: Mapping[str, str],
) -> DimensionVerdict:
    """Derive one 1-5 dimension score from a claim set. Pure and total.

    The rule, in full:

    * Every claim in `specs` must have a verdict; a missing one raises rather
      than being treated as `cannot_tell`, so a truncated response cannot look
      like an honest abstention.
    * If more than half the claims are `cannot_tell`, the dimension is
      unjudged (`score=None`), not scored low.
    * A `no` on any claim marked critical caps the score at
      `CRITICAL_FAILURE_CAP`, which is below the published acceptance minimum.
    * Otherwise the score is a linear map of the settled pass fraction onto
      1-5, so all-pass is 5 and all-fail is 1.
    """
    if not specs:
        raise ValueError(f"{dimension} has no claims to aggregate")
    yes = no = unknown = 0
    failed: list[str] = []
    critical_failed: list[str] = []
    for spec in specs:
        verdict = verdicts.get(spec.id)
        if verdict not in {"yes", "no", "cannot_tell"}:
            raise ValueError(f"{dimension} claim {spec.id} has no usable verdict")
        if verdict == "yes":
            yes += 1
        elif verdict == "cannot_tell":
            unknown += 1
        else:
            no += 1
            failed.append(spec.id)
            if spec.critical:
                critical_failed.append(spec.id)

    if unknown > INSUFFICIENT_EVIDENCE_FRACTION * len(specs):
        score: int | None = None
    else:
        settled = yes + no
        score = 1 + round(4 * (yes / settled)) if settled else 1
        if critical_failed:
            score = min(score, CRITICAL_FAILURE_CAP)
    return DimensionVerdict(
        dimension=dimension,
        score=score,
        yes=yes,
        no=no,
        cannot_tell=unknown,
        failed_claim_ids=tuple(failed),
        critical_failure_ids=tuple(critical_failed),
    )


def unanswered_graders(verdicts_by_grader: Mapping[str, Mapping[str, str]]) -> tuple[str, ...]:
    return tuple(name for name in GRADER_NAMES if name not in verdicts_by_grader)
