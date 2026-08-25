from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path
from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, Iterable, Literal, Mapping

from openai import OpenAI
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, create_model

from .judge_claims import (
    INSUFFICIENT_EVIDENCE_FRACTION,
    DimensionVerdict,
    aggregate_claims,
    claim_ids,
    claim_specs,
)
from .judge_prompts import (
    GRADER_NAMES,
    GRADER_SPECS,
    GraderName,
    family_for_intent,
    grader_prompt,
)
from .observability import SpanLike, get_tracer
from .planner import load_environment


DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"
DEFAULT_JUDGE_FALLBACK_MODEL = "gpt-5.6-terra"
DEFAULT_JUDGE_REASONING_EFFORT = "low"
DEFAULT_JUDGE_IMAGE_DETAIL = "high"
DEFAULT_JUDGE_MAX_IMAGE_DIMENSION_PX = 960
DEFAULT_JUDGE_ESCALATION_CONFIDENCE = 0.70

# Mirrors acceptance_criteria.yaml -> autonomous_pipeline.minimum_selected_scores.
# Every dimension listed there is part of the published release threshold, so the
# rule below must gate on all four, not a subset.
ACCEPTANCE_MINIMUM_SCORES: dict[str, int] = {
    "semantic_match": 4,
    "gesture_recognizability": 4,
    "anatomical_naturalness": 4,
    "overall": 4,
}


FailureTag = Literal[
    "wrong_gesture",
    "wrong_hand",
    "wrist_contortion",
    "arm_contortion",
    "self_collision",
    "finger_shape_error",
    "hand_cropped",
    "camera_mismatch",
    "timing_error",
    "weak_readability",
    "cross_view_inconsistency",
    "contact_error",
    "object_slip",
    "weak_lift",
    "wrong_strike_path",
    "missing_guard",
    "missing_follow_through",
    "missing_limb_motion",
    "coordination_error",
    "trajectory_error",
    "root_motion_error",
    "foot_sliding",
    "balance_error",
    "ground_contact_error",
    "none",
]


class EvidenceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str = Field(min_length=1, max_length=32)
    observation: str = Field(min_length=3, max_length=240)


class MotionJudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_match: int = Field(ge=1, le=5)
    gesture_recognizability: int = Field(ge=1, le=5)
    anatomical_naturalness: int = Field(ge=1, le=5)
    temporal_readability: int = Field(ge=1, le=5)
    egocentric_visibility: int = Field(ge=1, le=5)
    cross_view_consistency: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    accept: bool
    confidence: float = Field(ge=0.0, le=1.0)
    failure_tags: list[FailureTag] = Field(min_length=1, max_length=8)
    evidence: list[EvidenceCitation] = Field(min_length=2, max_length=8)
    summary: str = Field(min_length=5, max_length=400)
    suggested_adjustment: str = Field(min_length=3, max_length=300)


class PairwiseJudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Literal["A", "B", "tie", "neither"]
    confidence: float = Field(ge=0.0, le=1.0)
    semantic_winner: Literal["A", "B", "tie", "neither"]
    anatomy_winner: Literal["A", "B", "tie", "neither"]
    timing_winner: Literal["A", "B", "tie", "neither"]
    visibility_winner: Literal["A", "B", "tie", "neither"]
    evidence: list[EvidenceCitation] = Field(min_length=2, max_length=8)
    rationale: str = Field(min_length=5, max_length=400)


class RepairPatch(BaseModel):
    """Bounded deltas proposed after a visually rejected candidate."""

    model_config = ConfigDict(extra="forbid")

    arm_height_delta: float = Field(ge=-0.30, le=0.30)
    arm_depth_delta: float = Field(ge=-0.30, le=0.30)
    lateral_offset_delta: float = Field(ge=-0.30, le=0.30)
    wrist_pitch_delta: float = Field(ge=-0.25, le=0.25)
    wrist_yaw_delta: float = Field(ge=-0.25, le=0.25)
    wrist_roll_delta: float = Field(ge=-0.25, le=0.25)
    elbow_swivel_delta: float = Field(ge=-0.30, le=0.30)
    torso_participation_delta: float = Field(default=0.0, ge=-0.25, le=0.25)
    path_arc_delta: float = Field(default=0.0, ge=-0.30, le=0.30)
    finger_splay_delta: float = Field(ge=-0.50, le=0.50)
    thumb_curl_delta: float = Field(ge=-0.40, le=0.40)
    little_curl_delta: float = Field(ge=-0.40, le=0.40)
    wrist_shake_amplitude_delta: float = Field(ge=-0.25, le=0.25)
    trajectory_amplitude_m_delta: float = Field(default=0.0, ge=-0.05, le=0.05)
    axial_rotation_amplitude_delta: float = Field(default=0.0, ge=-0.25, le=0.25)
    present_duration_scale: float = Field(ge=0.80, le=1.25)
    hold_duration_scale: float = Field(ge=0.80, le=1.25)
    shake_duration_scale: float = Field(ge=0.80, le=1.25)
    recover_duration_scale: float = Field(ge=0.80, le=1.25)
    easing_delta: float = Field(ge=-0.25, le=0.25)
    pose_root_scale: float = Field(default=1.0, ge=0.85, le=1.12)
    pose_directional_scale: float = Field(default=1.0, ge=0.80, le=1.35)
    object_distance_scale: float = Field(default=1.0, ge=0.85, le=1.15)
    object_apex_scale: float = Field(default=1.0, ge=0.75, le=1.25)
    object_contact_height_delta: float = Field(default=0.0, ge=-0.12, le=0.12)
    object_contact_depth_delta: float = Field(default=0.0, ge=-0.10, le=0.10)
    rationale: str = Field(min_length=5, max_length=400)


class FiveWayCandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["A", "B", "C", "D", "E"]
    semantic_match: int = Field(ge=1, le=5)
    gesture_recognizability: int = Field(ge=1, le=5)
    anatomical_naturalness: int = Field(ge=1, le=5)
    temporal_readability: int = Field(ge=1, le=5)
    egocentric_visibility: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    accept: bool
    failure_tags: list[FailureTag] = Field(min_length=1, max_length=8)
    summary: str = Field(min_length=5, max_length=300)
    suggested_adjustment: str = Field(min_length=3, max_length=240)


class FiveWayJudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[FiveWayCandidateScore] = Field(min_length=5, max_length=5)
    winner: Literal["A", "B", "C", "D", "E", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceCitation] = Field(min_length=3, max_length=10)
    rationale: str = Field(min_length=5, max_length=400)


def meets_acceptance_thresholds(score: BaseModel | Mapping[str, Any]) -> bool:
    """Apply the published minimum-selected-score rule to one judged candidate.

    Pure and total: a dimension that is missing or non-numeric fails the rule
    rather than being skipped, so a truncated payload cannot pass by omission.
    """
    values: Mapping[str, Any] = (
        score.model_dump(mode="json") if isinstance(score, BaseModel) else score
    )
    for dimension, minimum in ACCEPTANCE_MINIMUM_SCORES.items():
        value = values.get(dimension)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if value < minimum:
            return False
    return True


# 07d: the split graders become the default. The combined path stays reachable
# for the calibration comparison plan 10 §10.4 needs, and because production
# selection is the five-way listwise call rather than this one (07 §6.1).
DEFAULT_JUDGE_GRADER_MODE = "split"
JUDGE_GRADER_MODES: tuple[str, ...] = ("combined", "split")


class Claim(BaseModel):
    """One verifiable observation with a required snapshot citation (§3.2)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    verdict: Literal["yes", "no", "cannot_tell"]
    confidence: float = Field(ge=0.0, le=1.0)
    snapshot_id: str = Field(min_length=1, max_length=48)


class GraderClaims(BaseModel):
    """Every split grader returns claims. The dimension score is derived, never asked for."""

    model_config = ConfigDict(extra="forbid")

    claims: list[Claim] = Field(min_length=1, max_length=24)
    summary: str = Field(min_length=5, max_length=240)


class SemanticGraderClaims(GraderClaims):
    """The semantic grader additionally proposes the repair hint the loop consumes."""

    suggested_adjustment: str = Field(min_length=3, max_length=240)


GRADER_OUTPUT_MODELS: dict[GraderName, type[BaseModel]] = {
    "semantic": SemanticGraderClaims,
    "anatomy": GraderClaims,
    "artifact": GraderClaims,
    "timing": GraderClaims,
    "crossview": GraderClaims,
}



def _citable_snapshot_ids(payload_audit: list[dict[str, Any]]) -> tuple[str, ...]:
    """Every snapshot id this grader was actually shown, in payload order.

    A contact sheet is one image carrying many frames, so both the sheet's own id
    and the source ids tiled into it are legitimate citations. Derived from the
    audit the payload builder already writes rather than re-deriving from the
    manifest, so the enum cannot drift from what was sent.
    """
    seen: list[str] = []
    for entry in payload_audit:
        for key in ("snapshot_id",):
            value = entry.get(key)
            if isinstance(value, str) and value not in seen:
                seen.append(value)
        for value in entry.get("source_snapshot_ids") or ():
            if isinstance(value, str) and value not in seen:
                seen.append(value)
    return tuple(seen)


@cache
def grader_output_model(
    grader: GraderName,
    intent: str | None = None,
    snapshot_ids: tuple[str, ...] = (),
) -> type[BaseModel]:
    """The output schema for one grader, with `Claim.id` closed to its own claim set.

    **The free-form `id` was a real defect and it was invisible until a real
    model ran.** `Claim.id` was `str(min_length=1, max_length=64)` and `claims`
    was `1..24`, so nothing tied the response to the claim set the prompt asks
    for. `aggregate_claims` then requires every spec id exactly and raises on a
    miss — deliberately, so a truncated response cannot pass as an honest
    abstention. Schema permissive, aggregator strict, nothing between them.

    Measured against `gpt-5.6-luna` on one clip, five calls: the grader returned
    twelve claims every time and rendered the id's **final** separator as an
    underscore — `anatomy.elbow_abduction` for `anatomy.elbow.abduction` — on
    four of five calls, up to eight ids at once. Complete: 1 of 5. At five
    graders a clip, that is a run which scores almost nothing.

    Every judge in the test suite is a fake client (`docs/testing.md`), so no
    test could see it: the fakes emit the ids the aggregator wants. The split
    path has been the default since 07d and this contract had never met a real
    model until 10f's first call.

    Closing the enum is the fix rather than normalising `_` to `.` on read.
    Normalising accepts a malformed id and silently repairs it, which is how the
    next divergence goes unnoticed; a `Literal` makes the wrong id unrepresentable
    in the response, and the provider enforces it rather than this module hoping
    for it. `claims` is also pinned to exactly the expected count, so an omission
    is refused at the same boundary rather than surviving to the aggregator.
    """
    ids = claim_ids(grader, intent=intent)
    if not ids:
        raise ValueError(f"{grader}: no claim ids for intent {intent!r}")
    fields: dict[str, Any] = {"id": (Literal[ids], ...)}  # type: ignore[valid-type]
    if snapshot_ids:
        # `snapshot_id` was the second open field and it failed the same way. The
        # grader returned "01-orbit, 02-orbit, 03-orbit, 04-orbit, 05-orbit" -- a
        # comma-joined list in a scalar -- which `Claim` accepted at 48 chars and
        # `MotionJudgeScore.evidence[].snapshot_id` then rejected at 32, so the
        # clip died two layers downstream of the field that let it through. The
        # citable set is known exactly: it is what the payload actually showed
        # this grader. Closing it makes "cite one snapshot" unrepresentable as
        # "cite five".
        fields["snapshot_id"] = (Literal[snapshot_ids], ...)  # type: ignore[valid-type]
    closed_claim = create_model(
        f"Claim_{grader}_{family_for_intent(intent) if grader == 'semantic' else 'any'}",
        __base__=Claim,
        **fields,
    )
    base = SemanticGraderClaims if grader == "semantic" else GraderClaims
    return create_model(
        f"{base.__name__}_{grader}_{family_for_intent(intent) if grader == 'semantic' else 'any'}",
        __base__=base,
        claims=(list[closed_claim], Field(min_length=len(ids), max_length=len(ids))),  # type: ignore[valid-type]
    )


# `overall` is a summary of the dimensions that are measured directly, not a
# seventh measurement.  Deriving it from the three published dimensions that a
# grader actually reports keeps the acceptance rule non-circular; deriving it
# from all six would silently promote `temporal_readability`,
# `egocentric_visibility` and `cross_view_consistency` into gating dimensions,
# which `acceptance_criteria.yaml` deliberately does not do.
OVERALL_SOURCE_DIMENSIONS: tuple[str, ...] = (
    "semantic_match",
    "gesture_recognizability",
    "anatomical_naturalness",
)

# What a dimension scores when its claims could not be settled.  Any number here
# is a lie of some kind: the honest answer is `None`, which is what
# `DimensionVerdict.score` carries and what the record reports.  3 is chosen for
# the legacy `MotionJudgeScore` field because it is below the acceptance minimum
# — so an unjudged clip is never accepted — without asserting the failure that a
# 1 would claim was observed.
UNJUDGED_DIMENSION_SCORE = 3


@dataclass(frozen=True)
class AcceptanceDecision:
    """The explicit combination plan 07 §3.3 asks for, with its reason attached.

    `accepted = deterministic_valid and grader_verdict and score_threshold`.

    Before 07d each term lived somewhere different: `deterministic_valid` in the
    structural layer, `score_threshold` in `meets_acceptance_thresholds`, and
    `grader_verdict` nowhere at all — the model's self-reported boolean was
    copied verbatim by five call sites and never checked against the published
    rule. Making the combination one function is what lets the graders be
    measured against the deterministic layer instead of quietly agreeing with it.

    `reasons` is ordered and complete rather than short-circuited: a clip that
    fails on two terms says so. A single first-failure reason reads as though the
    other terms passed.
    """

    accepted: bool
    deterministic_valid: bool
    meets_thresholds: bool
    fully_judged: bool
    reasons: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "deterministic_valid": self.deterministic_valid,
            "meets_thresholds": self.meets_thresholds,
            "fully_judged": self.fully_judged,
            "reasons": list(self.reasons),
        }


def decide(
    scores: BaseModel | Mapping[str, Any],
    *,
    deterministic_valid: bool,
    unjudged_dimensions: Iterable[str] = (),
) -> AcceptanceDecision:
    """Apply the published rule as the authority. Pure and total.

    `deterministic_valid` is a required keyword with no default. A default here
    would be a value indistinguishable from a measurement — a caller that forgot
    to pass it would get a decision that looks complete, and the term the plan
    calls "always the authority" would silently be assumed true.
    """
    unjudged = set(unjudged_dimensions)
    gating_unjudged = sorted(unjudged & set(ACCEPTANCE_MINIMUM_SCORES))
    fully_judged = not gating_unjudged
    meets = meets_acceptance_thresholds(scores)
    reasons: list[str] = []
    if not deterministic_valid:
        reasons.append("deterministic_invalid")
    if not meets:
        reasons.append("below_published_thresholds")
    if gating_unjudged:
        reasons.append("unjudged_gating_dimensions:" + ",".join(gating_unjudged))
    return AcceptanceDecision(
        accepted=deterministic_valid and meets and fully_judged,
        deterministic_valid=deterministic_valid,
        meets_thresholds=meets,
        fully_judged=fully_judged,
        reasons=tuple(reasons),
    )


def _claim_verdicts(payload: Mapping[str, Any]) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for claim in payload.get("claims", []):
        verdicts[str(claim.get("id"))] = str(claim.get("verdict"))
    return verdicts


def grader_dimension_verdicts(
    grader: GraderName,
    payload: Mapping[str, Any],
    *,
    intent: str | None = None,
) -> dict[str, DimensionVerdict]:
    """Aggregate one grader's claims into every dimension it owns.

    A grader that owns two dimensions answers one claim set; both dimensions are
    derived from it.  `semantic_match` and `gesture_recognizability` genuinely do
    move together — pretending otherwise by asking twice would manufacture the
    fake independence §1.5 complains about.
    """
    specs = claim_specs(grader, intent=intent)
    verdicts = _claim_verdicts(payload)
    return {
        dimension: aggregate_claims(dimension, specs, verdicts)
        for dimension in GRADER_SPECS[grader].dimensions
    }


def aggregate_grader_dimensions(
    parts: Mapping[str, Mapping[str, Any]],
    *,
    intent: str | None = None,
) -> dict[str, DimensionVerdict]:
    """Derive every dimension from the five graders' claims. Pure and total."""
    merged: dict[str, DimensionVerdict] = {}
    for name in GRADER_NAMES:
        payload = parts.get(name)
        if payload is None:
            raise ValueError(f"missing grader result: {name}")
        merged.update(grader_dimension_verdicts(name, payload, intent=intent))
    settled = [
        merged[dimension].score
        for dimension in OVERALL_SOURCE_DIMENSIONS
        if merged[dimension].score is not None
    ]
    merged["overall"] = DimensionVerdict(
        dimension="overall",
        score=min(settled) if len(settled) == len(OVERALL_SOURCE_DIMENSIONS) else None,
        yes=0,
        no=0,
        cannot_tell=0,
        failed_claim_ids=(),
        critical_failure_ids=(),
    )
    return merged


def unexpected_claim_ids(
    grader: GraderName,
    payload: Mapping[str, Any],
    *,
    intent: str | None = None,
) -> list[str]:
    """Claim ids a grader invented. Recorded rather than raised, so one hallucinated
    id does not discard four correct graders — but never silently ignored."""
    known = set(claim_ids(grader, intent=intent))
    return sorted({claim_id for claim_id in _claim_verdicts(payload) if claim_id not in known})


def _merge_failure_tags(
    verdicts: Mapping[str, DimensionVerdict],
    parts: Mapping[str, Mapping[str, Any]],
    *,
    intent: str | None = None,
) -> list[str]:
    """Failure tags are derived from which claims failed, not asked for separately."""
    tags: list[str] = []
    for name in GRADER_NAMES:
        by_id = {spec.id: spec for spec in claim_specs(name, intent=intent)}
        for dimension in GRADER_SPECS[name].dimensions:
            for claim_id in verdicts[dimension].failed_claim_ids:
                tag = by_id[claim_id].tag
                if tag != "none" and tag not in tags:
                    tags.append(tag)
    return tags[:8] or ["none"]


def _merge_evidence(parts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cite the claims that decided the outcome, worst news first.

    A `no` is what a reader needs to see; a `cannot_tell` is the second most
    informative thing, because it says the evidence could not settle the
    question.  Filling the citation list with passing claims would bury both.
    """
    ranked: dict[str, list[dict[str, Any]]] = {"no": [], "cannot_tell": [], "yes": []}
    for name in GRADER_NAMES:
        for claim in parts[name].get("claims", []):
            verdict = str(claim.get("verdict"))
            if verdict not in ranked:
                continue
            ranked[verdict].append(
                {
                    "snapshot_id": str(claim.get("snapshot_id")),
                    "observation": f"{name}/{claim.get('id')}: {verdict}"[:240],
                }
            )
    ordered = ranked["no"] + ranked["cannot_tell"] + ranked["yes"]
    return ordered[:8]


def _mean_claim_confidence(payload: Mapping[str, Any]) -> float:
    claims = payload.get("claims", [])
    values = [float(claim.get("confidence", 0.0)) for claim in claims]
    return sum(values) / len(values) if values else 0.0


def assemble_split_score(
    parts: Mapping[str, Mapping[str, Any]],
    *,
    intent: str | None = None,
) -> tuple[MotionJudgeScore, dict[str, DimensionVerdict]]:
    """Build one `MotionJudgeScore` from the five graders' claims.

    `accept` is *derived* here by applying the published rule, rather than being
    the model's own boolean as it is on the combined path.  A dimension the
    evidence could not settle is never accepted: you cannot accept what you could
    not judge.
    """
    verdicts = aggregate_grader_dimensions(parts, intent=intent)
    scores = {
        dimension: (
            UNJUDGED_DIMENSION_SCORE if verdict.score is None else verdict.score
        )
        for dimension, verdict in verdicts.items()
    }
    fully_judged = all(
        verdicts[dimension].score is not None for dimension in ACCEPTANCE_MINIMUM_SCORES
    )
    summary = " ".join(
        str(parts[name].get("summary", "")).strip() for name in GRADER_NAMES
    ).strip()[:400]
    score = MotionJudgeScore.model_validate(
        {
            **scores,
            "accept": fully_judged and meets_acceptance_thresholds(scores),
            "confidence": min(_mean_claim_confidence(parts[name]) for name in GRADER_NAMES),
            "failure_tags": _merge_failure_tags(verdicts, parts, intent=intent),
            "evidence": _merge_evidence(parts),
            "summary": summary,
            "suggested_adjustment": str(
                parts["semantic"].get("suggested_adjustment", "No adjustment proposed.")
            ),
        }
    )
    return score, verdicts


UNARY_SYSTEM_PROMPT = """You are a strict animation-quality judge for first-person humanoid arm motion.
Judge only the visible rendered evidence and the user's motion request. Evidence includes detailed pose frames
and chronological timeline sheets; every timeline tile preserves the complete uncropped 16:9 field of view and
is labeled with its snapshot id and time. Each phase appears in both egocentric and orbit views. Penalize unnatural
wrist bending, forearm/arm contortion, incorrect finger configuration, cropped hands, self-intersection,
unreadable silhouettes, abrupt or poorly staged motion, and disagreements between views. A hang-ten/shaka
requires thumb and little finger extended with the middle three fingers curled. Do not infer quality from
candidate names, parameters, seeds, or pass/fail gate outputs; they are intentionally hidden. Score 1 as clearly
failed, 3 as recognizable but flawed, and 5 as natural and immediately readable. Accept only when overall is
at least 4, anatomy is at least 4, recognizability is at least 4, and no severe failure is visible. Cite exact
snapshot ids for every important conclusion. For temporal readability, explicitly compare consecutive timeline
tiles: penalize a pose that stays nearly frozen and then jumps, advances in steps, reverses during presentation,
or fails to hold. Do not infer smooth timing from the final pose alone. Use failure tag 'none' only when no failure
tag applies. For finger semantics,
compare every measured normalized curl with the requested hand shape's reference: a hang-ten must have low thumb
and little-finger curl and high index, middle, and ring curl. Reject when these measurements contradict the gesture.
A thumbs-up requires an extended thumb with the other four fingers curled; a peace/victory sign requires extended,
separated index and middle fingers with the ring and little fingers curled.
For ordered intra-hand dexterity, require the named driver digit to make visibly distinct fingertip contacts in the
requested order, with a readable separation between contacts rather than one held pinch. Use the chronological
thumb-to-digit tiles and the measured fingertip-distance records together. If the request says to look at the hand,
require the head-mounted view to track that hand naturally while orbit evidence confirms the head turn; reject a
head-only substitute, wrong order, simultaneous collapse, skipped fingertip, or static pinch.
When the request asks for a shake or back-and-forth motion, require visible repeated direction reversals during
that phase, roughly the requested number of beats, and a clean return to the relaxed default after it. For a
hang-ten/shaka shake, the oscillation must be forearm pronation/supination about the forearm's long axis. Penalize
wrist flexion/extension or side-to-side wrist deviation used as the main oscillation. Reject a requested forearm shake whose oscillation is carried by
wrist flexion or side-to-side deviation instead of forearm rotation. For punches, require a closed fist, readable guard/load,
decisive impact path, controlled follow-through, and recovery. A hook must travel on a lateral curved arc with
a visibly bent elbow and torso participation; reject a hook that reads as a straight jab, static crossed-arm pose,
  or slow arm placement.
For physical object throws, require secure contact, a style-appropriate windup, hand opening at release, continuous
ballistic free flight, follow-through, and recovery; reject a block that stays glued to an open palm, teleports, or
flies before release. For catches, require the open hand to meet the incoming object, closure only at contact, a
short absorbing motion, and stable retention. Use both views to verify object/hand coincidence at attachment.
For pushes and pulls, require contact before object motion, continuous hand/object coincidence through the guided
phase, travel in the requested direction, no teleport or premature release, and a clean separation before recovery.
For a roll, require that same continuous guided contact and support-plane travel, plus visible rotation about the
axis perpendicular to travel; reject sliding without rotation, support-plane drift, or rotation without translation.
For a support-plane spin, require continuous contact while the object center stays fixed and completes the requested
yaw turns; reject translation masquerading as a spin, support drift, or a static object under a moving hand.
For controlled placement, require secure pickup, visible lift, continuous attached transport, lowering onto the
support surface before the hand opens, clean separation, and recovery without a reset or ballistic drop.
For drops, require a secured lift, visible hand opening before separation, gravity-driven vertical fall, supported
landing without tunneling, and recovery only after the object is free.
For hand-to-hand object transfers, require the named source hand to secure and present the object, the receiver to
make visible contact before the source opens, a continuous ownership transfer with no teleport, and stable retention
by the named receiver after the source withdraws.
For carries, require the initial grasp to remain closed and the object to follow the carrying hand continuously
through every requested step; reject a reset to the original table position, floating object, open-hand carry,
or object motion that fails to follow the translated/turned body.
When a carry is followed by a drop or throw, require the release to begin from the transported world position after
the requested travel, with no second pickup or reset to the original table; then require continuous free flight and
a supported landing before recovery.
For an overhead lift, require the grasp to stay closed and the object center to rise clearly above the head rather
than stopping at chest, shoulder, or face height.
For coordinated or multi-part movement, verify every requested limb moves, the relative phase and direction between
limbs is correct, the path type and repetition count are visible, and limbs do not pass through each other. For
an explicit basketball travel signal, require two closed-fist forearms held across the chest, one over the other,
with each fist staying near the opposite elbow. The complete parallel forearms must roll together around one shared
cross-body axis, repeatedly exchanging over/under and front/back order for about three revolutions. Reject two
independent hand circles, fixed elbows with small wrist flourishes, diverging forearm axes, intersecting forearms,
or a static crossed-arm substitute. Judge the whole chronological
sequence—including setup and recovery—not merely one attractive endpoint pose. For an
ordered mixed-action request, require every clause exactly once and in the requested order. Reject a dropped,
reordered, repeated, reset, or abruptly snapped step, and do not overlap actions connected by "then". For
full-body motion, use the orbit view to judge root travel, turns, leg/foot action, support changes, balance, ground
clearance, landing, and recovery; use the egocentric view to confirm coherent camera motion. The ego camera is the
performer's literal head-mounted first-person view: it is correct for the torso, legs, and complete performer to be
absent. Never lower visibility or tag camera_mismatch merely because ego and orbit have different framing; score ego
visibility from naturally visible arms/environment and use orbit for whole-body anatomy. Penalize foot sliding,
ground penetration, floating support feet, missing requested steps/repetitions, wrong travel direction or distance,
and a final unbalanced stance. A run must include faster cadence and a brief flight phase with both feet off the
ground; reject a run that is only a sped-up planted walk. A jumping jack must pair every requested jump with a clear
lateral foot spread and synchronized bilateral arm raise overhead, then close both feet and lower both arms before
the next repetition. A burpee must complete each requested crouch-to-plank transition, one palm-and-toe push-up,
a controlled return to foot support, one airborne overhead-arm jump and landing, with no skipped, reordered, or duplicated phase.
A rhythmic dance must expose every requested beat as an alternating foot lift and weight transfer, coordinate both
arms with the beat rather than freezing them, preserve planted-foot stability, and return to a balanced stance.
A squat must show the exact number of distinct down-and-up cycles, a visible hip descent with both feet planted,
and a full standing-height return before the next repetition.
A lunge must show the exact number of distinct stagger-and-return cycles, the requested lead leg/direction or a
clear alternating lead pattern, visible knee/root descent, stable planted feet, and a stance reset between cycles.
A single-leg balance or knee raise must keep the named support foot planted without sliding, visibly lift only the
free foot, shift the body over the support leg without tipping, and return both feet to a balanced stance.
A sit-up must begin and return supine for every requested cycle, visibly curl the head and shoulders above the floor,
keep a stable floor support set without hovering or penetration, and avoid extra hidden upright excursions.
A floor roll must complete the requested forward/backward turn while travelling in that direction, visibly tuck,
transfer support from the feet across the body, avoid penetration, and recover to balance. A cartwheel must complete
one lateral turn in the named direction with extended arms, a readable hand-support interval, feet passing above the
hands, and a controlled two-foot landing. An airborne flip or spin must complete the requested axis and angle while
both feet are visibly off the ground, then land without a snap, slide, or extra rotation.
For obstacle-aware locomotion, use the orbit/environment view: a step-over must pass the named swing foot above the
named object's full top surface with visible margin, while an around-path must bend to the requested side and preserve
body clearance throughout. Reject clipping through, walking beneath, teleporting past, or targeting the wrong object.
For ladder climbing, require the named ladder to remain visible, alternating contralateral hand/foot advances,
continuous vertical root travel in the requested direction, at least three stable limb contacts after acquisition,
and a supported terminal pose. Reject floating ascent, ground walking, wrong-object contact, or limbs passing through
the ladder instead of meeting its front/rungs.
For a grounded pose request, verify the requested root shift/drop and pelvis, torso, hip, knee, ankle,
and staggered-foot configuration at the decisive pose. Distinguish a waist bend from a crouch, a lateral lean from
a turn, a one-knee kneel from a symmetric squat, and a seated posture from kneeling. A lie-down request must rotate
the full body to a clearly horizontal axis, establish broad floor support without penetration or hovering, preserve
face-up supine versus face-down prone orientation, or the requested left/right side-lying orientation, and then return under control. An all-fours request must establish
a horizontal trunk supported by both open palms and both knees, with the pelvis above rather than flattened onto the floor. A crawl additionally requires opposed
hand/knee advances, visible root travel in the requested direction, and recovery at the destination rather than a reset. A push-up requires a straight plank,
both palms planted, the requested number of visible down/up torso cycles, and a return from the floor-support state. A static plank requires the same straight
palm-and-toe support alignment without push-up cycles or visible support drift."""


PAIRWISE_SYSTEM_PROMPT = """You are a strict blinded pairwise animation judge. Compare candidate A and B
against the same user request using only detailed full-frame poses and chronological sheets whose tiles preserve
the complete uncropped egocentric and orbit views. Explicitly compare consecutive tiles for snaps, steps, reversals,
and missing holds rather than inferring smooth timing from a final pose.
Prefer the candidate whose gesture is more immediately recognizable, anatomically natural, fully visible,
temporally readable, and consistent across views. Ignore candidate ordering. Choose tie only when differences
are genuinely negligible, and neither when both clearly fail. Cite snapshot ids that include the candidate
prefix. Never infer quality from hidden parameters, seeds,
filenames, or prior scores. When a shake is requested, compare the visible direction reversals and recovery—not
just the held endpoint—and penalize a single flourish, frozen motion, or failure to return to default. A
hang-ten/shaka shake should rotate through forearm pronation/supination about the long axis while the wrist joint
stays stable; penalize wrist flexion/extension or side-to-side deviation masquerading as the shake. For a hook,
prefer a closed fist moving through a lateral curved arc with a bent elbow, guarded opposite hand, torso drive,
follow-through, and clean recovery; penalize a straight extension or merely crossed forearms.
For ordered intra-hand dexterity, compare every named thumb-to-fingertip contact in chronological order and require
visible release between contacts. Prefer concurrent natural gaze at the hand when requested; reject a held pinch,
skipped or reordered digit, all-at-once closure, or head motion that substitutes for the finger action.
For a basketball travel signal, require closed fists across the chest near the opposite elbows and near-parallel
forearms rolling as one coupled pair around a shared cross-body axis. Prefer clear repeated exchanges of over/under
and front/back order with persistent clearance, the requested repetitions, and a clean recovery. Reject two
independent wrist circles, a static crossed pose, diverging forearms, or missing order exchanges.
For a physical throw, compare secure pickup, windup style, exact opening/release, free ballistic arc, requested
direction/distance, follow-through, and recovery. For a catch, compare interception timing, closure at contact,
absorption, and stable retention. Reject teleports, pre-contact attachment, post-release glue, or an object/palm gap.
For a support-plane roll, compare continuous guiding contact, requested travel direction/distance, visible rotation
coupled to translation, stable height, separation before recovery, and no teleport-sized step.
For a support-plane spin, compare fixed-center yaw completion, continuous guiding contact, stable support height,
clean separation, and exact requested direction/count when specified.
For controlled placement, compare pickup security, requested direction/distance, continuous transport, support
contact before release, hand opening, and a stationary placed object during recovery.
For a handoff, compare named source/receiver correctness, dual-hand overlap before release, continuous transfer,
source withdrawal, and final receiver retention.
When a carry precedes a drop or throw, require one continuous palm attachment through locomotion and the child-action
boundary, followed by release from the transported position, free flight, landing, and no replayed pickup.
For full-body requests, compare orbit sequences for root direction/distance, turn amount, step or repetition count,
leg coordination, support-foot stability, ground clearance, balance, landing, and final settle. The ego view should
move coherently with the head/root but is not expected to show the torso, legs, or complete performer. That absence is
correct first-person framing, not camera_mismatch or low visibility. Penalize foot sliding or ground penetration.
A run requires faster cadence and a visible brief flight phase, not merely faster planted walking. For jumping jacks,
compare exact repetition count, synchronized overhead bilateral arm raises, lateral foot spreading, and closed returns.
For burpees, compare exact repetition count and require a crouch, palm-and-toe plank with one push-up, return to both
feet, airborne overhead-arm jump, landing, and recovery in that order for every repetition. For squats, compare exact
down-and-up cycle count, visible depth, planted bilateral support, and complete stance recovery between repetitions.
For rhythmic dances, compare exact beat count, left/right foot-lift alternation, visible lateral weight transfer,
bilateral arm participation, planted support stability, and a clean balanced exit.
For lunges, compare exact cycle count, requested direction and lead-leg alternation, visible foot stagger/depth, and
complete stance recovery before the next repetition.
For single-leg balance, compare named support/free-leg correctness, raised-foot clearance, planted-foot stability,
upright balance, and the controlled two-foot recovery.
For sit-ups, compare exact curl-and-supine-return count, shoulder/head lift, bent-leg floor support, monotonic phase
transitions, and the final controlled return to standing.
For floor rolls, compare complete signed pitch rotation, travel direction, tuck, body-support transfer, and recovery.
For cartwheels, compare named lateral direction, hand support, inversion with both feet above the hands, extended
limbs, landing, and recovery. For airborne flips and spins, compare requested axis/angle, visible loss of foot support,
adequate clearance, controlled landing, and absence of hidden extra turns.
For obstacle-aware locomotion, compare the exact named object, over-versus-around relation, requested detour side,
swing-foot/top clearance or whole-body lateral clearance, and continuous arrival beyond the obstacle.
For ladder climbing, compare requested up/down direction and height, exact contact-cycle count, alternating
contralateral hand/foot advances, three-point support, limb-to-rung coincidence, and stable final ladder support.
For grounded
poses, prefer the candidate whose requested bend/lean/kneel/seated configuration is clearest without losing contact
or balance, and reject a visibly different posture family. For an ordered mixed-action request, compare whether
every clause appears exactly once in the requested order and whether every child-action boundary stays continuous;
reject dropped, reordered, overlapped, repeated, or reset steps. For lie-down poses, require a horizontal body axis,
the requested prone/supine/side facing, broad floor support, no penetration or hovering, and controlled recovery. For
all fours, require visible bilateral palm-and-knee support beneath a horizontal trunk rather than a prone lie-down. For crawling, require opposed limb alternation,
requested step count and travel direction, sustained low support, and no reset to the starting point. For push-ups, compare exact repetition count,
straight palm-and-toe plank alignment, stable planted palms, visible elbow/torso lowering, and full extension between repetitions. For a static plank, require
the same straight four-point support without cyclic lowering or support drift."""


REPAIR_SYSTEM_PROMPT = """You repair a bounded parametric humanoid arm/hand motion after a separate visual judge rejects it.
The motion may be a gesture, strike, object pickup, one-/two-arm compositional sequence, procedural full-body
sequence, or an ordered sequence of those executable skills. Return small parameter deltas, never a new animation or
new intent. arm_height moves the wrist vertically;
arm_depth moves it forward/back; lateral_offset moves it sideways; wrist_pitch/yaw are bounded hand-joint
flexion/deviation; wrist_roll is implemented anatomically as forearm pronation/supination; elbow_swivel changes
the elbow pole; positive finger_splay widens the gesture silhouette; negative thumb_curl/little_curl extends those
digits. For gestures, duration scales alter the present, hold, shake, and recover phases. For object pickup,
present_duration_scale alters reach/preshape, hold_duration_scale alters contact/close/hold,
shake_duration_scale alters lift, and recover_duration_scale alters recovery. Positive easing moves toward a
minimum-jerk trajectory. wrist_shake_amplitude_delta and shake_duration_scale may repair a requested repeated
shake by changing the amplitude and timing of forearm pronation/supination, but the cycle count is semantic and
must not change. Never repair a hang-ten shake by adding wrist flexion/extension or side-to-side deviation.
Preserve the requested hand, action or gesture shape, object, direction words, phase order, cycle count, and timing category.
For throw/catch, the duration scales adjust preparation, release/intercept, flight/absorption, and recovery.
object_distance_scale and object_apex_scale adjust the free-flight path without reversing it;
object_contact_height_delta and object_contact_depth_delta move a catch intercept within the reachable workspace.
Preserve action, style, hand, object, horizontal direction, lifecycle order, and whether the object is held or free.
For strikes, present_duration_scale alters guard/load, hold_duration_scale alters strike/follow-through, and
recover_duration_scale alters recovery. The existing bounded path arc and torso parameters must preserve the
selected hook/jab/cross/uppercut type. For compositional movement, present_duration_scale alters move phases,
shake_duration_scale alters cyclic phases, trajectory_amplitude_m_delta changes path size, and
axial_rotation_amplitude_delta changes visible pronation/supination about each forearm's long axis. Arm height/depth,
lateral, wrist, and elbow deltas apply symmetrically to every active effector. Use axial rotation—not wrist bending—
when the request calls for rolling forearms. Preserve limb count, relative phase, path type, and cycle count. Prefer
the smallest changes that directly address the cited visual failures.
For ordered dexterous motion, preserve every typed thumb-to-fingertip contact, its order, the open releases between
contacts, and any hand-gaze target. Repairs may restage the arm or timing but may not replace the sequence with one
pinch or change the contacted digits.
For full-body motion, present_duration_scale changes body-action timing and recover_duration_scale changes the final
balanced settle; preserve action type, root direction/distance, turn direction, lead side, height, repetition count,
continuous rotation axis/degrees/support mode, and every authored grounded-pose direction. A pose repair may adjust timing/easing but must not swap bend direction,
lean side, kneeling side, seated stance, or foot-contact mode. pose_root_scale changes only root drop and foot-placement
magnitudes within a narrow physical margin; pose_directional_scale changes root shift and joint-angle magnitudes while
preserving every sign. Use them when a requested pose is too subtle or too extreme.
Structural gates validate the result."""


FIVE_WAY_SYSTEM_PROMPT = """You are a strict blinded animation selector for first-person humanoid arm, hand,
full-body, gesture, locomotion, and object-manipulation motion. Rank exactly five candidates A-E against
one user request using only detailed full-frame poses and chronological sheets whose tiles preserve complete
uncropped egocentric and orbit views. Score every candidate using
the same standard: semantic match to every requested modifier, immediate action or gesture recognizability,
anatomical wrist/arm naturalness, timing readability, complete first-person visibility, and agreement between
views. For full-body requests, "agreement between views" means temporal/camera coherence, not identical framing: the
ego camera is mounted at the performer's head and should not show the complete performer, torso, or legs. Do not lower
egocentric_visibility or tag camera_mismatch for that correct first-person absence; judge whole-body semantics and
anatomy from orbit, and judge ego only for coherent head motion plus naturally visible arms/environment. Semantic
match includes the requested action, style, quick/balanced/slow timing, low/chest/high placement, close/natural/extended
reach, inward/center/outward staging, hand, and wrist pitch/yaw/roll. For object pickup, require a readable
approach, preshape, contact, close, lift, and stable hold without visible slip. A shaka requires thumb and little
finger extended with the middle three curled. Accept a candidate only when semantic match, overall, anatomy, and
recognizability are all at least 4 with no severe failure. Select one winner only when an accepted candidate has a
clear, non-negligible visual advantage in prompt fidelity or motion quality
that an independent human should notice without access to parameters or diagnostics. If accepted candidates are
effectively tied, return none; do not manufacture one-point score differences from tiny pose or timing changes.
For physical throws, require secure contact before windup, opening at release, a continuous free ballistic arc in
the requested direction, follow-through, and recovery. For catches, require open-hand interception, closure at
contact, absorption, and stable retention. Reject teleports, object/palm gaps, pre-contact glue, post-release glue,
wrong direction, or a flight unnecessarily hidden outside the complete egocentric FOV.
Visually indistinguishable candidates should receive the same scores. Also return none if none pass. Candidate
labels are randomized, so ignore ordering. Explicitly compare consecutive timeline tiles for snaps, steps,
reversals, and missing holds. Cite candidate-prefixed snapshot ids for the decisive visual
evidence. When the request includes a repeated shake, score its visible direction reversals, requested beat count,
and return to default as explicit semantic requirements rather than treating all matching endpoint poses as equal.
For an ordered thumb-to-fingertip action, require distinct contacts in the requested sequence with open separation
between them; reject a single held pinch, wrong order, skipped digit, or all fingers closing together. When looking
at the hand is requested, require the head/camera gaze to follow the active hand concurrently instead of replacing
the dexterous action.
For a hang-ten/shaka, require forearm pronation/supination about the forearm's long axis with a stable wrist joint;
penalize flexion/extension or side-to-side wrist deviation used as the oscillation. For a punch, require a closed
fist, readable guard/load, decisive strike, controlled follow-through, and recovery. A hook specifically requires
a lateral curved fist path, bent elbow, guarded opposite hand, and visible torso drive; score a straight jab-like
  extension, frozen crossed-arm pose, or missing recovery as a semantic failure. For coordinated or multi-part
  movement, require every requested limb, correct relative phase and direction, visible path shape and repetition
  count, collision-free coordination, and a complete setup-to-recovery sequence. A basketball travel signal
  requires closed fists near the opposite elbows and near-parallel forearms rolling as one coupled pair around a
  shared cross-body axis. Require repeated exchanges of over/under and front/back order, persistent clearance, the
  requested repetitions, and a clean recovery. Reject independent wrist circles, a static crossed pose, diverging
  forearms, or missing order exchanges. For an explicit ordered request,
  require every clause exactly once and in order, with no reset or snap between actions; dropping, reordering,
  overlapping "then" clauses, or repeating a step is a semantic failure. For full-body motion, use orbit
  views to verify root direction/distance, turn amount, leg and foot coordination, support changes, ground clearance,
  balance, landing, and final settle; use ego views to verify coherent head/root camera motion. Reject visible foot
  sliding, ground penetration, floating support feet, missing steps/repetitions, or an unbalanced recovery. A run
  must show faster cadence and a brief flight phase with both feet off the ground; a sped-up planted walk is not a
  semantically correct run. For a grounded pose, compare the visible root, torso, pelvis, leg, and foot configuration
  with every requested direction and support relationship; a crouch is not a bend, a squat is not a one-knee kneel,
  and kneeling is not sitting. A lie-down pose must become horizontal with correct prone/supine facing and broad,
  non-penetrating ground support; side-lying must visibly rotate laterally onto the requested side. Reject a deep crouch,
  kneel, floating body, or wrong-facing substitute. An all-fours
  pose must show a horizontal trunk with two open palms and two knees bearing support; reject prone, hovering-hand,
  standing, or crouched substitutes. A push-up must retain a straight palm-and-toe plank while the torso visibly lowers
  and rises for the requested number of cycles; reject all-fours rocking or planted-hand drift. A crawl must translate
  through alternating opposed palm/knee advances and stand at the destination rather than sliding or resetting.
  A floor roll must visibly tuck and transfer support across the body while completing its signed turn and travel.
  A cartwheel requires extended arms, hand support, feet above the hands, the named lateral direction, and a stable
  landing. An airborne flip or spin requires the requested axis/angle, genuine airborne clearance, and a controlled
  landing without snaps or hidden extra turns. For obstacle traversal, require the exact named scene object and
  visible over/around clearance without clipping, teleportation, or a straight-through substitute."""


def _usage(response: object) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return dict(usage) if isinstance(usage, dict) else {}


class CompactedEvidenceError(ValueError):
    """The run was archived, not broken.

    A `ValueError` subclass so existing handlers keep catching it, but nameable
    so a caller that wants to skip archived runs can do so without matching on a
    message string.
    """


def _manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence manifest must be an object")
    contract = value.get("capture_contract")
    if not isinstance(contract, dict):
        raise ValueError("evidence manifest is missing its capture contract")
    expected = {
        "raw_canvas_only": True,
        "width_px": 1600,
        "height_px": 900,
        "egocentric_vertical_fov_deg": 94.0,
        "device_pixel_ratio": 1,
        "ui_overlay_included": False,
    }
    for key, expected_value in expected.items():
        if contract.get(key) != expected_value:
            raise ValueError(f"capture contract {key} is not {expected_value!r}")
    # A compacted run's PNGs have been transcoded to WebP and deleted (PR 01d),
    # so every snapshot path below is genuinely missing. Saying so in the path
    # layer's vocabulary -- "snapshot path escapes or is missing" -- is true and
    # sends the reader to look for a capture bug. The run is archival by
    # decision, not broken.
    #
    # Keyed off the manifest rather than importing `evals.compact_run`: `src/`
    # should not depend on `evals/`, and the artifact-level key is what travels
    # with the evidence. `tests/test_compacted_run_is_not_judgeable.py` pins this
    # predicate against `is_compacted` so the two cannot drift apart.
    if isinstance(value.get("compaction"), dict):
        raise CompactedEvidenceError(
            "this run is compacted: its evidence is archival and not judgeable"
        )
    snapshots = value.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("evidence manifest contains no snapshots")
    verified: list[dict[str, Any]] = []
    ego_count = 0
    orbit_count = 0
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("snapshot record must be an object")
        view = snapshot.get("view")
        if view not in {"ego", "orbit"}:
            raise ValueError("snapshot view must be ego or orbit")
        ego_count += int(view == "ego")
        orbit_count += int(view == "orbit")
        if snapshot.get("width_px") != 1600 or snapshot.get("height_px") != 900:
            raise ValueError("snapshot does not preserve the full 1600x900 canvas")
        camera = snapshot.get("camera")
        if not isinstance(camera, dict) or camera.get("view") != view:
            raise ValueError("snapshot camera metadata does not match its view")
        if view == "ego" and camera.get("vertical_fov_deg") != 94:
            raise ValueError("egocentric snapshot did not use the complete 94-degree FOV")
        image_path = (path.parent / str(snapshot.get("path", ""))).resolve()
        if image_path.parent != path.parent.resolve() or not image_path.is_file():
            raise ValueError("snapshot path escapes or is missing from the evidence directory")
        image = image_path.read_bytes()
        if hashlib.sha256(image).hexdigest() != snapshot.get("sha256"):
            raise ValueError(f"snapshot hash mismatch: {snapshot.get('id')}")
        verified.append({**snapshot, "absolute_path": image_path, "image_bytes": image})
    if ego_count == 0 or orbit_count == 0 or ego_count != orbit_count:
        raise ValueError("judge evidence requires paired ego and orbit snapshots")
    return value, verified


def _key_pose_labels(manifest: dict[str, Any]) -> set[str]:
    intent = str(manifest.get("intent", ""))
    if intent == "grab":
        return {"hold_midpoint"}
    if intent == "object_interaction":
        return {
            "windup_ready",
            "release_pose",
            "flight_apex",
            "receive_ready",
            "flight_end",
            "absorb_end",
        }
    if intent == "strike":
        return {"strike_midpoint", "impact_pose"}
    return {"presented_pose"}


def _image_content(
    snapshots: list[dict[str, Any]],
    *,
    prefix: str = "",
    labels: set[str] | None = None,
    detail: str = DEFAULT_JUDGE_IMAGE_DETAIL,
    max_dimension_px: int = DEFAULT_JUDGE_MAX_IMAGE_DIMENSION_PX,
    payload_audit: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for snapshot in snapshots:
        if labels is not None and str(snapshot.get("label")) not in labels:
            continue
        snapshot_id = f"{prefix}{snapshot['id']}"
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"SNAPSHOT {snapshot_id}: phase={snapshot['phase']}, label={snapshot['label']}, "
                    f"view={snapshot['view']}, time={float(snapshot['rendered_time_s']):.3f}s"
                ),
            }
        )
        with Image.open(io.BytesIO(snapshot["image_bytes"])) as source:
            source.load()
            source_width, source_height = source.size
            scale = min(1.0, max_dimension_px / max(source.size))
            payload_width = max(1, round(source_width * scale))
            payload_height = max(1, round(source_height * scale))
            if (payload_width, payload_height) == source.size:
                payload_bytes = snapshot["image_bytes"]
            else:
                resized = source.resize((payload_width, payload_height), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                resized.save(buffer, format="PNG", optimize=True)
                payload_bytes = buffer.getvalue()
        if payload_audit is not None:
            payload_audit.append(
                {
                    "snapshot_id": snapshot_id,
                    "source_dimensions_px": [source_width, source_height],
                    "payload_dimensions_px": [payload_width, payload_height],
                    "full_frame_uncropped": True,
                    "detail": detail,
                }
            )
        encoded = base64.b64encode(payload_bytes).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{encoded}",
                "detail": detail,
            }
        )
    return content


def _timeline_content(
    snapshots: list[dict[str, Any]],
    *,
    prefix: str = "",
    detail: str = DEFAULT_JUDGE_IMAGE_DETAIL,
    max_dimension_px: int = DEFAULT_JUDGE_MAX_IMAGE_DIMENSION_PX,
    payload_audit: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    columns = 4
    tile_width = max_dimension_px // columns
    tile_height = round(tile_width * 9 / 16)
    label_height = 24
    for view in ("ego", "orbit"):
        selected = [snapshot for snapshot in snapshots if snapshot.get("view") == view]
        if not selected:
            continue
        rows = math.ceil(len(selected) / columns)
        sheet = Image.new(
            "RGB",
            (tile_width * columns, (tile_height + label_height) * rows),
            color=(12, 16, 22),
        )
        draw = ImageDraw.Draw(sheet)
        tile_descriptions: list[str] = []
        source_ids: list[str] = []
        for index, snapshot in enumerate(selected):
            row, column = divmod(index, columns)
            x = column * tile_width
            y = row * (tile_height + label_height)
            with Image.open(io.BytesIO(snapshot["image_bytes"])) as source:
                source.load()
                tile = source.convert("RGB").resize(
                    (tile_width, tile_height), Image.Resampling.LANCZOS
                )
            sheet.paste(tile, (x, y))
            snapshot_id = f"{prefix}{snapshot['id']}"
            time_s = float(snapshot["rendered_time_s"])
            draw.text((x + 4, y + tile_height + 4), f"{snapshot_id}  {time_s:.3f}s", fill="white")
            tile_descriptions.append(
                f"{snapshot_id}={snapshot['phase']}/{snapshot['label']}@{time_s:.3f}s"
            )
            source_ids.append(snapshot_id)
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": f"{prefix}CHRONOLOGICAL {view.upper()} TIMELINE: " + "; ".join(tile_descriptions),
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{encoded}",
                    "detail": detail,
                },
            ]
        )
        if payload_audit is not None:
            payload_audit.append(
                {
                    "snapshot_id": f"{prefix}timeline-{view}",
                    "source_snapshot_ids": source_ids,
                    "payload_dimensions_px": list(sheet.size),
                    "full_frame_uncropped_tiles": True,
                    "chronological": True,
                    "detail": detail,
                }
            )
    return content


def _call_record(response: object, parsed: BaseModel, model: str) -> dict[str, Any]:
    return {
        "response_id": getattr(response, "id", None),
        "model": getattr(response, "model", model),
        "usage": _usage(response),
        "parsed": parsed.model_dump(mode="json"),
    }


def _summed_usage(graders: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Token usage across every grader attempt that reported some.

    Attempts that raised carry no usage anywhere, so this is the *attributed*
    total rather than the cost of every dispatch. `grader_cost.cost_report`
    reports the ratio between them; summing here without that ratio beside it
    would understate cost by an unknown amount concentrated in the calls that
    failed hardest.
    """
    totals: dict[str, int] = {}
    for record in graders.values():
        for attempt in record.get("attempts", []):
            usage = attempt.get("usage") if isinstance(attempt, Mapping) else None
            if not isinstance(usage, Mapping):
                continue
            for key, value in usage.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
    return totals


def _never_escalate(_value: BaseModel) -> str | None:
    """Escalation hook for calls that have no quality signal to route on."""
    return None


class ModelCallBudgetExhausted(RuntimeError):
    """Raised before an API request would exceed the configured hard ceiling."""


class RoutedModelClient:
    """Retry, fallback, budget and observability for one structured model call.

    Extracted from `VLMJudge` so the text-only graders in
    `rigby_poc.llm_graders` route identically (plan 10 §4). Every model call in
    the judging layer goes through `_routed_parse`, which is deliberately the
    single wrap point PR 01b instruments.
    """

    def __init__(
        self,
        *,
        client: OpenAI | None,
        model: str,
        fallback_model: str | None,
        reasoning_effort: str | None = None,
        max_model_calls: int | None = None,
    ) -> None:
        load_environment()
        self.client = client or OpenAI()
        self.model = model
        self.fallback_model = fallback_model
        self.reasoning_effort = reasoning_effort or os.getenv(
            "OPENAI_JUDGE_REASONING_EFFORT", DEFAULT_JUDGE_REASONING_EFFORT
        )
        if max_model_calls is not None and max_model_calls < 0:
            raise ValueError("max_model_calls cannot be negative")
        self.max_model_calls = max_model_calls
        self.model_calls_made = 0

    @property
    def remaining_model_calls(self) -> int | None:
        if self.max_model_calls is None:
            return None
        return max(0, self.max_model_calls - self.model_calls_made)

    def _parse_response(
        self,
        *,
        model: str,
        input: list[dict[str, Any]],
        text_format: type[BaseModel],
        image_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> object:
        if self.remaining_model_calls == 0:
            raise ModelCallBudgetExhausted(
                f"model call budget exhausted after {self.model_calls_made} calls"
            )
        # One `attempt` span per *dispatch*, not per logical attempt: a transient retry
        # is a separate HTTP request that burns budget, and plan 01 section 1.2's whole
        # complaint is that those were invisible. The span wraps the dispatch rather than
        # any caller's `except`, so a failure that propagates -- including the
        # no-fallback re-raise in `_routed_parse` -- is still recorded on the way out.
        with get_tracer().span(
            "attempt",
            f"{type(self).__name__}.dispatch",
            model=model,
        ) as span:
            span.set(
                model=model,
                reasoning_effort=self.reasoning_effort,
                text_format=text_format.__name__,
                attempt_index=self.model_calls_made,
            )
            span.record_request(input, image_metadata=image_metadata)
            # Count immediately before dispatch so failed API requests remain inside
            # the same hard ceiling as successful responses.
            self.model_calls_made += 1
            response = self.client.responses.parse(
                model=model,
                reasoning={"effort": self.reasoning_effort},
                input=input,
                text_format=text_format,
            )
            span.record_response(response)
            span.set(usage=_usage(response), response_id=getattr(response, "id", None))
            return response

    @staticmethod
    def _is_transient_dispatch_error(error: Exception) -> bool:
        """Identify provider/SDK failures that are safe to retry once."""
        return type(error).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
        }

    def _parse_response_with_retry(
        self,
        *,
        model: str,
        input: list[dict[str, Any]],
        text_format: type[BaseModel],
        image_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> object:
        transient_attempt = 0
        while True:
            try:
                return self._parse_response(
                    model=model,
                    input=input,
                    text_format=text_format,
                    image_metadata=image_metadata,
                )
            except Exception as error:
                if (
                    not self._is_transient_dispatch_error(error)
                    or self.remaining_model_calls == 0
                    or transient_attempt >= 2
                ):
                    raise
                # Retry only explicit provider connection/timeout/server
                # failures. A short bounded backoff reuses the preserved
                # request without regenerating candidates or captures.
                time.sleep(1.5 * (2**transient_attempt))
                transient_attempt += 1

    def _routed_parse(
        self,
        *,
        input: list[dict[str, Any]],
        text_format: type[BaseModel],
        escalation_reason: Callable[[BaseModel], str | None],
        primary_model: str | None = None,
        allow_fallback: bool = True,
        image_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        span_refs: Mapping[str, Any] | None = None,
    ) -> tuple[object, BaseModel, list[dict[str, Any]], dict[str, Any]]:
        """Route one structured-output call, with an optional fallback escalation.

        The `model_call` span opened here is the parent of every `attempt` span the
        dispatches below produce, so the transcript shows a routed decision and the one
        or more HTTP requests it actually cost. It parents in turn to whichever `stage`
        span is open -- `vlm_judge` inside a flywheel run -- which is what makes per-stage
        model cost attributable without correlating by position.
        """
        with get_tracer().span(
            "model_call",
            f"{type(self).__name__}.{text_format.__name__}",
            **dict(span_refs or {}),
        ) as call_span:
            return self._routed_parse_inner(
                call_span=call_span,
                input=input,
                text_format=text_format,
                escalation_reason=escalation_reason,
                primary_model=primary_model,
                allow_fallback=allow_fallback,
                image_metadata=image_metadata,
            )

    def _routed_parse_inner(
        self,
        *,
        call_span: SpanLike,
        input: list[dict[str, Any]],
        text_format: type[BaseModel],
        escalation_reason: Callable[[BaseModel], str | None],
        primary_model: str | None = None,
        allow_fallback: bool = True,
        image_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[object, BaseModel, list[dict[str, Any]], dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        reason: str | None = None
        selected_primary = primary_model or self.model
        dispatches_before = self.model_calls_made
        try:
            response = self._parse_response_with_retry(
                model=selected_primary,
                input=input,
                text_format=text_format,
                image_metadata=image_metadata,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("VLM judge returned no parsed result")
            attempts.append(_call_record(response, parsed, selected_primary))
            reason = escalation_reason(parsed)
        except Exception as error:
            if not allow_fallback or not self.fallback_model or self.fallback_model == selected_primary:
                raise
            reason = f"primary_error:{type(error).__name__}"
            response = None
            parsed = None

        escalation_requested = bool(
            allow_fallback
            and reason
            and self.fallback_model
            and self.fallback_model != selected_primary
        )
        escalation_skipped_reason: str | None = None
        escalated = escalation_requested
        if escalated and self.remaining_model_calls == 0 and response is not None and parsed is not None:
            escalated = False
            escalation_skipped_reason = "model_call_budget_exhausted"
        if escalated:
            response = self._parse_response_with_retry(
                model=self.fallback_model,
                input=input,
                text_format=text_format,
                image_metadata=image_metadata,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("fallback VLM judge returned no parsed result")
            attempts.append(_call_record(response, parsed, self.fallback_model))
        assert response is not None and parsed is not None
        routing = {
            "primary_model": selected_primary,
            "fallback_model": self.fallback_model if allow_fallback else None,
            "reasoning_effort": self.reasoning_effort,
            "escalated": escalated,
            "escalation_reason": reason if escalated else None,
            "escalation_requested_reason": reason if escalation_requested else None,
            "escalation_skipped_reason": escalation_skipped_reason,
            "selected_attempt": len(attempts) - 1,
            "attempt_count": len(attempts),
            "dispatch_count": self.model_calls_made - dispatches_before,
            "transient_retry_count": max(
                0, self.model_calls_made - dispatches_before - len(attempts)
            ),
            "model_calls_made": self.model_calls_made,
            "remaining_model_calls": self.remaining_model_calls,
        }
        call_span.set(**routing)
        return response, parsed, attempts, routing


class VLMJudge(RoutedModelClient):
    def __init__(
        self,
        client: OpenAI | None = None,
        model: str | None = None,
        *,
        fallback_model: str | None = None,
        reasoning_effort: str | None = None,
        image_detail: str | None = None,
        max_image_dimension_px: int | None = None,
        escalation_confidence: float = DEFAULT_JUDGE_ESCALATION_CONFIDENCE,
        max_model_calls: int | None = None,
        grader_mode: str | None = None,
    ) -> None:
        super().__init__(
            client=client,
            model=model or os.getenv("OPENAI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            fallback_model=(
                fallback_model
                or (model if model is not None else None)
                or os.getenv("OPENAI_JUDGE_FALLBACK_MODEL", DEFAULT_JUDGE_FALLBACK_MODEL)
            ),
            reasoning_effort=reasoning_effort,
            max_model_calls=max_model_calls,
        )
        self.image_detail = image_detail or os.getenv(
            "OPENAI_JUDGE_IMAGE_DETAIL", DEFAULT_JUDGE_IMAGE_DETAIL
        )
        configured_dimension = max_image_dimension_px or int(
            os.getenv(
                "OPENAI_JUDGE_MAX_IMAGE_DIMENSION_PX",
                str(DEFAULT_JUDGE_MAX_IMAGE_DIMENSION_PX),
            )
        )
        if configured_dimension < 512:
            raise ValueError("judge image dimension must be at least 512 pixels")
        if self.image_detail not in {"low", "high", "original"}:
            raise ValueError("judge image detail must be low, high, or original")
        self.max_image_dimension_px = configured_dimension
        self.escalation_confidence = escalation_confidence
        # The split graders stay opt-in until 07d.  Production runs on a
        # four-call budget (`pipeline.py:167`) that five graders per candidate
        # cannot fit inside, so the flag defaults to the combined path.
        resolved_mode = grader_mode or os.getenv(
            "RIGBY_JUDGE_GRADER_MODE", DEFAULT_JUDGE_GRADER_MODE
        )
        if resolved_mode not in JUDGE_GRADER_MODES:
            raise ValueError(f"judge grader mode must be one of {JUDGE_GRADER_MODES}")
        self.grader_mode = resolved_mode

    def _images(
        self,
        snapshots: list[dict[str, Any]],
        *,
        prefix: str = "",
        labels: set[str] | None = None,
        payload_audit: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return _image_content(
            snapshots,
            prefix=prefix,
            labels=labels,
            detail=self.image_detail,
            max_dimension_px=self.max_image_dimension_px,
            payload_audit=payload_audit,
        )

    def _timeline(
        self,
        snapshots: list[dict[str, Any]],
        *,
        prefix: str = "",
        payload_audit: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return _timeline_content(
            snapshots,
            prefix=prefix,
            detail=self.image_detail,
            max_dimension_px=self.max_image_dimension_px,
            payload_audit=payload_audit,
        )

    def _evidence_contract(self, payload_audit: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "source_dimensions_px": [1600, 900],
            "max_payload_dimension_px": self.max_image_dimension_px,
            "image_detail": self.image_detail,
            "full_frame_uncropped": True,
            "payloads": payload_audit,
        }

    def _unary_escalation(self, value: BaseModel) -> str | None:
        score = MotionJudgeScore.model_validate(value)
        if score.confidence < self.escalation_confidence:
            return "low_confidence"
        accepted_by_scores = meets_acceptance_thresholds(score)
        if score.accept != accepted_by_scores:
            return "acceptance_score_inconsistency"
        if score.accept:
            return "positive_confirmation"
        return None

    def _pairwise_escalation(self, value: BaseModel) -> str | None:
        decision = PairwiseJudgeDecision.model_validate(value)
        if decision.confidence < self.escalation_confidence:
            return "low_confidence"
        if decision.winner in {"tie", "neither"}:
            return "ambiguous_pair"
        return None

    def _five_way_escalation(self, value: BaseModel) -> str | None:
        decision = FiveWayJudgeDecision.model_validate(value)
        if decision.confidence < self.escalation_confidence:
            return "low_confidence"
        accepted = {candidate.label for candidate in decision.candidates if candidate.accept}
        if decision.winner == "none" and accepted:
            return "accepted_candidate_without_winner"
        if decision.winner != "none" and decision.winner not in accepted:
            return "winner_not_accepted"
        return None

    def score_combined(self, evidence_manifest: Path) -> dict[str, Any]:
        """One mega-prompt call producing a self-reported score. Pre-07b behaviour."""
        manifest, snapshots = _manifest(evidence_manifest)
        prompt = str(manifest.get("prompt", ""))
        payload_audit: list[dict[str, Any]] = []
        response, parsed, attempts, routing = self._routed_parse(
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": UNARY_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"USER MOTION REQUEST: {prompt}\nDetailed presented-pose frames and full chronological timelines follow.",
                        },
                        *self._images(
                            snapshots, labels=_key_pose_labels(manifest), payload_audit=payload_audit
                        ),
                        *self._timeline(snapshots, payload_audit=payload_audit),
                    ],
                },
            ],
            text_format=MotionJudgeScore,
            escalation_reason=self._unary_escalation,
        )
        return {
            "schema_version": "1.0",
            "kind": "unary_motion_judgment",
            "result_id": manifest.get("result_id"),
            "evidence_manifest": str(evidence_manifest),
            "evidence_manifest_sha256": hashlib.sha256(evidence_manifest.read_bytes()).hexdigest(),
            "call": _call_record(response, parsed, self.model),
            "attempts": attempts,
            "routing": routing,
            "judge_evidence_contract": self._evidence_contract(payload_audit),
        }

    def _grader_escalation(self, value: BaseModel) -> str | None:
        """Route on the claim set: unconfident, or unable to settle its questions.

        `cannot_tell` is an honest answer, not a failure — but a grader that
        cannot settle most of its claims is exactly the case where a second,
        stronger model is worth the call.
        """
        claims = getattr(value, "claims", None) or []
        if not claims:
            return "no_claims_returned"
        confidences = [float(claim.confidence) for claim in claims]
        if sum(confidences) / len(confidences) < self.escalation_confidence:
            return "low_confidence"
        unresolved = sum(claim.verdict == "cannot_tell" for claim in claims)
        if unresolved > INSUFFICIENT_EVIDENCE_FRACTION * len(claims):
            return "insufficient_evidence"
        return None

    def _grader_content(
        self,
        name: GraderName,
        *,
        manifest: dict[str, Any],
        snapshots: list[dict[str, Any]],
        payload_audit: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Assemble one grader's user payload.

        Deterministic diagnostics are never assembled here for any grader, and a
        grader whose spec says `sees_prompt=False` never receives the request
        text.  `tests/test_grader_split.py` asserts both at string level so
        leakage cannot creep back in.
        """
        spec = GRADER_SPECS[name]
        content: list[dict[str, Any]] = []
        if spec.sees_prompt:
            content.append(
                {
                    "type": "input_text",
                    "text": f"USER MOTION REQUEST: {str(manifest.get('prompt', ''))}",
                }
            )
        else:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        "The motion request is deliberately withheld. Judge the rendered "
                        "evidence on its own terms."
                    ),
                }
            )
        if spec.evidence in {"key_poses", "both"}:
            content.extend(
                self._images(
                    snapshots,
                    labels=_key_pose_labels(manifest),
                    payload_audit=payload_audit,
                )
            )
        if spec.evidence in {"timelines", "both"}:
            content.extend(self._timeline(snapshots, payload_audit=payload_audit))
        return content

    def grade(
        self,
        name: GraderName,
        *,
        manifest: dict[str, Any],
        snapshots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run one split grader and return its full call record."""
        prompt = grader_prompt(name, intent=manifest.get("intent"))
        payload_audit: list[dict[str, Any]] = []
        content = self._grader_content(
            name, manifest=manifest, snapshots=snapshots, payload_audit=payload_audit
        )
        response, parsed, attempts, routing = self._routed_parse(
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": prompt.text}]},
                {"role": "user", "content": content},
            ],
            text_format=grader_output_model(
                name, manifest.get("intent"), _citable_snapshot_ids(payload_audit)
            ),
            escalation_reason=self._grader_escalation,
        )
        return {
            "grader": name,
            "prompt": prompt.record(),
            "blinding": {
                "sees_prompt": GRADER_SPECS[name].sees_prompt,
                "sees_diagnostics": GRADER_SPECS[name].sees_diagnostics,
                "evidence": GRADER_SPECS[name].evidence,
            },
            "call": _call_record(response, parsed, self.model),
            "attempts": attempts,
            "routing": routing,
            "judge_evidence_contract": self._evidence_contract(payload_audit),
        }

    def score_split(self, evidence_manifest: Path) -> dict[str, Any]:
        """Run the five single-purpose graders and derive one score from them.

        Costs five model calls where `score_combined` costs one, which is why it
        is calibration-only until 07d (plan 07 §6.1).
        """
        manifest, snapshots = _manifest(evidence_manifest)
        graders = {
            name: self.grade(name, manifest=manifest, snapshots=snapshots)
            for name in GRADER_NAMES
        }
        intent = manifest.get("intent")
        parts = {name: graders[name]["call"]["parsed"] for name in GRADER_NAMES}
        aggregated, verdicts = assemble_split_score(parts, intent=intent)
        invented = {
            name: unexpected_claim_ids(name, parts[name], intent=intent)
            for name in GRADER_NAMES
        }
        return {
            "schema_version": "1.2",
            "kind": "split_motion_judgment",
            "grader_mode": "split",
            "family": family_for_intent(intent),
            "dimension_verdicts": {
                dimension: verdict.record() for dimension, verdict in verdicts.items()
            },
            "insufficient_evidence_dimensions": sorted(
                dimension for dimension, verdict in verdicts.items() if verdict.score is None
            ),
            "unexpected_claim_ids": {
                name: ids for name, ids in invented.items() if ids
            },
            "result_id": manifest.get("result_id"),
            "evidence_manifest": str(evidence_manifest),
            "evidence_manifest_sha256": hashlib.sha256(evidence_manifest.read_bytes()).hexdigest(),
            "graders": graders,
            "prompt_versions": {
                name: graders[name]["prompt"] for name in GRADER_NAMES
            },
            # Materialized in the combined path's shape so every existing consumer
            # of `record["call"]["parsed"]["accept"]` keeps working unchanged.
            "call": {
                # No single response id: five calls produced this score. The
                # per-grader ids are under `graders`. `usage` is summed across
                # the attempts that carried one -- see `grader_cost` for why that
                # is not the same as the cost of every dispatch.
                "response_id": None,
                "model": self.model,
                "usage": _summed_usage(graders),
                "parsed": aggregated.model_dump(mode="json"),
            },
            "decision": None,
            "attempts": [
                attempt
                for name in GRADER_NAMES
                for attempt in graders[name]["attempts"]
            ],
            "routing": {
                "grader_mode": "split",
                "graders": {name: graders[name]["routing"] for name in GRADER_NAMES},
                "model_calls_made": self.model_calls_made,
                "remaining_model_calls": self.remaining_model_calls,
            },
        }

    def score(self, evidence_manifest: Path) -> dict[str, Any]:
        """Dispatch on `grader_mode`. Combined is the default until 07d."""
        if self.grader_mode == "split":
            return self.score_split(evidence_manifest)
        return self.score_combined(evidence_manifest)

    def compare(
        self,
        first_manifest: Path,
        second_manifest: Path,
        *,
        random_seed: int,
        reverse_check: bool = False,
        routing_policy: Literal[
            "authoritative", "lightweight_routed", "lightweight_only"
        ] = "authoritative",
    ) -> dict[str, Any]:
        if routing_policy not in {
            "authoritative",
            "lightweight_routed",
            "lightweight_only",
        }:
            raise ValueError(f"unsupported pairwise routing policy: {routing_policy}")
        first, first_snapshots = _manifest(first_manifest)
        second, second_snapshots = _manifest(second_manifest)
        if first.get("prompt") != second.get("prompt"):
            raise ValueError("pairwise evidence must use the same prompt")
        candidates = [
            ("first", first, first_snapshots),
            ("second", second, second_snapshots),
        ]
        random.Random(random_seed).shuffle(candidates)
        calls: list[dict[str, Any]] = []
        orders = [candidates]
        if reverse_check:
            orders.append(list(reversed(candidates)))
        for order in orders:
            payload_audit: list[dict[str, Any]] = []
            left_key, _, left_snapshots = order[0]
            right_key, _, right_snapshots = order[1]
            content: list[dict[str, Any]] = [
                {
                    "type": "input_text",
                    "text": f"USER MOTION REQUEST: {first.get('prompt')}\nCandidate A frames, then candidate B frames:",
                },
                *self._images(
                    left_snapshots, prefix="A-", labels=_key_pose_labels(order[0][1]), payload_audit=payload_audit
                ),
                *self._images(
                    right_snapshots, prefix="B-", labels=_key_pose_labels(order[1][1]), payload_audit=payload_audit
                ),
                *self._timeline(left_snapshots, prefix="A-", payload_audit=payload_audit),
                *self._timeline(right_snapshots, prefix="B-", payload_audit=payload_audit),
            ]
            response, parsed, attempts, routing = self._routed_parse(
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": PAIRWISE_SYSTEM_PROMPT}]},
                    {"role": "user", "content": content},
                ],
                text_format=PairwiseJudgeDecision,
                escalation_reason=self._pairwise_escalation,
                primary_model=(
                    self.model
                    if routing_policy in {"lightweight_routed", "lightweight_only"}
                    else self.fallback_model
                ),
                allow_fallback=routing_policy == "lightweight_routed",
            )
            mapped_winner = {
                "A": left_key,
                "B": right_key,
                "tie": "tie",
                "neither": "neither",
            }[parsed.winner]
            calls.append(
                {
                    "order": {"A": left_key, "B": right_key},
                    "mapped_winner": mapped_winner,
                    **_call_record(response, parsed, self.model),
                    "attempts": attempts,
                    "routing": routing,
                    "judge_evidence_contract": self._evidence_contract(payload_audit),
                }
            )
        return {
            "schema_version": "1.0",
            "kind": "pairwise_motion_judgment",
            "random_seed": random_seed,
            "routing_policy": routing_policy,
            "first_result_id": first.get("result_id"),
            "second_result_id": second.get("result_id"),
            "calls": calls,
            "order_consistent": len(calls) == 1 or calls[0]["mapped_winner"] == calls[1]["mapped_winner"],
        }

    def recommend_repair(
        self,
        *,
        prompt: str,
        motion_profile: dict[str, Any],
        parameters: dict[str, Any],
        judgment: dict[str, Any],
        structural_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        repair_model = os.getenv("OPENAI_JUDGE_REPAIR_MODEL", self.model)
        visible_metrics = {
            key: structural_metrics.get(key)
            for key in (
                "structural_valid",
                "structural_failures",
                "max_wrist_swing_rad",
                "max_wrist_twist_rad",
                "max_forearm_twist_rad",
                "self_collision_frames",
                "active_hand_visibility_fraction",
                "intra_hand_contact_expected_order",
                "intra_hand_contact_observed_order",
                "intra_hand_contact_count",
                "intra_hand_contact_records",
                "intra_hand_minimum_release_separation_m",
                "gaze_max_endpoint_angle_deg",
                "max_angular_velocity_rad_s",
                "max_angular_acceleration_rad_s2",
                "max_angular_jerk_rad_s3",
                "requested_dance_beats",
                "measured_dance_beats",
                "dance_alternating_lift_count",
                "dance_lateral_root_range_m",
                "dance_left_foot_peak_clearance_m",
                "dance_right_foot_peak_clearance_m",
                "requested_climb_height_m",
                "measured_climb_height_m",
                "climb_vertical_completion_fraction",
                "requested_climb_cycles",
                "climb_support_target_max_error_m",
                "climb_three_point_support_fraction",
                "climb_final_supported_limb_count",
                "climb_missing_support_object_count",
                "strike_wrist_path_length_m",
                "strike_lateral_excursion_m",
                "strike_forward_excursion_m",
                "impact_elbow_angle_deg",
                "requested_body_rotation_degrees",
                "measured_body_rotation_degrees",
                "minimum_body_rotation_completion_fraction",
                "minimum_rotation_travel_completion_fraction",
                "floor_roll_nonfoot_contact_fraction",
                "cartwheel_hand_contact_frame_count",
                "cartwheel_inverted_frame_count",
                "cartwheel_max_foot_clearance_m",
                "cartwheel_minimum_head_clearance_m",
                "airborne_rotation_airborne_frame_count",
                "obstacle_missing_target_count",
                "minimum_obstacle_step_foot_clearance_m",
                "maximum_obstacle_step_crossing_error_m",
                "minimum_obstacle_avoidance_root_clearance_m",
            )
        }
        response, parsed, attempts, routing = self._routed_parse(
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": REPAIR_SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "user_motion_request": prompt,
                                    "motion_profile": motion_profile,
                                    "current_present_parameters": parameters,
                                    "visual_judgment": judgment,
                                    "structural_metrics": visible_metrics,
                                },
                                ensure_ascii=False,
                            ),
                        }
                    ],
                },
            ],
            text_format=RepairPatch,
            # The repair patch is bounded by its own schema, so there is no
            # quality signal worth proactively escalating on. Routing it here
            # is for retry, fallback, and observability parity only.
            escalation_reason=_never_escalate,
            primary_model=repair_model,
            # Section 1.3: this record carried no identifier at all -- no round, no result
            # id, no routing block -- so a repair could not be located from anything else
            # in the run. `routing` now travels in the returned record as well.
            span_refs={"result_id": judgment.get("result_id"), "repair_model": repair_model},
        )
        return {
            "schema_version": "1.0",
            "kind": "bounded_motion_repair",
            "call": _call_record(response, parsed, repair_model),
            "attempts": attempts,
            "routing": routing,
        }

    def rank_five(self, evidence_manifests: list[Path], *, random_seed: int) -> dict[str, Any]:
        if len(evidence_manifests) != 5:
            raise ValueError("five-way judgment requires exactly five evidence manifests")
        loaded = [(*_manifest(path), path) for path in evidence_manifests]
        prompts = {str(manifest.get("prompt", "")) for manifest, _, _ in loaded}
        if len(prompts) != 1:
            raise ValueError("five-way evidence must use one shared prompt")
        randomized = list(enumerate(loaded))
        random.Random(random_seed).shuffle(randomized)
        labels = ("A", "B", "C", "D", "E")
        payload_audit: list[dict[str, Any]] = []
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": f"USER MOTION REQUEST: {next(iter(prompts))}\nCandidates A-E follow:",
            }
        ]
        label_map: dict[str, int] = {}
        for label, (original_index, (candidate_manifest, snapshots, _)) in zip(labels, randomized):
            label_map[label] = original_index
            content.append({"type": "input_text", "text": f"CANDIDATE {label}"})
            content.extend(
                self._images(
                    snapshots,
                    prefix=f"{label}-",
                    labels=_key_pose_labels(candidate_manifest),
                    payload_audit=payload_audit,
                )
            )
            content.extend(
                self._timeline(snapshots, prefix=f"{label}-", payload_audit=payload_audit)
            )
        response, parsed, attempts, routing = self._routed_parse(
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": FIVE_WAY_SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ],
            text_format=FiveWayJudgeDecision,
            escalation_reason=self._five_way_escalation,
            # Five-way selection is the normal production path, so begin on
            # the lightweight visual model and reserve the larger model for
            # genuinely uncertain rankings.
            primary_model=self.model,
            allow_fallback=True,
            span_refs={
                # Section 1.3: `order` and `mapped_scores` are keyed by list position and
                # the position-to-candidate map lived only in the caller. Recording the
                # ids and the blinding seed makes the ranking interpretable on its own.
                "result_ids": [str(manifest.get("result_id", "")) for manifest, _, _ in loaded],
                "evidence_manifests": [str(path) for _, _, path in loaded],
                "label_order": [labels[index] for index, _ in enumerate(randomized)],
                "randomized_source_indices": [index for index, _ in randomized],
                "random_seed": random_seed,
            },
        )
        returned_labels = [item.label for item in parsed.candidates]
        if len(set(returned_labels)) != 5 or set(returned_labels) != set(labels):
            raise ValueError("five-way judge did not score every candidate exactly once")
        mapped_winner = None if parsed.winner == "none" else label_map[parsed.winner]
        mapped_scores = {
            label_map[item.label]: item.model_dump(mode="json") for item in parsed.candidates
        }
        return {
            "schema_version": "1.0",
            "kind": "five_way_motion_judgment",
            "random_seed": random_seed,
            "order": label_map,
            "mapped_winner_index": mapped_winner,
            "mapped_scores": mapped_scores,
            "call": _call_record(response, parsed, self.model),
            "attempts": attempts,
            "routing": routing,
            "judge_evidence_contract": self._evidence_contract(payload_audit),
        }


def write_judge_record(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
