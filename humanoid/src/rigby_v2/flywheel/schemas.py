"""Frozen interfaces shared by evidence, retrieval, and calibration subsystems."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from rigby_core.contracts import ArtifactRefV1, Contract, MotionProgramV2
from rigby_core.hashing import validate_sha256


class PathFamily(StrEnum):
    DIRECT = "direct"
    ARC = "arc"
    BODY_LED = "body_led"
    HAND_LED = "hand_led"
    CONTACT_FIRST = "contact_first"


class CandidateVariationV1(Contract):
    retrieval_seed: int = Field(ge=0)
    path_family: PathFamily
    timing_style: Literal["relaxed", "neutral", "energetic"]
    energy: float = Field(ge=0.0, le=1.0)
    body_participation: tuple[str, ...] = Field(min_length=1)
    contact_strategy: str = Field(min_length=1)


class SemanticPlanV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    plan_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    base_program: MotionProgramV2
    retrieval_release: str = Field(min_length=1)
    retrieval_examples: tuple[str, ...] = Field(default=(), max_length=3)

    @field_validator("retrieval_examples")
    @classmethod
    def hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_sha256(value) for value in values)


class CandidateProposalV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    candidate_id: str = Field(min_length=1)
    semantic_plan_hash: str
    variation: CandidateVariationV1
    program: MotionProgramV2
    compile_attempt: int = Field(ge=1, le=20)
    repaired: bool = False

    @field_validator("semantic_plan_hash")
    @classmethod
    def plan_hash(cls, value: str) -> str:
        return validate_sha256(value)


class CandidateSetV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    semantic_plan: SemanticPlanV1
    candidates: tuple[CandidateProposalV1, ...] = Field(min_length=5, max_length=5)
    total_compile_attempts: int = Field(ge=5, le=20)
    repair_rounds: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def exactly_five_diverse_candidates(self) -> Self:
        ids = [item.candidate_id for item in self.candidates]
        variations = [item.variation.content_hash() for item in self.candidates]
        if len(ids) != len(set(ids)) or len(variations) != len(set(variations)):
            raise ValueError("Candidate set requires five nonduplicate candidates")
        plan_hash = self.semantic_plan.content_hash()
        if any(item.semantic_plan_hash != plan_hash for item in self.candidates):
            raise ValueError("Every candidate must bind the one semantic plan")
        return self


class EvidenceCameraRole(StrEnum):
    ORBIT = "orbit"
    EGOCENTRIC = "egocentric"
    TASK_CLOSEUP = "task_closeup"


class CameraEvidenceV1(Contract):
    camera_id: str = Field(min_length=1)
    role: EvidenceCameraRole
    fps: Literal[30] = 30
    raw_frames: ArtifactRefV1
    video: ArtifactRefV1
    timestamps_s: tuple[float, ...] = Field(min_length=2)
    world_from_camera: tuple[tuple[float, ...], ...] = Field(min_length=2)
    intrinsics: tuple[float, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def aligned_frames(self) -> Self:
        if len(self.timestamps_s) != len(self.world_from_camera):
            raise ValueError("Camera timestamps and matrices must align")
        if any(len(matrix) != 16 for matrix in self.world_from_camera):
            raise ValueError("Camera matrices must be flattened 4x4 transforms")
        if any(
            right <= left
            for left, right in zip(self.timestamps_s, self.timestamps_s[1:])
        ):
            raise ValueError("Evidence timestamps must be strictly increasing")
        return self


class CandidateEvidenceV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    anonymous_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    trace: ArtifactRefV1
    cameras: tuple[CameraEvidenceV1, ...] = Field(min_length=3)
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def required_views(self) -> Self:
        roles = {camera.role for camera in self.cameras}
        required = {
            EvidenceCameraRole.ORBIT,
            EvidenceCameraRole.EGOCENTRIC,
            EvidenceCameraRole.TASK_CLOSEUP,
        }
        if not required <= roles:
            raise ValueError("Evidence requires orbit, egocentric, and close-up views")
        return self


class RubricScoresV1(Contract):
    semantic_fidelity: float = Field(ge=0.0, le=5.0)
    physical_plausibility: float = Field(ge=0.0, le=5.0)
    contact_quality: float = Field(ge=0.0, le=5.0)
    timing_energy: float = Field(ge=0.0, le=5.0)
    whole_body_quality: float = Field(ge=0.0, le=5.0)
    visual_clarity: float = Field(ge=0.0, le=5.0)

    @property
    def weakest(self) -> float:
        return min(
            self.semantic_fidelity,
            self.physical_plausibility,
            self.contact_quality,
            self.timing_energy,
            self.whole_body_quality,
            self.visual_clarity,
        )

    @property
    def aggregate(self) -> float:
        return sum(
            (
                self.semantic_fidelity,
                self.physical_plausibility,
                self.contact_quality,
                self.timing_energy,
                self.whole_body_quality,
                self.visual_clarity,
            )
        ) / 6.0


class AnonymousJudgeEvaluationV1(Contract):
    anonymous_id: str = Field(min_length=1)
    scores: RubricScoresV1
    independently_clears_rubric: bool
    uncertainty: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


class JudgePassV1(Contract):
    pass_id: str = Field(min_length=1)
    judge_id: str = Field(min_length=1)
    presentation_order: tuple[str, ...] = Field(min_length=5, max_length=5)
    evaluations: tuple[AnonymousJudgeEvaluationV1, ...] = Field(
        min_length=5, max_length=5
    )

    @model_validator(mode="after")
    def anonymous_set_matches(self) -> Self:
        if len(set(self.presentation_order)) != 5:
            raise ValueError("Judge presentation order must contain five unique IDs")
        evaluated = [item.anonymous_id for item in self.evaluations]
        if len(set(evaluated)) != 5 or set(evaluated) != set(self.presentation_order):
            raise ValueError("Judge evaluations must cover the anonymous order exactly")
        return self


class SelectionOutcome(StrEnum):
    SELECTED = "selected"
    UNCERTAIN = "uncertain"
    ALL_CANDIDATES_REJECTED = "all_candidates_rejected"


class BestOfFiveSelectionV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    outcome: SelectionOutcome
    selected_candidate_id: str | None = None
    judge_passes: tuple[JudgePassV1, ...] = Field(min_length=2, max_length=4)
    fallback_used: bool = False
    repair_calls: int = Field(default=0, ge=0, le=1)
    total_judge_and_repair_calls: int = Field(ge=2, le=4)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def outcome_matches_selection(self) -> Self:
        if (self.outcome is SelectionOutcome.SELECTED) != (
            self.selected_candidate_id is not None
        ):
            raise ValueError("Only a selected outcome may name a candidate")
        return self


class RetrievalIndex(StrEnum):
    TEXT = "text"
    PROGRAM = "structured_program"
    MOTION = "task_space_motion"
    VIDEO = "video"
    KEYFRAME = "salient_keyframe"


class RetrievalFiltersV1(Contract):
    certified_only: Literal[True] = True
    release: str = Field(min_length=1)
    allowed_licenses: tuple[str, ...] = Field(min_length=1)
    schema_versions: tuple[str, ...] = Field(min_length=1)
    rig_ids: tuple[str, ...] = Field(min_length=1)
    object_affordances: tuple[str, ...] = ()
    limbs: tuple[str, ...] = ()
    contact_requirements: tuple[str, ...] = ()
    excluded_splits: tuple[str, ...] = ("test", "benchmark")
    excluded_lineage: tuple[str, ...] = ()


class RankedRetrievalHitV1(Contract):
    record_id: str = Field(min_length=1)
    index: RetrievalIndex
    rank: int = Field(ge=1)
    distance: float = Field(ge=0.0)
    release: str = Field(min_length=1)


class FusedRetrievalResultV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    query_hash: str
    release: str = Field(min_length=1)
    hits_by_index: dict[RetrievalIndex, tuple[RankedRetrievalHitV1, ...]]
    selected_examples: tuple[str, ...] = Field(min_length=2, max_length=3)
    reciprocal_rank_k: int = Field(default=60, gt=0)

    @field_validator("query_hash")
    @classmethod
    def query_digest(cls, value: str) -> str:
        return validate_sha256(value)


class DefectKind(StrEnum):
    REALISTIC = "realistic"
    NEAR_TIE = "near_tie"
    ORDER_REVERSAL = "order_reversal"
    LEFT_RIGHT = "left_right"
    TIMING = "timing"
    MISSING_VIEW = "missing_view"
    HAND = "hand"
    CONTACT = "contact"


class CalibrationRatingV1(Contract):
    rater_id_hash: str
    presentation_order: Literal["left_right", "right_left"]
    verdict: Literal["left", "right", "tie", "abstain", "both_fail"]
    rubric: RubricScoresV1

    @field_validator("rater_id_hash")
    @classmethod
    def rater_hash(cls, value: str) -> str:
        return validate_sha256(value)


class HumanComparisonPairV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    pair_id: str = Field(min_length=1)
    action_family: str = Field(min_length=1)
    left_record_id: str = Field(min_length=1)
    right_record_id: str = Field(min_length=1)
    defects: tuple[DefectKind, ...] = Field(min_length=1)
    ratings: tuple[CalibrationRatingV1, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def three_independent_raters(self) -> Self:
        if self.left_record_id == self.right_record_id:
            raise ValueError("Calibration requires two distinct records")
        raters = {rating.rater_id_hash for rating in self.ratings}
        if len(raters) != 3:
            raise ValueError("Calibration comparison requires three independent raters")
        return self


class DatasetNamespacePolicyV1(Contract):
    namespace: str = Field(min_length=1)
    research_only: bool
    allowed_uses: tuple[str, ...] = Field(min_length=1)
    prohibited_claims: tuple[str, ...] = ()
    requires_expert_review: bool = False

