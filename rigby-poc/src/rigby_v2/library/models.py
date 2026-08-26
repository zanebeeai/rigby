"""Immutable records and release metadata for the certified animation library."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from rigby_v2.flywheel.schemas import FusedRetrievalResultV1, RetrievalIndex
from rigby_v2.hashing import content_hash, validate_sha256


class AnimationStatus(StrEnum):
    LEGACY_CANDIDATE = "legacy_candidate"
    CANDIDATE = "candidate"
    STAGED = "staged"
    CERTIFIED = "certified"
    QUARANTINED = "quarantined"
    DEPRECATED = "deprecated"


class ReleaseStatus(StrEnum):
    BUILDING = "building"
    FROZEN = "frozen"
    ACTIVE = "active"
    RETIRED = "retired"


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"
    L2 = "l2"


@dataclass(frozen=True)
class EmbeddingNamespaceConfig:
    index: RetrievalIndex
    namespace: str
    version: str
    model_name: str
    model_sha256: str
    preprocessing_version: str
    dimensions: int
    distance_metric: DistanceMetric = DistanceMetric.COSINE

    def __post_init__(self) -> None:
        if not all((self.namespace, self.version, self.model_name, self.preprocessing_version)):
            raise ValueError("embedding namespace fields cannot be blank")
        validate_sha256(self.model_sha256)
        if self.dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")


@dataclass(frozen=True)
class ReleaseRecord:
    release_id: str
    parent_release_id: str | None
    status: ReleaseStatus
    manifest_sha256: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnimationRecordInput:
    schema_version: str
    split: str
    prompt_text: str
    program: dict[str, Any]
    world: dict[str, Any]
    motion: dict[str, Any]
    evidence: dict[str, Any]
    labels: dict[str, Any]
    evaluation: dict[str, Any]
    provenance: dict[str, Any]
    license_id: str
    rig_id: str
    lineage_id: str
    compact_example: str
    object_affordances: tuple[str, ...] = ()
    limbs: tuple[str, ...] = ()
    contact_requirements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required = (
            self.schema_version,
            self.split,
            self.prompt_text,
            self.license_id,
            self.rig_id,
            self.lineage_id,
            self.compact_example,
        )
        if any(not value.strip() for value in required):
            raise ValueError("record identity, compatibility, and compact context are required")
        if len(self.compact_example) > 4_000:
            raise ValueError("compact examples are limited to 4000 characters")


@dataclass(frozen=True)
class AnimationRecord:
    record_id: str
    schema_version: str
    status: AnimationStatus
    release_id: str | None
    split: str
    prompt: dict[str, Any]
    program: dict[str, Any]
    world: dict[str, Any]
    motion: dict[str, Any]
    evidence: dict[str, Any]
    labels: dict[str, Any]
    evaluation: dict[str, Any]
    provenance: dict[str, Any]
    program_sha256: str
    world_sha256: str
    motion_sha256: str
    parent_record_id: str | None = None

    @property
    def compact_example(self) -> str:
        return str(self.labels["compact_example"])

    @classmethod
    def from_input(
        cls,
        record_id: str,
        source: AnimationRecordInput,
        *,
        status: AnimationStatus,
        release_id: str | None = None,
        parent_record_id: str | None = None,
        evaluation_overlay: dict[str, Any] | None = None,
    ) -> "AnimationRecord":
        world = {
            **source.world,
            "rig_id": source.rig_id,
            "object_affordances": sorted(set(source.object_affordances)),
        }
        labels = {
            **source.labels,
            "compact_example": source.compact_example,
            "limbs": sorted(set(source.limbs)),
            "contact_requirements": sorted(set(source.contact_requirements)),
        }
        provenance = {
            **source.provenance,
            "license_id": source.license_id,
            "lineage_id": source.lineage_id,
        }
        return cls(
            record_id=record_id,
            schema_version=source.schema_version,
            status=status,
            release_id=release_id,
            split=source.split,
            prompt={"text": source.prompt_text},
            program=dict(source.program),
            world=world,
            motion=dict(source.motion),
            evidence=dict(source.evidence),
            labels=labels,
            evaluation={**source.evaluation, **(evaluation_overlay or {})},
            provenance=provenance,
            program_sha256=content_hash(source.program),
            world_sha256=content_hash(world),
            motion_sha256=content_hash(source.motion),
            parent_record_id=parent_record_id,
        )


@dataclass(frozen=True)
class PromotionProof:
    independently_certified: bool
    resimulated: bool
    independently_evaluated: bool
    certification_id: str
    resimulation_result_sha256: str
    independent_evaluation_sha256: str
    evaluator_id: str

    def validate(self) -> None:
        if not (
            self.independently_certified
            and self.resimulated
            and self.independently_evaluated
        ):
            raise ValueError(
                "promotion requires independent certification, re-simulation, and evaluation"
            )
        if not self.certification_id.strip() or not self.evaluator_id.strip():
            raise ValueError("promotion requires independent certification and evaluator IDs")
        validate_sha256(self.resimulation_result_sha256)
        validate_sha256(self.independent_evaluation_sha256)


@dataclass(frozen=True)
class EmbeddingVector:
    record_id: str
    index: RetrievalIndex
    namespace: str
    namespace_version: str
    values: tuple[float, ...]
    segment_id: str = "global"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardNegativeRecord:
    failure_id: str
    schema_version: str
    stage: str
    failure_code: str
    failed_predicate: str | None
    measurements: dict[str, Any]
    artifacts: dict[str, Any]
    provenance: dict[str, Any]
    candidate_record_id: str | None = None


@dataclass(frozen=True)
class CompactExample:
    record_id: str
    context: str


@dataclass(frozen=True)
class RetrievalResponse:
    result: FusedRetrievalResultV1
    examples: tuple[CompactExample, ...]


def staged_copy(
    source: AnimationRecord,
    *,
    record_id: str,
    release_id: str,
    proof: PromotionProof,
) -> AnimationRecord:
    proof.validate()
    return replace(
        source,
        record_id=record_id,
        status=AnimationStatus.STAGED,
        release_id=release_id,
        parent_record_id=source.record_id,
        evaluation={
            **source.evaluation,
            "independently_certified": True,
            "resimulated": True,
            "independently_evaluated": True,
            "certification_id": proof.certification_id,
            "resimulation_result_sha256": proof.resimulation_result_sha256,
            "independent_evaluation_sha256": proof.independent_evaluation_sha256,
            "evaluator_id": proof.evaluator_id,
        },
    )


def has_promotion_proof(record: AnimationRecord) -> bool:
    evaluation = record.evaluation
    try:
        validate_sha256(str(evaluation["resimulation_result_sha256"]))
        validate_sha256(str(evaluation["independent_evaluation_sha256"]))
    except (KeyError, ValueError):
        return False
    return (
        evaluation.get("independently_certified") is True
        and evaluation.get("resimulated") is True
        and evaluation.get("independently_evaluated") is True
        and bool(str(evaluation.get("certification_id", "")).strip())
        and bool(str(evaluation.get("evaluator_id", "")).strip())
    )
