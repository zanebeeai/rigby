from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, field_validator

from .contracts import ArtifactRefV1, Contract, SimulationJobV1, SimulationResultV1, utc_now
from .errors import FailureCode
from .hashing import validate_sha256


class CertificationStatus(StrEnum):
    LEGACY_CANDIDATE = "legacy_candidate"
    STAGED = "staged"
    CERTIFIED = "certified"
    REJECTED = "rejected"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    BENCHMARK = "benchmark"


class LicenseRecordV1(Contract):
    identifier: str = Field(min_length=1)
    research_only: bool = False
    attribution: str | None = None
    source_url: str | None = None


class LineageRefV1(Contract):
    relation: Literal["derived_from", "repair_of", "variant_of", "imported_from"]
    record_hash: str

    @field_validator("record_hash")
    @classmethod
    def valid_record_hash(cls, value: str) -> str:
        return validate_sha256(value)


class AnimationRecordV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    record_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    prompt: str = Field(min_length=1)
    program_hash: str
    scene_hash: str
    rig_hash: str
    result_hash: str
    status: CertificationStatus
    release: str = Field(min_length=1)
    staged_for_release: str | None = None
    split: DatasetSplit
    licenses: tuple[LicenseRecordV1, ...] = ()
    artifacts: tuple[ArtifactRefV1, ...] = Field(min_length=1)
    retrieval_namespaces: tuple[str, ...] = ()
    embedding_models: dict[str, str] = Field(default_factory=dict)
    evaluation_version: str = Field(min_length=1)
    evidence_hashes: tuple[str, ...] = ()
    lineage: tuple[LineageRefV1, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("program_hash", "scene_hash", "rig_hash", "result_hash")
    @classmethod
    def valid_required_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("evidence_hashes")
    @classmethod
    def valid_evidence_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_sha256(value) for value in values)

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)


class FailureRecordV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    failure_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    job_id: str | None = None
    code: FailureCode
    message: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    artifacts: tuple[ArtifactRefV1, ...] = ()
    lineage: tuple[LineageRefV1, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)


class JobState(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRecord(Contract):
    job: SimulationJobV1
    state: JobState
    priority: int = 0
    attempts: int = Field(default=0, ge=0)
    available_at: datetime
    created_at: datetime
    updated_at: datetime
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    cancel_requested: bool = False
    result: SimulationResultV1 | None = None
    failure: FailureRecordV1 | None = None
    idempotency_key: str | None = None

    @field_validator("available_at", "created_at", "updated_at", "lease_expires_at", "heartbeat_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("Job timestamps must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None
