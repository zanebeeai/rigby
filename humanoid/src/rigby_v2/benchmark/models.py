from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from rigby_core.contracts import Contract
from rigby_core.hashing import content_hash, validate_sha256


class BenchmarkFamily(StrEnum):
    COMMUNICATIVE_GESTURE = "communicative_gesture"
    GRASP_PLACE = "grasp_place"
    TOOL_ARTICULATED = "tool_articulated"
    BIMANUAL_MULTIPHASE = "bimanual_multiphase"


class BenchmarkCaseKind(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED_ADVERSARIAL = "unsupported_adversarial"


class ExpectedOutcome(StrEnum):
    CERTIFIED = "certified"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"
    INVALID_ASSET = "invalid_asset"


class ObservedOutcome(StrEnum):
    CERTIFIED = "certified"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"
    INVALID_ASSET = "invalid_asset"
    FAILED = "failed"


class SupportedTemplateGridV1(Contract):
    family: BenchmarkFamily
    actions: tuple[str, ...] = Field(min_length=5, max_length=5)
    entities: tuple[str, ...] = Field(min_length=5, max_length=5)
    modifiers: tuple[str, ...] = Field(min_length=3, max_length=3)


class AdversarialTemplateV1(Contract):
    category: str = Field(min_length=1)
    prompt_template: str = Field(min_length=1)
    expected_outcome: Literal[
        ExpectedOutcome.UNSUPPORTED,
        ExpectedOutcome.INFEASIBLE,
        ExpectedOutcome.INVALID_ASSET,
    ]
    tags: tuple[str, ...] = Field(min_length=1)

    @field_validator("prompt_template")
    @classmethod
    def index_placeholder(cls, value: str) -> str:
        if "{index}" not in value:
            raise ValueError("adversarial prompt template must contain {index}")
        return value


class BenchmarkSpecV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: str = Field(min_length=1)
    generator_version: Literal["rigby-benchmark-generator.v1"]
    seed: int = Field(ge=0)
    supported_per_family: Literal[75] = 75
    adversarial_count: Literal[100] = 100
    supported_grids: tuple[SupportedTemplateGridV1, ...] = Field(min_length=4, max_length=4)
    adversarial_templates: tuple[AdversarialTemplateV1, ...] = Field(
        min_length=10, max_length=10
    )
    expanded_cases_hash: str

    @field_validator("expanded_cases_hash")
    @classmethod
    def digest(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def exact_families(self) -> Self:
        if {grid.family for grid in self.supported_grids} != set(BenchmarkFamily):
            raise ValueError("benchmark spec requires each supported family exactly once")
        categories = [template.category for template in self.adversarial_templates]
        if len(categories) != len(set(categories)):
            raise ValueError("adversarial template categories must be unique")
        return self


class BenchmarkCaseV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    kind: BenchmarkCaseKind
    family: BenchmarkFamily | None = None
    prompt: str = Field(min_length=1)
    expected_outcome: ExpectedOutcome
    seed: int = Field(ge=0)
    tags: tuple[str, ...] = Field(min_length=1)
    case_hash: str = ""

    def hash_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"case_hash"})

    @model_validator(mode="after")
    def seal_and_validate(self) -> Self:
        if self.kind is BenchmarkCaseKind.SUPPORTED:
            if self.family is None or self.expected_outcome is not ExpectedOutcome.CERTIFIED:
                raise ValueError("supported cases require a family and certified expected outcome")
        elif self.family is not None or self.expected_outcome is ExpectedOutcome.CERTIFIED:
            raise ValueError("adversarial cases cannot declare supported certification")
        expected = content_hash(self.hash_payload())
        if self.case_hash and self.case_hash != expected:
            raise ValueError("benchmark case hash does not match its immutable content")
        object.__setattr__(self, "case_hash", expected)
        return self


class BenchmarkManifestV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    benchmark_id: str = Field(min_length=1)
    generator_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    spec_artifact_hash: str
    expanded_cases_hash: str
    cases: tuple[BenchmarkCaseV1, ...] = Field(min_length=400, max_length=400)

    @field_validator("spec_artifact_hash", "expanded_cases_hash")
    @classmethod
    def digest(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def exact_release_matrix(self) -> Self:
        identifiers = [case.case_id for case in self.cases]
        hashes = [case.case_hash for case in self.cases]
        if len(identifiers) != len(set(identifiers)) or len(hashes) != len(set(hashes)):
            raise ValueError("benchmark case IDs and hashes must be unique")
        supported = [case for case in self.cases if case.kind is BenchmarkCaseKind.SUPPORTED]
        adversarial = [
            case for case in self.cases if case.kind is BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL
        ]
        if len(supported) != 300 or len(adversarial) != 100:
            raise ValueError("benchmark requires exactly 300 supported and 100 adversarial cases")
        for family in BenchmarkFamily:
            if sum(case.family is family for case in supported) != 75:
                raise ValueError(f"benchmark family {family.value} must contain exactly 75 cases")
        expected = content_hash([case.model_dump(mode="json") for case in self.cases])
        if expected != self.expanded_cases_hash:
            raise ValueError("expanded benchmark case hash does not match the sealed spec")
        return self


class SupportedCaseResultV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["supported"] = "supported"
    case_id: str = Field(min_length=1)
    case_hash: str
    observed_outcome: ObservedOutcome
    max_fingertip_error_m: float | None = Field(default=None, ge=0.0)
    max_penetration_m: float | None = Field(default=None, ge=0.0)
    repeat_task_outcomes: tuple[str, ...] = ()
    vlm_human_agrees: bool
    judge_order_consistent: bool
    critical_judge_false_accept: bool = False
    retrieval_relevant_top3: int = Field(ge=0, le=3)
    retrieval_relevant_top5: int = Field(ge=0, le=5)
    retrieval_total_relevant: int = Field(gt=0)
    rag_off_success: bool
    rag_on_success: bool
    local_pipeline_seconds: float = Field(ge=0.0)
    candidate_count: Literal[5] = 5
    candidate_duration_s: Literal[10.0] = 10.0
    external_vlm_time_excluded: Literal[True] = True
    result_hash: str = ""

    @field_validator("case_hash")
    @classmethod
    def digest(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def seal_and_validate(self) -> Self:
        if self.retrieval_relevant_top5 < self.retrieval_relevant_top3:
            raise ValueError("top-5 relevant count cannot be below top-3")
        if self.retrieval_relevant_top5 > self.retrieval_total_relevant:
            raise ValueError("retrieved relevant count cannot exceed total relevant records")
        payload = self.model_dump(mode="json", exclude={"result_hash"})
        expected = content_hash(payload)
        if self.result_hash and self.result_hash != expected:
            raise ValueError("supported result hash does not match immutable content")
        object.__setattr__(self, "result_hash", expected)
        return self


class AdversarialCaseResultV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["adversarial"] = "adversarial"
    case_id: str = Field(min_length=1)
    case_hash: str
    observed_outcome: ObservedOutcome
    critical_judge_false_accept: bool = False
    result_hash: str = ""

    @field_validator("case_hash")
    @classmethod
    def digest(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def seal_and_validate(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"result_hash"})
        expected = content_hash(payload)
        if self.result_hash and self.result_hash != expected:
            raise ValueError("adversarial result hash does not match immutable content")
        object.__setattr__(self, "result_hash", expected)
        return self


BenchmarkCaseResultV1 = SupportedCaseResultV1 | AdversarialCaseResultV1


class BenchmarkRunV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    manifest_hash: str
    environment_hash: str
    completed_at: datetime
    results: tuple[BenchmarkCaseResultV1, ...] = Field(min_length=400, max_length=400)
    result_set_hash: str = ""

    @field_validator("manifest_hash", "environment_hash")
    @classmethod
    def digest(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("completed_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("benchmark completion time must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def seal_results(self) -> Self:
        identifiers = [result.case_id for result in self.results]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("benchmark run must contain one result per case")
        expected = content_hash([result.model_dump(mode="json") for result in self.results])
        if self.result_set_hash and self.result_set_hash != expected:
            raise ValueError("benchmark result-set hash does not match immutable results")
        object.__setattr__(self, "result_set_hash", expected)
        return self


class BenchmarkMetricsV1(Contract):
    supported_count: int
    adversarial_count: int
    critical_deterministic_false_accepts: int
    adversarial_typed_outcome_rate: float
    supported_certified_winner_rate: float
    max_fingertip_error_m: float
    max_penetration_m: float
    repeat_task_outcome_identical_rate: float
    vlm_human_agreement_rate: float
    judge_order_consistency_rate: float
    critical_judge_false_accept_rate: float
    retrieval_precision_at_3: float
    retrieval_recall_at_5: float
    rag_success_lift: float
    rag_lift_ci95_low: float
    rag_lift_ci95_high: float
    local_pipeline_p95_seconds: float
    missing_physics_measurements: int


class ReleaseGateCheckV1(Contract):
    name: str = Field(min_length=1)
    passed: bool
    measured: float | int | bool | str
    requirement: str = Field(min_length=1)


class ReleaseGateReportV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    release_allowed: bool
    manifest_hash: str
    result_set_hash: str
    metrics: BenchmarkMetricsV1
    checks: tuple[ReleaseGateCheckV1, ...] = Field(min_length=1)
    bootstrap_seed: int
    bootstrap_samples: int = Field(ge=1_000)
    report_hash: str = ""

    @field_validator("manifest_hash", "result_set_hash")
    @classmethod
    def digest(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def seal_report(self) -> Self:
        if self.release_allowed != all(check.passed for check in self.checks):
            raise ValueError("release decision must equal the conjunction of all hard gates")
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        expected = content_hash(payload)
        if self.report_hash and self.report_hash != expected:
            raise ValueError("release report hash does not match immutable content")
        object.__setattr__(self, "report_hash", expected)
        return self
