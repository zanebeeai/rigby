"""Bounded end-to-end flywheel orchestration.

The pipeline deliberately depends on small protocols: learned proposal, VLM, and
embedding implementations remain replaceable, while deterministic certification
and release isolation stay authoritative.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rigby_v2.flywheel.schemas import (
    BestOfFiveSelectionV1,
    CandidateEvidenceV1,
    CandidateProposalV1,
    CandidateSetV1,
    RetrievalIndex,
    SelectionOutcome,
    SemanticPlanV1,
)
from rigby_v2.hashing import content_hash, validate_sha256
from rigby_v2.library import (
    AnimationRecord,
    AnimationRecordInput,
    CertifiedLibraryService,
    HardNegativeRecord,
    PromotionProof,
)
from rigby_v2.selection import BestOfFiveOrchestrator, CandidateCompiler, generate_candidate_set


@dataclass(frozen=True)
class CandidateExecution:
    """Evidence and independent deterministic verdict for one proposal."""

    candidate_id: str
    evidence: CandidateEvidenceV1
    certified: bool
    certification_id: str
    certification_sha256: str
    resimulation_result_sha256: str
    certifier_id: str
    repeat_count: int
    replay_exact: bool
    failure_code: str | None = None
    failed_predicate: str | None = None

    def __post_init__(self) -> None:
        if self.evidence.candidate_id != self.candidate_id:
            raise ValueError("execution evidence must bind the candidate")
        validate_sha256(self.certification_sha256)
        validate_sha256(self.resimulation_result_sha256)
        if not self.certification_id.strip() or not self.certifier_id.strip():
            raise ValueError("certification and certifier IDs are required")
        if self.certified and (self.repeat_count < 3 or not self.replay_exact):
            raise ValueError("certification requires three repeats and exact replay")
        if not self.certified and not self.failure_code:
            raise ValueError("a rejected execution requires a typed failure code")

    def gated_evidence(self) -> CandidateEvidenceV1:
        metrics = dict(self.evidence.metrics)
        metrics["deterministic_gates_passed"] = 1.0 if self.certified else 0.0
        metrics["repeat_count"] = float(self.repeat_count)
        metrics["exact_replay"] = 1.0 if self.replay_exact else 0.0
        return self.evidence.model_copy(update={"metrics": metrics})


@runtime_checkable
class CandidateExecutor(Protocol):
    def execute(self, proposal: CandidateProposalV1) -> CandidateExecution: ...


@dataclass(frozen=True)
class SurvivorValidation:
    passed: bool
    variation_count: int
    validation_sha256: str
    failure_code: str | None = None

    def __post_init__(self) -> None:
        validate_sha256(self.validation_sha256)
        if self.variation_count < 5:
            raise ValueError("survivor validation requires all calibrated variations")
        if not self.passed and not self.failure_code:
            raise ValueError("failed survivor validation requires a typed failure")


RecordFactory = Callable[
    [CandidateProposalV1, CandidateExecution, BestOfFiveSelectionV1],
    AnimationRecordInput,
]
IndexPayloadFactory = Callable[
    [CandidateProposalV1, CandidateExecution], Mapping[RetrievalIndex, object]
]


@dataclass(frozen=True)
class FlywheelRunResult:
    candidates: CandidateSetV1
    executions: tuple[CandidateExecution, ...]
    selection: BestOfFiveSelectionV1
    survivor_validation: SurvivorValidation | None = None
    candidate_record: AnimationRecord | None = None
    staged_record: AnimationRecord | None = None

    @property
    def promoted(self) -> bool:
        return self.staged_record is not None


class FlywheelPipeline:
    """Run exactly five proposals and stage only an independently proven winner."""

    def __init__(
        self,
        *,
        executor: CandidateExecutor,
        selector: BestOfFiveOrchestrator,
        library: CertifiedLibraryService | None = None,
    ) -> None:
        self.executor = executor
        self.selector = selector
        self.library = library

    def run(
        self,
        plan: SemanticPlanV1,
        *,
        compiler: CandidateCompiler | None = None,
        target_release_id: str | None = None,
        record_factory: RecordFactory | None = None,
        index_payload_factory: IndexPayloadFactory | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> FlywheelRunResult:
        resolved_compiler = compiler
        if resolved_compiler is None:
            resolved_compiler = getattr(self.executor, "candidate_compiler", None)
        candidates = generate_candidate_set(plan, resolved_compiler)
        execute_many = getattr(self.executor, "execute_many", None)
        if callable(execute_many):
            executions = tuple(
                execute_many(candidates.candidates, cancel_check=cancel_check)
                if cancel_check is not None
                else execute_many(candidates.candidates)
            )
        else:
            executions_list: list[CandidateExecution] = []
            for item in candidates.candidates:
                if cancel_check is not None and cancel_check():
                    from rigby_v2.errors import FailureCode, RigbyV2Error

                    raise RigbyV2Error(FailureCode.CANCELLED, "candidate execution cancelled")
                executions_list.append(self.executor.execute(item))
            executions = tuple(executions_list)
        by_candidate = {item.candidate_id: item for item in executions}
        expected_ids = {item.candidate_id for item in candidates.candidates}
        if len(executions) != 5 or set(by_candidate) != expected_ids:
            raise ValueError("executor must return exactly one bound result per candidate")

        selection = self.selector.select(
            candidates,
            tuple(by_candidate[item.candidate_id].gated_evidence() for item in candidates.candidates),
        )
        for execution in executions:
            if not execution.certified and self.library is not None:
                self.library.record_failure(
                    HardNegativeRecord(
                        failure_id="failure-" + content_hash(
                            {
                                "plan": plan.content_hash(),
                                "candidate": execution.candidate_id,
                                "certification": execution.certification_sha256,
                            }
                        )[:24],
                        schema_version="1.0",
                        stage="deterministic_certification",
                        failure_code=execution.failure_code or "unknown",
                        failed_predicate=execution.failed_predicate,
                        measurements=dict(execution.evidence.metrics),
                        artifacts={"trace_sha256": execution.evidence.trace.sha256},
                        provenance={"semantic_plan_hash": plan.content_hash()},
                    )
                )

        if selection.outcome is not SelectionOutcome.SELECTED:
            return FlywheelRunResult(candidates, executions, selection)

        selected_id = selection.selected_candidate_id
        if selected_id is None:
            raise RuntimeError("selected outcome omitted its candidate")
        selected_execution = by_candidate[selected_id]
        if not selected_execution.certified:
            raise RuntimeError("judge selected a deterministically rejected candidate")
        proposal = next(item for item in candidates.candidates if item.candidate_id == selected_id)
        survivor_validator = getattr(self.executor, "validate_survivor", None)
        survivor_validation = (
            survivor_validator(proposal, selected_execution)
            if callable(survivor_validator)
            else None
        )
        if self.library is None:
            return FlywheelRunResult(
                candidates,
                executions,
                selection,
                survivor_validation=survivor_validation,
            )
        if not target_release_id or record_factory is None or index_payload_factory is None:
            raise ValueError("library promotion requires target release, record, and index payloads")
        if survivor_validation is None:
            raise ValueError("library promotion requires calibrated survivor validation")
        if not survivor_validation.passed:
            self.library.record_failure(
                HardNegativeRecord(
                    failure_id="failure-" + survivor_validation.validation_sha256[:24],
                    schema_version="1.0",
                    stage="survivor_robustness",
                    failure_code=survivor_validation.failure_code or "robustness_failed",
                    failed_predicate=None,
                    measurements={
                        "variation_count": survivor_validation.variation_count,
                    },
                    artifacts={
                        "validation_sha256": survivor_validation.validation_sha256,
                    },
                    provenance={"candidate_id": selected_id},
                )
            )
            return FlywheelRunResult(
                candidates,
                executions,
                selection,
                survivor_validation=survivor_validation,
            )
        judge_ids = {judge_pass.judge_id for judge_pass in selection.judge_passes}
        if selected_execution.certifier_id in judge_ids:
            raise ValueError("certifier and selection judge must be independent")
        candidate_record = self.library.submit_candidate(
            record_factory(proposal, selected_execution, selection)
        )
        evaluation_sha256 = content_hash(selection)
        staged_record = self.library.promote_for_next_release(
            candidate_record.record_id,
            target_release_id,
            PromotionProof(
                independently_certified=True,
                resimulated=True,
                independently_evaluated=True,
                certification_id=selected_execution.certification_id,
                resimulation_result_sha256=content_hash(
                    {
                        "baseline": selected_execution.resimulation_result_sha256,
                        "robustness": survivor_validation.validation_sha256,
                    }
                ),
                independent_evaluation_sha256=evaluation_sha256,
                evaluator_id="+".join(sorted(judge_ids)),
            ),
        )
        self.library.index_record(
            staged_record.record_id,
            index_payload_factory(proposal, selected_execution),
        )
        return FlywheelRunResult(
            candidates,
            executions,
            selection,
            survivor_validation=survivor_validation,
            candidate_record=candidate_record,
            staged_record=staged_record,
        )
