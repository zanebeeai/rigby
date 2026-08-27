"""Resumable, evidence-backed execution for the supported benchmark.

This module deliberately does not synthesize ``SupportedCaseResultV1`` rows.  A
case is complete only when every production stage ran and a completion provider
supplied the full, hash-verified benchmark evidence set.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_core.contracts import ArtifactRefV1
from rigby_v2.flywheel.schemas import (
    CandidateEvidenceV1,
    CandidateSetV1,
    SemanticPlanV1,
)
from rigby_v2.guardrails import enforce_planning_intake
from rigby_core.hashing import canonical_json_bytes
from rigby_v2.library import CertifiedLibraryService
from rigby_v2.library.embeddings import RetrievalQuery
from rigby_v2.pipeline import CandidateExecution
from rigby_v2.selection import BestOfFiveOrchestrator, generate_candidate_set

from .execution import SUPPORTED_EVIDENCE_KINDS
from .manifest import load_benchmark_manifest
from .models import (
    BenchmarkCaseKind,
    BenchmarkCaseV1,
    BenchmarkFamily,
    BenchmarkManifestV1,
    ObservedOutcome,
    SupportedCaseResultV1,
)


class SupportedLiveStage(StrEnum):
    MANIFEST_PREFLIGHT = "manifest_preflight"
    EXTERNAL_MODEL_GATE = "external_model_gate"
    RETRIEVAL = "retrieval"
    PLANNING = "planning"
    CANDIDATE_GENERATION = "candidate_generation"
    FIVE_CANDIDATE_EXECUTION = "five_candidate_execution"
    JUDGMENT = "judgment"
    COMPLETION_EVIDENCE = "completion_evidence"


PIPELINE_STAGES = tuple(SupportedLiveStage)[1:]


class SupportedLiveStatus(StrEnum):
    PENDING = "pending"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETE = "complete"


class SupportedLiveBlockCode(StrEnum):
    EXTERNAL_MODEL_UNAVAILABLE = "external_model_unavailable"
    CASE_BINDING_MISSING = "case_binding_missing"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"
    EXTERNAL_EVIDENCE_UNAVAILABLE = "external_evidence_unavailable"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    MODEL_CALL_BUDGET_EXHAUSTED = "model_call_budget_exhausted"
    ARTIFACT_INTEGRITY = "artifact_integrity"
    STAGE_FAILED = "stage_failed"


class SupportedStageBlocked(RuntimeError):
    """A typed, persisted stop that is never confused with benchmark success."""

    def __init__(
        self,
        code: SupportedLiveBlockCode,
        message: str,
        *,
        retryable: bool,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class SupportedStageOutput:
    data: Mapping[str, object]
    model_calls: int = 0

    def __post_init__(self) -> None:
        if self.model_calls < 0:
            raise ValueError("model_calls cannot be negative")


class SupportedStagePipeline(Protocol):
    pipeline_id: str

    def model_call_upper_bound(self, stage: SupportedLiveStage) -> int: ...

    def run_stage(
        self,
        case: BenchmarkCaseV1,
        stage: SupportedLiveStage,
        completed: Mapping[SupportedLiveStage, Mapping[str, object]],
    ) -> SupportedStageOutput: ...


@dataclass(frozen=True, slots=True)
class SupportedLiveRunSummary:
    run_root: Path
    selected_case_ids: tuple[str, ...]
    status_counts: Mapping[str, int]
    cases: tuple[Mapping[str, object], ...]

    @property
    def all_complete(self) -> bool:
        return self.status_counts.get(SupportedLiveStatus.COMPLETE.value, 0) == len(
            self.selected_case_ids
        )


def select_supported_cases(
    manifest: BenchmarkManifestV1,
    *,
    case_ids: Sequence[str] = (),
    families: Sequence[BenchmarkFamily | str] = (),
) -> tuple[BenchmarkCaseV1, ...]:
    supported = tuple(
        case for case in manifest.cases if case.kind is BenchmarkCaseKind.SUPPORTED
    )
    by_id = {case.case_id: case for case in supported}
    requested_ids = set(case_ids)
    unknown = sorted(requested_ids - set(by_id))
    if unknown:
        raise ValueError(f"unknown supported benchmark case IDs: {', '.join(unknown)}")
    requested_families = {BenchmarkFamily(item) for item in families}
    if not requested_ids and not requested_families:
        return supported
    return tuple(
        case
        for case in supported
        if case.case_id in requested_ids or case.family in requested_families
    )


def communicative_canary_cases(
    manifest: BenchmarkManifestV1, count: int = 3
) -> tuple[BenchmarkCaseV1, ...]:
    if not 3 <= count <= 5:
        raise ValueError("the communicative canary must contain three to five cases")
    cases = select_supported_cases(
        manifest, families=(BenchmarkFamily.COMMUNICATIVE_GESTURE,)
    )
    return cases[:count]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _reference(value: Mapping[str, object]) -> ArtifactRefV1:
    return ArtifactRefV1.model_validate(value)


class ResumableSupportedBenchmarkRunner:
    """Case-isolated runner suitable for sequential use or a process scheduler."""

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        *,
        run_root: Path,
        manifest: BenchmarkManifestV1,
        pipeline: SupportedStagePipeline,
        selected_cases: Sequence[BenchmarkCaseV1],
        stage_attempt_limit: int = 2,
        max_model_calls_per_case: int = 5,
        resume: bool = True,
    ) -> None:
        if not pipeline.pipeline_id.strip():
            raise ValueError("pipeline_id is required")
        if stage_attempt_limit < 1 or max_model_calls_per_case < 0:
            raise ValueError("execution budgets must be nonnegative and attempts positive")
        selected_cases = tuple(selected_cases)
        if not selected_cases or any(
            case.kind is not BenchmarkCaseKind.SUPPORTED for case in selected_cases
        ):
            raise ValueError("live supported execution requires supported cases")
        if len({case.case_id for case in selected_cases}) != len(selected_cases):
            raise ValueError("selected benchmark cases must be unique")
        self.run_root = run_root.resolve()
        self.manifest = manifest
        self.pipeline = pipeline
        self.selected_cases = selected_cases
        self.stage_attempt_limit = stage_attempt_limit
        self.max_model_calls_per_case = max_model_calls_per_case
        self.artifacts = ContentAddressedArtifactStore(self.run_root / "artifacts")
        bind_artifacts = getattr(self.pipeline, "bind_artifact_store", None)
        if callable(bind_artifacts):
            bind_artifacts(self.artifacts)
        self.cases_root = self.run_root / "cases"
        self.config_path = self.run_root / "run-config.json"
        config = self._config()
        if self.config_path.exists():
            if not resume:
                raise FileExistsError(self.config_path)
            observed = json.loads(self.config_path.read_text(encoding="utf-8"))
            if observed != config:
                raise ValueError("resume configuration differs from the sealed run config")
        else:
            _atomic_json(self.config_path, config)

    def _config(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "manifest_hash": self.manifest.content_hash(),
            "expanded_cases_hash": self.manifest.expanded_cases_hash,
            "pipeline_id": self.pipeline.pipeline_id,
            "selected_case_ids": [case.case_id for case in self.selected_cases],
            "stage_attempt_limit": self.stage_attempt_limit,
            "max_model_calls_per_case": self.max_model_calls_per_case,
        }

    def _state_path(self, case: BenchmarkCaseV1) -> Path:
        return self.cases_root / f"{case.case_id}.json"

    def _initial_state(self, case: BenchmarkCaseV1) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "case_id": case.case_id,
            "case_hash": case.case_hash,
            "status": SupportedLiveStatus.PENDING.value,
            "completed_stages": {},
            "attempts": {},
            "attempt_artifacts": {},
            "model_calls_used": 0,
            "blocker": None,
        }

    def _load_state(self, case: BenchmarkCaseV1) -> dict[str, Any]:
        path = self._state_path(case)
        if not path.exists():
            return self._initial_state(case)
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("case_id") != case.case_id or state.get("case_hash") != case.case_hash:
            raise ValueError("case checkpoint does not bind the immutable benchmark case")
        self._verified_completed(state)
        for stage_name, values in state.get("attempt_artifacts", {}).items():
            for value in values:
                self._read_attempt_artifact(state, stage_name, value)
        return state

    def _read_attempt_artifact(
        self,
        state: Mapping[str, Any],
        expected_stage: str,
        value: Mapping[str, object],
    ) -> dict[str, object]:
        payload = json.loads(
            self.artifacts.read_bytes(_reference(value), verify=True).decode("utf-8")
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != self.SCHEMA_VERSION
            or payload.get("case_id") != state.get("case_id")
            or payload.get("case_hash") != state.get("case_hash")
            or payload.get("pipeline_id") != self.pipeline.pipeline_id
            or payload.get("stage") != expected_stage
            or payload.get("outcome") not in {"completed", "blocked"}
        ):
            raise ValueError("stage attempt artifact does not bind its case and pipeline")
        return payload

    def _verified_completed(
        self, state: Mapping[str, Any]
    ) -> dict[SupportedLiveStage, Mapping[str, object]]:
        completed: dict[SupportedLiveStage, Mapping[str, object]] = {}
        for stage_name, value in state.get("completed_stages", {}).items():
            stage = SupportedLiveStage(stage_name)
            payload = self._read_attempt_artifact(state, stage_name, value)
            if payload.get("stage") != stage.value or payload.get("outcome") != "completed":
                raise ValueError("completed stage artifact has invalid contents")
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("completed stage artifact data must be an object")
            if stage is SupportedLiveStage.FIVE_CANDIDATE_EXECUTION:
                self._verify_execution_evidence(data)
            completed[stage] = data
        return completed

    def _verify_execution_evidence(self, data: Mapping[str, object]) -> None:
        executions = data.get("executions")
        if not isinstance(executions, list) or len(executions) != 5:
            raise ValueError("execution stage must retain exactly five evidence bundles")
        candidate_ids: set[str] = set()
        for value in executions:
            if not isinstance(value, Mapping):
                raise ValueError("execution evidence entry must be an object")
            evidence = CandidateEvidenceV1.model_validate(value.get("evidence"))
            candidate_ids.add(evidence.candidate_id)
            self.artifacts.read_bytes(evidence.trace, verify=True)
            for camera in evidence.cameras:
                self.artifacts.read_bytes(camera.raw_frames, verify=True)
                self.artifacts.read_bytes(camera.video, verify=True)
        if len(candidate_ids) != 5:
            raise ValueError("execution evidence must cover five unique candidates")

    def _record_attempt(
        self,
        state: dict[str, Any],
        stage: SupportedLiveStage,
        payload: Mapping[str, object],
    ) -> ArtifactRefV1:
        bound_payload = {
            "schema_version": self.SCHEMA_VERSION,
            "case_id": state["case_id"],
            "case_hash": state["case_hash"],
            "pipeline_id": self.pipeline.pipeline_id,
            **payload,
        }
        reference = self.artifacts.put_json(
            bound_payload,
            filename=f"{state['case_id']}-{stage.value}-attempt.json",
        )
        state.setdefault("attempt_artifacts", {}).setdefault(stage.value, []).append(
            reference.model_dump(mode="json")
        )
        return reference

    def _block(
        self,
        state: dict[str, Any],
        stage: SupportedLiveStage,
        blocked: SupportedStageBlocked,
        attempt: int,
    ) -> None:
        blocker = {
            "stage": stage.value,
            "code": blocked.code.value,
            "message": blocked.message,
            "retryable": blocked.retryable,
            "details": blocked.details,
            "attempt": attempt,
        }
        self._record_attempt(
            state,
            stage,
            {"stage": stage.value, "outcome": "blocked", "blocker": blocker},
        )
        state["status"] = SupportedLiveStatus.BLOCKED.value
        state["blocker"] = blocker

    def _complete_stage(
        self,
        state: dict[str, Any],
        stage: SupportedLiveStage,
        output: SupportedStageOutput,
    ) -> None:
        payload = {
            "stage": stage.value,
            "outcome": "completed",
            "data": dict(output.data),
            "model_calls": output.model_calls,
        }
        reference = self._record_attempt(state, stage, payload)
        state.setdefault("completed_stages", {})[stage.value] = reference.model_dump(
            mode="json"
        )
        state["model_calls_used"] += output.model_calls
        state["blocker"] = None

    def _validate_completion(
        self, case: BenchmarkCaseV1, data: Mapping[str, object]
    ) -> None:
        try:
            result = SupportedCaseResultV1.model_validate(data["result"])
            evidence = data["evidence"]
            if not isinstance(evidence, Mapping) or set(evidence) != SUPPORTED_EVIDENCE_KINDS:
                raise ValueError("completion requires the exact supported evidence set")
            if result.case_id != case.case_id or result.case_hash != case.case_hash:
                raise ValueError("completion result does not bind the benchmark case")
            if result.observed_outcome is not ObservedOutcome.CERTIFIED:
                raise ValueError("supported completion is not certified")
            for value in evidence.values():
                if not isinstance(value, Mapping):
                    raise ValueError("completion evidence reference is invalid")
                self.artifacts.read_bytes(_reference(value), verify=True)
        except Exception as error:
            raise SupportedStageBlocked(
                SupportedLiveBlockCode.INCOMPLETE_EVIDENCE,
                "the case lacks a complete, certified, hash-verified evidence set",
                retryable=True,
                details={"validation_error": type(error).__name__},
            ) from error

    def run_case(self, case: BenchmarkCaseV1) -> Mapping[str, object]:
        if case.case_id not in {item.case_id for item in self.selected_cases}:
            raise ValueError("case is outside this sealed run selection")
        try:
            state = self._load_state(case)
        except Exception as error:
            state = self._initial_state(case)
            blocked = SupportedStageBlocked(
                SupportedLiveBlockCode.ARTIFACT_INTEGRITY,
                "checkpoint artifact integrity verification failed",
                retryable=False,
                details={"error_type": type(error).__name__},
            )
            self._block(state, SupportedLiveStage.MANIFEST_PREFLIGHT, blocked, 0)
            _atomic_json(self._state_path(case), state)
            return state
        if state["status"] == SupportedLiveStatus.COMPLETE.value:
            return state
        completed = self._verified_completed(state)
        if SupportedLiveStage.MANIFEST_PREFLIGHT not in completed:
            decision = enforce_planning_intake(case.prompt)
            output = SupportedStageOutput(
                {
                    "case": case.model_dump(mode="json"),
                    "manifest_hash": self.manifest.content_hash(),
                    "guardrail": decision.model_dump(mode="json"),
                }
            )
            state["attempts"][SupportedLiveStage.MANIFEST_PREFLIGHT.value] = 1
            self._complete_stage(state, SupportedLiveStage.MANIFEST_PREFLIGHT, output)
            completed[SupportedLiveStage.MANIFEST_PREFLIGHT] = output.data
            _atomic_json(self._state_path(case), state)

        for stage in PIPELINE_STAGES:
            if stage in completed:
                continue
            attempts = int(state["attempts"].get(stage.value, 0))
            if attempts >= self.stage_attempt_limit:
                blocked = SupportedStageBlocked(
                    SupportedLiveBlockCode.ATTEMPT_BUDGET_EXHAUSTED,
                    "the stage attempt budget is exhausted",
                    retryable=False,
                    details={"limit": self.stage_attempt_limit},
                )
                self._block(state, stage, blocked, attempts)
                break
            upper_bound = self.pipeline.model_call_upper_bound(stage)
            remaining = self.max_model_calls_per_case - int(state["model_calls_used"])
            if upper_bound > remaining:
                blocked = SupportedStageBlocked(
                    SupportedLiveBlockCode.MODEL_CALL_BUDGET_EXHAUSTED,
                    "the remaining model-call budget cannot cover this stage",
                    retryable=False,
                    details={"required": upper_bound, "remaining": remaining},
                )
                self._block(state, stage, blocked, attempts)
                break
            attempts += 1
            state["attempts"][stage.value] = attempts
            try:
                output = self.pipeline.run_stage(case, stage, completed)
                if output.model_calls > upper_bound or output.model_calls > remaining:
                    raise RuntimeError("pipeline exceeded its declared model-call budget")
                if stage is SupportedLiveStage.FIVE_CANDIDATE_EXECUTION:
                    self._verify_execution_evidence(output.data)
                if stage is SupportedLiveStage.COMPLETION_EVIDENCE:
                    self._validate_completion(case, output.data)
                self._complete_stage(state, stage, output)
                completed[stage] = output.data
                _atomic_json(self._state_path(case), state)
            except SupportedStageBlocked as blocked:
                self._block(state, stage, blocked, attempts)
                break
            except Exception as error:
                blocked = SupportedStageBlocked(
                    SupportedLiveBlockCode.STAGE_FAILED,
                    "the production stage raised an exception",
                    retryable=True,
                    details={"error_type": type(error).__name__},
                )
                self._block(state, stage, blocked, attempts)
                state["status"] = SupportedLiveStatus.FAILED.value
                break
        else:
            state["status"] = SupportedLiveStatus.COMPLETE.value
            state["blocker"] = None
        _atomic_json(self._state_path(case), state)
        return state

    def run(self) -> SupportedLiveRunSummary:
        states = tuple(self.run_case(case) for case in self.selected_cases)
        counts = {status.value: 0 for status in SupportedLiveStatus}
        for state in states:
            counts[str(state["status"])] += 1
        summary = SupportedLiveRunSummary(
            run_root=self.run_root,
            selected_case_ids=tuple(case.case_id for case in self.selected_cases),
            status_counts=counts,
            cases=states,
        )
        _atomic_json(
            self.run_root / "latest-summary.json",
            {
                "schema_version": self.SCHEMA_VERSION,
                "selected_case_ids": list(summary.selected_case_ids),
                "status_counts": dict(summary.status_counts),
                "all_complete": summary.all_complete,
                "cases": [
                    {
                        "case_id": item["case_id"],
                        "status": item["status"],
                        "blocker": item["blocker"],
                        "completed_stages": sorted(item["completed_stages"]),
                    }
                    for item in states
                ],
            },
        )
        return summary


class ProductionSupportedStagePipeline:
    """Adapter over the real retrieval/planner/compiler/executor/judge interfaces."""

    def __init__(
        self,
        *,
        planner: Any | None = None,
        retrieval: CertifiedLibraryService | None = None,
        retrieval_release: str = "",
        retrieval_query: Callable[[BenchmarkCaseV1], RetrievalQuery] | None = None,
        retrieval_filters: Callable[[BenchmarkCaseV1], Any] | None = None,
        rig: Any | None = None,
        scene: Any | None = None,
        executor: Any | None = None,
        artifacts: ContentAddressedArtifactStore | None = None,
        selector: BestOfFiveOrchestrator | None = None,
        completion_provider: Callable[
            [BenchmarkCaseV1, Mapping[SupportedLiveStage, Mapping[str, object]]],
            Mapping[str, object],
        ]
        | None = None,
        external_model_diagnostic: Mapping[str, object] | None = None,
        pipeline_id: str = "rigby-v2-production-supported.v1",
    ) -> None:
        self.pipeline_id = pipeline_id
        self.planner = planner
        self.retrieval = retrieval
        self.retrieval_release = retrieval_release
        self.retrieval_query = retrieval_query
        self.retrieval_filters = retrieval_filters
        self.rig = rig
        self.scene = scene
        self.executor = executor
        self.artifacts = artifacts
        self.selector = selector
        self.completion_provider = completion_provider
        self.external_model_diagnostic = dict(external_model_diagnostic or {})

    def readiness_diagnostic(self) -> dict[str, object]:
        """Return a value-free inventory of every authority needed downstream."""

        missing = []
        if self.planner is None:
            missing.append("live_planner_adapter")
        if self.selector is None:
            missing.append("live_judge_adapter")
        if self.rig is None:
            missing.append("canonical_rig_binding")
        if self.scene is None:
            missing.append("compiled_scene_binding")
        if self.executor is None:
            missing.append("exact_five_executor_binding")
        elif not callable(getattr(self.executor, "execute_many", None)):
            missing.append("exact_five_execute_many")
        if self.retrieval is None:
            missing.append("certified_retrieval_service")
        if self.retrieval_query is None:
            missing.append("retrieval_query_binding")
        if self.retrieval_filters is None:
            missing.append("retrieval_filter_binding")
        if not self.retrieval_release:
            missing.append("active_certified_retrieval_release")
        if self.completion_provider is None:
            missing.append("completion_evidence_provider")
        return {
            "schema_version": "1.0",
            "external_model_authority": self.external_model_diagnostic,
            "bindings": {
                "planner": self.planner is not None,
                "judge": self.selector is not None,
                "canonical_rig": self.rig is not None,
                "compiled_scene": self.scene is not None,
                "exact_five_executor": self.executor is not None
                and callable(getattr(self.executor, "execute_many", None)),
                "certified_retrieval": self.retrieval is not None,
                "retrieval_query": self.retrieval_query is not None,
                "retrieval_filters": self.retrieval_filters is not None,
                "active_retrieval_release": bool(self.retrieval_release),
                "completion_evidence_provider": self.completion_provider is not None,
            },
            "missing_authority": missing,
        }

    def bind_artifact_store(self, artifacts: ContentAddressedArtifactStore) -> None:
        """Bind nested execution evidence to the runner's authoritative CAS."""

        stores = [
            store
            for store in (self.artifacts, getattr(self.executor, "artifacts", None))
            if store is not None
        ]
        if any(store.root != artifacts.root for store in stores):
            raise ValueError(
                "production executor and supported runner must share one artifact store"
            )
        self.artifacts = artifacts

    def model_call_upper_bound(self, stage: SupportedLiveStage) -> int:
        if stage is SupportedLiveStage.PLANNING:
            return 1
        if stage is SupportedLiveStage.JUDGMENT:
            return BestOfFiveOrchestrator.MAX_TOTAL_CALLS
        return 0

    def _external_gate(self) -> SupportedStageOutput:
        missing = []
        if self.planner is None:
            missing.append("planner")
        if self.selector is None:
            missing.append("judge")
        if missing:
            raise SupportedStageBlocked(
                SupportedLiveBlockCode.EXTERNAL_MODEL_UNAVAILABLE,
                "configured live planner and judge are not both available",
                retryable=True,
                details={
                    "missing": missing,
                    "readiness": self.readiness_diagnostic(),
                    "model_call_attempted": False,
                },
            )
        return SupportedStageOutput(
            {
                "planner_id": str(getattr(self.planner, "planner_id", "configured")),
                "judge_id": str(getattr(self.selector.judge, "judge_id", "configured")),
                "readiness": self.readiness_diagnostic(),
            }
        )

    def run_stage(
        self,
        case: BenchmarkCaseV1,
        stage: SupportedLiveStage,
        completed: Mapping[SupportedLiveStage, Mapping[str, object]],
    ) -> SupportedStageOutput:
        if stage is SupportedLiveStage.EXTERNAL_MODEL_GATE:
            return self._external_gate()
        if self.rig is None or self.scene is None or self.executor is None:
            raise SupportedStageBlocked(
                SupportedLiveBlockCode.CASE_BINDING_MISSING,
                "the benchmark case has no staged canonical rig and scene binding",
                retryable=True,
                details={
                    "readiness": self.readiness_diagnostic(),
                    "model_call_attempted": False,
                },
            )
        if stage is SupportedLiveStage.RETRIEVAL:
            if (
                self.retrieval is None
                or self.retrieval_query is None
                or self.retrieval_filters is None
                or not self.retrieval_release
            ):
                raise SupportedStageBlocked(
                    SupportedLiveBlockCode.RETRIEVAL_UNAVAILABLE,
                    "certified active-release retrieval is not configured",
                    retryable=True,
                    details={
                        "readiness": self.readiness_diagnostic(),
                        "model_call_attempted": False,
                    },
                )
            response = self.retrieval.retrieve(
                self.retrieval_query(case), self.retrieval_filters(case)
            )
            return SupportedStageOutput(
                {
                    "fused_result": response.result.model_dump(mode="json"),
                    "examples": [
                        {"record_id": item.record_id, "context": item.context}
                        for item in response.examples
                    ],
                }
            )
        if stage is SupportedLiveStage.PLANNING:
            from rigby_v2.library.models import CompactExample

            retrieval = completed[SupportedLiveStage.RETRIEVAL]
            examples = tuple(
                CompactExample(str(item["record_id"]), str(item["context"]))
                for item in retrieval["examples"]  # type: ignore[index]
            )
            plan = self.planner.plan(
                prompt=case.prompt,
                rig=self.rig.manifest,
                rig_artifact_sha256=self.rig.reference.sha256,
                scene=self.scene.manifest,
                retrieval_release=self.retrieval_release,
                retrieval_examples=examples,
                seed=case.seed,
            )
            return SupportedStageOutput(
                {"semantic_plan": plan.model_dump(mode="json")}, model_calls=1
            )
        if stage is SupportedLiveStage.CANDIDATE_GENERATION:
            plan = SemanticPlanV1.model_validate(
                completed[SupportedLiveStage.PLANNING]["semantic_plan"]
            )
            candidates = generate_candidate_set(
                plan, getattr(self.executor, "candidate_compiler", None)
            )
            return SupportedStageOutput(
                {"candidate_set": candidates.model_dump(mode="json")}
            )
        if stage is SupportedLiveStage.FIVE_CANDIDATE_EXECUTION:
            candidates = CandidateSetV1.model_validate(
                completed[SupportedLiveStage.CANDIDATE_GENERATION]["candidate_set"]
            )
            execute_many = getattr(self.executor, "execute_many", None)
            if not callable(execute_many):
                raise RuntimeError("production executor lacks exact-five batch execution")
            executions = tuple(execute_many(candidates.candidates))
            if len(executions) != 5:
                raise RuntimeError("production executor did not return exactly five results")
            evidence_store = self.artifacts or getattr(self.executor, "artifacts", None)
            if evidence_store is None:
                raise RuntimeError("production evidence has no authoritative artifact store")
            for execution in executions:
                evidence_store.read_bytes(execution.evidence.trace, verify=True)
                for camera in execution.evidence.cameras:
                    evidence_store.read_bytes(camera.raw_frames, verify=True)
                    evidence_store.read_bytes(camera.video, verify=True)
            return SupportedStageOutput(
                {"executions": [_execution_payload(item) for item in executions]}
            )
        if stage is SupportedLiveStage.JUDGMENT:
            candidates = CandidateSetV1.model_validate(
                completed[SupportedLiveStage.CANDIDATE_GENERATION]["candidate_set"]
            )
            executions = tuple(
                _execution_from_payload(item)
                for item in completed[SupportedLiveStage.FIVE_CANDIDATE_EXECUTION][
                    "executions"
                ]  # type: ignore[index]
            )
            selection = self.selector.select(
                candidates, tuple(item.gated_evidence() for item in executions)
            )
            return SupportedStageOutput(
                {"selection": selection.model_dump(mode="json")},
                model_calls=selection.total_judge_and_repair_calls,
            )
        if stage is SupportedLiveStage.COMPLETION_EVIDENCE:
            if self.completion_provider is None:
                raise SupportedStageBlocked(
                    SupportedLiveBlockCode.EXTERNAL_EVIDENCE_UNAVAILABLE,
                    "human, RAG-off/on, retrieval, and performance evidence is incomplete",
                    retryable=True,
                    details={"readiness": self.readiness_diagnostic()},
                )
            return SupportedStageOutput(dict(self.completion_provider(case, completed)))
        raise ValueError(f"unsupported pipeline stage: {stage.value}")


def _execution_payload(item: CandidateExecution) -> dict[str, object]:
    return {
        "candidate_id": item.candidate_id,
        "evidence": item.evidence.model_dump(mode="json"),
        "certified": item.certified,
        "certification_id": item.certification_id,
        "certification_sha256": item.certification_sha256,
        "resimulation_result_sha256": item.resimulation_result_sha256,
        "certifier_id": item.certifier_id,
        "repeat_count": item.repeat_count,
        "replay_exact": item.replay_exact,
        "failure_code": item.failure_code,
        "failed_predicate": item.failed_predicate,
    }


def _execution_from_payload(value: Mapping[str, object]) -> CandidateExecution:
    return CandidateExecution(
        candidate_id=str(value["candidate_id"]),
        evidence=CandidateEvidenceV1.model_validate(value["evidence"]),
        certified=bool(value["certified"]),
        certification_id=str(value["certification_id"]),
        certification_sha256=str(value["certification_sha256"]),
        resimulation_result_sha256=str(value["resimulation_result_sha256"]),
        certifier_id=str(value["certifier_id"]),
        repeat_count=int(value["repeat_count"]),
        replay_exact=bool(value["replay_exact"]),
        failure_code=(str(value["failure_code"]) if value.get("failure_code") else None),
        failed_predicate=(
            str(value["failed_predicate"]) if value.get("failed_predicate") else None
        ),
    )


def unavailable_live_pipeline() -> ProductionSupportedStagePipeline:
    """A production adapter that records a truthful external-model blocker."""

    return ProductionSupportedStagePipeline()


def run_communicative_canary(
    run_root: Path,
    *,
    count: int = 3,
    stage_attempt_limit: int = 2,
    max_model_calls_per_case: int = 5,
) -> SupportedLiveRunSummary:
    manifest = load_benchmark_manifest()
    cases = communicative_canary_cases(manifest, count)
    return ResumableSupportedBenchmarkRunner(
        run_root=run_root,
        manifest=manifest,
        pipeline=unavailable_live_pipeline(),
        selected_cases=cases,
        stage_attempt_limit=stage_attempt_limit,
        max_model_calls_per_case=max_model_calls_per_case,
    ).run()
