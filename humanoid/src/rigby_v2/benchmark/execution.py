"""Evidence-backed execution and atomic recording of the release benchmark."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

from rigby_core.hashing import canonical_json_bytes, hash_file, validate_sha256

from .evaluator import BOOTSTRAP_SAMPLES, evaluate_release_gate
from .models import (
    AdversarialCaseResultV1,
    BenchmarkCaseKind,
    BenchmarkCaseResultV1,
    BenchmarkCaseV1,
    BenchmarkManifestV1,
    BenchmarkRunV1,
    ReleaseGateReportV1,
    SupportedCaseResultV1,
)


SUPPORTED_EVIDENCE_KINDS = frozenset(
    {
        "simulation",
        "repeatability",
        "vlm_judgment",
        "human_rating",
        "retrieval",
        "rag_off",
        "rag_on",
        "performance",
    }
)
ADVERSARIAL_EVIDENCE_KINDS = frozenset({"deterministic_outcome", "judge"})


class BenchmarkExecutionFailureCode(StrEnum):
    RUNNER_FAILED = "runner_failed"
    INVALID_RESULT = "invalid_result"
    MISSING_EVIDENCE = "missing_evidence"
    ARTIFACT_INTEGRITY = "artifact_integrity"
    OUTPUT_EXISTS = "output_exists"


class BenchmarkExecutionError(RuntimeError):
    def __init__(
        self,
        code: BenchmarkExecutionFailureCode,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class BenchmarkEvidenceArtifact:
    path: str
    sha256: str
    media_type: str = "application/json"

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.path)
        if (
            not self.path
            or "\\" in self.path
            or pure.is_absolute()
            or ".." in pure.parts
            or not pure.name
        ):
            raise ValueError("benchmark evidence paths must be safe relative POSIX paths")
        object.__setattr__(self, "sha256", validate_sha256(self.sha256))
        if not self.media_type.strip():
            raise ValueError("benchmark evidence media type is required")


@dataclass(frozen=True, slots=True)
class EvidencedBenchmarkCaseResult:
    result: BenchmarkCaseResultV1
    evidence: Mapping[str, BenchmarkEvidenceArtifact]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


class BenchmarkCaseRunner(Protocol):
    runner_id: str

    def run_case(self, case: BenchmarkCaseV1) -> EvidencedBenchmarkCaseResult: ...


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionArtifacts:
    output_root: Path
    run_path: Path
    run_sha256: str
    evidence_ledger_path: Path
    evidence_ledger_sha256: str
    release_gate_report_path: Path
    release_gate_report_sha256: str
    execution_manifest_path: Path
    run: BenchmarkRunV1
    report: ReleaseGateReportV1


def _evidence_path(root: Path, reference: BenchmarkEvidenceArtifact) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / Path(*PurePosixPath(reference.path).parts)).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise BenchmarkExecutionError(
            BenchmarkExecutionFailureCode.ARTIFACT_INTEGRITY,
            "benchmark evidence path escapes its root",
            details={"path": reference.path},
        ) from error
    if not path.is_file():
        raise BenchmarkExecutionError(
            BenchmarkExecutionFailureCode.MISSING_EVIDENCE,
            "benchmark evidence artifact is missing",
            details={"path": reference.path},
        )
    observed = hash_file(path)
    if observed != reference.sha256:
        raise BenchmarkExecutionError(
            BenchmarkExecutionFailureCode.ARTIFACT_INTEGRITY,
            "benchmark evidence artifact hash mismatch",
            details={
                "path": reference.path,
                "expected_sha256": reference.sha256,
                "observed_sha256": observed,
            },
        )
    return path


def _validate_case_result(
    case: BenchmarkCaseV1,
    recorded: EvidencedBenchmarkCaseResult,
    *,
    evidence_root: Path,
) -> None:
    result = recorded.result
    if result.case_id != case.case_id or result.case_hash != case.case_hash:
        raise BenchmarkExecutionError(
            BenchmarkExecutionFailureCode.INVALID_RESULT,
            "case runner returned a result for the wrong immutable benchmark case",
            details={"expected_case_id": case.case_id, "observed_case_id": result.case_id},
        )
    if case.kind is BenchmarkCaseKind.SUPPORTED:
        if not isinstance(result, SupportedCaseResultV1):
            raise BenchmarkExecutionError(
                BenchmarkExecutionFailureCode.INVALID_RESULT,
                "supported case runner returned an adversarial result contract",
                details={"case_id": case.case_id},
            )
        required = SUPPORTED_EVIDENCE_KINDS
    else:
        if not isinstance(result, AdversarialCaseResultV1):
            raise BenchmarkExecutionError(
                BenchmarkExecutionFailureCode.INVALID_RESULT,
                "adversarial case runner returned a supported result contract",
                details={"case_id": case.case_id},
            )
        required = ADVERSARIAL_EVIDENCE_KINDS
    observed = set(recorded.evidence)
    if observed != required:
        raise BenchmarkExecutionError(
            BenchmarkExecutionFailureCode.MISSING_EVIDENCE,
            "case result does not have the exact required evidence set",
            details={
                "case_id": case.case_id,
                "missing": sorted(required - observed),
                "unknown": sorted(observed - required),
            },
        )
    for reference in recorded.evidence.values():
        _evidence_path(evidence_root, reference)


def execute_and_record_benchmark(
    manifest: BenchmarkManifestV1,
    runner: BenchmarkCaseRunner,
    *,
    evidence_root: Path,
    environment_evidence: BenchmarkEvidenceArtifact,
    output_root: Path,
    run_id: str,
    completed_at: datetime | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> BenchmarkExecutionArtifacts:
    """Execute all 400 cases and publish a sealed run only after evidence validates."""

    if not run_id.strip() or not getattr(runner, "runner_id", "").strip():
        raise ValueError("run ID and case-runner ID are required")
    completed = completed_at or datetime.now(UTC)
    if completed.tzinfo is None:
        raise ValueError("benchmark completion time must be timezone-aware")
    output_root = output_root.resolve()
    if output_root.exists():
        raise BenchmarkExecutionError(
            BenchmarkExecutionFailureCode.OUTPUT_EXISTS,
            "benchmark output root already exists",
            details={"output_root": str(output_root)},
        )
    _evidence_path(evidence_root, environment_evidence)

    recorded_results = []
    ledger_cases = []
    for case in manifest.cases:
        try:
            recorded = runner.run_case(case)
        except BenchmarkExecutionError:
            raise
        except Exception as error:
            raise BenchmarkExecutionError(
                BenchmarkExecutionFailureCode.RUNNER_FAILED,
                "benchmark case runner failed",
                details={"case_id": case.case_id, "runner_error": type(error).__name__},
            ) from error
        _validate_case_result(case, recorded, evidence_root=evidence_root)
        recorded_results.append(recorded.result)
        ledger_cases.append(
            {
                "case_id": case.case_id,
                "case_hash": case.case_hash,
                "result_hash": recorded.result.result_hash,
                "evidence": {
                    key: {
                        "path": reference.path,
                        "sha256": reference.sha256,
                        "media_type": reference.media_type,
                    }
                    for key, reference in sorted(recorded.evidence.items())
                },
            }
        )
    run = BenchmarkRunV1(
        run_id=run_id,
        manifest_hash=manifest.content_hash(),
        environment_hash=environment_evidence.sha256,
        completed_at=completed,
        results=tuple(recorded_results),
    )
    report = evaluate_release_gate(
        manifest, run, bootstrap_samples=bootstrap_samples
    )
    ledger = {
        "schema_version": "1.0",
        "run_id": run_id,
        "runner_id": runner.runner_id,
        "manifest_hash": manifest.content_hash(),
        "environment_evidence": {
            "path": environment_evidence.path,
            "sha256": environment_evidence.sha256,
            "media_type": environment_evidence.media_type,
        },
        "cases": ledger_cases,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        run_path = temporary / "benchmark-run.json"
        ledger_path = temporary / "case-evidence-ledger.json"
        report_path = temporary / "release-gate-report.json"
        run_path.write_bytes(canonical_json_bytes(run.model_dump(mode="json")))
        ledger_path.write_bytes(canonical_json_bytes(ledger))
        report_path.write_bytes(canonical_json_bytes(report.model_dump(mode="json")))
        execution_manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "runner_id": runner.runner_id,
            "release_allowed": report.release_allowed,
            "artifacts": {
                "benchmark_run": {
                    "path": run_path.name,
                    "sha256": hash_file(run_path),
                },
                "case_evidence_ledger": {
                    "path": ledger_path.name,
                    "sha256": hash_file(ledger_path),
                },
                "release_gate_report": {
                    "path": report_path.name,
                    "sha256": hash_file(report_path),
                },
            },
        }
        manifest_path = temporary / "execution-manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(execution_manifest))
        try:
            os.rename(temporary, output_root)
        except FileExistsError:
            raise BenchmarkExecutionError(
                BenchmarkExecutionFailureCode.OUTPUT_EXISTS,
                "benchmark output root was created concurrently",
                details={"output_root": str(output_root)},
            ) from None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return BenchmarkExecutionArtifacts(
        output_root=output_root,
        run_path=output_root / run_path.name,
        run_sha256=execution_manifest["artifacts"]["benchmark_run"]["sha256"],
        evidence_ledger_path=output_root / ledger_path.name,
        evidence_ledger_sha256=execution_manifest["artifacts"]["case_evidence_ledger"][
            "sha256"
        ],
        release_gate_report_path=output_root / report_path.name,
        release_gate_report_sha256=execution_manifest["artifacts"][
            "release_gate_report"
        ]["sha256"],
        execution_manifest_path=output_root / manifest_path.name,
        run=run,
        report=report,
    )
