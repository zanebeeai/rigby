from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rigby_v2.benchmark import (
    ADVERSARIAL_EVIDENCE_KINDS,
    SUPPORTED_EVIDENCE_KINDS,
    AdversarialCaseResultV1,
    BenchmarkCaseKind,
    BenchmarkEvidenceArtifact,
    BenchmarkExecutionError,
    BenchmarkExecutionFailureCode,
    EvidencedBenchmarkCaseResult,
    ExpectedOutcome,
    ObservedOutcome,
    SupportedCaseResultV1,
    execute_and_record_benchmark,
    load_benchmark_manifest,
)
from rigby_v2.hashing import hash_file

pytestmark = pytest.mark.fast


def _evidence(root: Path, name: str) -> BenchmarkEvidenceArtifact:
    path = root / f"{name}.json"
    path.write_text(json.dumps({"fixture_kind": name}), encoding="utf-8")
    return BenchmarkEvidenceArtifact(path=path.name, sha256=hash_file(path))


class FixtureCaseRunner:
    runner_id = "injected-test-fixture-runner"

    def __init__(self, references, *, omit=None, wrong_case=False, corrupt=False):
        self.references = references
        self.omit = omit
        self.wrong_case = wrong_case
        self.corrupt = corrupt
        self.calls = []
        self.supported_index = 0

    def run_case(self, case):
        self.calls.append(case.case_id)
        case_id = "wrong-case" if self.wrong_case else case.case_id
        if case.kind is BenchmarkCaseKind.SUPPORTED:
            result = SupportedCaseResultV1(
                case_id=case_id,
                case_hash=case.case_hash,
                observed_outcome=ObservedOutcome.CERTIFIED,
                max_fingertip_error_m=0.007,
                max_penetration_m=0.001,
                repeat_task_outcomes=("success", "success", "success"),
                vlm_human_agrees=True,
                judge_order_consistent=True,
                critical_judge_false_accept=False,
                retrieval_relevant_top3=3,
                retrieval_relevant_top5=5,
                retrieval_total_relevant=5,
                rag_off_success=self.supported_index < 270,
                rag_on_success=True,
                local_pipeline_seconds=20.0,
            )
            self.supported_index += 1
            evidence = {key: self.references[key] for key in SUPPORTED_EVIDENCE_KINDS}
        else:
            observed = {
                ExpectedOutcome.UNSUPPORTED: ObservedOutcome.UNSUPPORTED,
                ExpectedOutcome.INFEASIBLE: ObservedOutcome.INFEASIBLE,
                ExpectedOutcome.INVALID_ASSET: ObservedOutcome.INVALID_ASSET,
            }[case.expected_outcome]
            result = AdversarialCaseResultV1(
                case_id=case_id,
                case_hash=case.case_hash,
                observed_outcome=observed,
            )
            evidence = {key: self.references[key] for key in ADVERSARIAL_EVIDENCE_KINDS}
        if self.omit:
            evidence.pop(self.omit, None)
        if self.corrupt:
            first = next(iter(evidence))
            prior = evidence[first]
            evidence[first] = BenchmarkEvidenceArtifact(
                path=prior.path, sha256="f" * 64, media_type=prior.media_type
            )
        return EvidencedBenchmarkCaseResult(result=result, evidence=evidence)


def _references(root: Path):
    kinds = SUPPORTED_EVIDENCE_KINDS | ADVERSARIAL_EVIDENCE_KINDS
    return {kind: _evidence(root, kind) for kind in kinds}


def test_executes_all_400_cases_and_atomically_records_evidence_bound_run(tmp_path) -> None:
    manifest = load_benchmark_manifest()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    references = _references(evidence_root)
    environment = _evidence(evidence_root, "environment")
    runner = FixtureCaseRunner(references)
    output = tmp_path / "sealed-run"
    artifacts = execute_and_record_benchmark(
        manifest,
        runner,
        evidence_root=evidence_root,
        environment_evidence=environment,
        output_root=output,
        run_id="fixture-complete-run",
        completed_at=datetime(2026, 8, 10, 23, tzinfo=UTC),
        bootstrap_samples=1_000,
    )
    assert len(runner.calls) == 400
    assert len(artifacts.run.results) == 400
    assert artifacts.report.release_allowed
    assert artifacts.run.environment_hash == environment.sha256
    assert hash_file(artifacts.run_path) == artifacts.run_sha256
    assert hash_file(artifacts.evidence_ledger_path) == artifacts.evidence_ledger_sha256
    assert hash_file(artifacts.release_gate_report_path) == artifacts.release_gate_report_sha256
    execution_manifest = json.loads(artifacts.execution_manifest_path.read_text(encoding="utf-8"))
    assert execution_manifest["release_allowed"] is True
    ledger = json.loads(artifacts.evidence_ledger_path.read_text(encoding="utf-8"))
    assert ledger["runner_id"] == runner.runner_id
    assert len(ledger["cases"]) == 400
    first_supported = next(value for value in ledger["cases"] if value["case_id"].startswith("supported-"))
    assert set(first_supported["evidence"]) == SUPPORTED_EVIDENCE_KINDS
    with pytest.raises(BenchmarkExecutionError) as captured:
        execute_and_record_benchmark(
            manifest,
            runner,
            evidence_root=evidence_root,
            environment_evidence=environment,
            output_root=output,
            run_id="must-not-overwrite",
        )
    assert captured.value.code is BenchmarkExecutionFailureCode.OUTPUT_EXISTS


@pytest.mark.parametrize("missing", ("human_rating", "vlm_judgment", "rag_off", "rag_on"))
def test_refuses_supported_results_with_missing_external_evidence(tmp_path, missing) -> None:
    manifest = load_benchmark_manifest()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    references = _references(evidence_root)
    runner = FixtureCaseRunner(references, omit=missing)
    with pytest.raises(BenchmarkExecutionError) as captured:
        execute_and_record_benchmark(
            manifest,
            runner,
            evidence_root=evidence_root,
            environment_evidence=_evidence(evidence_root, "environment"),
            output_root=tmp_path / "must-not-publish",
            run_id="missing-evidence",
            bootstrap_samples=1_000,
        )
    assert captured.value.code is BenchmarkExecutionFailureCode.MISSING_EVIDENCE
    assert missing in captured.value.details["missing"]
    assert not (tmp_path / "must-not-publish").exists()


@pytest.mark.parametrize(
    ("runner_options", "code"),
    (
        ({"wrong_case": True}, BenchmarkExecutionFailureCode.INVALID_RESULT),
        ({"corrupt": True}, BenchmarkExecutionFailureCode.ARTIFACT_INTEGRITY),
    ),
)
def test_refuses_wrong_case_binding_or_tampered_artifact(
    tmp_path, runner_options, code
) -> None:
    manifest = load_benchmark_manifest()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    runner = FixtureCaseRunner(_references(evidence_root), **runner_options)
    with pytest.raises(BenchmarkExecutionError) as captured:
        execute_and_record_benchmark(
            manifest,
            runner,
            evidence_root=evidence_root,
            environment_evidence=_evidence(evidence_root, "environment"),
            output_root=tmp_path / "must-not-publish",
            run_id="invalid-run",
            bootstrap_samples=1_000,
        )
    assert captured.value.code is code
    assert not (tmp_path / "must-not-publish").exists()
