from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.benchmark.live_runner import (
    ProductionSupportedStagePipeline,
    ResumableSupportedBenchmarkRunner,
    SupportedLiveBlockCode,
    SupportedLiveStage,
    SupportedLiveStatus,
    SupportedStageBlocked,
    SupportedStageOutput,
    communicative_canary_cases,
    select_supported_cases,
)
from rigby_v2.benchmark.live_configuration import (
    discover_live_model_configuration,
)
from rigby_v2.benchmark.manifest import load_benchmark_manifest
from rigby_v2.benchmark.models import BenchmarkFamily
from rigby_core.contracts import ArtifactRefV1

pytestmark = pytest.mark.medium


class _ResumePipeline:
    pipeline_id = "test-resume-pipeline.v1"

    def __init__(self) -> None:
        self.retrieval_calls = 0

    def model_call_upper_bound(self, stage: SupportedLiveStage) -> int:
        return 1 if stage is SupportedLiveStage.PLANNING else 0

    def run_stage(self, case, stage, completed):  # type: ignore[no-untyped-def]
        del case, completed
        if stage is SupportedLiveStage.EXTERNAL_MODEL_GATE:
            return SupportedStageOutput({"models": "configured"})
        if stage is SupportedLiveStage.RETRIEVAL:
            self.retrieval_calls += 1
            if self.retrieval_calls == 1:
                raise SupportedStageBlocked(
                    SupportedLiveBlockCode.RETRIEVAL_UNAVAILABLE,
                    "active release is temporarily unavailable",
                    retryable=True,
                )
            return SupportedStageOutput({"examples": ["a", "b"]})
        if stage is SupportedLiveStage.PLANNING:
            raise SupportedStageBlocked(
                SupportedLiveBlockCode.EXTERNAL_MODEL_UNAVAILABLE,
                "planner is temporarily unavailable",
                retryable=True,
            )
        raise AssertionError(stage)


class _BudgetPipeline:
    pipeline_id = "test-budget-pipeline.v1"

    def model_call_upper_bound(self, stage: SupportedLiveStage) -> int:
        return 1 if stage is SupportedLiveStage.PLANNING else 0

    def run_stage(self, case, stage, completed):  # type: ignore[no-untyped-def]
        del case, completed
        if stage is SupportedLiveStage.EXTERNAL_MODEL_GATE:
            return SupportedStageOutput({"models": "configured"})
        if stage is SupportedLiveStage.RETRIEVAL:
            return SupportedStageOutput({"examples": ["a", "b"]})
        raise AssertionError("the model call must be stopped before invocation")


class _IncompletePipeline:
    pipeline_id = "test-incomplete-pipeline.v1"

    def bind_artifact_store(self, artifacts):  # type: ignore[no-untyped-def]
        self.artifacts = artifacts

    def model_call_upper_bound(self, stage: SupportedLiveStage) -> int:
        return 0

    def run_stage(self, case, stage, completed):  # type: ignore[no-untyped-def]
        del case, completed
        if stage is SupportedLiveStage.FIVE_CANDIDATE_EXECUTION:
            reference = self.artifacts.put_json({"authoritative": "test-trace"})
            matrix = [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            executions = []
            for index in range(5):
                cameras = [
                    {
                        "camera_id": f"camera-{role}-{index}",
                        "role": role,
                        "fps": 30,
                        "raw_frames": reference.model_dump(mode="json"),
                        "video": reference.model_dump(mode="json"),
                        "timestamps_s": [0.0, 1.0],
                        "world_from_camera": [matrix, matrix],
                        "intrinsics": [1.0] * 9,
                    }
                    for role in ("orbit", "egocentric", "task_closeup")
                ]
                executions.append(
                    {
                        "evidence": {
                            "schema_version": "1.0",
                            "anonymous_id": f"anonymous-{index}",
                            "candidate_id": f"candidate-{index}",
                            "trace": reference.model_dump(mode="json"),
                            "cameras": cameras,
                            "metrics": {},
                        }
                    }
                )
            return SupportedStageOutput({"executions": executions})
        if stage is SupportedLiveStage.COMPLETION_EVIDENCE:
            return SupportedStageOutput({"result": {}, "evidence": {}})
        return SupportedStageOutput({"stage_ran": stage.value})


def _runner(
    root: Path,
    pipeline,
    *,
    attempts: int = 2,
    model_calls: int = 5,
) -> ResumableSupportedBenchmarkRunner:
    manifest = load_benchmark_manifest()
    case = communicative_canary_cases(manifest, 3)[0]
    return ResumableSupportedBenchmarkRunner(
        run_root=root,
        manifest=manifest,
        pipeline=pipeline,
        selected_cases=(case,),
        stage_attempt_limit=attempts,
        max_model_calls_per_case=model_calls,
    )


def test_supported_selection_is_the_sealed_300_and_can_filter() -> None:
    manifest = load_benchmark_manifest()
    assert len(select_supported_cases(manifest)) == 300
    communicative = select_supported_cases(
        manifest, families=(BenchmarkFamily.COMMUNICATIVE_GESTURE,)
    )
    assert len(communicative) == 75
    assert communicative_canary_cases(manifest, 5) == communicative[:5]
    selected = select_supported_cases(
        manifest,
        case_ids=(communicative[7].case_id,),
        families=(BenchmarkFamily.GRASP_PLACE,),
    )
    assert len(selected) == 76
    assert selected[0].family is BenchmarkFamily.COMMUNICATIVE_GESTURE


def test_real_canary_preflight_seals_external_model_blockers(tmp_path: Path) -> None:
    manifest = load_benchmark_manifest()
    cases = communicative_canary_cases(manifest, 3)
    runner = ResumableSupportedBenchmarkRunner(
        run_root=tmp_path / "run",
        manifest=manifest,
        pipeline=ProductionSupportedStagePipeline(),
        selected_cases=cases,
    )

    summary = runner.run()

    assert summary.status_counts[SupportedLiveStatus.BLOCKED.value] == 3
    assert not summary.all_complete
    for state in summary.cases:
        assert state["blocker"]["code"] == "external_model_unavailable"
        assert state["blocker"]["details"]["model_call_attempted"] is False
        readiness = state["blocker"]["details"]["readiness"]
        assert readiness["bindings"]["planner"] is False
        assert readiness["bindings"]["judge"] is False
        assert "canonical_rig_binding" in readiness["missing_authority"]
        assert "active_certified_retrieval_release" in readiness["missing_authority"]
        assert "completion_evidence_provider" in readiness["missing_authority"]
        assert set(state["completed_stages"]) == {"manifest_preflight"}
        assert state["model_calls_used"] == 0
        preflight = ArtifactRefV1.model_validate(
            state["completed_stages"]["manifest_preflight"]
        )
        payload = json.loads(runner.artifacts.read_bytes(preflight))
        assert payload["case_id"] == state["case_id"]
        assert payload["case_hash"] == state["case_hash"]
        assert payload["pipeline_id"] == runner.pipeline.pipeline_id
        assert payload["data"]["case"]["case_hash"] == state["case_hash"]
        assert payload["data"]["guardrail"]["outcome"] == "allow"
        attempt = ArtifactRefV1.model_validate(
            state["attempt_artifacts"]["external_model_gate"][-1]
        )
        attempt_payload = json.loads(runner.artifacts.read_bytes(attempt))
        assert attempt_payload["case_id"] == state["case_id"]
        assert attempt_payload["case_hash"] == state["case_hash"]


def test_dotenv_model_discovery_reports_presence_without_values(tmp_path: Path) -> None:
    secret = "credential-must-never-enter-diagnostics"
    planner = "planner-model-must-not-enter-diagnostics"
    judge = "judge-model-must-not-enter-diagnostics"
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            (
                f"OPENAI_API_KEY={secret}",
                f"OPENAI_PLANNER_MODEL={planner}",
                f"OPENAI_JUDGE_MODEL={judge}",
            )
        ),
        encoding="utf-8",
    )

    configuration = discover_live_model_configuration(
        environ={}, dotenv_paths=(dotenv,)
    )
    diagnostic = configuration.diagnostic()
    serialized = json.dumps(diagnostic, sort_keys=True)

    assert configuration.configured
    assert diagnostic["api_credential_present"] is True
    assert diagnostic["planner_model_present"] is True
    assert diagnostic["judge_model_present"] is True
    assert diagnostic["missing_authority"] == []
    assert secret not in serialized
    assert planner not in serialized
    assert judge not in serialized
    assert secret not in repr(configuration)
    assert planner not in repr(configuration)
    assert judge not in repr(configuration)


def test_partial_authority_names_precise_missing_model_configuration(
    tmp_path: Path,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=present-but-redacted\n", encoding="utf-8")

    diagnostic = discover_live_model_configuration(
        environ={}, dotenv_paths=(dotenv,)
    ).diagnostic()

    assert diagnostic["configured"] is False
    assert diagnostic["api_credential_present"] is True
    assert diagnostic["missing_authority"] == [
        "planner_model_configuration",
        "judge_model_configuration",
    ]


def test_resume_reuses_completed_artifacts_and_retries_only_blocked_stage(
    tmp_path: Path,
) -> None:
    pipeline = _ResumePipeline()
    runner = _runner(tmp_path / "run", pipeline)
    first = runner.run().cases[0]
    first_preflight = first["completed_stages"]["manifest_preflight"]
    first_gate = first["completed_stages"]["external_model_gate"]
    assert first["blocker"]["code"] == "retrieval_unavailable"

    second = runner.run().cases[0]

    assert second["completed_stages"]["manifest_preflight"] == first_preflight
    assert second["completed_stages"]["external_model_gate"] == first_gate
    assert second["attempts"]["external_model_gate"] == 1
    assert second["attempts"]["retrieval"] == 2
    assert second["attempts"]["planning"] == 1
    assert second["blocker"]["code"] == "external_model_unavailable"


def test_model_budget_stops_stage_before_call(tmp_path: Path) -> None:
    state = _runner(
        tmp_path / "run", _BudgetPipeline(), model_calls=0
    ).run().cases[0]

    assert state["status"] == SupportedLiveStatus.BLOCKED.value
    assert state["blocker"]["code"] == "model_call_budget_exhausted"
    assert state["blocker"]["stage"] == "planning"
    assert "planning" not in state["attempts"]
    assert state["model_calls_used"] == 0


def test_incomplete_completion_evidence_never_becomes_passed(tmp_path: Path) -> None:
    state = _runner(tmp_path / "run", _IncompletePipeline()).run().cases[0]

    assert state["status"] == SupportedLiveStatus.BLOCKED.value
    assert state["blocker"]["code"] == "incomplete_evidence"
    assert "completion_evidence" not in state["completed_stages"]


def test_tampered_immutable_stage_artifact_fails_closed(tmp_path: Path) -> None:
    runner = _runner(tmp_path / "run", ProductionSupportedStagePipeline())
    first = runner.run().cases[0]
    reference = ArtifactRefV1.model_validate(
        first["completed_stages"]["manifest_preflight"]
    )
    runner.artifacts.resolve(reference).write_bytes(b"tampered")

    second = runner.run().cases[0]

    assert second["status"] == SupportedLiveStatus.BLOCKED.value
    assert second["blocker"]["code"] == "artifact_integrity"
    assert second["blocker"]["retryable"] is False


def test_resume_reverifies_nested_execution_evidence_artifacts(tmp_path: Path) -> None:
    runner = _runner(tmp_path / "run", _IncompletePipeline())
    first = runner.run().cases[0]
    execution_ref = ArtifactRefV1.model_validate(
        first["completed_stages"]["five_candidate_execution"]
    )
    execution_payload = json.loads(runner.artifacts.read_bytes(execution_ref))
    trace_ref = ArtifactRefV1.model_validate(
        execution_payload["data"]["executions"][0]["evidence"]["trace"]
    )
    runner.artifacts.resolve(trace_ref).write_bytes(b"tampered nested trace")

    second = runner.run().cases[0]

    assert second["status"] == SupportedLiveStatus.BLOCKED.value
    assert second["blocker"]["code"] == "artifact_integrity"
    assert second["blocker"]["retryable"] is False


def test_production_executor_must_share_the_runner_artifact_store(
    tmp_path: Path,
) -> None:
    manifest = load_benchmark_manifest()
    pipeline = ProductionSupportedStagePipeline(
        executor=SimpleNamespace(
            artifacts=ContentAddressedArtifactStore(tmp_path / "other-artifacts")
        )
    )
    with pytest.raises(ValueError, match="share one artifact store"):
        ResumableSupportedBenchmarkRunner(
            run_root=tmp_path / "run",
            manifest=manifest,
            pipeline=pipeline,
            selected_cases=(communicative_canary_cases(manifest, 3)[0],),
        )
