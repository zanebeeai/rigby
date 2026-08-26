from __future__ import annotations

from itertools import count

from rigby_v2.flywheel.schemas import RetrievalIndex, SelectionOutcome
from rigby_v2.library import (
    AnimationRecordInput,
    CertifiedLibraryService,
    EmbeddingRouter,
    InMemoryLibraryRepository,
    standard_namespace_configs,
)
from rigby_v2.pipeline import CandidateExecution, FlywheelPipeline, SurvivorValidation
from rigby_v2.selection import BestOfFiveOrchestrator, generate_candidate_set

from test_v2_selection_generation import semantic_plan
from test_v2_selection_orchestrator import ScriptedJudge, _scores, _script, evidence_set


class Passthrough:
    def embed(self, payload):  # type: ignore[no-untyped-def]
        return payload


def _library() -> tuple[CertifiedLibraryService, InMemoryLibraryRepository]:
    configs = standard_namespace_configs(
        cosmos_model_sha256="a" * 64,
        siglip2_model_sha256="b" * 64,
        descriptor_spec_sha256="c" * 64,
        cosmos_dimensions=4,
        siglip2_dimensions=4,
        descriptor_dimensions=4,
    )
    ids = count(1)
    repository = InMemoryLibraryRepository()
    service = CertifiedLibraryService(
        repository,
        EmbeddingRouter(configs, {index: Passthrough() for index in RetrievalIndex}),
        id_factory=lambda: f"00000000-0000-0000-0000-{next(ids):012d}",
    )
    service.create_release("release-1", parent_release_id=None)
    return service, repository


class Executor:
    def __init__(self) -> None:
        evidence = evidence_set()
        self.by_candidate = {item.candidate_id: item for item in evidence}

    def execute(self, proposal):  # type: ignore[no-untyped-def]
        item = self.by_candidate[proposal.candidate_id]
        failed = item.anonymous_id == "anon-0"
        return CandidateExecution(
            candidate_id=proposal.candidate_id,
            evidence=item,
            certified=not failed,
            certification_id=f"cert-{item.anonymous_id}",
            certification_sha256="d" * 64,
            resimulation_result_sha256="e" * 64,
            certifier_id="deterministic-certifier",
            repeat_count=3,
            replay_exact=True,
            failure_code="penetration" if failed else None,
        )

    def validate_survivor(self, proposal, execution):  # type: ignore[no-untyped-def]
        assert proposal.candidate_id == execution.candidate_id
        return SurvivorValidation(
            passed=True,
            variation_count=5,
            validation_sha256="f" * 64,
        )


def test_pipeline_uses_one_ordered_five_candidate_batch() -> None:
    class BatchExecutor(Executor):
        def __init__(self) -> None:
            super().__init__()
            self.batch_ids = ()

        def execute(self, proposal):  # type: ignore[no-untyped-def]
            raise AssertionError("pipeline must not use the serial path")

        def execute_many(self, proposals, *, cancel_check=None):  # type: ignore[no-untyped-def]
            assert cancel_check is None
            self.batch_ids = tuple(item.candidate_id for item in proposals)
            return tuple(Executor.execute(self, item) for item in proposals)

    executor = BatchExecutor()
    script = _script({"anon-1": _scores(4.0)})
    result = FlywheelPipeline(
        executor=executor,
        selector=BestOfFiveOrchestrator(ScriptedJudge([script, script])),
    ).run(semantic_plan())
    expected = tuple(
        item.candidate_id for item in generate_candidate_set(semantic_plan()).candidates
    )
    assert executor.batch_ids == expected
    assert tuple(item.candidate_id for item in result.executions) == expected


def _record(proposal, execution, selection):  # type: ignore[no-untyped-def]
    return AnimationRecordInput(
        schema_version="2.0",
        split="train",
        prompt_text="reach and press the button",
        program=proposal.program.model_dump(mode="json"),
        world={"scene": proposal.program.scene_id},
        motion={"certification": execution.certification_sha256},
        evidence={"trace": execution.evidence.trace.sha256},
        labels={"selection": selection.outcome.value},
        evaluation={},
        provenance={"source": "flywheel"},
        license_id="generated-local",
        rig_id=proposal.program.rig_id,
        lineage_id=proposal.semantic_plan_hash,
        compact_example="Press the button with a certified approach/contact/settle motion.",
        object_affordances=("pressable",),
        limbs=("right_hand",),
        contact_requirements=("button_contact",),
    )


def _payloads(proposal, execution):  # type: ignore[no-untyped-def]
    del proposal, execution
    return {index: (1.0, 0.0, 0.0, 0.0) for index in RetrievalIndex}


def test_full_flywheel_vetoes_failed_favorite_and_stages_only_certified_winner() -> None:
    service, repository = _library()
    script = _script({"anon-0": _scores(5.0), "anon-1": _scores(4.0)})
    pipeline = FlywheelPipeline(
        executor=Executor(),
        selector=BestOfFiveOrchestrator(ScriptedJudge([script, script])),
        library=service,
    )

    result = pipeline.run(
        semantic_plan(),
        target_release_id="release-1",
        record_factory=_record,
        index_payload_factory=_payloads,
    )

    expected = generate_candidate_set(semantic_plan()).candidates[1]
    assert result.selection.outcome is SelectionOutcome.SELECTED
    assert result.selection.selected_candidate_id == expected.candidate_id
    assert result.promoted
    assert result.staged_record is not None
    assert result.staged_record.release_id == "release-1"
    assert result.survivor_validation is not None
    assert result.survivor_validation.passed
    assert result.staged_record.evaluation["independently_certified"] is True
    assert len(repository.failures()) == 1
    assert repository.failures()[0].failure_code == "penetration"
    assert len(repository.records_for_release("release-1")) == 1


def test_pipeline_refuses_same_certifier_and_judge_identity() -> None:
    service, _ = _library()
    executor = Executor()
    for candidate_id, evidence in executor.by_candidate.items():
        if evidence.anonymous_id == "anon-1":
            original = executor.execute

            def same_identity(proposal):  # type: ignore[no-untyped-def]
                execution = original(proposal)
                if proposal.candidate_id == candidate_id:
                    return CandidateExecution(
                        **{**execution.__dict__, "certifier_id": "fake-judge"}
                    )
                return execution

            executor.execute = same_identity  # type: ignore[method-assign]
            break
    script = _script({"anon-1": _scores(4.0)})
    pipeline = FlywheelPipeline(
        executor=executor,
        selector=BestOfFiveOrchestrator(ScriptedJudge([script, script])),
        library=service,
    )

    import pytest

    with pytest.raises(ValueError, match="independent"):
        pipeline.run(
            semantic_plan(),
            target_release_id="release-1",
            record_factory=_record,
            index_payload_factory=_payloads,
        )


def test_failed_survivor_variation_is_a_hard_negative_not_positive_context() -> None:
    class FragileExecutor(Executor):
        def validate_survivor(self, proposal, execution):  # type: ignore[no-untyped-def]
            return SurvivorValidation(
                passed=False,
                variation_count=5,
                validation_sha256="9" * 64,
                failure_code="foot_slip",
            )

    service, repository = _library()
    script = _script({"anon-1": _scores(4.0)})
    result = FlywheelPipeline(
        executor=FragileExecutor(),
        selector=BestOfFiveOrchestrator(ScriptedJudge([script, script])),
        library=service,
    ).run(
        semantic_plan(),
        target_release_id="release-1",
        record_factory=_record,
        index_payload_factory=_payloads,
    )
    assert not result.promoted
    assert result.survivor_validation is not None
    assert not result.survivor_validation.passed
    assert not repository.records_for_release("release-1")
    assert {item.failure_code for item in repository.failures()} == {
        "penetration",
        "foot_slip",
    }
