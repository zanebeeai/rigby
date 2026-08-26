from __future__ import annotations

from dataclasses import replace
from itertools import count

import pytest

from rigby_v2.flywheel.schemas import RetrievalFiltersV1, RetrievalIndex
from rigby_v2.library import (
    AnimationRecord,
    AnimationRecordInput,
    AnimationStatus,
    CertifiedLibraryService,
    EmbeddingRouter,
    HardNegativeRecord,
    InMemoryLibraryRepository,
    InsufficientCertifiedExamples,
    PromotionProof,
    ReleaseStatus,
    RetrievalQuery,
    standard_namespace_configs,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class PassthroughEmbedder:
    def embed(self, payload):
        return payload


def _router() -> EmbeddingRouter:
    configs = standard_namespace_configs(
        cosmos_model_sha256=HASH_A,
        siglip2_model_sha256=HASH_B,
        descriptor_spec_sha256=HASH_C,
        cosmos_dimensions=4,
        siglip2_dimensions=4,
        descriptor_dimensions=4,
    )
    return EmbeddingRouter(
        configs,
        {index: PassthroughEmbedder() for index in RetrievalIndex},
    )


def _service() -> tuple[CertifiedLibraryService, InMemoryLibraryRepository]:
    ids = count(1)
    repository = InMemoryLibraryRepository()
    service = CertifiedLibraryService(
        repository,
        _router(),
        id_factory=lambda: f"00000000-0000-0000-0000-{next(ids):012d}",
    )
    return service, repository


def _input(
    name: str,
    *,
    license_id: str = "research-ok",
    split: str = "train",
    lineage: str | None = None,
    rig_id: str = "rig-a",
    schema_version: str = "2.0",
    affordances: tuple[str, ...] = ("graspable", "hinged"),
    limbs: tuple[str, ...] = ("left_hand", "right_hand"),
    contacts: tuple[str, ...] = ("bimanual",),
) -> AnimationRecordInput:
    return AnimationRecordInput(
        schema_version=schema_version,
        split=split,
        prompt_text=f"perform {name}",
        program={"schema_version": "2.0", "name": name},
        world={"scene": "workbench"},
        motion={"phases": ["setup", "action", "settle"]},
        evidence={"trace": f"trace-{name}"},
        labels={"family": "tool"},
        evaluation={"rubric": 4.8},
        provenance={"source": "unit"},
        license_id=license_id,
        rig_id=rig_id,
        lineage_id=lineage or f"lineage-{name}",
        compact_example=f"Prompt: perform {name}\nProgram: setup, action, settle",
        object_affordances=affordances,
        limbs=limbs,
        contact_requirements=contacts,
    )


def _proof(*, valid: bool = True) -> PromotionProof:
    return PromotionProof(
        independently_certified=valid,
        resimulated=valid,
        independently_evaluated=valid,
        certification_id="cert-independent",
        resimulation_result_sha256=HASH_A,
        independent_evaluation_sha256=HASH_B,
        evaluator_id="judge-independent",
    )


def _payload(vector):
    return {index: vector for index in RetrievalIndex}


def _query(vector=(1.0, 0.0, 0.0, 0.0)) -> RetrievalQuery:
    return RetrievalQuery(
        prompt=vector,
        structured_program=vector,
        task_space_motion=vector,
        video=vector,
        salient_keyframes=vector,
    )


def _filters(*, release="release-1", excluded_lineage=()) -> RetrievalFiltersV1:
    return RetrievalFiltersV1(
        release=release,
        allowed_licenses=("research-ok",),
        schema_versions=("2.0",),
        rig_ids=("rig-a",),
        object_affordances=("graspable",),
        limbs=("left_hand", "right_hand"),
        contact_requirements=("bimanual",),
        excluded_splits=("test", "benchmark"),
        excluded_lineage=excluded_lineage,
    )


def _add_certified(
    service: CertifiedLibraryService,
    name: str,
    vector: tuple[float, ...],
    **record_options,
):
    candidate = service.submit_candidate(_input(name, **record_options))
    staged = service.promote_for_next_release(candidate.record_id, "release-1", _proof())
    service.index_record(staged.record_id, _payload(vector))
    return candidate, staged


def test_legacy_is_never_positive_and_promotion_requires_three_independent_gates() -> None:
    service, repository = _service()
    legacy = service.import_legacy(_input("legacy"))
    service.create_release("release-1", parent_release_id=None)

    assert legacy.status is AnimationStatus.LEGACY_CANDIDATE
    assert legacy.release_id is None
    with pytest.raises(ValueError, match="re-simulation"):
        service.promote_for_next_release(legacy.record_id, "release-1", _proof(valid=False))
    assert repository.get_record(legacy.record_id).status is AnimationStatus.LEGACY_CANDIDATE
    bypass = AnimationRecord.from_input(
        "90000000-0000-0000-0000-000000000001",
        _input("attempted-bypass"),
        status=AnimationStatus.STAGED,
        release_id="release-1",
    )
    repository.insert_record(bypass)
    with pytest.raises(ValueError, match="promotion proof"):
        service.freeze_release("release-1")


def test_five_independent_indexes_fuse_to_three_compact_certified_examples() -> None:
    service, repository = _service()
    service.create_release("release-1", parent_release_id=None)
    records = [
        _add_certified(service, "alpha", (1.0, 0.0, 0.0, 0.0))[1],
        _add_certified(service, "beta", (0.9, 0.1, 0.0, 0.0))[1],
        _add_certified(service, "gamma", (0.8, 0.2, 0.0, 0.0))[1],
    ]
    service.freeze_release("release-1")
    frozen = repository.get_release("release-1")
    assert frozen is not None
    assert len(frozen.manifest["embedding_namespaces"]) == 5
    assert all(len(item["embeddings"]) == 5 for item in frozen.manifest["records"])
    service.activate_release("release-1")

    response = service.retrieve(_query(), _filters())

    assert repository.active_release().status is ReleaseStatus.ACTIVE
    assert set(response.result.hits_by_index) == set(RetrievalIndex)
    assert len(response.result.selected_examples) == 3
    assert response.result.selected_examples == tuple(record.record_id for record in records)
    assert tuple(example.record_id for example in response.examples) == response.result.selected_examples
    assert all(len(example.context) <= 4_000 for example in response.examples)
    assert all(
        hit.release == "release-1"
        for hits in response.result.hits_by_index.values()
        for hit in hits
    )


def test_reciprocal_rank_fusion_combines_channel_specific_rankings() -> None:
    service, _ = _service()
    service.create_release("release-1", parent_release_id=None)
    alpha_source = service.submit_candidate(_input("alpha-channel-specialist"))
    beta_source = service.submit_candidate(_input("beta-four-channel-winner"))
    gamma_source = service.submit_candidate(_input("gamma-baseline"))
    alpha = service.promote_for_next_release(alpha_source.record_id, "release-1", _proof())
    beta = service.promote_for_next_release(beta_source.record_id, "release-1", _proof())
    gamma = service.promote_for_next_release(gamma_source.record_id, "release-1", _proof())
    exact = (1.0, 0.0, 0.0, 0.0)
    second = (0.8, 0.2, 0.0, 0.0)
    third = (0.6, 0.4, 0.0, 0.0)
    service.index_record(
        alpha.record_id,
        {index: (exact if index is RetrievalIndex.TEXT else second) for index in RetrievalIndex},
    )
    service.index_record(
        beta.record_id,
        {index: (second if index is RetrievalIndex.TEXT else exact) for index in RetrievalIndex},
    )
    service.index_record(gamma.record_id, _payload(third))
    service.freeze_release("release-1")
    service.activate_release("release-1")

    response = service.retrieve(_query(exact), _filters())

    assert response.result.hits_by_index[RetrievalIndex.TEXT][0].record_id == alpha.record_id
    assert response.result.hits_by_index[RetrievalIndex.VIDEO][0].record_id == beta.record_id
    assert response.result.selected_examples[0] == beta.record_id


def test_hard_filters_run_before_ranking_against_license_split_lineage_and_rig_attacks() -> None:
    service, _ = _service()
    service.create_release("release-1", parent_release_id=None)
    allowed = [
        _add_certified(service, "allowed-a", (0.8, 0.2, 0.0, 0.0))[1],
        _add_certified(service, "allowed-b", (0.7, 0.3, 0.0, 0.0))[1],
    ]
    blocked = [
        _add_certified(service, "license-leak", (1.0, 0.0, 0.0, 0.0), license_id="forbidden-commercial")[1],
        _add_certified(service, "test-leak", (1.0, 0.0, 0.0, 0.0), split="test")[1],
        _add_certified(service, "lineage-leak", (1.0, 0.0, 0.0, 0.0), lineage="heldout-family")[1],
        _add_certified(service, "rig-leak", (1.0, 0.0, 0.0, 0.0), rig_id="rig-b")[1],
        _add_certified(service, "schema-leak", (1.0, 0.0, 0.0, 0.0), schema_version="1.0")[1],
        _add_certified(service, "affordance-leak", (1.0, 0.0, 0.0, 0.0), affordances=("pressable",))[1],
        _add_certified(service, "limb-leak", (1.0, 0.0, 0.0, 0.0), limbs=("right_hand",))[1],
        _add_certified(service, "contact-leak", (1.0, 0.0, 0.0, 0.0), contacts=("single_contact",))[1],
    ]
    legacy = service.import_legacy(_input("uncertified-leak"))
    service.index_record(legacy.record_id, _payload((1.0, 0.0, 0.0, 0.0)))
    blocked.append(legacy)
    service.freeze_release("release-1")
    service.activate_release("release-1")

    response = service.retrieve(
        _query(),
        _filters(excluded_lineage=("heldout-family",)),
        example_count=2,
    )

    selected = set(response.result.selected_examples)
    assert selected == {record.record_id for record in allowed}
    assert selected.isdisjoint(record.record_id for record in blocked)
    for hits in response.result.hits_by_index.values():
        assert {hit.record_id for hit in hits} == selected


def test_n_plus_one_winner_and_failures_cannot_leak_into_active_release() -> None:
    service, repository = _service()
    service.create_release("release-1", parent_release_id=None)
    _add_certified(service, "base-a", (0.8, 0.2, 0.0, 0.0))
    _add_certified(service, "base-b", (0.7, 0.3, 0.0, 0.0))
    service.freeze_release("release-1")
    service.activate_release("release-1")

    service.create_release("release-2", parent_release_id="release-1")
    winner = service.submit_candidate(_input("current-cycle-winner"))
    staged_winner = service.promote_for_next_release(winner.record_id, "release-2", _proof())
    service.index_record(staged_winner.record_id, _payload((1.0, 0.0, 0.0, 0.0)))
    service.record_failure(
        HardNegativeRecord(
            failure_id="10000000-0000-0000-0000-000000000001",
            schema_version="1.0",
            stage="certification",
            failure_code="penetration",
            failed_predicate="max_penetration",
            measurements={"penetration_m": 0.01},
            artifacts={},
            provenance={"cycle": "current"},
            candidate_record_id=winner.record_id,
        )
    )

    active_response = service.retrieve(_query(), _filters(), example_count=2)
    assert staged_winner.record_id not in active_response.result.selected_examples
    assert staged_winner.record_id not in {
        hit.record_id for hits in active_response.result.hits_by_index.values() for hit in hits
    }
    assert repository.failures()[0].candidate_record_id == winner.record_id
    with pytest.raises(ValueError, match="active release"):
        service.retrieve(_query(), _filters(release="release-2"), example_count=2)

    service.freeze_release("release-2")
    service.activate_release("release-2")
    next_response = service.retrieve(_query(), _filters(release="release-2"))
    assert staged_winner.record_id in next_response.result.selected_examples
    assert len(next_response.result.selected_examples) == 3
    assert repository.get_release("release-1").status is ReleaseStatus.RETIRED


def test_namespace_versions_are_immutable_and_too_few_examples_fail_closed() -> None:
    service, repository = _service()
    config = service.embeddings.configs[RetrievalIndex.TEXT]
    with pytest.raises(ValueError, match="immutable"):
        repository.register_namespace(replace(config, dimensions=5))

    service.create_release("release-1", parent_release_id=None)
    _add_certified(service, "only-one", (1.0, 0.0, 0.0, 0.0))
    service.freeze_release("release-1")
    service.activate_release("release-1")
    with pytest.raises(InsufficientCertifiedExamples):
        service.retrieve(_query(), _filters(), example_count=2)


def test_deterministic_descriptor_channels_do_not_download_weights() -> None:
    from rigby_v2.library import DeterministicDescriptorEmbedder

    embedder = DeterministicDescriptorEmbedder(32)
    payload = {"phases": ["setup", "contact", "release"], "contacts": ["thumb-object"]}
    assert tuple(embedder.embed(payload)) == tuple(embedder.embed(payload))
    assert len(embedder.embed(payload)) == 32
