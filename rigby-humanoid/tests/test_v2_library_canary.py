from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest

from rigby_v2.flywheel.schemas import RetrievalFiltersV1, RetrievalIndex
from rigby_v2.hashing import content_hash, hash_file
from rigby_v2.library import (
    AnimationRecord,
    AnimationRecordInput,
    AnimationStatus,
    DistanceMetric,
    EmbeddingNamespaceConfig,
    EmbeddingVector,
    InMemoryLibraryRepository,
    PromotionProof,
    ReleaseRecord,
    ReleaseStatus,
    audit_building_release_retrieval,
    staging_filter_rejection,
)
from rigby_v2.library.models import staged_copy

pytestmark = pytest.mark.fast


def _record(status: AnimationStatus = AnimationStatus.CANDIDATE) -> AnimationRecord:
    source = AnimationRecordInput(
        schema_version="2.0",
        split="production",
        prompt_text="Press the button and release it.",
        program={"action": "button_press"},
        world={"object_pack": "lever_button"},
        motion={"phases": ["approach", "contact", "release"]},
        evidence={"trace_sha256": "a" * 64},
        labels={},
        evaluation={"certified": True},
        provenance={},
        license_id="Apache-2.0",
        rig_id="rigby-canonical-human-medium",
        lineage_id="button-press-canary",
        compact_example="approach_clear physical_button_contact button_pressed release_clear",
        object_affordances=("push_button",),
        limbs=("left_arm", "left_hand"),
        contact_requirements=("button_contact",),
    )
    return AnimationRecord.from_input("candidate", source, status=status)


def _staged_record() -> AnimationRecord:
    return staged_copy(
        _record(),
        record_id="staged",
        release_id="button-canary-v1",
        proof=PromotionProof(
            independently_certified=True,
            resimulated=True,
            independently_evaluated=True,
            certification_id="button-certification",
            resimulation_result_sha256="b" * 64,
            independent_evaluation_sha256="c" * 64,
            evaluator_id="rigby-v2-certification-engine",
        ),
    )


def _filters(**changes: object) -> RetrievalFiltersV1:
    values = {
        "release": "button-canary-v1",
        "allowed_licenses": ("Apache-2.0",),
        "schema_versions": ("2.0",),
        "rig_ids": ("rigby-canonical-human-medium",),
        "object_affordances": ("push_button",),
        "limbs": ("left_arm", "left_hand"),
        "contact_requirements": ("button_contact",),
    }
    values.update(changes)
    return RetrievalFiltersV1(**values)


def _config(index: RetrievalIndex) -> EmbeddingNamespaceConfig:
    return EmbeddingNamespaceConfig(
        index=index,
        namespace=f"test_{index.value}",
        version="1.0",
        model_name="test/model",
        model_sha256=content_hash({"index": index.value}),
        preprocessing_version="test-v1",
        dimensions=3,
        distance_metric=DistanceMetric.COSINE,
    )


def _repository() -> tuple[
    InMemoryLibraryRepository,
    dict[RetrievalIndex, EmbeddingNamespaceConfig],
]:
    repository = InMemoryLibraryRepository()
    repository.create_release(
        ReleaseRecord("button-canary-v1", None, ReleaseStatus.BUILDING)
    )
    repository.insert_record(_staged_record())
    indexes = (RetrievalIndex.KEYFRAME, RetrievalIndex.PROGRAM, RetrievalIndex.MOTION)
    configs = {index: _config(index) for index in indexes}
    for config in configs.values():
        repository.register_namespace(config)
        repository.put_embedding(
            EmbeddingVector(
                record_id="staged",
                index=config.index,
                namespace=config.namespace,
                namespace_version=config.version,
                values=(1.0, 0.0, 0.0),
            )
        )
    return repository, configs


def test_staging_audit_applies_hard_filters_and_rrf_without_promoting() -> None:
    repository, configs = _repository()
    query = {index: (1.0, 0.0, 0.0) for index in configs}

    audit = audit_building_release_retrieval(repository, configs, query, _filters())

    assert audit.selected_record_ids == ("staged",)
    assert audit.reciprocal_rank_scores == {"staged": pytest.approx(3 / 61)}
    assert set(audit.missing_indexes) == {RetrievalIndex.TEXT, RetrievalIndex.VIDEO}
    assert audit.promotable is False
    assert repository.get_release("button-canary-v1").status is ReleaseStatus.BUILDING
    assert repository.get_record("staged").status is AnimationStatus.STAGED
    # The production path continues to hide the staged record.
    assert repository.rank(
        configs[RetrievalIndex.KEYFRAME],
        query[RetrievalIndex.KEYFRAME],
        _filters(),
        limit=3,
    ) == ()


@pytest.mark.parametrize(
    ("filters", "reason"),
    [
        (_filters(allowed_licenses=("CC-BY-4.0",)), "license_not_allowed"),
        (_filters(excluded_lineage=("button-press-canary",)), "lineage_excluded"),
        (_filters(limbs=("right_hand",)), "limb_missing"),
    ],
)
def test_staging_audit_rejects_incompatible_hard_filters(
    filters: RetrievalFiltersV1,
    reason: str,
) -> None:
    repository, configs = _repository()
    staged = repository.get_record("staged")
    assert staged is not None
    assert staging_filter_rejection(staged, filters) == reason

    audit = audit_building_release_retrieval(
        repository,
        configs,
        {index: (1.0, 0.0, 0.0) for index in configs},
        filters,
    )
    assert audit.selected_record_ids == ()


def test_staging_filter_rejects_candidate_and_missing_proof() -> None:
    assert staging_filter_rejection(_record(), _filters()) == "status_not_staged"
    incomplete = replace(
        _record(),
        record_id="incomplete",
        status=AnimationStatus.STAGED,
        release_id="button-canary-v1",
    )
    assert staging_filter_rejection(incomplete, _filters()) == "missing_promotion_proof"


def test_sealed_button_canary_is_real_but_staging_only() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "assets/v2/library/button_rag_canary_staging.v1.json"
    seal = evidence.with_suffix(evidence.suffix + ".sha256")
    assert hash_file(evidence) == seal.read_text(encoding="ascii").strip()
    document = json.loads(evidence.read_text(encoding="utf-8"))

    assert document["status"] == "verified_staging_only"
    assert document["release"]["status"] == "building"
    assert document["release"]["freeze_allowed"] is False
    assert document["release"]["activation_allowed"] is False
    assert document["retrieval_audit"]["promotable"] is False
    assert set(document["retrieval_audit"]["missing_indexes"]) == {"text", "video"}
    assert document["retrieval_audit"]["hard_filters"]["production_rank_count"] == 0
    assert document["retrieval_audit"]["hard_filters"]["wrong_license_selected"] == []
    assert document["retrieval_audit"]["hard_filters"]["excluded_lineage_selected"] == []
    assert document["cosmos"]["status"] == "unavailable"
    assert document["cosmos"]["substitution_used"] is False
    assert document["siglip2"]["network_at_runtime"] is False
    assert document["siglip2"]["repeat_count"] == 2
    assert document["siglip2"]["maximum_repeat_delta"] == 0.0
    assert document["certification"]["baseline"]["outcome"] == "certified"
    assert document["certification"]["independent"]["outcome"] == "certified"
    assert document["certification"]["baseline"]["trace_sha256"] == (
        document["certification"]["independent"]["trace_sha256"]
    )
    assert len(document["certification"]["robustness"]) == 5
    assert all(
        item["certification"]["outcome"] == "certified"
        for item in document["certification"]["robustness"]
    )
    assert set(document["embeddings"]) == {
        "salient_keyframe",
        "structured_program",
        "task_space_motion",
    }
    assert all(
        abs(item["l2_norm"] - 1.0) < 1e-12
        for item in document["embeddings"].values()
    )
    metric = document["rag_context_coverage_canary"]
    assert (metric["rag_off_hits"], metric["rag_on_hits"], metric["denominator"]) == (
        0,
        5,
        5,
    )
    assert "not the Goal 10" in metric["scope"]


@pytest.mark.skipif(
    "RIGBY_V2_CANARY_DATABASE_URL" not in os.environ,
    reason="live canary database verification is explicitly opt-in",
)
def test_live_button_canary_database_inventory() -> None:
    database_url = os.environ["RIGBY_V2_CANARY_DATABASE_URL"]
    with psycopg.connect(database_url) as connection:
        assert connection.execute(
            "SELECT status FROM rigby_v2.library_releases WHERE release_id = %s",
            ("button-rag-canary-v1-building",),
        ).fetchone() == ("building",)
        assert connection.execute(
            """SELECT status, count(*) FROM rigby_v2.animation_records
               GROUP BY status ORDER BY status"""
        ).fetchall() == [
            ("candidate", 1),
            ("legacy_candidate", 6354),
            ("staged", 1),
        ]
        assert connection.execute(
            "SELECT count(*) FROM rigby_v2.embedding_namespaces"
        ).fetchone() == (3,)
        assert connection.execute(
            "SELECT count(*) FROM rigby_v2.embeddings"
        ).fetchone() == (3,)
        assert connection.execute(
            """SELECT count(*) FROM rigby_v2.animation_records
               WHERE status = 'certified'"""
        ).fetchone() == (0,)
