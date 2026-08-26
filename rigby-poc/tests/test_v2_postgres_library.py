from __future__ import annotations

import os
import json
from dataclasses import replace
from uuid import uuid4

import psycopg
import pytest

from rigby_v2.flywheel.schemas import RetrievalFiltersV1, RetrievalIndex
from rigby_v2.library import (
    AnimationRecordInput,
    CertifiedLibraryService,
    DeferredLegacyIndexImporter,
    EmbeddingRouter,
    LegacyRepositorySink,
    PostgresLibraryRepository,
    PromotionProof,
    RetrievalQuery,
    standard_namespace_configs,
)
from rigby_v2.release import apply_rigby_v2_postgres_migrations


pytestmark = pytest.mark.skipif(
    os.getenv("RIGBY_TEST_POSTGRES") != "1",
    reason="set RIGBY_TEST_POSTGRES=1 with the local v2 database running",
)

DATABASE_URL = os.getenv(
    "RIGBY_V2_DATABASE_URL",
    "postgresql://rigby:rigby-local-only@127.0.0.1:54329/rigby",
)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class Passthrough:
    def embed(self, payload):
        return payload


def test_postgres_pgvector_release_filters_and_rrf() -> None:
    prefix = f"library-{uuid4()}"
    repository = PostgresLibraryRepository(DATABASE_URL)
    prior_active = repository.active_release()
    configs = standard_namespace_configs(
        cosmos_model_sha256=HASH_A,
        siglip2_model_sha256=HASH_B,
        descriptor_spec_sha256=HASH_C,
        cosmos_dimensions=4,
        siglip2_dimensions=4,
        descriptor_dimensions=4,
    )
    configs = {
        index: replace(config, namespace=f"{config.namespace}_{prefix}")
        for index, config in configs.items()
    }
    router = EmbeddingRouter(
        configs, {index: Passthrough() for index in RetrievalIndex}
    )
    created_ids: list[str] = []

    def next_id() -> str:
        value = str(uuid4())
        created_ids.append(value)
        return value

    service = CertifiedLibraryService(repository, router, id_factory=next_id)
    release_id = f"{prefix}-release"
    proof = PromotionProof(
        independently_certified=True,
        resimulated=True,
        independently_evaluated=True,
        certification_id=f"{prefix}-cert",
        resimulation_result_sha256=HASH_A,
        independent_evaluation_sha256=HASH_B,
        evaluator_id=f"{prefix}-judge",
    )

    def source(name: str, license_id: str = "postgres-test-license") -> AnimationRecordInput:
        return AnimationRecordInput(
            schema_version="2.0",
            split="train",
            prompt_text=name,
            program={"name": name},
            world={},
            motion={"phase": "action"},
            evidence={},
            labels={},
            evaluation={},
            provenance={"test_prefix": prefix},
            license_id=license_id,
            rig_id=f"{prefix}-rig",
            lineage_id=f"{prefix}-{name}",
            compact_example=f"compact {name}",
            object_affordances=("graspable",),
            limbs=("right_hand",),
            contact_requirements=("grasp",),
        )

    try:
        service.create_release(
            release_id,
            parent_release_id=(prior_active.release_id if prior_active else None),
        )
        for name, vector, license_id in (
            ("first", (1.0, 0.0, 0.0, 0.0), "postgres-test-license"),
            ("second", (0.8, 0.2, 0.0, 0.0), "postgres-test-license"),
            ("blocked", (1.0, 0.0, 0.0, 0.0), "forbidden-license"),
        ):
            candidate = service.submit_candidate(source(name, license_id))
            staged = service.promote_for_next_release(candidate.record_id, release_id, proof)
            service.index_record(staged.record_id, {index: vector for index in RetrievalIndex})
        service.freeze_release(release_id)
        service.activate_release(release_id)
        query_vector = (1.0, 0.0, 0.0, 0.0)
        response = service.retrieve(
            RetrievalQuery(
                prompt=query_vector,  # type: ignore[arg-type]
                structured_program=query_vector,
                task_space_motion=query_vector,
                video=query_vector,
                salient_keyframes=query_vector,
            ),
            RetrievalFiltersV1(
                release=release_id,
                allowed_licenses=("postgres-test-license",),
                schema_versions=("2.0",),
                rig_ids=(f"{prefix}-rig",),
                object_affordances=("graspable",),
                limbs=("right_hand",),
                contact_requirements=("grasp",),
            ),
            example_count=2,
        )
        assert len(response.result.selected_examples) == 2
        assert set(response.result.hits_by_index) == set(RetrievalIndex)
        assert all(len(hits) == 2 for hits in response.result.hits_by_index.values())
    finally:
        with psycopg.connect(DATABASE_URL) as connection, connection.transaction():
            connection.execute(
                "DELETE FROM rigby_v2.embeddings WHERE record_id = ANY(%s::uuid[])",
                (created_ids,),
            )
            connection.execute(
                """
                DELETE FROM rigby_v2.embeddings WHERE record_id IN (
                    SELECT record_id FROM rigby_v2.animation_records WHERE release_id = %s
                )
                """,
                (release_id,),
            )
            connection.execute(
                "DELETE FROM rigby_v2.animation_records WHERE release_id = %s",
                (release_id,),
            )
            connection.execute(
                "DELETE FROM rigby_v2.animation_records WHERE provenance->>'test_prefix' = %s",
                (prefix,),
            )
            connection.execute(
                "DELETE FROM rigby_v2.library_releases WHERE release_id = %s",
                (release_id,),
            )
            if prior_active:
                connection.execute(
                    "UPDATE rigby_v2.library_releases SET status = 'active' WHERE release_id = %s",
                    (prior_active.release_id,),
                )
            for config in configs.values():
                connection.execute(
                    """
                    DELETE FROM rigby_v2.embedding_namespaces
                    WHERE namespace = %s AND version = %s
                    """,
                    (config.namespace, config.version),
                )


def test_postgres_metadata_only_legacy_import_is_quarantined_and_idempotent(
    tmp_path,
) -> None:
    apply_rigby_v2_postgres_migrations(DATABASE_URL)
    assert apply_rigby_v2_postgres_migrations(DATABASE_URL) == ()
    source_id = f"legacy-canary-{uuid4().hex}"
    archive = tmp_path / "archive"
    folder = archive / source_id
    folder.mkdir(parents=True)
    entry = {
        "id": source_id,
        "prompt": "Guarded PostgreSQL metadata import fixture.",
        "intent": "gesture",
        "success": True,
        "metrics": {"max_penetration_m": 0.0},
    }
    (archive / "index.json").write_text(
        json.dumps({"schema_version": "1.0", "results": [entry]}),
        encoding="utf-8",
    )
    repository = PostgresLibraryRepository(DATABASE_URL)
    sink = LegacyRepositorySink(repository)
    importer = DeferredLegacyIndexImporter(archive)
    record_ids: list[str] = []
    try:
        first = importer.import_all(sink)
        second = importer.import_all(sink)
        record_ids.append(first[0].record_id)
        assert first == second
        assert first[0].split == "legacy"
        assert first[0].status.value == "legacy_candidate"
        assert first[0].evaluation["requires_hydration"] is True
        with psycopg.connect(DATABASE_URL) as connection:
            row = connection.execute(
                """
                SELECT split, status, release_id, evaluation, provenance
                FROM rigby_v2.animation_records WHERE record_id = %s
                """,
                (first[0].record_id,),
            ).fetchone()
            assert row[0] == "legacy"
            assert row[1] == "legacy_candidate"
            assert row[2] is None
            assert row[3]["requires_hydration"] is True
            assert row[4]["legacy_index_entry_sha256"]
            assert connection.execute(
                "SELECT count(*) FROM rigby_v2.embeddings WHERE record_id = %s",
                (first[0].record_id,),
            ).fetchone()[0] == 0
    finally:
        if record_ids:
            with psycopg.connect(DATABASE_URL) as connection:
                connection.execute(
                    "DELETE FROM rigby_v2.animation_records WHERE record_id = ANY(%s::uuid[])",
                    (record_ids,),
                )
