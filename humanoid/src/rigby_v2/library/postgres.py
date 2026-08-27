"""PostgreSQL/pgvector repository for the certified animation library."""

from __future__ import annotations

import json
from math import exp
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from rigby_v2.flywheel.schemas import RankedRetrievalHitV1, RetrievalFiltersV1

from .models import (
    AnimationRecord,
    AnimationStatus,
    DistanceMetric,
    EmbeddingNamespaceConfig,
    EmbeddingVector,
    HardNegativeRecord,
    ReleaseRecord,
    ReleaseStatus,
)


class PostgresLibraryRepository:
    def __init__(self, database_url: str) -> None:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgresLibraryRepository requires a PostgreSQL URL")
        self.database_url = database_url

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    @staticmethod
    def _record(row: dict[str, Any]) -> AnimationRecord:
        return AnimationRecord(
            record_id=str(row["record_id"]),
            schema_version=row["schema_version"],
            status=AnimationStatus(row["status"]),
            release_id=row["release_id"],
            split=row["split"],
            prompt=row["prompt"],
            program=row["program"],
            world=row["world"],
            motion=row["motion"],
            evidence=row["evidence"],
            labels=row["labels"],
            evaluation=row["evaluation"],
            provenance=row["provenance"],
            program_sha256=row["program_sha256"],
            world_sha256=row["world_sha256"],
            motion_sha256=row["motion_sha256"],
            parent_record_id=(str(row["parent_record_id"]) if row["parent_record_id"] else None),
        )

    @staticmethod
    def _release(row: dict[str, Any]) -> ReleaseRecord:
        return ReleaseRecord(
            release_id=row["release_id"],
            parent_release_id=row["parent_release_id"],
            status=ReleaseStatus(row["status"]),
            manifest_sha256=row["manifest_sha256"],
            manifest=row["manifest"],
        )

    def register_namespace(self, config: EmbeddingNamespaceConfig) -> None:
        with self._connect() as connection, connection.transaction():
            existing = connection.execute(
                """
                SELECT * FROM rigby_v2.embedding_namespaces
                WHERE namespace = %s AND version = %s FOR UPDATE
                """,
                (config.namespace, config.version),
            ).fetchone()
            values = (
                config.model_name,
                config.model_sha256,
                config.preprocessing_version,
                config.dimensions,
                config.distance_metric.value,
            )
            if existing is not None:
                observed = (
                    existing["model_name"],
                    existing["model_sha256"],
                    existing["preprocessing_version"],
                    int(existing["dimensions"]),
                    existing["distance_metric"],
                )
                if observed != values:
                    raise ValueError("embedding namespace version is immutable")
                return
            connection.execute(
                """
                INSERT INTO rigby_v2.embedding_namespaces (
                    namespace, version, model_name, model_sha256,
                    preprocessing_version, dimensions, distance_metric, active
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, true)
                """,
                (config.namespace, config.version, *values),
            )

    def create_release(self, release: ReleaseRecord) -> None:
        if release.status is not ReleaseStatus.BUILDING:
            raise ValueError("new releases must start building")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rigby_v2.library_releases (
                    release_id, parent_release_id, status, manifest
                ) VALUES (%s, %s, 'building', %s)
                """,
                (release.release_id, release.parent_release_id, Jsonb(release.manifest)),
            )

    def get_release(self, release_id: str) -> ReleaseRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM rigby_v2.library_releases WHERE release_id = %s",
                (release_id,),
            ).fetchone()
        return self._release(row) if row else None

    def active_release(self) -> ReleaseRecord | None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM rigby_v2.library_releases WHERE status = 'active'"
            ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("multiple active library releases")
        return self._release(rows[0]) if rows else None

    def freeze_release(self, release_id: str, manifest: dict, manifest_sha256: str) -> None:
        with self._connect() as connection, connection.transaction():
            row = connection.execute(
                "SELECT status FROM rigby_v2.library_releases WHERE release_id = %s FOR UPDATE",
                (release_id,),
            ).fetchone()
            if row is None or row["status"] != ReleaseStatus.BUILDING.value:
                raise ValueError("only a building release can be frozen")
            invalid = connection.execute(
                """
                SELECT record_id FROM rigby_v2.animation_records
                WHERE release_id = %s AND status = 'staged' AND NOT (
                    evaluation->>'independently_certified' = 'true'
                    AND evaluation->>'resimulated' = 'true'
                    AND evaluation->>'independently_evaluated' = 'true'
                    AND COALESCE(evaluation->>'certification_id', '') <> ''
                    AND COALESCE(evaluation->>'evaluator_id', '') <> ''
                    AND COALESCE(evaluation->>'resimulation_result_sha256', '') ~ '^[0-9a-f]{64}$'
                    AND COALESCE(evaluation->>'independent_evaluation_sha256', '') ~ '^[0-9a-f]{64}$'
                ) LIMIT 1
                """,
                (release_id,),
            ).fetchone()
            if invalid is not None:
                raise ValueError(
                    "release contains a staged record without independent promotion proof"
                )
            connection.execute(
                """
                UPDATE rigby_v2.animation_records SET status = 'certified', updated_at = now()
                WHERE release_id = %s AND status = 'staged'
                """,
                (release_id,),
            )
            connection.execute(
                """
                UPDATE rigby_v2.library_releases
                SET status = 'frozen', manifest = %s, manifest_sha256 = %s, frozen_at = now()
                WHERE release_id = %s
                """,
                (Jsonb(manifest), manifest_sha256, release_id),
            )

    def activate_release(self, release_id: str) -> None:
        with self._connect() as connection, connection.transaction():
            row = connection.execute(
                "SELECT status FROM rigby_v2.library_releases WHERE release_id = %s FOR UPDATE",
                (release_id,),
            ).fetchone()
            if row is None or row["status"] != ReleaseStatus.FROZEN.value:
                raise ValueError("only a frozen release can become active")
            connection.execute(
                "UPDATE rigby_v2.library_releases SET status = 'retired' WHERE status = 'active'"
            )
            connection.execute(
                "UPDATE rigby_v2.library_releases SET status = 'active' WHERE release_id = %s",
                (release_id,),
            )

    def insert_record(self, record: AnimationRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rigby_v2.animation_records (
                    record_id, schema_version, status, release_id, split,
                    prompt, program, world, motion, evidence, labels, evaluation,
                    provenance, program_sha256, world_sha256, motion_sha256,
                    parent_record_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                """,
                (
                    record.record_id,
                    record.schema_version,
                    record.status.value,
                    record.release_id,
                    record.split,
                    Jsonb(record.prompt),
                    Jsonb(record.program),
                    Jsonb(record.world),
                    Jsonb(record.motion),
                    Jsonb(record.evidence),
                    Jsonb(record.labels),
                    Jsonb(record.evaluation),
                    Jsonb(record.provenance),
                    record.program_sha256,
                    record.world_sha256,
                    record.motion_sha256,
                    record.parent_record_id,
                ),
            )

    def replace_legacy_record(self, record_id: str, record: AnimationRecord) -> None:
        if (
            record.record_id != record_id
            or record.status is not AnimationStatus.LEGACY_CANDIDATE
            or record.release_id is not None
        ):
            raise ValueError("hydration replacement must remain the same legacy candidate")
        with self._connect() as connection, connection.transaction():
            existing = connection.execute(
                """
                SELECT status, release_id, provenance FROM rigby_v2.animation_records
                WHERE record_id = %s FOR UPDATE
                """,
                (record_id,),
            ).fetchone()
            if (
                existing is None
                or existing["status"] != AnimationStatus.LEGACY_CANDIDATE.value
                or existing["release_id"] is not None
                or existing["provenance"].get("lineage_id")
                != record.provenance.get("lineage_id")
            ):
                raise ValueError("only the same quarantined legacy record may be hydrated")
            embedded = connection.execute(
                "SELECT 1 FROM rigby_v2.embeddings WHERE record_id = %s LIMIT 1",
                (record_id,),
            ).fetchone()
            if embedded is not None:
                raise ValueError("a legacy record with embeddings cannot be hydrated in place")
            connection.execute(
                """
                UPDATE rigby_v2.animation_records SET
                    schema_version = %s, split = %s, prompt = %s, program = %s,
                    world = %s, motion = %s, evidence = %s, labels = %s,
                    evaluation = %s, provenance = %s, program_sha256 = %s,
                    world_sha256 = %s, motion_sha256 = %s, parent_record_id = %s,
                    updated_at = now()
                WHERE record_id = %s
                """,
                (
                    record.schema_version,
                    record.split,
                    Jsonb(record.prompt),
                    Jsonb(record.program),
                    Jsonb(record.world),
                    Jsonb(record.motion),
                    Jsonb(record.evidence),
                    Jsonb(record.labels),
                    Jsonb(record.evaluation),
                    Jsonb(record.provenance),
                    record.program_sha256,
                    record.world_sha256,
                    record.motion_sha256,
                    record.parent_record_id,
                    record_id,
                ),
            )

    def get_record(self, record_id: str) -> AnimationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM rigby_v2.animation_records WHERE record_id = %s",
                (record_id,),
            ).fetchone()
        return self._record(row) if row else None

    def find_legacy_by_lineage(self, lineage_id: str) -> AnimationRecord | None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM rigby_v2.animation_records
                WHERE status = 'legacy_candidate'
                  AND provenance->>'lineage_id' = %s
                ORDER BY record_id LIMIT 2
                """,
                (lineage_id,),
            ).fetchall()
        if len(rows) > 1:
            raise RuntimeError("legacy lineage uniqueness was violated")
        return self._record(rows[0]) if rows else None

    def records_for_release(self, release_id: str) -> tuple[AnimationRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM rigby_v2.animation_records
                WHERE release_id = %s ORDER BY record_id
                """,
                (release_id,),
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def copy_embeddings(self, source_record_id: str, target_record_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rigby_v2.embeddings (
                    record_id, namespace, namespace_version, segment_id,
                    embedding, metadata
                )
                SELECT %s, namespace, namespace_version, segment_id, embedding, metadata
                FROM rigby_v2.embeddings WHERE record_id = %s
                """,
                (target_record_id, source_record_id),
            )

    def put_embedding(self, embedding: EmbeddingVector) -> None:
        vector = json.dumps(list(embedding.values), separators=(",", ":"))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO rigby_v2.embeddings (
                    record_id, namespace, namespace_version, segment_id,
                    embedding, metadata
                ) VALUES (%s, %s, %s, %s, %s::vector, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    embedding.record_id,
                    embedding.namespace,
                    embedding.namespace_version,
                    embedding.segment_id,
                    vector,
                    Jsonb(embedding.metadata),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("embedding segment is immutable")

    def embedding_manifest(self, record_id: str) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT namespace, namespace_version, segment_id,
                       embedding::text AS values, metadata
                FROM rigby_v2.embeddings
                WHERE record_id = %s
                ORDER BY namespace, namespace_version, segment_id
                """,
                (record_id,),
            ).fetchall()
        return tuple(
            {
                "namespace": row["namespace"],
                "namespace_version": row["namespace_version"],
                "segment_id": row["segment_id"],
                "values": json.loads(row["values"]),
                "metadata": row["metadata"],
            }
            for row in rows
        )

    def rank(
        self,
        config: EmbeddingNamespaceConfig,
        query: tuple[float, ...],
        filters: RetrievalFiltersV1,
        *,
        limit: int,
    ) -> tuple[RankedRetrievalHitV1, ...]:
        operator = {
            DistanceMetric.COSINE: "<=>",
            DistanceMetric.INNER_PRODUCT: "<#>",
            DistanceMetric.L2: "<->",
        }[config.distance_metric]
        vector = json.dumps(list(query), separators=(",", ":"))
        lineage_sql = ""
        params: list[Any] = [
            vector,
            config.namespace,
            config.version,
            filters.release,
            list(filters.schema_versions),
            list(filters.allowed_licenses),
            list(filters.rig_ids),
            list(filters.excluded_splits),
            Jsonb(list(filters.object_affordances)),
            Jsonb(list(filters.limbs)),
            Jsonb(list(filters.contact_requirements)),
        ]
        if filters.excluded_lineage:
            lineage_sql = "AND NOT (COALESCE(a.provenance->>'lineage_id', '') = ANY(%s))"
            params.append(list(filters.excluded_lineage))
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT a.record_id, e.embedding {operator} %s::vector AS raw_distance
                FROM rigby_v2.animation_records AS a
                JOIN rigby_v2.embeddings AS e ON e.record_id = a.record_id
                WHERE e.namespace = %s AND e.namespace_version = %s
                  AND e.segment_id = 'global'
                  AND a.status = 'certified' AND a.release_id = %s
                  AND a.schema_version = ANY(%s)
                  AND a.provenance->>'license_id' = ANY(%s)
                  AND a.world->>'rig_id' = ANY(%s)
                  AND NOT (a.split = ANY(%s))
                  AND COALESCE(a.world->'object_affordances', '[]'::jsonb) @> %s::jsonb
                  AND COALESCE(a.labels->'limbs', '[]'::jsonb) @> %s::jsonb
                  AND COALESCE(a.labels->'contact_requirements', '[]'::jsonb) @> %s::jsonb
                  {lineage_sql}
                ORDER BY raw_distance, a.record_id
                LIMIT %s
                """,
                params,
            ).fetchall()
        hits = []
        for rank, row in enumerate(rows, start=1):
            raw = float(row["raw_distance"])
            distance = 1.0 / (1.0 + exp(-raw)) if config.distance_metric is DistanceMetric.INNER_PRODUCT else max(0.0, raw)
            hits.append(
                RankedRetrievalHitV1(
                    record_id=str(row["record_id"]),
                    index=config.index,
                    rank=rank,
                    distance=distance,
                    release=filters.release,
                )
            )
        return tuple(hits)

    def insert_failure(self, failure: HardNegativeRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rigby_v2.failure_records (
                    failure_id, schema_version, candidate_record_id, stage,
                    failure_code, failed_predicate, measurements, artifacts, provenance
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    failure.failure_id,
                    failure.schema_version,
                    failure.candidate_record_id,
                    failure.stage,
                    failure.failure_code,
                    failure.failed_predicate,
                    Jsonb(failure.measurements),
                    Jsonb(failure.artifacts),
                    Jsonb(failure.provenance),
                ),
            )

    def failures(self) -> tuple[HardNegativeRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM rigby_v2.failure_records ORDER BY failure_id"
            ).fetchall()
        return tuple(
            HardNegativeRecord(
                failure_id=str(row["failure_id"]),
                schema_version=row["schema_version"],
                stage=row["stage"],
                failure_code=row["failure_code"],
                failed_predicate=row["failed_predicate"],
                measurements=row["measurements"],
                artifacts=row["artifacts"],
                provenance=row["provenance"],
                candidate_record_id=(str(row["candidate_record_id"]) if row["candidate_record_id"] else None),
            )
            for row in rows
        )
