"""Deterministic in-memory repository used by unit tests and offline tooling."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

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
    has_promotion_proof,
)


def _eligible(record: AnimationRecord, filters: RetrievalFiltersV1) -> bool:
    if record.status is not AnimationStatus.CERTIFIED or record.release_id != filters.release:
        return False
    if record.provenance.get("license_id") not in filters.allowed_licenses:
        return False
    if record.schema_version not in filters.schema_versions:
        return False
    if record.world.get("rig_id") not in filters.rig_ids:
        return False
    if record.split in filters.excluded_splits:
        return False
    if record.provenance.get("lineage_id") in filters.excluded_lineage:
        return False
    checks = (
        (filters.object_affordances, record.world.get("object_affordances", [])),
        (filters.limbs, record.labels.get("limbs", [])),
        (filters.contact_requirements, record.labels.get("contact_requirements", [])),
    )
    return all(set(required) <= set(available) for required, available in checks)


def _distance(metric: DistanceMetric, left: np.ndarray, right: np.ndarray) -> float:
    if metric is DistanceMetric.L2:
        return float(np.linalg.norm(left - right))
    dot = float(np.dot(left, right))
    if metric is DistanceMetric.INNER_PRODUCT:
        return float(1.0 / (1.0 + np.exp(dot)))
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    return float(1.0 - dot / (left_norm * right_norm))


class InMemoryLibraryRepository:
    def __init__(self) -> None:
        self.namespaces: dict[tuple[str, str], EmbeddingNamespaceConfig] = {}
        self.releases: dict[str, ReleaseRecord] = {}
        self.records: dict[str, AnimationRecord] = {}
        self.embeddings: dict[tuple[str, str, str, str], EmbeddingVector] = {}
        self._failures: dict[str, HardNegativeRecord] = {}

    def register_namespace(self, config: EmbeddingNamespaceConfig) -> None:
        key = (config.namespace, config.version)
        existing = self.namespaces.get(key)
        if existing is not None and existing != config:
            raise ValueError("embedding namespace version is immutable")
        self.namespaces[key] = config

    def create_release(self, release: ReleaseRecord) -> None:
        if release.release_id in self.releases:
            raise ValueError("release already exists")
        if release.status is not ReleaseStatus.BUILDING:
            raise ValueError("new releases must start building")
        if release.parent_release_id and release.parent_release_id not in self.releases:
            raise ValueError("parent release does not exist")
        self.releases[release.release_id] = release

    def get_release(self, release_id: str) -> ReleaseRecord | None:
        return self.releases.get(release_id)

    def active_release(self) -> ReleaseRecord | None:
        active = [release for release in self.releases.values() if release.status is ReleaseStatus.ACTIVE]
        if len(active) > 1:
            raise RuntimeError("multiple active releases")
        return active[0] if active else None

    def freeze_release(self, release_id: str, manifest: dict, manifest_sha256: str) -> None:
        release = self.releases[release_id]
        if release.status is not ReleaseStatus.BUILDING:
            raise ValueError("only a building release can be frozen")
        staged = [
            record
            for record in self.records.values()
            if record.release_id == release_id and record.status is AnimationStatus.STAGED
        ]
        if any(not has_promotion_proof(record) for record in staged):
            raise ValueError("release contains a staged record without independent promotion proof")
        self.releases[release_id] = replace(
            release,
            status=ReleaseStatus.FROZEN,
            manifest=dict(manifest),
            manifest_sha256=manifest_sha256,
        )
        for record in staged:
            self.records[record.record_id] = replace(record, status=AnimationStatus.CERTIFIED)

    def activate_release(self, release_id: str) -> None:
        release = self.releases[release_id]
        if release.status is not ReleaseStatus.FROZEN:
            raise ValueError("only a frozen release can become active")
        for active_id, active in tuple(self.releases.items()):
            if active.status is ReleaseStatus.ACTIVE:
                self.releases[active_id] = replace(active, status=ReleaseStatus.RETIRED)
        self.releases[release_id] = replace(release, status=ReleaseStatus.ACTIVE)

    def insert_record(self, record: AnimationRecord) -> None:
        if record.record_id in self.records:
            raise ValueError("animation records are immutable")
        self.records[record.record_id] = record

    def replace_legacy_record(self, record_id: str, record: AnimationRecord) -> None:
        existing = self.records.get(record_id)
        if (
            existing is None
            or existing.status is not AnimationStatus.LEGACY_CANDIDATE
            or record.record_id != record_id
            or record.status is not AnimationStatus.LEGACY_CANDIDATE
            or record.release_id is not None
            or record.provenance.get("lineage_id")
            != existing.provenance.get("lineage_id")
        ):
            raise ValueError("only the same quarantined legacy record may be hydrated")
        if any(embedding.record_id == record_id for embedding in self.embeddings.values()):
            raise ValueError("a legacy record with embeddings cannot be hydrated in place")
        self.records[record_id] = record

    def get_record(self, record_id: str) -> AnimationRecord | None:
        return self.records.get(record_id)

    def find_legacy_by_lineage(self, lineage_id: str) -> AnimationRecord | None:
        matches = [
            record
            for record in self.records.values()
            if record.status is AnimationStatus.LEGACY_CANDIDATE
            and record.provenance.get("lineage_id") == lineage_id
        ]
        if len(matches) > 1:
            raise RuntimeError("legacy lineage uniqueness was violated")
        return matches[0] if matches else None

    def records_for_release(self, release_id: str) -> tuple[AnimationRecord, ...]:
        return tuple(
            record
            for record in sorted(self.records.values(), key=lambda item: item.record_id)
            if record.release_id == release_id
        )

    def copy_embeddings(self, source_record_id: str, target_record_id: str) -> None:
        source_embeddings = [
            embedding
            for embedding in self.embeddings.values()
            if embedding.record_id == source_record_id
        ]
        for embedding in source_embeddings:
            self.put_embedding(replace(embedding, record_id=target_record_id))

    def put_embedding(self, embedding: EmbeddingVector) -> None:
        key = (
            embedding.record_id,
            embedding.namespace,
            embedding.namespace_version,
            embedding.segment_id,
        )
        if key in self.embeddings:
            raise ValueError("embedding segment is immutable")
        config = self.namespaces.get((embedding.namespace, embedding.namespace_version))
        if config is None or config.index is not embedding.index:
            raise ValueError("embedding namespace is not registered for this index")
        if len(embedding.values) != config.dimensions:
            raise ValueError("embedding dimension mismatch")
        self.embeddings[key] = embedding

    def embedding_manifest(self, record_id: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "index": embedding.index.value,
                "namespace": embedding.namespace,
                "namespace_version": embedding.namespace_version,
                "segment_id": embedding.segment_id,
                "values": list(embedding.values),
                "metadata": dict(embedding.metadata),
            }
            for embedding in sorted(
                (
                    item
                    for item in self.embeddings.values()
                    if item.record_id == record_id
                ),
                key=lambda item: (
                    item.index.value,
                    item.namespace,
                    item.namespace_version,
                    item.segment_id,
                ),
            )
        )

    def rank(
        self,
        config: EmbeddingNamespaceConfig,
        query: tuple[float, ...],
        filters: RetrievalFiltersV1,
        *,
        limit: int,
    ) -> tuple[RankedRetrievalHitV1, ...]:
        candidates: list[tuple[float, str]] = []
        query_array = np.asarray(query, dtype=np.float64)
        for embedding in self.embeddings.values():
            record = self.records.get(embedding.record_id)
            if (
                embedding.namespace != config.namespace
                or embedding.namespace_version != config.version
                or embedding.index is not config.index
                or record is None
                or not _eligible(record, filters)
            ):
                continue
            distance = _distance(config.distance_metric, query_array, np.asarray(embedding.values))
            candidates.append((distance, record.record_id))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return tuple(
            RankedRetrievalHitV1(
                record_id=record_id,
                index=config.index,
                rank=rank,
                distance=max(0.0, distance),
                release=filters.release,
            )
            for rank, (distance, record_id) in enumerate(candidates[:limit], start=1)
        )

    def insert_failure(self, failure: HardNegativeRecord) -> None:
        if failure.failure_id in self._failures:
            raise ValueError("failure records are immutable")
        self._failures[failure.failure_id] = failure

    def failures(self) -> tuple[HardNegativeRecord, ...]:
        return tuple(self._failures[key] for key in sorted(self._failures))
