"""Fail-closed retrieval audit for an incomplete, building library release.

This module deliberately does not expose staged records to production retrieval.
It exists so a release candidate can be audited while one or more mandatory model
namespaces are unavailable.  A successful audit is not a freeze or activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

from rigby_v2.flywheel.schemas import (
    RankedRetrievalHitV1,
    RetrievalFiltersV1,
    RetrievalIndex,
)

from .models import (
    AnimationRecord,
    AnimationStatus,
    EmbeddingNamespaceConfig,
    ReleaseStatus,
    has_promotion_proof,
)
from .repository import LibraryRepository


@dataclass(frozen=True)
class StagingRetrievalAudit:
    """Audited staging-only ranks; never a production retrieval response."""

    release_id: str
    hits_by_index: Mapping[RetrievalIndex, tuple[RankedRetrievalHitV1, ...]]
    reciprocal_rank_scores: Mapping[str, float]
    selected_record_ids: tuple[str, ...]
    available_indexes: tuple[RetrievalIndex, ...]
    missing_indexes: tuple[RetrievalIndex, ...]
    promotable: bool


def staging_filter_rejection(
    record: AnimationRecord,
    filters: RetrievalFiltersV1,
) -> str | None:
    """Return a stable reason when a record is ineligible for a staging audit."""

    if record.status is not AnimationStatus.STAGED:
        return "status_not_staged"
    if record.release_id != filters.release:
        return "release_mismatch"
    if not has_promotion_proof(record):
        return "missing_promotion_proof"
    if record.provenance.get("license_id") not in filters.allowed_licenses:
        return "license_not_allowed"
    if record.schema_version not in filters.schema_versions:
        return "schema_not_allowed"
    if record.world.get("rig_id") not in filters.rig_ids:
        return "rig_not_allowed"
    if record.split in filters.excluded_splits:
        return "split_excluded"
    if record.provenance.get("lineage_id") in filters.excluded_lineage:
        return "lineage_excluded"
    required_sets = (
        ("object_affordance_missing", filters.object_affordances, record.world.get("object_affordances", ())),
        ("limb_missing", filters.limbs, record.labels.get("limbs", ())),
        (
            "contact_requirement_missing",
            filters.contact_requirements,
            record.labels.get("contact_requirements", ()),
        ),
    )
    for reason, required, available in required_sets:
        if not set(required) <= set(available):
            return reason
    return None


def _cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("query and indexed vectors must have the same positive dimension")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine vectors cannot be zero")
    return max(0.0, 1.0 - dot / (left_norm * right_norm))


def audit_building_release_retrieval(
    repository: LibraryRepository,
    configs: Mapping[RetrievalIndex, EmbeddingNamespaceConfig],
    query_vectors: Mapping[RetrievalIndex, tuple[float, ...]],
    filters: RetrievalFiltersV1,
    *,
    reciprocal_rank_k: int = 60,
    limit: int = 3,
) -> StagingRetrievalAudit:
    """Rank eligible staged records without changing their release visibility."""

    if reciprocal_rank_k <= 0 or limit <= 0:
        raise ValueError("ranking limits must be positive")
    if set(configs) != set(query_vectors):
        raise ValueError("each available namespace requires exactly one query vector")
    release = repository.get_release(filters.release)
    if release is None or release.status is not ReleaseStatus.BUILDING:
        raise ValueError("staging audit requires a building release")

    ranked: dict[RetrievalIndex, tuple[RankedRetrievalHitV1, ...]] = {}
    fused: dict[str, float] = {}
    for index in sorted(configs, key=lambda item: item.value):
        config = configs[index]
        query = query_vectors[index]
        candidates: list[tuple[float, str]] = []
        for record in repository.records_for_release(filters.release):
            if staging_filter_rejection(record, filters) is not None:
                continue
            embeddings = repository.embedding_manifest(record.record_id)
            for embedding in embeddings:
                if (
                    embedding["namespace"] == config.namespace
                    and embedding["namespace_version"] == config.version
                    and embedding["segment_id"] == "global"
                ):
                    values = tuple(float(value) for value in embedding["values"])
                    candidates.append((_cosine_distance(query, values), record.record_id))
                    break
        candidates.sort(key=lambda item: (item[0], item[1]))
        hits = tuple(
            RankedRetrievalHitV1(
                record_id=record_id,
                index=index,
                rank=rank,
                distance=distance,
                release=filters.release,
            )
            for rank, (distance, record_id) in enumerate(candidates[:limit], start=1)
        )
        ranked[index] = hits
        for hit in hits:
            fused[hit.record_id] = fused.get(hit.record_id, 0.0) + 1.0 / (
                reciprocal_rank_k + hit.rank
            )

    ordered = tuple(sorted(fused, key=lambda record_id: (-fused[record_id], record_id)))
    available = tuple(sorted(configs, key=lambda item: item.value))
    missing = tuple(sorted(set(RetrievalIndex) - set(available), key=lambda item: item.value))
    return StagingRetrievalAudit(
        release_id=filters.release,
        hits_by_index=ranked,
        reciprocal_rank_scores={record_id: fused[record_id] for record_id in ordered},
        selected_record_ids=ordered[:limit],
        available_indexes=available,
        missing_indexes=missing,
        promotable=not missing,
    )
