"""Release-safe promotion, indexing, and reciprocal-rank retrieval."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from uuid import uuid4

from rigby_v2.flywheel.schemas import (
    FusedRetrievalResultV1,
    RetrievalFiltersV1,
    RetrievalIndex,
)
from rigby_v2.hashing import content_hash

from .embeddings import EmbeddingRouter, RetrievalQuery
from .legacy import prepare_hydrated_legacy_source
from .models import (
    AnimationRecord,
    AnimationRecordInput,
    AnimationStatus,
    CompactExample,
    EmbeddingVector,
    HardNegativeRecord,
    PromotionProof,
    ReleaseRecord,
    ReleaseStatus,
    RetrievalResponse,
    has_promotion_proof,
    staged_copy,
)
from .repository import LibraryRepository


class InsufficientCertifiedExamples(RuntimeError):
    pass


class CertifiedLibraryService:
    def __init__(
        self,
        repository: LibraryRepository,
        embeddings: EmbeddingRouter,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self._id_factory = id_factory or (lambda: str(uuid4()))
        for config in embeddings.configs.values():
            repository.register_namespace(config)

    def create_release(self, release_id: str, *, parent_release_id: str | None) -> None:
        active = self.repository.active_release()
        if active is None:
            if parent_release_id is not None:
                raise ValueError("the bootstrap release cannot name a parent")
        elif parent_release_id != active.release_id:
            raise ValueError("a building release must be the active release's N+1 child")
        self.repository.create_release(
            ReleaseRecord(release_id, parent_release_id, ReleaseStatus.BUILDING)
        )
        # N+1 is cumulative, but active records remain immutable: copy each
        # certified parent record and its exact embeddings into staging.
        if active is not None:
            for source in self.repository.records_for_release(active.release_id):
                carried = replace(
                    source,
                    record_id=self._id_factory(),
                    status=AnimationStatus.STAGED,
                    release_id=release_id,
                    parent_record_id=source.record_id,
                )
                self.repository.insert_record(carried)
                self.repository.copy_embeddings(source.record_id, carried.record_id)

    def import_legacy(self, source: AnimationRecordInput) -> AnimationRecord:
        existing = self.repository.find_legacy_by_lineage(source.lineage_id)
        if existing is not None:
            expected = AnimationRecord.from_input(
                existing.record_id,
                source,
                status=AnimationStatus.LEGACY_CANDIDATE,
            )
            if existing == expected:
                return existing
            if existing.evaluation.get("requires_hydration") is True and not source.evaluation.get(
                "requires_hydration", False
            ):
                hydrated_source = prepare_hydrated_legacy_source(existing, source)
                hydrated = AnimationRecord.from_input(
                    existing.record_id,
                    hydrated_source,
                    status=AnimationStatus.LEGACY_CANDIDATE,
                )
                self.repository.replace_legacy_record(existing.record_id, hydrated)
                return hydrated
            raise ValueError("legacy lineage already exists with different content")
        record = AnimationRecord.from_input(
            self._id_factory(), source, status=AnimationStatus.LEGACY_CANDIDATE
        )
        self.repository.insert_record(record)
        return record

    def submit_candidate(self, source: AnimationRecordInput) -> AnimationRecord:
        record = AnimationRecord.from_input(
            self._id_factory(), source, status=AnimationStatus.CANDIDATE
        )
        self.repository.insert_record(record)
        return record

    def promote_for_next_release(
        self,
        source_record_id: str,
        target_release_id: str,
        proof: PromotionProof,
    ) -> AnimationRecord:
        source = self.repository.get_record(source_record_id)
        if source is None:
            raise KeyError(source_record_id)
        if source.evaluation.get("requires_hydration") is True:
            raise ValueError(
                "deferred legacy metadata cannot be promoted before source-file hydration"
            )
        release = self.repository.get_release(target_release_id)
        if release is None or release.status is not ReleaseStatus.BUILDING:
            raise ValueError("promotion target must be a building release")
        active = self.repository.active_release()
        expected_parent = active.release_id if active else None
        if release.parent_release_id != expected_parent:
            raise ValueError("promotion target is not the active release's N+1 staging release")
        promoted = staged_copy(
            source,
            record_id=self._id_factory(),
            release_id=target_release_id,
            proof=proof,
        )
        self.repository.insert_record(promoted)
        return promoted

    def index_record(
        self,
        record_id: str,
        payloads: Mapping[RetrievalIndex, object],
    ) -> None:
        record = self.repository.get_record(record_id)
        if record is None:
            raise KeyError(record_id)
        if record.evaluation.get("requires_hydration") is True:
            raise ValueError(
                "deferred legacy metadata cannot be indexed before source-file hydration"
            )
        if set(payloads) != set(RetrievalIndex):
            raise ValueError("indexing requires payloads for all five indexes")
        for index in RetrievalIndex:
            config = self.embeddings.configs[index]
            self.repository.put_embedding(
                EmbeddingVector(
                    record_id=record_id,
                    index=index,
                    namespace=config.namespace,
                    namespace_version=config.version,
                    values=self.embeddings.embed(index, payloads[index]),
                )
            )

    def freeze_release(self, release_id: str) -> None:
        release = self.repository.get_release(release_id)
        if release is None:
            raise KeyError(release_id)
        records = self.repository.records_for_release(release_id)
        namespace_by_key = {
            (config.namespace, config.version): config
            for config in self.embeddings.configs.values()
        }
        record_entries = []
        for record in records:
            if not has_promotion_proof(record):
                raise ValueError(
                    "release contains a staged record without independent promotion proof"
                )
            embedding_entries = []
            observed_indexes = set()
            for embedding in self.repository.embedding_manifest(record.record_id):
                config = namespace_by_key.get(
                    (embedding["namespace"], embedding["namespace_version"])
                )
                if config is None:
                    raise ValueError("release contains an unversioned embedding namespace")
                observed_indexes.add(config.index)
                embedding_entries.append(
                    {
                        "index": config.index.value,
                        "namespace": config.namespace,
                        "namespace_version": config.version,
                        "segment_id": embedding["segment_id"],
                        "values_sha256": content_hash(embedding["values"]),
                        "metadata_sha256": content_hash(embedding["metadata"]),
                    }
                )
            if observed_indexes != set(RetrievalIndex):
                raise ValueError("every released record requires all five retrieval indexes")
            record_entries.append(
                {
                    "record_id": record.record_id,
                    "program_sha256": record.program_sha256,
                    "world_sha256": record.world_sha256,
                    "motion_sha256": record.motion_sha256,
                    "embeddings": sorted(
                        embedding_entries,
                        key=lambda item: (
                            item["index"],
                            item["namespace"],
                            item["namespace_version"],
                            item["segment_id"],
                        ),
                    ),
                }
            )
        manifest = {
            "release_id": release_id,
            "parent": release.parent_release_id,
            "embedding_namespaces": sorted(
                (
                    {
                        "index": config.index.value,
                        "namespace": config.namespace,
                        "version": config.version,
                        "model_name": config.model_name,
                        "model_sha256": config.model_sha256,
                        "preprocessing_version": config.preprocessing_version,
                        "dimensions": config.dimensions,
                        "distance_metric": config.distance_metric.value,
                    }
                    for config in self.embeddings.configs.values()
                ),
                key=lambda item: item["index"],
            ),
            "records": record_entries,
        }
        self.repository.freeze_release(release_id, manifest, content_hash(manifest))

    def activate_release(self, release_id: str) -> None:
        self.repository.activate_release(release_id)

    def record_failure(self, failure: HardNegativeRecord) -> None:
        self.repository.insert_failure(failure)

    def retrieve(
        self,
        query: RetrievalQuery,
        filters: RetrievalFiltersV1,
        *,
        example_count: int = 3,
        reciprocal_rank_k: int = 60,
        per_index_limit: int = 50,
    ) -> RetrievalResponse:
        if example_count not in (2, 3):
            raise ValueError("retrieval returns exactly two or three examples")
        if reciprocal_rank_k <= 0 or per_index_limit <= 0:
            raise ValueError("ranking limits must be positive")
        active = self.repository.active_release()
        if active is None or filters.release != active.release_id:
            raise ValueError("queries are restricted to the one immutable active release")
        hits_by_index = {}
        fused: dict[str, float] = {}
        for index in RetrievalIndex:
            config = self.embeddings.configs[index]
            vector = self.embeddings.embed(index, query.payload(index))
            hits = self.repository.rank(
                config, vector, filters, limit=per_index_limit
            )
            hits_by_index[index] = hits
            for hit in hits:
                fused[hit.record_id] = fused.get(hit.record_id, 0.0) + 1.0 / (
                    reciprocal_rank_k + hit.rank
                )
        ordered = sorted(fused, key=lambda record_id: (-fused[record_id], record_id))
        selected = ordered[: min(example_count, len(ordered))]
        if len(selected) < 2:
            raise InsufficientCertifiedExamples(
                "fewer than two independently certified compatible examples are available"
            )
        compact_examples: list[CompactExample] = []
        for record_id in selected:
            record = self.repository.get_record(record_id)
            if record is None:
                raise RuntimeError(f"ranked record {record_id!r} disappeared")
            compact_examples.append(CompactExample(record_id, record.compact_example))
        examples = tuple(compact_examples)
        query_hash = content_hash(
            {
                "query": {
                    "prompt": query.prompt,
                    "structured_program": query.structured_program,
                    "task_space_motion": query.task_space_motion,
                    "video": query.video,
                    "salient_keyframes": query.salient_keyframes,
                },
                "filters": filters.model_dump(mode="python"),
                "namespaces": {
                    index.value: (
                        self.embeddings.configs[index].namespace,
                        self.embeddings.configs[index].version,
                    )
                    for index in RetrievalIndex
                },
            }
        )
        return RetrievalResponse(
            result=FusedRetrievalResultV1(
                query_hash=query_hash,
                release=filters.release,
                hits_by_index=hits_by_index,
                selected_examples=tuple(selected),
                reciprocal_rank_k=reciprocal_rank_k,
            ),
            examples=examples,
        )
