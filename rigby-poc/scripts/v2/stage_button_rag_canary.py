"""Build one real button RAG canary while keeping the incomplete release staged.

The command is intentionally one-shot and fail-closed.  It performs all costly
simulation, rendering, and offline model inference before atomically adding the
candidate, staged copy, three honest namespaces, and embeddings to PostgreSQL.
Cosmos text/video absence is recorded and never substituted.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import mujoco
import numpy as np
import psycopg
from PIL import Image
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from rigby_v2.acceptance.articulated_packs import certify_button_press
from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.certification import CertificationEngine
from rigby_v2.evidence import MujocoEvidenceRenderer
from rigby_v2.evidence.renderer import trace_archive_bytes
from rigby_v2.flywheel.schemas import EvidenceCameraRole, RetrievalFiltersV1, RetrievalIndex
from rigby_v2.hashing import canonical_json, canonical_json_bytes, content_hash, hash_file, sha256_bytes
from rigby_v2.library import (
    SIGLIP2_SPEC,
    AnimationRecord,
    AnimationRecordInput,
    AnimationStatus,
    DeterministicDescriptorEmbedder,
    DistanceMetric,
    EmbeddingModelIdentity,
    EmbeddingNamespaceConfig,
    PostgresLibraryRepository,
    PromotionProof,
    ReleaseRecord,
    ReleaseStatus,
    SigLIP2KeyframeProvider,
    audit_building_release_retrieval,
    manifest_path,
    verify_snapshot,
)
from rigby_v2.library.models import staged_copy
from rigby_v2.simulation import load_model_source


RELEASE_ID = "button-rag-canary-v1-building"
CANDIDATE_ID = str(uuid5(NAMESPACE_URL, "rigby-v2/button-rag-canary/source"))
STAGED_ID = str(uuid5(NAMESPACE_URL, "rigby-v2/button-rag-canary/staged"))
LINEAGE_ID = "button-press-physical-v2-canary"
ANCHORS = (
    "approach_clear",
    "physical_button_contact",
    "button_pressed",
    "release_clear",
    "no_object_actuator",
)


def _trace_hashes(certification: object) -> tuple[str, ...]:
    runs = getattr(certification, "simulation_runs")
    return tuple(sha256_bytes(trace_archive_bytes(run.trace)) for run in runs)


def _certification_summary(certification: object) -> dict[str, object]:
    predicates = getattr(certification, "predicate_evaluations")
    violations = getattr(certification, "violations")
    return {
        "outcome": getattr(certification, "outcome").value,
        "candidate_id": getattr(certification, "candidate_id"),
        "trace_sha256": list(_trace_hashes(certification)),
        "predicates": [asdict(value) for value in predicates],
        "violations": [
            {**asdict(value), "code": value.code.value}
            for value in violations
        ],
    }


def _counts(database_url: str) -> dict[str, object]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        records = connection.execute(
            """SELECT status, count(*) AS count FROM rigby_v2.animation_records
               GROUP BY status ORDER BY status"""
        ).fetchall()
        releases = connection.execute(
            """SELECT status, count(*) AS count FROM rigby_v2.library_releases
               GROUP BY status ORDER BY status"""
        ).fetchall()
        return {
            "records": {row["status"]: int(row["count"]) for row in records},
            "releases": {row["status"]: int(row["count"]) for row in releases},
            "namespaces": int(
                connection.execute(
                    "SELECT count(*) AS count FROM rigby_v2.embedding_namespaces"
                ).fetchone()["count"]
            ),
            "embeddings": int(
                connection.execute(
                    "SELECT count(*) AS count FROM rigby_v2.embeddings"
                ).fetchone()["count"]
            ),
        }


def _preflight(database_url: str, output: Path) -> dict[str, object]:
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    counts = _counts(database_url)
    if counts["releases"] or counts["namespaces"] or counts["embeddings"]:
        raise RuntimeError(
            "one-shot bootstrap requires no pre-existing releases, namespaces, or embeddings"
        )
    if counts["records"] != {"legacy_candidate": 6354}:
        raise RuntimeError(f"unexpected live record inventory: {counts['records']!r}")
    return counts


def _salient_frame(
    store: ContentAddressedArtifactStore,
    evidence: object,
    model: mujoco.MjModel,
    trace: object,
) -> tuple[np.ndarray, dict[str, object]]:
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "obj__control_panel__button_slide"
    )
    if joint < 0:
        raise RuntimeError("compiled button joint is missing")
    address = int(model.jnt_qposadr[joint])
    peak_sample = int(np.argmin(trace.qpos[:, address]))
    peak_time = float(trace.times_s[peak_sample])
    camera = next(
        item for item in evidence.cameras if item.role is EvidenceCameraRole.TASK_CLOSEUP
    )
    frame_index = int(np.argmin(np.abs(np.asarray(camera.timestamps_s) - peak_time)))
    frame_name = f"frames/{frame_index:06d}.png"
    archive_bytes = store.read_bytes(camera.raw_frames)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["role"] != EvidenceCameraRole.TASK_CLOSEUP.value:
            raise RuntimeError("selected evidence archive is not task-closeup")
        frame_bytes = archive.read(frame_name)
    with Image.open(io.BytesIO(frame_bytes)) as source:
        source.load()
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    return image, {
        "camera_id": camera.camera_id,
        "role": camera.role.value,
        "raw_frames": camera.raw_frames.model_dump(mode="json"),
        "video": camera.video.model_dump(mode="json"),
        "frame_name": frame_name,
        "frame_sha256": sha256_bytes(frame_bytes),
        "peak_button_time_s": peak_time,
        "peak_button_position_m": float(trace.qpos[peak_sample, address]),
    }


def _configs(tree_sha256: str, descriptor_sha256: str) -> dict[
    RetrievalIndex, EmbeddingNamespaceConfig
]:
    return {
        RetrievalIndex.KEYFRAME: EmbeddingNamespaceConfig(
            index=RetrievalIndex.KEYFRAME,
            namespace="siglip2_salient_keyframe",
            version="1.0",
            model_name=f"{SIGLIP2_SPEC.repository}@{SIGLIP2_SPEC.revision}",
            model_sha256=tree_sha256,
            preprocessing_version=SIGLIP2_SPEC.preprocessing_version,
            dimensions=SIGLIP2_SPEC.dimensions,
            distance_metric=DistanceMetric.COSINE,
        ),
        RetrievalIndex.PROGRAM: EmbeddingNamespaceConfig(
            index=RetrievalIndex.PROGRAM,
            namespace="rigby_program_descriptor",
            version="1.0",
            model_name="rigby/deterministic-program-descriptor",
            model_sha256=descriptor_sha256,
            preprocessing_version="motion-program-v2-feature-hash-v1",
            dimensions=256,
            distance_metric=DistanceMetric.COSINE,
        ),
        RetrievalIndex.MOTION: EmbeddingNamespaceConfig(
            index=RetrievalIndex.MOTION,
            namespace="rigby_phase_contact_descriptor",
            version="1.0",
            model_name="rigby/deterministic-phase-contact-descriptor",
            model_sha256=descriptor_sha256,
            preprocessing_version="phase-contact-task-space-v1",
            dimensions=256,
            distance_metric=DistanceMetric.COSINE,
        ),
    }


def _write_record(connection: object, record: AnimationRecord) -> None:
    connection.execute(
        """
        INSERT INTO rigby_v2.animation_records (
            record_id, schema_version, status, release_id, split, prompt,
            program, world, motion, evidence, labels, evaluation, provenance,
            program_sha256, world_sha256, motion_sha256, parent_record_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s)
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


def _commit(
    database_url: str,
    release: ReleaseRecord,
    source: AnimationRecord,
    staged: AnimationRecord,
    configs: dict[RetrievalIndex, EmbeddingNamespaceConfig],
    vectors: dict[RetrievalIndex, tuple[float, ...]],
    vector_metadata: dict[RetrievalIndex, dict[str, object]],
) -> None:
    with psycopg.connect(database_url) as connection, connection.transaction():
        connection.execute(
            """INSERT INTO rigby_v2.library_releases
               (release_id, parent_release_id, status, manifest)
               VALUES (%s, NULL, 'building', %s)""",
            (release.release_id, Jsonb(release.manifest)),
        )
        _write_record(connection, source)
        _write_record(connection, staged)
        for index in sorted(configs, key=lambda item: item.value):
            config = configs[index]
            connection.execute(
                """INSERT INTO rigby_v2.embedding_namespaces
                   (namespace, version, model_name, model_sha256,
                    preprocessing_version, dimensions, distance_metric, active)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, true)""",
                (
                    config.namespace,
                    config.version,
                    config.model_name,
                    config.model_sha256,
                    config.preprocessing_version,
                    config.dimensions,
                    config.distance_metric.value,
                ),
            )
            connection.execute(
                """INSERT INTO rigby_v2.embeddings
                   (record_id, namespace, namespace_version, segment_id,
                    embedding, metadata)
                   VALUES (%s, %s, %s, 'global', %s::vector, %s)""",
                (
                    staged.record_id,
                    config.namespace,
                    config.version,
                    json.dumps(list(vectors[index]), separators=(",", ":")),
                    Jsonb(vector_metadata[index]),
                ),
            )


def _write_sealed(path: Path, document: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = canonical_json(document) + "\n"
    path.write_text(rendered, encoding="utf-8", newline="\n")
    digest = hash_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        digest + "\n", encoding="ascii", newline="\n"
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts-v2/library/button-rag-canary"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/v2/library/button_rag_canary_staging.v1.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    before = _preflight(arguments.database_url, arguments.output)

    attempt = certify_button_press(arguments.artifact_root)
    if not attempt.certification.certified or not attempt.robust:
        raise RuntimeError("current button baseline and five variations must all certify")
    independent = CertificationEngine().certify(attempt.request)
    if not independent.certified:
        raise RuntimeError("independent re-simulation/evaluation did not certify")
    baseline_summary = _certification_summary(attempt.certification)
    independent_summary = _certification_summary(independent)
    if baseline_summary != independent_summary:
        raise RuntimeError("independent certification did not reproduce the baseline exactly")
    resimulation_sha = content_hash(independent_summary["trace_sha256"])
    independent_evaluation_sha = content_hash(independent_summary)
    certification_id = "button-cert-" + content_hash(baseline_summary)[:20]

    store = ContentAddressedArtifactStore(arguments.artifact_root)
    baseline_run = attempt.certification.simulation_runs[0]
    model, _ = load_model_source(attempt.request.simulation)
    rendered = MujocoEvidenceRenderer(store).render(
        model,
        baseline_run.trace,
        candidate_id=attempt.certification.candidate_id,
        anonymous_id="button-rag-canary",
        metrics={
            "button_terminal_m": attempt.raw_evidence.terminal_object_state,
            "button_minimum_m": attempt.raw_evidence.maximum_object_state,
            "contact_force_n": attempt.raw_evidence.contact.maximum_normal_force_n,
        },
        task_site="obj__control_panel__button_surface",
    )
    image, frame_evidence = _salient_frame(store, rendered, model, baseline_run.trace)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _, snapshot_manifest = verify_snapshot(SIGLIP2_SPEC)
    tree_sha256 = str(snapshot_manifest["snapshot"]["tree_sha256"])
    identity = EmbeddingModelIdentity(
        repository=SIGLIP2_SPEC.repository,
        revision=SIGLIP2_SPEC.revision,
        snapshot_sha256=tree_sha256,
        preprocessing_version=SIGLIP2_SPEC.preprocessing_version,
        dimensions=SIGLIP2_SPEC.dimensions,
    )
    provider = SigLIP2KeyframeProvider(identity, device=arguments.device, batch_size=1)
    keyframe_vector = tuple(provider.embed(image))
    repeated = tuple(provider.embed(image))
    maximum_repeat_delta = max(
        abs(left - right) for left, right in zip(keyframe_vector, repeated, strict=True)
    )
    if maximum_repeat_delta != 0.0:
        raise RuntimeError("current SigLIP2 keyframe inference is not repeat-identical")

    descriptor_spec = {
        "schema_version": "rigby.deterministic_descriptors.v1",
        "dimensions": 256,
        "canonical_input": "rigby_v2.hashing.canonical_json",
        "program": "motion-program-v2-feature-hash-v1",
        "motion": "phase-contact-task-space-v1",
    }
    descriptor_sha = content_hash(descriptor_spec)
    program_payload = {
        "schema_version": "rigby.button_acceptance_program.v1",
        "action": "press_then_release",
        "effector": "left_index_tip",
        "target_site": "obj__control_panel__button_surface",
        "phases": ["approach", "contact", "hold", "release"],
        "planned_contact_s": [1.17, 1.94],
        "state_predicate": "control_panel.button_pressed",
        "object_actuation": False,
    }
    trajectory = attempt.request.simulation.trajectory
    motion_payload = {
        "schema_version": "rigby.phase_contact_motion.v1",
        "duration_s": attempt.request.simulation.config.duration_s,
        "trajectory_times_sha256": content_hash(trajectory.times_s.tolist()),
        "trajectory_qpos_sha256": content_hash(trajectory.qpos.tolist()),
        "phase_anchors_s": [0.0, 0.3, 1.5, 1.9, 2.6, 2.8],
        "contact_lifecycle": ["clear", "form", "press", "break", "clear"],
        "terminal_button_position_m": attempt.raw_evidence.terminal_object_state,
    }
    embedder = DeterministicDescriptorEmbedder(256)
    vectors = {
        RetrievalIndex.KEYFRAME: keyframe_vector,
        RetrievalIndex.PROGRAM: tuple(float(value) for value in embedder.embed(program_payload)),
        RetrievalIndex.MOTION: tuple(float(value) for value in embedder.embed(motion_payload)),
    }
    configs = _configs(tree_sha256, descriptor_sha)
    vector_hashes = {
        index: sha256_bytes(canonical_json_bytes(vector))
        for index, vector in vectors.items()
    }

    compact_example = " ".join(ANCHORS) + (
        "; left index approaches clear, makes measured contact, depresses the free "
        "unactuated button past its predicate, then releases clear."
    )
    robustness = [
        {
            "variation": asdict(variation),
            "certification": _certification_summary(result),
        }
        for variation, result in attempt.robustness_variations
    ]
    source_input = AnimationRecordInput(
        schema_version="2.0",
        split="production",
        prompt_text="Press the control-panel button with the left index finger, then release.",
        program=program_payload,
        world={
            "object_pack": "lever_button",
            "selected_task": "control_panel.button_pressed",
            "model_sha256": attempt.request.expected_model_hash,
        },
        motion=motion_payload,
        evidence={
            "authoritative_trace": rendered.trace.model_dump(mode="json"),
            "task_closeup": frame_evidence,
            "contact": asdict(attempt.raw_evidence.contact),
            "robustness": robustness,
        },
        labels={"canary_anchors": list(ANCHORS)},
        evaluation={
            "certified": True,
            "baseline": baseline_summary,
            "robust": True,
            "robustness_variation_count": len(robustness),
        },
        provenance={
            "source": "rigby_v2.acceptance.articulated_packs.certify_button_press",
            "runtime": "native_mujoco_240hz",
            "learned_motion_model": False,
        },
        license_id="Apache-2.0",
        rig_id="rigby-canonical-human-medium",
        lineage_id=LINEAGE_ID,
        compact_example=compact_example,
        object_affordances=("push_button",),
        limbs=("left_arm", "left_hand"),
        contact_requirements=("button_contact",),
    )
    source = AnimationRecord.from_input(
        CANDIDATE_ID, source_input, status=AnimationStatus.CANDIDATE
    )
    proof = PromotionProof(
        independently_certified=True,
        resimulated=True,
        independently_evaluated=True,
        certification_id=certification_id,
        resimulation_result_sha256=resimulation_sha,
        independent_evaluation_sha256=independent_evaluation_sha,
        evaluator_id="rigby-v2-certification-engine-independent-pass",
    )
    staged = staged_copy(source, record_id=STAGED_ID, release_id=RELEASE_ID, proof=proof)
    release_manifest = {
        "schema_version": "rigby.library_building_release.v1",
        "purpose": "real certified button RAG canary",
        "record_id": STAGED_ID,
        "available_indexes": sorted(index.value for index in vectors),
        "missing_indexes": [RetrievalIndex.TEXT.value, RetrievalIndex.VIDEO.value],
        "missing_model": "nvidia/Cosmos-Embed1-336p",
        "freeze_allowed": False,
        "activation_allowed": False,
    }
    release = ReleaseRecord(RELEASE_ID, None, ReleaseStatus.BUILDING, manifest=release_manifest)
    vector_metadata = {
        RetrievalIndex.KEYFRAME: {
            "vector_sha256": vector_hashes[RetrievalIndex.KEYFRAME],
            "frame_sha256": frame_evidence["frame_sha256"],
            "repeat_count": 2,
            "maximum_repeat_delta": maximum_repeat_delta,
            "network_at_runtime": False,
        },
        RetrievalIndex.PROGRAM: {
            "vector_sha256": vector_hashes[RetrievalIndex.PROGRAM],
            "payload_sha256": content_hash(program_payload),
            "learned_model": False,
        },
        RetrievalIndex.MOTION: {
            "vector_sha256": vector_hashes[RetrievalIndex.MOTION],
            "payload_sha256": content_hash(motion_payload),
            "learned_model": False,
        },
    }
    _commit(
        arguments.database_url,
        release,
        source,
        staged,
        configs,
        vectors,
        vector_metadata,
    )

    repository = PostgresLibraryRepository(arguments.database_url)
    filters = RetrievalFiltersV1(
        release=RELEASE_ID,
        allowed_licenses=("Apache-2.0",),
        schema_versions=("2.0",),
        rig_ids=("rigby-canonical-human-medium",),
        object_affordances=("push_button",),
        limbs=("left_arm", "left_hand"),
        contact_requirements=("button_contact",),
    )
    audit = audit_building_release_retrieval(repository, configs, vectors, filters)
    wrong_license = filters.model_copy(update={"allowed_licenses": ("CC-BY-4.0",)})
    excluded_lineage = filters.model_copy(update={"excluded_lineage": (LINEAGE_ID,)})
    filter_results = {
        "compatible_selected": list(audit.selected_record_ids),
        "wrong_license_selected": list(
            audit_building_release_retrieval(
                repository, configs, vectors, wrong_license
            ).selected_record_ids
        ),
        "excluded_lineage_selected": list(
            audit_building_release_retrieval(
                repository, configs, vectors, excluded_lineage
            ).selected_record_ids
        ),
        "production_rank_count": len(
            repository.rank(
                configs[RetrievalIndex.KEYFRAME],
                vectors[RetrievalIndex.KEYFRAME],
                filters,
                limit=3,
            )
        ),
    }
    if filter_results != {
        "compatible_selected": [STAGED_ID],
        "wrong_license_selected": [],
        "excluded_lineage_selected": [],
        "production_rank_count": 0,
    }:
        raise RuntimeError(f"hard-filter isolation audit failed: {filter_results!r}")
    retrieved = repository.get_record(STAGED_ID)
    if retrieved is None:
        raise RuntimeError("staged record disappeared")
    rag_off = 0
    rag_on = sum(anchor in retrieved.compact_example.split(";")[0].split() for anchor in ANCHORS)
    after = _counts(arguments.database_url)
    blocker = Path("assets/v2/library/cosmos_embed1_blocker.json")
    blocker_seal = blocker.with_suffix(blocker.suffix + ".sha256")
    if not blocker.is_file() or not blocker_seal.is_file():
        raise RuntimeError("sealed Cosmos admission blocker is missing")
    if hash_file(blocker) != blocker_seal.read_text(encoding="ascii").strip():
        raise RuntimeError("Cosmos admission blocker seal is invalid")

    document: dict[str, object] = {
        "schema_version": "rigby.button_rag_canary_staging.v1",
        "status": "verified_staging_only",
        "release": {
            **release_manifest,
            "release_id": RELEASE_ID,
            "status": "building",
            "manifest_sha256": content_hash(release_manifest),
        },
        "record": {
            "source_record_id": CANDIDATE_ID,
            "staged_record_id": STAGED_ID,
            "program_sha256": staged.program_sha256,
            "world_sha256": staged.world_sha256,
            "motion_sha256": staged.motion_sha256,
            "certification_id": certification_id,
            "resimulation_result_sha256": resimulation_sha,
            "independent_evaluation_sha256": independent_evaluation_sha,
        },
        "certification": {
            "baseline": baseline_summary,
            "independent": independent_summary,
            "robustness": robustness,
            "baseline_repeat_count": 3,
            "independent_repeat_count": 3,
            "robustness_variation_count": 5,
        },
        "evidence": {
            "authoritative_trace": rendered.trace.model_dump(mode="json"),
            "task_closeup": frame_evidence,
        },
        "embeddings": {
            index.value: {
                "namespace": configs[index].namespace,
                "version": configs[index].version,
                "dimensions": len(vectors[index]),
                "l2_norm": float(np.linalg.norm(vectors[index])),
                "vector_sha256": vector_hashes[index],
            }
            for index in sorted(vectors, key=lambda item: item.value)
        },
        "siglip2": {
            "repository": SIGLIP2_SPEC.repository,
            "revision": SIGLIP2_SPEC.revision,
            "snapshot_tree_sha256": tree_sha256,
            "snapshot_manifest_sha256": hash_file(manifest_path(SIGLIP2_SPEC)),
            "network_at_runtime": False,
            "repeat_count": 2,
            "maximum_repeat_delta": maximum_repeat_delta,
        },
        "cosmos": {
            "status": "unavailable",
            "substitution_used": False,
            "blocker_sha256": hash_file(blocker),
        },
        "retrieval_audit": {
            "hard_filters": filter_results,
            "available_indexes": [item.value for item in audit.available_indexes],
            "missing_indexes": [item.value for item in audit.missing_indexes],
            "rrf_k": 60,
            "rrf_scores": dict(audit.reciprocal_rank_scores),
            "selected_record_ids": list(audit.selected_record_ids),
            "promotable": audit.promotable,
        },
        "rag_context_coverage_canary": {
            "metric": "certified_anchor_context_coverage",
            "anchors": list(ANCHORS),
            "rag_off_hits": rag_off,
            "rag_on_hits": rag_on,
            "denominator": len(ANCHORS),
            "rag_off": rag_off / len(ANCHORS),
            "rag_on": rag_on / len(ANCHORS),
            "absolute_improvement": (rag_on - rag_off) / len(ANCHORS),
            "scope": "smoke metric only; not the Goal 10 paired-bootstrap quality gate",
        },
        "database": {"before": before, "after": after},
    }
    seal = _write_sealed(arguments.output, document)
    print(
        json.dumps(
            {
                "evidence": str(arguments.output.resolve()),
                "evidence_sha256": seal,
                "database": after,
                "release_status": "building",
                "promotable": False,
                "record_id": STAGED_ID,
                "vector_sha256": {index.value: value for index, value in vector_hashes.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
