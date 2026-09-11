from __future__ import annotations

import json
import subprocess
import sys
from itertools import count
from pathlib import Path

import pytest

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.flywheel.schemas import RetrievalIndex
from rigby_v2.library import (
    AnimationStatus,
    CertifiedLibraryService,
    DeferredLegacyIndexImporter,
    EmbeddingRouter,
    InMemoryLibraryRepository,
    LegacyArchiveImporter,
    PromotionProof,
    standard_namespace_configs,
)
from rigby_core.hashing import content_hash, hash_file

pytestmark = pytest.mark.medium


class Passthrough:
    def embed(self, payload):  # type: ignore[no-untyped-def]
        return payload


def test_sealed_live_legacy_inventory_remains_quarantined() -> None:
    root = Path(__file__).resolve().parents[1] / "assets" / "v2" / "library"
    inventory_path = root / "legacy_import_inventory.json"
    expected = (root / "legacy_import_inventory.json.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    assert hash_file(inventory_path) == expected
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["indexed_results"] == inventory["database_records"] == 6354
    assert inventory["requires_hydration"] == 6354
    assert inventory["requires_resimulation"] == 6354
    assert inventory["requires_evaluation"] == 6354
    assert inventory["released_records"] == 0
    assert inventory["embedding_rows"] == 0
    assert inventory["artifact_payloads_copied"] == 0
    assert inventory["positive_context_eligible"] is False


def _service():
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
    return (
        CertifiedLibraryService(
            repository,
            EmbeddingRouter(
                configs, {index: Passthrough() for index in RetrievalIndex}
            ),
            id_factory=lambda: f"00000000-0000-0000-0000-{next(ids):012d}",
        ),
        repository,
    )


def _archive(root: Path) -> Path:
    folder = root / "000001-legacy-motion"
    folder.mkdir(parents=True)
    documents = {
        "program.json": {
            "schema_version": "1.0",
            "source_text": "Pick up the block.",
            "intent": "interaction",
            "hand": "right",
            "primitives": [{"kind": "grasp"}, {"kind": "lift"}],
        },
        "scene.json": {
            "schema_version": "1.0",
            "rig": {"id": "legacy-rig"},
            "objects": [{"id": "block", "kind": "block"}],
        },
        "clip.json": {"schema_version": "1.0", "frames": [0, 1]},
        "metrics.json": {"max_penetration_m": 0.0},
        "provenance.json": {"rig_id": "legacy-rig", "seed": 0},
        "request.json": {"schema_version": "1.0"},
    }
    for name, value in documents.items():
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    (folder / "animation.glb").write_bytes(b"glTF legacy test")
    return folder


def _index(root: Path, *, source_id: str = "000001-legacy-motion"):
    entry = {
        "id": source_id,
        "created_at": "2026-08-07T07:02:58+00:00",
        "prompt": "Pick up the block.",
        "intent": "interaction",
        "success": True,
        "metrics": {"max_penetration_m": 0.0},
    }
    document = {"schema_version": "1.0", "next_sequence": 2, "results": [entry]}
    path = root / "index.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, entry


def _proof():
    return PromotionProof(
        independently_certified=True,
        resimulated=True,
        independently_evaluated=True,
        certification_id="cert-after-hydration",
        resimulation_result_sha256="a" * 64,
        independent_evaluation_sha256="b" * 64,
        evaluator_id="independent-evaluator",
    )


def test_archive_import_is_idempotent_and_never_certifies_legacy(tmp_path: Path) -> None:
    root = tmp_path / "results"
    folder = _archive(root)
    service, repository = _service()
    importer = LegacyArchiveImporter(
        root,
        artifacts=ContentAddressedArtifactStore(tmp_path / "artifacts"),
    )

    first = importer.import_all(service)
    second = importer.import_all(service)

    assert len(first) == len(second) == 1
    assert first[0] == second[0]
    assert first[0].status is AnimationStatus.LEGACY_CANDIDATE
    assert first[0].release_id is None
    assert first[0].evaluation["requires_resimulation"] is True
    assert first[0].provenance["lineage_id"] == "legacy:000001-legacy-motion"
    assert len(repository.records) == 1
    assert importer.candidate(folder).source.prompt_text == "Pick up the block."


def test_changed_legacy_source_is_refused_under_the_same_lineage(tmp_path: Path) -> None:
    root = tmp_path / "results"
    folder = _archive(root)
    service, _ = _service()
    importer = LegacyArchiveImporter(
        root,
        artifacts=ContentAddressedArtifactStore(tmp_path / "artifacts"),
    )
    importer.import_all(service)
    program = json.loads((folder / "program.json").read_text(encoding="utf-8"))
    program["source_text"] = "Changed after import."
    (folder / "program.json").write_text(json.dumps(program), encoding="utf-8")

    with pytest.raises(ValueError, match="different content"):
        importer.import_all(service)


def test_archive_import_rejects_escape_and_size_overflow(tmp_path: Path) -> None:
    root = tmp_path / "results"
    folder = _archive(root)
    importer = LegacyArchiveImporter(
        root,
        artifacts=ContentAddressedArtifactStore(tmp_path / "artifacts"),
        maximum_animation_bytes=2,
    )
    with pytest.raises(ValueError, match="size limit"):
        importer.candidate(folder)
    with pytest.raises(ValueError, match="immediate archive child"):
        importer.candidate(tmp_path)


def test_metadata_only_index_import_is_hash_bound_idempotent_and_copies_no_payloads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    _archive(root)
    index_path, entry = _index(root)
    service, repository = _service()
    importer = DeferredLegacyIndexImporter(root)

    first = importer.import_all(service)
    second = importer.import_all(service)

    assert first == second
    assert len(first) == len(repository.records) == 1
    record = first[0]
    assert record.status is AnimationStatus.LEGACY_CANDIDATE
    assert record.release_id is None
    assert record.evaluation == {
        "certified": False,
        "requires_hydration": True,
        "hydration_verified": False,
        "requires_resimulation": True,
        "requires_evaluation": True,
    }
    assert record.provenance["legacy_source_index_sha256"] == hash_file(index_path)
    assert record.provenance["legacy_index_entry_sha256"] == content_hash(entry)
    assert record.evidence["legacy_archive_pointer"]["source_folder_id"] == entry["id"]
    assert not (tmp_path / "artifacts").exists()


def test_metadata_only_record_cannot_index_or_promote_until_full_hydration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    _archive(root)
    _index(root)
    service, repository = _service()
    deferred = DeferredLegacyIndexImporter(root).import_all(service)[0]
    service.create_release("release-1", parent_release_id=None)
    vectors = {index: (1.0, 0.0, 0.0, 0.0) for index in RetrievalIndex}

    with pytest.raises(ValueError, match="before source-file hydration"):
        service.index_record(deferred.record_id, vectors)
    with pytest.raises(ValueError, match="before source-file hydration"):
        service.promote_for_next_release(deferred.record_id, "release-1", _proof())

    artifact_root = tmp_path / "artifacts"
    hydrated = LegacyArchiveImporter(
        root, artifacts=ContentAddressedArtifactStore(artifact_root)
    ).import_all(service)[0]
    assert hydrated.record_id == deferred.record_id
    assert hydrated.evaluation["requires_hydration"] is False
    assert hydrated.evaluation["hydration_verified"] is True
    assert hydrated.evaluation["requires_resimulation"] is True
    assert hydrated.evaluation["requires_evaluation"] is True
    assert hydrated.provenance["legacy_source_index_sha256"]
    assert hydrated.provenance["legacy_index_entry_sha256"]
    assert len(repository.records) == 1
    assert (
        len(
            [
                path
                for path in (artifact_root / "objects" / "sha256").rglob("*")
                if path.is_file()
            ]
        )
        == 1
    )

    promoted = service.promote_for_next_release(hydrated.record_id, "release-1", _proof())
    assert promoted.status is AnimationStatus.STAGED


def test_metadata_import_refuses_index_drift_duplicate_or_unsafe_child_before_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    _archive(root)
    index_path, entry = _index(root)
    service, repository = _service()
    importer = DeferredLegacyIndexImporter(root)
    importer.import_all(service)

    document = json.loads(index_path.read_text(encoding="utf-8"))
    document["results"][0]["prompt"] = "Index changed after quarantine."
    index_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="different content"):
        DeferredLegacyIndexImporter(root).import_all(service)

    duplicate_root = tmp_path / "duplicate"
    _archive(duplicate_root)
    _, duplicate_entry = _index(duplicate_root)
    (duplicate_root / "index.json").write_text(
        json.dumps({"schema_version": "1.0", "results": [duplicate_entry, duplicate_entry]}),
        encoding="utf-8",
    )
    duplicate_service, duplicate_repository = _service()
    with pytest.raises(ValueError, match="duplicate ID"):
        DeferredLegacyIndexImporter(duplicate_root).import_all(duplicate_service)
    assert duplicate_repository.records == {}

    unsafe_root = tmp_path / "unsafe"
    unsafe_root.mkdir()
    _index(unsafe_root, source_id="../escape")
    unsafe_service, unsafe_repository = _service()
    with pytest.raises(ValueError, match="unsafe legacy index result ID"):
        DeferredLegacyIndexImporter(unsafe_root).import_all(unsafe_service)
    assert unsafe_repository.records == {}


def test_metadata_mode_is_an_explicit_cli_choice() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "v2" / "import_legacy_archive.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--mode {full,metadata-only}" in result.stdout
    assert "without copying payloads" in result.stdout
