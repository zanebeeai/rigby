from __future__ import annotations

import json
import zipfile

import pytest

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.hashing import canonical_json_bytes
from rigby_v2.jobs import SQLiteJobStore
from rigby_v2.release import (
    ReleaseIntegrityError,
    create_backup,
    export_sqlite_job_metadata,
    restore_backup,
    restore_sqlite_job_metadata,
    verify_backup,
)

from test_v2_artifacts_jobs import simulation_job

pytestmark = pytest.mark.fast


def _source(tmp_path):
    artifacts = ContentAddressedArtifactStore(tmp_path / "source-artifacts")
    trace = artifacts.put_bytes(b"authoritative trace", filename="trace.json")
    video = artifacts.put_bytes(b"orbit evidence", media_type="video/mp4")
    jobs = (
        {
            "job_id": "job-1",
            "state": "succeeded",
            "trace_sha256": trace.sha256,
        },
        {"job_id": "job-2", "state": "queued"},
    )
    library = {
        "releases": [
            {"release_id": "release-1", "status": "active"},
            {"release_id": "release-2", "status": "building"},
        ],
        "records": [
            {
                "record_id": "record-active",
                "release_id": "release-1",
                "status": "certified",
            },
            {
                "record_id": "record-next",
                "release_id": "release-2",
                "status": "staged",
            },
        ],
    }
    return artifacts, trace, video, jobs, library


def test_backup_restore_round_trip_preserves_artifacts_jobs_and_library_metadata(tmp_path) -> None:
    artifacts, trace, video, jobs, library = _source(tmp_path)
    archive = tmp_path / "release.backup"

    manifest = create_backup(
        archive,
        backup_id="backup-1",
        artifact_root=artifacts.root,
        jobs=jobs,
        library=library,
    )
    verified = verify_backup(archive)
    restored = restore_backup(verified, tmp_path / "restored")

    assert manifest.artifact_count == 2
    assert manifest.job_count == 2
    assert manifest.active_release_ids == ("release-1",)
    assert manifest.staged_release_ids == ("release-2",)
    assert restored.jobs == jobs
    assert restored.library == library
    restored_store = ContentAddressedArtifactStore(restored.artifacts_root)
    assert restored_store.read_bytes(trace) == b"authoritative trace"
    assert restored_store.read_bytes(video) == b"orbit evidence"
    assert json.loads((restored.root / "metadata/jobs.json").read_text()) == list(jobs)


def test_backup_and_restore_never_overwrite_existing_paths(tmp_path) -> None:
    artifacts, _, _, jobs, library = _source(tmp_path)
    archive = tmp_path / "release.backup"
    create_backup(
        archive,
        backup_id="backup-no-overwrite",
        artifact_root=artifacts.root,
        jobs=jobs,
        library=library,
    )
    with pytest.raises(FileExistsError):
        create_backup(
            archive,
            backup_id="second",
            artifact_root=artifacts.root,
            jobs=jobs,
            library=library,
        )

    target = tmp_path / "existing-restore"
    target.mkdir()
    marker = target / "preserve.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        restore_backup(verify_backup(archive), target)
    assert marker.read_text() == "preserve"


def _rewrite(source, destination, transform):
    with zipfile.ZipFile(source, "r") as input_archive, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as output_archive:
        for info in input_archive.infolist():
            output_archive.writestr(
                info.filename,
                transform(info.filename, input_archive.read(info.filename)),
            )


def test_corrupt_backup_and_tampered_manifest_are_refused_before_restore(tmp_path) -> None:
    artifacts, _, _, jobs, library = _source(tmp_path)
    archive = tmp_path / "release.backup"
    manifest = create_backup(
        archive,
        backup_id="backup-corruption",
        artifact_root=artifacts.root,
        jobs=jobs,
        library=library,
    )
    artifact_entry = next(
        item for item in manifest.files if item.kind == "content_addressed_artifact"
    )
    corrupt = tmp_path / "corrupt.backup"
    _rewrite(
        archive,
        corrupt,
        lambda name, payload: b"corrupt" if name == artifact_entry.path else payload,
    )
    with pytest.raises(ReleaseIntegrityError, match="corrupt"):
        verify_backup(corrupt)
    assert not (tmp_path / "corrupt-restore").exists()

    tampered = tmp_path / "tampered.backup"

    def alter_manifest(name, payload):
        if name != "backup.json":
            return payload
        envelope = json.loads(payload)
        envelope["manifest"]["backup_id"] = "attacker"
        return canonical_json_bytes(envelope)

    _rewrite(archive, tampered, alter_manifest)
    with pytest.raises(ReleaseIntegrityError, match="manifest checksum"):
        verify_backup(tampered)


def test_backup_refuses_corrupt_source_artifact(tmp_path) -> None:
    artifacts, trace, _, jobs, library = _source(tmp_path)
    artifacts.resolve(trace, verify=False).write_bytes(b"tampered source")

    with pytest.raises(ReleaseIntegrityError, match="corrupt before backup"):
        create_backup(
            tmp_path / "must-not-exist.backup",
            backup_id="bad-source",
            artifact_root=artifacts.root,
            jobs=jobs,
            library=library,
        )
    assert not (tmp_path / "must-not-exist.backup").exists()


def test_sqlite_job_metadata_restores_to_a_fresh_durable_store(tmp_path) -> None:
    source_database = tmp_path / "source-jobs.sqlite3"
    source_store = SQLiteJobStore(source_database)
    source_store.submit(simulation_job("release-backup-job"), idempotency_key="request-1")
    rows = export_sqlite_job_metadata(source_database)
    artifacts = ContentAddressedArtifactStore(tmp_path / "job-artifacts")
    archive = tmp_path / "jobs.backup"
    create_backup(
        archive,
        backup_id="sqlite-job-roundtrip",
        artifact_root=artifacts.root,
        jobs=rows,
        library={"releases": [], "records": []},
    )
    restored_bundle = restore_backup(verify_backup(archive), tmp_path / "job-restore")
    restored_database = restore_sqlite_job_metadata(
        restored_bundle.root / "jobs.sqlite3", restored_bundle.jobs
    )

    restored = SQLiteJobStore(restored_database).get("release-backup-job")
    assert restored.job == source_store.get("release-backup-job").job
    assert restored.idempotency_key == "request-1"
    with pytest.raises(FileExistsError):
        restore_sqlite_job_metadata(restored_database, rows)
