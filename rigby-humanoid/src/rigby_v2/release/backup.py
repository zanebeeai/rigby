"""Integrity-checked backup and non-overwriting atomic restore."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rigby_v2.hashing import canonical_json_bytes, hash_file, sha256_bytes, validate_sha256

from .errors import ReleaseIntegrityError


MAX_BACKUP_ENTRIES = 100_000
MAX_BACKUP_UNCOMPRESSED_BYTES = 64 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class BackupFile:
    path: str
    sha256: str
    size_bytes: int
    kind: str

    def __post_init__(self) -> None:
        validate_sha256(self.sha256)
        pure = PurePosixPath(self.path)
        if pure.is_absolute() or ".." in pure.parts or "\\" in self.path:
            raise ValueError("backup paths must be safe relative POSIX paths")
        if self.size_bytes < 0 or not self.kind:
            raise ValueError("backup file metadata is invalid")


@dataclass(frozen=True)
class BackupManifest:
    schema_version: str
    backup_id: str
    files: tuple[BackupFile, ...]
    active_release_ids: tuple[str, ...]
    staged_release_ids: tuple[str, ...]
    artifact_count: int
    job_count: int

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or not self.backup_id:
            raise ValueError("unsupported backup manifest")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("backup manifest contains duplicate paths")
        if len(self.active_release_ids) > 1:
            raise ValueError("backup metadata cannot contain multiple active releases")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackupManifest":
        return cls(
            schema_version=str(value["schema_version"]),
            backup_id=str(value["backup_id"]),
            files=tuple(BackupFile(**item) for item in value["files"]),
            active_release_ids=tuple(value["active_release_ids"]),
            staged_release_ids=tuple(value["staged_release_ids"]),
            artifact_count=int(value["artifact_count"]),
            job_count=int(value["job_count"]),
        )


@dataclass(frozen=True)
class VerifiedBackup:
    path: Path
    manifest: BackupManifest
    manifest_sha256: str
    jobs: tuple[dict[str, Any], ...]
    library: dict[str, Any]


@dataclass(frozen=True)
class RestoredBackup:
    root: Path
    artifacts_root: Path
    jobs: tuple[dict[str, Any], ...]
    library: dict[str, Any]


def _artifact_files(artifact_root: Path) -> list[tuple[Path, str]]:
    objects = artifact_root.resolve() / "objects" / "sha256"
    if not objects.exists():
        return []
    files: list[tuple[Path, str]] = []
    for path in sorted(item for item in objects.rglob("*") if item.is_file()):
        relative = path.relative_to(objects)
        parts = relative.parts
        if len(parts) != 2 or len(parts[0]) != 2:
            raise ReleaseIntegrityError(f"non-content-addressed artifact path: {relative}")
        digest = parts[0] + parts[1]
        validate_sha256(digest)
        if hash_file(path) != digest:
            raise ReleaseIntegrityError(f"artifact {digest} is corrupt before backup")
        files.append((path, f"artifacts/objects/sha256/{parts[0]}/{parts[1]}"))
    return files


def _publish_file_no_replace(temporary: Path | str, destination: Path) -> None:
    """Atomically publish a same-filesystem file without overwriting a race winner."""

    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(destination) from None
    Path(temporary).unlink()


def _release_ids(library: Mapping[str, Any], status: str) -> tuple[str, ...]:
    releases = library.get("releases")
    if not isinstance(releases, Sequence) or isinstance(releases, (str, bytes)):
        raise ValueError("library metadata must contain a releases sequence")
    return tuple(
        sorted(
            str(release["release_id"])
            for release in releases
            if isinstance(release, Mapping) and release.get("status") == status
        )
    )


def _staged_release_ids(library: Mapping[str, Any]) -> tuple[str, ...]:
    records = library.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("library metadata must contain a records sequence")
    return tuple(
        sorted(
            {
                str(record["release_id"])
                for record in records
                if isinstance(record, Mapping)
                and record.get("status") == "staged"
                and record.get("release_id")
            }
        )
    )


SQLITE_JOB_COLUMNS = (
    "job_id",
    "job_json",
    "job_hash",
    "state",
    "priority",
    "attempts",
    "available_at",
    "created_at",
    "updated_at",
    "lease_owner",
    "lease_expires_at",
    "heartbeat_at",
    "cancel_requested",
    "result_json",
    "failure_json",
    "idempotency_key",
)


def export_sqlite_job_metadata(database: Path) -> tuple[dict[str, Any], ...]:
    """Read a consistent, portable snapshot of the local durable job table."""

    database = database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        )
        if columns != SQLITE_JOB_COLUMNS:
            raise ReleaseIntegrityError("SQLite jobs schema is incompatible with this release")
        rows = connection.execute("SELECT * FROM jobs ORDER BY job_id").fetchall()
        connection.commit()
    return tuple({column: row[column] for column in SQLITE_JOB_COLUMNS} for row in rows)


def restore_sqlite_job_metadata(
    destination: Path,
    jobs: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically create a fresh SQLite job database; never merge or overwrite."""

    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.restore-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    sidecars = (
        Path(f"{temporary}-wal"),
        Path(f"{temporary}-shm"),
    )
    try:
        with closing(
            sqlite3.connect(temporary, timeout=30, isolation_level=None)
        ) as connection:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE jobs (
                    job_id TEXT PRIMARY KEY,
                    job_json TEXT NOT NULL,
                    job_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('queued','leased','succeeded','failed','cancelled')
                    ),
                    priority INTEGER NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    available_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (
                        cancel_requested IN (0,1)
                    ),
                    result_json TEXT,
                    failure_json TEXT,
                    idempotency_key TEXT UNIQUE
                );
                CREATE INDEX jobs_ready_idx
                    ON jobs(state, available_at, priority DESC, created_at ASC);
                CREATE INDEX jobs_lease_expiry_idx
                    ON jobs(state, lease_expires_at);
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in SQLITE_JOB_COLUMNS)
            columns_sql = ",".join(SQLITE_JOB_COLUMNS)
            for job in jobs:
                if set(job) != set(SQLITE_JOB_COLUMNS):
                    raise ReleaseIntegrityError("job metadata columns are incomplete or unexpected")
                connection.execute(
                    f"INSERT INTO jobs ({columns_sql}) VALUES ({placeholders})",
                    tuple(job[column] for column in SQLITE_JOB_COLUMNS),
                )
            connection.commit()
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ReleaseIntegrityError("restored SQLite job database failed integrity check")
        _publish_file_no_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        for sidecar in sidecars:
            sidecar.unlink(missing_ok=True)
    return destination


def create_backup(
    destination: Path,
    *,
    backup_id: str,
    artifact_root: Path,
    jobs: Sequence[Mapping[str, Any]],
    library: Mapping[str, Any],
) -> BackupManifest:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "records" not in library:
        raise ValueError("library metadata must include active/staged records")
    jobs_bytes = canonical_json_bytes(list(jobs))
    library_bytes = canonical_json_bytes(dict(library))
    source_files = _artifact_files(artifact_root)
    entries: list[BackupFile] = [
        BackupFile("metadata/jobs.json", sha256_bytes(jobs_bytes), len(jobs_bytes), "job_metadata"),
        BackupFile(
            "metadata/library.json",
            sha256_bytes(library_bytes),
            len(library_bytes),
            "library_metadata",
        ),
    ]
    for source, archive_path in source_files:
        entries.append(
            BackupFile(
                archive_path,
                hash_file(source),
                source.stat().st_size,
                "content_addressed_artifact",
            )
        )
    manifest = BackupManifest(
        schema_version="1.0",
        backup_id=backup_id,
        files=tuple(entries),
        active_release_ids=_release_ids(library, "active"),
        staged_release_ids=_staged_release_ids(library),
        artifact_count=len(source_files),
        job_count=len(jobs),
    )
    manifest_dict = manifest.to_dict()
    manifest_hash = sha256_bytes(canonical_json_bytes(manifest_dict))
    envelope = canonical_json_bytes(
        {"manifest": manifest_dict, "manifest_sha256": manifest_hash}
    )
    source_by_archive = {archive: source for source, archive in source_files}

    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.staging-", delete=False
        ) as stream:
            temporary = stream.name
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("backup.json", envelope)
            archive.writestr("metadata/jobs.json", jobs_bytes)
            archive.writestr("metadata/library.json", library_bytes)
            for archive_path, source in sorted(source_by_archive.items()):
                archive.write(source, archive_path)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        _publish_file_no_replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return manifest


def verify_backup(path: Path) -> VerifiedBackup:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(infos) > MAX_BACKUP_ENTRIES or len(names) != len(set(names)):
                raise ReleaseIntegrityError("backup has too many or duplicate entries")
            if sum(info.file_size for info in infos) > MAX_BACKUP_UNCOMPRESSED_BYTES:
                raise ReleaseIntegrityError("backup expands beyond the safety limit")
            for name in names:
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                    raise ReleaseIntegrityError("backup contains an unsafe path")
            if "backup.json" not in names:
                raise ReleaseIntegrityError("backup manifest is missing")
            envelope = json.loads(archive.read("backup.json"))
            manifest_dict = envelope["manifest"]
            expected_hash = validate_sha256(envelope["manifest_sha256"])
            actual_hash = sha256_bytes(canonical_json_bytes(manifest_dict))
            if actual_hash != expected_hash:
                raise ReleaseIntegrityError("backup manifest checksum mismatch")
            manifest = BackupManifest.from_dict(manifest_dict)
            expected_names = {"backup.json", *(item.path for item in manifest.files)}
            if set(names) != expected_names:
                raise ReleaseIntegrityError("backup has missing or unbound files")
            payloads: dict[str, bytes] = {}
            for item in manifest.files:
                payload = archive.read(item.path)
                if len(payload) != item.size_bytes or sha256_bytes(payload) != item.sha256:
                    raise ReleaseIntegrityError(f"backup member {item.path!r} is corrupt")
                payloads[item.path] = payload
            jobs = tuple(json.loads(payloads["metadata/jobs.json"]))
            library = dict(json.loads(payloads["metadata/library.json"]))
            if len(jobs) != manifest.job_count:
                raise ReleaseIntegrityError("backup job count does not match metadata")
            if _release_ids(library, "active") != manifest.active_release_ids:
                raise ReleaseIntegrityError("active library release metadata does not match manifest")
            if _staged_release_ids(library) != manifest.staged_release_ids:
                raise ReleaseIntegrityError("staged library release metadata does not match manifest")
            artifact_count = sum(
                item.kind == "content_addressed_artifact" for item in manifest.files
            )
            if artifact_count != manifest.artifact_count:
                raise ReleaseIntegrityError("backup artifact count does not match manifest")
    except (zipfile.BadZipFile, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ReleaseIntegrityError):
            raise
        raise ReleaseIntegrityError(f"invalid backup: {exc}") from exc
    return VerifiedBackup(path.resolve(), manifest, actual_hash, jobs, library)


def restore_backup(backup: VerifiedBackup, destination: Path) -> RestoredBackup:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reverify immediately before materializing anything. The archive may have
    # changed after an earlier verification call.
    verified = verify_backup(backup.path)
    if verified.manifest_sha256 != backup.manifest_sha256:
        raise ReleaseIntegrityError("backup changed after verification")
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent)
    )
    try:
        with zipfile.ZipFile(backup.path, "r") as archive:
            for item in verified.manifest.files:
                assert staging is not None
                target = staging / item.path
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = archive.read(item.path)
                with target.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
        if destination.exists():
            raise FileExistsError(destination)
        assert staging is not None
        # On the supported local Windows runtime os.rename is an atomic,
        # no-replace directory publication. The immediately preceding check
        # also keeps the intent explicit on other platforms.
        os.rename(staging, destination)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
    return RestoredBackup(
        root=destination,
        artifacts_root=destination / "artifacts",
        jobs=verified.jobs,
        library=verified.library,
    )
