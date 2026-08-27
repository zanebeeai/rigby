from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
from psycopg import sql

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.config import RuntimeSettings
from rigby_v2.release import create_backup, restore_backup, verify_backup
from rigby_v2.release_ops import seal_release_evidence


ROOT = Path(__file__).resolve().parents[2]
_DISPOSABLE_DATABASE = re.compile(r"^rigby_restore_[0-9a-f]{32}$")


def _database_url(base: str, database: str) -> str:
    parsed = urlsplit(base)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"/{database}", parsed.query, parsed.fragment)
    )


def _run_docker_postgres(
    *arguments: str,
    input_bytes: bytes | None = None,
) -> bytes:
    command = [
        "docker",
        "compose",
        "-f",
        str(ROOT / "compose.v2.yaml"),
        "exec",
        "-T",
        "postgres",
        *arguments,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Docker PostgreSQL command failed ({arguments[0]}): "
            + completed.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    return completed.stdout


def _table_counts(database_url: str) -> dict[str, int]:
    with psycopg.connect(database_url) as connection:
        names = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'rigby_v2' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ).fetchall()
        ]
        return {
            name: int(
                connection.execute(
                    sql.SQL("SELECT count(1) FROM rigby_v2.{}").format(
                        sql.Identifier(name)
                    )
                ).fetchone()[0]
            )
            for name in names
        }


def main() -> int:
    settings = RuntimeSettings.from_env()
    source_url = settings.database_url
    admin_url = _database_url(source_url, "postgres")
    restored_name = f"rigby_restore_{uuid4().hex}"
    if not _DISPOSABLE_DATABASE.fullmatch(restored_name):
        raise RuntimeError("refusing unsafe disposable database name")
    restored_url = _database_url(source_url, restored_name)
    evidence_root = settings.project_root / "artifacts-v2" / "release-evidence"
    payload: dict[str, object] | None = None

    with tempfile.TemporaryDirectory(prefix="rigby-backup-restore-") as temporary:
        temporary_root = Path(temporary)
        artifact_store = ContentAddressedArtifactStore(settings.artifact_root)
        canary = artifact_store.put_bytes(
            b"rigby-v2-backup-restore-canary-v1",
            media_type="application/octet-stream",
            filename="backup-restore-canary.bin",
        )
        portable_backup_path = temporary_root / "artifacts.rigby-backup"
        portable_manifest = create_backup(
            portable_backup_path,
            backup_id=f"backup-restore-{uuid4().hex}",
            artifact_root=settings.artifact_root,
            jobs=(),
            library={"releases": [], "records": []},
        )
        verified_portable = verify_backup(portable_backup_path)
        restored_portable = restore_backup(
            verified_portable, temporary_root / "portable-restored"
        )
        restored_canary = (
            restored_portable.artifacts_root
            / "objects"
            / "sha256"
            / canary.sha256[:2]
            / canary.sha256[2:]
        )
        if (
            not restored_canary.is_file()
            or hashlib.sha256(restored_canary.read_bytes()).hexdigest() != canary.sha256
        ):
            raise RuntimeError("restored content-addressed canary failed integrity")

        dump_bytes = _run_docker_postgres(
            "pg_dump",
            "-Fc",
            "--no-owner",
            "--no-privileges",
            "-U",
            "rigby",
            "-d",
            urlsplit(source_url).path.lstrip("/"),
        )
        if not dump_bytes:
            raise RuntimeError("PostgreSQL dump is empty")
        source_counts = _table_counts(source_url)
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(restored_name))
            )
        try:
            _run_docker_postgres(
                "pg_restore",
                "--no-owner",
                "--no-privileges",
                "-U",
                "rigby",
                "-d",
                restored_name,
                input_bytes=dump_bytes,
            )
            restored_counts = _table_counts(restored_url)
            if source_counts != restored_counts:
                raise RuntimeError("restored PostgreSQL table counts differ from source")
            if source_counts.get("animation_records") != 6354:
                raise RuntimeError("source backup does not contain the full legacy inventory")
            payload = {
                "source_database": "rigby",
                "dump_format": "postgresql_custom",
                "dump_sha256": hashlib.sha256(dump_bytes).hexdigest(),
                "dump_size_bytes": len(dump_bytes),
                "table_counts": source_counts,
                "restored_table_counts_identical": True,
                "portable_backup_manifest_sha256": verified_portable.manifest_sha256,
                "portable_artifact_count": portable_manifest.artifact_count,
                "content_addressed_canary_sha256": canary.sha256,
                "content_addressed_canary_restored": True,
            }
        finally:
            with psycopg.connect(admin_url, autocommit=True) as admin:
                admin.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND pid <> pg_backend_pid()",
                    (restored_name,),
                )
                admin.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(restored_name)
                    )
                )

    if payload is None:
        raise RuntimeError("backup/restore acceptance did not produce evidence")
    evidence = seal_release_evidence(
        evidence_root,
        requirement_id="database_backup_restore",
        passed=True,
        payload={**payload, "disposable_database_cleaned": True},
    )
    print(
        json.dumps(
            {
                "passed": True,
                "evidence": evidence.artifact_path,
                "evidence_sha256": evidence.artifact_sha256,
                **payload,
                "disposable_database_cleaned": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
