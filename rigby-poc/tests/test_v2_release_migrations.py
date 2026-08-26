from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from rigby_v2.release import (
    ALLOW_LEGACY_SPLIT_POSTGRES,
    Migration,
    MigrationChecksumError,
    MigrationHistoryError,
    MigrationRunner,
    PostgresMigrationAdapter,
    RIGBY_V2_SCHEMA_MIGRATIONS,
    SQLiteMigrationAdapter,
)


MIGRATIONS = (
    Migration(
        1,
        "create_release_items",
        ("CREATE TABLE release_items (item_id TEXT PRIMARY KEY, value TEXT NOT NULL)",),
    ),
    Migration(
        2,
        "add_release_item_index",
        ("CREATE INDEX release_items_value_idx ON release_items(value)",),
    ),
)


def test_canonical_forward_schema_migration_and_fresh_init_allow_legacy_split() -> None:
    root = Path(__file__).resolve().parents[1]
    init_sql = (root / "db" / "init" / "001_rigby_v2.sql").read_text(encoding="utf-8")
    forward_sql = (
        root / "db" / "migrations" / "001_allow_legacy_animation_split.sql"
    ).read_text(encoding="utf-8")
    assert "'unassigned', 'legacy'" in init_sql
    assert "'unassigned', 'legacy'" in forward_sql
    assert "VALIDATE CONSTRAINT animation_records_split_check" in forward_sql
    migration = RIGBY_V2_SCHEMA_MIGRATIONS[0]
    assert migration.name == "allow_quarantined_legacy_animation_split"
    assert any("'unassigned', 'legacy'" in value for value in migration.postgres_statements)
    executable_sql = "\n".join(
        line for line in forward_sql.splitlines() if not line.lstrip().startswith("--")
    )
    file_statements = tuple(
        value.strip()
        for value in executable_sql.split(";")
        if value.strip()
    )
    def normalize(value: str) -> str:
        return " ".join(value.split())

    assert tuple(map(normalize, file_statements)) == tuple(
        map(normalize, ALLOW_LEGACY_SPLIT_POSTGRES)
    )


def test_fresh_sqlite_migration_is_forward_only_idempotent_and_restart_safe(tmp_path) -> None:
    database = tmp_path / "fresh.sqlite3"
    first = MigrationRunner(SQLiteMigrationAdapter(database), MIGRATIONS)

    applied = first.apply()

    assert [migration.version for migration in applied] == [1, 2]
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO release_items VALUES ('preserved', 'value')")

    restarted = MigrationRunner(SQLiteMigrationAdapter(database), MIGRATIONS)
    assert restarted.apply() == ()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT value FROM release_items WHERE item_id = 'preserved'"
        ).fetchone() == ("value",)
        assert connection.execute(
            "SELECT count(*) FROM rigby_release_migrations"
        ).fetchone() == (2,)


def test_applied_migration_checksum_drift_is_refused_without_touching_data(tmp_path) -> None:
    database = tmp_path / "drift.sqlite3"
    MigrationRunner(SQLiteMigrationAdapter(database), MIGRATIONS).apply()
    drifted = (
        Migration(
            1,
            "create_release_items",
            ("CREATE TABLE release_items (item_id TEXT PRIMARY KEY, changed TEXT)",),
        ),
        MIGRATIONS[1],
    )

    with pytest.raises(MigrationChecksumError):
        MigrationRunner(SQLiteMigrationAdapter(database), drifted).apply()

    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(release_items)")]
    assert columns == ["item_id", "value"]


def test_failed_sqlite_migration_rolls_back_statements_and_history(tmp_path) -> None:
    database = tmp_path / "transaction.sqlite3"
    base = (Migration(1, "create", ("CREATE TABLE durable (value TEXT)",)),)
    MigrationRunner(SQLiteMigrationAdapter(database), base).apply()
    broken = base + (
        Migration(
            2,
            "broken",
            (
                "INSERT INTO durable VALUES ('must-roll-back')",
                "INSERT INTO missing_table VALUES ('failure')",
            ),
        ),
    )

    with pytest.raises(sqlite3.OperationalError):
        MigrationRunner(SQLiteMigrationAdapter(database), broken).apply()

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM durable").fetchall() == []
        versions = connection.execute(
            "SELECT version FROM rigby_release_migrations ORDER BY version"
        ).fetchall()
    assert versions == [(1,)]


def test_inserting_a_migration_behind_applied_history_is_refused(tmp_path) -> None:
    database = tmp_path / "forward-only.sqlite3"
    original = (
        Migration(1, "create", ("CREATE TABLE forward_only (value TEXT)",)),
        Migration(3, "index", ("CREATE INDEX forward_only_idx ON forward_only(value)",)),
    )
    MigrationRunner(SQLiteMigrationAdapter(database), original).apply()
    rewritten_history = (
        original[0],
        Migration(2, "late_insert", ("ALTER TABLE forward_only ADD COLUMN late TEXT",)),
        original[1],
    )

    with pytest.raises(MigrationHistoryError, match="not a prefix"):
        MigrationRunner(SQLiteMigrationAdapter(database), rewritten_history).apply()

    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(forward_only)")]
    assert columns == ["value"]


@pytest.mark.skipif(
    os.getenv("RIGBY_TEST_POSTGRES") != "1",
    reason="set RIGBY_TEST_POSTGRES=1 with the local v2 database running",
)
def test_postgres_adapter_applies_and_restarts_transactionally() -> None:
    database_url = os.getenv(
        "RIGBY_V2_DATABASE_URL",
        "postgresql://rigby:rigby-local-only@127.0.0.1:54329/rigby",
    )
    suffix = uuid4().hex
    migration_set = f"release-test-{suffix}"
    table = f"release_migration_{suffix}"
    migrations = (
        Migration(
            1,
            "create_postgres_release_table",
            (f"CREATE TABLE rigby_v2.{table} (value text)",),
        ),
    )
    adapter = PostgresMigrationAdapter(database_url, migration_set=migration_set)
    try:
        assert len(MigrationRunner(adapter, migrations).apply()) == 1
        assert MigrationRunner(
            PostgresMigrationAdapter(database_url, migration_set=migration_set),
            migrations,
        ).apply() == ()
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute(f"DROP TABLE IF EXISTS rigby_v2.{table}")
            connection.execute(
                """
                DELETE FROM rigby_v2.release_schema_migrations
                WHERE migration_set = %s
                """,
                (migration_set,),
            )
