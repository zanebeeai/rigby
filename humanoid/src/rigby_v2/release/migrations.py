"""Forward-only, checksum-pinned transactional migrations."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psycopg
from psycopg.rows import dict_row

from rigby_core.hashing import content_hash

from .errors import MigrationChecksumError, MigrationHistoryError


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sqlite_statements: tuple[str, ...]
    postgres_statements: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.version <= 0 or not self.name.strip() or not self.sqlite_statements:
            raise ValueError("migration version, name, and SQL statements are required")
        statements = self.sqlite_statements + (self.postgres_statements or ())
        transaction_tokens = re.compile(r"\b(BEGIN|COMMIT|ROLLBACK)\b", re.IGNORECASE)
        if any(not statement.strip() or transaction_tokens.search(statement) for statement in statements):
            raise ValueError("migration statements cannot be blank or manage transactions")

    @property
    def checksum(self) -> str:
        return content_hash(
            {
                "version": self.version,
                "name": self.name,
                "sqlite_statements": self.sqlite_statements,
                "postgres_statements": self.postgres_statements,
            }
        )

    def statements(self, dialect: str) -> tuple[str, ...]:
        if dialect == "postgres" and self.postgres_statements is not None:
            return self.postgres_statements
        return self.sqlite_statements


@dataclass(frozen=True)
class AppliedMigration:
    version: int
    name: str
    checksum: str


class MigrationAdapter(Protocol):
    dialect: str

    def initialize(self) -> None: ...
    def applied(self) -> tuple[AppliedMigration, ...]: ...
    def apply(self, migration: Migration) -> bool: ...


def _validate_migration_set(value: str) -> str:
    if not value or len(value) > 128 or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("migration_set must use 1-128 safe identifier characters")
    return value


class SQLiteMigrationAdapter:
    dialect = "sqlite"

    def __init__(self, path: Path, *, migration_set: str = "rigby-v2") -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migration_set = _validate_migration_set(migration_set)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rigby_release_migrations (
                    migration_set TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version > 0),
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL CHECK (length(checksum) = 64),
                    applied_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    PRIMARY KEY (migration_set, version)
                )
                """
            )

    def applied(self) -> tuple[AppliedMigration, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT version, name, checksum FROM rigby_release_migrations
                WHERE migration_set = ? ORDER BY version
                """,
                (self.migration_set,),
            ).fetchall()
        return tuple(AppliedMigration(int(row["version"]), row["name"], row["checksum"]) for row in rows)

    def apply(self, migration: Migration) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT name, checksum FROM rigby_release_migrations
                    WHERE migration_set = ? AND version = ?
                    """,
                    (self.migration_set, migration.version),
                ).fetchone()
                if row is not None:
                    if row["name"] != migration.name or row["checksum"] != migration.checksum:
                        raise MigrationChecksumError(
                            f"migration {migration.version} checksum or name drifted"
                        )
                    connection.commit()
                    return False
                for statement in migration.statements(self.dialect):
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO rigby_release_migrations (
                        migration_set, version, name, checksum
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (self.migration_set, migration.version, migration.name, migration.checksum),
                )
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise


class PostgresMigrationAdapter:
    dialect = "postgres"

    def __init__(self, database_url: str, *, migration_set: str = "rigby-v2") -> None:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgresMigrationAdapter requires a PostgreSQL URL")
        self.database_url = database_url
        self.migration_set = _validate_migration_set(migration_set)

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize(self) -> None:
        with self._connect() as connection, connection.transaction():
            connection.execute("CREATE SCHEMA IF NOT EXISTS rigby_v2")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rigby_v2.release_schema_migrations (
                    migration_set text NOT NULL,
                    version integer NOT NULL CHECK (version > 0),
                    name text NOT NULL,
                    checksum text NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
                    applied_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (migration_set, version)
                )
                """
            )

    def applied(self) -> tuple[AppliedMigration, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT version, name, checksum
                FROM rigby_v2.release_schema_migrations
                WHERE migration_set = %s ORDER BY version
                """,
                (self.migration_set,),
            ).fetchall()
        return tuple(AppliedMigration(int(row["version"]), row["name"], row["checksum"]) for row in rows)

    def apply(self, migration: Migration) -> bool:
        with self._connect() as connection, connection.transaction():
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))", (self.migration_set,)
            )
            row = connection.execute(
                """
                SELECT name, checksum FROM rigby_v2.release_schema_migrations
                WHERE migration_set = %s AND version = %s FOR UPDATE
                """,
                (self.migration_set, migration.version),
            ).fetchone()
            if row is not None:
                if row["name"] != migration.name or row["checksum"] != migration.checksum:
                    raise MigrationChecksumError(
                        f"migration {migration.version} checksum or name drifted"
                    )
                return False
            for statement in migration.statements(self.dialect):
                connection.execute(statement)
            connection.execute(
                """
                INSERT INTO rigby_v2.release_schema_migrations (
                    migration_set, version, name, checksum
                ) VALUES (%s, %s, %s, %s)
                """,
                (self.migration_set, migration.version, migration.name, migration.checksum),
            )
            return True


class MigrationRunner:
    def __init__(self, adapter: MigrationAdapter, migrations: tuple[Migration, ...]) -> None:
        self.adapter = adapter
        self.migrations = migrations
        versions = [migration.version for migration in migrations]
        if versions != sorted(versions) or len(versions) != len(set(versions)):
            raise ValueError("migrations must have unique strictly increasing versions")

    def apply(self) -> tuple[AppliedMigration, ...]:
        self.adapter.initialize()
        applied = self.adapter.applied()
        if len(applied) > len(self.migrations):
            raise MigrationHistoryError("database contains migrations newer than this binary")
        for index, observed in enumerate(applied):
            expected = self.migrations[index]
            if observed.version != expected.version:
                raise MigrationHistoryError(
                    "database migration history is not a prefix of the supplied plan"
                )
            if observed.name != expected.name or observed.checksum != expected.checksum:
                raise MigrationChecksumError(
                    f"migration {observed.version} checksum or name drifted"
                )
        newly_applied: list[AppliedMigration] = []
        for migration in self.migrations[len(applied) :]:
            if self.adapter.apply(migration):
                newly_applied.append(
                    AppliedMigration(migration.version, migration.name, migration.checksum)
                )
        return tuple(newly_applied)
