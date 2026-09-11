"""Canonical forward migrations for the persistent Rigby v2 PostgreSQL schema."""

from __future__ import annotations

from .migrations import AppliedMigration, Migration, MigrationRunner, PostgresMigrationAdapter


RIGBY_V2_SCHEMA_MIGRATION_SET = "rigby-v2-schema"

ALLOW_LEGACY_SPLIT_POSTGRES = (
    """
    ALTER TABLE rigby_v2.animation_records
    DROP CONSTRAINT IF EXISTS animation_records_split_check
    """,
    """
    ALTER TABLE rigby_v2.animation_records
    ADD CONSTRAINT animation_records_split_check CHECK (
        split IN ('train', 'development', 'test', 'production', 'unassigned', 'legacy')
    ) NOT VALID
    """,
    """
    ALTER TABLE rigby_v2.animation_records
    VALIDATE CONSTRAINT animation_records_split_check
    """,
)

RIGBY_V2_SCHEMA_MIGRATIONS = (
    Migration(
        version=1,
        name="allow_quarantined_legacy_animation_split",
        sqlite_statements=("SELECT 1",),
        postgres_statements=ALLOW_LEGACY_SPLIT_POSTGRES,
    ),
)


def apply_rigby_v2_postgres_migrations(database_url: str) -> tuple[AppliedMigration, ...]:
    """Apply the checksum-pinned canonical schema plan transactionally."""

    return MigrationRunner(
        PostgresMigrationAdapter(
            database_url,
            migration_set=RIGBY_V2_SCHEMA_MIGRATION_SET,
        ),
        RIGBY_V2_SCHEMA_MIGRATIONS,
    ).apply()
