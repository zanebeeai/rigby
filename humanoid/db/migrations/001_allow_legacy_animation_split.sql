-- Forward migration for databases initialized before deferred legacy quarantine.
-- Applied transactionally by rigby_v2.release.schema_migrations.
ALTER TABLE rigby_v2.animation_records
    DROP CONSTRAINT IF EXISTS animation_records_split_check;

ALTER TABLE rigby_v2.animation_records
    ADD CONSTRAINT animation_records_split_check CHECK (
        split IN ('train', 'development', 'test', 'production', 'unassigned', 'legacy')
    ) NOT VALID;

ALTER TABLE rigby_v2.animation_records
    VALIDATE CONSTRAINT animation_records_split_check;
