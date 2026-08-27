CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS rigby_v2;

CREATE TABLE IF NOT EXISTS rigby_v2.artifacts (
    sha256 text PRIMARY KEY CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    byte_size bigint NOT NULL CHECK (byte_size >= 0),
    media_type text NOT NULL,
    relative_path text NOT NULL UNIQUE,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rigby_v2.simulation_jobs (
    job_id text PRIMARY KEY,
    schema_version text NOT NULL,
    status text NOT NULL CONSTRAINT simulation_jobs_status_check CHECK (
        status IN ('queued', 'leased', 'cancelled', 'succeeded', 'failed')
    ),
    priority integer NOT NULL DEFAULT 0,
    payload jsonb NOT NULL,
    payload_sha256 text NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
    result jsonb,
    result_sha256 text CHECK (result_sha256 IS NULL OR result_sha256 ~ '^[0-9a-f]{64}$'),
    failure jsonb,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    max_attempts integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    available_at timestamptz NOT NULL DEFAULT now(),
    lease_owner text,
    lease_expires_at timestamptz,
    heartbeat_at timestamptz,
    cancel_requested boolean NOT NULL DEFAULT false,
    idempotency_key text UNIQUE,
    error_code text,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    UNIQUE (payload_sha256, job_id),
    CHECK ((status = 'leased') = (lease_owner IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS simulation_jobs_claim_idx
    ON rigby_v2.simulation_jobs (priority DESC, created_at, job_id)
    WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS simulation_jobs_lease_idx
    ON rigby_v2.simulation_jobs (lease_expires_at)
    WHERE status = 'leased';

CREATE TABLE IF NOT EXISTS rigby_v2.animation_records (
    record_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version text NOT NULL,
    status text NOT NULL CHECK (
        status IN ('legacy_candidate', 'candidate', 'staged', 'certified', 'quarantined', 'deprecated')
    ),
    release_id text,
    split text NOT NULL CONSTRAINT animation_records_split_check CHECK (
        split IN ('train', 'development', 'test', 'production', 'unassigned', 'legacy')
    ),
    prompt jsonb NOT NULL,
    program jsonb NOT NULL,
    world jsonb NOT NULL,
    motion jsonb NOT NULL,
    evidence jsonb NOT NULL,
    labels jsonb NOT NULL,
    evaluation jsonb NOT NULL,
    provenance jsonb NOT NULL,
    program_sha256 text NOT NULL CHECK (program_sha256 ~ '^[0-9a-f]{64}$'),
    world_sha256 text NOT NULL CHECK (world_sha256 ~ '^[0-9a-f]{64}$'),
    motion_sha256 text NOT NULL CHECK (motion_sha256 ~ '^[0-9a-f]{64}$'),
    parent_record_id uuid REFERENCES rigby_v2.animation_records(record_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS animation_records_filter_idx
    ON rigby_v2.animation_records (status, release_id, split);
CREATE INDEX IF NOT EXISTS animation_records_program_hash_idx
    ON rigby_v2.animation_records (program_sha256);
CREATE INDEX IF NOT EXISTS animation_records_motion_hash_idx
    ON rigby_v2.animation_records (motion_sha256);

CREATE TABLE IF NOT EXISTS rigby_v2.failure_records (
    failure_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version text NOT NULL,
    job_id text REFERENCES rigby_v2.simulation_jobs(job_id),
    candidate_record_id uuid REFERENCES rigby_v2.animation_records(record_id),
    stage text NOT NULL,
    failure_code text NOT NULL,
    failed_predicate text,
    measurements jsonb NOT NULL DEFAULT '{}'::jsonb,
    artifacts jsonb NOT NULL DEFAULT '{}'::jsonb,
    attempted_repair jsonb,
    provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rigby_v2.embedding_namespaces (
    namespace text NOT NULL,
    version text NOT NULL,
    model_name text NOT NULL,
    model_sha256 text NOT NULL CHECK (model_sha256 ~ '^[0-9a-f]{64}$'),
    preprocessing_version text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions > 0),
    distance_metric text NOT NULL CHECK (distance_metric IN ('cosine', 'inner_product', 'l2')),
    active boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, version)
);

CREATE TABLE IF NOT EXISTS rigby_v2.embeddings (
    record_id uuid NOT NULL REFERENCES rigby_v2.animation_records(record_id) ON DELETE CASCADE,
    namespace text NOT NULL,
    namespace_version text NOT NULL,
    segment_id text NOT NULL DEFAULT 'global',
    embedding vector NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (record_id, namespace, namespace_version, segment_id),
    FOREIGN KEY (namespace, namespace_version)
        REFERENCES rigby_v2.embedding_namespaces(namespace, version)
);

CREATE TABLE IF NOT EXISTS rigby_v2.library_releases (
    release_id text PRIMARY KEY,
    parent_release_id text REFERENCES rigby_v2.library_releases(release_id),
    status text NOT NULL CHECK (status IN ('building', 'frozen', 'active', 'retired')),
    manifest_sha256 text CHECK (manifest_sha256 IS NULL OR manifest_sha256 ~ '^[0-9a-f]{64}$'),
    manifest jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    frozen_at timestamptz
);

CREATE TABLE IF NOT EXISTS rigby_v2.benchmark_cases (
    case_id text PRIMARY KEY,
    benchmark_version text NOT NULL,
    family text NOT NULL,
    supported boolean NOT NULL,
    prompt text NOT NULL,
    scene_manifest jsonb NOT NULL,
    expected_outcomes jsonb NOT NULL,
    lineage_group text NOT NULL,
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (benchmark_version, content_sha256)
);

CREATE TABLE IF NOT EXISTS rigby_v2.human_comparisons (
    comparison_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    calibration_version text NOT NULL,
    family text NOT NULL,
    left_record_id uuid REFERENCES rigby_v2.animation_records(record_id),
    right_record_id uuid REFERENCES rigby_v2.animation_records(record_id),
    presentation_order jsonb NOT NULL,
    rater_id_hash text NOT NULL,
    verdict text NOT NULL CHECK (verdict IN ('left', 'right', 'tie', 'abstain', 'both_fail')),
    rubric jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (calibration_version, left_record_id, right_record_id, rater_id_hash, presentation_order)
);
