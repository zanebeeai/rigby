-- Rigby General: per-robot morphology and certified primitive bindings.
--
-- Deliberately a separate schema from rigby_v2 rather than an extension of it.
-- The v2 animation_records table is keyed to one certified humanoid; here every
-- record is keyed to a robot whose morphology was measured at upload time, and
-- the two lifecycles should be able to diverge without a shared migration.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS rigby_general;

-- One uploaded robot. `base_tree_sha256` records which rigby_v2 source measured
-- it, so a morphology can always be traced to the code that produced it.
CREATE TABLE IF NOT EXISTS rigby_general.robots (
    robot_id text PRIMARY KEY CHECK (robot_id ~ '^[a-z0-9][a-z0-9_-]{0,63}$'),
    display_name text NOT NULL,
    source_format text NOT NULL CHECK (source_format IN ('urdf', 'mjcf')),
    source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    mjcf_sha256 text NOT NULL CHECK (mjcf_sha256 ~ '^[0-9a-f]{64}$'),
    morphology_class text NOT NULL CHECK (
        morphology_class IN (
            'fixed_base_arm',
            'fixed_base_bimanual',
            'unsupported_floating_base',
            'unsupported_topology'
        )
    ),
    manifest jsonb NOT NULL,
    manifest_sha256 text NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    morphology jsonb NOT NULL,
    morphology_sha256 text NOT NULL CHECK (morphology_sha256 ~ '^[0-9a-f]{64}$'),
    -- An intrinsic frame that no person has confirmed cannot serve a deictic or
    -- INTRINSIC-frame schema; the grounder refuses rather than guessing "front".
    frame_source text NOT NULL DEFAULT 'derived'
        CHECK (frame_source IN ('derived', 'operator_confirmed')),
    asset_licenses jsonb NOT NULL DEFAULT '{}'::jsonb,
    base_tree_sha256 text NOT NULL CHECK (base_tree_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- One bake run over one robot. Records the ceilings it ran under, so an
-- incomplete library is always distinguishable from a complete one.
CREATE TABLE IF NOT EXISTS rigby_general.bakes (
    bake_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    robot_id text NOT NULL REFERENCES rigby_general.robots (robot_id) ON DELETE CASCADE,
    status text NOT NULL CHECK (
        status IN ('running', 'complete', 'budget_exhausted', 'failed')
    ),
    inventory_sha256 text NOT NULL CHECK (inventory_sha256 ~ '^[0-9a-f]{64}$'),
    budget_seconds integer NOT NULL CHECK (budget_seconds > 0),
    max_attempts integer NOT NULL CHECK (max_attempts > 0),
    attempted integer NOT NULL DEFAULT 0 CHECK (attempted >= 0),
    certified integer NOT NULL DEFAULT 0 CHECK (certified >= 0),
    elapsed_seconds double precision NOT NULL DEFAULT 0.0 CHECK (elapsed_seconds >= 0.0),
    base_tree_sha256 text NOT NULL CHECK (base_tree_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    CHECK (certified <= attempted),
    CHECK (attempted <= max_attempts)
);

CREATE INDEX IF NOT EXISTS bakes_robot_idx ON rigby_general.bakes (robot_id, created_at DESC);

-- A certified primitive: one grounded schema binding that simulated and passed
-- every gate for its morphology class. `schema_key` is a discrete symbol, which
-- is why retrieval here is exact-match and needs no text embedding model.
CREATE TABLE IF NOT EXISTS rigby_general.primitive_bindings (
    binding_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    robot_id text NOT NULL REFERENCES rigby_general.robots (robot_id) ON DELETE CASCADE,
    bake_id uuid NOT NULL REFERENCES rigby_general.bakes (bake_id) ON DELETE CASCADE,
    schema_key text NOT NULL,
    segment_key text NOT NULL,
    figure_role text NOT NULL,
    ground_role text NOT NULL,
    figure_site text NOT NULL,
    ground_site text,
    region_remove text NOT NULL,
    region_dimensionality text NOT NULL,
    reference_frame text NOT NULL,
    status text NOT NULL CHECK (status IN ('certified', 'quarantined', 'deprecated')),
    program jsonb NOT NULL,
    program_sha256 text NOT NULL CHECK (program_sha256 ~ '^[0-9a-f]{64}$'),
    trajectory_sha256 text NOT NULL CHECK (trajectory_sha256 ~ '^[0-9a-f]{64}$'),
    measurements jsonb NOT NULL DEFAULT '{}'::jsonb,
    certification jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (robot_id, segment_key, status)
);

CREATE INDEX IF NOT EXISTS primitive_lookup_idx
    ON rigby_general.primitive_bindings (robot_id, schema_key)
    WHERE status = 'certified';

-- A binding that failed. The set of these is the robot's refusal boundary: the
-- firewall consults it so an unafforded request returns a typed UNSUPPORTED
-- instead of a fabricated motion. Failures are part of the product.
CREATE TABLE IF NOT EXISTS rigby_general.binding_failures (
    failure_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    robot_id text NOT NULL REFERENCES rigby_general.robots (robot_id) ON DELETE CASCADE,
    bake_id uuid NOT NULL REFERENCES rigby_general.bakes (bake_id) ON DELETE CASCADE,
    schema_key text NOT NULL,
    segment_key text NOT NULL,
    stage text NOT NULL CHECK (
        stage IN ('affordance', 'grounding', 'compilation', 'simulation', 'certification')
    ),
    failure_code text NOT NULL,
    failed_gate text,
    measurements jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS binding_failure_lookup_idx
    ON rigby_general.binding_failures (robot_id, schema_key);
