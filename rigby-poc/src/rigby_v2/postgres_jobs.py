from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .contracts import SimulationJobV1, SimulationResultV1, utc_now
from .errors import InvalidJobTransitionError, JobLeaseError, JobNotFoundError
from .hashing import canonical_json
from .records import FailureRecordV1, JobRecord, JobState


def _as_utc(value: datetime | None) -> datetime:
    resolved = value or utc_now()
    if resolved.tzinfo is None:
        raise ValueError("Job store timestamps must be timezone-aware")
    return resolved.astimezone(UTC)


def _jsonb(value: object) -> Jsonb:
    return Jsonb(json.loads(canonical_json(value)))


class PostgresJobStore:
    """PostgreSQL implementation of the durable lease queue.

    Claims use ``FOR UPDATE SKIP LOCKED`` so adding workers never changes the
    public job contract or lets two workers own the same simulation.  Each
    method opens a short transaction; no process-local state is authoritative.
    """

    def __init__(self, database_url: str) -> None:
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgresJobStore requires a PostgreSQL URL")
        self.database_url = database_url

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> JobRecord:
        return JobRecord(
            job=SimulationJobV1.model_validate(row["payload"]),
            state=JobState(row["status"]),
            priority=int(row["priority"]),
            attempts=int(row["attempts"]),
            available_at=_as_utc(row["available_at"]),
            created_at=_as_utc(row["created_at"]),
            updated_at=_as_utc(row["updated_at"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=(
                _as_utc(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            ),
            heartbeat_at=(
                _as_utc(row["heartbeat_at"]) if row["heartbeat_at"] is not None else None
            ),
            cancel_requested=bool(row["cancel_requested"]),
            result=(
                SimulationResultV1.model_validate(row["result"])
                if row["result"] is not None
                else None
            ),
            failure=(
                FailureRecordV1.model_validate(row["failure"])
                if row["failure"] is not None
                else None
            ),
            idempotency_key=row["idempotency_key"],
        )

    @staticmethod
    def _require_row(
        connection: psycopg.Connection[dict[str, Any]],
        job_id: str,
        *,
        for_update: bool = False,
    ) -> dict[str, Any]:
        suffix = " FOR UPDATE" if for_update else ""
        row = connection.execute(
            f"SELECT * FROM rigby_v2.simulation_jobs WHERE job_id = %s{suffix}",
            (job_id,),
        ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return row

    def health(self) -> dict[str, str]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT current_setting('server_version') AS postgres_version,
                       extversion AS vector_version
                FROM pg_extension WHERE extname = 'vector'
                """
            ).fetchone()
        if row is None:
            raise RuntimeError("pgvector extension is not installed")
        return {key: str(value) for key, value in row.items()}

    def submit(
        self,
        job: SimulationJobV1,
        *,
        priority: int = 0,
        available_at: datetime | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord:
        if idempotency_key is not None and not idempotency_key.strip():
            raise ValueError("idempotency_key cannot be blank")
        now = utc_now()
        ready_at = _as_utc(available_at or now)
        job_hash = job.content_hash()
        with self._connect() as connection:
            with connection.transaction():
                if idempotency_key is not None:
                    existing = connection.execute(
                        """
                        SELECT * FROM rigby_v2.simulation_jobs
                        WHERE idempotency_key = %s FOR UPDATE
                        """,
                        (idempotency_key,),
                    ).fetchone()
                    if existing is not None:
                        return self._row_to_record(existing)
                existing = connection.execute(
                    """
                    SELECT * FROM rigby_v2.simulation_jobs
                    WHERE job_id = %s FOR UPDATE
                    """,
                    (job.job_id,),
                ).fetchone()
                if existing is not None:
                    if existing["payload_sha256"] != job_hash:
                        raise ValueError(f"Job id {job.job_id!r} already has different content")
                    return self._row_to_record(existing)
                row = connection.execute(
                    """
                    INSERT INTO rigby_v2.simulation_jobs (
                        job_id, schema_version, status, priority, payload,
                        payload_sha256, attempts, max_attempts, available_at,
                        idempotency_key, created_at, updated_at
                    ) VALUES (%s, %s, 'queued', %s, %s, %s, 0, 3, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        job.job_id,
                        job.schema_version,
                        priority,
                        _jsonb(job),
                        job_hash,
                        ready_at,
                        idempotency_key,
                        now,
                        now,
                    ),
                ).fetchone()
                assert row is not None
                return self._row_to_record(row)

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            return self._row_to_record(self._require_row(connection, job_id))

    def list(
        self, *, states: Sequence[JobState] | None = None, limit: int = 100
    ) -> list[JobRecord]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            if states:
                rows = connection.execute(
                    """
                    SELECT * FROM rigby_v2.simulation_jobs
                    WHERE status = ANY(%s)
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    ([state.value for state in states], limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM rigby_v2.simulation_jobs
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def lease(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        now: datetime | None = None,
    ) -> JobRecord | None:
        if not worker_id.strip():
            raise ValueError("worker_id cannot be blank")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        current = _as_utc(now)
        expiry = current + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            with connection.transaction():
                row = connection.execute(
                    """
                    WITH ready AS (
                        SELECT job_id FROM rigby_v2.simulation_jobs
                        WHERE status = 'queued' AND available_at <= %s
                        ORDER BY priority DESC, created_at, job_id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE rigby_v2.simulation_jobs AS job
                    SET status = 'leased', attempts = job.attempts + 1,
                        lease_owner = %s, lease_expires_at = %s,
                        heartbeat_at = %s, updated_at = %s,
                        started_at = COALESCE(job.started_at, %s)
                    FROM ready
                    WHERE job.job_id = ready.job_id
                    RETURNING job.*
                    """,
                    (current, worker_id, expiry, current, current, current),
                ).fetchone()
                return self._row_to_record(row) if row is not None else None

    @staticmethod
    def _assert_live_lease(row: dict[str, Any], worker_id: str, now: datetime) -> None:
        if row["status"] != JobState.LEASED.value:
            raise InvalidJobTransitionError(
                f"Job {row['job_id']!r} is {row['status']!r}, not leased"
            )
        if row["lease_owner"] != worker_id or row["lease_expires_at"] <= now:
            raise JobLeaseError(f"Worker {worker_id!r} does not hold a live lease")

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        now: datetime | None = None,
    ) -> JobRecord:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        current = _as_utc(now)
        expiry = current + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            with connection.transaction():
                row = self._require_row(connection, job_id, for_update=True)
                self._assert_live_lease(row, worker_id, current)
                updated = connection.execute(
                    """
                    UPDATE rigby_v2.simulation_jobs
                    SET heartbeat_at = %s, lease_expires_at = %s, updated_at = %s
                    WHERE job_id = %s RETURNING *
                    """,
                    (current, expiry, current, job_id),
                ).fetchone()
                assert updated is not None
                return self._row_to_record(updated)

    def cancel(self, job_id: str, *, now: datetime | None = None) -> JobRecord:
        current = _as_utc(now)
        with self._connect() as connection:
            with connection.transaction():
                row = self._require_row(connection, job_id, for_update=True)
                if row["status"] in {JobState.QUEUED.value, JobState.LEASED.value}:
                    row = connection.execute(
                        """
                        UPDATE rigby_v2.simulation_jobs
                        SET status = 'cancelled', cancel_requested = true,
                            lease_owner = NULL, lease_expires_at = NULL,
                            updated_at = %s, finished_at = %s
                        WHERE job_id = %s RETURNING *
                        """,
                        (current, current, job_id),
                    ).fetchone()
                    assert row is not None
                return self._row_to_record(row)

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result: SimulationResultV1,
        *,
        now: datetime | None = None,
    ) -> JobRecord:
        if result.job_id != job_id:
            raise ValueError("Result job_id does not match the completed job")
        current = _as_utc(now)
        with self._connect() as connection:
            with connection.transaction():
                row = self._require_row(connection, job_id, for_update=True)
                self._assert_live_lease(row, worker_id, current)
                updated = connection.execute(
                    """
                    UPDATE rigby_v2.simulation_jobs
                    SET status = 'succeeded', result = %s, result_sha256 = %s,
                        lease_owner = NULL, lease_expires_at = NULL,
                        updated_at = %s, finished_at = %s
                    WHERE job_id = %s RETURNING *
                    """,
                    (_jsonb(result), result.content_hash(), current, current, job_id),
                ).fetchone()
                assert updated is not None
                return self._row_to_record(updated)

    def fail(
        self,
        job_id: str,
        worker_id: str,
        failure: FailureRecordV1,
        *,
        now: datetime | None = None,
    ) -> JobRecord:
        if failure.job_id is not None and failure.job_id != job_id:
            raise ValueError("Failure job_id does not match the failed job")
        current = _as_utc(now)
        with self._connect() as connection:
            with connection.transaction():
                row = self._require_row(connection, job_id, for_update=True)
                self._assert_live_lease(row, worker_id, current)
                updated = connection.execute(
                    """
                    UPDATE rigby_v2.simulation_jobs
                    SET status = 'failed', failure = %s,
                        error_code = %s, error_message = %s,
                        lease_owner = NULL, lease_expires_at = NULL,
                        updated_at = %s, finished_at = %s
                    WHERE job_id = %s RETURNING *
                    """,
                    (
                        _jsonb(failure),
                        failure.code.value,
                        failure.message,
                        current,
                        current,
                        job_id,
                    ),
                ).fetchone()
                assert updated is not None
                return self._row_to_record(updated)

    def requeue_expired(self, *, now: datetime | None = None) -> int:
        current = _as_utc(now)
        with self._connect() as connection:
            with connection.transaction():
                cursor = connection.execute(
                    """
                    UPDATE rigby_v2.simulation_jobs
                    SET status = CASE WHEN cancel_requested THEN 'cancelled' ELSE 'queued' END,
                        lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                        available_at = %s, updated_at = %s,
                        finished_at = CASE WHEN cancel_requested THEN %s ELSE NULL END
                    WHERE status = 'leased' AND lease_expires_at <= %s
                    """,
                    (current, current, current, current),
                )
                return cursor.rowcount

    def recover(self, *, now: datetime | None = None) -> int:
        return self.requeue_expired(now=now)
