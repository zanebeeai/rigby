from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import SimulationJobV1, SimulationResultV1, utc_now
from .errors import InvalidJobTransitionError, JobLeaseError, JobNotFoundError
from .hashing import canonical_json
from .records import FailureRecordV1, JobRecord, JobState


@runtime_checkable
class JobStore(Protocol):
    def submit(
        self,
        job: SimulationJobV1,
        *,
        priority: int = 0,
        available_at: datetime | None = None,
        idempotency_key: str | None = None,
    ) -> JobRecord: ...

    def get(self, job_id: str) -> JobRecord: ...

    def lease(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        now: datetime | None = None,
    ) -> JobRecord | None: ...

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
        now: datetime | None = None,
    ) -> JobRecord: ...

    def cancel(self, job_id: str, *, now: datetime | None = None) -> JobRecord: ...

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result: SimulationResultV1,
        *,
        now: datetime | None = None,
    ) -> JobRecord: ...

    def fail(
        self,
        job_id: str,
        worker_id: str,
        failure: FailureRecordV1,
        *,
        now: datetime | None = None,
    ) -> JobRecord: ...

    def requeue_expired(self, *, now: datetime | None = None) -> int: ...

    def list(
        self, *, states: Sequence[JobState] | None = None, limit: int = 100
    ) -> list[JobRecord]: ...

    def recover(self, *, now: datetime | None = None) -> int: ...


def _as_utc(value: datetime | None) -> datetime:
    value = value or utc_now()
    if value.tzinfo is None:
        raise ValueError("Job store timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _encode_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _as_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _decode_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class SQLiteJobStore:
    """Durable local JobStore with lease-based crash recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_json TEXT NOT NULL,
                    job_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('queued','leased','succeeded','failed','cancelled')),
                    priority INTEGER NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    available_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    heartbeat_at TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0,1)),
                    result_json TEXT,
                    failure_json TEXT,
                    idempotency_key TEXT UNIQUE
                );
                CREATE INDEX IF NOT EXISTS jobs_ready_idx
                    ON jobs(state, available_at, priority DESC, created_at ASC);
                CREATE INDEX IF NOT EXISTS jobs_lease_expiry_idx
                    ON jobs(state, lease_expires_at);
                """
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job=SimulationJobV1.model_validate_json(row["job_json"]),
            state=JobState(row["state"]),
            priority=row["priority"],
            attempts=row["attempts"],
            available_at=_decode_time(row["available_at"]),
            created_at=_decode_time(row["created_at"]),
            updated_at=_decode_time(row["updated_at"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=_decode_time(row["lease_expires_at"]),
            heartbeat_at=_decode_time(row["heartbeat_at"]),
            cancel_requested=bool(row["cancel_requested"]),
            result=(
                SimulationResultV1.model_validate_json(row["result_json"])
                if row["result_json"] is not None
                else None
            ),
            failure=(
                FailureRecordV1.model_validate_json(row["failure_json"])
                if row["failure_json"] is not None
                else None
            ),
            idempotency_key=row["idempotency_key"],
        )

    @staticmethod
    def _require_row(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return row

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
        available_at = _as_utc(available_at or now)
        job_json = canonical_json(job)
        job_hash = job.content_hash()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing is not None:
                    connection.commit()
                    return self._row_to_record(existing)
            existing = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job.job_id,)
            ).fetchone()
            if existing is not None:
                if existing["job_hash"] != job_hash:
                    connection.rollback()
                    raise ValueError(f"Job id {job.job_id!r} already has different content")
                connection.commit()
                return self._row_to_record(existing)
            encoded_now = _encode_time(now)
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, job_json, job_hash, state, priority, attempts,
                    available_at, created_at, updated_at, idempotency_key
                ) VALUES (?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job_json,
                    job_hash,
                    priority,
                    _encode_time(available_at),
                    encoded_now,
                    encoded_now,
                    idempotency_key,
                ),
            )
            row = self._require_row(connection, job.job_id)
            connection.commit()
            return self._row_to_record(row)

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            return self._row_to_record(self._require_row(connection, job_id))

    def list(self, *, states: Sequence[JobState] | None = None, limit: int = 100) -> list[JobRecord]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        parameters: list[object] = []
        clause = ""
        if states:
            placeholders = ",".join("?" for _ in states)
            clause = f" WHERE state IN ({placeholders})"
            parameters.extend(state.value for state in states)
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs{clause} ORDER BY created_at DESC LIMIT ?", parameters
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
        now = _as_utc(now)
        expiry = now + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE state = 'queued' AND available_at <= ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (_encode_time(now),),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            cursor = connection.execute(
                """
                UPDATE jobs SET state = 'leased', attempts = attempts + 1,
                    lease_owner = ?, lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'queued'
                """,
                (
                    worker_id,
                    _encode_time(expiry),
                    _encode_time(now),
                    _encode_time(now),
                    row["job_id"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise JobLeaseError("Ready job changed while acquiring its lease")
            leased = self._require_row(connection, row["job_id"])
            connection.commit()
            return self._row_to_record(leased)

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
        now = _as_utc(now)
        expiry = now + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_row(connection, job_id)
            if (
                row["state"] != JobState.LEASED.value
                or row["lease_owner"] != worker_id
                or row["lease_expires_at"] <= _encode_time(now)
            ):
                connection.rollback()
                raise JobLeaseError(f"Worker {worker_id!r} does not hold a live lease for {job_id!r}")
            connection.execute(
                """
                UPDATE jobs SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (_encode_time(now), _encode_time(expiry), _encode_time(now), job_id),
            )
            updated = self._require_row(connection, job_id)
            connection.commit()
            return self._row_to_record(updated)

    def cancel(self, job_id: str, *, now: datetime | None = None) -> JobRecord:
        now = _as_utc(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_row(connection, job_id)
            if row["state"] in {JobState.QUEUED.value, JobState.LEASED.value}:
                connection.execute(
                    """
                    UPDATE jobs SET state = 'cancelled', cancel_requested = 1,
                        lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
                    WHERE job_id = ?
                    """,
                    (_encode_time(now), job_id),
                )
            updated = self._require_row(connection, job_id)
            connection.commit()
            return self._row_to_record(updated)

    @staticmethod
    def _assert_live_lease(row: sqlite3.Row, worker_id: str, now: datetime) -> None:
        if row["state"] != JobState.LEASED.value:
            raise InvalidJobTransitionError(
                f"Job {row['job_id']!r} is {row['state']!r}, not leased"
            )
        if row["lease_owner"] != worker_id or row["lease_expires_at"] <= _encode_time(now):
            raise JobLeaseError(f"Worker {worker_id!r} does not hold a live lease")

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
        now = _as_utc(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_row(connection, job_id)
            self._assert_live_lease(row, worker_id, now)
            connection.execute(
                """
                UPDATE jobs SET state = 'succeeded', result_json = ?, updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE job_id = ?
                """,
                (canonical_json(result), _encode_time(now), job_id),
            )
            updated = self._require_row(connection, job_id)
            connection.commit()
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
        now = _as_utc(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._require_row(connection, job_id)
            self._assert_live_lease(row, worker_id, now)
            connection.execute(
                """
                UPDATE jobs SET state = 'failed', failure_json = ?, updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE job_id = ?
                """,
                (canonical_json(failure), _encode_time(now), job_id),
            )
            updated = self._require_row(connection, job_id)
            connection.commit()
            return self._row_to_record(updated)

    def requeue_expired(self, *, now: datetime | None = None) -> int:
        now = _as_utc(now)
        encoded_now = _encode_time(now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE jobs SET
                    state = CASE WHEN cancel_requested = 1 THEN 'cancelled' ELSE 'queued' END,
                    lease_owner = NULL, lease_expires_at = NULL, heartbeat_at = NULL,
                    available_at = ?, updated_at = ?
                WHERE state = 'leased' AND lease_expires_at <= ?
                """,
                (encoded_now, encoded_now, encoded_now),
            )
            count = cursor.rowcount
            connection.commit()
            return count

    def recover(self, *, now: datetime | None = None) -> int:
        """Reclaim expired work after an API or worker process restart."""

        return self.requeue_expired(now=now)
