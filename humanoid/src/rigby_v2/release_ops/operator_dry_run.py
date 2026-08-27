"""Fail-closed local operator incident and rollback drill.

The drill deliberately starts an API canary with an unreachable disposable
database, captures a credential-redacted incident bundle, stops that canary,
and returns to the exact-lock known-good local environment.  It is read-only
with respect to the database and content-addressed artifact store.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol
from urllib.parse import urlsplit
import urllib.request
from uuid import uuid4

import psycopg

from rigby_core.hashing import hash_file

from rigby_v2.config import resolve_lock_path

from .checklist import ReleaseEvidence
from .evidence import seal_release_evidence


_CREDENTIAL_KEY = re.compile(r"password|passwd|secret|token|credential|database_url", re.I)
_URL_WITH_USERINFO = re.compile(r"\b(postgresql|postgres)://[^\s/@]+(?::[^\s/@]*)?@", re.I)


class OperatorDrillError(RuntimeError):
    """The operator drill failed and therefore must not seal pass evidence."""


@dataclass(frozen=True, slots=True)
class OperatorDryRunConfig:
    project_root: Path
    evidence_root: Path
    artifact_root: Path
    database_url: str
    python_executable: str = sys.executable
    timeout_s: float = 20.0


@dataclass(frozen=True, slots=True)
class ApiProcessSpec:
    port: int
    database_url: str
    artifact_root: Path
    project_root: Path
    python_executable: str
    log_path: Path
    label: str


class ProcessHandle(Protocol):
    label: str

    def poll(self) -> int | None: ...


class ProcessRunner(Protocol):
    def start_api(self, specification: ApiProcessSpec) -> ProcessHandle: ...

    def stop(self, process: ProcessHandle) -> None: ...


class ProbeRunner(Protocol):
    def wait_health(
        self,
        process: ProcessHandle,
        port: int,
        *,
        expected_status: str,
        timeout_s: float,
    ) -> Mapping[str, object]: ...

    def database_counts(self, database_url: str) -> Mapping[str, int]: ...

    def artifact_readiness(self, artifact_root: Path) -> Mapping[str, object]: ...


@dataclass(slots=True)
class _SubprocessHandle:
    label: str
    process: subprocess.Popen[bytes]
    log_stream: Any

    def poll(self) -> int | None:
        return self.process.poll()


class LocalProcessRunner:
    """Start and stop only the exact API child processes created by the drill."""

    def start_api(self, specification: ApiProcessSpec) -> _SubprocessHandle:
        specification.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = specification.log_path.open("ab", buffering=0)
        environment = dict(os.environ)
        existing_pythonpath = environment.get("PYTHONPATH")
        source = str((specification.project_root / "src").resolve())
        environment["PYTHONPATH"] = source + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        environment.update(
            {
                "RIGBY_V2_API_PORT": str(specification.port),
                "RIGBY_V2_DATABASE_URL": specification.database_url,
                "RIGBY_V2_ARTIFACT_DIR": str(specification.artifact_root),
                "RIGBY_V2_PROJECT_ROOT": str(specification.project_root),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        try:
            process = subprocess.Popen(
                [
                    specification.python_executable,
                    "-c",
                    "from rigby_v2.app import run; run()",
                ],
                cwd=specification.project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            log_stream.close()
            raise
        return _SubprocessHandle(specification.label, process, log_stream)

    def stop(self, process: ProcessHandle) -> None:
        if not isinstance(process, _SubprocessHandle):
            raise TypeError("LocalProcessRunner can stop only its own child process")
        try:
            if process.process.poll() is None:
                process.process.terminate()
                try:
                    process.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.process.kill()
                    process.process.wait(timeout=10)
        finally:
            process.log_stream.close()


class LocalProbeRunner:
    def wait_health(
        self,
        process: ProcessHandle,
        port: int,
        *,
        expected_status: str,
        timeout_s: float,
    ) -> Mapping[str, object]:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        url = f"http://127.0.0.1:{port}/api/v2/health"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise OperatorDrillError(
                    f"{process.label} API exited before its health observation"
                )
            try:
                # The degraded probe deliberately includes a bounded one-second
                # database connect failure before the API can answer.
                with urllib.request.urlopen(url, timeout=min(10.0, timeout_s)) as response:
                    payload = json.loads(response.read())
                if response.status == 200 and payload.get("status") == expected_status:
                    return payload
            except Exception as error:
                last_error = error
            time.sleep(0.1)
        raise OperatorDrillError(
            f"timed out waiting for {process.label}={expected_status}; "
            f"last error={type(last_error).__name__ if last_error else 'none'}"
        )

    def database_counts(self, database_url: str) -> Mapping[str, int]:
        with psycopg.connect(database_url) as connection:
            row = connection.execute(
                """
                SELECT
                  (SELECT count(1) FROM rigby_v2.simulation_jobs) AS jobs,
                  (SELECT count(1) FROM rigby_v2.library_releases) AS releases,
                  (SELECT count(1) FROM rigby_v2.animation_records
                   WHERE status = 'legacy_candidate') AS legacy
                """
            ).fetchone()
        if row is None:
            raise OperatorDrillError("database count snapshot returned no row")
        return {"jobs": int(row[0]), "releases": int(row[1]), "legacy": int(row[2])}

    def artifact_readiness(self, artifact_root: Path) -> Mapping[str, object]:
        root = artifact_root.resolve()
        if not root.is_dir() or artifact_root.is_symlink():
            raise OperatorDrillError("artifact root is absent or an unsafe symlink")
        objects = root / "objects" / "sha256"
        files = tuple(path for path in objects.glob("*/*") if path.is_file()) if objects.is_dir() else ()
        inventory = hashlib.sha256()
        for path in sorted(files):
            relative = path.relative_to(objects).as_posix().replace("/", "")
            actual = hash_file(path)
            if relative != actual:
                raise OperatorDrillError(f"artifact object {relative!r} failed its content hash")
            inventory.update(f"{actual}\n".encode())
        return {
            "ready": True,
            "content_addressed_objects": len(files),
            "inventory_sha256": inventory.hexdigest(),
            "read_only_probe": True,
        }


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _two_distinct_free_ports() -> tuple[int, int]:
    listeners = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
    try:
        for listener in listeners:
            listener.bind(("127.0.0.1", 0))
        return (
            int(listeners[0].getsockname()[1]),
            int(listeners[1].getsockname()[1]),
        )
    finally:
        for listener in listeners:
            listener.close()


def _unreachable_database_url(port: int) -> str:
    return (
        "postgresql://operator_canary:operator-canary-secret@"
        f"127.0.0.1:{port}/unreachable?connect_timeout=1"
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _redact(value: object, *, secrets: tuple[str, ...]) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if _CREDENTIAL_KEY.search(str(key))
                else _redact(item, secrets=secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_redact(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        redacted = _URL_WITH_USERINFO.sub(r"\1://[REDACTED]@", value)
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    return value


def _evidence_document(
    evidence_root: Path, requirement_id: str
) -> tuple[dict[str, object], str]:
    manifest_path = evidence_root / "manifest.json"
    artifact_path = evidence_root / f"{requirement_id}.json"
    if not manifest_path.is_file() or not artifact_path.is_file():
        raise OperatorDrillError(f"sealed {requirement_id} evidence is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(
        (item for item in manifest if item.get("requirement_id") == requirement_id),
        None,
    )
    if row is None or row.get("status") != "pass":
        raise OperatorDrillError(f"{requirement_id} is absent or not sealed pass evidence")
    actual = hash_file(artifact_path)
    if row.get("artifact_path") != artifact_path.name or row.get("artifact_sha256") != actual:
        raise OperatorDrillError(f"sealed {requirement_id} evidence hash is invalid")
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    if document.get("requirement_id") != requirement_id or document.get("passed") is not True:
        raise OperatorDrillError(f"{requirement_id} document is not a passing envelope")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise OperatorDrillError(f"{requirement_id} payload is invalid")
    return payload, actual


def _verify_prior_evidence(evidence_root: Path) -> dict[str, object]:
    backup, backup_hash = _evidence_document(evidence_root, "database_backup_restore")
    counts = backup.get("table_counts")
    if (
        backup.get("restored_table_counts_identical") is not True
        or backup.get("content_addressed_canary_restored") is not True
        or backup.get("disposable_database_cleaned") is not True
        or not isinstance(backup.get("dump_size_bytes"), int)
        or int(backup["dump_size_bytes"]) <= 0
        or not isinstance(counts, dict)
        or int(counts.get("animation_records", 0)) <= 0
    ):
        raise OperatorDrillError("backup/restore evidence is not populated and complete")
    replay, replay_hash = _evidence_document(evidence_root, "deterministic_replay")
    if (
        replay.get("api_restarted_in_new_process") is not True
        or replay.get("live_replay_endpoint") is not True
        or replay.get("authoritative_trace_sha256") != replay.get("replay_trace_sha256")
        or int(replay.get("attempts", 0)) < 2
    ):
        raise OperatorDrillError("restart/replay evidence is incomplete or disagrees")
    return {
        "backup_evidence_sha256": backup_hash,
        "backup_dump_sha256": backup["dump_sha256"],
        "backup_table_counts": counts,
        "replay_evidence_sha256": replay_hash,
        "replay_trace_sha256": replay["replay_trace_sha256"],
        "known_good_lock_sha256": replay["dependency_lock_sha256"],
    }


def _verify_runbook(project_root: Path) -> str:
    runbook = project_root / "docs" / "v2" / "operator-runbook.md"
    if not runbook.is_file():
        raise OperatorDrillError("operator runbook is missing")
    text = runbook.read_text(encoding="utf-8").lower()
    required = (
        "backup and restore",
        "upgrade and rollback",
        "incident response",
        "database unavailable",
        "rollback is forward-restorative",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise OperatorDrillError(f"operator runbook omits required steps: {missing}")
    return hash_file(runbook)


def _log_tail(path: Path, maximum_bytes: int = 64 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - maximum_bytes))
        return stream.read().decode("utf-8", errors="replace")


def run_operator_dry_run(
    config: OperatorDryRunConfig,
    *,
    process_runner: ProcessRunner | None = None,
    probe_runner: ProbeRunner | None = None,
) -> ReleaseEvidence:
    """Exercise the documented incident/rollback path and seal only on success."""

    processes = process_runner or LocalProcessRunner()
    probes = probe_runner or LocalProbeRunner()
    prior = _verify_prior_evidence(config.evidence_root)
    runbook_sha256 = _verify_runbook(config.project_root)
    lock_sha256 = hash_file(resolve_lock_path(config.project_root))
    if lock_sha256 != prior["known_good_lock_sha256"]:
        raise OperatorDrillError("current dependency lock differs from known-good replay evidence")
    before = dict(probes.database_counts(config.database_url))
    expected_count_keys = {"jobs", "releases", "legacy"}
    if set(before) != expected_count_keys:
        raise OperatorDrillError("database snapshot omitted protected count categories")

    run_id = uuid4().hex
    work_root = Path(tempfile.mkdtemp(prefix="rigby-operator-dry-run-"))
    bad_log = work_root / "bad-canary.log"
    good_log = work_root / "rollback-good.log"
    bad_process: ProcessHandle | None = None
    good_process: ProcessHandle | None = None
    bad_stopped = False
    good_stopped = False
    try:
        bad_port, unreachable_port = _two_distinct_free_ports()
        bad_url = _unreachable_database_url(unreachable_port)
        bad_process = processes.start_api(
            ApiProcessSpec(
                bad_port,
                bad_url,
                config.artifact_root,
                config.project_root,
                config.python_executable,
                bad_log,
                "bad-canary",
            )
        )
        degraded = dict(
            probes.wait_health(
                bad_process,
                bad_port,
                expected_status="degraded",
                timeout_s=config.timeout_s,
            )
        )
        if degraded.get("status") != "degraded" or not degraded.get("database_error"):
            raise OperatorDrillError("unreachable DB canary did not fail closed as degraded")
        processes.stop(bad_process)
        bad_stopped = True

        parsed = urlsplit(config.database_url)
        known_secrets = tuple(
            value
            for value in (
                parsed.username,
                parsed.password,
                "operator_canary",
                "operator-canary-secret",
            )
            if value
        )
        incident = _redact(
            {
                "schema_version": "1.0",
                "incident_id": f"operator-dry-run-{run_id}",
                "classification": "database_unavailable_canary",
                "database_target": bad_url,
                "health": degraded,
                "log_tail": _log_tail(bad_log),
                "actions": [
                    "submission traffic never enabled",
                    "diagnostics preserved and redacted",
                    "bad canary stopped",
                    "verified backup selected for forward-restorative rollback",
                ],
            },
            secrets=known_secrets,
        )
        incident_path = (
            config.evidence_root / "incidents" / f"operator-dry-run-{run_id}.json"
        )
        _atomic_json(incident_path, incident)  # type: ignore[arg-type]
        incident_bytes = incident_path.read_bytes()
        if any(secret.encode() in incident_bytes for secret in known_secrets):
            raise OperatorDrillError("incident redaction retained a credential")

        good_port = _free_port()
        good_process = processes.start_api(
            ApiProcessSpec(
                good_port,
                config.database_url,
                config.artifact_root,
                config.project_root,
                config.python_executable,
                good_log,
                "rollback-known-good",
            )
        )
        healthy = dict(
            probes.wait_health(
                good_process,
                good_port,
                expected_status="ok",
                timeout_s=config.timeout_s,
            )
        )
        if healthy.get("status") != "ok" or not healthy.get("database"):
            raise OperatorDrillError("known-good rollback API/DB readiness failed")
        artifact = dict(probes.artifact_readiness(config.artifact_root))
        if artifact.get("ready") is not True:
            raise OperatorDrillError("known-good artifact readiness failed")
        processes.stop(good_process)
        good_stopped = True
        after = dict(probes.database_counts(config.database_url))
        if after != before:
            raise OperatorDrillError(
                f"operator rollback mutated protected counts: before={before}, after={after}"
            )

        payload: dict[str, object] = {
            **prior,
            "runbook_sha256": runbook_sha256,
            "exact_lock_sha256": lock_sha256,
            "bad_canary_degraded": True,
            "bad_canary_stopped": bad_stopped,
            "incident_bundle": {
                "path": incident_path.relative_to(config.evidence_root).as_posix(),
                "sha256": hashlib.sha256(incident_bytes).hexdigest(),
                "credentials_redacted": True,
            },
            "rollback_known_good_health": "ok",
            "rollback_database_ready": True,
            "rollback_artifact_readiness": artifact,
            "protected_counts_before": before,
            "protected_counts_after": after,
            "protected_counts_unchanged": True,
            "process_cleanup": {
                "bad_canary_stopped": bad_stopped,
                "known_good_probe_stopped": good_stopped,
            },
            "documented_steps_exercised": [
                "verify sealed populated backup/restore evidence",
                "verify restart/replay trace agreement",
                "observe degraded DB-unavailable health",
                "preserve redacted incident diagnostics",
                "stop bad canary",
                "select exact-lock known-good environment",
                "confirm API/DB/artifact readiness",
                "confirm jobs/releases/legacy counts unchanged",
            ],
            "read_only_database_drill": True,
        }
        return seal_release_evidence(
            config.evidence_root,
            requirement_id="operator_dry_run",
            passed=True,
            payload=payload,
        )
    finally:
        if good_process is not None and not good_stopped:
            processes.stop(good_process)
        if bad_process is not None and not bad_stopped:
            processes.stop(bad_process)
        # Logs contain no authoritative state.  Remove only this exact private
        # temporary directory after redacted incident capture.
        for path in (bad_log, good_log):
            path.unlink(missing_ok=True)
        try:
            work_root.rmdir()
        except OSError:
            pass
