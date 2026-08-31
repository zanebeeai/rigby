"""Persistent, cancellable runs for the VLM-directed gripper.

Each attempt lives in its own process.  That boundary is intentional: stopping
an attempt must stop MuJoCo and any in-flight model request, not merely change a
label in the browser.  State is file-backed so the UI can reconnect after a
refresh and completed attempts remain a replay library.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .compiler import PROJECT_ROOT
from .io_utils import atomic_write_json


_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[a-f0-9]{8}$")
_TERMINAL = {"completed", "failed", "aborted", "interrupted"}


class GripperRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=400)
    seconds: float = Field(default=32.0, ge=5.0, le=90.0)
    max_model_calls: int = Field(default=10, ge=1, le=20)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class GripperRunStore:
    def __init__(self, root: Path | None = None) -> None:
        configured = os.environ.get("RIGBY_GRIPPER_RUNS_DIR")
        self.root = root or (Path(configured) if configured else PROJECT_ROOT / "results" / "gripper-runs")
        self.root.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.Lock()

    def _directory(self, run_id: str) -> Path:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("invalid gripper run id")
        return self.root / run_id

    def _read(self, run_id: str) -> dict[str, Any] | None:
        try:
            path = self._directory(run_id) / "run.json"
        except ValueError:
            return None
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _public(self, value: dict[str, Any]) -> dict[str, Any]:
        run_id = str(value["id"])
        out = dict(value)
        directory = self.root / run_id
        out["clip_url"] = (
            f"/api/v1/gripper-runs/{run_id}/clip"
            if (directory / "clip.json").is_file()
            else None
        )
        out["can_abort"] = out.get("status") in {"queued", "running"}
        # Process identifiers are an implementation detail, not a UI contract.
        out.pop("pid", None)
        return out

    def start(self, request: GripperRunRequest) -> dict[str, Any]:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        directory = self._directory(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        request_document = request.model_dump(mode="json")
        atomic_write_json(directory / "request.json", request_document)
        record: dict[str, Any] = {
            "schema_version": "1.0",
            "id": run_id,
            "task": request.task.strip(),
            "status": "queued",
            "stage": "starting",
            "message": "Starting an isolated gripper process",
            "started_at": _now(),
            "updated_at": _now(),
            "finished_at": None,
            "progress": 0.0,
            "elapsed_s": 0.0,
            "model_calls": 0,
            "events": [{"at": _now(), "kind": "queued", "message": "Run queued"}],
            "latest": None,
            "achieved": None,
            "error": None,
        }
        atomic_write_json(directory / "run.json", record)

        command = [
            sys.executable,
            "-m",
            "rigby_poc.gripper.run_worker",
            "--root",
            str(self.root),
            "--run-id",
            run_id,
        ]
        log = (directory / "worker.log").open("ab")
        try:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except Exception as error:
            log.close()
            record.update(
                status="failed",
                stage="failed",
                message="The gripper process could not start",
                error=str(error)[:500],
                updated_at=_now(),
                finished_at=_now(),
            )
            atomic_write_json(directory / "run.json", record)
            return self._public(record)
        finally:
            log.close()

        record["pid"] = process.pid
        record["status"] = "running"
        record["stage"] = "initializing"
        record["message"] = "Building the simulated body and cameras"
        record["updated_at"] = _now()
        atomic_write_json(directory / "run.json", record)
        # The child waits for this marker before reading run.json. This keeps a
        # fast-starting worker from overwriting its own first progress update
        # with the API process's initial pid/status write.
        atomic_write_json(directory / "ready.json", {"pid": process.pid})
        with self._lock:
            self._processes[run_id] = process
        return self._public(record)

    def get(self, run_id: str) -> dict[str, Any] | None:
        value = self._read(run_id)
        if value is None:
            return None
        with self._lock:
            process = self._processes.get(run_id)
            if process is not None and process.poll() is not None:
                self._processes.pop(run_id, None)
        return self._public(value)

    def clip_path(self, run_id: str) -> Path | None:
        try:
            path = self._directory(run_id) / "clip.json"
        except ValueError:
            return None
        return path if path.is_file() else None

    def list(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/run.json"), reverse=True):
            value = self._read(path.parent.name)
            if value is not None:
                values.append(self._public(value))
        return values

    def abort(self, run_id: str) -> dict[str, Any] | None:
        value = self._read(run_id)
        if value is None:
            return None
        if value.get("status") in _TERMINAL:
            return self._public(value)

        with self._lock:
            process = self._processes.pop(run_id, None)
        pid = int(value.get("pid") or 0)
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
            elif pid > 0:
                # Supports abort after an API reload, when the child still lives
                # but its Popen handle no longer does. On Windows SIGTERM maps to
                # TerminateProcess.
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

        value.update(
            status="aborted",
            stage="aborted",
            message="Stopped by the operator",
            updated_at=_now(),
            finished_at=_now(),
        )
        events = list(value.get("events") or [])
        events.append({"at": _now(), "kind": "aborted", "message": "Run aborted"})
        value["events"] = events[-100:]
        atomic_write_json(self._directory(run_id) / "run.json", value)
        return self._public(value)
