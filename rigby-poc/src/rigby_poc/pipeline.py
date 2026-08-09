from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .compiler import PROJECT_ROOT
from .models import PipelineRunRequest
from .io_utils import atomic_write_json


SAFE_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[a-f0-9]{8}$")


def capture_in_subprocess(
    result_id: str,
    output_dir: Path,
    *,
    base_url: str,
) -> Path:
    """Give every browser capture a main-thread Playwright lifecycle."""
    command = [
        sys.executable,
        "-m",
        "evals.capture",
        result_id,
        str(output_dir),
        "--base-url",
        base_url,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        # Browser grandchildren can briefly retain inherited pipe handles even
        # after the capture process has written its final manifest. Sending
        # output to the null device lets `run` observe the helper's exit
        # directly instead of waiting for every descendant to close a pipe.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"evidence capture failed with exit code {completed.returncode}")
    manifest = output_dir / "evidence-manifest.json"
    if not manifest.is_file():
        raise RuntimeError("evidence capture completed without a manifest")
    return manifest


class PipelineRunStore:
    """Persistent background orchestration with a UI-friendly event trace."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PROJECT_ROOT / "results" / "pipeline-runs"
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _path(self, run_id: str) -> Path:
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("invalid pipeline run id")
        return self.root / run_id / "run.json"

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        atomic_write_json(path, value)

    def _read(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id)
        if not path.is_file():
            raise FileNotFoundError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _update(self, run_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            record = self._read(run_id)
            record.update(updates)
            record["updated_at"] = self._now()
            self._atomic_json(self._path(run_id), record)
            return record

    def _event(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            record = self._read(run_id)
            events = record.setdefault("events", [])
            events.append(
                {
                    "sequence": len(events) + 1,
                    "at": self._now(),
                    **payload,
                }
            )
            record["stage"] = payload.get("stage", record.get("stage"))
            if (
                payload.get("stage") != "failed"
                and isinstance(record.get("error"), dict)
                and record["error"].get("type") == "ServerRestarted"
            ):
                record["status"] = "running"
                record["error"] = None
            record["updated_at"] = self._now()
            self._atomic_json(self._path(run_id), record)

    def start(self, request: PipelineRunRequest, *, base_url: str) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        created = self._now()
        record = {
            "schema_version": "1.0",
            "run_id": run_id,
            "prompt": request.text,
            "status": "queued",
            "stage": "queued",
            "created_at": created,
            "updated_at": created,
            "winner_result_id": None,
            "trace_url": None,
            "error": None,
            "events": [
                {
                    "sequence": 1,
                    "at": created,
                    "event": "run_queued",
                    "stage": "queued",
                    "message": "The autonomous motion run is queued.",
                    "data": {},
                }
            ],
        }
        self._atomic_json(self._path(run_id), record)
        thread = threading.Thread(
            target=self._execute,
            args=(run_id, request, base_url),
            daemon=True,
            name=f"rigby-pipeline-{run_id}",
        )
        with self._lock:
            self._threads[run_id] = thread
        thread.start()
        return record

    def _execute(self, run_id: str, request: PipelineRunRequest, base_url: str) -> None:
        try:
            from evals.flywheel import run_best_of_five

            self._update(run_id, status="running", stage="planning")
            output_dir = self._path(run_id).parent / "artifacts"
            trace_path = run_best_of_five(
                request.text,
                output_dir,
                provider=request.provider,
                base_url=base_url,
                max_rounds=request.max_rounds,
                selection_mode="five_way",
                scene_manifest=request.scene,
                progress_callback=lambda event: self._event(run_id, event),
                max_model_calls=4,
                capture_fn=capture_in_subprocess,
            )
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            winner = trace.get("winner_result_id")
            completed = bool(winner)
            unsupported = trace.get("status") == "unsupported_motion"
            unsupported_reason = trace.get("unsupported_reason")
            attempted_candidate_count = sum(
                len(item.get("candidates", [])) for item in trace.get("rounds", [])
            )
            judging_batch_count = sum(
                sum(
                    bool(candidate.get("perceptually_rankable"))
                    for candidate in item.get("candidates", [])
                )
                for item in trace.get("rounds", [])
            )
            judged_candidate_count = sum(
                sum(
                    isinstance(candidate.get("judgment"), dict)
                    for candidate in item.get("candidates", [])
                )
                for item in trace.get("rounds", [])
            )
            self._update(
                run_id,
                status=(
                    "completed"
                    if completed
                    else "unsupported"
                    if unsupported
                    else "no_acceptable_candidate"
                ),
                stage="finalize",
                winner_result_id=winner,
                trace_url=f"/results/pipeline-runs/{run_id}/artifacts/flywheel-trace.json",
                error=(
                    {"type": "UnsupportedMotion", "message": unsupported_reason}
                    if unsupported
                    else None
                ),
                summary={
                    "round_count": len(trace.get("rounds", [])),
                    "candidate_count": judging_batch_count or attempted_candidate_count,
                    "attempted_candidate_count": attempted_candidate_count,
                    "judging_batch_count": judging_batch_count,
                    "judged_candidate_count": judged_candidate_count,
                    "repair_count": len(trace.get("repairs", [])),
                    "selection_mode": trace.get("selection_mode"),
                },
            )
        except Exception as error:  # The typed error is surfaced in the run UI.
            self._event(
                run_id,
                {
                    "event": "pipeline_failed",
                    "stage": "failed",
                    "message": str(error),
                    "data": {"error_type": type(error).__name__},
                },
            )
            self._update(
                run_id,
                status="failed",
                stage="failed",
                error={"type": type(error).__name__, "message": str(error)},
            )
        finally:
            with self._lock:
                self._threads.pop(run_id, None)

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            try:
                record = self._read(run_id)
            except (FileNotFoundError, ValueError):
                return None
            # The worker removes itself only after its terminal record is
            # committed. A thread can report is_alive=False in the tiny gap
            # before that cleanup; treating that gap as a server restart
            # overwrites a legitimate completed/unsupported outcome.
            live = run_id in self._threads
        if record.get("status") in {"queued", "running"} and not live:
            self._event(
                run_id,
                {
                    "event": "server_restart",
                    "stage": "failed",
                    "message": "The server restarted before this run finished; start a new run.",
                    "data": {},
                },
            )
            record = self._update(
                run_id,
                status="failed",
                stage="failed",
                error={
                    "type": "ServerRestarted",
                    "message": "The server restarted before this run finished; start a new run.",
                },
            )
        trace_path = self._path(run_id).parent / "artifacts" / "flywheel-trace.json"
        if trace_path.is_file():
            try:
                record["trace"] = json.loads(trace_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                record["trace"] = None
        return record
