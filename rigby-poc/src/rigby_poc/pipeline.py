from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from .compiler import PROJECT_ROOT
from .observability import Tracer, get_tracer
from .run_config import effective_configuration
from .models import PipelineRunRequest
from .io_utils import atomic_write_json


SAFE_RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[a-f0-9]{8}$")


def _capture_budget_s(result_count: int) -> float:
    """Capture's own deadline plus process grace, never a number of its own.

    Importing this from `evals.capture` is deliberate: plan 05 section 1.4 traced the
    orphaned-Chrome failure to a hardcoded 300 s here that was *below* capture's worst
    case, so the timeout fired during normal work and escaped uncaught.
    """
    from evals.capture import subprocess_timeout_s

    return subprocess_timeout_s(result_count=result_count)


def _spawn_capture(command: list[str], stdout: Any, stderr: Any) -> subprocess.Popen[bytes]:
    """Start the capture helper in its own process group so its browser can be reaped."""
    if os.name == "posix":
        return subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr, start_new_session=True
        )
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=stdout,
        stderr=stderr,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
    )


def terminate_capture_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill the capture helper *and its browser descendants*, on either platform.

    `subprocess.run(timeout=...)` kills only the direct child, which is why a timed-out
    capture used to leave Chrome running. POSIX gets a process-group signal; Windows has
    no process groups for orphan reaping, so `taskkill /T` walks the tree instead.
    """
    if os.name == "posix":
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    else:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    with suppress(OSError):
        process.kill()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=30)


def _run_capture(command: list[str], log_dir: Path, *, budget_s: float, description: str) -> str:
    """Run the capture helper, persisting its output and never leaking its browser.

    Output goes to a file rather than a pipe. The original code sent it to the null
    device because browser grandchildren can hold an inherited pipe open past the
    helper's exit, which makes `communicate()` block; a file has no such coupling, so
    the output survives without reintroducing that hang.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "capture.log"
    # Plan 01 section 1.6: a non-model stage produced artifacts and no record at all, and
    # capture's output went to the null device, so a failure surfaced only as an exit
    # code. The span parents to whichever stage is running, so capture time is attributed
    # rather than merely elapsed.
    with get_tracer().span("tool", "capture.subprocess", description=description) as span:
        span.set(command=command, budget_s=budget_s, log_path=str(log_path))
        with log_path.open("wb") as sink:
            process = _spawn_capture(command, sink, subprocess.STDOUT)
            try:
                returncode = process.wait(timeout=budget_s)
            except subprocess.TimeoutExpired:
                terminate_capture_tree(process)
                span.set(timed_out=True)
                raise RuntimeError(
                    f"evidence capture for {description} exceeded {budget_s:.0f}s and was killed; "
                    f"see {log_path}"
                ) from None
        output = log_path.read_text(encoding="utf-8", errors="replace")
        span.set(returncode=returncode, output_bytes=len(output))
        # Raised inside the span, not after it. Outside, the span closes `ok` and the
        # transcript records a successful capture for a run that failed -- the failure
        # class this whole PR exists to stop producing.
        if returncode != 0:
            tail = "\n".join(output.strip().splitlines()[-20:])
            raise RuntimeError(
                f"evidence capture for {description} failed with exit code {returncode}:\n{tail}"
            )
    return output


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
    _run_capture(command, output_dir, budget_s=_capture_budget_s(1), description=result_id)
    manifest = output_dir / "evidence-manifest.json"
    if not manifest.is_file():
        raise RuntimeError("evidence capture completed without a manifest")
    return manifest


def capture_batch_in_subprocess(
    requests: Sequence[tuple[str, Path]],
    *,
    base_url: str,
) -> list[Path]:
    """Capture a whole round through one browser launch.

    The subprocess is not for isolation -- the flywheel runs on a daemon thread inside
    the FastAPI process and Playwright's sync API refuses a thread with a live asyncio
    loop -- so one subprocess per round amortises a launch that is pure overhead.
    """
    if not requests:
        return []
    batch_root = Path(requests[0][1]).parent
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_path = batch_root / "capture-batch.json"
    atomic_write_json(
        batch_path,
        {
            "base_url": base_url,
            "requests": [
                {"result_id": result_id, "output_dir": str(output_dir)}
                for result_id, output_dir in requests
            ],
        },
    )
    command = [sys.executable, "-m", "evals.capture", "--batch", str(batch_path)]
    _run_capture(
        command,
        batch_root,
        budget_s=_capture_budget_s(len(requests)),
        description=f"{len(requests)} candidates",
    )
    manifests: list[Path] = []
    for result_id, output_dir in requests:
        manifest = Path(output_dir) / "evidence-manifest.json"
        if not manifest.is_file():
            raise RuntimeError(f"evidence capture completed without a manifest for {result_id}")
        manifests.append(manifest)
    return manifests


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
            "inspection_result_id": None,
            "inspection_reason": None,
            "candidate_result_ids": [],
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
            run_dir = self._path(run_id).parent
            output_dir = run_dir / "artifacts"
            # The tracer and its root span are built *here*, inside the worker, not in
            # `start`. Plan 01 section 8.3, verified: a `ContextVar` set before
            # `thread.start()` reads back as None in the worker, so a root span opened on
            # the caller's side would leave every child span orphaned -- and a detached
            # span is a valid second root rather than an error, so nothing would report it.
            tracer = Tracer(run_dir, run_id=run_id)
            atomic_write_json(
                run_dir / "config.json",
                effective_configuration(
                    run_id=run_id,
                    prompt=request.text,
                    provider=request.provider,
                    selection_mode="five_way",
                    max_rounds=request.max_rounds,
                    max_model_calls=4,
                    extra={"launched_by": "api", "base_url": base_url},
                ),
            )
            with tracer.span(
                "run",
                "pipeline.run",
                prompt=request.text,
                provider=request.provider,
                selection_mode="five_way",
                launched_by="api",
            ):
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
                    capture_batch_fn=capture_batch_in_subprocess,
                    tracer=tracer,
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
                # Surfaced on the record rather than left in the trace file: a
                # rejection nobody can open is indistinguishable from the
                # pipeline breaking, and the UI reads the record, not the trace.
                inspection_result_id=trace.get("inspection_result_id"),
                inspection_reason=trace.get("inspection_reason"),
                candidate_result_ids=[
                    candidate.get("result_id")
                    for item in trace.get("rounds", [])
                    for candidate in item.get("candidates", [])
                    if candidate.get("result_id")
                ],
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
