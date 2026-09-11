"""Child-process entry point for one live gripper attempt."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..io_utils import atomic_write_json
from .decision.planner import Planner
from .runs.directed import run


def _now() -> str:
    return datetime.now(UTC).isoformat()


def execute(root: Path, run_id: str) -> None:
    directory = root / run_id
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    state = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    frames: list[dict[str, Any]] = []

    def write() -> None:
        state["updated_at"] = _now()
        atomic_write_json(directory / "run.json", state)

    def progress(event: dict[str, Any]) -> None:
        kind = str(event.get("kind", "progress"))
        if kind == "frame":
            frame = event.get("frame")
            if isinstance(frame, dict):
                frames.append(frame)
            state["elapsed_s"] = float(event.get("elapsed_s", state.get("elapsed_s", 0.0)))
            state["progress"] = float(event.get("progress", 0.0))
            state["latest"] = event.get("latest")
            # Kept out of the frame list on purpose: the clip holds a thousand
            # frames and two images each would make it unopenable.
            state["cameras"] = event.get("cameras")
            state["model_calls"] = int(event.get("model_calls", state.get("model_calls", 0)))
            state["stage"] = "acting"
            state["message"] = str(event.get("message", "The search is pursuing the current target"))
            clip = event.get("clip")
            if isinstance(clip, dict):
                clip["frames"] = frames
                # Carried through so a run that is stopped still explains itself.
                clip["transcript"] = event.get("transcript") or []
                atomic_write_json(directory / "clip.json", clip)
        else:
            state["stage"] = str(event.get("stage", kind))
            state["message"] = str(event.get("message", kind.replace("_", " ")))
            state["model_calls"] = int(event.get("model_calls", state.get("model_calls", 0)))
            events = list(state.get("events") or [])
            events.append({
                "at": _now(),
                "kind": kind,
                "message": state["message"],
                "detail": event.get("detail"),
            })
            state["events"] = events[-100:]
        write()

    try:
        ready = directory / "ready.json"
        deadline = time.monotonic() + 5.0
        while not ready.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("gripper run did not receive its startup marker")
            time.sleep(0.01)
        state.update(status="running", stage="initializing", message="Building the simulated body and cameras")
        write()
        planner = Planner(
            task=str(request["task"]),
            max_calls=int(request.get("max_model_calls", 10)),
        )
        achieved = run(
            task=str(request["task"]),
            seconds=float(request.get("seconds", 32.0)),
            planner=planner,
            name=run_id,
            verbose=False,
            output_path=directory / "clip.json",
            on_progress=progress,
        )
        state.update(
            status="completed",
            stage="completed",
            message=("Placed the object in the target" if achieved.get("in_target") else "Run finished without completing the task"),
            progress=1.0,
            model_calls=int(achieved.get("vlm_calls", planner.calls)),
            achieved=achieved,
            finished_at=_now(),
        )
        state["events"] = [
            *(state.get("events") or []),
            {"at": _now(), "kind": "completed", "message": state["message"]},
        ][-100:]
        write()
    except BaseException as error:  # child must persist a useful terminal state
        state.update(
            status="failed",
            stage="failed",
            message="The gripper run failed",
            error=f"{type(error).__name__}: {error}"[:500],
            finished_at=_now(),
        )
        state["events"] = [
            *(state.get("events") or []),
            {"at": _now(), "kind": "failed", "message": state["error"]},
        ][-100:]
        write()
        (directory / "traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    execute(args.root.resolve(), args.run_id)


if __name__ == "__main__":
    main()
