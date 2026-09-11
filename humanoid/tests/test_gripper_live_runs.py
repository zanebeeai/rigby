from __future__ import annotations

import json

import pytest

from rigby_poc.gripper.decision.goals import NumericTarget
from rigby_poc.gripper.runs.directed import run
from rigby_poc.gripper_runs import GripperRunRequest, GripperRunStore


pytestmark = pytest.mark.fast


class _FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        return -15 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        del timeout
        return -15


def test_persistent_run_can_be_listed_and_really_aborted(tmp_path, monkeypatch) -> None:
    process = _FakeProcess()
    monkeypatch.setattr(
        "rigby_poc.gripper_runs.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    store = GripperRunStore(tmp_path)

    started = store.start(GripperRunRequest(task="pick up the block"))

    assert started["status"] == "running"
    assert started["can_abort"] is True
    assert started["id"] in {run["id"] for run in store.list()}
    assert (tmp_path / started["id"] / "request.json").is_file()
    assert json.loads(
        (tmp_path / started["id"] / "ready.json").read_text(encoding="utf-8")
    ) == {"pid": process.pid}

    stopped = store.abort(started["id"])

    assert stopped is not None
    assert stopped["status"] == "aborted"
    assert stopped["can_abort"] is False
    assert process.terminated is True
    persisted = json.loads(
        (tmp_path / started["id"] / "run.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "aborted"
    assert persisted["events"][-1]["kind"] == "aborted"


class _OfflinePlanner:
    """A fixed numeric goal exercises streaming without a paid model call."""

    def __init__(self) -> None:
        self.held = NumericTarget(metric="hand_z_m", value=0.95, using=("move",))
        self.spans = {}
        self.calls = 0
        self.transcript: list[dict] = []

    def due(self, body, seen, now):
        del body, seen, now
        return False


def test_directed_run_streams_replayable_physical_frames_without_api(tmp_path) -> None:
    events: list[dict] = []
    destination = tmp_path / "clip.json"

    achieved = run(
        task="raise the hand",
        seconds=0.3,
        fps=10,
        planner=_OfflinePlanner(),  # type: ignore[arg-type]
        output_path=destination,
        on_progress=events.append,
        verbose=False,
    )

    clip = json.loads(destination.read_text(encoding="utf-8"))
    assert clip["protocol"] == "directed_v2"
    assert clip["frames"]
    assert len(clip["frames"][0]["links"]) == 7
    assert clip["frames"][0]["target"] == "hand_z_m"
    assert any(event["kind"] == "frame" for event in events)
    assert achieved["vlm_calls"] == 0
