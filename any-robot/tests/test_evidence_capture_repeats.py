"""A prompt-run reference change must not hide behind replays of run one."""

from dataclasses import replace

import pytest

from rigby_general.evidence import capture


def test_second_prompt_run_motion_change_is_detected(monkeypatch, tmp_path):
    original = capture.answer
    calls = 0

    def drifting_answer(*args, **kwargs):
        nonlocal calls
        run = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            positions = run.trajectory.qpos.copy()
            positions[:, 0] += 0.01
            return replace(run, trajectory=replace(run.trajectory, qpos=positions))
        return run

    monkeypatch.setattr(capture, "answer", drifting_answer)
    with pytest.raises(ValueError, match="Independent prompt runs produced different reference data"):
        capture.capture_canonical("zoo_compact_arm", tmp_path / "must-not-publish")
    assert calls == 3
    assert not (tmp_path / "must-not-publish").exists()
