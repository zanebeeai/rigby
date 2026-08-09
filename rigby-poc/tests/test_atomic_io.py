from __future__ import annotations

import json

from rigby_poc import io_utils


def test_atomic_write_json_retries_transient_windows_lock(tmp_path, monkeypatch) -> None:
    target = tmp_path / "run.json"
    real_replace = io_utils.os.replace
    calls = 0

    def intermittently_locked(source, destination):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(5, "Access is denied")
        return real_replace(source, destination)

    monkeypatch.setattr(io_utils.os, "replace", intermittently_locked)
    monkeypatch.setattr(io_utils.time, "sleep", lambda _: None)

    io_utils.atomic_write_json(target, {"status": "running"})

    assert calls == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "running"}
    assert not list(tmp_path.glob("*.tmp"))
