"""A result becomes a demo with one request, and carries who asked for it."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from rigby_general.app import create_app
from rigby_general.config import GeneralSettings
from rigby_general.trace import RunTrace, TraceStore


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "demos").mkdir(parents=True)
    (root / "any-robot").mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Angelo Wei"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "angelo@example.com"], check=True)
    (root / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "init"], check=True)
    return root


def test_a_result_with_a_clip_registers_as_a_demo(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    project = root / "any-robot"
    traces = TraceStore(project / "results")
    trace = RunTrace(prompt="pick up the block", robot_id="so101", accepted=True)
    traces.write(trace, clip_bytes=b"GIF89a-stand-in")
    client = TestClient(create_app(GeneralSettings(project_root=project)))

    response = client.post(f"/api/v3/results/{trace.trace_id}/demo", json={"notes": "first lift"})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["who"] == "AngeloWhey"
    entry = json.loads((root / body["registry"]).read_text(encoding="utf-8"))
    assert entry["prompt"] == "pick up the block"
    assert entry["embodiment"] == "so101"
    assert entry["tier"] == "any-robot"
    assert entry["outcome"]["state"] == "ok"
    assert entry["media"][0]["path"].startswith(f"demos/media/{entry['id']}/")
    assert (root / entry["media"][0]["path"]).read_bytes() == b"GIF89a-stand-in"
    assert f"trace:{trace.trace_id}" in entry["tags"]
    assert (root / "demos" / "index.html").is_file()


def test_a_result_without_a_clip_is_refused(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    project = root / "any-robot"
    trace = RunTrace(prompt="wave", robot_id="so101", accepted=False)
    TraceStore(project / "results").write(trace)
    client = TestClient(create_app(GeneralSettings(project_root=project)))

    assert client.post(f"/api/v3/results/{trace.trace_id}/demo").status_code == 409
    assert client.post("/api/v3/results/nope--nothing/demo").status_code == 404
