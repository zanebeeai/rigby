"""A compiled result becomes a registry demo with one request, carrying its pose payload."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rigby_poc import app as app_module
from rigby_poc.store import ResultStore

pytestmark = pytest.mark.fast


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "demos").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Tony Pan"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "tony@example.com"], check=True)
    (root / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "init"], check=True)
    return root


def _result(results: Path, result_id: str) -> None:
    folder = results / result_id
    folder.mkdir(parents=True)
    (folder / "clip.json").write_text(json.dumps({"schema_version": "1.0", "fps": 30, "duration_s": 1.0, "frames": []}), encoding="utf-8")
    (folder / "program.json").write_text(json.dumps({"source_text": "wave hello", "intent": "gesture"}), encoding="utf-8")
    (folder / "provenance.json").write_text(json.dumps({"rig_id": "mesh2motion-human-vrm1", "planner_provider": "offline", "planner_model": "rule-planner-v1", "seed": 0}), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps({"structural_valid": True}), encoding="utf-8")
    (folder / "scene.json").write_text("{}", encoding="utf-8")


def test_a_result_registers_with_its_clip_as_the_payload(tmp_path: Path, monkeypatch) -> None:
    root = _checkout(tmp_path)
    results = root / "humanoid" / "results"
    _result(results, "000042-wave-hello")
    monkeypatch.setattr(app_module, "store", ResultStore(results))
    monkeypatch.setattr(app_module, "_demo_repo_root", lambda: root)
    client = TestClient(app_module.app)

    response = client.post("/api/v1/results/000042-wave-hello/demo", json={"notes": "first wave"})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["who"] == "tpypan"
    entry = json.loads((root / body["registry"]).read_text(encoding="utf-8"))
    assert entry["prompt"] == "wave hello"
    assert entry["kind"] == "humanoid-bones-v1"
    assert entry["embodiment"] == "mesh2motion-human-vrm1"
    assert entry["outcome"] == {"state": "ok", "text": "accepted"}
    assert entry["payload"]["path"].startswith(f"demos/media/{entry['id']}/")
    assert (root / entry["payload"]["path"]).name == "clip.json"
    assert "result:000042-wave-hello" in entry["tags"]
    assert (root / "demos" / "index.html").is_file()


def test_unknown_results_are_refused(tmp_path: Path, monkeypatch) -> None:
    root = _checkout(tmp_path)
    monkeypatch.setattr(app_module, "store", ResultStore(root / "humanoid" / "results"))
    monkeypatch.setattr(app_module, "_demo_repo_root", lambda: root)
    client = TestClient(app_module.app)

    assert client.post("/api/v1/results/000001-nope/demo").status_code == 404
    # A traversal never reaches the handler: the router normalizes it to a
    # different route (405) or the store refuses it (404). Never 201.
    assert client.post("/api/v1/results/../escape/demo").status_code in (404, 405, 422)
