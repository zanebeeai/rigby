"""The demo registrar: an entry carries its provenance and the viewer renders it."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rigby_core import demos


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "demos").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Tony Pan"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "tony@example.com"], check=True)
    (root / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "init"], check=True)
    return root


def _gif(path: Path, payload: bytes = b"GIF89a-not-really") -> Path:
    path.write_bytes(payload)
    return path


def test_register_copies_media_and_records_provenance(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    clip = _gif(tmp_path / "clip.gif")
    out = demos.register(
        demos.DemoSpec(
            title="so101 lifts the block", prompt="pick up the block", tier="any-robot",
            embodiment="so101", kind="gif", how="uv run python scripts/run_trials.py", files=[clip],
            outcome={"state": "ok", "text": "lifted"},
        ),
        root,
    )
    entry = json.loads(out.read_text(encoding="utf-8"))
    assert out.parent == root / "demos" / "registry"
    assert entry["id"] == out.stem
    assert entry["id"].startswith(entry["produced_at"][:10] + "-pick-up-the-block-")
    assert entry["who"] == "tpypan", "git identity maps to the GitHub handle"
    assert entry["source"]["branch"] in ("main", "master")
    assert entry["source"]["dirty"] is False
    assert len(entry["source"]["commit"]) == 10
    media = entry["media"][0]
    assert media["path"] == f"demos/media/{entry['id']}/clip.gif"
    assert (root / media["path"]).read_bytes() == clip.read_bytes()
    assert media["sha256"] == demos.sha256_file(clip)


def test_same_prompt_different_media_gets_a_different_id(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    spec = dict(title="t", prompt="wave", tier="humanoid", embodiment="mesh2motion-human-vrm1", kind="gif", how="x")
    a = demos.register(demos.DemoSpec(**spec, files=[_gif(tmp_path / "a.gif", b"a")]), root)
    b = demos.register(demos.DemoSpec(**spec, files=[_gif(tmp_path / "b.gif", b"b")]), root)
    assert a != b


def test_register_refuses_what_the_validator_would(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    clip = _gif(tmp_path / "clip.gif")
    with pytest.raises(ValueError):
        demos.register(demos.DemoSpec(title="t", prompt="p", tier="nope", embodiment="e", kind="gif", how="h", files=[clip]), root)
    with pytest.raises(ValueError):
        demos.register(demos.DemoSpec(title="t", prompt="p", tier="core", embodiment="e", kind="gif", how="h"), root)
    with pytest.raises(ValueError):
        demos.register(demos.DemoSpec(title="t", prompt="p", tier="core", embodiment="e", kind="gif", how="h", payload=clip, files=[clip]), root)


def test_index_renders_every_entry(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    demos.register(demos.DemoSpec(title="A title", prompt="wave hello", tier="humanoid", embodiment="rig", kind="gif", how="h", files=[_gif(tmp_path / "a.gif")]), root)
    page = demos.write_index(root)
    text = page.read_text(encoding="utf-8")
    assert "wave hello" in text and "A title" in text
    assert "<title>Rigby Demos</title>" in text
