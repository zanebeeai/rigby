"""The published invariance count must distinguish robots from readings."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_recorded_six_body_demo_counts_bodies_and_readings_separately(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "any-robot" / "scripts"))
    build_demos = importlib.import_module("build_demos")
    manifest = json.loads(
        (root / "docs/results/media/any-robot-zoo-demos/demo-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    entries = manifest["clips"]
    prompt = manifest["shared_prompt"]

    assert manifest["schema_invariance"] == build_demos.schema_invariance(entries)
    assert build_demos.schema_invariance(entries)[prompt] == {
        "robots": 6,
        "distinct_readings": 1,
    }

    # A replay is not a new body. A conflicting reading must stay visible.
    duplicate = next(entry.copy() for entry in entries if entry["prompt"] == prompt)
    duplicate["role_normalized_hash"] = "a-different-semantic-reading"
    assert build_demos.schema_invariance([*entries, duplicate])[prompt] == {
        "robots": 6,
        "distinct_readings": 2,
    }
