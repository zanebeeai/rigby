"""Reproduce the bounded, offline embodiment probe used in the research review.

Run from the repository root with .venv/Scripts/python.exe on Windows.
This writes only review evidence; it does not alter the product or baked library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy

from rigby_general.bake.enumerate import build_candidate
from rigby_general.bake.runner import _attempt
from rigby_general.config import base_tree_fingerprint
from rigby_general.pipeline import ingest_robot
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.primitives import PrimitiveRecord
from rigby_general.run import answer
from rigby_general.schema.inventory import afforded_entries, load_inventory


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PROMPT = "reach out as far as you can and then come back"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory()
    fingerprint = base_tree_fingerprint().sha256
    rows = []
    for source in sorted((ROOT / "any-robot/assets/general/zoo").glob("*/robot.urdf")):
        robot = ingest_robot(source, robot_id=source.parent.name)
        planner = OfflineSchemaPlanner(inventory)
        program = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
        records, failures, attempts = [], [], []
        for segment in program.segments:
            entry = inventory.by_binding_key(
                f"{segment.motion_schema.canonical_key}|{segment.figure.role.value}->{segment.ground.role.value}"
            )
            outcome = _attempt(
                build_candidate(entry, segment.region.remove), robot.manifest,
                robot.finalized.model, inventory, policy=None, fingerprint=fingerprint,
            )
            certified = isinstance(outcome, PrimitiveRecord)
            (records if certified else failures).append(outcome)
            attempts.append({"entry": entry.entry_id, "remove": segment.region.remove.value,
                             "certified": certified,
                             "failure": None if certified else str(outcome)})
        result = answer(PROMPT, robot.manifest, robot.finalized.model, inventory,
                        tuple(records), tuple(failures), planner=planner)
        row = result.summary()
        row.update({"source_urdf": str(source.relative_to(ROOT)).replace("\\", "/"),
                    "dof": len(robot.manifest.dofs),
                    "reach_m": robot.morphology.scale.reach_radius_m,
                    "fresh_bake_attempts": attempts,
                    "bound_hash": result.bound.program.role_normalized_hash() if result.bound else None})
        rows.append(row)
        print(json.dumps(row), flush=True)

    old = json.loads((ROOT / "docs/results/media/any-robot-zoo-demos/demo-manifest.json").read_text())
    grouped = defaultdict(list)
    for clip in old["clips"]:
        grouped[clip["prompt"]].append(clip)
    corrected = {
        prompt: {"robot_count_from_clip_rows": len({c["robot_id"] for c in clips}),
                 "distinct_hashes_from_clip_rows": len({c["role_normalized_hash"] for c in clips}),
                 "published_summary": old["schema_invariance"].get(prompt)}
        for prompt, clips in grouped.items()
    }
    payload = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_patch_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
        ).hexdigest(),
        "platform": platform.platform(), "python": sys.version,
        "mujoco": mujoco.__version__, "numpy": numpy.__version__,
        "core_fingerprint": fingerprint, "prompt": PROMPT,
        "inventory_entry_count": len(inventory.entries),
        "experiment_scope": "Fresh exact-region bake for two requested free-space segments per robot, then complete prompt path. No neural model, no old baked library, no object task.",
        "results": rows,
        "historical_demo_summary_recomputed": corrected,
    }
    (arguments.out / "embodiment-probe.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
