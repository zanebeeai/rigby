"""Render the demo set: several robots, the same plain-language prompts.

The point of the demo is not that one arm moves. It is that *the same words*
produce the corresponding motion on bodies that share no dimension, no joint
count, and no gripper -- and that the body-neutral reading of those words is
byte-identical across all of them.

So the manifest records the role-normalized schema hash beside each clip. Where a
prompt ran on more than one robot, that hash is the claim: one prompt, one
meaning, many bodies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rigby_general.pipeline import ingest_robot
from rigby_general.primitives import PrimitiveLibrary
from rigby_general.run import answer, unsupported_reason
from rigby_general.schema.inventory import load_inventory

import render_demo


ROOT = Path(__file__).resolve().parents[1]
ZOO_ROOT = ROOT / "assets" / "general" / "zoo"
MEDIA = ROOT / "docs" / "media"

# One shared prompt every robot gets, so the clips can be compared directly, plus
# one that suits each body in particular.
SHARED_PROMPT = "reach out as far as you can and then come back"

PER_ROBOT_PROMPTS: dict[str, tuple[str, ...]] = {
    "zoo_tool_arm": ("sweep slowly across in front of you",),
    "zoo_jaw_arm": ("wave three times",),
    "zoo_hand_arm": ("trace a big circle",),
    "zoo_dual_arm": ("lower the tool down toward the table",),
    "zoo_compact_arm": ("reach out quickly, just a little",),
    "zoo_long_arm": ("sweep across the workspace, then hold still",),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "robots")
    parser.add_argument("--out", type=Path, default=MEDIA)
    parser.add_argument("--robots", nargs="*", default=None)
    arguments = parser.parse_args()

    inventory = load_inventory()
    library = PrimitiveLibrary(arguments.library)
    arguments.out.mkdir(parents=True, exist_ok=True)

    robot_ids = arguments.robots or sorted(
        directory.name
        for directory in ZOO_ROOT.iterdir()
        if (directory / "robot.urdf").is_file()
    )

    entries: list[dict[str, object]] = []
    shared_hashes: dict[str, set[str]] = {}
    failures = 0

    for robot_id in robot_ids:
        if not library.has_library(robot_id):
            print(f"{robot_id:<18} no baked library; skipping")
            failures += 1
            continue

        robot = ingest_robot(ZOO_ROOT / robot_id / "robot.urdf", robot_id=robot_id)
        records = library.load(robot_id)
        refusals = library.load_failures(robot_id)
        scale = robot.morphology.scale
        centre = np.array(
            [
                scale.workspace_centroid_m.x,
                scale.workspace_centroid_m.y,
                scale.workspace_centroid_m.z,
            ]
        )

        prompts = (SHARED_PROMPT, *PER_ROBOT_PROMPTS.get(robot_id, ()))
        for prompt in prompts:
            started = time.perf_counter()
            result = answer(
                prompt,
                robot.manifest,
                robot.finalized.model,
                inventory,
                records,
                refusals,
            )
            if not result.accepted:
                print(
                    f"{robot_id:<18} {prompt[:40]:<42} REFUSED "
                    f"[{result.failure_stage}] {unsupported_reason(result)[:60]}"
                )
                failures += 1
                continue

            trace = result.certification.trace
            frames = render_demo.render_trace(
                robot.finalized.model,
                trace.qpos,
                trace.times_s,
                centre=centre,
                reach=scale.reach_radius_m,
            )
            slug = "-".join(prompt.lower().split())[:44].strip("-")
            path = arguments.out / f"{robot_id}-{slug}.gif"
            render_demo.write_gif(frames, path)

            digest = result.schema_program.role_normalized_hash()
            shared_hashes.setdefault(prompt, set()).add(digest)
            entries.append(
                {
                    "robot_id": robot_id,
                    "prompt": prompt,
                    "gif": path.name,
                    "duration_s": round(result.duration_s, 2),
                    "frames": len(frames),
                    "reach_radius_m": scale.reach_radius_m,
                    "dof": len(robot.manifest.dofs),
                    "effectors": [
                        effector.kind.value for effector in robot.morphology.effectors
                    ],
                    "schema_keys": list(result.schema_program.canonical_keys),
                    "role_normalized_hash": digest,
                    "tracking_error_m": round(
                        float(trace.tracking_error_m.max()), 6
                    ),
                    "certified_primitives_used": result.bound.used_certified_primitives,
                    "render_seconds": round(time.perf_counter() - started, 1),
                }
            )
            print(
                f"{robot_id:<18} {prompt[:40]:<42} {result.duration_s:5.1f}s "
                f"{len(frames):>3}f  -> {path.name}"
            )

    invariance = {
        prompt: {"robots": len(hashes), "distinct_readings": len(hashes)}
        for prompt, hashes in shared_hashes.items()
    }
    manifest = {
        "schema_version": "1.0",
        "shared_prompt": SHARED_PROMPT,
        "clips": entries,
        "schema_invariance": invariance,
    }
    manifest_path = arguments.out / "demo-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\n{len(entries)} clips -> {arguments.out}")
    shared = shared_hashes.get(SHARED_PROMPT, set())
    print(
        f"shared prompt read {len(shared)} distinct way(s) across "
        f"{len([e for e in entries if e['prompt'] == SHARED_PROMPT])} robots"
    )
    return 0 if failures == 0 and len(shared) <= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
