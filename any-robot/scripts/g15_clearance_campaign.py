"""Run the registered G15 protocol: clearance episodes on physics, paired across bodies, chain lengths and executors.

``nominal --body B --objects N`` runs the fifty registered seeds through the
generated tree; ``disturbed --body B`` runs the thirty disturbed seeds for
five objects through the tree and through its flat twin, the same seed, the
same world, the same disturbance, the same caps. Each episode is a chain of
TransferObject sessions on one clock; the root's verdict is the executor's,
and the oracle judges the final poses apart. Every episode keeps its row
(node verdicts, per-object outcomes, segment timings, leaf calls, the
work-area observations, planning calls and reuse counts); failed, undecided
and interrupted episodes and the first successes per cell are sealed segment
by segment, locally. Rendering is a separate pass. No API or model calls.

    python any-robot/scripts/g15_clearance_campaign.py nominal --body zoo_jaw_arm --objects 3 --out docs/results/g15-clearance --local any-robot/results/g15-clearance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from rigby_core.skills import Belief, Interrupt, Verdict, execute
from rigby_core.skills.clearance import clear_work_area_library

from rigby_general.evidence.capture import json_bytes
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.sensing import load_policy
from rigby_general.skills import disturbance_named, seal_tree_run
from rigby_general.skills.clear_work_area import LAYOUT, ChainClock, ClearWorkAreaRuntime, ClearanceWorld, Layout, build_clearance_world, final_object_poses, oracle_clear, predicates_for_clearance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as g10  # noqa: E402
import g15_protocol as protocol  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EPISODE_PROTOCOL = "rigby.clearance-episode/1"
KEEP_SUCCESSES = 2
KEEP_FAILURES = 15
"""Sealed failures per cell: every row is kept, every failure's segments up to this many; a ten-object failure is fifty megabytes."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registration() -> tuple[dict, dict]:
    registration = json.loads((protocol.PROTOCOL / "registration.json").read_bytes())
    if sha256_of(protocol.PROTOCOL / "corpus.json") != registration["files"]["corpus.json"]:
        raise SystemExit("corpus.json does not match its registration; refusing to run")
    if sha256_of(g10.G09 / "policy.json") != registration["files"]["g09/policy.json"]:
        raise SystemExit("the G09 policy does not match the registration; refusing to run")
    if hashlib.sha256(json_bytes(g10.environment().model_dump(mode="json"))).hexdigest() != registration["base_environment_sha256"]:
        raise SystemExit("the base world no longer hashes to the registration")
    corpus = json.loads((protocol.PROTOCOL / "corpus.json").read_bytes())
    if registration.get("layout") != LAYOUT.name:
        raise SystemExit("the layout in code is not the registered one; refusing to run")
    for count in protocol.CHAIN_LENGTHS:
        world = build_clearance_world(g10.environment(), count)
        if hashlib.sha256(json_bytes(world.environment.model_dump(mode="json"))).hexdigest() != registration["worlds"][str(count)]:
            raise SystemExit(f"the {count}-object world no longer hashes to the registration")
        for flat in (False, True):
            library = clear_work_area_library(world.objects, tuple(world.cells[o] for o in world.objects), flat=flat)
            if library.content_hash() != registration["libraries"][f"{count}-{'flat' if flat else 'tree'}"]:
                raise SystemExit(f"the {count}-object {'flat' if flat else 'tree'} library no longer hashes to the registration")
    return corpus, registration


def world_for(count: int, draw: dict, layout: Layout = LAYOUT) -> ClearanceWorld:
    return build_clearance_world(g10.environment(), count, draw=draw, layout=layout)


def run_episode(body: str, source: Path, robot, world: ClearanceWorld, policy: dict, *, flat: bool, disturbance: tuple[str, str] | None, seed_label: str, observe_each_pass: bool = True) -> dict:
    """One clearance episode: the tree (or its flat twin) over the chain of
    sessions. ``observe_each_pass=False`` runs the first structure, for the
    baseline comparison only; the campaign never does."""

    library = clear_work_area_library(world.objects, tuple(world.cells[o] for o in world.objects), flat=flat, observe_each_pass=observe_each_pass)
    interrupt = Interrupt()
    attached = None
    if disturbance is not None:
        attached = (disturbance[0], disturbance_named(disturbance[1]))
    runtime = ClearWorkAreaRuntime(body=body, source=source, robot=robot, world=world, policy=policy, interrupt=interrupt, configuration=protocol.CONFIGURATION, disturbance=attached, seed_label=seed_label,
                                   report_placements=observe_each_pass)
    effector = robot.morphology.grasping_effectors[0].chain_id
    started = time.perf_counter()
    planning = {"library_generated": 1, "tree_expansions": 1}
    tree = library.expand("clear_work_area", {"effector": effector})
    record = execute(tree, library, runtime, predicates_for_clearance(world), clock=ChainClock(runtime), interrupt=interrupt, belief=Belief())
    poses = final_object_poses(runtime)
    wall = time.perf_counter() - started
    oracle = oracle_clear(world, poses)
    success = record.verdict is Verdict.SUCCESS
    arguments = {n.node_id: n.arguments for n in tree.root.walk()}
    per_object = {}
    for node in record.root.walk():
        if node.skill_id == "transfer_object":
            name = arguments[node.node_id]["object"]
            per_object.setdefault(name, []).append({"verdict": node.verdict.value, "reason": node.reason, "attempts": node.attempts, "started_s": node.started_s, "ended_s": node.ended_s})
    placed_by_belief = [o for o, c in world.cells.items() if record.belief.get(f"placed:{o}:{c}") is True]
    nodes = {"total": sum(1 for _ in tree.root.walk()), "max_depth": tree.max_depth, "definitions": len(library.skills), "transfer_instances": sum(1 for n in tree.root.walk() if n.skill_id == "transfer_object")}
    reuse = {"shared_transfer_definitions": 11, "instances_per_definition": len(world.objects), "reused_bindings": (len(world.objects) - 1) * 11, "sessions_opened": runtime.switches, "groundings_per_object": 1}
    looks = [o for o in runtime.observations]
    return {
        "verdict": record.verdict.value, "root_reason": record.root.reason, "interrupted": record.interrupted, "interrupt_reason": record.interrupt_reason, "skill_success": success,
        "oracle": oracle, "false_completion": bool(success and not oracle["root_success"]), "physics_s": float(runtime.clock_s), "wall_seconds": wall,
        "passes": next((n.evidence.get("attempts", n.attempts) for n in record.root.walk() if n.skill_id == "clear_until_clear"), None),
        "per_object": per_object, "placed_by_belief": placed_by_belief, "placed_by_oracle": oracle["placed"], "work_area_clear_observed": record.belief.get("work_area_clear"),
        "looks": len(looks), "placements_undone_by_look": sorted({name for look in looks for name, in_cell in look.get("in_cells", {}).items() if in_cell is False}),
        "library_id": library.library_id, "layout": world.layout.name,
        "segments": [{"index": s.index, "object": s.object_name, "started_s": s.started_s, "ended_s": s.ended_s, "leaves": s.leaves} for s in runtime.segments],
        "leaf_calls": runtime.calls, "verdict_trail": runtime.verdicts, "observations": runtime.observations, "events": [e for s in runtime.segments for e in s.session.events],
        "disturbance_log": [entry for s in runtime.segments if s.session.disturbance is not None for entry in s.session.disturbance.log],
        "planning_calls": planning, "tree": nodes, "reuse": reuse, "effector": effector,
        "_runtime": runtime, "_record": record, "_tree": tree, "_library": library,
    }


def seal_episode(row: dict, *, local: Path, label: str, caption: str, task_extra: dict) -> dict:
    """Every segment of the chain as its own bundle in the G01 layout, the full execution record on each."""

    runtime, record, tree, library = row["_runtime"], row["_record"], row["_tree"], row["_library"]
    sealed = []
    for segment in runtime.segments:
        destination = local / label / f"segment-{segment.index:02d}-{segment.object_name}" / "physical"
        if destination.exists():
            shutil.rmtree(destination)
        bundle = seal_tree_run(destination, session=segment.session, library=library, tree=tree, record=record, label=f"{label}-{segment.index:02d}", caption=f"{caption} | segment {segment.index} {segment.object_name}",
                               runtime_calls=[c for c in runtime.calls if c.get("segment") == segment.index], goal="G15", protocol=EPISODE_PROTOCOL,
                               task_extra={**task_extra, "segment": segment.index, "segment_object": segment.object_name, "segment_started_s": segment.started_s, "segment_ended_s": segment.ended_s,
                                           "objects": list(runtime.world.objects), "cells": dict(runtime.world.cells), "chain": "one clock across every segment; each segment compiled with its object under the canonical names"},
                               outcome_extra={"skill_success": row["skill_success"], "oracle": row["oracle"], "false_completion": row["false_completion"], "per_object": row["per_object"], "passes": row["passes"]},
                               observation={"policy": "declared_sensors_with_oracle_labels_kept_apart", "inputs": ["joint encoders", "grasp point through the model", "contact force per opposition group", "cameras by ray visibility"], "vlm": False})
        sealed.append({"segment": segment.index, "object": segment.object_name, "bundle": bundle["bundle"], "sha256": bundle["sha256"], "trace_sha256": bundle["trace_sha256"], "simulation_duration_s": bundle["simulation_duration_s"], "replay_agrees": bundle["replay_agrees"]})
    return {"segments": sealed}


def public_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("nominal", "disturbed"))
    parser.add_argument("--body", required=True)
    parser.add_argument("--objects", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N seeds (engineering smoke; never a scored run)")
    parser.add_argument("--resume", action="store_true", help="keep the episodes already in the cell's trials.json (their sealed bundles intact) and run only the rest")
    args = parser.parse_args()
    corpus, registration = load_registration()
    policy = load_policy(g10.G09 / "policy.json")
    body = args.body
    source = ROOT / "assets/general/zoo" / body / "robot.urdf"
    robot = ingest_robot(source, robot_id=body)
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                  "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__, "registration_sha256": registration["registration_sha256"],
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "generation_calls": 0, "scored": args.limit is None, "phase": args.phase, "body": body}
    if args.phase == "nominal":
        count = int(args.objects)
        name = f"nominal-{body}-{count}"
        seeds = corpus["nominal"][: args.limit] if args.limit else corpus["nominal"]
        runs = [(draw, False, None) for draw in seeds]
    else:
        count = corpus["disturbed_count"]
        name = f"disturbed-{body}"
        seeds = corpus["disturbed"][: args.limit] if args.limit else corpus["disturbed"]
        runs = [(draw, flat, (f"cube_{draw['object_index'] + 1:02d}", draw["kind"])) for draw in seeds for flat in (False, True)]
    out = args.out / name
    local = args.local / name
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    rows = []
    kept: dict[str, int] = {}
    if args.resume and (out / "trials.json").exists() and (out / "provenance.json").exists():
        # A cell interrupted mid-run (the machine went down) keeps every
        # episode it finished, provided the sealed bundles it claims are
        # whole; a row whose bundle is missing or unfinished is dropped and
        # its seed runs again. The kept counts are rebuilt from the rows.
        earlier = json.loads((out / "provenance.json").read_bytes())
        if earlier["registration_sha256"] != registration["registration_sha256"]:
            raise SystemExit("the cell on disk was run under another registration; refusing to resume")
        for row in json.loads((out / "trials.json").read_bytes()):
            if "sealed" in row and not all((REPO / seg["bundle"] / "manifest.json").is_file() for seg in row["sealed"]["segments"]):
                print(json.dumps({"dropped": row["episode_id"], "why": "sealed bundle incomplete"}), flush=True)
                continue
            rows.append(row)
            cell = f"{row['executor']}-{row['disturbance']['kind'] if row['disturbance'] else 'nominal'}"
            if "sealed" in row:
                key = cell if row["skill_success"] else cell + "-fail"
                kept[key] = kept.get(key, 0) + 1
        provenance = {**earlier, "resumed_at_utc": datetime.now(timezone.utc).isoformat(), "resumed_with": len(rows), "resumed_commit": provenance["commit"], "resumed_working_tree_dirty": provenance["working_tree_dirty"]}
        print(json.dumps({"resumed": name, "episodes_kept": len(rows)}), flush=True)
    (out / "provenance.json").write_bytes(json_bytes(provenance))
    done = {row["episode_id"] for row in rows}
    for draw, flat, disturbance in runs:
        executor = "flat" if flat else "tree"
        episode_id = f"{body}-{count}-{executor}-{draw['seed']}" + (f"-{disturbance[1]}" if disturbance else "")
        if episode_id in done:
            continue
        world = world_for(count, draw)
        row = run_episode(body, source, robot, world, policy, flat=flat, disturbance=disturbance, seed_label=episode_id)
        row.update({"episode_id": episode_id, "zoo_id": body, "objects": count, "executor": executor, "seed": draw["seed"], "draw_sha256": hashlib.sha256(json_bytes(draw)).hexdigest(),
                    "disturbance": None if disturbance is None else {"object": disturbance[0], "kind": disturbance[1]}, "cap_s": corpus["budgets"]["cap_s"][str(count)], "within_cap": row["physics_s"] <= corpus["budgets"]["cap_s"][str(count)] + 1e-9})
        cell = f"{executor}-{disturbance[1] if disturbance else 'nominal'}"
        keep = (not row["skill_success"] and kept.get(cell + "-fail", 0) < KEEP_FAILURES) or (row["skill_success"] and kept.get(cell, 0) < KEEP_SUCCESSES)
        if keep:
            if not row["skill_success"]:
                kept[cell + "-fail"] = kept.get(cell + "-fail", 0) + 1
            caption = f"{body} | {count} objects | {executor} | seed {draw['seed']}" + (f" | {disturbance[1]} on {disturbance[0]}" if disturbance else "") + f" | {row['verdict']}" + (f" | {row['root_reason']}" if row["root_reason"] else "")
            row["sealed"] = seal_episode(row, local=local, label=episode_id, caption=caption, task_extra={"episode_id": episode_id, "seed": draw["seed"], "draw": draw, "executor": executor, "objects": count, "disturbance": row["disturbance"],
                                                                                                          "registration_sha256": registration["registration_sha256"], "sensor_configuration": protocol.CONFIGURATION})
            if row["skill_success"]:
                kept[cell] = kept.get(cell, 0) + 1
        rows.append(public_row(row))
        print(json.dumps({"episode": episode_id, "verdict": row["verdict"], "reason": row["root_reason"], "placed": f"{row['placed_by_oracle']}/{count}", "oracle_root": row["oracle"]["root_success"], "false_completion": row["false_completion"],
                          "physics_s": round(row["physics_s"], 1), "wall": round(row["wall_seconds"], 1), "passes": row["passes"], "kept": keep}), flush=True)
        (out / "trials.json").write_bytes(json_bytes(rows))
        del row
    summary = {"phase": args.phase, "body": body, "objects": count, "episodes": len(rows), "provenance": provenance, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "by_executor": {}}
    for executor in sorted({r["executor"] for r in rows}):
        subset = [r for r in rows if r["executor"] == executor]
        summary["by_executor"][executor] = {"episodes": len(subset), "successes": sum(r["skill_success"] for r in subset), "false_completions": sum(r["false_completion"] for r in subset),
                                            "oracle_root": sum(r["oracle"]["root_success"] for r in subset), "within_cap": sum(r["within_cap"] for r in subset),
                                            "mean_placed": float(np.mean([r["placed_by_oracle"] for r in subset])), "physics_s_mean": float(np.mean([r["physics_s"] for r in subset])), "wall_s_mean": float(np.mean([r["wall_seconds"] for r in subset])),
                                            "reasons": {}}
        for r in subset:
            if not r["skill_success"]:
                key = r["root_reason"] or r["verdict"]
                summary["by_executor"][executor]["reasons"][key] = summary["by_executor"][executor]["reasons"].get(key, 0) + 1
    (out / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary["by_executor"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
