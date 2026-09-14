"""Run the registered G17 protocol for one body: the hundred travel trials and the thirty perturbation trials, recorded on physics.

    python any-robot/scripts/g17_locomotion_campaign.py --body mobile_dog_arm --out docs/results/g17-locomotion --local any-robot/results/g17-locomotion

Every trial keeps its row: the outcome and its reason, the waypoints
reached with their times, falls and collisions, the actuator energy,
the distance and speeds, the controller latency, the time against the
cap, the actuation log (which actuators saturated and for what fraction
of the run, the peak forces, what external input acted) and the support
log (which declared members bore the body and for what fraction, any
undeclared contact). Every failure and the first successes of each kind
are sealed as replayable bundles, locally; rendering is a separate pass.
`--resume` keeps the rows already on disk whose sealed bundles are
whole and runs only the rest. No API or model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_general.evidence.capture import json_bytes
from rigby_general.mobility import load_mobile_body, measure_mobile_body
from rigby_general.mobility.evidence import seal_run
from rigby_general.mobility.trials import Perturbation, course_world_xml, run_travel
from rigby_general.mobility.world import course_v1


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MOBILE = ROOT / "assets/general/mobile"
PROTOCOL = ROOT / "assets/general/research-protocols/g17-locomotion-v1"
TRIAL_PROTOCOL = "rigby.mobile-locomotion-trial/1"
KEEP_SUCCESSES = 2
CAMERA = {"centre": [1.5, 0.0, 0.25], "reach": 2.3}


def load_protocol() -> tuple[dict, dict]:
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    for name, digest in registration["files"].items():
        if hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{name} no longer matches its registration; refusing to run")
    return json.loads((PROTOCOL / "protocol.json").read_bytes()), registration


def perturbation_of(spec: dict | None) -> Perturbation | None:
    if spec is None:
        return None
    return Perturbation(kind=spec["kind"], detail=spec["detail"], at_s=float(spec.get("at_s", 0.0)), duration_s=float(spec.get("duration_s", 0.0)), magnitude=float(spec.get("magnitude", 0.0)),
                        direction=tuple(spec.get("direction", (1.0, 0.0, 0.0))), target=spec.get("target", ""), patch_friction=float(spec.get("patch_friction", 0.15)),
                        patch_centre_m=tuple(spec.get("patch_centre_m", (1.5, 0.0))), patch_half_m=tuple(spec.get("patch_half_m", (0.4, 0.8))))


def controls_note(body_kind: str, perturbation: Perturbation | None) -> str:
    controller = {"legged": "trot controller on the leg servos", "wheeled": "balance controller on the wheel drives and leg servos", "crawling": "tripod crawl on the tentacle servos"}[body_kind]
    if perturbation is None:
        return f"{controller}; navigator; nothing else acting"
    if perturbation.kind == "push":
        return f"{controller}; navigator; {perturbation.detail} (declared, recorded)"
    if perturbation.kind == "patch":
        return f"{controller}; navigator; a slick patch in the world (friction {perturbation.patch_friction:.2f})"
    return f"{controller}; navigator; support disturbance: {perturbation.detail}"


def row_of(trial: dict, result, wall: float) -> dict:
    saturated = {a: f for a, f in result.actuation["saturation_fraction"].items() if f >= 0.05}
    return {"trial_id": trial["trial_id"], "body": trial["body"], "kind": trial["kind"], "seed": trial["seed"], "perturbation": trial["perturbation"], "cap_s": trial["cap_s"],
            "success": result.success, "reason": result.reason, "fell": result.fell, "fall_time_s": result.fall_time_s, "reached": result.reached, "waypoints_reached": len(result.reached), "navigator_log": result.navigator_log,
            "perturbation_log": result.perturbation_log, "collisions": result.collisions, "energy_j": round(result.energy_j, 2), "distance_m": round(result.distance_m, 3), "mean_speed_mps": round(result.mean_speed_mps, 4), "peak_speed_mps": round(result.peak_speed_mps, 4),
            "duration_s": round(result.duration_s, 3), "time_budget_fraction": round(result.duration_s / result.cap_s, 4), "control_latency_s": {k: round(v, 6) for k, v in result.control_latency_s.items()}, "stable_at_end": result.stable_at_end,
            "final_distance_m": round(result.final_distance_m, 4), "contacts_at_end": list(result.contacts_at_end), "samples": result.samples,
            "actuation": {"provenance": result.actuation["provenance"], "controller": result.actuation["controller"], "control_hz": result.actuation["control_hz"], "saturated_actuators": saturated, "saturation_max": max(result.actuation["saturation_fraction"].values()),
                          "peak_force_max": max(result.actuation["peak_force"].values()), "external_inputs": result.actuation["external_inputs"], "root_writes": result.actuation["root_writes"], "artificial_support": result.actuation["artificial_support"]},
            "support": result.support, "wall_seconds": round(wall, 2)}


def summarise(rows: list[dict], protocol: dict, body_id: str) -> dict:
    kinds = {}
    for kind in ("travel", "push", "patch", "support"):
        cell = [r for r in rows if r["kind"] == kind]
        if not cell:
            continue
        successes = [r for r in cell if r["success"]]
        kinds[kind] = {"trials": len(cell), "successes": len(successes), "rate": round(len(successes) / len(cell), 4), "falls": sum(1 for r in cell if r["fell"]), "time_caps": sum(1 for r in cell if r["reason"].startswith("time cap")),
                       "unstable_ends": sum(1 for r in cell if r["reason"].startswith("not stable")), "drifted": sum(1 for r in cell if r["reason"].startswith("drifted")), "collisions_total": sum(r["collisions"] for r in cell), "trials_with_collisions": sum(1 for r in cell if r["collisions"] > 0),
                       "energy_j": {"mean": round(float(np.mean([r["energy_j"] for r in cell])), 1), "max": round(max(r["energy_j"] for r in cell), 1)},
                       "mean_speed_mps": {"mean": round(float(np.mean([r["mean_speed_mps"] for r in successes])), 4) if successes else None, "min": round(min(r["mean_speed_mps"] for r in successes), 4) if successes else None},
                       "peak_speed_mps": {"max": round(max(r["peak_speed_mps"] for r in cell), 3)}, "duration_s": {"mean": round(float(np.mean([r["duration_s"] for r in successes])), 1) if successes else None, "max": round(max(r["duration_s"] for r in cell), 1)},
                       "time_budget_fraction": {"mean": round(float(np.mean([r["time_budget_fraction"] for r in successes])), 4) if successes else None, "max": round(max(r["time_budget_fraction"] for r in cell), 4)},
                       "control_latency_ms": {"p99_max": round(1e3 * max(r["control_latency_s"]["p99"] for r in cell), 3), "mean": round(1e3 * float(np.mean([r["control_latency_s"]["mean"] for r in cell])), 4)},
                       "saturation_max": round(max(r["actuation"]["saturation_max"] for r in cell), 4), "undeclared_contacts": sorted({m for r in cell for m in r["support"]["undeclared_contact_fraction"]}), "sealed": sum(1 for r in cell if "sealed" in r)}
    perturbation = [r for r in rows if r["kind"] != "travel"]
    targets = protocol["targets"]
    travel = kinds.get("travel", {"trials": 0, "successes": 0})
    return {"body": body_id, "kinds": kinds, "travel": {"successes": travel["successes"], "trials": travel["trials"], "target": f">= {targets['travel_min']} of {targets['travel_of']}", "met": travel["trials"] == targets["travel_of"] and travel["successes"] >= targets["travel_min"]},
            "perturbation": {"successes": sum(1 for r in perturbation if r["success"]), "trials": len(perturbation), "target": f">= {targets['perturbation_min']} of {targets['perturbation_of']}", "met": len(perturbation) == targets["perturbation_of"] and sum(1 for r in perturbation if r["success"]) >= targets["perturbation_min"],
                             "by_kind": {k: f"{v['successes']}/{v['trials']}" for k, v in kinds.items() if k != "travel"}},
            "provenance": sorted({r["actuation"]["provenance"] for r in rows}), "controller": sorted({r["actuation"]["controller"] for r in rows}), "root_writes": sorted({r["actuation"]["root_writes"] for r in rows}), "artificial_support": sorted({r["actuation"]["artificial_support"] for r in rows})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--body", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--kinds", default="travel,push,patch,support")
    parser.add_argument("--limit", type=int, default=None, help="only the first N trials of each kind (engineering smoke; never a scored run)")
    parser.add_argument("--resume", action="store_true", help="keep the rows already on disk (their sealed bundles whole) and run only the rest")
    args = parser.parse_args()
    protocol, registration = load_protocol()
    course = course_v1()
    if course.sha256() != protocol["course_sha256"]:
        raise SystemExit("the course no longer hashes to the protocol's; refusing to run")
    body = load_mobile_body(MOBILE / args.body)
    if body.model_sha256 != protocol["bodies"][args.body]["model_sha256"]:
        raise SystemExit(f"{args.body} no longer hashes to the protocol's; refusing to run")
    manifest = measure_mobile_body(body)
    out = args.out / args.body
    out.mkdir(parents=True, exist_ok=True)
    local = args.local / args.body
    kinds = args.kinds.split(",")
    trials = [t for t in protocol["trials"][args.body] if t["kind"] in kinds]
    if args.limit is not None:
        trials = [t for k in kinds for t in [t for t in trials if t["kind"] == k][: args.limit]]
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                  "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__, "registration_sha256": registration["registration_sha256"], "protocol_sha256": registration["files"]["protocol.json"],
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "generation_calls": 0, "scored": args.limit is None, "kinds": kinds}
    rows: list[dict] = []
    if args.resume and (out / "trials.json").exists() and (out / "provenance.json").exists():
        earlier = json.loads((out / "provenance.json").read_bytes())
        if earlier.get("registration_sha256") != registration["registration_sha256"]:
            raise SystemExit("the rows on disk were run under another registration; refusing to resume")
        for row in json.loads((out / "trials.json").read_bytes()):
            if "sealed" in row and not (REPO / row["sealed"]["bundle"] / "manifest.json").is_file():
                print(json.dumps({"dropped": row["trial_id"], "why": "sealed bundle incomplete"}), flush=True)
                continue
            rows.append(row)
        provenance = {**earlier, "resumed_at_utc": datetime.now(timezone.utc).isoformat(), "resumed_with": len(rows), "resumed_commit": provenance["commit"], "resumed_working_tree_dirty": provenance["working_tree_dirty"]}
        print(json.dumps({"resumed": args.body, "trials_kept": len(rows)}), flush=True)
    done = {r["trial_id"] for r in rows}
    kept = {}
    for row in rows:
        if "sealed" in row and row["success"]:
            kept[row["kind"]] = kept.get(row["kind"], 0) + 1
    (out / "provenance.json").write_bytes(json_bytes(provenance))
    for trial in trials:
        if trial["trial_id"] in done:
            continue
        perturbation = perturbation_of(trial["perturbation"])
        started = time.perf_counter()
        result = run_travel(body, course, seed=trial["seed"], waypoints=[tuple(w) for w in trial["waypoints"]], cap_s=trial["cap_s"], jitter_xy_m=protocol["jitter"]["xy_m"], jitter_yaw_deg=protocol["jitter"]["yaw_deg"], perturbation=perturbation, settle_s=protocol["settle_s"])
        row = row_of(trial, result, time.perf_counter() - started)
        keep = (not result.success) or kept.get(trial["kind"], 0) < KEEP_SUCCESSES
        if keep:
            if result.success:
                kept[trial["kind"]] = kept.get(trial["kind"], 0) + 1
            world_xml = course_world_xml(body, course, perturbation)
            model = mujoco.MjSpec.from_string(world_xml).compile()
            from dataclasses import replace

            on_course = replace(body, floor_model=model, floor_xml=world_xml)
            status = "success" if result.success else ("fell" if result.fell else ("time_cap" if result.reason.startswith("time cap") else ("unstable" if result.reason.startswith("not stable") else "drifted")))
            caption = f"{trial['kind']} trial {trial['trial_id']}: {trial['perturbation']['detail'] if trial['perturbation'] else 'travel start -> station -> start'}"
            sealed = seal_run(local / trial["trial_id"] / "physical", body=on_course, manifest=manifest, run=result.run, label=trial["trial_id"], caption=caption[:150], goal="G17", protocol=TRIAL_PROTOCOL,
                              test={"test": "locomotion_trial", "trial_id": trial["trial_id"], "kind": trial["kind"], "seed": trial["seed"], "waypoints": trial["waypoints"], "cap_s": trial["cap_s"], "perturbation": trial["perturbation"], "course_id": course.course_id, "course_sha256": course.sha256(),
                                    "registration_sha256": registration["registration_sha256"], "controller": {"name": result.actuation["controller"], "provenance": result.actuation["provenance"]}},
                              outcome={"status": status, "success": result.success, "reason": result.reason, "fell": result.fell, "fall_time_s": result.fall_time_s, "reached": result.reached, "collisions": result.collisions, "energy_j": result.energy_j, "distance_m": result.distance_m,
                                       "duration_s": result.duration_s, "final_distance_m": result.final_distance_m, "stable_at_end": result.stable_at_end, "contacts_at_end": list(result.contacts_at_end), "perturbation_log": result.perturbation_log, "navigator_log": result.navigator_log, "support": result.support},
                              camera=CAMERA, world_xml=world_xml, model=model, controls_note=controls_note(body.declaration["base_kind"], perturbation))
            row["sealed"] = {**sealed, "bundle": Path(sealed["bundle"]).relative_to(REPO).as_posix() if Path(sealed["bundle"]).is_absolute() else sealed["bundle"]}
        rows.append(row)
        (out / "trials.json").write_bytes(json_bytes(rows))
        print(json.dumps({"trial": trial["trial_id"], "success": result.success, "reason": result.reason, "t": round(result.duration_s, 1), "collisions": result.collisions, "energy_j": round(result.energy_j), "wall": round(row["wall_seconds"], 1), "kept": keep}), flush=True)
    summary = summarise(rows, protocol, args.body)
    (out / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps({"body": args.body, "travel": summary["travel"], "perturbation": summary["perturbation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
