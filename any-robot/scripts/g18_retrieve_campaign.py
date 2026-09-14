"""Run the registered G18 protocol for one body: the hundred nominal retrieves and the thirty disturbed ones, recorded on physics.

    python any-robot/scripts/g18_retrieve_campaign.py --body mobile_dog_arm_v2 --out docs/results/g18-retrieve --local any-robot/results/g18-retrieve

Every trial keeps its row: the outcome and its reason, every phase with
its attempts, timing, support set and holding limb (the resource
schedule), the hold events, the invariant checks, the object's slip
while carried, falls and collisions, energy, distance, latency, the
time against the cap, the actuation and support logs. Every failure and
the first successes of each stage are sealed as replayable bundles,
locally; rendering is a separate pass. `--resume` keeps the rows on
disk whose sealed bundles are whole. No API or model calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_general.evidence.capture import json_bytes
from rigby_general.mobility import load_mobile_body, measure_mobile_body
from rigby_general.mobility.evidence import seal_run
from rigby_general.mobility.retrieve import RetrieveDisturbance, run_retrieve
from rigby_general.mobility.trials import course_world_xml
from rigby_general.mobility.world import course_v1


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MOBILE = ROOT / "assets/general/mobile"
PROTOCOL = ROOT / "assets/general/research-protocols/g18-retrieve-v1"
TRIAL_PROTOCOL = "rigby.mobile-retrieve-trial/1"
KEEP_SUCCESSES = 2
CAMERA = {"centre": [1.7, 1.1, 0.25], "reach": 2.6}


def load_protocol() -> tuple[dict, dict]:
    registration = json.loads((PROTOCOL / "registration.json").read_bytes())
    for name, digest in registration["files"].items():
        if hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{name} no longer matches its registration; refusing to run")
    return json.loads((PROTOCOL / "protocol.json").read_bytes()), registration


def disturbance_of(spec: dict | None) -> RetrieveDisturbance | None:
    if spec is None:
        return None
    return RetrieveDisturbance(kind=spec["kind"], detail=spec["detail"], phase=spec["phase"], offset_s=float(spec["offset_s"]), duration_s=float(spec["duration_s"]), magnitude=float(spec["magnitude"]),
                               direction=tuple(spec.get("direction", (1.0, 0.0, 0.0))), target=spec.get("target", ""))


def controls_note(kind: str, disturbance: RetrieveDisturbance | None) -> str:
    controller = {"legged": "trot + arm reach", "wheeled": "balance + arm reach", "crawling": "tripod crawl + tentacle reach"}[kind]
    if disturbance is None:
        return f"{controller}; navigator; nothing else acting"
    return f"{controller}; navigator; {disturbance.detail} (declared, recorded)"


def row_of(trial: dict, result, wall: float) -> dict:
    saturated = {a: f for a, f in result.actuation["saturation_fraction"].items() if f >= 0.05}
    return {"trial_id": trial["trial_id"], "body": trial["body"], "kind": trial["kind"], "stage": trial["stage"], "seed": trial["seed"], "disturbance": trial["disturbance"], "cap_s": trial["cap_s"],
            "success": result.success, "reason": result.reason, "fell": result.fell, "fall_time_s": result.fall_time_s, "cube_in_tray": result.cube_in_tray, "cube_final": result.cube_final, "at_start": result.at_start, "stable_at_end": result.stable_at_end,
            "phases": [{"phase": p.phase, "attempt": p.attempts, "started_s": round(p.started_s, 3), "ended_s": round(p.ended_s, 3), "success": p.success, "reason": p.reason, "support": list(p.support), "holding": p.holding or None, "detail": {k: v for k, v in p.detail.items() if k != "navigator_log"}} for p in result.phases],
            "phases_completed": sorted({p.phase for p in result.phases if p.success}), "schedule": result.schedule, "hold_events": result.hold_events, "invariant_violations": result.invariant_violations[:20], "invariant_violation_count": len(result.invariant_violations),
            "max_object_slip_m": round(result.max_object_slip_m, 4), "collisions": result.collisions, "energy_j": round(result.energy_j, 2), "distance_m": round(result.distance_m, 3), "duration_s": round(result.duration_s, 3), "time_budget_fraction": round(result.duration_s / result.cap_s, 4),
            "control_latency_s": {k: round(v, 6) for k, v in result.control_latency_s.items()}, "contacts_at_end": list(result.contacts_at_end), "samples": result.samples, "disturbance_log": result.disturbance_log,
            "actuation": {"provenance": result.actuation["provenance"], "controller": result.actuation["controller"], "saturated_actuators": saturated, "saturation_max": max(result.actuation["saturation_fraction"].values()), "peak_force_max": max(result.actuation["peak_force"].values()),
                          "external_inputs": result.actuation["external_inputs"], "root_writes": result.actuation["root_writes"], "object_writes": result.actuation["object_writes"], "artificial_support": result.actuation["artificial_support"]},
            "support": result.support, "wall_seconds": round(wall, 2)}


def summarise(rows: list[dict], protocol: dict, body_id: str) -> dict:
    stages = {}
    for stage in ("nominal", "navigation", "carrying", "placement"):
        cell = [r for r in rows if r["stage"] == stage]
        if not cell:
            continue
        successes = [r for r in cell if r["success"]]
        farthest = {}
        for r in cell:
            last = r["phases"][-1]["phase"] if r["phases"] else "none"
            farthest[last] = farthest.get(last, 0) + 1
        stages[stage] = {"trials": len(cell), "successes": len(successes), "rate": round(len(successes) / len(cell), 4), "falls": sum(1 for r in cell if r["fell"]), "cube_in_tray": sum(1 for r in cell if r["cube_in_tray"]),
                         "hold_lost": sum(1 for r in cell if any(e["event"] == "lost" for e in r["hold_events"])), "recovered_after_loss": sum(1 for r in cell if r["success"] and any(e["event"] == "lost" for e in r["hold_events"])),
                         "retries_used": sum(1 for r in cell if any(p["attempt"] > 1 for p in r["phases"])), "invariant_violations": sum(r["invariant_violation_count"] for r in cell), "collisions_total": sum(r["collisions"] for r in cell),
                         "last_phase_reached": farthest, "max_object_slip_m": round(max((r["max_object_slip_m"] for r in cell), default=0.0), 4), "energy_j_mean": round(float(np.mean([r["energy_j"] for r in cell])), 1),
                         "duration_s": {"mean": round(float(np.mean([r["duration_s"] for r in successes])), 1) if successes else None, "max": round(max(r["duration_s"] for r in cell), 1)},
                         "time_budget_fraction_max": round(max(r["time_budget_fraction"] for r in cell), 4), "control_latency_ms_p99_max": round(1e3 * max(r["control_latency_s"]["p99"] for r in cell), 3), "sealed": sum(1 for r in cell if "sealed" in r)}
    nominal = [r for r in rows if r["kind"] == "nominal"]
    disturbed = [r for r in rows if r["kind"] == "disturbed"]
    targets = protocol["targets"]
    reasons = {}
    for r in rows:
        if not r["success"]:
            key = r["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    return {"body": body_id, "stages": stages,
            "nominal": {"successes": sum(1 for r in nominal if r["success"]), "trials": len(nominal), "target": f">= {targets['nominal_min']} of {targets['nominal_of']}", "met": len(nominal) == targets["nominal_of"] and sum(1 for r in nominal if r["success"]) >= targets["nominal_min"]},
            "disturbed": {"successes": sum(1 for r in disturbed if r["success"]), "trials": len(disturbed), "target": f">= {targets['disturbed_min']} of {targets['disturbed_of']}", "met": len(disturbed) == targets["disturbed_of"] and sum(1 for r in disturbed if r["success"]) >= targets["disturbed_min"],
                          "by_stage": {s: f"{v['successes']}/{v['trials']}" for s, v in stages.items() if s != "nominal"}, "explicit_failures": sum(1 for r in disturbed if not r["success"] and r["reason"])},
            "failure_reasons": reasons, "provenance": sorted({r["actuation"]["provenance"] for r in rows}), "controller": sorted({r["actuation"]["controller"] for r in rows}), "root_writes": sorted({r["actuation"]["root_writes"] for r in rows}), "object_writes": sorted({r["actuation"]["object_writes"] for r in rows}),
            "artificial_support": sorted({r["actuation"]["artificial_support"] for r in rows}), "invariant_violations_total": sum(r["invariant_violation_count"] for r in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--body", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--kinds", default="nominal,disturbed")
    parser.add_argument("--limit", type=int, default=None, help="only the first N trials of each kind (engineering smoke; never a scored run)")
    parser.add_argument("--resume", action="store_true")
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
            kept[row["stage"]] = kept.get(row["stage"], 0) + 1
    (out / "provenance.json").write_bytes(json_bytes(provenance))
    for trial in trials:
        if trial["trial_id"] in done:
            continue
        disturbance = disturbance_of(trial["disturbance"])
        started = time.perf_counter()
        result = run_retrieve(body, course, seed=trial["seed"], cap_s=trial["cap_s"], jitter_xy_m=protocol["jitter"]["xy_m"], jitter_yaw_deg=protocol["jitter"]["yaw_deg"], disturbance=disturbance, settle_s=protocol["settle_s"], retry_budget=protocol["retry_budget"])
        row = row_of(trial, result, time.perf_counter() - started)
        keep = (not result.success) or kept.get(trial["stage"], 0) < KEEP_SUCCESSES
        if keep:
            if result.success:
                kept[trial["stage"]] = kept.get(trial["stage"], 0) + 1
            world_xml = course_world_xml(body, course, None)
            model = mujoco.MjSpec.from_string(world_xml).compile()
            on_course = replace(body, floor_model=model, floor_xml=world_xml)
            status = "success" if result.success else ("fell" if result.fell else "failed")
            caption = f"{trial['stage']} trial {trial['trial_id']}: {trial['disturbance']['detail'] if trial['disturbance'] else 'retrieve: start -> station -> tray -> start'}"
            sealed = seal_run(local / trial["trial_id"] / "physical", body=on_course, manifest=manifest, run=result.run, label=trial["trial_id"], caption=caption[:150], goal="G18", protocol=TRIAL_PROTOCOL,
                              test={"test": "retrieve_trial", "trial_id": trial["trial_id"], "kind": trial["kind"], "stage": trial["stage"], "seed": trial["seed"], "cap_s": trial["cap_s"], "disturbance": trial["disturbance"], "course_id": course.course_id, "course_sha256": course.sha256(),
                                    "registration_sha256": registration["registration_sha256"], "controller": {"name": result.actuation["controller"], "provenance": result.actuation["provenance"]}},
                              outcome={"status": status, "success": result.success, "reason": result.reason, "fell": result.fell, "fall_time_s": result.fall_time_s, "cube_in_tray": result.cube_in_tray, "cube_final": result.cube_final, "at_start": result.at_start, "stable_at_end": result.stable_at_end,
                                       "phases": row["phases"], "schedule": result.schedule, "hold_events": result.hold_events, "invariant_violation_count": len(result.invariant_violations), "collisions": result.collisions, "energy_j": result.energy_j, "distance_m": result.distance_m, "duration_s": result.duration_s,
                                       "disturbance_log": result.disturbance_log, "support": result.support},
                              camera=CAMERA, world_xml=world_xml, model=model, controls_note=controls_note(body.declaration["base_kind"], disturbance))
            row["sealed"] = {**sealed, "bundle": Path(sealed["bundle"]).relative_to(REPO).as_posix() if Path(sealed["bundle"]).is_absolute() else sealed["bundle"]}
        rows.append(row)
        (out / "trials.json").write_bytes(json_bytes(rows))
        print(json.dumps({"trial": trial["trial_id"], "success": result.success, "reason": result.reason[:80], "t": round(result.duration_s, 1), "phases": row["phases_completed"], "slip": row["max_object_slip_m"], "wall": round(row["wall_seconds"], 1), "kept": keep}), flush=True)
    summary = summarise(rows, protocol, args.body)
    (out / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps({"body": args.body, "nominal": summary["nominal"], "disturbed": summary["disturbed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
