"""Execute the registered G05 composition roster and record every outcome.

One invocation, one attempt per trial, in roster order. For every body: the
canonical trial is captured as a full G01-style evidence bundle (physics,
replay, full-duration video); the twenty predeclared feasible variants and the
predeclared invalid requests are run through the same fresh-bake-then-answer
pipeline with the certification trace hashed and saved locally. Nothing is
retried, re-drawn or removed. No API or model calls.

    python any-robot/scripts/g05_composition_campaign.py --out docs/results/g05-campaign \
        --local any-robot/results/g05-campaign [--render] [--bodies zoo_x,zoo_y]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_general.capabilities.intake import ingest_capability_body
from rigby_general.evidence.capture import replay_bundle
from rigby_general.evidence.composition import capture_prompt, run_prompt
from rigby_general.evidence.render import render_bundle


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ROSTER_DIR = ROOT / "assets/general/research-protocols/g05-composition-v1"


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def load_roster() -> tuple[dict, dict]:
    roster_bytes = (ROSTER_DIR / "roster.json").read_bytes()
    registration = json.loads((ROSTER_DIR / "registration.json").read_bytes())
    digest = hashlib.sha256(roster_bytes).hexdigest()
    if digest != registration["roster_sha256"]:
        raise SystemExit("The roster does not match its registration; refusing to run an unregistered campaign")
    return json.loads(roster_bytes), registration


def trace_arrays(run) -> dict[str, np.ndarray]:
    trace = run.certification.trace
    return {
        "time_s": trace.times_s, "qpos": trace.qpos, "qvel": trace.qvel,
        "ctrl": trace.ctrl, "demand": trace.demand, "tracking_error_m": trace.tracking_error_m,
    }


def summarize(runs, records, failures, program, prompt, wall_s: float) -> dict:
    run = runs[0]
    row = {
        "prompt": prompt, "accepted": run.accepted, "wall_seconds": round(wall_s, 3),
        "failure": {"stage": run.failure_stage, "code": run.failure_code, "detail": run.failure_detail} if not run.accepted else None,
        "requested_semantic_hash": program.role_normalized_hash() if program is not None else None,
        "fresh_leaves_certified": len(records), "leaf_failures": [
            {"entry": f.entry_id, "stage": f.stage.value if hasattr(f.stage, "value") else str(f.stage),
             "code": f.failure_code, "gate": f.failed_gate, "detail": f.detail[:300], "measurements": f.measurements}
            for f in failures
        ],
        "leaf_duration_scales": [float(r.measurements.get("duration_scale", 1.0)) for r in records],
        "repeat_outcomes_agree": len({(r.accepted, r.failure_code, r.failure_stage) for r in runs}) == 1,
        "stages_ms": {item["name"]: round(float(item.get("elapsed_ms", 0.0)), 3) for item in run.trace.to_json().get("stages", [])
                      if isinstance(item, dict) and "name" in item},
    }
    if run.bound is not None:
        row["region_substitutions"] = run.bound.substitutions
        row["authored_duration_s"] = run.duration_s
        row["grounded_program_sha256"] = run.bound.grounded.program.content_hash()
        row["path_repairs"] = list(run.bound.grounded.program.metadata.get("path_repairs", []))
        row["leaf_path_repairs"] = [int(r.measurements.get("path_repairs", 0)) for r in records]
    if run.certification is not None:
        trace = run.certification.trace
        row.update({
            "certified": run.certification.certified,
            "replay_hashes": list(run.certification.replay_hashes),
            "replay_agreement": len(set(run.certification.replay_hashes)) == 1,
            "actual_physics_duration_s": float(trace.times_s[-1]),
            "tracking_error_m": float(trace.tracking_error_m.max()),
            "base_drift_m": float(trace.base_drift_m),
            "unexpected_contacts": [list(p) for p in trace.unexpected_contacts],
            "violations": [{"code": v.code.value, "subject": v.subject, "measured": v.measured, "limit": v.limit, "detail": v.detail}
                           for v in run.certification.violations],
            "peak_demand_fraction": float(np.max(np.abs(trace.demand) / np.maximum(1e-9, np.abs(run.certification.trace.demand).max()))) if trace.demand.size else None,
        })
    return row


def run_trial(robot, prompt: str, start: np.ndarray | None) -> tuple[dict, object]:
    started = time.perf_counter()
    runs, records, failures, program = run_prompt(robot, prompt, start_qpos=start)
    wall = time.perf_counter() - started
    return summarize(runs, records, failures, program, prompt, wall), runs[0]


def save_trace(local: Path, run) -> str | None:
    if run.certification is None:
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    np.savez_compressed(stream, **trace_arrays(run))
    local.write_bytes(stream.getvalue())
    return hashlib.sha256(local.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="tracked evidence directory (summaries, bundles, media)")
    parser.add_argument("--local", type=Path, required=True, help="untracked directory for full variant traces")
    parser.add_argument("--render", action="store_true", help="render the canonical bundles to full video")
    parser.add_argument("--bodies", help="comma-separated zoo ids; default every roster body")
    args = parser.parse_args()
    roster, registration = load_roster()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; a campaign writes only to a fresh destination")
    args.out.mkdir(parents=True)
    args.local.mkdir(parents=True, exist_ok=True)
    wanted = set(args.bodies.split(",")) if args.bodies else None
    provenance = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
        "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__,
        "roster_sha256": registration["roster_sha256"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_calls": 0,
    }
    (args.out / "provenance.json").write_bytes(json_bytes(provenance))
    bodies_out = []
    for body in roster["bodies"]:
        zoo_id = body["zoo_id"]
        if wanted and zoo_id not in wanted:
            continue
        source = REPO / body["source_urdf"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != body["source_sha256"]:
            raise SystemExit(f"{zoo_id}: source URDF changed since registration")
        capability = ingest_capability_body(source)
        robot = capability.robot
        if capability.manifest.package_sha256 != body["package_sha256"] or robot.manifest.rig_id != body["rig_id"]:
            raise SystemExit(f"{zoo_id}: structural intake no longer matches the registered package")
        starts = {s["start_id"]: np.asarray(s["qpos"], dtype=float) for s in body["starts"]}
        out = args.out / zoo_id
        out.mkdir()
        print(json.dumps({"body": zoo_id, "rig_id": robot.manifest.rig_id, "trials": len(body["feasible_trials"])}), flush=True)

        canonical = capture_prompt(
            robot, roster["canonical_prompt"], out / "canonical" / "physical",
            label=zoo_id, source_urdf=source, start_qpos=starts["start_0"], goal="G05",
            protocol_reference={"roster_sha256": registration["roster_sha256"], "trial_id": "pace+0_start_0"},
            extra_payloads={"capability-manifest.json": json_bytes(capability.manifest.model_dump(mode="json"))},
            caption=f"{zoo_id} | reach/return | structural intake | canonical (pace 0, measured rest)",
        )
        canonical["replay"] = replay_bundle(out / "canonical" / "physical", canonical["sha256"])
        if not canonical["replay"]["agrees"]:
            raise SystemExit(f"{zoo_id}: canonical physical replay disagrees")
        if args.render:
            canonical["media"] = render_bundle(out / "canonical" / "physical", out / "canonical" / "media", expected_digest=canonical["sha256"])
        print(json.dumps({"body": zoo_id, "canonical": canonical["outcome"], "physics_s": canonical["simulation_duration_s"], "reference_s": canonical["reference_duration_s"]}), flush=True)

        trials = []
        for trial in body["feasible_trials"]:
            row, run = run_trial(robot, trial["prompt"], starts[trial["start_id"]])
            row.update({"trial_id": trial["trial_id"], "pace_ordinal": trial["pace_ordinal"], "start_id": trial["start_id"], "kind": "feasible"})
            row["trace_file_sha256"] = save_trace(args.local / zoo_id / f"{trial['trial_id']}.npz", run)
            trials.append(row)
            print(json.dumps({"body": zoo_id, "trial": trial["trial_id"], "accepted": row["accepted"],
                              "duration_s": row.get("authored_duration_s"), "failure": row["failure"]}), flush=True)
        invalid = []
        for case in body["invalid_requests"]:
            if case["prompt"] is None:
                invalid.append({"case_id": case["case_id"], "kind": "invalid", "not_constructible": True, "reason": case["reason"]})
                continue
            row, run = run_trial(robot, case["prompt"], np.asarray(case["start_qpos"], dtype=float))
            expected = case["expected"]
            observed_codes = {row["failure"]["code"]} if row["failure"] else set()
            observed_codes |= {f["code"] for f in row["leaf_failures"]}
            observed_measurements = {f["gate"] for f in row["leaf_failures"] if f["gate"]}
            correct = (not row["accepted"]) and expected["refusal_code"] in observed_codes and (
                expected["measurement"] is None or expected["measurement"] in observed_measurements
            )
            row.update({"case_id": case["case_id"], "kind": "invalid", "reason": case["reason"], "expected": expected,
                        "observed_codes": sorted(observed_codes), "observed_measurements": sorted(observed_measurements),
                        "correct_typed_refusal": bool(correct)})
            invalid.append(row)
            print(json.dumps({"body": zoo_id, "invalid": case["case_id"], "accepted": row["accepted"], "correct": row["correct_typed_refusal"], "codes": sorted(observed_codes)}), flush=True)
        successes = sum(1 for t in trials if t["accepted"])
        repaired = sum(1 for t in trials if t["accepted"] and t.get("path_repairs"))
        body_summary = {
            "zoo_id": zoo_id, "rig_id": robot.manifest.rig_id, "package_sha256": capability.manifest.package_sha256,
            "canonical": canonical, "feasible_trials": trials, "invalid_requests": invalid,
            "feasible_successes": successes, "feasible_count": len(trials),
            "successes_with_path_repair": repaired, "successes_straight": successes - repaired,
            "meets_19_of_20": successes >= 19 and len(trials) >= 20,
            "invalid_correctly_refused": sum(1 for c in invalid if c.get("correct_typed_refusal")),
            "invalid_constructed": sum(1 for c in invalid if not c.get("not_constructible")),
        }
        (out / "trials.json").write_bytes(json_bytes(body_summary))
        bodies_out.append({k: body_summary[k] for k in ("zoo_id", "rig_id", "feasible_successes", "feasible_count", "successes_with_path_repair", "successes_straight", "meets_19_of_20", "invalid_correctly_refused", "invalid_constructed")}
                          | {"canonical_outcome": canonical["outcome"], "canonical_reference_s": canonical["reference_duration_s"], "canonical_physics_s": canonical["simulation_duration_s"], "canonical_sha256": canonical["sha256"]})
    summary = {
        "goal": "G05", "roster_sha256": registration["roster_sha256"], "provenance": provenance,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "bodies": bodies_out,
        "all_bodies_meet_19_of_20": all(b["meets_19_of_20"] for b in bodies_out) and len(bodies_out) == len(roster["bodies"]),
        "all_canonical_success": all(b["canonical_outcome"] == "success" for b in bodies_out) and len(bodies_out) == len(roster["bodies"]),
    }
    (args.out / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps({k: summary[k] for k in ("all_bodies_meet_19_of_20", "all_canonical_success")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
