"""Validate the three mobile bodies: integrity, manifests, every declared stance settled on physics, recovery trials, an inspection sweep; every run sealed.

    python any-robot/scripts/g16_validate.py --out docs/results/g16-bodies --local any-robot/results/g16-bodies

Per body: the integrity report (the rules a floating base can meet), the
measured manifest, a settling test per declared stance (does it come to
rest upright on the members its author said would bear it?), a fixed
list of recovery trials in the working stance -- rolled, pitched and
dropped, supported and unsupported alike, the unsupported ones being the
manoeuvres the body is not meant to survive without a controller -- and
an inspection sweep of every joint. Every run is a sealed, replayable
bundle; nothing acts on the body but its own position servos.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_general.evidence.capture import json_bytes
from rigby_general.mobility import check_mobile_integrity, inspection_sweep, load_mobile_body, measure_mobile_body, measure_stance, recovery_trial
from rigby_general.mobility.contracts import MobileIntegrityReportV1
from rigby_general.mobility.evidence import seal_run


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MOBILE = ROOT / "assets/general/mobile"
BODIES = ("mobile_dog_arm", "mobile_wheeled_biped", "mobile_octopus")
SETTLE_S = 4.0
RECOVERY_S = 8.0


def recovery_plan(body_id: str, declaration: dict) -> list[dict]:
    """The trials every body runs in its working stance (the biped also in its parked stance, which is the one it holds without a controller), and the manoeuvres declared unsupported."""

    working = declaration["working_stance"]
    holdable = working if declaration["stances"][working]["statically_stable"] else next(n for n, s in declaration["stances"].items() if s["statically_stable"])
    trials = [
        {"trial_id": "drop_10cm", "stance": holdable, "perturbation": "released 10 cm higher than its stance", "roll_deg": 0.0, "pitch_deg": 0.0, "drop_m": 0.10, "supported": True},
        {"trial_id": "roll_8", "stance": holdable, "perturbation": "released rolled 8 degrees", "roll_deg": 8.0, "pitch_deg": 0.0, "drop_m": 0.03, "supported": True},
        {"trial_id": "roll_-8", "stance": holdable, "perturbation": "released rolled -8 degrees", "roll_deg": -8.0, "pitch_deg": 0.0, "drop_m": 0.03, "supported": True},
        {"trial_id": "pitch_8", "stance": holdable, "perturbation": "released pitched 8 degrees nose down", "roll_deg": 0.0, "pitch_deg": 8.0, "drop_m": 0.03, "supported": True},
        {"trial_id": "pitch_-8", "stance": holdable, "perturbation": "released pitched 8 degrees nose up", "roll_deg": 0.0, "pitch_deg": -8.0, "drop_m": 0.03, "supported": True},
        {"trial_id": "roll_25", "stance": holdable, "perturbation": "released rolled 25 degrees", "roll_deg": 25.0, "pitch_deg": 0.0, "drop_m": 0.03, "supported": True},
        {"trial_id": "pitch_25", "stance": holdable, "perturbation": "released pitched 25 degrees nose down", "roll_deg": 0.0, "pitch_deg": 25.0, "drop_m": 0.03, "supported": True},
        {"trial_id": "roll_60", "stance": holdable, "perturbation": "released rolled 60 degrees: an unsupported manoeuvre, kept to show what the body does without a controller", "roll_deg": 60.0, "pitch_deg": 0.0, "drop_m": 0.05, "supported": False},
    ]
    if not declaration["stances"][working]["statically_stable"]:
        trials.insert(0, {"trial_id": "released_standing", "stance": working, "perturbation": "released in its working stance with no balance controller: declared unsupported", "roll_deg": 0.0, "pitch_deg": 0.0, "drop_m": 0.02, "supported": False})
    if body_id == "mobile_dog_arm":
        trials.append({"trial_id": "rearing", "stance": "rearing", "perturbation": "released rearing on its hind legs: an unsupported manoeuvre", "roll_deg": 0.0, "pitch_deg": -35.0, "drop_m": 0.05, "supported": False})
    if body_id == "mobile_octopus":
        trials.append({"trial_id": "upside_down", "stance": working, "perturbation": "released upside down: an unsupported manoeuvre, it has no righting reflex", "roll_deg": 180.0, "pitch_deg": 0.0, "drop_m": 0.10, "supported": False})
    return trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--bodies", default=",".join(BODIES))
    args = parser.parse_args()
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                  "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__, "started_at_utc": datetime.now(timezone.utc).isoformat(), "generation_calls": 0}
    summary = {}
    for body_id in args.bodies.split(","):
        body = load_mobile_body(MOBILE / body_id)
        declaration = body.declaration
        # a manoeuvre the dog does not declare, for the unsupported rearing trial: hind legs folded, fore legs stretched; the declaration itself is untouched
        undeclared_joints = {"rearing": {**declaration["stances"]["standing"]["joints"], "fl_hip_pitch": -0.9, "fr_hip_pitch": -0.9, "fl_knee": -0.3, "fr_knee": -0.3}} if body_id == "mobile_dog_arm" else {}
        report = check_mobile_integrity(body)
        manifest = measure_mobile_body(body)
        out = args.out / body_id
        local = args.local / body_id
        out.mkdir(parents=True, exist_ok=True)
        local.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_bytes(json_bytes(manifest.model_dump(mode="json")))
        (out / "provenance.json").write_bytes(json_bytes({**provenance, "body": body_id, "model_sha256": body.model_sha256}))
        tests = {"stances": [], "recoveries": [], "inspection": None}
        stances = []
        for stance in declaration["stances"]:
            measurement, run = measure_stance(body, stance, duration_s=SETTLE_S)
            stances.append(measurement)
            status = "stable" if measurement.statically_stable else ("unstable_as_declared" if not measurement.declared_statically_stable else "unstable")
            sealed = seal_run(local / f"stance-{stance}" / "physical", body=body, manifest=manifest, run=run, label=f"{body_id}-stance-{stance}",
                              caption=f"settling test: stance '{stance}' | declared {'stable' if measurement.declared_statically_stable else 'unstable without a controller'} | measured {status.replace('_', ' ')}",
                              test={"test": "stance", "stance": stance, "declared": declaration["stances"][stance]}, outcome={"status": status, "measurement": measurement.model_dump(mode="json")})
            tests["stances"].append({**measurement.model_dump(mode="json"), "status": status, "sealed": sealed})
            print(json.dumps({"body": body_id, "stance": stance, "status": status, "height": round(measurement.base_height_m, 3), "tilt": round(measurement.tilt_deg, 1), "contacts": measurement.support_contacts}), flush=True)
        recoveries = []
        measured_height = {s.stance: s.base_height_m for s in stances if s.statically_stable}
        for plan in recovery_plan(body_id, declaration):
            trial, run = recovery_trial(body, plan["stance"] if plan["stance"] in declaration["stances"] else declaration["working_stance"], plan["trial_id"], perturbation=plan["perturbation"], roll_deg=plan["roll_deg"], pitch_deg=plan["pitch_deg"],
                                        drop_m=plan["drop_m"], supported=plan["supported"], duration_s=RECOVERY_S, expected_height_m=measured_height.get(plan["stance"], measured_height.get(declaration["working_stance"])),
                                        joints=undeclared_joints.get(plan["stance"]))
            recoveries.append(trial)
            status = "recovered" if trial.recovered else ("not_recovered_unsupported" if not plan["supported"] else "not_recovered")
            sealed = seal_run(local / f"recovery-{plan['trial_id']}" / "physical", body=body, manifest=manifest, run=run, label=f"{body_id}-recovery-{plan['trial_id']}",
                              caption=f"recovery trial: {plan['perturbation']} | {'SUPPORTED' if plan['supported'] else 'UNSUPPORTED'} | {status.replace('_', ' ')}",
                              test={"test": "recovery", "trial": plan}, outcome={"status": status, "trial": trial.model_dump(mode="json")})
            tests["recoveries"].append({**trial.model_dump(mode="json"), "stance": plan["stance"], "status": status, "sealed": sealed})
            print(json.dumps({"body": body_id, "trial": plan["trial_id"], "supported": plan["supported"], "recovered": trial.recovered, "tilt": round(trial.final_tilt_deg, 1), "height": round(trial.final_height_m, 3)}), flush=True)
        holdable = next(s["stance"] for s in tests["stances"] if s["statically_stable"])
        # a leg that bears the body is swept less far than an arm that does not; a parked wheeled body's legs least of all, since a knee folding a wheel off the ground tips it
        fractions = {"leg": 0.15 if declaration["base_kind"] == "wheeled" else 0.35, "arm": 0.5, "grip": 0.5, "tentacle": 0.4}
        run, schedule = inspection_sweep(body, holdable, per_joint_s=1.2 if body_id != "mobile_octopus" else 0.6, fraction_by_role=fractions)
        sealed = seal_run(local / "inspection" / "physical", body=body, manifest=manifest, run=run, label=f"{body_id}-inspection",
                          caption=f"physical inspection: every joint swept from its rest in stance '{holdable}', the rest of the body holding",
                          test={"test": "inspection", "stance": holdable, "schedule": schedule}, outcome={"status": "stable" if run.settled and run.upright else "disturbed", "tilt_deg": run.tilt_deg, "height_m": run.base_height_m, "contacts_at_end": run.contacts_at_end})
        tests["inspection"] = {"stance": holdable, "schedule": schedule, "duration_s": float(run.record.arrays["time_s"][-1]), "tilt_deg": run.tilt_deg, "settled": run.settled, "upright": run.upright, "sealed": sealed}
        full = MobileIntegrityReportV1(**{**report.model_dump(mode="json"), "stances": [s.model_dump(mode="json") for s in stances], "recoveries": [r.model_dump(mode="json") for r in recoveries]})
        (out / "integrity.json").write_bytes(json_bytes(full.model_dump(mode="json")))
        (out / "tests.json").write_bytes(json_bytes(tests))
        summary[body_id] = {"integrity": report.passed, "violations": list(report.violations), "mass_kg": round(report.mass_kg, 3), "joints": report.joint_count, "actuators": report.actuator_count,
                            "stances": {s["stance"]: s["status"] for s in tests["stances"]},
                            "recoveries": {"supported": [r["trial_id"] for r in tests["recoveries"] if r["supported"]], "supported_recovered": sum(1 for r in tests["recoveries"] if r["supported"] and r["recovered"]),
                                           "supported_total": sum(1 for r in tests["recoveries"] if r["supported"]),
                                           "unsupported": [r["trial_id"] for r in tests["recoveries"] if not r["supported"]], "unsupported_recovered": sum(1 for r in tests["recoveries"] if not r["supported"] and r["recovered"]),
                                           "unsupported_total": sum(1 for r in tests["recoveries"] if not r["supported"])},
                            "inspection_upright": run.upright}
        print(json.dumps({"body": body_id, **summary[body_id]}), flush=True)
    (args.out / "summary.json").write_bytes(json_bytes({"provenance": provenance, "bodies": summary, "finished_at_utc": datetime.now(timezone.utc).isoformat()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
