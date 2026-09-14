"""Freeze the common course and the goal set, judge every body's feasibility on it from its measured manifest, and register both with the expert baseline.

    python any-robot/scripts/g16_course.py --bodies docs/results/g16-bodies --out any-robot/assets/general/mobile/course-v1 --results docs/results/g16-course --local any-robot/results/g16-course

The course is one world for the three bodies, frozen here before any
locomotion controller exists and independent of which one later works.
The feasibility map reads each body's manifest and settling measurements
against the course's dimensions; the expert baseline is labelled by hand
and kept in its own file with its own hash, so the report can say where
the analysis and the expert agree and where they do not. Each body is
also compiled into the course and settled on the start pad, sealed, as
the proof that the world is one it can stand in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from rigby_general.evidence.capture import json_bytes
from rigby_general.mobility import load_mobile_body
from rigby_general.mobility.contracts import MobileBodyManifestV1, StanceMeasurementV1
from rigby_general.mobility.evidence import seal_run
from rigby_general.mobility.validate import run_held
from rigby_general.mobility.world import course_v1, feasibility


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MOBILE = ROOT / "assets/general/mobile"
BODIES = ("mobile_dog_arm", "mobile_wheeled_biped", "mobile_octopus")
GEOMETRY = {
    "mobile_dog_arm": {"arm_base_height_above_base_m": 0.10, "wheel_radius_m": None, "leg_length_m": 0.41, "mantle_rise_m": None},
    "mobile_wheeled_biped": {"arm_base_height_above_base_m": 0.18, "wheel_radius_m": 0.08, "leg_length_m": None, "mantle_rise_m": None},
    "mobile_octopus": {"arm_base_height_above_base_m": -0.06, "wheel_radius_m": None, "leg_length_m": None, "mantle_rise_m": 0.017},
}
"""Read off the models: where each body's manipulator hangs from relative to its base origin (the arm base's z offset on the torso, the tentacle roots' on the mantle), its wheel radius, its leg length (thigh + shank), or how
far its mantle's underside sits above the floor at rest (0.127 m base height less the 0.11 m half height of the ellipsoid)."""

EXPERT_LABELS = {
    "schema": "rigby.mobile-feasibility-expert-labels/1",
    "labeller": "internal (the author of the bodies and the course); not an independent reviewer",
    "labels": {
        "mobile_dog_arm": {"corridor": True, "doorway": True, "ramp": True, "step": True, "station": True, "tray": True, "travel": True, "retrieve": True,
                           "why": "0.6 m across, fits both openings; rubber feet on a 6 degree ramp; a 5 cm step is within a leg's stride; the arm hangs from 0.48 m and reaches 0.44 m, so 0.09 m and 0.07 m are inside its band"},
        "mobile_wheeled_biped": {"corridor": True, "doorway": False, "ramp": True, "step": False, "station": True, "tray": True, "travel": True, "retrieve": True,
                                 "why": "0.67 m across the wheels: the corridor fits, a 0.7 m doorway leaves 1.5 cm a side and a balancing body cannot hold that line; rubber wheels hold a 6 degree ramp; an 8 cm wheel cannot climb a 5 cm step; the arm hangs from the torso top and its 0.34 m reach spans the station and tray with the knees flexed"},
        "mobile_octopus": {"corridor": True, "doorway": False, "ramp": True, "step": False, "station": True, "tray": True, "travel": True, "retrieve": True,
                           "why": "1.2 m across at rest: the corridor fits, the 0.7 m doorway does not; the tentacles' friction holds the ramp; a 5 cm step is above its mantle's rise; the pincer tentacles reach 0.69 m along the floor and up to the station and the tray floor"},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bodies", type=Path, required=True, help="the g16_validate output (manifests and tests)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    args = parser.parse_args()
    course = course_v1()
    args.out.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.out / "course.json").write_bytes(json_bytes(course.as_json()))
    (args.out / "expert-labels.json").write_bytes(json_bytes(EXPERT_LABELS))
    feasibility_map = {"schema": "rigby.mobile-feasibility-map/1", "course_id": course.course_id, "course_sha256": course.sha256(), "bodies": {}}
    agreement = {}
    for body_id in BODIES:
        manifest = MobileBodyManifestV1.model_validate_json((args.bodies / body_id / "manifest.json").read_bytes())
        tests = json.loads((args.bodies / body_id / "tests.json").read_bytes())
        stances = tuple(StanceMeasurementV1.model_validate({k: v for k, v in s.items() if k not in ("status", "sealed")}) for s in tests["stances"])
        verdicts = feasibility(course, manifest, stances, **GEOMETRY[body_id])
        feasibility_map["bodies"][body_id] = {**verdicts, "geometry": GEOMETRY[body_id]}
        labels = EXPERT_LABELS["labels"][body_id]
        agreement[body_id] = {v["branch_id"]: {"analysis": v["feasible"], "expert": labels[v["branch_id"]], "agree": v["feasible"] == labels[v["branch_id"]]} for v in verdicts["verdicts"]}
        print(json.dumps({"body": body_id, "verdicts": {v["branch_id"]: v["feasible"] for v in verdicts["verdicts"]}, "main_route": verdicts["all_main_route_feasible"]}), flush=True)
    feasibility_map["agreement"] = agreement
    (args.out / "feasibility.json").write_bytes(json_bytes(feasibility_map))
    files = {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest() for name in ("course.json", "feasibility.json", "expert-labels.json")}
    registration = {"schema": "g16.course-registration.v1", "course_id": course.course_id, "course_sha256": course.sha256(), "files": files, "bodies": list(BODIES),
                    "frozen_before": "any locomotion controller (G17) or loco-manipulation policy (G18); the course does not depend on which controller later succeeds",
                    "registered_at_utc": datetime.now(timezone.utc).isoformat()}
    registration["registration_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in registration.items() if k != "registered_at_utc"}, sort_keys=True).encode("utf-8")).hexdigest()
    (args.out / "registration.json").write_bytes(json_bytes(registration))
    # each body compiled into the course and settled on the start pad
    provenance = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "platform": platform.platform(), "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__,
                  "registration_sha256": registration["registration_sha256"], "started_at_utc": datetime.now(timezone.utc).isoformat(), "generation_calls": 0}
    settled = {}
    for body_id in BODIES:
        body = load_mobile_body(MOBILE / body_id)
        manifest = MobileBodyManifestV1.model_validate_json((args.bodies / body_id / "manifest.json").read_bytes())
        world_xml = body.floor_xml.replace('<light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>', '<light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>\n    ' + course.mjcf_features(), 1)
        model = mujoco.MjSpec.from_string(world_xml).compile()
        holdable = next(n for n, s in body.declaration["stances"].items() if s["statically_stable"]) if not body.declaration["stances"][body.declaration["working_stance"]]["statically_stable"] else body.declaration["working_stance"]
        from dataclasses import replace

        on_course = replace(body, floor_model=model, floor_xml=world_xml)
        run = run_held(on_course, body.declaration["stances"][holdable]["joints"], duration_s=3.0)
        sealed = seal_run(args.local / body_id / "start-pad" / "physical", body=on_course, manifest=manifest, run=run, label=f"{body_id}-course-start", caption=f"on the course: settled on the start pad in stance '{holdable}'",
                          test={"test": "course_start", "stance": holdable, "course_id": course.course_id, "course_sha256": course.sha256()}, outcome={"status": "stable" if run.settled and run.upright else "unstable", "tilt_deg": run.tilt_deg, "height_m": run.base_height_m, "contacts": run.contacts_at_end},
                          world_xml=world_xml, model=model, camera={"centre": [1.2, 0.6, 0.2], "reach": 2.2})
        settled[body_id] = {"stance": holdable, "stable": bool(run.settled and run.upright), "tilt_deg": run.tilt_deg, "height_m": run.base_height_m, "contacts": run.contacts_at_end, "sealed": sealed, "course_model_sha256": hashlib.sha256(world_xml.encode("utf-8")).hexdigest()}
        print(json.dumps({"body": body_id, "course_start": settled[body_id]["stable"], "contacts": run.contacts_at_end}), flush=True)
    (args.results / "course-settle.json").write_bytes(json_bytes({"provenance": provenance, "bodies": settled}))
    print(json.dumps({"registration_sha256": registration["registration_sha256"], "agreement": {b: sum(1 for v in a.values() if v["agree"]) for b, a in agreement.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
