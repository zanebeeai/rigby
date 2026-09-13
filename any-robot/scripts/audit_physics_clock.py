"""Reproduce the old clock error and measure the repaired public-zoo pipeline.

This is a diagnostic prerequisite to G05. It does not substitute for the
structural-profile integration, predeclared variant campaign or D05 videos.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
from rigby_general.bake.enumerate import build_candidate
from rigby_general.bake.runner import _attempt
from rigby_general.config import base_tree_fingerprint
from rigby_general.gates.certify import simulate
from rigby_general.pipeline import ingest_robot
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.primitives import PrimitiveRecord
from rigby_general.run import answer
from rigby_general.schema.inventory import afforded_entries, load_inventory


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BASELINE_COMMIT = "cfbe16e90544fc22b3acc9d95ce7b2d9cc6a24ba"
PROMPT = "reach out as far as you can and then come back"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main(destination):
    if destination.exists():
        raise ValueError("Choose a fresh diagnostic destination")
    destination.mkdir(parents=True)
    old_source = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:any-robot/src/rigby_general/gates/certify.py"], cwd=REPO)
    (destination / "certify-before.py").write_bytes(old_source)
    before = load_module(destination / "certify-before.py", "rigby_general.gates.clock_baseline")
    helpers = load_module(ROOT / "tests/test_certification_clock.py", "physics_clock_fixtures")
    probes = []
    for dt in (.002, 1/240, .003):
        for label, function in (("before", before.simulate), ("after", simulate)):
            model, manifest = helpers.fixture(dt)
            reference = helpers.linear_reference(.2)
            physics_times = [0.]
            step = mujoco.mj_step
            def observe_step(m, d):
                step(m, d)
                physics_times.append(float(d.time))
            mujoco.mj_step = observe_step
            try:
                trace = function(model, manifest, reference, site_name="tip")
            finally:
                mujoco.mj_step = step
            assert len(physics_times) == len(trace.times_s)
            prefix = f"{label}-{round(1/dt)}hz"
            np.savez_compressed(destination / f"{prefix}.npz", physical_times_s=physics_times,
                reported_times_s=trace.times_s, qpos=trace.qpos, qvel=trace.qvel, ctrl=trace.ctrl,
                tracking_error_m=trace.tracking_error_m)
            value = {"runner": label, "timestep_s": dt, "reference_duration_s": .2,
                "physics_steps": len(physics_times)-1, "actual_duration_s": physics_times[-1],
                "reported_duration_s": float(trace.times_s[-1]),
                "max_clock_disagreement_s": float(np.max(np.abs(trace.times_s - physics_times))),
                "max_tracking_error_m": float(trace.tracking_error_m.max())}
            if label == "after":
                assert value["max_clock_disagreement_s"] == 0
            probes.append(value)
    write_json(destination / "clock-probes.json", probes)
    inventory = load_inventory()
    fingerprint = base_tree_fingerprint().sha256
    bodies = []
    for source in sorted((ROOT / "assets/general/zoo").glob("*/robot.urdf")):
        print(f"Fresh exact-region bakes and composition: {source.parent.name}", flush=True)
        started = time.perf_counter()
        robot = ingest_robot(source)
        manifest, model = robot.manifest, robot.finalized.model
        planner = OfflineSchemaPlanner(inventory)
        program = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
        records, failures = [], []
        for segment in program.segments:
            entry = inventory.by_binding_key(f"{segment.motion_schema.canonical_key}|{segment.figure.role.value}->{segment.ground.role.value}")
            outcome = _attempt(build_candidate(entry, segment.region.remove), manifest, model, inventory,
                               policy=None, fingerprint=fingerprint)
            (records if isinstance(outcome, PrimitiveRecord) else failures).append(outcome)
        result = answer(PROMPT, manifest, model, inventory, tuple(records), tuple(failures), planner=planner)
        folder = destination / source.parent.name
        folder.mkdir()
        write_json(folder / "primitive-records.json", [r.to_json() for r in records])
        write_json(folder / "primitive-failures.json", [r.to_json() for r in failures])
        summary = result.summary()
        summary.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       intake_profile="legacy; structural profile integration remains pending",
                       fresh_exact_region_bakes=len(records), bake_failures=len(failures),
                       duration_scales=[r.measurements["duration_scale"] for r in records],
                       total_wall_seconds=time.perf_counter()-started)
        if result.certification:
            c = result.certification
            t = c.trace
            np.savez_compressed(folder / "rollout.npz", time_s=t.times_s, qpos=t.qpos, qvel=t.qvel,
                                ctrl=t.ctrl, demand=t.demand, tracking_error_m=t.tracking_error_m)
            summary.update(actual_physics_duration_s=float(t.times_s[-1]), native_timestep_s=float(model.opt.timestep),
                repeats=c.repeats, replay_hashes=c.replay_hashes, violations=[asdict(v) for v in c.violations])
        write_json(folder / "summary.json", summary)
        bodies.append(summary)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    report = {"schema": "g05.clock_diagnostic.v1", "source_commit": commit, "baseline_commit": BASELINE_COMMIT,
        "mujoco_version": mujoco.__version__, "prompt": PROMPT, "bodies": bodies,
        "canonical_successes": sum(b["accepted"] for b in bodies), "goal_G05_complete": False,
        "clock_probes": probes, "api_calls": 0,
        "limitations": ["Legacy-profile diagnostic; G01/G03 integration and G05's predeclared 120 variants and D05 media remain pending.",
            "No reference duration was changed by the clock repair; comparison to previously mislabeled physical durations needs explicit treatment in G05."]}
    write_json(destination / "report.json", report)
    write_json(destination / "payloads.json", {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sorted(destination.rglob("*")) if p.is_file()})
    print(json.dumps({"source_commit": commit, "canonical_successes": report["canonical_successes"], "G05_complete": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    main(parser.parse_args().destination)
