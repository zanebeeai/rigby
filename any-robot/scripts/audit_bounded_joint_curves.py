"""Public-zoo prerequisite probe; this is not the scored G05 campaign.

Loads the separately reviewed clock fix from its pinned Git source so physics
uses native time without stacking production branches. No API/model calls.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
from numpy.polynomial import Polynomial
from rigby_core.motion.ownership import build_joint_series
from rigby_general.bake.enumerate import build_candidate
from rigby_general.bake.runner import _attempt
from rigby_general.config import base_tree_fingerprint
from rigby_general.grounding import ground
from rigby_general.pipeline import ingest_robot
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.primitives import PrimitiveRecord
from rigby_general.run import answer
from rigby_general.schema.inventory import afforded_entries, load_inventory


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BASE = "cfbe16e90544fc22b3acc9d95ce7b2d9cc6a24ba"
CLOCK = "8d4d1c7cbeb0986e471dd1d743711d1b8e0d3119"
PROMPT = "reach out as far as you can and then come back"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n",
                    encoding="utf-8", newline="\n")


def pinned_module(commit, source, destination, name):
    payload = subprocess.check_output(["git", "show", f"{commit}:{source}"], cwd=REPO)
    destination.write_bytes(payload)
    spec = importlib.util.spec_from_file_location(name, destination)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def interval_extrema(program):
    rows = []
    for joint, tracks in build_joint_series(program.tracks, program.duration_s).items():
        for track in tracks:
            for i, segment in enumerate(track._segments):
                # Normalize the time axis before finding all stationary points.
                h = segment.duration_s
                poly = Polynomial([float(c)*h**j for j, c in enumerate(segment._coefficients)])
                roots = poly.deriv().roots()
                points = [0., 1.] + [float(r.real) for r in roots
                                    if abs(r.imag) < 1e-7 and 0 < r.real < 1]
                values = poly(points)
                low, high = sorted(track.values[i:i+2])
                rows.append({"joint": joint, "track": track.track.track_id, "interval": i,
                             "start_s": float(track.times[i]), "end_s": float(track.times[i+1]),
                             "key_min": float(low), "key_max": float(high),
                             "curve_min": float(values.min()), "curve_max": float(values.max()),
                             "endpoint_range_excess": float(max(0., low-values.min(), values.max()-high))})
    return rows


def main(destination):
    if destination.exists():
        raise ValueError("Choose a fresh destination")
    destination.mkdir(parents=True)
    old_grounder = pinned_module(BASE, "any-robot/src/rigby_general/grounding/grounder.py",
                                destination / "grounder-before.py", "rigby_general.grounding.curve_baseline")
    native = pinned_module(CLOCK, "any-robot/src/rigby_general/gates/certify.py",
                           destination / "certify-native.py", "rigby_general.gates.curve_native_clock")
    gates = importlib.import_module("rigby_general.gates.certify")
    original_simulate = gates.simulate
    gates.simulate = native.simulate
    inventory = load_inventory()
    fingerprint = base_tree_fingerprint().sha256
    bodies = []
    try:
        for source in sorted((ROOT / "assets/general/zoo").glob("*/robot.urdf")):
            print(f"Native physics + bounded curves: {source.parent.name}", flush=True)
            started = time.perf_counter()
            robot = ingest_robot(source)
            manifest, model = robot.manifest, robot.finalized.model
            planner = OfflineSchemaPlanner(inventory)
            schema = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
            before = old_grounder.ground(schema, manifest, model, inventory).program
            after = ground(schema, manifest, model, inventory).program
            assert before.duration_s == after.duration_s
            assert before.phases == after.phases
            assert [t.keyframes for t in before.tracks] == [t.keyframes for t in after.tracks]
            assert all(t.interpolation.value == "bounded_quintic" for t in after.tracks)
            before_extrema, after_extrema = interval_extrema(before), interval_extrema(after)
            assert max(r["endpoint_range_excess"] for r in after_extrema) < 1e-9
            records, failures = [], []
            for segment in schema.segments:
                key = f"{segment.motion_schema.canonical_key}|{segment.figure.role.value}->{segment.ground.role.value}"
                outcome = _attempt(build_candidate(inventory.by_binding_key(key), segment.region.remove),
                                   manifest, model, inventory, policy=None, fingerprint=fingerprint)
                (records if isinstance(outcome, PrimitiveRecord) else failures).append(outcome)
            result = answer(PROMPT, manifest, model, inventory, tuple(records), tuple(failures), planner=planner)
            folder = destination / source.parent.name
            folder.mkdir()
            write_json(folder / "program-before.json", before.model_dump(mode="json"))
            write_json(folder / "program-after.json", after.model_dump(mode="json"))
            write_json(folder / "extrema-before.json", before_extrema)
            write_json(folder / "extrema-after.json", after_extrema)
            write_json(folder / "primitive-records.json", [r.to_json() for r in records])
            write_json(folder / "primitive-failures.json", [f.to_json() for f in failures])
            summary = result.summary()
            summary.update(intake_profile="legacy", fresh_exact_region_leaves=len(records),
                           leaf_failures=len(failures), duration_scales=[r.measurements["duration_scale"] for r in records],
                           authored_duration_s=after.duration_s, keys_and_phases_unchanged=True,
                           scalar_intervals_checked=len(after_extrema),
                           max_endpoint_range_excess=max(r["endpoint_range_excess"] for r in after_extrema),
                           total_wall_seconds=time.perf_counter()-started,
                           native_timestep_s=float(model.opt.timestep))
            if result.bound:
                assert result.bound.grounded.program == after
            if result.trajectory:
                tr = result.trajectory
                np.savez_compressed(folder / "reference.npz", time_s=tr.times_s,
                                    qpos=tr.qpos, qvel=tr.qvel, qacc=tr.qacc)
            if result.certification:
                certification = result.certification
                tr = certification.trace
                np.savez_compressed(folder / "rollout.npz", time_s=tr.times_s, qpos=tr.qpos,
                                    qvel=tr.qvel, ctrl=tr.ctrl, demand=tr.demand,
                                    tracking_error_m=tr.tracking_error_m)
                summary.update(actual_physics_duration_s=float(tr.times_s[-1]),
                               repeats=certification.repeats, replay_hashes=certification.replay_hashes,
                               violations=[asdict(v) for v in certification.violations])
            write_json(folder / "summary.json", summary)
            bodies.append(summary)
            print(json.dumps(summary), flush=True)
    finally:
        gates.simulate = original_simulate
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    report = {"schema": "g05.bounded_curves_diagnostic.v1", "source_commit": commit,
              "baseline_commit": BASE, "native_clock_commit": CLOCK, "mujoco_version": mujoco.__version__,
              "prompt": PROMPT, "bodies": bodies, "canonical_successes": sum(b["accepted"] for b in bodies),
              "goal_G05_complete": False, "api_calls": 0,
              "limitations": ["Legacy-profile prerequisite diagnostic, not the G05 120-variant campaign.",
                              "G01 replay/media and G03 structural-profile integration remain pending.",
                              "Authored durations unchanged; comparison to old actual physics durations remains a G05 requirement.",
                              "Per-scalar curve bounds do not establish task success or bounds after additive/refinement operations."]}
    write_json(destination / "report.json", report)
    write_json(destination / "payloads.json", {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sorted(destination.rglob("*")) if p.is_file()})
    print(json.dumps({"canonical_successes": report["canonical_successes"], "G05_complete": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    main(parser.parse_args().destination)
