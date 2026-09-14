"""Before/after pairs for the fixes made while the G17 controllers were being finished, each pair on identical physics.

    python any-robot/scripts/g17_before_after.py --out docs/results/g17-before-after --local any-robot/results/g17-before-after

Each pair runs the same body, the same course, the same seed and the
same disturbance twice: once with the fix switched off (the controller
or navigator as it first was) and once as committed. Both runs are
sealed and rendered in full; the index says which is which and what
each did. The pairs: the biped's balance before its undamped leg servos
were held under the wheel torque (it rings against the torque limit);
the biped's arrival before the navigator's run-in and the gated integral
(after a push from behind it arrived fast, stopped at the route point
against the station platform, and could not turn); the octopus's
in-place turn before its sweep sense was corrected (at the station the
navigator drove the heading back to the seam instead of round); the
dog's in-place turn before its stride difference was boosted (a slow
turn that drifted). Rendering needs ffmpeg and ffprobe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import mujoco

from rigby_general.evidence.capture import json_bytes
from rigby_general.mobility import load_mobile_body, measure_mobile_body
from rigby_general.mobility.evidence import render_run, seal_run
from rigby_general.mobility.locomotion import DogTrot, OctopusCrawl, WheeledBalance
from rigby_general.mobility.trials import Perturbation, course_world_xml, run_travel
from rigby_general.mobility.world import course_v1

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MOBILE = ROOT / "assets/general/mobile"
CAMERA = {"centre": [1.5, 0.0, 0.25], "reach": 2.3}
WAYPOINTS = [(3.0, 0.0), (0.0, 0.0)]

PAIRS = [
    {"pair_id": "biped-legs-held", "body": "mobile_wheeled_biped", "seed": 1, "cap_s": 90.0, "perturbation": None,
     "fix": "the undamped leg servos held under the wheel torque (feed-forward) and damped through the same servos", "before": {"WheeledBalance.HOLD_LEGS": False}, "after": {},
     "symptom": "the balance rings against the 12 N m torque limit: the legs flex in series with the pitch loop"},
    {"pair_id": "biped-arrival", "body": "mobile_wheeled_biped", "seed": 201, "cap_s": 90.0, "perturbation": Perturbation(kind="push", detail="80 N aft push on the base for 0.3 s at 4 s", at_s=4.0, duration_s=0.3, magnitude=80.0, direction=(-1.0, 0.0, 0.0)),
     "fix": "the navigator's run-in and floor at the drive's minimum speed, the lean fed forward, the integral gated during transients", "before": {"WheeledBalance.GATE_INTEGRAL": False, "navigator.run_in": False}, "after": {},
     "symptom": "after a push from behind the integral winds up, the body arrives fast and stops at the route point against the station platform, where a turn in place is blocked"},
    {"pair_id": "octopus-turn-sense", "body": "mobile_octopus", "seed": 1, "cap_s": 300.0, "perturbation": None,
     "fix": "the in-place sweep difference's sense corrected (it is the opposite of the moving differential's)", "before": {"OctopusCrawl.IN_PLACE_SENSE": 1.0, "navigator.commit_turns": False}, "after": {},
     "symptom": "at the station the body turns the wrong way for the command; the navigator's feedback then holds the heading at the seam and it never comes back"},
    {"pair_id": "dog-turn-boost", "body": "mobile_dog_arm", "seed": 1, "cap_s": 120.0, "perturbation": None,
     "fix": "the in-place stride difference half again as large", "before": {"DogTrot.BOOST_IN_PLACE_TURN": False}, "after": {},
     "symptom": "two-centimetre strides mostly slip: the turn at the station is slow"},
]
CLASSES = {"WheeledBalance": WheeledBalance, "OctopusCrawl": OctopusCrawl, "DogTrot": DogTrot}
DEFAULTS = {"WheeledBalance.HOLD_LEGS": True, "WheeledBalance.GATE_INTEGRAL": True, "OctopusCrawl.IN_PLACE_SENSE": -1.0, "DogTrot.BOOST_IN_PLACE_TURN": True}


def apply(settings: dict) -> dict:
    navigator = {}
    for key, value in DEFAULTS.items():
        cls, attr = key.split(".")
        setattr(CLASSES[cls], attr, settings.get(key, value))
    for key, value in settings.items():
        if key.startswith("navigator."):
            navigator[key.split(".", 1)[1]] = value
    return navigator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--pairs", default=",".join(p["pair_id"] for p in PAIRS))
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not args.no_render and (not ffmpeg or not ffprobe):
        raise SystemExit("ffmpeg and ffprobe are required")
    args.out.mkdir(parents=True, exist_ok=True)
    course = course_v1()
    index = {"goal": "G17", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "generation_calls": 0, "pairs": []}
    for pair in [p for p in PAIRS if p["pair_id"] in args.pairs.split(",")]:
        body = load_mobile_body(MOBILE / pair["body"])
        manifest = measure_mobile_body(body)
        entry = {"pair_id": pair["pair_id"], "body": pair["body"], "seed": pair["seed"], "perturbation": pair["perturbation"].as_json() if pair["perturbation"] else None, "fix": pair["fix"], "symptom": pair["symptom"], "runs": {}}
        for side in ("before", "after"):
            navigator = apply(pair[side])
            result = run_travel(body, course, seed=pair["seed"], waypoints=WAYPOINTS, cap_s=pair["cap_s"], perturbation=pair["perturbation"], navigator_options=navigator)
            apply({})
            world_xml = course_world_xml(body, course, pair["perturbation"])
            model = mujoco.MjSpec.from_string(world_xml).compile()
            on_course = replace(body, floor_model=model, floor_xml=world_xml)
            status = "success" if result.success else ("fell" if result.fell else ("time_cap" if result.reason.startswith("time cap") else "unstable"))
            label = f"{pair['pair_id']}-{side}"
            caption = f"{side.upper()} the fix ({pair['fix'][:70]}): {pair['perturbation'].detail if pair['perturbation'] else 'travel start -> station -> start'}"
            sealed = seal_run(args.local / pair["pair_id"] / side / "physical", body=on_course, manifest=manifest, run=result.run, label=label, caption=caption[:150], goal="G17", protocol="rigby.mobile-locomotion-pair/1",
                              test={"test": "before_after_pair", "pair_id": pair["pair_id"], "side": side, "settings": {k: v for k, v in pair[side].items()}, "seed": pair["seed"], "waypoints": WAYPOINTS, "cap_s": pair["cap_s"], "perturbation": pair["perturbation"].as_json() if pair["perturbation"] else None,
                                    "course_sha256": course.sha256(), "controller": {"name": result.actuation["controller"], "provenance": result.actuation["provenance"]}},
                              outcome={"status": status, "success": result.success, "reason": result.reason, "fell": result.fell, "fall_time_s": result.fall_time_s, "reached": result.reached, "collisions": result.collisions, "energy_j": result.energy_j, "distance_m": result.distance_m,
                                       "duration_s": result.duration_s, "final_distance_m": result.final_distance_m, "stable_at_end": result.stable_at_end, "navigator_log": result.navigator_log, "perturbation_log": result.perturbation_log, "support": result.support},
                              camera=CAMERA, world_xml=world_xml, model=model, controls_note=f"{side} the fix: {result.actuation['controller']} + navigator" + (f"; {pair['perturbation'].detail}" if pair["perturbation"] else ""))
            run = {"success": result.success, "status": status, "reason": result.reason, "fell": result.fell, "duration_s": round(result.duration_s, 2), "distance_m": round(result.distance_m, 3), "peak_speed_mps": round(result.peak_speed_mps, 3), "energy_j": round(result.energy_j, 1),
                   "collisions": result.collisions, "final_distance_m": round(result.final_distance_m, 3), "navigator_log": result.navigator_log, "saturation_max": round(max(result.actuation["saturation_fraction"].values()), 4), "settings": pair[side],
                   "sealed": {**sealed, "bundle": Path(sealed["bundle"]).relative_to(REPO).as_posix() if Path(sealed["bundle"]).is_absolute() else sealed["bundle"]}}
            if not args.no_render:
                phases = [{"label": f"{side.upper()} the fix", "start_s": 0.0, "end_s": result.duration_s + 1.0}]
                rendered = render_run(Path(sealed["bundle"]), args.out / pair["pair_id"] / side, ffmpeg=ffmpeg, ffprobe=ffprobe, phases=phases)
                run["media"] = {"video": f"{pair['pair_id']}/{side}/media/episode.mp4", "preview": f"{pair['pair_id']}/{side}/media/preview.gif", "frames": f"{pair['pair_id']}/{side}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"]}
            entry["runs"][side] = run
            print(json.dumps({"pair": pair["pair_id"], "side": side, "success": result.success, "reason": result.reason, "t": round(result.duration_s, 1), "saturation": run["saturation_max"]}), flush=True)
        index["pairs"].append(entry)
        (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
