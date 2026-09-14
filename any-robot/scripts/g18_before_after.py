"""Before/after pairs for the fixes G18 made, each pair on identical physics: the same course, seed and controller, once as it was and once as committed.

    python any-robot/scripts/g18_before_after.py --out docs/results/g18-before-after --local any-robot/results/g18-before-after

Two pairs are body corrections: the dog's first-version jaw against
its second (the wrist capsule between the fingers, so the jaw took the
cube by its top and lost it on the lift), and the octopus's first-
version pincer against its second (fingers that parted less than the
cube, and no torsional friction on the pads). Two are controller fixes
switched off and on: the dog's placement from the arm's folded carry
branch against the best branch (the wrist came down on the tray's
rim), and the octopus's tripod crawl with a tentacle held out against
the wave it crawls now (one phase of the tripod stood on two
tentacles). Both runs of a pair are sealed and rendered in full with
the side named in the banner. The wave-gait pair does not go back for
a lost hold (a recovery budget of none): it shows the carry, and a
crawler's recovery attempts would seal to three quarters of a gigabyte
a side. Rendering needs ffmpeg and ffprobe.
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
from rigby_general.mobility.locomotion import OctopusCrawl
from rigby_general.mobility.retrieve import RetrieveSession, run_retrieve
from rigby_general.mobility.trials import course_world_xml
from rigby_general.mobility.world import course_v1

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
MOBILE = ROOT / "assets/general/mobile"
CAMERA = {"centre": [1.7, 1.1, 0.25], "reach": 2.6}
CAPS = {"mobile_dog_arm": 360.0, "mobile_dog_arm_v2": 360.0, "mobile_octopus": 800.0, "mobile_octopus_v2": 800.0}

PAIRS = [
    {"pair_id": "dog-jaw", "before": {"body": "mobile_dog_arm"}, "after": {"body": "mobile_dog_arm_v2"}, "seed": 1,
     "fix": "the jaw hung below the last arm link (second-version body)", "symptom": "the first version's wrist capsule ran between the fingers: the jaw closed on the cube's top few millimetres and lost it on the lift"},
    {"pair_id": "octopus-pincer", "before": {"body": "mobile_octopus"}, "after": {"body": "mobile_octopus_v2"}, "seed": 1,
     "fix": "pincer hinges that open to -0.9 rad and finger pads with torsional friction (second-version body)", "symptom": "the first version's open fingers parted less than the cube, so the pincer could not close on it"},
    {"pair_id": "dog-place-branch", "before": {"body": "mobile_dog_arm_v2", "RetrieveSession.PLACE_ON_BEST_BRANCH": False}, "after": {"body": "mobile_dog_arm_v2"}, "seed": 2,
     "fix": "the placement reaches over the tray on the arm's best branch", "symptom": "a straight-line move from the carry pose kept the arm folded back over the torso and brought the wrist down on the tray's rim"},
    {"pair_id": "octopus-wave-gait", "before": {"body": "mobile_octopus_v2", "OctopusCrawl.WAVE_WHEN_HOLDING": False}, "after": {"body": "mobile_octopus_v2"}, "seed": 1, "recovery_budget": 0,
     "fix": "a wave gait while a tentacle holds the object", "symptom": "the tripod with a tentacle held out stood on two tentacles in one of its phases, and the mantle's rocking shook the object out"},
]
DEFAULTS = {"RetrieveSession.PLACE_ON_BEST_BRANCH": True, "OctopusCrawl.WAVE_WHEN_HOLDING": True}
CLASSES = {"RetrieveSession": RetrieveSession, "OctopusCrawl": OctopusCrawl}


def apply(settings: dict) -> None:
    for key, default in DEFAULTS.items():
        cls, attr = key.split(".")
        setattr(CLASSES[cls], attr, settings.get(key, default))


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
    index = {"goal": "G18", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "generation_calls": 0, "pairs": []}
    if (args.out / "index.json").is_file():
        # pairs run earlier keep their entries; the pairs asked for are run afresh
        earlier = json.loads((args.out / "index.json").read_bytes())
        index["pairs"] = [p for p in earlier.get("pairs", []) if p["pair_id"] not in args.pairs.split(",")]
    for pair in [p for p in PAIRS if p["pair_id"] in args.pairs.split(",")]:
        recovery_budget = pair.get("recovery_budget", 2)
        entry = {"pair_id": pair["pair_id"], "seed": pair["seed"], "fix": pair["fix"], "symptom": pair["symptom"], "retry_budget": 2, "recovery_budget": recovery_budget, "runs": {}}
        for side in ("before", "after"):
            settings = dict(pair[side])
            body_id = settings.pop("body")
            body = load_mobile_body(MOBILE / body_id)
            manifest = measure_mobile_body(body)
            apply(settings)
            cap_s = pair.get("cap_s", CAPS[body_id])
            result = run_retrieve(body, course, seed=pair["seed"], cap_s=cap_s, recovery_budget=recovery_budget)
            apply({})
            world_xml = course_world_xml(body, course, None)
            model = mujoco.MjSpec.from_string(world_xml).compile()
            on_course = replace(body, floor_model=model, floor_xml=world_xml)
            status = "success" if result.success else ("fell" if result.fell else "failed")
            label = f"{pair['pair_id']}-{side}"
            caption = f"{side.upper()} the fix ({pair['fix'][:60]}): retrieve, {body_id}"
            sealed = seal_run(args.local / pair["pair_id"] / side / "physical", body=on_course, manifest=manifest, run=result.run, label=label, caption=caption[:150], goal="G18", protocol="rigby.mobile-retrieve-pair/1",
                              test={"test": "before_after_pair", "pair_id": pair["pair_id"], "side": side, "body": body_id, "settings": settings, "seed": pair["seed"], "cap_s": cap_s, "retry_budget": 2, "recovery_budget": recovery_budget, "course_sha256": course.sha256(),
                                    "controller": {"name": result.actuation["controller"], "provenance": result.actuation["provenance"]}},
                              outcome={"status": status, "success": result.success, "reason": result.reason, "fell": result.fell, "cube_in_tray": result.cube_in_tray, "at_start": result.at_start, "phases": [{"phase": p.phase, "attempt": p.attempts, "success": p.success, "reason": p.reason, "started_s": p.started_s, "ended_s": p.ended_s} for p in result.phases],
                                       "hold_events": result.hold_events, "duration_s": result.duration_s, "max_object_slip_m": result.max_object_slip_m, "invariant_violation_count": len(result.invariant_violations)},
                              camera=CAMERA, world_xml=world_xml, model=model, controls_note=f"{side} the fix: {result.actuation['controller']} + reach + navigator on {body_id}")
            run = {"body": body_id, "success": result.success, "status": status, "reason": result.reason, "fell": result.fell, "duration_s": round(result.duration_s, 2), "cap_s": cap_s, "phases_completed": sorted({p.phase for p in result.phases if p.success}),
                   "hold_events": result.hold_events, "max_object_slip_m": round(result.max_object_slip_m, 4), "settings": settings, "phases": [{"phase": p.phase, "attempt": p.attempts, "success": p.success, "reason": p.reason, "started_s": round(p.started_s, 2), "ended_s": round(p.ended_s, 2)} for p in result.phases],
                   "sealed": {**sealed, "bundle": Path(sealed["bundle"]).relative_to(REPO).as_posix() if Path(sealed["bundle"]).is_absolute() else sealed["bundle"]}}
            if not args.no_render:
                phases = [{"label": f"{side.upper()} the fix | {p.phase} #{p.attempts}" + ("" if p.success else f" | FAILED: {p.reason[:40]}"), "start_s": p.started_s, "end_s": max(p.ended_s, p.started_s + 0.1)} for p in result.phases]
                for event in result.hold_events:
                    phases.insert(0, {"label": f"{side.upper()} the fix | HOLD {event['event'].upper()}", "start_s": event["time_s"], "end_s": event["time_s"] + 1.2})
                rendered = render_run(Path(sealed["bundle"]), args.out / pair["pair_id"] / side, ffmpeg=ffmpeg, ffprobe=ffprobe, phases=phases)
                run["media"] = {"video": f"{pair['pair_id']}/{side}/media/episode.mp4", "preview": f"{pair['pair_id']}/{side}/media/preview.gif", "frames": f"{pair['pair_id']}/{side}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"]}
            entry["runs"][side] = run
            print(json.dumps({"pair": pair["pair_id"], "side": side, "body": body_id, "success": result.success, "reason": result.reason[:80], "t": round(result.duration_s, 1), "phases": run["phases_completed"]}), flush=True)
        index["pairs"].append(entry)
        (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
