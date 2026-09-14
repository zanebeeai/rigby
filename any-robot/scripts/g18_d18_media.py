"""D18: uninterrupted retrieve-and-deliver on the dog, the wheels and the tentacles, with carrying disturbances, from the sealed G18 trials.

    python any-robot/scripts/g18_d18_media.py --results docs/results/g18-retrieve --out docs/results/g18-d18

Per body, from the sealed runs of the campaign: the first successful
nominal retrieve with every phase and the holding limb in the banner;
the first recovered disturbance of each stage (navigation, carrying,
placement) with the disturbance window in the banner; and the first
failed trial, labelled as a failure with its reason. Every clip is the
full run at real-time playback from recorded states, frames.json beside
it, a labelled GIF summary for the report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rigby_general.evidence.capture import json_bytes
from rigby_general.mobility.evidence import render_run


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BODIES = ("mobile_dog_arm_v2", "mobile_wheeled_biped_v2", "mobile_octopus_v2")
NAMES = {"mobile_dog_arm_v2": "dog (v2 jaw)", "mobile_wheeled_biped_v2": "wheeled biped (v2 jaw)", "mobile_octopus_v2": "octopus (v2 pincer)"}


def phases_of(row: dict) -> list[dict]:
    """Every phase attempt with its support set and holding limb, plus the hold events and the disturbance window."""

    phases = []
    for p in row["phases"]:
        label = f"{p['phase']} #{p['attempt']}" + (f" | holding {p['holding']}" if p["holding"] else "") + f" | support: {len(p['support'])} members" + ("" if p["success"] else f" | FAILED: {p['reason'][:40]}")
        phases.append({"label": label[:110], "start_s": p["started_s"], "end_s": max(p["ended_s"], p["started_s"] + 0.1)})
    last = row["phases"][-1]["ended_s"] if row["phases"] else 0.0
    phases.append({"label": "DONE: SUCCESS" if row["success"] else f"DONE: FAILED ({row['reason'][:60]})", "start_s": last, "end_s": row["duration_s"] + 1.0})
    for event in row.get("disturbance_log", []):
        if event["event"] == "on":
            off = next((e["time_s"] for e in row["disturbance_log"] if e["event"] == "off" and e["time_s"] > event["time_s"]), event["time_s"] + 0.5)
            phases.insert(0, {"label": f"DISTURBANCE ON: {row['disturbance']['detail'][:70]}", "start_s": event["time_s"], "end_s": off})
    for event in row.get("hold_events", []):
        phases.insert(0, {"label": f"HOLD {event['event'].upper()}", "start_s": event["time_s"], "end_s": event["time_s"] + 1.2})
    return phases


def clip_entry(name: str, body_id: str, row: dict, rendered: dict, title: str) -> dict:
    return {"clip": name, "title": title, "trial_id": row["trial_id"], "kind": row["kind"], "stage": row["stage"], "outcome": "success" if row["success"] else ("fell" if row["fell"] else "failed"), "success": row["success"], "reason": row["reason"], "fell": row["fell"],
            "disturbance": row["disturbance"]["detail"] if row["disturbance"] else None, "phases_completed": row["phases_completed"], "duration_s": row["duration_s"], "video": f"{body_id}/{name}/media/episode.mp4", "preview": f"{body_id}/{name}/media/preview.gif",
            "frames": f"{body_id}/{name}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"], "source_sha256": row["sealed"]["sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bodies", default=",".join(BODIES))
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    args.out.mkdir(parents=True, exist_ok=True)
    index = {"goal": "G18", "demo": "D18", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "bodies": {}, "generation_calls": 0}
    if (args.out / "index.json").is_file():
        earlier = json.loads((args.out / "index.json").read_bytes())
        index["bodies"] = {b: c for b, c in earlier.get("bodies", {}).items() if b not in args.bodies.split(",")}
    for body_id in args.bodies.split(","):
        rows = json.loads((args.results / body_id / "trials.json").read_bytes())
        sealed = [r for r in rows if "sealed" in r]
        chosen = []
        nominal = next((r for r in sealed if r["kind"] == "nominal" and r["success"]), None)
        if nominal is not None:
            chosen.append(("retrieve", nominal, f"retrieve and deliver: start -> station -> tray -> start ({NAMES[body_id]})"))
        for stage in ("navigation", "carrying", "placement"):
            recovered = next((r for r in sealed if r["stage"] == stage and r["success"]), None)
            if recovered is not None:
                chosen.append((f"recovery-{stage}", recovered, f"{stage} disturbance, recovered: {recovered['disturbance']['detail']}"))
        failure = next((r for r in sealed if not r["success"] and r["kind"] == "nominal"), None) or next((r for r in sealed if not r["success"]), None)
        if failure is not None:
            chosen.append(("failure", failure, f"FAILURE ({failure['stage']}): {failure['disturbance']['detail'] if failure['disturbance'] else 'nominal'}; {failure['reason']}"))
        clips = []
        for name, row, title in chosen:
            rendered = render_run(REPO / row["sealed"]["bundle"], args.out / body_id / name, ffmpeg=ffmpeg, ffprobe=ffprobe, phases=phases_of(row))
            clips.append(clip_entry(name, body_id, row, rendered, title))
            print(json.dumps({"body": body_id, "clip": name, "trial": row["trial_id"], "success": row["success"], "frames": rendered["frames"]}), flush=True)
        index["bodies"][body_id] = clips
    (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
