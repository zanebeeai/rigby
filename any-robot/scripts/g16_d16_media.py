"""D16: physical inspection and contact/support tests for the three mobile bodies, including their unsupported manoeuvres.

    python any-robot/scripts/g16_d16_media.py --bodies docs/results/g16-bodies --course docs/results/g16-course --out docs/results/g16-d16

Per body, from the sealed runs: the inspection sweep with the joint under
inspection named on every frame, the working (or holdable) stance's
settling test, one supported recovery, and every unsupported manoeuvre,
labelled as such; and the body settled on the course's start pad. Every
clip is the full run at real-time playback with the simulation clock,
the base height and tilt, and the floor contacts at that frame in the
banner; frames.json beside each; a labelled GIF summary for the report.
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
BODIES = ("mobile_dog_arm", "mobile_wheeled_biped", "mobile_octopus")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bodies", type=Path, required=True)
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    args.out.mkdir(parents=True, exist_ok=True)
    index = {"goal": "G16", "demo": "D16", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "bodies": {}, "generation_calls": 0}
    course = json.loads((args.course / "course-settle.json").read_bytes())["bodies"]
    for body_id in BODIES:
        tests = json.loads((args.bodies / body_id / "tests.json").read_bytes())
        clips = []
        inspection = tests["inspection"]
        phases = [{"label": f"inspecting {e['joint']} ({e['limb']}, {e['role']})", "start_s": e["start_s"], "end_s": e["end_s"]} for e in inspection["schedule"]]
        rendered = render_run(Path(inspection["sealed"]["bundle"]), args.out / body_id / "inspection", ffmpeg=ffmpeg, ffprobe=ffprobe, phases=phases)
        clips.append({"clip": "inspection", "title": f"physical inspection in stance '{inspection['stance']}'", "outcome": "stable" if inspection["upright"] else "disturbed", "video": f"{body_id}/inspection/media/episode.mp4", "preview": f"{body_id}/inspection/media/preview.gif",
                      "frames": f"{body_id}/inspection/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"], "source_sha256": inspection["sealed"]["sha256"], "joints_swept": len(inspection["schedule"])})
        print(json.dumps({"body": body_id, "clip": "inspection", "frames": rendered["frames"]}), flush=True)
        for stance in tests["stances"]:
            rendered = render_run(Path(stance["sealed"]["bundle"]), args.out / body_id / f"stance-{stance['stance']}", ffmpeg=ffmpeg, ffprobe=ffprobe)
            clips.append({"clip": f"stance-{stance['stance']}", "title": f"settling test, stance '{stance['stance']}'", "outcome": stance["status"], "declared_stable": stance["declared_statically_stable"], "measured_stable": stance["statically_stable"],
                          "support_contacts": stance["support_contacts"], "video": f"{body_id}/stance-{stance['stance']}/media/episode.mp4", "preview": f"{body_id}/stance-{stance['stance']}/media/preview.gif",
                          "frames": f"{body_id}/stance-{stance['stance']}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"], "source_sha256": stance["sealed"]["sha256"]})
            print(json.dumps({"body": body_id, "clip": f"stance-{stance['stance']}", "frames": rendered["frames"]}), flush=True)
        supported = [r for r in tests["recoveries"] if r["supported"]]
        chosen = [max(supported, key=lambda r: abs(r["roll_deg"]) + abs(r["pitch_deg"]))] + [r for r in tests["recoveries"] if not r["supported"]]
        for trial in chosen:
            name = f"recovery-{trial['trial_id']}"
            rendered = render_run(Path(trial["sealed"]["bundle"]), args.out / body_id / name, ffmpeg=ffmpeg, ffprobe=ffprobe)
            clips.append({"clip": name, "title": f"recovery trial: {trial['perturbation']}", "supported": trial["supported"], "outcome": trial["status"], "recovered": trial["recovered"], "final_tilt_deg": trial["final_tilt_deg"],
                          "video": f"{body_id}/{name}/media/episode.mp4", "preview": f"{body_id}/{name}/media/preview.gif", "frames": f"{body_id}/{name}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"], "source_sha256": trial["sealed"]["sha256"]})
            print(json.dumps({"body": body_id, "clip": name, "supported": trial["supported"], "recovered": trial["recovered"], "frames": rendered["frames"]}), flush=True)
        start = course[body_id]
        rendered = render_run(Path(start["sealed"]["bundle"]), args.out / body_id / "course-start", ffmpeg=ffmpeg, ffprobe=ffprobe)
        clips.append({"clip": "course-start", "title": f"on the course: settled on the start pad in stance '{start['stance']}'", "outcome": "stable" if start["stable"] else "unstable", "video": f"{body_id}/course-start/media/episode.mp4",
                      "preview": f"{body_id}/course-start/media/preview.gif", "frames": f"{body_id}/course-start/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"], "source_sha256": start["sealed"]["sha256"]})
        index["bodies"][body_id] = clips
    (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
