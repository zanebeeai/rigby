"""D17: the synchronized walk / roll / crawl course, and uninterrupted perturbation recoveries, from the sealed G17 trials.

    python any-robot/scripts/g17_d17_media.py --results docs/results/g17-locomotion --out docs/results/g17-d17

Per body, from the sealed runs of the campaign: the first successful
travel trial, with the navigator's phases (out to the station, holding,
back to the start, holding) in the banner; the first recovered push,
slick patch and support disturbance, with the disturbance window in the
banner; and the first failed trial of any kind, labelled as a failure.
Then the three travel clips composed side by side on one simulation
clock: the dog walks, the biped rolls, the octopus crawls the same
course, each panel with its own outcome, until the slowest is done.
Every clip is the full run at real-time playback from recorded states,
frames.json beside it, a labelled GIF summary for the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rigby_core.evidence import verify_bundle, write_bundle
from rigby_core.simulation.recording import PhysicsRecord, STATE_SPEC

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import HEIGHT, PREVIEW_LIMIT, WIDTH
from rigby_general.mobility.evidence import BACKGROUND, BANNER, FPS, _camera, _contacts_at, render_run
from rigby_general.mobility.validate import robot_bodies


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
BODIES = ("mobile_dog_arm", "mobile_wheeled_biped", "mobile_octopus")
GAIT = {"mobile_dog_arm": "walks (trot)", "mobile_wheeled_biped": "rolls (balance)", "mobile_octopus": "crawls (tripod)"}


def phases_of(row: dict) -> list[dict]:
    """The navigator's phases and the disturbance window, for the banner."""

    phases = []
    events = row["navigator_log"]
    names = {0: "out to the station", 1: "back to the start"}
    start = 1.5
    for event in events:
        if event["event"] == "arrived":
            phases.append({"label": names.get(event["waypoint"], f"to waypoint {event['waypoint']}"), "start_s": start, "end_s": event["time_s"]})
            start = event["time_s"]
        elif event["event"] == "released":
            phases.append({"label": "holding at the station" if event["waypoint"] == 0 else "holding at the start", "start_s": start, "end_s": event["time_s"]})
            start = event["time_s"]
        elif event["event"] == "drifted":
            phases.append({"label": "drifted out, returning", "start_s": start, "end_s": event["time_s"]})
            start = event["time_s"]
    phases.append({"label": "done" if row["success"] else row["reason"], "start_s": start, "end_s": row["duration_s"] + 1.0})
    for event in row.get("perturbation_log", []):
        if event["event"] == "on":
            off = next((e["time_s"] for e in row["perturbation_log"] if e["event"] == "off" and e["time_s"] > event["time_s"]), event["time_s"] + 0.5)
            phases.insert(0, {"label": f"DISTURBANCE ON: {row['perturbation']['detail'][:60]}", "start_s": event["time_s"], "end_s": off})
    return phases


def clip_entry(name: str, body_id: str, row: dict, rendered: dict, title: str) -> dict:
    return {"clip": name, "title": title, "trial_id": row["trial_id"], "kind": row["kind"], "outcome": "success" if row["success"] else ("fell" if row["fell"] else "failed"), "success": row["success"], "reason": row["reason"], "fell": row["fell"],
            "perturbation": row["perturbation"]["detail"] if row["perturbation"] else None, "duration_s": row["duration_s"], "video": f"{body_id}/{name}/media/episode.mp4", "preview": f"{body_id}/{name}/media/preview.gif",
            "frames": f"{body_id}/{name}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"], "source_sha256": row["sealed"]["sha256"]}


def render_synchronized(roots: dict[str, Path], rows: dict[str, dict], destination: Path, *, ffmpeg: str, ffprobe: str) -> dict:
    """The three travel runs side by side on one simulation clock, each panel from its own sealed record; a body that finishes early holds its last state."""

    panels = {}
    for body_id, root in roots.items():
        manifest = verify_bundle(root)
        record = PhysicsRecord.from_bytes((root / "trace.npz").read_bytes())
        model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
        model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
        model.vis.headlight.ambient[:] = 0.7
        model.vis.headlight.diffuse[:] = 0.8
        panels[body_id] = {"manifest": manifest, "metadata": dict(manifest["metadata"]), "record": record, "model": model, "data": mujoco.MjData(model), "digest": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                           "base": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, manifest["metadata"]["base_body"]), "robot": robot_bodies(model, manifest["metadata"]["base_body"])}
    longest = max(float(p["record"].arrays["time_s"][-1]) for p in panels.values())
    frame_times = np.arange(0.0, longest + 1e-9, 1.0 / FPS)
    if frame_times[-1] < longest:
        frame_times = np.append(frame_times, longest)
    preview_at = set(np.linspace(0, len(frame_times) - 1, min(PREVIEW_LIMIT, len(frame_times))).astype(int).tolist())
    speed = (len(frame_times) / FPS) / (len(preview_at) / 10)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=12)
    frames = []
    previews = []
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    width = 3 * WIDTH
    with tempfile.TemporaryDirectory(prefix=".d17-", dir=destination) as temporary:
        staging = Path(temporary)
        video = staging / "episode.mp4"
        process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{HEIGHT + BANNER}", "-r", str(FPS), "-i", "pipe:0",
                                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            renderers = {body_id: mujoco.Renderer(p["model"], height=HEIGHT, width=WIDTH) for body_id, p in panels.items()}
            for number, time_s in enumerate(frame_times):
                canvas = Image.new("RGB", (width, HEIGHT + BANNER), BACKGROUND)
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 6), f"D17 | synchronized course | sim t={time_s:.3f}s | one clock, three bodies, each from its own sealed record", fill=(191, 210, 233), font=font)
                draw.text((12, 27), "FULL EPISODE | real-time playback | recorded physical states | each body's own controller and navigator, nothing else acting", fill="white", font=font)
                frame_row = {"frame": number, "playback_time_s": number / FPS, "simulation_time_s": float(time_s), "panels": {}}
                for column, (body_id, p) in enumerate(panels.items()):
                    times = p["record"].arrays["time_s"]
                    index = int(min(np.searchsorted(times, time_s, side="right") - 1, len(times) - 1))
                    index = max(index, 0)
                    mujoco.mj_setState(p["model"], p["data"], p["record"].arrays["state"][index], STATE_SPEC)
                    mujoco.mj_forward(p["model"], p["data"])
                    camera = _camera(**p["metadata"]["camera"], task=False)
                    renderers[body_id].update_scene(p["data"], camera=camera)
                    canvas.paste(Image.fromarray(np.asarray(renderers[body_id].render()).copy()), (column * WIDTH, BANNER))
                    up = np.zeros(3)
                    mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), p["data"].xquat[p["base"]])
                    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
                    contacts = _contacts_at(p["model"], p["record"], index, p["robot"])
                    row = rows[body_id]
                    finished = time_s >= float(times[-1]) - 1e-9
                    phase = next((ph["label"] for ph in phases_of(row) if ph["start_s"] <= time_s < ph["end_s"]), "")
                    status = ("DONE: " + ("SUCCESS" if row["success"] else row["reason"].upper())) if finished else phase
                    tint = (100, 230, 165) if row["success"] else (255, 189, 100)
                    draw.text((column * WIDTH + 12, 48), f"{body_id} {GAIT[body_id]} | {status}"[:70], fill=tint, font=small)
                    draw.text((column * WIDTH + 12, 64), f"tilt {tilt:.1f} deg | ground: {', '.join(contacts) if contacts else 'none'}"[:70], fill=(170, 220, 190), font=small)
                    draw.text((column * WIDTH + 12, 80), f"own clock {float(times[index]):.2f}s of {float(times[-1]):.1f}s", fill=(170, 185, 205), font=small)
                    frame_row["panels"][body_id] = {"recorded_sample": index, "own_time_s": float(times[index]), "finished": bool(finished), "tilt_deg": round(tilt, 2), "floor_contacts": contacts, "phase": phase}
                process.stdin.write(canvas.tobytes())
                frames.append(frame_row)
                if number in preview_at:
                    preview = canvas.copy()
                    ImageDraw.Draw(preview).text((12, 27), f"GIF SUMMARY | approximately {speed:.1f}x speed | full video: episode.mp4" + " " * 40, fill="white", font=font)
                    previews.append(preview)
            process.stdin.close()
            if process.wait(timeout=1200) != 0:
                raise RuntimeError("ffmpeg failed: " + process.stderr.read().decode(errors="replace")[-1500:])
        finally:
            for r in renderers.values():
                r.close()
            if process.poll() is None:
                process.kill()
        probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=600))["streams"][0]
        if int(probe["nb_read_frames"]) != len(frame_times):
            raise RuntimeError("the encoded video lost frames")
        previews[0].save(staging / "preview.gif", save_all=True, append_images=previews[1:], duration=100, loop=0, optimize=True)
        payloads = {"episode.mp4": video.read_bytes(), "preview.gif": (staging / "preview.gif").read_bytes(), "frames.json": json_bytes(frames),
                    "encoding.json": json_bytes({"probe": probe, "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]})}
        media = {"schema": "rigby.presentation/1", "sources": {b: {"bundle_sha256": p["digest"], "trace_sha256": p["metadata"]["trace_sha256"], "outcome": p["metadata"]["outcome"], "duration_s": float(p["record"].arrays["time_s"][-1])} for b, p in panels.items()},
                 "full_episode": True, "frame_count": len(frame_times), "fps": FPS, "simulation_duration_s": float(longest), "playback_duration_s": len(frame_times) / FPS, "preview_is_summary": True, "physics_replay_performed_by_renderer": False,
                 "overlay": {"kind": "synchronized_course", "shows": ["one simulation clock across three panels", "each body's phase, outcome, tilt and floor contacts", "a body that finishes early holds its final state"]},
                 "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        rendered = write_bundle(destination / "media", payloads, media)
    return {"media": (destination / "media").as_posix(), "sha256": rendered, "frames": len(frame_times), "playback_s": len(frame_times) / FPS, "simulation_s": float(longest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bodies", default=",".join(BODIES))
    parser.add_argument("--skip-synchronized", action="store_true")
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    args.out.mkdir(parents=True, exist_ok=True)
    index = {"goal": "G17", "demo": "D17", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "bodies": {}, "generation_calls": 0}
    travel_roots, travel_rows = {}, {}
    for body_id in args.bodies.split(","):
        rows = json.loads((args.results / body_id / "trials.json").read_bytes())
        sealed = [r for r in rows if "sealed" in r]
        chosen = []
        travel = next((r for r in sealed if r["kind"] == "travel" and r["success"]), None)
        if travel is not None:
            chosen.append(("travel", travel, f"travel: start -> station -> start ({GAIT[body_id]})"))
            travel_roots[body_id] = REPO / travel["sealed"]["bundle"]
            travel_rows[body_id] = travel
        for kind in ("push", "patch", "support"):
            recovered = next((r for r in sealed if r["kind"] == kind and r["success"]), None)
            if recovered is not None:
                chosen.append((f"recovery-{kind}", recovered, f"{kind} recovery: {recovered['perturbation']['detail']}"))
        failure = next((r for r in sealed if not r["success"]), None)
        if failure is not None:
            chosen.append(("failure", failure, f"FAILURE ({failure['kind']}): {failure['perturbation']['detail'] if failure['perturbation'] else 'travel'}; {failure['reason']}"))
        clips = []
        for name, row, title in chosen:
            rendered = render_run(REPO / row["sealed"]["bundle"], args.out / body_id / name, ffmpeg=ffmpeg, ffprobe=ffprobe, phases=phases_of(row))
            clips.append(clip_entry(name, body_id, row, rendered, title))
            print(json.dumps({"body": body_id, "clip": name, "trial": row["trial_id"], "success": row["success"], "frames": rendered["frames"]}), flush=True)
        index["bodies"][body_id] = clips
    if not args.skip_synchronized and len(travel_roots) == len(BODIES):
        rendered = render_synchronized(travel_roots, travel_rows, args.out / "synchronized", ffmpeg=ffmpeg, ffprobe=ffprobe)
        index["synchronized"] = {"clip": "synchronized", "title": "synchronized course: the dog walks, the biped rolls, the octopus crawls, on one clock", "video": "synchronized/media/episode.mp4", "preview": "synchronized/media/preview.gif", "frames": "synchronized/media/frames.json",
                                 "media_sha256": rendered["sha256"], "frame_count": rendered["frames"], "simulation_s": rendered["simulation_s"], "sources": {b: {"trial_id": r["trial_id"], "source_sha256": r["sealed"]["sha256"], "success": r["success"], "duration_s": r["duration_s"]} for b, r in travel_rows.items()}}
        print(json.dumps({"clip": "synchronized", "frames": rendered["frames"], "simulation_s": rendered["simulation_s"]}), flush=True)
    (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
