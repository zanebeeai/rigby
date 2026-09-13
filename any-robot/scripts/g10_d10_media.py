"""D10: the three-body fixed-world transfer, and slip and occlusion recoveries, uninterrupted, with the skill's own trail on every frame.

Three artifacts from the sealed campaign bundles, no physics re-run:

- the three enabled bodies' first nominal episodes tiled side by side on
  one clock, each rendered in full from its own record at real-time
  playback, the shorter ones held on their final frame;
- one induced-slip recovery per body, rendered in full and annotated
  frame by frame with what the skill was doing (the leaf and its attempt
  count), what its monitor last decided from the declared sensors (held,
  stably placed, with the sensors and their ages), the protocol's
  disturbance as it acts, and the oracle's labels from the full state,
  which the skill never reads;
- one temporary-occlusion recovery per body, annotated the same way.

Failures are shown as such: where a body's first slip or occlusion
episode did not recover, that episode is the one rendered. No API or
model calls.

    python any-robot/scripts/g10_d10_media.py --campaign docs/results/g10-campaign --local any-robot/results/g10-campaign --out docs/results/g10-d10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rigby_core.evidence import verify_bundle, write_bundle

from rigby_general.evidence.capture import json_bytes
from rigby_general.sensing import Episode, OracleLabeler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g10_corpus as corpus_module  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FPS = 12
PREVIEW_LIMIT = 60
STRIP = 118
BACKGROUND = (17, 24, 39)
COLOURS = {"pass": (52, 140, 92), "fail": (170, 62, 62), "unknown": (176, 120, 30), "none": (70, 78, 92)}


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def decode(ffmpeg: str, video: Path, width: int, height: int) -> list[Image.Image]:
    raw = subprocess.check_output([ffmpeg, "-v", "error", "-nostdin", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], timeout=600)
    count = len(raw) // (width * height * 3)
    return [Image.fromarray(f.copy()) for f in np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)]


def tile(ffmpeg: str, ffprobe: str, clips: list[Path], destination: Path) -> dict:
    streams = [probe(ffprobe, clip) for clip in clips]
    longest = max(float(s["duration"]) for s in streams)
    inputs, filters = [], []
    for index, (clip, stream) in enumerate(zip(clips, streams)):
        inputs += ["-i", str(clip)]
        filters.append(f"[{index}:v]tpad=stop_mode=clone:stop_duration={max(0.0, longest - float(stream['duration'])):.3f}[v{index}]")
    filters.append("".join(f"[v{i}]" for i in range(len(clips))) + f"xstack=inputs={len(clips)}:layout=0_0|w0_0|w0+w1_0[out]")
    command = [ffmpeg, "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-r", str(FPS), "-c:v", "libx264",
               "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    for attempt in range(3):
        if subprocess.run(command, timeout=1200).returncode == 0:
            break
    else:
        raise RuntimeError("ffmpeg could not tile the clips")
    result = probe(ffprobe, destination)
    return {"video": destination.name, "frames": int(result["nb_read_frames"]), "duration_s": float(result["duration"]), "inputs": [c.as_posix() for c in clips]}


def gif_summary(ffmpeg: str, video: Path, destination: Path, title: str, real_duration_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=".d10-gif-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        rate = max(0.25, PREVIEW_LIMIT / max(real_duration_s, 1e-6))
        subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(video), "-vf", f"fps={rate:.4f},scale=960:-1", str(staging / "f%04d.png")], timeout=600)
        frames = sorted(staging.glob("f*.png"))[:PREVIEW_LIMIT]
        speed = real_duration_s / (len(frames) / 10.0)
        font = ImageFont.load_default(size=16)
        images = []
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            canvas = Image.new("RGB", (image.width, image.height + 26), BACKGROUND)
            canvas.paste(image, (0, 26))
            ImageDraw.Draw(canvas).text((10, 5), f"{title} | GIF SUMMARY | approximately {speed:.1f}x speed | full video: {video.name}", fill=(255, 189, 100), font=font)
            images.append(canvas)
        images[0].save(destination, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
    return {"gif": destination.name, "frames": len(images), "approximate_speed": round(speed, 2), "summary_of": video.name}


def encode_frames(ffmpeg: str, video: Path, frames: list[Image.Image], width: int, height: int, *, attempts: int = 3) -> None:
    """Encode RGB frames through a pipe; the encoder is restarted on a failed attempt (a process crash mid-pipe, seen under load)."""

    last = None
    for attempt in range(attempts):
        process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0",
                                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for image in frames:
                process.stdin.write(image.tobytes())
            process.stdin.close()
            code = process.wait(timeout=600)
            if code == 0:
                return
            last = f"exit {code}: {process.stderr.read().decode(errors='replace')[-500:]}"
        except (BrokenPipeError, OSError) as error:
            try:
                process.kill()
                process.wait(timeout=30)
                last = f"{error!r}: {process.stderr.read().decode(errors='replace')[-500:]}"
            except Exception:
                last = repr(error)
    raise RuntimeError(f"ffmpeg failed after {attempts} attempts: {last}")


def timeline(outcome: dict, execution: dict) -> list[dict]:
    """Leaf spans from the runtime calls: what the skill was doing when."""

    spans = []
    for call in execution["runtime_calls"]:
        if call.get("physics_time_s"):
            spans.append({"leaf": call["leaf"], "attempt": call["attempt"], "start_s": call["physics_time_s"][0], "end_s": call["physics_time_s"][1], "verdict": call.get("verdict"), "reason": call.get("reason", "")})
        elif "trail" in call and call["trail"]:
            spans.append({"leaf": call["leaf"], "attempt": call["attempt"], "start_s": call["trail"][0]["time_s"] - 2.0, "end_s": call["trail"][-1]["time_s"], "verdict": call["trail"][-1]["decision"], "reason": call["trail"][-1]["reason"]})
    return spans


def annotate(frame: Image.Image, *, title: str, time_s: float, leaf: dict | None, monitor: dict, disturbance: str, labels: dict, attempts: dict, outcome: str, preview: str | None) -> Image.Image:
    canvas = Image.new("RGB", (frame.width, frame.height + STRIP), BACKGROUND)
    canvas.paste(frame, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    small = ImageFont.load_default(size=11)
    if preview:
        draw.rectangle((0, 51, frame.width, 68), fill=BACKGROUND)
        draw.text((12, 51), preview, fill=(191, 210, 233), font=ImageFont.load_default(size=16))
    y = frame.height + 4
    draw.text((10, y), f"{title} | sim t={time_s:.3f}s | skill verdict at end: {outcome.upper()}", fill=(255, 220, 160), font=font)
    y += 17
    doing = "between leaves" if leaf is None else f"{leaf['leaf']} (attempt {leaf['attempt']})" + (f" -> {leaf['verdict']}" + (f" {leaf['reason']}" if leaf.get("reason") else "") if leaf.get("done") else "")
    draw.text((10, y), f"skill: {doing}   |   retries: acquire_until_held {attempts['acquire']} of 3, place_until_placed {attempts['place']} of 3", fill="white", font=font)
    y += 17
    x = 10
    for name in ("reachable", "held", "stably_placed"):
        verdict = monitor.get(name)
        decision = "none" if verdict is None else verdict["decision"]
        draw.rectangle((x, y, x + 236, y + 15), fill=COLOURS[decision])
        text = f"{name}: {'not asked yet' if verdict is None else decision.upper() + ('' if not verdict['reason'] else ' ' + verdict['reason'].split(':')[0]) + f' @{verdict['time_s']:.2f}s'}"
        draw.text((x + 3, y + 1), text[:44], fill="white", font=small)
        x += 244
    y += 17
    sensors = "" if not monitor.get("cameras") else " | ".join(f"{k} {'seen' if v['visible_fraction'] and v['visible_fraction'] >= 0.5 else 'occluded'} {v['age_s'] * 1000:.0f}ms" if v["age_s"] is not None else f"{k} none" for k, v in monitor["cameras"].items())
    draw.text((10, y), f"monitor's sensors at its last decision: {sensors or 'contact only'}", fill=(170, 185, 205), font=small)
    y += 14
    draw.text((10, y), f"protocol: {disturbance}", fill=(255, 189, 100), font=small)
    y += 14
    oracle = "  ".join(f"{k}={'1' if v else '0'}" for k, v in labels.items())
    draw.text((10, y), f"oracle labels from the full state (never read by the skill): {oracle}", fill=(150, 200, 255), font=small)
    return canvas


def overlay(case: dict, physical: Path, media: Path, destination: Path, ffmpeg: str, ffprobe: str) -> dict:
    manifest = verify_bundle(media)
    frames_map = json.loads((media / "frames.json").read_bytes())
    stream = probe(ffprobe, media / "episode.mp4")
    width, height = int(stream["width"]), int(stream["height"])
    base = decode(ffmpeg, media / "episode.mp4", width, height)
    outcome = json.loads((physical / "outcome.json").read_bytes())
    execution = json.loads((physical / "execution.json").read_bytes())
    episode = Episode.load(physical)
    labeler = OracleLabeler(episode)
    effector = episode.effectors[0]
    spans = timeline(outcome, execution)
    trail = sorted(outcome["verdict_trail"], key=lambda v: v["time_s"])
    events = outcome.get("events", [])
    disturbance = outcome.get("disturbance_log", [])
    verdict = outcome["verdict"]
    annotated, previews, records = [], [], []
    preview_indices = set(np.linspace(0, len(base) - 1, min(PREVIEW_LIMIT, len(base))).astype(int).tolist())
    speed = (len(base) / FPS) / (len(preview_indices) / 10)
    for index, (frame, entry) in enumerate(zip(base, frames_map)):
        now = float(entry["simulation_time_s"])
        current = None
        for span in spans:
            if span["start_s"] - 1e-6 <= now <= span["end_s"] + 1e-6:
                current = {**span, "done": now >= span["end_s"] - 1e-6}
        if current is None:
            past = [s for s in spans if s["end_s"] <= now]
            if past:
                current = {**past[-1], "done": True}
        monitor: dict = {}
        for v in trail:
            if v["time_s"] <= now + 1e-9:
                monitor[v["conditional"]] = v
                monitor["cameras"] = v.get("cameras", {})
        for e in events:
            if e["event"] == "hold_lost" and e["time_s"] <= now + 1e-9 and (monitor.get("held") is None or monitor["held"]["time_s"] < e["time_s"]):
                monitor["held"] = {"decision": "fail", "reason": "hold_lost", "time_s": e["time_s"]}
        acquire_attempts = sum(1 for s in spans if s["leaf"] == "acquire" and s["start_s"] <= now + 1e-9)
        place_attempts = 1 + sum(1 for s in spans if s["leaf"] == "transport" and s["start_s"] <= now + 1e-9 and s["verdict"] == "failure") if spans else 0
        active = "none"
        for d in disturbance:
            start = d["start_s"]
            end = start + (d.get("pulled_for_s") or d.get("duration_s") or 0.0)
            if start <= now <= end + 1e-9:
                active = f"{d['kind'].upper()} ACTING" + (f": pull {d.get('peak_pull_n', 0):.1f} N" if d["kind"] == "slip" else f": push {np.linalg.norm(d['force_n']):.2f} N" if d["kind"] == "displaced" else ": occluder between camera and fixtures")
            elif now > end:
                active = f"{d['kind']} applied at {start:.2f} s" + (f", object left the fingers at {d['exit_speed_mps']:.2f} m/s" if d.get("exit_speed_mps") else "")
        labels = {"held": labeler.held(effector, now, 0.5).value, "stably_placed": labeler.stably_placed(now, 2.0).value, "area_clear": labeler.area_clear(now, 0.5).value}
        records.append({"frame": index, "simulation_time_s": now, "leaf": None if current is None else {k: current[k] for k in ("leaf", "attempt", "done")},
                        "monitor": {k: {"decision": v["decision"], "reason": v["reason"], "time_s": v["time_s"]} for k, v in monitor.items() if k != "cameras"}, "disturbance": active, "labels": labels})
        args = dict(title=case["title"], time_s=now, leaf=current, monitor=monitor, disturbance=active, labels=labels, attempts={"acquire": acquire_attempts, "place": max(1, place_attempts)}, outcome=verdict)
        annotated.append(annotate(frame, preview=None, **args))
        if index in preview_indices:
            previews.append(annotate(frame, preview=f"GIF SUMMARY | approximately {speed:.1f}x speed | full video: episode.mp4", **args))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".d10-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        video = staging / "episode.mp4"
        encode_frames(ffmpeg, video, annotated, width, height + STRIP)
        encoded = probe(ffprobe, video)
        if int(encoded["nb_read_frames"]) != len(annotated) or abs(float(encoded["duration"]) - len(annotated) / FPS) > 0.01:
            raise RuntimeError("the annotated video lost frames or changed its playback duration")
        previews[0].save(staging / "preview.gif", save_all=True, append_images=previews[1:], duration=100, loop=0, optimize=True)
        annotated[0].save(staging / "initial.png")
        annotated[-1].save(staging / "final.png")
        payloads = {name: (staging / name).read_bytes() for name in ("episode.mp4", "preview.gif", "initial.png", "final.png")}
        payloads["frames.json"] = (media / "frames.json").read_bytes()
        payloads["trail.json"] = json_bytes(records)
        payloads["encoding.json"] = json_bytes({"probe": encoded, "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]})
        metadata = {"schema": "rigby.presentation-overlay/1", "source_media_sha256": hashlib.sha256((media / "manifest.json").read_bytes()).hexdigest(),
                    "source_bundle_sha256": manifest["metadata"]["source_bundle_sha256"], "source_trace_sha256": manifest["metadata"]["source_trace_sha256"],
                    "robot_id": manifest["metadata"]["robot_id"], "outcome": manifest["metadata"]["outcome"], "full_episode": True, "frame_count": len(annotated), "fps": FPS,
                    "simulation_duration_s": manifest["metadata"]["simulation_duration_s"], "playback_duration_s": len(annotated) / FPS, "preview_is_summary": True,
                    "refusal_slate": manifest["metadata"].get("refusal_slate", False), "physics_replay_performed_by_renderer": False,
                    "overlay": {"kind": "skill_trail", "strip_px": STRIP, "shows": ["leaf and attempt", "monitor verdicts with sensors and ages", "protocol disturbance", "oracle labels"]},
                    "case": case["name"], "skill_verdict": verdict, "recovered": bool(outcome.get("skill_success")) and case["class"] != "nominal"}
        digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, "frames": len(annotated), "simulation_duration_s": metadata["simulation_duration_s"], "skill_verdict": verdict}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a fresh destination")
    args.out.mkdir(parents=True)
    rows = json.loads((args.campaign / "trials.json").read_bytes())
    corpus = corpus_module.load_registration()
    index = {"goal": "G10", "demo": "D10", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
             "campaign_sha256": hashlib.sha256((args.campaign / "summary.json").read_bytes()).hexdigest(), "generation_calls": 0, "cases": []}
    # -- the three bodies, nominal, on one clock ------------------------------------
    clips = []
    nominal_rows = []
    for body in corpus["bodies"]:
        row = next(r for r in rows if r["zoo_id"] == body and r["class"] == "nominal" and "bundle" in r)
        clips.append(args.campaign / body / "nominal" / f"seed-{row['seed']:03d}" / "media" / "episode.mp4")
        nominal_rows.append(row)
    tiled = tile(ffmpeg, ffprobe, clips, args.out / "three-body-nominal.mp4")
    tiled["preview"] = gif_summary(ffmpeg, args.out / "three-body-nominal.mp4", args.out / "three-body-nominal-preview.gif", "D10 three-body fixed-world transfer", tiled["duration_s"])
    index["three_body"] = {**tiled, "episodes": [{"episode_id": r["episode_id"], "verdict": r["verdict"], "physics_s": r["physics_s"], "media_sha256": r["media_sha256"]} for r in nominal_rows]}
    print(json.dumps({"three_body": tiled["frames"]}), flush=True)
    # -- recoveries -------------------------------------------------------------------------
    for kind in ("slip", "occlusion"):
        for body in corpus["bodies"]:
            subset = [r for r in rows if r["zoo_id"] == body and r["class"] == kind and "bundle" in r]
            row = next((r for r in subset if r["skill_success"]), subset[0]) if subset else None
            if row is None:
                continue
            name = f"{kind}-{body}"
            physical = args.local / body / kind / f"seed-{row['seed']:03d}" / "physical"
            media = args.campaign / body / kind / f"seed-{row['seed']:03d}" / "media"
            case = {"name": name, "class": kind, "body": body, "seed": row["seed"], "episode_id": row["episode_id"],
                    "title": f"D10 {kind.upper()} RECOVERY | {body} seed {row['seed']}" if row["skill_success"] else f"D10 {kind.upper()}, NOT RECOVERED | {body} seed {row['seed']}"}
            result = overlay(case, physical, media, args.out / name / "overlay", ffmpeg, ffprobe)
            shutil.copyfile(args.out / name / "overlay" / "episode.mp4", args.out / f"{name}.mp4")
            shutil.copyfile(args.out / name / "overlay" / "preview.gif", args.out / f"{name}-preview.gif")
            shutil.copyfile(args.out / name / "overlay" / "frames.json", args.out / f"{name}-frames.json")
            index["cases"].append({**case, "verdict": row["verdict"], "root_reason": row["root_reason"], "recovered": row["recovered"], "false_completion": row["false_completion"],
                                   "physics_s": row["physics_s"], "attempts": row["attempts"], "disturbance": row["disturbance"], "physical_sha256": row["bundle"]["sha256"], "media_sha256": row["media_sha256"],
                                   "overlay": result, "video": f"{name}.mp4", "preview": f"{name}-preview.gif", "frames": f"{name}-frames.json"})
            print(json.dumps({"case": name, "verdict": row["verdict"], "frames": result["frames"]}), flush=True)
    (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
