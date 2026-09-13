"""D15: uninterrupted 3-, 5- and 10-object clearances with the live tree, and a disturbance with and without recovery.

Each clearance video is one episode's chain of sealed segments rendered
from recorded states at real-time playback on the one clock the executor
ran on, with the tree's active path -- root to leaf, with each loop's
attempt count -- drawn on every frame from the execution record's node
timings, the object and cell in hand, and the placed count from the
verdicts so far. The disturbance example puts the same seed, the same
world and the same disturbance through the generated tree and through its
flat twin side by side on one clock.

    python any-robot/scripts/g15_d15_media.py --campaign docs/results/g15-clearance --local any-robot/results/g15-clearance --out docs/results/g15-d15 --body zoo_jaw_arm
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

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rigby_core.evidence import verify_bundle, write_bundle
from rigby_core.simulation.recording import PhysicsRecord, STATE_SPEC

from rigby_general.evidence.capture import json_bytes
from rigby_general.evidence.render import HEIGHT, PREVIEW_LIMIT, WIDTH, _camera, frame_schedule

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g15_protocol as protocol  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FPS = 12
BANNER = 88
STRIP = 160
BACKGROUND = (17, 24, 39)


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def active_path(record: dict, time_s: float) -> list[dict]:
    """The nodes whose span contains ``time_s``, root first, deepest last."""

    path = []

    def walk(node, depth):
        if node["started_s"] - 1e-6 <= time_s <= node["ended_s"] + 1e-6 or (node["started_s"] - 1e-6 <= time_s and node["ended_s"] == node["started_s"]):
            path.append({"skill_id": node["skill_id"], "node_id": node["node_id"], "attempts": node["attempts"], "verdict": node["verdict"], "arguments": {}, "depth": depth})
            candidates = [c for c in node.get("children", []) if c["started_s"] - 1e-6 <= time_s <= c["ended_s"] + 1e-6]
            if candidates:
                walk(candidates[-1], depth + 1)
                return
            for recovery in node.get("recoveries", []):
                if recovery["started_s"] - 1e-6 <= time_s <= recovery["ended_s"] + 1e-6:
                    walk(recovery, depth + 1)
                    return

    walk(record["root"], 1)
    return path


def placed_so_far(record: dict, time_s: float) -> int:
    count = 0
    for node in _walk(record["root"]):
        if node["skill_id"] == "verify_placement" and node["verdict"] == "success" and node["ended_s"] <= time_s + 1e-6:
            count += 1
    return count


def _walk(node):
    yield node
    for child in node.get("children", []):
        yield from _walk(child)
    for recovery in node.get("recoveries", []):
        yield from _walk(recovery)


def arguments_of(tree: dict, node_id: str) -> dict:
    for node in _walk_tree(tree["root"]):
        if node["node_id"] == node_id:
            return node.get("arguments", {})
    return {}


def _walk_tree(node):
    yield node
    for child in node.get("children", []):
        yield from _walk_tree(child)
    if node.get("recovery"):
        yield from _walk_tree(node["recovery"])


def label(frame: Image.Image, *, title: str, status: str, time_s: float, path: list[dict], tree: dict, placed: int, total: int, segment: str, preview: str | None, disturbance: str | None) -> Image.Image:
    canvas = Image.new("RGB", (2 * WIDTH, HEIGHT + BANNER + STRIP), BACKGROUND)
    canvas.paste(frame, (0, BANNER + STRIP))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=12)
    tint = (100, 230, 165) if status == "SUCCESS" else (255, 189, 100)
    draw.text((12, 6), f"{title} | {status} | sim t={time_s:.3f}s | placed {placed}/{total} | segment {segment}", fill=tint, font=font)
    draw.text((12, 28), preview or "FULL EPISODE | one clock across every segment | recorded physical states | live tree from the execution record", fill=(191, 210, 233), font=font)
    if disturbance:
        draw.text((12, 50), disturbance, fill=(255, 140, 140), font=font)
    draw.text((12, 70), "Global view", fill=(170, 185, 205), font=small)
    draw.text((WIDTH + 12, 70), "Task view | computed torque | model + joint encoders + cameras + contact", fill=(170, 185, 205), font=small)
    y = BANNER + 4
    shown = path if len(path) <= 10 else [path[0], None, *path[-8:]]
    for node in shown:
        if node is None:
            draw.text((24, y), "...", fill=(150, 165, 190), font=small)
            y += 15
            continue
        arguments = arguments_of(tree, node["node_id"])
        bound = " ".join(f"{k}={v}" for k, v in arguments.items() if k in ("object", "destination"))
        text = f"{'  ' * (node['depth'] - 1)}{node['skill_id']}" + (f" ({bound})" if bound else "") + (f" attempt {node['attempts']}" if node["attempts"] > 1 else "")
        draw.text((12, y), text[:110], fill=(230, 230, 240) if node is path[-1] else (150, 165, 190), font=small)
        y += 15
    return canvas


def render_episode(segments: list[dict], destination: Path, *, title: str, ffmpeg: str, ffprobe: str, total: int, disturbance: str | None = None) -> dict:
    """Every segment of one episode, in order, into one video on the chain's clock."""

    if destination.exists():
        shutil.rmtree(destination)
    frames_map = []
    previews = []
    frame_number = 0
    total_frames = 0
    durations = []
    for seg in segments:
        manifest = verify_bundle(Path(seg["bundle"]), seg["sha256"])
        record = PhysicsRecord.from_bytes((Path(seg["bundle"]) / "trace.npz").read_bytes())
        durations.append(len(frame_schedule(record.arrays["time_s"], FPS)[0]))
    total_frames = sum(durations)
    preview_at = set(np.linspace(0, total_frames - 1, min(PREVIEW_LIMIT, total_frames)).astype(int).tolist())
    speed = (total_frames / FPS) / (len(preview_at) / 10)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".d15-", dir=destination) as temporary:
        staging = Path(temporary)
        video = staging / "episode.mp4"
        process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * WIDTH}x{HEIGHT + BANNER + STRIP}", "-r", str(FPS), "-i", "pipe:0",
                                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for seg in segments:
                root = Path(seg["bundle"])
                manifest = verify_bundle(root, seg["sha256"])
                metadata = dict(manifest["metadata"])
                execution = json.loads((root / "execution.json").read_bytes())
                tree = json.loads((root / "tree.json").read_bytes())
                record = PhysicsRecord.from_bytes((root / "trace.npz").read_bytes())
                model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
                model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
                model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
                model.vis.headlight.ambient[:] = 0.7
                model.vis.headlight.diffuse[:] = 0.8
                model.vis.headlight.specular[:] = 0.2
                data = mujoco.MjData(model)
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "scene_block_center")
                indices, video_times = frame_schedule(record.arrays["time_s"], FPS)
                status = metadata["outcome"].upper().replace("_", " ")
                with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
                    for index in indices:
                        mujoco.mj_setState(model, data, record.arrays["state"][index], STATE_SPEC)
                        mujoco.mj_forward(model, data)
                        panels = []
                        for task in (False, True):
                            camera = _camera(**metadata["camera"], task=task)
                            if task and site >= 0:
                                camera.lookat[:] = data.site_xpos[site]
                            renderer.update_scene(data, camera=camera)
                            panels.append(np.asarray(renderer.render()).copy())
                        raw = Image.fromarray(np.concatenate(panels, axis=1))
                        time_s = float(record.arrays["time_s"][index])
                        path = active_path(execution["record"], time_s)
                        placed = placed_so_far(execution["record"], time_s)
                        full = label(raw, title=title, status=status, time_s=time_s, path=path, tree=tree, placed=placed, total=total, segment=f"{seg['segment']} {seg['object']}", preview=None, disturbance=disturbance)
                        process.stdin.write(full.tobytes())
                        if frame_number in preview_at:
                            previews.append(label(raw, title=title, status=status, time_s=time_s, path=path, tree=tree, placed=placed, total=total, segment=f"{seg['segment']} {seg['object']}",
                                                  preview=f"GIF SUMMARY | approximately {speed:.1f}x speed | full video: episode.mp4", disturbance=disturbance))
                        frames_map.append({"frame": frame_number, "playback_time_s": frame_number / FPS, "segment": seg["segment"], "object": seg["object"], "recorded_sample": int(index), "simulation_time_s": time_s,
                                           "active_path": [n["skill_id"] for n in path], "placed": placed})
                        frame_number += 1
            process.stdin.close()
            if process.wait(timeout=1200) != 0:
                raise RuntimeError("ffmpeg failed: " + process.stderr.read().decode(errors="replace")[-1500:])
        finally:
            if process.poll() is None:
                process.kill()
        stream = probe(ffprobe, video)
        if int(stream["nb_read_frames"]) != frame_number:
            raise RuntimeError("the encoded video lost frames")
        previews[0].save(staging / "preview.gif", save_all=True, append_images=previews[1:], duration=100, loop=0, optimize=True)
        payloads = {"episode.mp4": video.read_bytes(), "preview.gif": (staging / "preview.gif").read_bytes(), "frames.json": json_bytes(frames_map),
                    "encoding.json": json_bytes({"probe": stream, "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]})}
        media_metadata = {"schema": "rigby.presentation/1", "source_segments": [{"segment": s["segment"], "object": s["object"], "sha256": s["sha256"], "trace_sha256": s["trace_sha256"]} for s in segments],
                          "full_episode": True, "frame_count": frame_number, "fps": FPS, "playback_duration_s": frame_number / FPS, "preview_is_summary": True, "physics_replay_performed_by_renderer": False,
                          "overlay": {"kind": "live_tree", "shows": ["active path root to leaf with bound object and cell", "loop attempt counts", "placed count from verdicts so far", "segment and object"]},
                          "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        digest = write_bundle(destination / "media", payloads, media_metadata)
    return {"media": (destination / "media").as_posix(), "sha256": digest, "frames": frame_number, "playback_s": frame_number / FPS}


def tile_pair(ffmpeg: str, ffprobe: str, left: Path, right: Path, destination: Path) -> dict:
    streams = [probe(ffprobe, left), probe(ffprobe, right)]
    longest = max(float(s["duration"]) for s in streams)
    filters = [f"[{i}:v]tpad=stop_mode=clone:stop_duration={max(0.0, longest - float(s['duration'])):.3f},scale=trunc(iw/2/2)*2:trunc(ih/2/2)*2[v{i}]" for i, s in enumerate(streams)]
    filters.append("[v0][v1]xstack=inputs=2:layout=0_0|w0_0[out]")
    command = [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(left), "-i", str(right), "-filter_complex", ";".join(filters), "-map", "[out]", "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    if subprocess.run(command, timeout=1200).returncode != 0:
        raise RuntimeError("ffmpeg could not tile the pair")
    out = probe(ffprobe, destination)
    return {"video": destination.name, "frame_count": int(out["nb_read_frames"]), "duration_s": float(out["duration"])}


def gif_summary(ffmpeg: str, video: Path, destination: Path, title: str, real_duration_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=".d15-gif-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        rate = max(0.25, PREVIEW_LIMIT / max(real_duration_s, 1e-6))
        subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(video), "-vf", f"fps={rate:.4f},scale=1200:-2", str(staging / "f%04d.png")], timeout=600)
        frames = sorted(staging.glob("f*.png"))[:PREVIEW_LIMIT]
        speed = real_duration_s / (len(frames) / 10.0)
        font = ImageFont.load_default(size=15)
        images = []
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            canvas = Image.new("RGB", (image.width, image.height + 26), BACKGROUND)
            canvas.paste(image, (0, 26))
            ImageDraw.Draw(canvas).text((10, 5), f"{title} | GIF SUMMARY | approximately {speed:.1f}x speed | full video: {video.name}"[:150], fill=(255, 189, 100), font=font)
            images.append(canvas)
        images[0].save(destination, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
    return {"gif": destination.name, "frames": len(images), "approximate_speed": round(speed, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--body", default="zoo_jaw_arm")
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    args.out.mkdir(parents=True, exist_ok=True)
    index = {"goal": "G15", "demo": "D15", "created_at_utc": datetime.now(timezone.utc).isoformat(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "body": args.body, "clearances": [], "generation_calls": 0}
    for count in protocol.CHAIN_LENGTHS:
        rows = json.loads((args.campaign / f"nominal-{args.body}-{count}" / "trials.json").read_bytes())
        row = next((r for r in rows if r["skill_success"] and "sealed" in r), None)
        if row is None:
            index["clearances"].append({"objects": count, "episode_id": None, "note": "no sealed successful episode"})
            continue
        rendered = render_episode(row["sealed"]["segments"], args.out / f"clearance-{count}", title=f"{args.body} | clear {count} objects | seed {row['seed']}", ffmpeg=ffmpeg, ffprobe=ffprobe, total=count)
        index["clearances"].append({"objects": count, "episode_id": row["episode_id"], "seed": row["seed"], "verdict": row["verdict"], "physics_s": row["physics_s"], "passes": row["passes"], "segments": len(row["sealed"]["segments"]),
                                    "video": f"clearance-{count}/media/episode.mp4", "preview": f"clearance-{count}/media/preview.gif", "frames": f"clearance-{count}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"]})
        print(json.dumps({"clearance": count, "episode": row["episode_id"], "frames": rendered["frames"]}), flush=True)
    # the disturbance example: one seed the tree recovered and the flat twin did not, else the first sealed pair
    rows = json.loads((args.campaign / f"disturbed-{args.body}" / "trials.json").read_bytes())
    by_seed: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["executor"]] = r
    chosen = next((pair for pair in by_seed.values() if pair.get("tree", {}).get("skill_success") and not pair.get("flat", {}).get("skill_success") and "sealed" in pair["tree"] and "sealed" in pair["flat"]), None)
    if chosen is None:
        chosen = next((pair for pair in by_seed.values() if "sealed" in pair.get("tree", {}) and "sealed" in pair.get("flat", {})), None)
    if chosen is not None:
        clips = {}
        for executor in ("tree", "flat"):
            r = chosen[executor]
            note = f"{r['disturbance']['kind']} on {r['disturbance']['object']} | {executor}: {r['verdict']}" + (f" ({r['root_reason']})" if r["root_reason"] else "")
            rendered = render_episode(r["sealed"]["segments"], args.out / f"disturbed-{executor}", title=f"{args.body} | 5 objects | {executor} | seed {r['seed']}", ffmpeg=ffmpeg, ffprobe=ffprobe, total=5, disturbance=note)
            clips[executor] = {"episode_id": r["episode_id"], "verdict": r["verdict"], "root_reason": r["root_reason"], "placed": r["placed_by_oracle"], "video": f"disturbed-{executor}/media/episode.mp4", "preview": f"disturbed-{executor}/media/preview.gif",
                               "frames": f"disturbed-{executor}/media/frames.json", "media_sha256": rendered["sha256"], "frame_count": rendered["frames"]}
        pair = tile_pair(ffmpeg, ffprobe, args.out / clips["tree"]["video"], args.out / clips["flat"]["video"], args.out / "disturbed-pair.mp4")
        summary = gif_summary(ffmpeg, args.out / "disturbed-pair.mp4", args.out / "disturbed-pair-preview.gif", f"D15 {chosen['tree']['disturbance']['kind']}: tree (left) against flat (right), seed {chosen['tree']['seed']}", pair["duration_s"])
        index["disturbance"] = {"seed": chosen["tree"]["seed"], "kind": chosen["tree"]["disturbance"]["kind"], "object": chosen["tree"]["disturbance"]["object"], "tree": clips["tree"], "flat": clips["flat"],
                                "pair": {**pair, "preview": summary, "layout": "left the generated tree, right its flat twin; the same seed, world and disturbance; a shorter clip holds its final state"}}
        print(json.dumps({"disturbance": index["disturbance"]["kind"], "seed": index["disturbance"]["seed"], "tree": clips["tree"]["verdict"], "flat": clips["flat"]["verdict"]}), flush=True)
    (args.out / "index.json").write_bytes(json_bytes(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
