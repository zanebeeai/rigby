"""D04: 'a little' against 'as far as you can' against '5 cm', on three bodies.

Nine sealed episodes, one per body and prompt: each prompt read once, body-
neutrally, into the same program on every body (the reading's hash is on
every frame), then grounded by that body's own measured reach and executed
on native physics with the recorder. The banner carries the semantic fields
the reading produced -- the entry, the remove and the stated quantity, if
any -- and the effector's measured travel from where it started, updated on
every frame from the recorded state. 'A little' and 'as far as you can' come
out as different metres on different bodies under one reading; '5 cm' comes
out as five centimetres on every body that can reach it.

Everything is rendered from recorded physical states at real-time playback;
the nine clips are tiled on one clock and a shorter clip holds its final
state until the longest ends. No physics is advanced by the renderer.

    python any-robot/scripts/g04_d04_media.py --out docs/results/g04-d04 --local any-robot/results/g04-d04 [--model gpt-5-nano]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
from rigby_general.evidence.composition import capture_prompt
from rigby_general.evidence.render import HEIGHT, PREVIEW_LIMIT, WIDTH, _camera, frame_schedule
from rigby_general.pipeline import ingest_robot
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.planner.model_planner import Budget, CallLog, ModelSchemaPlanner, OpenAITransport, ResponseCache
from rigby_general.schema.inventory import load_inventory


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FIXTURE = ROOT / "assets/general/research-protocols/g04-semantics-v1"
BODIES = ("zoo_compact_arm", "zoo_jaw_arm", "zoo_long_arm")
PROMPTS = (("little", "reach out a little"), ("edge", "reach out as far as you can"), ("five_cm", "reach out 5 cm"))
FPS = 12
BANNER = 112
BACKGROUND = (17, 24, 39)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def probe(ffprobe: str, video: Path) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300)
    return json.loads(out)["streams"][0]


def encode_frames(ffmpeg: str, video: Path, frames: list[Image.Image], width: int, height: int, *, attempts: int = 3) -> None:
    last = None
    for _ in range(attempts):
        process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "pipe:0",
                                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for image in frames:
                process.stdin.write(image.tobytes())
            process.stdin.close()
            if process.wait(timeout=600) == 0:
                return
            last = process.stderr.read().decode(errors="replace")[-500:]
        except (BrokenPipeError, OSError) as error:
            try:
                process.kill()
                process.wait(timeout=30)
            except Exception:
                pass
            last = repr(error)
    raise RuntimeError(f"ffmpeg failed after {attempts} attempts: {last}")


def tile(ffmpeg: str, ffprobe: str, clips: list[Path], destination: Path, *, columns: int) -> dict:
    streams = [probe(ffprobe, clip) for clip in clips]
    longest = max(float(s["duration"]) for s in streams)
    inputs, filters = [], []
    for index, (clip, stream) in enumerate(zip(clips, streams)):
        inputs += ["-i", str(clip)]
        filters.append(f"[{index}:v]tpad=stop_mode=clone:stop_duration={max(0.0, longest - float(stream['duration'])):.3f},scale=iw/2:ih/2[v{index}]")
    layout = []
    for index in range(len(clips)):
        row, col = divmod(index, columns)
        x = "0" if col == 0 else "+".join(f"w{c}" for c in range(col))
        y = "0" if row == 0 else "+".join(f"h{r * columns}" for r in range(row))
        layout.append(f"{x}_{y}")
    filters.append("".join(f"[v{i}]" for i in range(len(clips))) + f"xstack=inputs={len(clips)}:layout={'|'.join(layout)}[out]")
    command = [ffmpeg, "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-r", str(FPS), "-c:v", "libx264",
               "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    for _ in range(3):
        if subprocess.run(command, timeout=1200).returncode == 0:
            break
    else:
        raise RuntimeError("ffmpeg could not tile the clips")
    result = probe(ffprobe, destination)
    return {"video": destination.name, "frames": int(result["nb_read_frames"]), "duration_s": float(result["duration"]), "width": int(result["width"]), "height": int(result["height"])}


def gif_summary(ffmpeg: str, video: Path, destination: Path, title: str, real_duration_s: float) -> dict:
    with tempfile.TemporaryDirectory(prefix=".d04-gif-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        rate = max(0.25, PREVIEW_LIMIT / max(real_duration_s, 1e-6))
        subprocess.check_call([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(video), "-vf", f"fps={rate:.4f},scale=960:-2", str(staging / "f%04d.png")], timeout=600)
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


def semantic_fields(root: Path, inventory) -> dict:
    """The reading the pipeline made of this prompt, from the sealed task and trace."""

    task = json.loads((root / "task.json").read_bytes())
    execution = json.loads((root / "execution.json").read_bytes())
    trace = execution["pipeline"]
    program = task.get("requested_semantics")
    fields = {"prompt": task["prompt"], "hash": trace.get("role_normalized_hash"), "planner": trace.get("planner"), "quantities": trace.get("requested_quantities", []), "segments": []}
    if program:
        by_binding = {entry.binding_key: entry.entry_id for entry in inventory.entries}
        for segment in program["segments"]:
            schema = segment["motion_schema"]
            if schema["kind"] == "path":
                key = f"path:{schema['vector']}.{schema['conformation']}.{schema['deixis']}.{schema['contour']}"
            else:
                key = f"stative:{schema['stative']}"
            binding = f"{key}|{segment['figure']['role']}->{segment['ground']['role']}"
            fields["segments"].append({"segment_id": segment["segment_id"], "entry_id": by_binding.get(binding, binding), "remove": segment["region"]["remove"],
                                       "manner": {k: v for k, v in segment["manner"].items() if v}})
    refusal = json.loads((root / "outcome.json").read_bytes()).get("refusal")
    fields["refusal"] = refusal
    return fields


def label(frame: Image.Image, *, body: str, status: str, time_s: float, fields: dict, travel_m: float, final_travel_m: float, reach_m: float, preview: str | None, phase: str = "") -> Image.Image:
    canvas = Image.new("RGB", (2 * WIDTH, HEIGHT + BANNER), BACKGROUND)
    canvas.paste(frame, (0, BANNER))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=12)
    tint = (100, 230, 165) if status == "SUCCESS" else (255, 189, 100)
    draw.text((12, 6), f"{body} | {status} | sim t={time_s:.3f}s | {phase}", fill=tint, font=font)
    if fields["segments"]:
        segment = fields["segments"][0]
        quantity = fields["quantities"][0] if fields["quantities"] else None
        stated = f"stated '{quantity['text']}' = {quantity['value'] * 100:.1f} cm" if quantity else "no stated quantity"
        reading = f"'{fields['prompt']}' read as {segment['entry_id']} remove={segment['remove']} | {stated}"
    else:
        reading = f"'{fields['prompt']}' | {(fields.get('refusal') or {}).get('code', 'refused')}: {(fields.get('refusal') or {}).get('detail', '')}"[:110]
    draw.text((12, 28), reading, fill="white", font=font)
    measured = f"effector travel {travel_m * 100:.1f} cm now, {final_travel_m * 100:.1f} cm at the terminus | reach {reach_m:.2f} m ({final_travel_m / max(reach_m, 1e-9) * 100:.0f}% of reach) | reading {(fields['hash'] or '')[:10]}"
    draw.text((12, 50), measured, fill=(191, 210, 233), font=font)
    detail = preview or "FULL EPISODE | real-time playback | recorded physical states | travel measured from the recorded state on every frame"
    draw.text((12, 72), detail, fill=(170, 185, 205), font=small)
    draw.text((12, 92), "Global view", fill=(170, 185, 205), font=small)
    draw.text((WIDTH + 12, 92), "Task view | computed torque | model + joint encoders", fill=(170, 185, 205), font=small)
    return canvas


def render_measured(root: Path, destination: Path, *, body: str, fields: dict, ffmpeg: str, ffprobe: str) -> dict:
    """Render one sealed episode with the reading and the measured travel on every frame."""

    manifest = verify_bundle(root, None)
    metadata = dict(manifest["metadata"])
    record = PhysicsRecord.from_bytes((root / "trace.npz").read_bytes())
    if record.content_hash() != metadata["trace_sha256"]:
        raise ValueError("physical array hash disagrees with the evidence manifest")
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    model.vis.headlight.ambient[:] = 0.7
    model.vis.headlight.diffuse[:] = 0.8
    model.vis.headlight.specular[:] = 0.2
    data = mujoco.MjData(model)
    site = metadata.get("task_site")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) if site else -1
    times = record.arrays["time_s"]
    states = record.arrays["state"]
    # travel of the figure site from the first recorded state, per recorded sample
    positions = np.zeros((len(states), 3))
    for index in range(len(states)):
        mujoco.mj_setState(model, data, states[index], STATE_SPEC)
        mujoco.mj_forward(model, data)
        positions[index] = data.site_xpos[site_id] if site_id >= 0 else 0.0
    travel = np.linalg.norm(positions - positions[0], axis=1)
    # Every grounded program ends with a recovery phase that retraces to rest,
    # so the travel that answers the prompt is the travel at the terminus of
    # the last action phase, not at the end of the recording.
    task = json.loads((root / "task.json").read_bytes())
    phases = (task.get("grounded_program") or {}).get("phases", [])
    action_end_s = max((float(phase["end_s"]) for phase in phases if phase["kind"] == "action"), default=float(times[-1]))
    terminus = int(np.clip(np.searchsorted(times, action_end_s, side="right") - 1, 0, len(times) - 1))
    final_travel = float(travel[terminus])
    farthest = float(travel.max())
    reach = float(metadata["camera"]["reach"])
    status = metadata["outcome"].upper().replace("_", " ")
    indices, video_times = frame_schedule(times, FPS)
    preview_indices = set(np.linspace(0, len(indices) - 1, min(PREVIEW_LIMIT, len(indices))).astype(int).tolist())
    speed = (len(indices) / FPS) / (len(preview_indices) / 10)
    frames, previews, frame_map = [], [], []
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for frame_number, index in enumerate(indices):
            mujoco.mj_setState(model, data, states[index], STATE_SPEC)
            mujoco.mj_forward(model, data)
            panels = []
            for task in (False, True):
                camera = _camera(**metadata["camera"], task=task)
                if task and site_id >= 0:
                    camera.lookat[:] = data.site_xpos[site_id]
                renderer.update_scene(data, camera=camera)
                panels.append(np.asarray(renderer.render()).copy())
            raw = Image.fromarray(np.concatenate(panels, axis=1))
            time_s = float(times[index])
            phase = "recovery: retracing to rest" if index > terminus else f"action: reaching, terminus at t={action_end_s:.2f}s"
            frames.append(label(raw, body=body, status=status, time_s=time_s, fields=fields, travel_m=float(travel[index]), final_travel_m=final_travel, reach_m=reach, preview=None, phase=phase))
            if frame_number in preview_indices:
                previews.append(label(raw, body=body, status=status, time_s=time_s, fields=fields, travel_m=float(travel[index]), final_travel_m=final_travel, reach_m=reach,
                                      preview=f"GIF SUMMARY | approximately {speed:.1f}x speed | full video: episode.mp4", phase=phase))
            frame_map.append({"frame": frame_number, "playback_time_s": frame_number / FPS, "requested_simulation_time_s": float(video_times[frame_number]), "recorded_sample": int(index),
                              "simulation_time_s": time_s, "effector_travel_m": round(float(travel[index]), 6), "phase": "recovery" if index > terminus else "action"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".d04-render-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        video = staging / "episode.mp4"
        encode_frames(ffmpeg, video, frames, 2 * WIDTH, HEIGHT + BANNER)
        stream = probe(ffprobe, video)
        if int(stream["nb_read_frames"]) != len(indices) or abs(float(stream["duration"]) - len(indices) / FPS) > 0.01:
            raise RuntimeError("encoded video lost frames or changed the required playback duration")
        frames[0].save(staging / "initial.png")
        frames[-1].save(staging / "final.png")
        previews[0].save(staging / "preview.gif", save_all=True, append_images=previews[1:], duration=100, loop=0, optimize=True)
        measurements = {
            "body": body, "prompt": fields["prompt"], "reading": fields, "task_site": site, "reach_radius_m": reach,
            "start_position_m": positions[0].round(6).tolist(), "final_position_m": positions[-1].round(6).tolist(),
            "final_travel_m": round(final_travel, 6), "terminus_time_s": float(times[terminus]), "action_end_s": action_end_s, "travel_at_end_of_recording_m": round(float(travel[-1]), 6),
            "farthest_travel_m": round(farthest, 6), "final_travel_fraction_of_reach": round(final_travel / max(reach, 1e-9), 6),
            "simulation_duration_s": float(times[-1] - times[0]), "outcome": metadata["outcome"],
        }
        payloads = {name: (staging / name).read_bytes() for name in ("episode.mp4", "preview.gif", "initial.png", "final.png")}
        payloads["frames.json"] = json_bytes(frame_map)
        payloads["measurements.json"] = json_bytes(measurements)
        payloads["encoding.json"] = json_bytes({"probe": stream, "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]})
        media_metadata = {
            "schema": "rigby.presentation/1", "source_bundle_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            "source_trace_sha256": metadata["trace_sha256"], "robot_id": body, "outcome": metadata["outcome"], "full_episode": True,
            "frame_count": len(indices), "fps": FPS, "simulation_duration_s": float(times[-1] - times[0]), "playback_duration_s": len(indices) / FPS,
            "preview_is_summary": True, "refusal_slate": len(times) == 1, "physics_replay_performed_by_renderer": False,
            "overlay": {"kind": "semantic_fields_and_measured_travel", "shows": ["entry and remove of the reading", "stated quantity or its absence", "effector travel from the recorded state", "reach fraction", "reading hash"]},
            "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if destination.exists():
            shutil.rmtree(destination)
        digest = write_bundle(destination, payloads, media_metadata)
    return {"media": destination.as_posix(), "sha256": digest, "measurements": measurements, "frames": len(indices)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--model", default=None, help="read the prompts through the model planner (cached replies cost nothing); default: the offline recognizer")
    parser.add_argument("--env", type=Path, default=None)
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    load_env(args.env or (REPO / ".env" if (REPO / ".env").exists() else REPO.parent / "rigby" / ".env"))
    inventory = load_inventory()
    registration = json.loads((FIXTURE / "registration.json").read_bytes())
    log = CallLog(args.out / "calls.jsonl")
    if args.model:
        planner = ModelSchemaPlanner(inventory, model=args.model, transport=OpenAITransport(), cache=ResponseCache(FIXTURE / "cache"), log=log, budget=Budget(), purpose="d04")
    else:
        planner = OfflineSchemaPlanner(inventory)
    args.out.mkdir(parents=True, exist_ok=True)
    args.local.mkdir(parents=True, exist_ok=True)
    clips, index_rows = [], []
    for body in BODIES:
        source = ROOT / "assets/general/zoo" / body / "robot.urdf"
        robot = ingest_robot(source, robot_id=body)
        for slug, prompt in PROMPTS:
            physical = args.local / body / slug / "physical"
            if physical.exists():
                shutil.rmtree(physical)
            captured = capture_prompt(robot, prompt, physical, label=body, source_urdf=source, goal="G04",
                                      protocol_reference={"fixture": "g04-semantics-v1", "registration_sha256": registration["registration_sha256"], "demo": "D04"},
                                      caption=prompt, planner=planner)
            fields = semantic_fields(physical, inventory)
            media = args.out / body / slug / "media"
            rendered = render_measured(physical, media, body=body, fields=fields, ffmpeg=ffmpeg, ffprobe=ffprobe)
            clips.append(media / "episode.mp4")
            index_rows.append({"body": body, "slug": slug, "prompt": prompt, "outcome": captured["outcome"], "physical_bundle": physical.as_posix(), "physical_sha256": captured["sha256"],
                               "media": media.relative_to(args.out).as_posix(), "media_sha256": rendered["sha256"], "video": (media / "episode.mp4").relative_to(args.out).as_posix(),
                               "preview": (media / "preview.gif").relative_to(args.out).as_posix(), "frames": (media / "frames.json").relative_to(args.out).as_posix(),
                               "reading": fields, "measurements": {k: v for k, v in rendered["measurements"].items() if k != "reading"}})
            print(f"{body} | {prompt}: {captured['outcome']}; travel {rendered['measurements']['final_travel_m']:.4f} m of reach {rendered['measurements']['reach_radius_m']:.3f} m", flush=True)
    tiled = tile(ffmpeg, ffprobe, clips, args.out / "nine-way-synchronized.mp4", columns=3)
    summary = gif_summary(ffmpeg, args.out / "nine-way-synchronized.mp4", args.out / "nine-way-preview.gif", "D04: rows compact, jaw, long arm; columns a little, as far as you can, 5 cm", tiled["duration_s"])
    tile_frames = []
    per_clip = [json.loads((args.out / row["frames"]).read_bytes()) for row in index_rows]
    for frame in range(tiled["frames"]):
        tile_frames.append({"frame": frame, "playback_time_s": frame / FPS, "clips": {
            f"{row['body']}/{row['slug']}": (maps[frame] if frame < len(maps) else {"held": True, "recorded_sample": maps[-1]["recorded_sample"], "simulation_time_s": maps[-1]["simulation_time_s"]})
            for row, maps in zip(index_rows, per_clip)}})
    (args.out / "nine-way-frames.json").write_bytes(json_bytes(tile_frames))
    by_prompt = {}
    for row in index_rows:
        by_prompt.setdefault(row["slug"], {"prompt": row["prompt"], "reading_hashes": set(), "travel_m": {}})
        by_prompt[row["slug"]]["reading_hashes"].add(row["reading"]["hash"])
        by_prompt[row["slug"]]["travel_m"][row["body"]] = row["measurements"]["final_travel_m"]
    comparison = {slug: {"prompt": v["prompt"], "one_reading_on_every_body": len(v["reading_hashes"]) == 1, "reading_hash": sorted(h for h in v["reading_hashes"] if h)[:1], "final_travel_m": v["travel_m"]} for slug, v in by_prompt.items()}
    index = {
        "goal": "G04", "demo": "D04", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "fixture_registration": registration["registration_sha256"], "planner": ("model:" + args.model) if args.model else "offline-recognizer-v1",
        "bodies": list(BODIES), "prompts": [p for _, p in PROMPTS], "clips": index_rows, "comparison": comparison,
        "tile": {**tiled, "preview": summary, "frames": "nine-way-frames.json", "layout": "rows: compact arm, jaw arm, long arm; columns: a little, as far as you can, 5 cm; each clip at half size; a shorter clip holds its final state"},
        "calls": log.totals(),
    }
    (args.out / "index.json").write_bytes(json_bytes(index))
    print(json.dumps(comparison, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
