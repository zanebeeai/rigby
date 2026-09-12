"""Render sealed physical states without advancing a physics episode."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rigby_core.evidence import verify_bundle, write_bundle
from rigby_core.simulation.recording import PhysicsRecord, STATE_SPEC

from .capture import json_bytes


WIDTH, HEIGHT, BANNER = 480, 320, 88
DEFAULT_FPS = 12
PREVIEW_LIMIT = 60


def frame_schedule(times: np.ndarray, fps: int = DEFAULT_FPS) -> tuple[np.ndarray, np.ndarray]:
    """Full real-time sampling, including the last state; never a frame cap.

    End-state hold is at most two frame periods. A zero-duration refusal gets
    a two-second initial-state slate and is explicitly labelled as unexecuted.
    """
    times = np.asarray(times, dtype=float)
    if not 1 <= fps <= 120 or len(times) == 0 or not np.isfinite(times).all():
        raise ValueError("Finite nonempty times and 1..120 FPS are required")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Simulation timestamps must strictly increase")
    duration = float(times[-1] - times[0])
    if duration == 0:
        return np.zeros(2 * fps, dtype=int), np.full(2 * fps, times[0])
    requested = np.minimum(times[0] + np.arange(math.ceil(duration * fps) + 1) / fps, times[-1])
    indices = np.maximum(0, np.searchsorted(times, requested, side="right") - 1)
    indices[-1] = len(times) - 1
    return indices, requested


def _camera(centre, reach: float, *, task: bool) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = centre
    camera.distance = max(0.2, reach) * (1.5 if task else 2.6)
    camera.azimuth = 55.0 if task else 135.0
    camera.elevation = -12.0 if task else -20.0
    return camera


def _label(frame: Image.Image, metadata: dict, time_s: float, *, preview: str | None = None) -> Image.Image:
    canvas = Image.new("RGB", (2 * WIDTH, HEIGHT + BANNER), (17, 24, 39))
    canvas.paste(frame, (0, BANNER))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    status = metadata["outcome"].upper().replace("_", " ")
    tint = (100, 230, 165) if status == "SUCCESS" else (255, 189, 100)
    draw.text((12, 7), f"{metadata['robot_id']} | {status} | sim t={time_s:.3f}s", fill=tint, font=font)
    qualifier = "ZERO ACTUATOR GAIN: INJECTED FAILURE" if metadata["fault"] else "Canonical reach/return | free-space baseline"
    if not metadata["fault"] and metadata.get("reference_clock_matches_physics") is False:
        qualifier = f"Reach/return | {metadata['reference_duration_s']:.2f}s reference / {metadata['simulation_duration_s']:.2f}s physics | CLOCKS DIFFER"
    draw.text((12, 29), qualifier, fill="white", font=font)
    detail = preview or "FULL EPISODE | real-time playback | recorded physical states"
    if status == "PRE EXECUTION REFUSAL":
        detail = "PRE-EXECUTION REFUSAL | initial state only | no motion was executed"
    draw.text((12, 51), detail, fill=(191, 210, 233), font=font)
    draw.text((12, 70), "Global view", fill=(170, 185, 205), font=ImageFont.load_default(size=12))
    draw.text((WIDTH + 12, 70), "Task view | computed torque | model + joint encoders", fill=(170, 185, 205), font=ImageFont.load_default(size=12))
    return canvas


def render_bundle(root: Path, destination: Path, *, expected_digest: str | None = None, fps: int = DEFAULT_FPS) -> dict:
    manifest = verify_bundle(root, expected_digest)
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("Full episode rendering requires ffmpeg and ffprobe on PATH")
    if destination.exists():
        raise ValueError("Render destination already exists; choose a new immutable bundle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = PhysicsRecord.from_bytes((root / "trace.npz").read_bytes())
    if record.content_hash() != manifest["metadata"]["trace_sha256"]:
        raise ValueError("Physical array hash disagrees with the evidence manifest")
    indices, video_times = frame_schedule(record.arrays["time_s"], fps)
    preview_indices = set(np.linspace(0, len(indices) - 1, min(PREVIEW_LIMIT, len(indices))).astype(int).tolist())
    preview_frames = []
    metadata = manifest["metadata"]
    source = json.loads((root / "source.json").read_bytes())
    if source["dependencies"]["mujoco"] != mujoco.__version__:
        raise ValueError("MJB rendering requires the recorded MuJoCo version")
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    # Offscreen buffer size is a renderer setting, not a change to episode physics.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    model.vis.headlight.ambient[:] = 0.7
    model.vis.headlight.diffuse[:] = 0.8
    model.vis.headlight.specular[:] = 0.2
    task_site = metadata.get("task_site")
    task_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, task_site) if task_site else -1
    data = mujoco.MjData(model)
    speed = (len(indices) / fps) / (len(preview_indices) / 10)
    frame_map = []
    with tempfile.TemporaryDirectory(prefix=".rigby-render-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        video = staging / "episode.mp4"
        with (staging / "ffmpeg.log").open("wb") as log:
            process = subprocess.Popen([
                ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{2 * WIDTH}x{HEIGHT + BANNER}", "-r", str(fps), "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(video),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
                    for frame_number, index in enumerate(indices):
                        mujoco.mj_setState(model, data, record.arrays["state"][index], STATE_SPEC)
                        mujoco.mj_forward(model, data)  # Render-only reconstruction, never replay evidence.
                        panels = []
                        for task in (False, True):
                            camera = _camera(**metadata["camera"], task=task)
                            if task and task_site_id >= 0:
                                camera.lookat[:] = data.site_xpos[task_site_id]
                            renderer.update_scene(data, camera=camera)
                            panels.append(np.asarray(renderer.render()).copy())
                        raw = Image.fromarray(np.concatenate(panels, axis=1))
                        time_s = float(record.arrays["time_s"][index])
                        full = _label(raw, metadata, time_s)
                        process.stdin.write(full.tobytes())
                        if frame_number == 0:
                            full.save(staging / "initial.png")
                        if frame_number == len(indices) - 1:
                            full.save(staging / "final.png")
                        if frame_number in preview_indices:
                            preview_frames.append(_label(raw, metadata, time_s, preview=f"GIF SUMMARY | approximately {speed:.1f}x speed | full video: episode.mp4"))
                        frame_map.append({"frame": frame_number, "playback_time_s": frame_number / fps, "requested_simulation_time_s": float(video_times[frame_number]), "recorded_sample": int(index), "simulation_time_s": time_s})
                process.stdin.close()
                if process.wait(timeout=120) != 0:
                    raise RuntimeError("ffmpeg failed: " + (staging / "ffmpeg.log").read_text(errors="replace")[-2000:])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
        probe = json.loads(subprocess.check_output([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video),
        ], timeout=120))
        stream = probe["streams"][0]
        if int(stream["nb_read_frames"]) != len(indices) or abs(float(stream["duration"]) - len(indices) / fps) > 0.01:
            raise RuntimeError("Encoded video lost frames or changed the required playback duration")
        preview_frames[0].save(staging / "preview.gif", save_all=True, append_images=preview_frames[1:], duration=100, loop=0, optimize=True)
        payloads = {name: (staging / name).read_bytes() for name in ("episode.mp4", "preview.gif", "initial.png", "final.png")}
        payloads["frames.json"] = json_bytes(frame_map)
        payloads["encoding.json"] = json_bytes({"probe": probe, "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]})
        media_metadata = {
            "schema": "rigby.presentation/1", "source_bundle_sha256": digest,
            "source_trace_sha256": metadata["trace_sha256"], "robot_id": metadata["robot_id"], "outcome": metadata["outcome"],
            "full_episode": True, "frame_count": len(indices), "fps": fps,
            "simulation_duration_s": float(record.arrays["time_s"][-1] - record.arrays["time_s"][0]),
            "playback_duration_s": len(indices) / fps, "preview_is_summary": True,
            "refusal_slate": len(record.arrays["time_s"]) == 1,
            "physics_replay_performed_by_renderer": False,
            "presentation_lighting": {"headlight_ambient": 0.7, "diffuse": 0.8, "specular": 0.2},
            "task_camera_tracks_recorded_site": task_site,
            "presentation_views_are_policy_observations": False,
        }
        rendered_digest = write_bundle(destination, payloads, media_metadata)
    return {"bundle": str(destination), "sha256": rendered_digest, **media_metadata}
