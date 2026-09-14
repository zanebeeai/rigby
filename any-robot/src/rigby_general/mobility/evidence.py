"""Seal a mobile body's physics run as a replayable bundle, and render it with the support contacts on every frame.

The bundle keeps the G01 layout: the compiled model, the physics record,
the declaration and manifest the body ran under, the world, the test's
task and outcome, the source provenance; the recorded controls replay to
the recorded states before the seal is written. The renderer draws the
banner the other goals use -- the body, the outcome, the simulation clock
-- and, under it, which members are touching the floor at that frame,
read from the recorded contacts, so a support test can be watched rather
than trusted.
"""

from __future__ import annotations

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
from rigby_core.simulation.recording import PhysicsRecord, STATE_SPEC, replay_physics

from ..evidence.capture import json_bytes, source_provenance
from ..evidence.render import HEIGHT, PREVIEW_LIMIT, WIDTH, frame_schedule
from .contracts import MobileBodyManifestV1
from .ingest import MobileBody
from .validate import Run, robot_bodies


PROTOCOL = "rigby.mobile-body-test/1"
BANNER = 96
FPS = 12
BACKGROUND = (17, 24, 39)


def seal_run(destination: Path, *, body: MobileBody, manifest: MobileBodyManifestV1, run: Run, label: str, caption: str, test: dict, outcome: dict, goal: str = "G16",
             camera: dict | None = None, world_xml: str | None = None, model: mujoco.MjModel | None = None) -> dict:
    """One bundle for one recorded run; the recorded controls must replay to the recorded states."""

    model = model or body.floor_model
    replay = replay_physics(model, run.record)
    if not replay["agrees"]:
        raise RuntimeError(f"{label}: the recorded controls do not replay to the recorded states")
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance, archive = source_provenance()
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    times = run.record.arrays["time_s"]
    payloads = {
        "model.mjb": buffer.tobytes(), "model.xml": (world_xml or body.floor_xml).encode("utf-8"), "robot.xml": body.xml.encode("utf-8"),
        "mobility.json": json_bytes(body.declaration), "body_manifest.json": json_bytes(manifest.model_dump(mode="json")), "provenance.json": json_bytes(body.provenance),
        "world.json": json_bytes({"mode": "level_floor" if world_xml is None else "course", "timestep_s": float(model.opt.timestep), "gravity": model.opt.gravity.tolist(), "floor_friction": 1.0}),
        "task.json": json_bytes({"goal": goal, "protocol": PROTOCOL, "label": label, **test, "clock_disclosure": {"physics_timestep_s": float(model.opt.timestep), "controls": "position servos held at the stance (velocity servos at zero); no external wrench, no teleport, no artificial support"}}),
        "outcome.json": json_bytes({**outcome, "actual_physics_duration_s": float(times[-1] - times[0]), "physical_steps": len(times) - 1, "refusal": None}),
        "trace.npz": run.record.to_bytes(),
        "repeats.json": json_bytes({"count": 1, "recorded_control_replay": replay, "note": "one run; the recorded controls replay to the recorded states"}),
        "source.json": json_bytes(provenance), "source.zip": archive,
    }
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.declaration["base_body"])
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, run.record.arrays["state"][0], STATE_SPEC)
    mujoco.mj_forward(model, data)
    centre = [float(v) for v in data.xpos[base]]
    centre[2] = max(0.15, centre[2] * 0.6)
    metadata = {"goal": goal, "protocol": PROTOCOL, "robot_id": body.robot_id, "created_at_utc": datetime.now(timezone.utc).isoformat(), "outcome": outcome["status"], "fault": False,
                "simulation_duration_s": float(times[-1] - times[0]), "reference_duration_s": float(times[-1] - times[0]), "reference_clock_matches_physics": True, "caption": caption,
                "trace_sha256": run.record.content_hash(), "world_sha256": hashlib.sha256(payloads["world.json"]).hexdigest(),
                "camera": camera or {"centre": centre, "reach": max(0.6, 1.2 * max(manifest.footprint_m))}, "task_site": None, "floor_geom": int(run.floor_geom), "base_body": body.declaration["base_body"],
                "support_members": list(body.declaration["support_members"]), "test": test.get("test")}
    digest = write_bundle(destination, payloads, metadata)
    return {"bundle": destination.as_posix(), "sha256": digest, "trace_sha256": metadata["trace_sha256"], "simulation_duration_s": metadata["simulation_duration_s"], "replay_agrees": True}


def _camera(centre, reach: float, *, task: bool) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = centre
    camera.distance = max(0.5, reach) * (1.6 if task else 2.8)
    camera.azimuth = 40.0 if task else 135.0
    camera.elevation = -18.0 if task else -22.0
    return camera


def _contacts_at(model: mujoco.MjModel, record: PhysicsRecord, index: int, robot: frozenset[int]) -> list[str]:
    """The robot's bodies touching the ground (any static geom: floor, pad, platform, ramp) at the recorded sample."""

    offsets = record.arrays["contact_offsets"]
    pairs = record.arrays["contact_geom"][int(offsets[index]): int(offsets[index + 1])]
    bodies = set()
    for g1, g2 in pairs:
        ground1, ground2 = int(model.geom_bodyid[int(g1)]) == 0, int(model.geom_bodyid[int(g2)]) == 0
        if ground1 != ground2:
            other = int(g2) if ground1 else int(g1)
            body = int(model.geom_bodyid[other])
            if body in robot:
                bodies.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "?")
    return sorted(bodies)


def _label(frame: Image.Image, metadata: dict, time_s: float, contacts: list[str], tilt_deg: float, height_m: float, *, preview: str | None = None, phase: str = "") -> Image.Image:
    canvas = Image.new("RGB", (2 * WIDTH, HEIGHT + BANNER), BACKGROUND)
    canvas.paste(frame, (0, BANNER))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=12)
    status = metadata["outcome"].upper().replace("_", " ")
    tint = (100, 230, 165) if status in ("SUCCESS", "STABLE", "RECOVERED") else (255, 189, 100)
    draw.text((12, 6), f"{metadata['robot_id']} | {status} | sim t={time_s:.3f}s | base {height_m:.3f} m, tilt {tilt_deg:.1f} deg", fill=tint, font=font)
    draw.text((12, 27), (str(metadata.get("caption", ""))[:120] + (f" | {phase}" if phase else ""))[:150], fill="white", font=font)
    draw.text((12, 48), preview or "FULL EPISODE | real-time playback | recorded physical states | servos holding the stance, nothing else acting", fill=(191, 210, 233), font=font)
    declared = set(metadata.get("support_members", []))
    undeclared = [c for c in contacts if c not in declared]
    text = "ground contact now: " + (", ".join(contacts) if contacts else "none")
    draw.text((12, 68), text[:150], fill=(255, 140, 140) if undeclared else (170, 220, 190), font=small)
    draw.text((12, 82), "Global view", fill=(170, 185, 205), font=small)
    draw.text((WIDTH + 12, 82), "Close view | position servos | joint encoders + IMU + touch", fill=(170, 185, 205), font=small)
    return canvas


def render_run(root: Path, destination: Path, *, ffmpeg: str, ffprobe: str, phases: list[dict] | None = None) -> dict:
    """Render a sealed run into destination/media: episode.mp4 at real time, preview.gif, frames.json."""

    manifest = verify_bundle(root)
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    metadata = dict(manifest["metadata"])
    record = PhysicsRecord.from_bytes((root / "trace.npz").read_bytes())
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    model.vis.headlight.ambient[:] = 0.7
    model.vis.headlight.diffuse[:] = 0.8
    model.vis.headlight.specular[:] = 0.2
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, metadata["base_body"])
    robot = robot_bodies(model, metadata["base_body"])
    indices, video_times = frame_schedule(record.arrays["time_s"], FPS)
    preview_at = set(np.linspace(0, len(indices) - 1, min(PREVIEW_LIMIT, len(indices))).astype(int).tolist())
    speed = (len(indices) / FPS) / (len(preview_at) / 10)
    previews = []
    frames = []
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".d16-", dir=destination) as temporary:
        staging = Path(temporary)
        video = staging / "episode.mp4"
        process = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * WIDTH}x{HEIGHT + BANNER}", "-r", str(FPS), "-i", "pipe:0",
                                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
                for number, index in enumerate(indices):
                    mujoco.mj_setState(model, data, record.arrays["state"][index], STATE_SPEC)
                    mujoco.mj_forward(model, data)
                    up = np.zeros(3)
                    mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), data.xquat[base])
                    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
                    panels = []
                    for task in (False, True):
                        camera = _camera(**metadata["camera"], task=task)
                        if task:
                            camera.lookat[:] = data.xpos[base]
                        renderer.update_scene(data, camera=camera)
                        panels.append(np.asarray(renderer.render()).copy())
                    raw = Image.fromarray(np.concatenate(panels, axis=1))
                    time_s = float(record.arrays["time_s"][index])
                    contacts = _contacts_at(model, record, int(index), robot)
                    phase = next((p["label"] for p in (phases or []) if p["start_s"] <= time_s < p["end_s"]), "")
                    full = _label(raw, metadata, time_s, contacts, tilt, float(data.xpos[base][2]), phase=phase)
                    process.stdin.write(full.tobytes())
                    if number in preview_at:
                        previews.append(_label(raw, metadata, time_s, contacts, tilt, float(data.xpos[base][2]), preview=f"GIF SUMMARY | approximately {speed:.1f}x speed | full video: episode.mp4", phase=phase))
                    frames.append({"frame": number, "playback_time_s": number / FPS, "recorded_sample": int(index), "simulation_time_s": time_s, "floor_contacts": contacts, "tilt_deg": round(tilt, 2), "base_height_m": round(float(data.xpos[base][2]), 4), "phase": phase})
            process.stdin.close()
            if process.wait(timeout=600) != 0:
                raise RuntimeError("ffmpeg failed: " + process.stderr.read().decode(errors="replace")[-1500:])
        finally:
            if process.poll() is None:
                process.kill()
        probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate,duration,width,height", "-of", "json", str(video)], timeout=300))["streams"][0]
        if int(probe["nb_read_frames"]) != len(indices):
            raise RuntimeError("the encoded video lost frames")
        previews[0].save(staging / "preview.gif", save_all=True, append_images=previews[1:], duration=100, loop=0, optimize=True)
        payloads = {"episode.mp4": video.read_bytes(), "preview.gif": (staging / "preview.gif").read_bytes(), "frames.json": json_bytes(frames),
                    "encoding.json": json_bytes({"probe": probe, "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]})}
        media = {"schema": "rigby.presentation/1", "source_bundle_sha256": digest, "source_trace_sha256": metadata["trace_sha256"], "robot_id": metadata["robot_id"], "outcome": metadata["outcome"],
                 "full_episode": True, "frame_count": len(indices), "fps": FPS, "simulation_duration_s": float(record.arrays["time_s"][-1] - record.arrays["time_s"][0]), "playback_duration_s": len(indices) / FPS,
                 "preview_is_summary": True, "physics_replay_performed_by_renderer": False, "overlay": {"kind": "support_contacts", "shows": ["floor contacts by body at every frame, undeclared ones in red", "base height and tilt", "the inspection phase"]},
                 "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        rendered = write_bundle(destination / "media", payloads, media)
    return {"media": (destination / "media").as_posix(), "sha256": rendered, "frames": len(indices), "playback_s": len(indices) / FPS}


__all__ = ["PROTOCOL", "render_run", "seal_run"]
