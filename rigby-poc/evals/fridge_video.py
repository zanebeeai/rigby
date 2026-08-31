"""Record the cabinet task straight out of MuJoCo, both cameras, to a video.

Not through the browser viewer: that viewer draws the arm from the clip's link
list and knows nothing about a cabinet, a door or a handle, so it would show the
arm moving in an empty room. These frames are the simulator's own render, which
means what you watch is what was simulated.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "src")

from rigby_poc.gripper.body.manifest import spec  # noqa: E402
from rigby_poc.gripper.decision.solver import _reach_for  # noqa: E402
from rigby_poc.gripper.decision.cabinet import (  # noqa: E402
    PHASES, READY_AT, advance, decide, look_global, on_the_bench,
)
from rigby_poc.gripper.physics.model import JOINTS, computed_torque, make  # noqa: E402

_OUT = Path("milestones/video")
_ROOM = (640, 460)
_WRIST = (240, 180)
_BUNDLED = sorted(
    (Path.home() / "AppData/Local/ms-playwright").glob("ffmpeg-*/ffmpeg-*.exe"))


def _encoder() -> str | None:
    return shutil.which("ffmpeg") or (str(_BUNDLED[-1]) if _BUNDLED else None)


def record(seconds: float = 30.0, fps: int = 30, name: str = "fridge-attempt",
           table_top: float = 0.72) -> list[Path]:
    document = spec()["scene"]["fridge"]
    at = np.asarray([document["contents_at"][0], document["contents_at"][2],
                     document["contents_at"][1]])
    body = make(np.asarray([0.03, 0.03, 0.03]), at, scene="fridge")
    per_joint = spec().get("rate_limits", {}).get("joints_deg_per_s", {})
    ceiling = np.asarray(
        [float(np.radians(per_joint.get(n, 120.0))) for n in JOINTS[:4]]
        + [0.035] * 2)

    ready = _reach_for(body, READY_AT, None, table_top)
    for slot, value in zip([body.address(n) for n in JOINTS[:4]], ready):
        body.data.qpos[slot] = value
    mujoco.mj_forward(body.model, body.data)

    latched: dict = {"on": False}
    held = np.asarray(body.q())
    squeeze = 0.0
    phase = 0
    per_frame = max(1, int(round((1.0 / fps) / body.model.opt.timestep)))
    shots: list[Image.Image] = []

    for index in range(int(seconds * fps)):
        now = index / fps
        seen = look_global(body, held, squeeze, latched)
        phase = advance(body, seen, phase, now)
        command = decide(body, seen, phase, table_top, now)
        squeeze = command.squeeze_n
        if not command.hold_station:
            held = held + np.clip(command.target - held,
                                  -ceiling / fps, ceiling / fps)
        for _ in range(per_frame):
            body.data.ctrl[:] = computed_torque(body, held, command.squeeze_n)
            mujoco.mj_step(body.model, body.data)

        frame = Image.fromarray(body.view(*_ROOM, camera="room")).convert("RGB")
        frame.paste(Image.fromarray(body.view(*_WRIST, camera="wrist")),
                    (_ROOM[0] - _WRIST[0] - 10, 10))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([(_ROOM[0] - _WRIST[0] - 10, 10),
                        (_ROOM[0] - 10, 10 + _WRIST[1])],
                       outline=(120, 140, 170))
        draw.text((12, 10), f"t={now:5.2f}s   phase {phase}: "
                            f"{PHASES[phase][0]}", fill=(225, 232, 245))
        draw.text((12, 26), f"door {seen.door_deg:6.1f} deg    "
                            f"jaws {body.opening() * 100:4.1f} cm    "
                            f"holding {seen.holding()}", fill=(190, 200, 215))
        draw.text((12, 42), command.note or "", fill=(230, 170, 150))
        draw.text((_ROOM[0] - _WRIST[0] - 10, 14 + _WRIST[1]), "wrist camera",
                  fill=(150, 165, 190))
        shots.append(frame)

    _OUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    encoder = _encoder()
    if encoder:
        full = shutil.which("ffmpeg") is not None
        suffix, codec = (".mp4", "libx264") if full else (".webm", "libvpx")
        movie = _OUT / f"{name}{suffix}"
        with tempfile.TemporaryDirectory() as scratch:
            reel = Path(scratch) / "frames.mjpeg"
            with reel.open("wb") as handle:
                for shot in shots:
                    shot.save(handle, format="JPEG", quality=93)
            command_line = [encoder, "-y", "-f", "image2pipe", "-framerate",
                            str(fps), "-c:v", "mjpeg", "-i", str(reel),
                            "-c:v", codec, "-pix_fmt", "yuv420p",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
            command_line += ["-crf", "20"] if full else ["-b:v", "4M"]
            done = subprocess.run(command_line + [str(movie)],
                                  capture_output=True)
            if done.returncode == 0:
                written.append(movie)
            else:
                tail = done.stderr.decode("utf-8", "replace").splitlines()
                print(f"  encoding failed: {tail[-1] if tail else '?'}")

    small = [s.resize((s.width // 2, s.height // 2), Image.LANCZOS)
             for s in shots[::2]]
    gif = _OUT / f"{name}.gif"
    small[0].save(gif, save_all=True, append_images=small[1:],
                  duration=int(2000 / fps), loop=0, optimize=True)
    written.append(gif)
    print(f"  door reached {max(0.0, seen.door_deg):.1f} deg, "
          f"phase {phase + 1}/{len(PHASES)}, on bench {on_the_bench(body)}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--name", default="fridge-attempt")
    args = parser.parse_args()
    for path in record(seconds=args.seconds, name=args.name):
        print(f"  {path}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
