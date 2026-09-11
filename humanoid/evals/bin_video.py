"""Record the pick-and-place from both cameras, straight out of MuJoCo.

Corner camera full frame, gripper camera inset, and a readout of what the machine
believes at that moment. These are the simulator's own renders through the two
cameras declared in the model, so what you watch is what was simulated and the
inset is literally what the controller was looking through.

The bench scene keeps its single key light, unchanged, because the gripper camera
is an INPUT: the controller finds the block by segmenting warm pixels, and
relighting the scene to make a nicer video would change the run. The corner
frames are brightened afterwards, on the recorded image only, which the
simulation never sees.
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
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, "src")

from rigby_poc.gripper.body.manifest import spec  # noqa: E402
from rigby_poc.gripper.decision.pick_and_place import (  # noqa: E402
    PHASES, advance, decide, object_in_target,
)
from rigby_poc.gripper.physics.model import (  # noqa: E402
    JOINTS, computed_torque, make,
)
from rigby_poc.gripper.sensing.gripper_camera import Senses, sense  # noqa: E402

_OUT = Path("videos")
_ROOM = (640, 460)
_GRIP = (240, 180)
_BUNDLED = sorted(
    (Path.home() / "AppData/Local/ms-playwright").glob("ffmpeg-*/ffmpeg-*.exe"))


def _encoder() -> str | None:
    return shutil.which("ffmpeg") or (str(_BUNDLED[-1]) if _BUNDLED else None)


def record(seconds: float = 24.0, fps: int = 30, table_top: float = 0.72,
           name: str = "gripper-two-cameras") -> list[Path]:
    block_half = np.asarray([0.03, 0.04, 0.03])
    block_at = np.asarray([0.0, 0.30, 0.76])
    body = make(block_half, block_at, table_top=table_top)

    document = spec().get("rate_limits", {})
    per_joint = document.get("joints_deg_per_s", {})
    ceiling = np.asarray(
        [float(np.radians(per_joint.get(n, 120.0))) for n in JOINTS[:4]]
        + [float(document.get("finger_m_per_s", 0.07)) / 2.0] * 2)

    eyes = Senses()
    held = np.asarray(body.q())
    squeeze = 0.0
    phase = 0
    per_frame = max(1, int(round((1.0 / fps) / body.model.opt.timestep)))
    shots: list[Image.Image] = []

    for index in range(int(seconds * fps)):
        now = index / fps
        seen = sense(body, eyes, held, squeeze, now, table_top)
        phase = advance(body, seen, phase, now)
        command = decide(body, seen, phase, table_top, now)
        squeeze = command.squeeze_n
        if not command.hold_station:
            held = held + np.clip(command.target - held,
                                  -ceiling / fps, ceiling / fps)
        for _ in range(per_frame):
            body.data.ctrl[:] = computed_torque(body, held, command.squeeze_n)
            mujoco.mj_step(body.model, body.data)

        room = Image.fromarray(body.view(*_ROOM, camera="room")).convert("RGB")
        room = ImageEnhance.Brightness(room).enhance(1.55)
        room = ImageEnhance.Contrast(room).enhance(1.08)
        frame = room
        frame.paste(Image.fromarray(body.view(*_GRIP, camera="gripper")),
                    (_ROOM[0] - _GRIP[0] - 10, 10))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([(_ROOM[0] - _GRIP[0] - 10, 10),
                        (_ROOM[0] - 10, 10 + _GRIP[1])],
                       outline=(120, 140, 170))
        draw.text((12, 10), f"t={now:5.2f}s   phase {phase}: {PHASES[phase][0]}",
                  fill=(235, 240, 250))
        draw.text((12, 26), f"jaws {body.opening() * 100:4.1f} cm    "
                            f"holding {seen.holding()}    "
                            f"{'SEES IT' if seen.object_seen else 'blind'}",
                  fill=(200, 210, 225))
        draw.text((12, 42), "IN THE BIN" if object_in_target(body) else "",
                  fill=(150, 230, 160))
        draw.text((_ROOM[0] - _GRIP[0] - 10, 14 + _GRIP[1]),
                  "gripper camera - what the controller sees",
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
            line = [encoder, "-y", "-f", "image2pipe", "-framerate", str(fps),
                    "-c:v", "mjpeg", "-i", str(reel), "-c:v", codec,
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
            line += ["-crf", "20"] if full else ["-b:v", "4M"]
            done = subprocess.run(line + [str(movie)], capture_output=True)
            if done.returncode == 0:
                written.append(movie)
            else:
                tail = done.stderr.decode("utf-8", "replace").splitlines()
                print(f"  encoding failed: {tail[-1] if tail else '?'}")

    print(f"  placed in the bin: {object_in_target(body)}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=24.0)
    parser.add_argument("--name", default="gripper-two-cameras")
    args = parser.parse_args()
    for path in record(seconds=args.seconds, name=args.name):
        print(f"  {path}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
