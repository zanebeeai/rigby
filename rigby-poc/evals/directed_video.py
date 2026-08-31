"""Record a VLM-directed run as the model saw it: both cameras, side by side.

Two panels, each rendered at the SAME resolution the planner sends and with the
SAME pixels -- corner at 480x340, wrist at 320x240, neither brightened. An
earlier version of this recorder showed the corner view full-frame at 640x460
with brightness 1.55 applied, which is a nicer picture of the run but is not
what the model was looking at, and the whole point of showing both feeds is to
be able to ask what the model could actually have seen.

The planner here REPLAYS a recorded run rather than calling the model again.
The physics are deterministic and the control loop is the same one runs/directed
uses, so replaying the targets at the times they were set reproduces the run
frame for frame -- and asking the model a second time would pay twice for the
same decisions, and could not be guaranteed to give the same ones.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "src")

from rigby_poc.gripper.body.manifest import spec  # noqa: E402
from rigby_poc.gripper.decision.goals import NumericTarget  # noqa: E402
from rigby_poc.gripper.decision.greedy import pursue  # noqa: E402
from rigby_poc.gripper.decision.pick_and_place import (  # noqa: E402
    object_in_target,
)
from rigby_poc.gripper.physics.model import (  # noqa: E402
    JOINTS, computed_torque, make,
)
from rigby_poc.gripper.sensing.gripper_camera import Senses, sense  # noqa: E402

_OUT = Path("videos")
#: EXACTLY the sizes decision/planner.py renders for the model. Kept here as a
#: pair of constants so a change there and a silent divergence here is visible.
_CORNER = (480, 340)
_GRIP = (320, 240)
_PAD = 10
_LABEL = 20
_READOUT = 92
_W = _PAD * 3 + _CORNER[0] + _GRIP[0]
_H = _PAD * 2 + _LABEL + _CORNER[1] + _READOUT
_BUNDLED = sorted(
    (Path.home() / "AppData/Local/ms-playwright").glob("ffmpeg-*/ffmpeg-*.exe"))


def _encoder() -> str | None:
    return shutil.which("ffmpeg") or (str(_BUNDLED[-1]) if _BUNDLED else None)


@dataclass
class Replay:
    """A planner that hands back decisions the model already made."""

    decisions: list = field(default_factory=list)
    held: NumericTarget | None = field(default=None, repr=False)
    spans: dict = field(default_factory=dict, repr=False)
    at: int = field(default=0, repr=False)
    why: str = field(default="", repr=False)
    step_text: str = field(default="", repr=False)

    def due(self, now: float) -> bool:
        return (self.at < len(self.decisions)
                and now >= float(self.decisions[self.at]["t"]) - 1e-9)

    def take(self, body, seen, now: float) -> None:
        entry = self.decisions[self.at]
        self.at += 1
        self.held = NumericTarget(
            metric=entry["target"], value=float(entry["value"]), set_at_s=now,
            also=tuple((a[0], float(a[1]), float(a[2]) if len(a) > 2 else 0.5)
                       for a in entry.get("also") or []),
            using=tuple(entry.get("using") or []))
        self.spans = self.held.spans(body, seen)
        self.why = str(entry.get("why", ""))
        self.step_text = str(entry.get("step_text") or "")


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines[:2]


def record(source: Path, seconds: float = 30.0, fps: int = 30,
           table_top: float = 0.72,
           name: str = "gripper-vlm-directed") -> list[Path]:
    recorded = json.loads(source.read_text(encoding="utf-8"))
    task = recorded.get("task", "")
    decisions = [e for e in recorded.get("transcript", []) if "target" in e]
    plan = next((e.get("steps") for e in recorded.get("transcript", [])
                 if e.get("steps")), [])
    planner = Replay(decisions=decisions)

    body = make(np.asarray([0.03, 0.04, 0.03]), np.asarray([0.0, 0.30, 0.76]),
                table_top=table_top)
    document = spec().get("rate_limits", {})
    per_joint = document.get("joints_deg_per_s", {})
    ceiling = np.asarray(
        [float(np.radians(per_joint.get(n, 120.0))) for n in JOINTS[:4]]
        + [float(document.get("finger_m_per_s", 0.07)) / 2.0] * 2)

    eyes = Senses()
    held = np.asarray(body.q())
    squeeze = 0.0
    per_frame = max(1, int(round((1.0 / fps) / body.model.opt.timestep)))
    shots: list[Image.Image] = []
    corner_light: list[float] = []
    grip_light: list[float] = []

    for index in range(int(seconds * fps)):
        now = index / fps
        seen = sense(body, eyes, held, squeeze, now, table_top)
        if planner.due(now):
            planner.take(body, seen, now)
        target = planner.held

        if target is not None:
            wanted, error, how = pursue(body, seen, target, planner.spans,
                                        start_from=held)
            held = held + np.clip(wanted - held, -ceiling / fps, ceiling / fps)
            wants_wider = False
            for metric, value, _weight in target.terms():
                if metric == "grip_tip_spread_m":
                    wants_wider = float(value) > body.opening() + 0.004
            squeeze = 0.0 if wants_wider else 12.0
            for _ in range(per_frame):
                body.data.ctrl[:] = computed_torque(body, held, squeeze)
                mujoco.mj_step(body.model, body.data)
        else:
            error, how = 0.0, {"part": "-", "move": "waiting"}

        # THE PIXELS THE MODEL WAS SENT, unretouched and at the sent size.
        corner = Image.fromarray(
            body.view(*_CORNER, camera="room")).convert("RGB")
        wrist = Image.fromarray(
            body.view(*_GRIP, camera="gripper")).convert("RGB")
        corner_light.append(float(np.asarray(corner).mean()))
        grip_light.append(float(np.asarray(wrist).mean()))

        frame = Image.new("RGB", (_W, _H), (16, 18, 24))
        frame.paste(corner, (_PAD, _PAD + _LABEL))
        frame.paste(wrist, (_PAD * 2 + _CORNER[0], _PAD + _LABEL))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([(_PAD, _PAD + _LABEL),
                        (_PAD + _CORNER[0], _PAD + _LABEL + _CORNER[1])],
                       outline=(120, 140, 170))
        draw.rectangle([(_PAD * 2 + _CORNER[0], _PAD + _LABEL),
                        (_PAD * 2 + _CORNER[0] + _GRIP[0],
                         _PAD + _LABEL + _GRIP[1])],
                       outline=(120, 140, 170))
        draw.text((_PAD, _PAD), "CORNER camera - sent to the model, 480x340",
                  fill=(190, 205, 230))
        draw.text((_PAD * 2 + _CORNER[0], _PAD),
                  "GRIPPER camera - sent to the model, 320x240",
                  fill=(190, 205, 230))

        base = _PAD + _LABEL + _CORNER[1] + 8
        draw.text((_PAD, base), chr(34) + task + chr(34),
                  fill=(245, 235, 200))
        step = f"step: {planner.step_text}" if planner.step_text else "step: -"
        draw.text((_PAD, base + 16), f"t={now:5.2f}s   {step}",
                  fill=(200, 210, 225))
        if target is not None:
            draw.text((_PAD, base + 32),
                      f"asked for: {target.metric} = {target.value:g}   "
                      f"chasing via {how['part']}/{how['move']}   "
                      f"error {error:5.3f}", fill=(235, 240, 250))
        draw.text((_PAD, base + 48),
                  f"jaws {body.opening() * 100:4.1f} cm    "
                  f"range {seen.range_ahead_m:4.2f} m    "
                  f"pads {seen.tip_force_left_n:4.1f}/"
                  f"{seen.tip_force_right_n:4.1f} N    "
                  f"holding {seen.holding()}    "
                  f"{'SEES IT' if seen.object_seen else 'BLIND'}",
                  fill=(200, 210, 225))
        for offset, line in enumerate(_wrap(planner.why, 96)):
            draw.text((_PAD, base + 64 + 13 * offset), line,
                      fill=(165, 180, 205))
        if object_in_target(body):
            draw.text((_W - 100, base), "IN THE BIN", fill=(150, 230, 160))
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

    print(f"  task: {task}")
    for line in plan:
        print(f"    step: {line}")
    print(f"  decisions replayed: {planner.at}/{len(decisions)}")
    # How bright the sent images actually are, 0-255. If the model is being
    # handed a near-black picture then "it did not see the block" is a fact
    # about the render, not about the model.
    print(f"  mean brightness sent -- corner "
          f"{sum(corner_light) / max(1, len(corner_light)):5.1f}/255, "
          f"wrist {sum(grip_light) / max(1, len(grip_light)):5.1f}/255")
    print(f"  placed in the bin: {object_in_target(body)}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",
                        default="frontend/public/directed-vlm.json")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--name", default="gripper-vlm-directed")
    args = parser.parse_args()
    for path in record(Path(args.source), seconds=args.seconds,
                       name=args.name):
        print(f"  {path}  {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
