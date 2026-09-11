"""Run the cabinet task and save what both cameras saw, side by side.

This is the picture the VLM will be handed: the room view on the left, the wrist
view on the right, one pair per sample. It doubles as the fastest way to see
what a run actually did, which a column of numbers is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "src")

from rigby_poc.gripper.body.manifest import spec  # noqa: E402
from rigby_poc.gripper.decision.solver import _reach_for  # noqa: E402
from rigby_poc.gripper.decision.cabinet import (  # noqa: E402
    PHASES, READY_AT, advance, decide, look_global,
)
from rigby_poc.gripper.physics.model import JOINTS, computed_torque, make  # noqa: E402

_ROOM = (440, 330)
_GRIP = (240, 180)


def filmstrip(seconds: float = 40.0, fps: int = 30, samples: int = 6,
              out: str = "milestones/fridge-filmstrip.png") -> None:
    document = spec()["scene"]["fridge"]
    at = np.asarray([document["contents_at"][0], document["contents_at"][2],
                     document["contents_at"][1]])
    body = make(np.asarray([0.03, 0.03, 0.03]), at, scene="fridge")
    per_joint = spec().get("rate_limits", {}).get("joints_deg_per_s", {})
    ceiling = np.asarray(
        [float(np.radians(per_joint.get(n, 120.0))) for n in JOINTS[:4]]
        + [0.035] * 2)

    ready = _reach_for(body, READY_AT, None, 0.72)
    for slot, value in zip([body.address(n) for n in JOINTS[:4]], ready):
        body.data.qpos[slot] = value
    mujoco.mj_forward(body.model, body.data)

    latched = {"on": False}
    held = np.asarray(body.q())
    squeeze = 0.0
    phase = 0
    per_frame = max(1, int(round((1.0 / fps) / body.model.opt.timestep)))
    total = int(seconds * fps)
    want = {int(total * i / samples) for i in range(samples)}
    shots: list[tuple[str, Image.Image, Image.Image]] = []

    for index in range(total):
        now = index / fps
        seen = look_global(body, held, squeeze, latched)
        phase = advance(body, seen, phase, now)
        command = decide(body, seen, phase, 0.72, now)
        squeeze = command.squeeze_n
        if not command.hold_station:
            held = held + np.clip(command.target - held,
                                  -ceiling / fps, ceiling / fps)
        for _ in range(per_frame):
            body.data.ctrl[:] = computed_torque(body, held, command.squeeze_n)
            mujoco.mj_step(body.model, body.data)
        if index in want:
            label = (f"t={now:.1f}s  {PHASES[phase][0]}  "
                     f"door={seen.door_deg:.0f} deg")
            shots.append((
                label,
                Image.fromarray(body.view(*_ROOM, camera="room")),
                Image.fromarray(body.view(*_GRIP, camera="gripper"))))

    pad, bar = 8, 20
    cell_w = _ROOM[0] + _GRIP[0] + pad
    sheet = Image.new("RGB", (cell_w + pad * 2,
                              (_ROOM[1] + bar + pad) * len(shots) + pad),
                      (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    for row, (label, room, wrist) in enumerate(shots):
        top = pad + row * (_ROOM[1] + bar + pad)
        draw.text((pad, top), label, fill=(190, 200, 215))
        sheet.paste(room, (pad, top + bar))
        sheet.paste(wrist, (pad + _ROOM[0] + pad, top + bar))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"  {out}  {sheet.size[0]}x{sheet.size[1]}")


if __name__ == "__main__":
    filmstrip()
