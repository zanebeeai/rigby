"""Run the torque-driven gripper and write a clip that can be watched.

The welded version exported the pose it commanded. This exports the pose the
simulation actually reached, which is not the same thing and is the reason the
whole exercise moved to torque: a commanded pose is what you asked for, and a
body under force control is allowed to disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .gripper import spec
from .gripper_control import (
    PHASES,
    advance,
    decide,
    holding,
    object_above_rim_m,
    object_in_grasp_m,
    object_in_target,
    object_over_target_m,
    palm_facing,
    palm_to_object_m,
)
from .gripper_torque import JOINTS, computed_torque, make

_PUBLIC = Path(__file__).resolve().parents[2] / "frontend" / "public"

#: Joint speed ceilings, so the arm moves like a machine rather than a cut.
#: Applied to the TARGET, not to the body: a rate limit on a torque controller
#: belongs on what it is asked for, since the body's own speed is then a
#: consequence of physics rather than something clamped after the fact.
def _rate_ceiling() -> np.ndarray:
    document = spec().get("rate_limits", {})
    per_joint = document.get("joints_deg_per_s", {})
    return np.asarray(
        [float(np.radians(per_joint.get(name, 120.0))) for name in JOINTS[:4]]
        + [float(document.get("finger_m_per_s", 0.07))] * 2)


def _app(point: np.ndarray) -> list[float]:
    """MuJoCo Z-up to the viewer's Y-up, at the edge and nowhere else."""
    return [float(point[0]), float(point[2]), float(point[1])]


def run(block_half=None, block_at=None, table_top: float = 0.72,
        seconds: float = 22.0, fps: int = 30, name: str = "gripper-run",
        verbose: bool = True) -> dict:
    block_half = np.asarray(block_half if block_half is not None
                            else [0.03, 0.04, 0.03])
    block_at = np.asarray(block_at if block_at is not None else [0.0, 0.30, 0.76])
    body = make(block_half, block_at, table_top=table_top)
    ceiling = _rate_ceiling()

    phase = 0
    per_frame = max(1, int(round((1.0 / fps) / body.model.opt.timestep)))
    frames: list[dict] = []
    worst_penetration = 0.0
    held = np.asarray(body.q())

    radii = spec()["kinematics"].get("link_radius_m", [0.022, 0.019, 0.016])
    finger = spec()["kinematics"]["finger"]

    for index in range(int(seconds * fps)):
        now = index / fps
        phase = advance(body, phase, now)
        command = decide(body, phase, table_top, now)
        # The rate limit lives here, once, between deciding and doing -- the
        # same place the humanoid's does, and for the same reason: no primitive
        # can bypass it and no new one has to remember it.
        step = np.clip(command.target - held, -ceiling / fps, ceiling / fps)
        held = held + step
        for _ in range(per_frame):
            body.data.ctrl[:] = computed_torque(body, held, command.squeeze_n)
            mujoco.mj_step(body.model, body.data)
        worst_penetration = max(worst_penetration, body.penetration_mm())

        joints = [body.body_at(n) for n in ("base", "link1", "link2", "link3")]
        joints.append(body.body_at("plate"))
        links = [
            {"kind": "segment", "from": _app(joints[i]), "to": _app(joints[i + 1]),
             "radius": float(radii[min(i, len(radii) - 1)]), "simulated": False}
            for i in range(len(joints) - 1)
        ]
        for pad in ("left_geom", "right_geom"):
            centre = body.geom_at(pad)
            back = centre - body.approach() * float(finger["length_m"]) / 2.0
            tip = centre + body.approach() * float(finger["length_m"]) / 2.0
            links.append({"kind": "finger", "from": _app(back), "to": _app(tip),
                          "half": [float(finger["thickness_m"]),
                                   float(finger["pad_width_m"]) / 2.0],
                          "simulated": True})
        links.append({"kind": "plate", "at": _app(body.body_at("plate")),
                      "approach": _app(body.approach()),
                      "across": _app(np.asarray([1.0, 0.0, 0.0])),
                      "simulated": True})

        quat = body.block_quat()
        frames.append({
            "t": round(now, 4),
            "phase": int(phase),
            "links": links,
            "block": _app(body.block()),
            # wxyz, with the same axis swap the positions get, once.
            "block_quat": [float(quat[0]), float(quat[1]),
                           float(quat[3]), float(quat[2])],
            "forces": {k: round(v, 2) for k, v in body.forces().items()},
            "opening_m": round(body.opening(), 5),
            "over_target_m": round(object_over_target_m(body), 4),
            "above_rim_m": round(object_above_rim_m(body), 4),
            "in_target": bool(object_in_target(body)),
            "penetration_mm": round(body.penetration_mm(), 3),
        })
        if verbose and index % 45 == 0:
            print(f"  t={now:5.2f} ph{phase} {PHASES[phase][0]:<11} "
                  f"palm={palm_to_object_m(body) * 100:5.1f} "
                  f"in={object_in_grasp_m(body) * 100:5.1f} "
                  f"open={body.opening() * 100:5.2f} "
                  f"pen={body.penetration_mm():4.2f}mm "
                  f"f={ {k: round(v, 1) for k, v in body.forces().items() if v > 0.3} }")

    heights = [f["block"][1] for f in frames]
    bin_doc = spec()["scene"]["bin"]
    document = {
        "protocol": "gripper_clip_v1",
        "embodiment": "gripper",
        "control": "computed torque",
        "fps": fps,
        "table_top_m": float(table_top),
        "block_half_m": [float(v) for v in block_half],
        "phase_names": ["approach", "open", "engulf", "close", "squeeze",
                        "lift", "carry", "release"],
        "simulated_geoms": ["left_geom", "right_geom", "plate_geom",
                            "block_geom", "table", "bin"],
        "pedestal": spec()["kinematics"].get("pedestal"),
        "bin": bin_doc,
        "achieved": {
            "peak_lift_m": round(max(heights) - heights[0], 5),
            "final_lift_m": round(heights[-1] - heights[0], 5),
            "in_target": bool(frames[-1]["in_target"]),
            "deepest_penetration_mm": round(worst_penetration, 3),
        },
        "frames": frames,
    }
    _PUBLIC.mkdir(parents=True, exist_ok=True)
    (_PUBLIC / f"{name}.json").write_text(json.dumps(document), encoding="utf-8")
    return document["achieved"]
