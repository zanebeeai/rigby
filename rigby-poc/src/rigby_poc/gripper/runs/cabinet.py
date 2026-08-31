"""Run the cabinet task and write a clip that can be watched."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from ..body.manifest import spec
from ..decision.solver import _reach_for
from ..decision.cabinet import (
    PHASES,
    READY_AT,
    Working,
    advance,
    decide,
    geometry,
    look_global,
    on_the_bench,
)
from ..physics.model import JOINTS, computed_torque, make

_PUBLIC = Path(__file__).resolve().parents[4] / "frontend" / "public"


def _ceiling() -> np.ndarray:
    document = spec().get("rate_limits", {})
    per_joint = document.get("joints_deg_per_s", {})
    return np.asarray(
        [float(np.radians(per_joint.get(name, 120.0))) for name in JOINTS[:4]]
        # Halved: the declared figure is how fast the GAP closes,
        # and both fingers contribute to the gap.
        + [float(document.get("finger_m_per_s", 0.07)) / 2.0] * 2)


def _app(point) -> list[float]:
    return [float(point[0]), float(point[2]), float(point[1])]


def run(seconds: float = 40.0, fps: int = 30, table_top: float = 0.72,
        name: str = "fridge-run", verbose: bool = True,
        sensing: str = "global") -> dict:
    document = spec()["scene"]["fridge"]
    half = np.asarray([0.03, 0.03, 0.03])
    at = np.asarray([document["contents_at"][0], document["contents_at"][2],
                     document["contents_at"][1]])
    body = make(half, at, table_top=table_top, scene="fridge")
    ceiling = _ceiling()
    place = geometry()

    # Fold the arm up in front before anything else. Its zero pose reaches
    # straight out over the cabinet, so the first move of every phase started by
    # dragging the gripper across the cabinet roof, which is where it stopped.
    ready = _reach_for(body, READY_AT, None, table_top)
    if ready is not None:
        for slot, value in zip([body.address(n) for n in JOINTS[:4]], ready):
            body.data.qpos[slot] = value
        mujoco.mj_forward(body.model, body.data)

    latched = {"on": False}
    work = Working()
    held = np.asarray(body.q())
    squeeze = 0.0
    phase = 0
    per_frame = max(1, int(round((1.0 / fps) / body.model.opt.timestep)))
    frames: list[dict] = []
    radii = spec()["kinematics"].get("link_radius_m", [0.022, 0.019, 0.016])
    finger = spec()["kinematics"]["finger"]
    hinge = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_JOINT,
                              "door_hinge")
    slot = body.model.jnt_qposadr[hinge]
    reached = 0

    for index in range(int(seconds * fps)):
        now = index / fps
        seen = look_global(body, held, squeeze, latched)
        phase = advance(body, seen, phase, now, work)
        reached = max(reached, phase)
        command = decide(body, seen, phase, table_top, now, work)
        squeeze = command.squeeze_n
        if not command.hold_station:
            held = held + np.clip(command.target - held,
                                  -ceiling / fps, ceiling / fps)
        for _ in range(per_frame):
            body.data.ctrl[:] = computed_torque(body, held, command.squeeze_n)
            mujoco.mj_step(body.model, body.data)

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
            "t": round(now, 4), "phase": int(phase), "links": links,
            "block": _app(body.block()),
            "block_quat": [float(quat[0]), float(quat[1]),
                           float(quat[3]), float(quat[2])],
            # Recorded for the viewer and the grader. The CONTROLLER never
            # reads it -- it works out how far the thing has opened from how far
            # its own hand has travelled.
            "door_deg": round(float(np.degrees(body.data.qpos[slot])), 2),
            "opened_m": round(float(work.mechanism.opened_m), 4),
            "forces": {k: round(v, 2) for k, v in body.forces().items()},
            "opening_m": round(body.opening(), 5),
            "on_bench": bool(on_the_bench(body)),
            "penetration_mm": round(body.penetration_mm(), 3),
        })
        if verbose and index % 60 == 0:
            print(f"  t={now:5.2f} ph{phase} {PHASES[phase][0]:<15} "
                  f"door={frames[-1]['door_deg']:6.1f} "
                  f"open={body.opening() * 100:5.2f} "
                  f"block={np.round(body.block(), 3)} {command.note}")

    document_out = {
        "protocol": "gripper_clip_v1", "embodiment": "gripper",
        "task": "open the cabinet, take out the object, put it on the bench",
        "control": "computed torque", "sensing": sensing, "fps": fps,
        "table_top_m": float(table_top), "block_half_m": [float(v) for v in half],
        "phase_names": [p[0] for p in PHASES],
        "fridge": spec()["scene"]["fridge"],
        "pedestal": spec()["kinematics"].get("pedestal"),
        "achieved": {
            "phases_reached": int(reached) + 1,
            "of_phases": len(PHASES),
            "door_opened_deg": round(max(f["door_deg"] for f in frames), 1),
            "on_bench": bool(frames[-1]["on_bench"]),
            "deepest_penetration_mm": round(
                max(f["penetration_mm"] for f in frames), 3),
        },
        "frames": frames,
    }
    _PUBLIC.mkdir(parents=True, exist_ok=True)
    (_PUBLIC / f"{name}.json").write_text(json.dumps(document_out),
                                          encoding="utf-8")
    return document_out["achieved"]
