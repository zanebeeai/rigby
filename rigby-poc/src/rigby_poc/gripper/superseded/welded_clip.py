"""Export a gripper run as something that can be watched.

The humanoid's clips are bone poses against a rigged mesh. This machine has no
skeleton and no mesh -- it is six joint values and a handful of boxes -- so
forcing it into that format would mean inventing a rig it does not have. It gets
its own format instead: per frame, where each link actually is.

That is the honest version of "the viewer supports two bodies", and it keeps the
gripper's numbers from being laundered through a representation built for
something else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gripper import (
    GripperState, forward, object_above_rim_m, object_in_target,
    object_over_target_m, spec,
)
from .gripper_sim import GripperRun

_PUBLIC = Path(__file__).resolve().parents[4] / "frontend" / "public"


def _links(state: GripperState) -> list[dict]:
    """Each segment as a start and an end, plus the two pads and the plate."""
    place = forward(state)
    joints = [list(map(float, p)) for p in place["joints"]]
    document = spec()
    finger = document["kinematics"]["finger"]
    # WHAT IS ACTUALLY SIMULATED is marked, because it is not everything drawn.
    # The physics contains the two finger boxes, the plate, the block and the
    # table -- nothing else. The arm segments are kinematic scaffolding: they
    # place the gripper and collide with nothing, so an arm shown solid beside a
    # solid block invites exactly the wrong conclusion about what the solver is
    # resolving.
    radii = document["kinematics"].get("link_radius_m", [0.022, 0.019, 0.016])
    out = [
        {"kind": "segment", "from": joints[i], "to": joints[i + 1],
         "radius": float(radii[min(i, len(radii) - 1)]),
         "simulated": False}
        for i in range(len(joints) - 1)
    ]
    for name in ("left_pad", "right_pad"):
        pad = place[name]
        # The pad is a box reaching back toward the plate.
        back = pad - place["approach"] * float(finger["length_m"])
        out.append({"kind": "finger", "from": list(map(float, back)),
                    "to": list(map(float, pad)),
                    "half": [float(finger["thickness_m"]),
                             float(finger["pad_width_m"]) / 2.0],
                    "simulated": True})
    out.append({"kind": "plate", "at": list(map(float, place["plate"])),
                "approach": list(map(float, place["approach"])),
                "across": list(map(float, place["across"])),
                "simulated": True})
    return out


def export(run: GripperRun, block_half: np.ndarray, table_top: float,
           name: str = "gripper-run") -> Path:
    """Write a run where the browser can fetch it."""
    _PUBLIC.mkdir(parents=True, exist_ok=True)
    frames = []
    for state, block, spin, forces, phase, time_s in zip(
            run.states, run.block, run.block_quat, run.forces, run.phases,
            run.times):
        frames.append({
            "t": round(float(time_s), 4),
            "phase": int(phase),
            "links": _links(state),
            "block": [float(v) for v in block],
            # MuJoCo quaternions are wxyz and this scene is Z-up against the
            # viewer's Y-up, so the axes are swapped to match the positions.
            "block_quat": [float(spin[0]), float(spin[1]),
                           float(spin[3]), float(spin[2])],
            "forces": {k: round(float(v), 2) for k, v in forces.items()},
            "opening_m": round(forward(state)["opening"], 5),
            "over_target_m": round(object_over_target_m(block), 4),
            "above_rim_m": round(object_above_rim_m(block, block_half), 4),
            "in_target": bool(object_in_target(block)),
        })
    achieved = run.lift_achieved()
    document = {
        "protocol": "gripper_clip_v1",
        "embodiment": "gripper",
        "fps": 30,
        "table_top_m": float(table_top),
        "block_half_m": [float(v) for v in block_half],
        "phase_names": ["approach", "open", "engulf", "close", "squeeze",
                        "lift", "carry", "release"],
        "simulated_geoms": ["finger_left", "finger_right", "plate", "block", "table"],
        "pedestal": spec()["kinematics"].get("pedestal"),
        "bin": spec().get("scene", {}).get("bin"),
        "support_height_m": spec()["kinematics"].get("support_height_m"),
        "achieved": {k: round(float(v), 5) for k, v in achieved.items()},
        "frames": frames,
    }
    path = _PUBLIC / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
