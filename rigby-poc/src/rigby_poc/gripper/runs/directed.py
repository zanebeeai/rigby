"""Run the arm under a planner: it names numbers, a greedy search chases them.

No phase list. Nothing here knows the order of a pick-and-place, that a grip
comes before a lift, or that the bin is the destination. The sequence is
whatever the planner asks for, one number at a time, and the only fixed
machinery is: read the sensors, ask when due, chase what was asked.

That is the difference between this and runs/pick_and_place.py, which executes a
sequence I wrote. The body, the physics, the sensing and the search are the
same; what is removed is my judgement about what to do next.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from ..body.manifest import spec
from ..decision.goals import readable
from ..decision.greedy import pursue
from ..decision.pick_and_place import (
    object_above_rim_m,
    object_in_target,
    object_over_target_m,
)
from ..decision.planner import Planner
from ..physics.model import JOINTS, computed_torque, make
from ..sensing.gripper_camera import Senses, sense

_PUBLIC = Path(__file__).resolve().parents[4] / "frontend" / "public"


def _app(point: np.ndarray) -> list[float]:
    """MuJoCo Z-up to the viewer's Y-up, at the serialization edge."""
    return [float(point[0]), float(point[2]), float(point[1])]


def _frame(body, seen, now: float, target, how: dict, error: float) -> dict[str, Any]:
    """The actual simulated body state, suitable for live playback."""
    radii = spec()["kinematics"].get("link_radius_m", [0.022, 0.019, 0.016])
    finger = spec()["kinematics"]["finger"]
    joints = [body.body_at(n) for n in ("base", "link1", "link2", "link3", "plate")]
    links: list[dict[str, Any]] = [
        {
            "kind": "segment",
            "from": _app(joints[index]),
            "to": _app(joints[index + 1]),
            "radius": float(radii[min(index, len(radii) - 1)]),
            "simulated": False,
        }
        for index in range(len(joints) - 1)
    ]
    for pad in ("left_geom", "right_geom"):
        centre = body.geom_at(pad)
        back = centre - body.approach() * float(finger["length_m"]) / 2.0
        tip = centre + body.approach() * float(finger["length_m"]) / 2.0
        links.append({
            "kind": "finger",
            "from": _app(back),
            "to": _app(tip),
            "half": [float(finger["thickness_m"]), float(finger["pad_width_m"]) / 2.0],
            "simulated": True,
        })
    links.append({
        "kind": "plate",
        "at": _app(body.body_at("plate")),
        "approach": _app(body.approach()),
        "across": _app(np.asarray([1.0, 0.0, 0.0])),
        "simulated": True,
    })
    quat = body.block_quat()
    return {
        "t": round(now, 4),
        "phase": 0,
        "links": links,
        "block": _app(body.block()),
        "block_quat": [float(quat[0]), float(quat[1]), float(quat[3]), float(quat[2])],
        "forces": {key: round(value, 2) for key, value in body.forces().items()},
        "opening_m": round(body.opening(), 5),
        "over_target_m": round(object_over_target_m(body, seen), 4),
        "above_rim_m": round(object_above_rim_m(body, seen), 4),
        "object_seen": bool(seen.object_seen),
        "holding": bool(seen.holding()),
        "in_target": bool(object_in_target(body)),
        "penetration_mm": round(body.penetration_mm(), 3),
        "target": None if target is None else target.metric,
        "target_value": None if target is None else float(target.value),
        "part": str(how.get("part", "-")),
        "move": str(how.get("move", "hold")),
        "error": round(float(error), 4),
    }


def _clip_shell(task: str, fps: int, table_top: float) -> dict[str, Any]:
    bin_doc = spec()["scene"]["bin"]
    return {
        "protocol": "directed_v2",
        "embodiment": "gripper",
        "control": "VLM numeric goals; greedy named-move search; computed torque",
        "sensing": "room and gripper RGB cameras, range, joint encoders and tip force",
        "task": task,
        "fps": fps,
        "table_top_m": float(table_top),
        "block_half_m": [0.03, 0.04, 0.03],
        "phase_names": ["directed"],
        "pedestal": spec()["kinematics"].get("pedestal"),
        "bin": bin_doc,
        "achieved": {},
        "transcript": [],
    }


def _ceiling() -> np.ndarray:
    document = spec().get("rate_limits", {})
    per_joint = document.get("joints_deg_per_s", {})
    return np.asarray(
        [float(np.radians(per_joint.get(name, 120.0))) for name in JOINTS[:4]]
        + [float(document.get("finger_m_per_s", 0.07)) / 2.0] * 2)


def run(task: str = "put the orange block into the bin",
        seconds: float = 26.0, fps: int = 30, table_top: float = 0.72,
        planner: Planner | None = None, name: str = "directed-run",
        verbose: bool = True, watch: bool = False,
        output_path: Path | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict:
    """One sentence in; a sequence of numbers pursued until the task is done."""
    body = make(np.asarray([0.03, 0.04, 0.03]), np.asarray([0.0, 0.30, 0.76]),
                table_top=table_top)
    ceiling = _ceiling()
    planner = planner or Planner(task=task)

    eyes = Senses()
    held = np.asarray(body.q())
    squeeze = 0.0
    per_frame = max(1, int(round((1.0 / fps) / body.model.opt.timestep)))
    frames: list[dict] = []
    note = "just started"

    # A window onto the same simulation, for a person to watch. It only reads
    # the state the loop already computed, so a run with the window open and a
    # run without it are the same run.
    window = None
    if watch:
        from mujoco import viewer as mujoco_viewer

        window = mujoco_viewer.launch_passive(
            body.model, body.data, show_left_ui=False, show_right_ui=False)

    for index in range(int(seconds * fps)):
        now = index / fps
        seen = sense(body, eyes, held, squeeze, now, table_top)

        if planner.due(body, seen, now):
            before = planner.held
            reached = before is not None and before.reached(body, seen,
                                                          planner.spans)
            if on_progress is not None:
                on_progress({
                    "kind": "model_call_started",
                    "stage": "model",
                    "message": "The model is choosing the next measurable goal",
                    "model_calls": planner.calls,
                })
            planner.ask(body, seen, now,
                        ("reached it" if reached else note) if before else note)
            if on_progress is not None:
                decision = planner.transcript[-1] if planner.transcript else None
                on_progress({
                    "kind": "model_decision",
                    "stage": "search",
                    "message": (
                        f"Model chose {planner.held.metric} = {planner.held.value:g}"
                        if planner.held is not None
                        else "The model did not provide a usable goal"
                    ),
                    "model_calls": planner.calls,
                    "detail": decision,
                })
            if planner.held is not None and planner.held is not before:
                note = ""
                if verbose:
                    held_target = planner.held
                    print(f"  t={now:5.1f} PLAN {held_target.metric}"
                          f"={held_target.value} also="
                          f"{[list(a) for a in held_target.also]} "
                          f"using={list(held_target.using)}")
        target = planner.held
        if target is None:
            continue

        wanted, error, how = pursue(body, seen, target, planner.spans,
                                    start_from=held)
        # The search says where the joints should be; the rate limit says how
        # fast they may get there. Same place as everywhere else in this system.
        held = held + np.clip(wanted - held, -ceiling / fps, ceiling / fps)
        # SQUEEZE UNLESS THE PLAN ASKS FOR AN OPENING. Stateless, and that is
        # the point: two cleverer versions of this rule both failed, and both
        # failed by inferring a persistent condition from an instantaneous one.
        #
        # First it squeezed only while the search was actively closing, so the
        # moment the plan moved on to lifting nothing counted as closing and the
        # block was dropped. Then it latched on `holding`, which flickers true
        # for a few frames while the jaws are still travelling -- so the plan
        # stepped off the grasp early, the jaws stopped closing at 4.9 cm of
        # travel, and the arm lifted nothing.
        #
        # What actually determines whether the hand should be gripping is what
        # the plan is trying to do. If the current goal wants the jaws WIDER
        # than they are, it is an opening and the squeeze comes off. Anything
        # else, hold on.
        wants_wider = False
        for metric, value, _weight in target.terms():
            if metric == "grip_tip_spread_m":
                wants_wider = float(value) > body.opening() + 0.004
        squeeze = 0.0 if wants_wider else 12.0

        for _ in range(per_frame):
            body.data.ctrl[:] = computed_torque(body, held, squeeze)
            mujoco.mj_step(body.model, body.data)

        if window is not None:
            if not window.is_running():
                break
            window.sync()

        if index % 30 == 0:
            note = (f"chasing {target.metric}={target.value} with "
                    f"{how['part']}/{how['move']}; error {error:.3f}; "
                    f"{'holding' if seen.holding() else 'not holding'}")
        frame = _frame(body, seen, now, target, how, error)
        frames.append(frame)
        if on_progress is not None and index % max(1, fps // 5) == 0:
            on_progress({
                "kind": "frame",
                "elapsed_s": now,
                "progress": min(1.0, now / max(seconds, 1e-6)),
                "model_calls": planner.calls,
                "latest": {
                    "target": target.metric,
                    "value": target.value,
                    "part": how["part"],
                    "move": how["move"],
                    "error": round(float(error), 4),
                    "holding": bool(seen.holding()),
                    "object_seen": bool(seen.object_seen),
                    "in_target": bool(frame["in_target"]),
                },
                "message": f"Trying {how['part']} / {how['move']} toward {target.metric}",
                "frame": frame,
                "clip": _clip_shell(task, fps, table_top),
            })
        if verbose and index % 60 == 0:
            print(f"  t={now:5.1f} {target.metric:<22}->{target.value:7.3f} "
                  f"via {how['part']}/{how['move']:<18} err={error:6.3f} "
                  f"open={body.opening() * 100:5.2f} "
                  f"sq={squeeze:4.1f} q4={np.asarray(body.q())[4]:.3f} "
                  f"w4={wanted[4]:.3f} "
                  f"{'HOLDING' if seen.holding() else '       '} "
                  f"{'IN BIN' if object_in_target(body) else ''}")

    if window is not None:
        window.close()

    heights = [f["block"][1] for f in frames] or [0.0]
    achieved = {
        "in_target": bool(frames[-1]["in_target"]) if frames else False,
        "peak_lift_m": round(max(heights) - heights[0], 5),
        "deepest_penetration_mm": round(
            max((f["penetration_mm"] for f in frames), default=0.0), 3),
        "vlm_calls": planner.calls,
    }
    document = _clip_shell(task, fps, table_top)
    document.update(achieved=achieved, transcript=planner.transcript, frames=frames)
    destination = output_path or (_PUBLIC / f"{name}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document), encoding="utf-8")
    return achieved
