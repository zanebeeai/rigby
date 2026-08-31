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

import mujoco
import numpy as np

from ..body.manifest import spec
from ..decision.goals import readable
from ..decision.greedy import pursue
from ..decision.pick_and_place import object_in_target
from ..decision.planner import Planner
from ..physics.model import JOINTS, computed_torque, make
from ..sensing.gripper_camera import Senses, sense

_PUBLIC = Path(__file__).resolve().parents[4] / "frontend" / "public"


def _ceiling() -> np.ndarray:
    document = spec().get("rate_limits", {})
    per_joint = document.get("joints_deg_per_s", {})
    return np.asarray(
        [float(np.radians(per_joint.get(name, 120.0))) for name in JOINTS[:4]]
        + [float(document.get("finger_m_per_s", 0.07)) / 2.0] * 2)


def run(task: str = "put the orange block into the bin",
        seconds: float = 26.0, fps: int = 30, table_top: float = 0.72,
        planner: Planner | None = None, name: str = "directed-run",
        verbose: bool = True, watch: bool = False) -> dict:
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
        import mujoco.viewer

        window = mujoco.viewer.launch_passive(
            body.model, body.data, show_left_ui=False, show_right_ui=False)

    for index in range(int(seconds * fps)):
        now = index / fps
        seen = sense(body, eyes, held, squeeze, now, table_top)

        if planner.due(body, seen, now):
            before = planner.held
            reached = before is not None and before.reached(body, seen,
                                                          planner.spans)
            planner.ask(body, seen, now,
                        ("reached it" if reached else note) if before else note)
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
        frames.append({
            "t": round(now, 4),
            "target": target.metric, "value": target.value,
            "part": how["part"], "move": how["move"],
            "error": round(float(error), 4),
            "block": [float(v) for v in body.block()],
            "opening_m": round(body.opening(), 5),
            "in_target": bool(object_in_target(body)),
            "penetration_mm": round(body.penetration_mm(), 3),
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

    heights = [f["block"][2] for f in frames] or [0.0]
    achieved = {
        "in_target": bool(frames[-1]["in_target"]) if frames else False,
        "peak_lift_m": round(max(heights) - heights[0], 5),
        "deepest_penetration_mm": round(
            max((f["penetration_mm"] for f in frames), default=0.0), 3),
        "vlm_calls": planner.calls,
    }
    _PUBLIC.mkdir(parents=True, exist_ok=True)
    (_PUBLIC / f"{name}.json").write_text(json.dumps({
        "protocol": "directed_v1", "fps": fps,
        "task": task,
        "decision": "planner names numbers, greedy search chases them",
        "achieved": achieved,
        "transcript": planner.transcript,
        "frames": frames,
    }), encoding="utf-8")
    return achieved
