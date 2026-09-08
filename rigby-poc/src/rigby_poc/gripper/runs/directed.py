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
from ..decision.lookahead import Foresight, foresee
from ..decision.pick_and_place import (
    object_above_rim_m,
    object_in_target,
    object_over_target_m,
)
from ..decision.planner import Planner
from ..decision.grip import DEFAULT as GRIP_DEFAULT, jaw_command
from ..decision.primitives import object_between_jaws
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
        # ORIENTATION NEEDS MORE THAN A SWAP. _app converts a POSITION from
        # MuJoCo's Z-up to the viewer's Y-up by exchanging y and z, which is
        # fine for a point and wrong for a rotation: exchanging two axes is a
        # reflection, determinant -1, so under it a rotation transforms as
        # P M P^-1 and the quaternion's vector part must be NEGATED as well as
        # permuted. Without the negation the block is drawn mirrored -- it turns
        # the wrong way and looks as though it is not held, while the physics
        # has it clamped to within 2.7 mm over seventeen seconds. Measured on a
        # 55 degree rotation: 0.366 of error, against 0.000 with the negation.
        #
        # The arm escapes this because it is drawn from endpoints, not angles.
        "block_quat": [float(quat[0]), -float(quat[1]),
                       -float(quat[3]), -float(quat[2])],
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


def _thumbnail(body, camera: str, width: int = 192, height: int = 144) -> str:
    """One camera, as a data URI small enough to ship with every progress tick.

    RENDERED AT A SIZE SOMETHING ELSE ALREADY USES, then shrunk in PIL. Every
    distinct size is another MuJoCo Renderer and another GL context, and a run
    died on "Default framebuffer is not complete" with five of them open --
    sensing at two sizes, the planner's two images, and these. Downscaling an
    existing render costs nothing and adds no context.

    And it never raises. A picture for the dashboard must not be able to end a
    forty-call run: if the renderer is unhappy the panel goes blank and the arm
    carries on.
    """
    import base64
    import io

    from PIL import Image

    try:
        raw = body.view(320, 240, camera=camera)
    except Exception:
        return ""
    picture = Image.fromarray(raw).convert("RGB")
    if (width, height) != picture.size:
        picture = picture.resize((width, height))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=70)
    return ("data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii"))


def _said(planner, decision) -> str:
    """What just happened, named accurately enough to act on."""
    if getattr(planner, "held", None) is not None:
        return (f"Model chose {planner.held.metric} = "
                f"{planner.held.value:g}")
    problem = (decision or {}).get("error") if isinstance(decision, dict) else None
    if problem:
        return f"The model could not be reached: {str(problem)[:140]}"
    if not getattr(planner, "calls", 0):
        return ("No model call has completed -- check the API key and the "
                "model name before reading anything into the run")
    return "The model replied without a usable goal"


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
        "block_half_m": [0.025, 0.025, 0.03],
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


#: How often the identified world is re-measured, in frames. At 30 fps this is
#: 10 Hz. Five camera renders cost about 90 ms, and a belief that is corrected
#: ten times a second is corrected far faster than the arm can invalidate it.
_WORLD_EVERY = 3
#: How often to search for a route, in frames. Looking three moves ahead through
#: five thousand poses costs about 350 ms while carrying something, so this
#: cannot run every frame -- and does not need to: what it produces is a
#: setpoint, and the arm spends the frames in between slewing toward it under
#: the rate limits. Every sixth frame is 5 Hz.
_PLAN_EVERY = 6


def run(task: str = "put the orange block into the bin",
        seconds: float = 26.0, fps: int = 30, table_top: float = 0.72,
        planner: Planner | None = None, name: str = "directed-run",
        verbose: bool = True, watch: bool = False,
        output_path: Path | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict:
    """One sentence in; a sequence of numbers pursued until the task is done."""
    # A BLOCK THE JAWS CAN TAKE AT ANY YAW. The old one was 60 x 80 mm in
    # cross-section: 60 mm across the grasp axis, which the 86 mm jaws clear
    # easily, and 100 mm on the diagonal, which they cannot clear at all. The
    # block is a free body and does get knocked askew, and the moment it turns
    # the grasp stops being merely hard and becomes impossible -- with nothing
    # in the readings to say so, since the width the camera measures is the
    # width it happens to see.
    #
    # Square cross-section, 50 mm across, 70.7 mm on the diagonal: 15 mm of
    # margin whichever way it is facing.
    body = make(np.asarray([0.025, 0.025, 0.03]),
                np.asarray([0.0, 0.30, 0.76]), table_top=table_top)
    ceiling = _ceiling()
    planner = planner or Planner(task=task)

    eyes = Senses()
    sight = Foresight(body)
    route = None
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
        # THE BELIEF THE MODEL BUILT IS THE BELIEF THE ARM STEERS BY. Until
        # now these were two different pictures: the model identified the
        # scene into `world`, was shown that world, and then every goal it set
        # was measured against a separate warm-pixel estimate that could not
        # see the bin and got the block's size from a hardcoded literal. The
        # model reasoned about one world and drove another.
        #
        # Refreshed at 10 Hz rather than every frame because it costs 90 ms to
        # look through five cameras and the answer does not change in 33 ms.
        # Between refreshes the belief is held, which is what a belief is for.
        if getattr(planner, "world", None) is not None and planner.world.looks:
            if index % _WORLD_EVERY == 0:
                planner.world.refresh(body, now)
            believed = planner.world.believed(planner.figure or "block")
        else:
            believed = None
        seen = sense(body, eyes, held, squeeze, now, table_top,
                     believed=believed)

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
            # A NEW GOAL INVALIDATES THE ROUTE. The route is only the best way
            # to reach the goal it was planned for; carrying one over to a
            # different goal would mean the arm spending up to six frames
            # driving toward something nobody asked for any more.
            route = None
            if on_progress is not None:
                decision = planner.transcript[-1] if planner.transcript else None
                on_progress({
                    "kind": "model_decision",
                    "stage": "search",
                    # A CALL THAT NEVER HAPPENED IS NOT A MODEL THAT DECLINED.
                    # An expired key produced three "The model did not provide
                    # a usable goal" events and a model_calls count stuck at
                    # zero, which reads as the model being unhelpful and is
                    # actually a 401. The transcript had the real reason all
                    # along; nothing surfaced it, and the run had to be
                    # reproduced by hand to find out. Blame the right thing.
                    "message": _said(planner, decision),
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
        # A DIRECT ACT SKIPS THE SEARCH. The joints go where the act says, at
        # the same rate limit as everything else -- what is bypassed is the
        # question of WHICH joints, not the physics of moving them.
        if getattr(planner, "doing", ""):
            from ..decision.primitives import direct as build_direct

            act = build_direct(planner.doing)
            if act is not None:
                wanted = act.pose(body, held)
                held = held + np.clip(wanted - held,
                                      -ceiling / fps, ceiling / fps)
                squeeze = 0.0
                for _ in range(per_frame):
                    body.data.ctrl[:] = computed_torque(body, held, squeeze)
                    mujoco.mj_step(body.model, body.data)
                frames.append(_frame(body, seen, now, None,
                                     {"part": "direct", "move": planner.doing},
                                     0.0))
                continue

        # THE JAWS FIRST, AND ALWAYS. They are a state, so they are driven every
        # frame whether or not the arm has a goal. Putting this after the
        # `target is None` check meant a planner with nothing to chase left the
        # hand frozen in whatever the fingers last happened to be doing.
        held, squeeze = jaw_command(getattr(planner, "jaws", GRIP_DEFAULT),
                                    body, held)

        target = planner.held
        if target is None:
            for _ in range(per_frame):
                body.data.ctrl[:] = computed_torque(body, held, squeeze)
                mujoco.mj_step(body.model, body.data)
            continue

        # A ROUTE, NOT A NUDGE. The greedy search this replaces scored one
        # move ahead with no idea that anything was in the way, so a probe that
        # drove the block into the outside of the bin wall scored as an
        # improvement -- it was moving the block closer to the bin, and
        # sideways through a wall is closer. Measured: 87 N through the plate
        # until the contact prised the jaws open.
        #
        # Recomputed every _PLAN_EVERY frames rather than every frame, because
        # what comes back is a setpoint and the arm takes several frames to
        # slew to it anyway.
        # WHICH SEARCH IS THE MODEL'S CALL. "direct" is the one-move greedy
        # search: fast, and blind to anything solid. "ahead" routes three moves
        # and refuses the ones that hit something, for about twenty times the
        # cost. The model sets it from what it can see, because whether there
        # is a wall near what you are carrying is exactly the sort of thing a
        # camera answers and a metric does not.
        looking_ahead = getattr(planner, "search", "direct") == "ahead"
        if looking_ahead:
            if index % _PLAN_EVERY == 0 or route is None:
                route = foresee(body, seen, target, planner.spans,
                                start_from=held, sight=sight)
            wanted, error, how = route
        else:
            # Cheap enough to run every frame, which is what it was built for.
            route = None
            wanted, error, how = pursue(body, seen, target, planner.spans,
                                        start_from=held)
        # The search says where the joints should be; the rate limit says how
        # fast they may get there. Same place as everywhere else in this system.
        # OPTIONAL, because it is instrumentation. A planner has to decide --
        # due() and ask() -- and everything else here is this loop telling it
        # how things went. Requiring the newer hooks broke a test double that
        # implemented the actual contract perfectly well, which is a sign the
        # contract had quietly grown rather than that the double was wrong.
        watching = getattr(planner, "observe", None)
        if callable(watching):
            watching(error, how, now)
        # The search moves the ARM. Whatever it thinks the fingers should do is
        # discarded -- the jaws answer to their state, not to a goal.
        jaws_now = held[4:].copy()
        held = held + np.clip(wanted - held, -ceiling / fps, ceiling / fps)
        held[4:] = jaws_now
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
                # BOTH FEEDS, as the model is given them, small enough to send
                # five times a second. A dashboard that draws the simulated
                # body shows where things ARE; these show what the machine can
                # SEE, and the difference between the two is most of what goes
                # wrong here.
                "cameras": {
                    "room": _thumbnail(body, "room"),
                    "gripper": _thumbnail(body, "gripper"),
                },
                # THE DECISIONS SO FAR, shipped with every tick. The transcript
                # used to be written only when a run finished, so an aborted run
                # kept its thousand frames and lost every decision that produced
                # them -- the exact runs most worth reading, since a run is
                # usually stopped because something looked wrong.
                "transcript": list(planner.transcript),
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
