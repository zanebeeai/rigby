"""The brain: a model that looks at the situation and calls a primitive.

Language and vision in, one primitive call out, about once a second. Everything
below it stays deterministic -- the primitive it names runs for the next second
at the loop's own rate, bounded by the joint speed ceilings, with the simulation
stepping and sensing every frame.

WHY 1 Hz AND NOT EVERY FRAME

A model cannot be asked 30 times a second, and should not be. What changes at 30
Hz is joint angles, which is control; what changes at 1 Hz is what the body
ought to be doing, which is the decision. Holding a primitive between calls is
not a compromise for latency, it is the right granularity: "close the hand" is
an instruction that stays true for a while.

WHAT IT REPLACES

``SuperPrimitiveSelector`` scores every primitive at every magnitude by
simulating each and taking the lowest error -- roughly 400 forward-kinematics
passes a frame. That is hill-climbing on a hand-written error function, and it
has no idea what it is doing. It cannot tell an overshoot from an approach
except through a number I chose, and every failure of the loop has traced back
to one of those numbers being wrong.

The model gets the image, the task, the stage it is on, and the menu. It is
scored the same way, against the same goals, and drops into the same
``choose`` interface, so the two are directly comparable and the loop cannot
tell which is driving.

WHAT IT SEES

An offscreen MuJoCo render from a camera at the head, looking where the head
looks. That is a real image of the hand, the object and the table, not a
description of them -- and it is why the gaze angle is worth measuring again:
where the head points now decides what the model is shown.

The render is of the collision geometry, not the character mesh: capsules for
the digits, a box for the palm and the block. The model is told this, because a
render that looks like a diagram and is described as a photograph invites the
model to explain away what it sees.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .closed_loop import Sensing, Step, super_primitives
from .models import BonePose, Hand

#: Seconds a decision stands before the model is asked again.
DECISION_PERIOD_S = 0.7

#: Offscreen render size. Small on purpose: the model needs to see which side of
#: the block the thumb is on, not read a serial number, and every pixel is
#: latency on a call that already gates the loop.
RENDER_WIDTH = 512
RENDER_HEIGHT = 384

SYSTEM_PROMPT = """You are the motor cortex of a simulated humanoid. Each second you see one \
image and choose ONE action for the body to run until you are asked again.

The image is an offscreen render of COLLISION GEOMETRY from a camera at the robot's head. \
Capsules are finger segments, the flat box is the palm, and the separate box on the surface \
is the object. It is a diagram, not a photograph; do not infer detail that is not drawn.

"your_plan" is the plan YOU wrote for this task, before seeing any of this, and \
"your_step" is where you are in it. Follow it. Advance with "next_stage" when the step's \
"done_when" is true, go back with "previous_stage" when something it assumed has stopped being \
true, and say why either way. Nothing else is enforcing the plan and nothing else will advance \
it: if you never call "next_stage" you will work on step one until the run ends. Change the plan \
when it turns out to be wrong -- you wrote it without seeing the scene.

The only thing required of you is the task itself. You will be asked again in a second.

PREFER SETTING A NUMBER over naming an action. A named action runs unchecked, exactly \
as given. A target is searched, and the search REFUSES any control that would move the \
number the wrong way -- which is the protection you want, because a control's name says \
what it is for and not what it does here. "orient_palm" was named directly eleven times \
in one run to bring the fingertips closer, and every magnitude of it moved them 0.4 to \
1.2 cm further away; set as a target it would have been rejected on the first frame.\n\nSetting a number: reply with "target" (a metric name), "value" (the number you want), and \
"using" -- the list of controls the search may touch to get there. You are not choosing the \
control or the magnitude; the search tries each one you name, thirty times a second, and keeps \
whichever moves the number closest. You are saying which part of the body the problem is in.

Name three to six. "using" may hold any action on the menu, and also single joints written \
"<part>:<move>" when one finger needs adjusting rather than a whole hand re-shaped.

Leaving "using" out makes the search try every control it has, which takes seventeen seconds \
per decision -- most of it spent confirming that the little finger does not help. A shortlist \
that turns out to be the wrong part of the body costs you one decision; you will see the number \
fail to move and can name different controls.

You are not asked on a clock while a number is being driven. The search reports back when it \
is finished with it, and that is when you are woken. "woke_because" says which happened: \
"reached" -- the number arrived, so inspect the result and set the next one; "stuck" -- the \
number stopped moving, so the controls you named are not the ones that move it, or the number \
was the wrong thing to ask for. Say what you now think and set a different target.

Rules that matter more than they look:

A grasp needs the THUMB loaded against at least one FINGER, on opposite faces of the object. \
Fingers loaded without the thumb are pressing the object into free space, which pushes it away \
rather than holding it. If you see contact but no opposition, the answer is usually to bring the \
thumb across, not to close harder.

Closing harder before a pair opposes is a shove. It has moved this object 30 cm across the table.

The aperture between the thumb and fingers must stand roughly square to the palm before closing. \
An aperture lying flat in the palm means the object rests against the palm rather than sitting \
between the digits, and closing from there cannot grip.

"stage_requires" is the whole task, not a sub-goal: the object off the surface. It is the only condition anything checks, and it is checked on the OBJECT -- a hand in the right shape with the object still on the table has achieved nothing.

Read the SPEEDS before deciding. "object_speed_m_s" above zero means the object is already
moving -- you are pushing it, and pushing harder will not grip it. A still frame cannot show
you this, which is why it is given: an object sliding away and one sitting still look identical
in a single image, and repeating the action that moved it is the usual mistake.

If the object is moving and you have no grip, stop pushing and re-form the hand around it.

YOU CAN SET A NUMBER INSTEAD OF AN ACTION, and usually should. Reply with
{"target": "<metric>", "value": <number>, "why": "..."} and a numeric search will drive that
metric there every frame until you change it -- thirty times a second, rather than once per
call. You decide WHAT should be true; it finds which control does it.

Metrics you may target:
  c_closure          PREFER THIS for shaping a grip. -1 is a C that closes AROUND something:
                     the thumb and index antiparallel AND pointing at each other. It reads +1
                     whenever they point away from each other, so it cannot be satisfied by
                     opening the hand flat -- which is what a bare direction test allows, and
                     what a search found when given one.
  ray_dot            just the direction test: -1 is antiparallel, which a splayed-open hand
                     satisfies as easily as a closing C. Rarely what you want alone.
  rays_toward        +1 when each fingertip ray points at the other, negative when they point
                     apart
  ray_gap_m          metres by which the two fingertip rays miss each other; 0 closes the loop
  tips_to_object_m   fingertip mean to the object
  thumb_opposition   +1 when the thumb is across the object from the fingers
  thumb_to_fingers_m thumb tip to the middle of the finger group
  aperture_deg       degrees the apertures stand off the palm

Prefer a target when you want a SHAPE, and an action when you want a discrete move. "Form a C"
is a target: ray_dot = -1.

Magnitude is how much of the action to apply, 0.15 to 1.0. Small values when close and \
adjusting, large when far.

Reply with a JSON object and nothing else:
{"action": "<one action name from the menu>", "magnitude": <number>, "why": "<one short sentence>"}"""



PLAN_PROMPT = """You are planning a physical action for a simulated humanoid hand, and you will
carry it out yourself afterwards.

Write the plan you would actually follow. Break the task into ordered steps, and for each say
what must become TRUE before moving on -- as a number wherever you can, because you are given
those numbers live while executing, and a step whose completion you cannot read is one you
cannot finish.

You know how a hand works. Say what it should do, in order, at the level of detail that makes
the difference between gripping something and pushing it.

Metrics you can read and target while executing:
  c_closure          -1 when thumb and index point straight at each other around something (a C
                     that closes); +1 whenever they point apart, including a hand splayed flat,
                     so it cannot be satisfied by opening
  grip_closure       the same averaged over all four fingers: -1 is a whole hand closed on it
  object_in_grasp_m  metres from the object to the line between thumb and fingers; near 0 means
                     the object is INSIDE the opening rather than beside the hand
  tips_to_object_m   fingertip mean to the object centre
  thumb_opposition   +1 when the thumb is across the object from the fingers
  thumb_to_fingers_m thumb tip to the middle of the finger group
  aperture_deg       degrees the apertures stand off the palm
  ray_dot            raw direction dot product of the thumb and index rays
  rays_toward        +1 when each fingertip ray points at the other

You will also read contact force per digit, the object speed, and where things are.

Reply with JSON and nothing else:
{"plan": [{"step": "<short name>", "do": "<what the hand does>", "done_when": "<condition,
naming a metric and value where you can>"}], "why": "<one sentence>"}"""


@lru_cache(maxsize=1)
def _load_env() -> bool:
    """Load ``.env`` the way the planner does, so one key serves everything.

    The judge's credentials already live there. Reading them from the same
    place means a working judge implies a working selector, rather than the
    selector failing with a credentials error on a machine where the rest of
    the pipeline runs.
    """
    from dotenv import load_dotenv

    here = Path(__file__).resolve()
    for candidate in (
        here.parents[2] / ".env",
        here.parents[3] / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return True
    return False


@dataclass
class VLMSelector:
    """Chooses a primitive from an image, the task, and what the body senses."""

    hand: Hand
    client: Any | None = None
    model: str | None = None
    period_s: float = DECISION_PERIOD_S
    #: Hard ceiling on calls per run. Past it the last decision stands, which
    #: bounds the worst case of a loop that would otherwise keep paying to
    #: repeat itself -- one run spent seventeen calls hovering 4 cm from a block
    #: it had already knocked away.
    max_calls: int = 40
    #: The standing decision, held between calls.
    _held: tuple[str, float] | None = field(default=None, repr=False)
    _decided_at: float = field(default=-1e9, repr=False)
    _renderer: Any | None = field(default=None, repr=False)
    calls: int = field(default=0, repr=False)
    #: Every exchange, so a run can be read back and argued with.
    transcript: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def _model(self) -> str:
        """The judge's vision model by default.

        The judge already looks at rendered frames of this rig and has a model
        configured for it, so the selector uses the same one rather than
        introducing a second choice to keep in sync. ``RIGBY_VLM_MODEL``
        overrides it when the two should differ.
        """
        _load_env()
        return self.model or os.getenv(
            "RIGBY_VLM_MODEL", os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-luna")
        )

    def _openai(self) -> Any:
        if self.client is None:
            _load_env()
            from openai import OpenAI

            self.client = OpenAI()
        return self.client

    def render(self, model: Any, data: Any) -> bytes | None:
        """One PNG from the head's camera, or None if rendering is unavailable.

        Headless rendering needs a GL context and there is not always one. A
        missing image is reported rather than silently swapped for a text-only
        prompt: a "VLM" that never receives an image is an LLM, and the
        difference is exactly what is being tested here.
        """
        import mujoco

        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(model, RENDER_HEIGHT, RENDER_WIDTH)
            camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
            self._renderer.update_scene(data, camera=camera if camera >= 0 else -1)
            pixels = self._renderer.render()
        except Exception:
            return None
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(buffer, format="PNG")
        return buffer.getvalue()

    def _menu(self, step: Step) -> list[dict[str, str]]:
        """The primitives for this stage, plus control of the plan itself."""
        allowed = set(getattr(step, "allows", ()) or ())
        menu = [
            {"action": p.name, "does": p.describes}
            for p in super_primitives(self.hand)
            if (set(p.parts) & set(step.active_parts))
            and (not allowed or p.name in allowed)
        ]
        menu.append({
            "action": "next_stage",
            "does": "this stage is done; move on to the next one",
        })
        menu.append({
            "action": "previous_stage",
            "does": "a precondition this stage assumed is not true; go back",
        })
        menu.append({
            "action": "abandon",
            "does": "the task cannot be done from here -- the object is gone, "
                    "not visible, or out of reach. Stop.",
        })
        return menu

    #: Set by the loop, so the model can see where it is in the plan.
    plan: tuple[str, ...] = ()
    stage_index: int = 0
    #: What is in the world besides the thing being grasped. Set by the loop.
    scene_context: dict[str, Any] = field(default_factory=dict)
    #: The standing numeric goal, if the model set one.
    active_target: Any = None
    #: The plan the model wrote for itself, and where it is in it.
    own_plan: list = field(default_factory=list)
    own_step: int = 0

    def make_plan(self, task: str) -> list:
        """Ask the model to plan the task before doing it.

        The plan used to be mine: a fixed sequence of stages in a JSON file with
        goals I wrote and thresholds I guessed. Nearly every failure in this
        project has been one of those goals -- a shape satisfiable without
        touching the object, a gate that demanded the grip in order to permit
        the grip, a distance measured to a point inside the block. The model
        diagnosed several of them out loud while being made to pursue them.

        So it writes its own, in its own terms, and revises as it goes.
        """
        try:
            response = self._openai().chat.completions.create(
                model=self._model(),
                messages=[
                    {"role": "system", "content": PLAN_PROMPT},
                    {"role": "user", "content": json.dumps(
                        {"task": task, "scene": self.scene_context}, sort_keys=True)},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=900,
            )
            self.calls += 1
            parsed = json.loads(response.choices[0].message.content or "{}")
            self.own_plan = list(parsed.get("plan", []))
            self.transcript.append(
                {"time_s": 0.0, "planned": self.own_plan, "why": parsed.get("why")})
        except Exception as error:  # noqa: BLE001
            self.transcript.append({"time_s": 0.0, "error": str(error)[:200]})
            self.own_plan = []
        return self.own_plan

    def _situation(self, sensing: Sensing, step: Step) -> dict[str, Any]:
        """What the body senses, in the terms the prompt uses."""
        from .skills import aperture_orthogonality_deg, opposition_pairs

        return {
            "plan": list(self.plan),
            "stage_index": self.stage_index,
            "stage": step.name,
            "stage_requires": getattr(step, "requires", {}),
            "stage_error": round(step.error(sensing), 4),
            "stage_reached": bool(step.reached(sensing)),
            "contact_force_n": {k: round(v, 2) for k, v in sensing.contact_force_n.items()},
            "opposition_pairs": opposition_pairs(sensing.contact_force_n, 0.5),
            "aperture_to_palm_deg": round(
                aperture_orthogonality_deg(sensing.bones, self.hand.value), 1
            ),
            "fingertips_to_object_m": round(
                float(np.linalg.norm(sensing.convergence - sensing.object_position)), 4
            ),
            "head_off_axis_deg": round(sensing.gaze_error_deg, 1),
            # WHAT IS ALREADY MOVING. A pose is a still photograph: a block
            # sliding away looks exactly like one at rest, so a controller reads
            # "not there yet" and issues the push that is moving it. Speed is
            # only visible between two frames, and the model sees one.
            "object_speed_m_s": round(
                float(np.linalg.norm(sensing.object_velocity)), 3
            ),
            "object_velocity_m_s": [
                round(float(v), 3) for v in sensing.object_velocity
            ],
            "hand_speed_m_s": round(
                float(np.linalg.norm(sensing.hand_velocity)), 3
            ),
            "scene": self.scene_context,
            "woke_because": self.last_wake,
            "your_plan": self.own_plan,
            "your_step": self.own_step,
            "active_target": (
                self.active_target.to_dict() if self.active_target else None
            ),
            # EVERY metric the model can target, under the names it targets
            # them by. It was being asked to plan in terms of ten numbers, given
            # three of them, and one of those under a different name -- so the
            # "done_when" conditions it wrote for its own plan named quantities
            # it could not read, and no step could ever be concluded finished.
            # Across runs 000488 and 000489 it never advanced its plan once in
            # fifty-two decisions, which is not stubbornness: it had no way to
            # tell that step one was over.
            **self._readable(sensing),
        }

    def _readable(self, sensing: Sensing) -> dict[str, Any]:
        """Every targetable metric, read now.

        Sourced from the same registry the targets resolve against, so the two
        cannot drift apart again: a metric the model can aim at is by
        construction a metric it can see.
        """
        from .closed_loop import METRICS, _register_metrics

        _register_metrics()
        readings: dict[str, Any] = {}
        for name, read in METRICS.items():
            try:
                readings[name] = round(float(read(sensing, self.hand)), 4)
            except Exception:  # noqa: BLE001
                readings[name] = None
        return readings

    def decide(
        self, sensing: Sensing, step: Step, image: bytes | None
    ) -> tuple[str, float]:
        """Ask the model. Returns the action name and its magnitude."""
        menu = self._menu(step)
        situation = self._situation(sensing, step)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {"situation": situation, "menu": menu}, sort_keys=True
                ),
            }
        ]
        if image is not None:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,"
                    + base64.b64encode(image).decode("ascii")
                },
            })
        response = self._openai().chat.completions.create(
            model=self._model(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
            # This model family rejects `max_tokens`. Sent under the name it
            # accepts rather than dropped, so a runaway answer still cannot
            # stall a loop that is gated on the call returning.
            max_completion_tokens=400,
        )
        self.calls += 1
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        # A target is a standing instruction, not one move.
        metric = parsed.get("target") or parsed.get("metric")
        if metric:
            # Which controls the search may touch while chasing the number.
            # Without it the search sweeps two thousand candidates a frame to
            # rediscover that the little finger is not the answer.
            using = parsed.get("using") or parsed.get("with") or []
            self.pending_using = tuple(str(u) for u in using if isinstance(u, str))
            self.transcript.append({
                "time_s": round(sensing.time_s, 2),
                "saw_image": image is not None,
                "situation": situation,
                "target": str(metric),
                "value": float(parsed.get("value", 0.0)),
                "using": list(self.pending_using),
                "why": parsed.get("reason") or parsed.get("why"),
            })
            return f"target:{metric}", float(parsed.get("value", 0.0))
        action = str(parsed.get("action", "hold"))
        amount = float(parsed.get("magnitude", parsed.get("amount", 0.5)))
        names = {entry["action"] for entry in menu} | {"hold"}
        if action not in names:
            # A named action that is not on the menu is a refusal to choose, not
            # a choice. Held rather than guessed at.
            action, amount = "hold", 0.0
        self.transcript.append({
            "time_s": round(sensing.time_s, 2),
            "saw_image": image is not None,
            "situation": situation,
            "chose": action,
            "magnitude": amount,
            "why": parsed.get("reason") or parsed.get("why"),
        })
        return action, float(np.clip(amount, 0.0, 1.5))

    def cadence_for(self, sensing: Sensing) -> float:
        """How often to decide, given how precise the moment is.

        A constant rate is wrong at both ends. Crossing a metre of empty room
        does not need a decision every 0.7 s -- nothing has changed and the
        answer is the same. Closing the last two centimetres onto an object
        needs several, because that is where overshooting happens and where a
        stale decision does damage: the hand carried on past the point it should
        have stopped, because the instruction to keep going was still standing.

        So the rate follows the precision required, which here is distance to
        the thing being worked on.
        """
        reach = float(np.linalg.norm(sensing.convergence - sensing.object_position))
        if reach > 0.30:
            return 1.0
        if reach > 0.12:
            return 0.5
        return 0.25

    #: How close to the number counts as arrived. Metres are held tighter than
    #: unit-scale metrics because a centimetre matters and 0.05 of a dot
    #: product does not.
    def _tolerance(self, metric: str) -> float:
        return 0.01 if metric.endswith("_m") else 0.05

    def _target_settled(self, sensing: Sensing) -> str | None:
        """Whether the search is done with the standing number, and why.

        A clock is the wrong thing to ask the model on. Mid-pursuit the answer
        is always "keep going", and the call is spent confirming it; the moment
        that actually needs a decision is the one the clock cannot see -- the
        number arrived, or stopped moving. So the search reports back instead.

        Returns "reached", "stuck", or None to keep pursuing.
        """
        target = self.active_target
        if target is None:
            return None
        from .closed_loop import NumericTarget  # noqa: F401

        error = target.error(sensing, self.hand)
        if error <= self._tolerance(target.metric):
            return "reached"
        if error < self._best_error - 1e-3:
            self._best_error, self._since_gain = error, 0
        else:
            self._since_gain += 1
        # About a second and a half at frame rate: long enough for a slow
        # approach, short enough not to spend the run pushing a number that has
        # stopped answering.
        return "stuck" if self._since_gain >= 45 else None

    #: Progress of the standing number, for deciding when to wake the model.
    _best_error: float = field(default=float("inf"), repr=False)
    _since_gain: int = field(default=0, repr=False)
    #: Why the model was last woken, shown to it so it knows what happened.
    last_wake: str = ""

    def choose(self, sensing: Sensing, step: Step) -> tuple[str, float, dict[str, Any]]:
        """The loop's interface. Asks the model when the moment warrants it."""
        if self.calls >= self.max_calls:
            return (self._held or ("hold", 0.0))[0], 0.0, {}
        settled = self._target_settled(sensing)
        period = min(self.period_s, self.cadence_for(sensing))
        elapsed = sensing.time_s - self._decided_at
        if self.active_target is not None:
            # While a number is being driven the model is not asked on a timer.
            # It is asked when the search has finished with it -- arrived, or
            # gone as far as it can -- which is the only moment its answer can
            # differ from the one already standing. The long ceiling is a
            # backstop for a metric that drifts without ever settling.
            due = settled is not None or elapsed >= max(period, 6.0)
        else:
            due = self._held is None or elapsed >= period
        if due:
            self.last_wake = settled or ("first" if self._held is None else "timer")
            if settled is not None:
                self.transcript.append({
                    "time_s": round(sensing.time_s, 2),
                    "target_done": self.active_target.metric,
                    "outcome": settled,
                    "error": round(self._best_error, 4),
                })
                self.active_target = None
                self._best_error, self._since_gain = float("inf"), 0
            try:
                action, amount = self.decide(sensing, step, self.pending_image)
            except Exception as error:  # noqa: BLE001
                # A failed call holds the last decision rather than freezing the
                # body, and says so in the transcript.
                self.transcript.append(
                    {"time_s": round(sensing.time_s, 2), "error": str(error)[:200]}
                )
                action, amount = self._held or ("hold", 0.0)
            self._held = (action, amount)
            self._decided_at = sensing.time_s

        action, amount = self._held
        if action.startswith("target:"):
            from .closed_loop import NumericTarget, pursue_target

            target = NumericTarget(action.split(":", 1)[1], amount, sensing.time_s,
                                   using=tuple(self.pending_using))
            if (self.active_target is None
                    or self.active_target.metric != target.metric
                    or self.active_target.value != target.value):
                self._best_error, self._since_gain = float("inf"), 0
            self.active_target = target
            name, chosen, rotations = pursue_target(sensing, self.hand, target)
            return f"{action} via {name}", chosen, rotations
        if action == "abandon":
            return "abandon", 0.0, {}
        if action in ("next_stage", "previous_stage"):
            # The model's own plan is the one being stepped through, so the
            # move lands here rather than in the loop -- the loop has a single
            # open step and would clamp both directions to a no-op.
            if self.own_plan:
                moved = 1 if action == "next_stage" else -1
                self.own_step = int(
                    min(max(self.own_step + moved, 0), len(self.own_plan) - 1)
                )
            # Consumed rather than held: it has already been acted on, and
            # holding it would re-fire the move every frame until the next call.
            self._held = None
            return action, 0.0, {}
        if action == "hold":
            return "hold", 0.0, {}
        for primitive in super_primitives(self.hand):
            if primitive.name == action:
                try:
                    return action, amount, primitive.solve(sensing, self.hand, amount)
                except Exception:  # noqa: BLE001
                    return "hold", 0.0, {}
        return "hold", 0.0, {}

    #: Controls the last decision said the search may use. Held with the
    #: target, since a standing number is pursued between calls and the
    #: controls meant to reach it stand with it.
    pending_using: tuple = ()

    #: Set by the loop each frame, so ``choose`` can stay the shared interface.
    pending_image: bytes | None = field(default=None, repr=False)
