"""The model that decides what should become true, from two camera images.

It never names a motion. It names a NUMBER it wants and which controls may
pursue it, and the greedy search downstairs finds the joint angles. That split
is the point of the whole system: judging what ought to be true next is a
question about the task, and finding the pose that makes it true is a search.

TWO CAMERAS, and they answer different questions. The corner camera shows where
everything is -- the arm, the block, the bin, and their relation, which a camera
buried in the workspace cannot see. The gripper camera shows what the hand is
actually pointed at, in detail, and is the image the perception layer segments.
Neither is sufficient: a plan made only from the corner view cannot tell whether
the jaws are around the block, and one made only from the wrist view does not
know the bin exists.

Everything numeric it reads comes through the same Sensed the controller uses.
It is not shown the object's true pose, because the machine does not have it.
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
from PIL import Image

from ..body.manifest import spec
from ..physics.model import Body
from ..sensing.gripper_camera import Sensed
from .goals import OUTCOMES, READABLE, NumericTarget, readable
from .greedy import changers

PLAN_PROMPT = """You are directing a robot arm with a two-finger parallel gripper.

You do not choose motions. You choose a NUMBER that should become true, and
which controls may be used to reach it. A greedy search finds the joint angles;
it is good at that and cannot do anything else. You are the only part of this
system that can decide what ought to be true next.

Reply as JSON: {"target": <metric>, "value": <number>, "also": [[metric, value,
weight], ...], "using": [controls], "why": "<one sentence>"}.

YOU SEE TWO CAMERAS EVERY TIME, and they answer different questions.
  CORNER  fixed in the corner of the room, sees the whole bench at once: the
          arm, the block, the bin, and how they stand relative to each other.
          Use it to decide WHAT to do and WHERE things are.
  WRIST   bolted to the gripper, looking out along the way the hand reaches.
          Sees detail and only what the hand is pointed at. Use it to judge
          whether the jaws are actually around something. It is also the image
          the machine segments to find the block, so if the block is not in
          this view the machine is working from memory.

THE NUMBERS YOU MAY ASK FOR

Always readable, because they are facts about the arm:
  hand_x_m hand_y_m hand_z_m   where the hand is. The bench runs roughly
                      x -0.25..0.25, y 0.10..0.50, and its surface is z=0.72.
  hand_pointing_down  +1 when the hand points straight down, 0 level. The
                      camera is bolted to the hand, so this is also where the
                      camera looks -- pointing the hand down is how you look at
                      the bench.
  grip_tip_spread_m   the gap between the pads. Open wider than the object
                      before approaching; close after.

Readable only once something has been SEEN, and simply absent from your reading
until then. Asking for one while blind is asking for a number nobody can
compute, and the search will correctly do nothing:
  palm_to_object_m    metres from the gripping surfaces to the object's surface
  object_in_grasp_m   metres from the object to the LINE BETWEEN THE PADS. Near
                      zero means it is INSIDE the opening, which is the
                      condition for closing on it rather than beside it.
  palm_facing         +1 looking straight at the face approached, 0 edge-on
  object_over_target_m  horizontal metres from what is held to the bin's middle
  object_above_rim_m  how far the held object's underside clears the rim.
                      Negative would strike the wall on the way across.

READABLE BUT NOT ASKABLE: object_in_target is the outcome, holding is a state,
object_seen is a fact about the camera. Steer the hand instead.

THE PARTS YOU MAY NAME IN "using", and what each one can do:
{parts_table}Name PACKAGES, not parts and not moves -- the search picks which part and which
move. A package is a group of parts that work together: "move" is the whole arm,
"gripper" is the fingers. Naming a package says what KIND of work is meant to
happen, and leaves how to a search that can measure the result.

Do not try to pick individual joints. Asked to bring the hand closer, an earlier
run named the last hinge alone -- which changes where the hand points, not how
far it reaches -- so nothing could fold the middle segment, and the arm stayed
stretched out past the block for the whole run.

YOU BEGIN BLIND. Nothing object-relative can be read until the camera has found
the block. Point the hand down over the bench until object_seen becomes 1.

NAME MORE THAN ONE NUMBER. Every metric is contested: approaching improves the
distance and destroys the facing; centring the object drags the hand out of
reach. A search told to care about exactly one will pay any price in the others.
Put what must not be lost in "also", weighted -- the number you chase at 1.0,
the ones you protect at 0.3 to 0.6.

WHAT THE TASK NEEDS, IN ORDER: find the block, get the opening around it, close,
lift it clear, carry it over the bin high enough to clear the rim, let go.
Closing is not a commitment -- the jaws stop on the object's width, so an early
close simply arrives and stops. Waiting is the expensive mistake.
"""

def _parts_table() -> str:
    """The parts and their moves, from the manifest that declares them.

    This was prose in the prompt, hand-copied from the manifest, and it went
    stale the instant the parts were renamed -- still offering shoulder, elbow
    and wrist. The model named them, they matched nothing, and the search
    correctly found that no control moved the number. A list of what the body
    can do belongs to the body.
    """
    from .greedy import moves_for, packages

    rows = []
    for group, members in packages().items():
        moves = []
        for part in members:
            moves += [m for m in moves_for(part)
                      if m not in ("hold", "neutral") and m not in moves]
        rows.append(f"  {group:<9} {' '.join(moves)}")
        rows.append(f"  {'':<9} (parts: {', '.join(members)})")
    return chr(10).join(rows)


EXPAND_PROMPT = """You are directing a robot arm with a two-finger parallel
gripper. You will be given one instruction in plain words.

Break it into the SMALLEST NUMBER OF PHYSICAL STEPS that actually have to
happen, in order, each written as a short English sentence naming what moves and
where it goes. Do not describe joint motions, and do not invent steps the
instruction does not require.

Write each step so it names the thing that moves and, where there is one, the
thing it moves relative to -- "pick up the block", "put the block into the bin".
Those two nouns and the relation between them are what the rest of the system
partitions the step into, and a step written without them cannot be partitioned.

Reply as JSON: {"steps": ["...", "..."], "why": "<one sentence>"}.
"""

STEP_PROMPT = """Same rules. Here is what the arm senses now and what it has
just been doing. Give the next number to pursue, or repeat the current one if it
is still the right one and simply has not been reached yet.

You are working through a plan, one step at a time, and you are told which step
you are on. When that step has actually happened in the world -- not when its
number has been reached, but when the thing the step describes is true -- reply
with "step_done": true and give the first number of the NEXT step. A number
being satisfied is not the same as a step being finished, and a step being
finished is not the same as the task being done.

READ packages_that_change_each_number BEFORE CHOOSING. It is measured on this
body, in this pose, this instant. If a number has an EMPTY list, nothing the
body can do will move it now, and asking for it wastes the whole interval until
you are asked again. That is usually a sign the number belongs to a later step:
nothing moves the block toward the bin while the jaws are empty, because moving
the arm does not move a block it is not holding. Grip it first.

Say plainly if the last target was a mistake -- a number that cannot be moved by
the controls you named, or one already satisfied while the task did not advance.
If you are handed back a target you already reached and the task has not moved,
that is the signal to advance the step, not to name it again.

Reply as JSON: {"target": "<name>", "value": <number>,
"also": [["<name>", <number>, <weight>], ...], "using": ["<package>", ...],
"step_done": <true|false>, "why": "<one sentence>"}.
"""


@lru_cache(maxsize=1)
def _load_env() -> bool:
    """Same credentials as the rest of the pipeline, from the same .env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    here = Path(__file__).resolve()
    for candidate in (here.parents[4] / ".env", here.parents[5] / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return True
    return False


def _as_png(pixels: np.ndarray, scale: int = 1) -> str:
    picture = Image.fromarray(pixels)
    if scale != 1:
        picture = picture.resize((picture.width // scale, picture.height // scale))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass
class Planner:
    """Asks a model what number to chase, from two views and the sensed numbers."""

    #: What it was asked to do, in words. The only instruction from outside.
    task: str = "put the block into the bin"
    model: str | None = None
    #: ASKED AGAIN WHEN THE GOAL IS REACHED, not on a clock. The model decides
    #: what should become true; the search makes it true; then, and only then,
    #: there is a new question worth paying for. A timer asks while the arm is
    #: still halfway through the last answer, which spends money to be told the
    #: same thing.
    #:
    #: The clock is only a backstop, for a goal the search cannot reach at all
    #: -- otherwise one impossible target ends the run in silence.
    stuck_after_s: float = 8.0
    #: Hard ceiling on calls per run, because a loop that repeats itself pays
    #: for every repetition.
    max_calls: int = 14
    #: The soonest a new decision may follow the last one, seconds.
    #:
    #: A goal is re-asked the moment it is reached, and a goal can be reached
    #: the instant it is set -- the model asked for the jaws to be at 0.00 cm
    #: when they were already at 0.00 cm, which was true on arrival, so it was
    #: re-asked on the very next frame, and again, and again: six decisions in
    #: two tenths of a second, a fifth of the run's whole budget spent before
    #: the arm had moved. Deciding is not free and a body does not change fast
    #: enough to be worth re-deciding at 30 Hz.
    min_gap_s: float = 0.6
    client: Any | None = None

    #: The steps the instruction was expanded into, and which one is current.
    plan: list = field(default_factory=list, repr=False)
    #: Each step partitioned by Talmy: what moves, with respect to what, along
    #: which path. Structure the model does not have to re-derive every call.
    partitioned: list = field(default_factory=list, repr=False)
    step: int = field(default=0, repr=False)

    held: NumericTarget | None = field(default=None, repr=False)
    spans: dict = field(default_factory=dict, repr=False)
    calls: int = field(default=0, repr=False)
    #: Consecutive re-asks that named an already-reached target unchanged.
    repeats: int = field(default=0, repr=False)
    transcript: list = field(default_factory=list, repr=False)
    _asked_at: float = field(default=-1e9, repr=False)

    def _openai(self):
        if self.client is None:
            _load_env()
            from openai import OpenAI

            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self.client

    def _model(self) -> str:
        return self.model or os.environ.get("RIGBY_VLM_MODEL", "gpt-4o")

    def scene(self) -> dict:
        """What the machine is entitled to know without looking: the furniture."""
        bin_doc = spec()["scene"]["bin"]
        return {
            "task": self.task,
            "bin_centre_xyz": bin_doc["centre"],
            "bin_rim_height_m": bin_doc["rim_height_m"],
            "note": "the bin is fixed furniture; the block's position is not "
                    "known and must be seen",
        }

    def _scene(self):
        """The gripper's world, in the shape Talmy expects to be handed.

        Coordinates are the viewer's Y-up, which is what SceneManifest is
        written in, so the bin's manifest entry goes in unconverted and the
        block's is stated the same way. Talmy only needs to know WHICH things
        are present and what they are called -- it partitions a sentence, it
        does not measure anything -- but the schema insists on real dimensions
        and a graspable socket, and inventing plausible ones would be inventing
        facts. These are the scene's own numbers.
        """
        from ...models import (AffordanceSocket, SceneManifest, SceneObject,
                               Transform, Vec3)

        bin_doc = spec()["scene"]["bin"]
        inner = bin_doc["inner_half_m"]
        return SceneManifest(
            objects=[
                SceneObject(
                    id="block", kind="block",
                    transform=Transform(translation=Vec3(x=0.0, y=0.76, z=0.30)),
                    dimensions_m=Vec3(x=0.06, y=0.08, z=0.06), mass_kg=0.25,
                    sockets=[AffordanceSocket(
                        id="side", transform=Transform(
                            translation=Vec3(x=0.0, y=0.76, z=0.30)),
                        approach_normal=Vec3(x=0.0, y=0.0, z=-1.0),
                        grasp_span_m=0.06)],
                ),
                SceneObject(
                    id="bin", kind="bin",
                    transform=Transform(translation=Vec3(
                        x=float(bin_doc["centre"][0]),
                        y=float(bin_doc["centre"][1]),
                        z=float(bin_doc["centre"][2]))),
                    dimensions_m=Vec3(x=float(inner[0]) * 2, y=float(inner[1]) * 2,
                                      z=float(inner[2]) * 2),
                    sockets=[AffordanceSocket(
                        id="mouth", transform=Transform(translation=Vec3(
                            x=float(bin_doc["centre"][0]),
                            y=float(bin_doc["rim_height_m"]),
                            z=float(bin_doc["centre"][2]))),
                        approach_normal=Vec3(x=0.0, y=1.0, z=0.0),
                        grasp_span_m=0.06)],
                ),
            ],
            support_height_m=0.72,
        )

    def expand(self, task: str | None = None) -> list:
        """Turn one instruction into the steps it actually requires.

        Two stages rather than one, and the reason is that they are different
        questions. "What does 'put the block in the bin' consist of" is answered
        once, from the words. "What number should be true right now" is answered
        repeatedly, from what the cameras show. Folding them together makes the
        model re-derive the whole task on every call, and it drifts.

        Each step is then partitioned by Talmy into FIGURE, GROUND and PATH --
        what moves, with respect to what, and the respect in which it moves. The
        model is handed that partition rather than asked to hold the structure
        of the sentence in its head while also reading two camera images.
        """
        from ...talmy import interpret

        wanted = task or self.task
        try:
            response = self._openai().chat.completions.create(
                model=self._model(),
                messages=[
                    {"role": "system", "content": EXPAND_PROMPT},
                    {"role": "user", "content": json.dumps(
                        {"instruction": wanted, "scene": self.scene()},
                        sort_keys=True)},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=400,
            )
            self.calls += 1
            parsed = json.loads(response.choices[0].message.content or "{}")
            self.plan = [str(s) for s in parsed.get("steps", [])]
        except Exception as error:  # noqa: BLE001
            self.transcript.append({"t": 0.0, "error": str(error)[:200]})
            self.plan = []

        scene = self._scene()
        self.partitioned = []
        for sentence in self.plan:
            situation = interpret(sentence, scene)
            self.partitioned.append(
                None if situation is None else situation.to_dict())
        self.transcript.append({
            "t": 0.0, "instruction": wanted, "steps": list(self.plan),
            "talmy": list(self.partitioned),
            "why": str(parsed.get("why", ""))[:200] if self.plan else "",
        })
        return self.plan

    def due(self, body: Body, seen: Sensed, now: float) -> bool:
        """Time for a new decision: there is none, it is done, or it is stuck."""
        if self.held is None:
            return True
        if self.calls >= self.max_calls:
            return False
        if now - self._asked_at < self.min_gap_s:
            return False
        if self.held.reached(body, seen, self.spans):
            return True
        return now - self._asked_at >= self.stuck_after_s

    def ask(self, body: Body, seen: Sensed, now: float,
            note: str = "") -> NumericTarget | None:
        """One decision, from both cameras and every number the body senses."""
        # THE INSTRUCTION IS EXPANDED BEFORE THE FIRST DECISION. expand() was
        # written, documented and reachable, and nothing ever called it: the
        # whole run went by with an empty plan, every transcript entry reading
        # step_text: null, and the model deciding from the bare sentence and the
        # sensed numbers with no steps and no Talmy partition behind it. It
        # belongs here rather than in the run loop so that expansion happens for
        # any caller, exactly once, on the way to the first decision.
        if not self.plan:
            self.expand()
        numbers = readable(body, seen)
        situation = {
            "instruction": self.task,
            "plan": list(self.plan),
            "step_index": self.step,
            "step": self.plan[self.step] if self.step < len(self.plan) else None,
            "step_partitioned": (self.partitioned[self.step]
                                 if self.step < len(self.partitioned) else None),
            "sensed": numbers,
            "seeing_the_object_now": bool(seen.object_seen),
            "current_target": None if self.held is None else {
                "metric": self.held.metric, "value": self.held.value,
                "also": [list(a) for a in self.held.also],
                "using": list(self.held.using),
            },
            "steps_total": len(self.plan),
            "steps_left": max(0, len(self.plan) - self.step - 1),
            "since_last_decision": note,
            # WHAT MOVES WHAT, measured on this body at this pose. Without it
            # the planner has only the part names to reason from, and the names
            # do not say that the last hinge cannot shorten the arm's reach.
            "packages_that_change_each_number": changers(
                body, seen, tuple(sorted(READABLE))),
            "askable": sorted(READABLE),
            "not_askable": list(OUTCOMES),
        }
        content = [
            {"type": "text", "text": json.dumps(
                {"scene": self.scene(), "now": situation}, sort_keys=True)},
            {"type": "text", "text": "CORNER camera -- the whole bench:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
             + _as_png(body.view(480, 340, camera="room"))}},
            {"type": "text", "text": "GRIPPER camera -- what the hand is pointed "
                                     "at, and what the machine segments:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
             + _as_png(body.view(320, 240, camera="gripper"))}},
        ]
        try:
            response = self._openai().chat.completions.create(
                model=self._model(),
                messages=[
                    {"role": "system",
                     "content": (PLAN_PROMPT.replace("{parts_table}", _parts_table())
                                if self.held is None else STEP_PROMPT)},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=400,
            )
            self.calls += 1
            parsed = json.loads(response.choices[0].message.content or "{}")
        except Exception as error:  # noqa: BLE001
            self.transcript.append({"t": round(now, 2), "error": str(error)[:200]})
            return self.held

        metric = str(parsed.get("target", ""))
        if metric not in READABLE:
            self.transcript.append(
                {"t": round(now, 2), "refused": metric,
                 "why": "not a number this body can be steered by"})
            return self.held

        also = []
        for entry in parsed.get("also") or []:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 \
                    and str(entry[0]) in READABLE:
                also.append((str(entry[0]), float(entry[1]),
                             float(entry[2]) if len(entry) > 2 else 0.5))
        target = NumericTarget(
            metric=metric, value=float(parsed.get("value", 0.0)), set_at_s=now,
            also=tuple(also),
            using=tuple(str(u) for u in (parsed.get("using") or [])),
        )
        # ADVANCING THE PLAN. self.step was read in three places and written in
        # none, so the model was shown step 0 for the whole run: it could never
        # be told a step had finished, and on every re-ask it correctly named
        # step 0's goal again, spending a call to repeat itself. The plan and
        # its Talmy partition were decorative after the first decision.
        #
        # The model says when a step is done, because whether "pick up the
        # block" has happened is a judgement about the world, not a threshold on
        # one number -- a goal can be reached while the step it belongs to has
        # not occurred.
        was = self.step
        if bool(parsed.get("step_done")) and self.step < len(self.plan) - 1:
            self.step += 1
            self.repeats = 0
        else:
            # A satisfied goal handed back unchanged is not progress. If that
            # happens twice running, advance rather than pay a third time for
            # the same answer -- and say so in the transcript, because a plan
            # that moved on for a reason the model did not give is a thing the
            # reader needs to be able to see.
            same = (self.held is not None
                    and self.held.metric == metric
                    and abs(self.held.value - float(target.value)) < 1e-9)
            settled = self.held is not None and self.held.reached(
                body, seen, self.spans)
            self.repeats = self.repeats + 1 if (same and settled) else 0
            if self.repeats >= 2 and self.step < len(self.plan) - 1:
                self.step += 1
                self.repeats = 0
                self.transcript.append({
                    "t": round(now, 2), "step": self.step,
                    "advanced_without_being_told": True,
                    "why": "the same reached target was named twice running",
                })

        self.held = target
        self.spans = target.spans(body, seen)
        self._asked_at = now
        self.transcript.append({
            "t": round(now, 2), "target": metric, "value": target.value,
            "also": [list(a) for a in target.also], "using": list(target.using),
            "step": self.step, "step_was": was,
            "step_text": self.plan[self.step] if self.step < len(self.plan) else None,
            "step_done": bool(parsed.get("step_done")),
            "why": str(parsed.get("why", ""))[:200], "sensed": numbers,
        })
        return target
