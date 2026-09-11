"""Named actions that fire a group of moves on a schedule and check each step.

A super primitive binds three things that were previously separate and could
therefore disagree: the timed group of body-part moves that make up an action,
the sensor reading that decides whether each step happened, and the order the
steps must happen in.

The binding is the point. A grasp is not "the fingers reached these angles" --
it is "the thumb was loaded before the fingers closed". Kept apart, a hand that
formed a perfect shape around nothing passes a shape check, which is how this
repository shipped a pickup whose rendered hand never touched the block.

Two sensors, because the two failure modes are different. **Force** reads contact
normal force on named digits out of the simulated contacts: it is the finger's
own sense of touch, and it can only report a load that actually existed.
**Vision** reads where the head was looking: a step that could not be seen cannot
be visually verified, and says so instead of passing by default.

Every step reports ``pass``, ``fail`` or ``unverified``. The third is not a
softer failure -- it is the honest answer when no sensor could have observed the
step, and collapsing it into either of the others is what makes a gate lie.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .body_parts import sequence_rotations
from .models import ClipFrame, ContactEvent, Hand, Quat, Vec3

CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "super_primitives.v1.json"
)

SENSOR_KINDS = ("none", "force", "vision")
StepStatus = Literal["pass", "fail", "unverified"]

_DIGITS = ("thumb", "index", "middle", "ring", "little")
#: A digit opposes the thumb; the thumb opposes any finger. Enough to tell a
#: grasp from several digits pressing the same way, without pretending to
#: compute force closure from event records.
_FINGERS = ("index", "middle", "ring", "little")


class SuperPrimitiveError(LookupError):
    """A super primitive the catalog does not define, or defines incoherently."""


@dataclass(frozen=True)
class StepVerdict:
    label: str
    at: float
    window_s: tuple[float, float]
    sensor: str
    status: StepStatus
    detail: str
    measured: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "at": self.at,
            "window_s": list(self.window_s),
            "sensor": self.sensor,
            "status": self.status,
            "detail": self.detail,
            "measured": self.measured,
        }


@dataclass(frozen=True)
class SuperPrimitiveVerdict:
    name: str
    passed: bool
    first_failure: str | None
    steps: tuple[StepVerdict, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "super_primitive_v1",
            "name": self.name,
            "passed": self.passed,
            "first_failure": self.first_failure,
            "steps": [s.to_dict() for s in self.steps],
        }


@lru_cache(maxsize=1)
def catalog() -> dict[str, Any]:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for name, spec in document["super_primitives"].items():
        steps = spec["steps"]
        times = [float(s["at"]) for s in steps]
        if times != sorted(times) or times[0] != 0.0 or times[-1] != 1.0:
            raise SuperPrimitiveError(
                f"{name!r} has steps that are not ordered over [0, 1]"
            )
        for step in steps:
            kind = step["sensor"]["kind"]
            if kind not in SENSOR_KINDS:
                raise SuperPrimitiveError(
                    f"{name}.{step['label']} names unknown sensor {kind!r}"
                )
            unknown = set(step["sensor"].get("digits", ())) - set(_DIGITS)
            if unknown:
                raise SuperPrimitiveError(
                    f"{name}.{step['label']} names unknown digits {sorted(unknown)}"
                )
            # A step may name the body-part move it commands. Validated here so
            # the schedule and the movement vocabulary cannot drift apart: a
            # step demanding thumb load while naming a move that does not exist
            # (or that carries the thumb away from the object) is the kind of
            # disagreement that only shows up as an unexplained failed grasp.
            for part_suffix, move in (step.get("moves") or {}).items():
                from .body_parts import moves_for

                available = moves_for(f"right_{part_suffix}")
                if move not in available:
                    raise SuperPrimitiveError(
                        f"{name}.{step['label']} commands unknown move "
                        f"{part_suffix}.{move!r}"
                    )
            if not step.get("note"):
                raise SuperPrimitiveError(
                    f"{name}.{step['label']} has no note; a step whose acceptance "
                    "criterion is not explained cannot be reviewed when it stops firing"
                )
    return document


def names() -> tuple[str, ...]:
    return tuple(catalog()["super_primitives"])


def resolve(text: str) -> str | None:
    """The super primitive a command asks for, longest alias winning."""
    lowered = f" {text.lower().strip()} "
    best: tuple[int, str] | None = None
    for name, spec in catalog()["super_primitives"].items():
        for alias in spec["aliases"]:
            if f" {alias.lower()} " in lowered or lowered.strip() == alias.lower():
                if best is None or len(alias) > best[0]:
                    best = (len(alias), name)
    return best[1] if best else None


def spec_for(name: str) -> dict[str, Any]:
    document = catalog()["super_primitives"]
    if name not in document:
        raise SuperPrimitiveError(f"no such super primitive: {name}")
    return document[name]


def expand(name: str, hand: Hand, progress: float) -> dict[str, Quat]:
    """The bone rotations this super primitive asks for at ``progress``.

    Delegates to the body-part sequence, so a super primitive cannot express a
    pose the movement vocabulary could not -- and therefore cannot express one
    outside the ROM envelope the vocabulary is written in.
    """
    sequence = spec_for(name)["sequence"]
    return sequence_rotations(f"{hand.value}_{sequence}", progress)


def _window(steps: list[dict[str, Any]], index: int, duration_s: float) -> tuple[float, float]:
    """The interval a step's evidence must fall in.

    A step is a checkpoint, so its window is symmetric about its own time,
    reaching halfway to the neighbour on each side. The obvious alternative --
    "the span since the previous step" -- degenerates at both ends: the first
    step at 0.0 gets a zero-width window nothing can satisfy, and a two-step
    action like ``release`` gets two windows that both cover the whole clip, so
    "was it held" and "is it now clear" are asked of the same evidence and
    cannot both be true.
    """
    at = float(steps[index]["at"])
    previous = float(steps[index - 1]["at"]) if index > 0 else at
    following = float(steps[index + 1]["at"]) if index + 1 < len(steps) else at
    start = at if index == 0 else (previous + at) / 2.0
    end = at if index + 1 == len(steps) else (at + following) / 2.0
    if index == 0:
        start = 0.0
    if index + 1 == len(steps):
        end = 1.0
    return (start * duration_s, end * duration_s)


def _force_verdict(
    sensor: dict[str, Any],
    contacts: list[ContactEvent],
    window: tuple[float, float],
) -> tuple[StepStatus, str, dict[str, Any]]:
    inside = [
        c for c in contacts if window[0] - 1e-9 <= c.time_s <= window[1] + 1e-9
    ]
    peak: dict[str, float] = {}
    for contact in inside:
        peak[contact.digit] = max(peak.get(contact.digit, 0.0), contact.normal_force_n)
    measured = {"peak_normal_n": {k: round(v, 4) for k, v in sorted(peak.items())}}

    # The inverse criterion: this step passes on the ABSENCE of load.
    if "max_normal_n" in sensor:
        limit = float(sensor["max_normal_n"])
        worst = max(peak.values(), default=0.0)
        measured["max_normal_n"] = limit
        if worst <= limit:
            return "pass", f"no digit loaded above {limit} N", measured
        return "fail", f"a digit still carries {worst:.3f} N", measured

    wanted = list(sensor.get("digits", ()))
    threshold = float(sensor.get("min_normal_n", 0.0))
    loaded = [d for d in wanted if peak.get(d, 0.0) >= threshold]
    measured["loaded"] = loaded
    measured["min_normal_n"] = threshold

    if sensor.get("require_all") and len(loaded) != len(wanted):
        missing = sorted(set(wanted) - set(loaded))
        return "fail", f"not loaded above {threshold} N: {missing}", measured
    minimum = int(sensor.get("minimum_count", len(wanted) if sensor.get("require_all") else 1))
    if len(loaded) < minimum:
        return (
            "fail",
            f"{len(loaded)} of a required {minimum} digits loaded above {threshold} N",
            measured,
        )
    if sensor.get("require_opposition"):
        # Opposition is judged over EVERY digit loaded in the window, not just
        # the ones this step asked about. close_fingers asks about the four
        # fingers; the thumb it opposes was loaded by the previous step and is
        # still holding. Checking only the requested subset made opposition
        # unsatisfiable there by construction.
        #
        # Thumb against any finger. A record of events cannot prove force
        # closure, so this is deliberately the weaker claim it can support.
        held = [d for d, f in peak.items() if f >= threshold]
        opposed = "thumb" in held and any(f in held for f in _FINGERS)
        measured["opposing_candidates"] = sorted(held)
        measured["opposed"] = opposed
        if not opposed:
            return "fail", "loaded digits are not in opposition", measured
    return "pass", f"{len(loaded)} digits loaded above {threshold} N", measured


def _vision_verdict(
    sensor: dict[str, Any],
    frames: list[ClipFrame],
    window: tuple[float, float],
    watch_target: Vec3 | np.ndarray | None,
) -> tuple[StepStatus, str, dict[str, Any]]:
    if watch_target is None:
        return "unverified", "no target was supplied for the eye to watch", {}
    from .gaze_controller import gaze_error_deg

    inside = [f for f in frames if window[0] - 1e-9 <= f.time_s <= window[1] + 1e-9]
    if not inside:
        return "unverified", "no frame falls inside this step's window", {}
    errors = [gaze_error_deg(f.bones, watch_target) for f in inside]
    limit = float(sensor.get("max_off_axis_deg", 35.0))
    best = float(min(errors))
    measured = {
        "best_off_axis_deg": round(best, 3),
        "mean_off_axis_deg": round(float(np.mean(errors)), 3),
        "max_off_axis_deg": limit,
        "watch": sensor.get("watch", "object"),
    }
    if best <= limit:
        return "pass", f"target held within {best:.1f} deg of the gaze axis", measured
    return "fail", f"target never came within {limit} deg (best {best:.1f})", measured


def evaluate(
    name: str,
    *,
    contacts: list[ContactEvent],
    frames: list[ClipFrame],
    duration_s: float,
    object_position: Vec3 | np.ndarray | None = None,
    grasp_position: Vec3 | np.ndarray | None = None,
) -> SuperPrimitiveVerdict:
    """Judge each step by the sensor it declared, in order."""
    spec = spec_for(name)
    steps = spec["steps"]
    verdicts: list[StepVerdict] = []
    for index, step in enumerate(steps):
        sensor = step["sensor"]
        window = _window(steps, index, duration_s)
        kind = sensor["kind"]
        if kind == "none":
            status, detail, measured = (
                "unverified",
                "positioning step with nothing to sense yet",
                {},
            )
        elif kind == "force":
            status, detail, measured = _force_verdict(sensor, contacts, window)
        else:
            target = (
                grasp_position if sensor.get("watch") == "grasp" else object_position
            )
            status, detail, measured = _vision_verdict(sensor, frames, window, target)
        verdicts.append(
            StepVerdict(
                label=step["label"],
                at=float(step["at"]),
                window_s=window,
                sensor=kind,
                status=status,
                detail=detail,
                measured=measured,
            )
        )
    failure = next((v.label for v in verdicts if v.status == "fail"), None)
    return SuperPrimitiveVerdict(
        name=name,
        passed=failure is None
        and any(v.status == "pass" for v in verdicts),
        first_failure=failure,
        steps=tuple(verdicts),
    )
