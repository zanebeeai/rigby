"""The search half: whichever control moves the number closest to what was asked.

The planner says what should become true. This finds the joint motion that makes
it truer, by trying each control it is allowed to touch at a range of
magnitudes, probing each through the simulation's own kinematics, and keeping
whichever probe scores best against the target.

It is deliberately stupid. It does not know what a grasp is, which phase the
task is in, or what the object is for. It knows one number and which joints it
may turn. Everything clever is upstairs, and everything reliable is here --
which is the trade the whole architecture is built on, because the model cannot
do this search and does not need to, and the search cannot make this judgement
and should not try.

Probing goes through mj_kinematics on the real model, so a candidate cannot
drift from what the body will actually do. The alternative -- a separate
analytic chain -- is a second source of truth, and this project has already paid
for one of those.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..physics.model import JOINTS, Body
from ..sensing.gripper_camera import Sensed
from .goals import NumericTarget
from .solver import _limits

#: HOW FAR TOWARD A NAMED MOVE TO GO. A move names a POSE -- a fraction of the
#: joint's range -- not a nudge, so applying it twice does nothing: the part is
#: already there. Five poses per joint is a lattice far too coarse to settle a
#: number like "point down to 0.95", and the search took one step and then
#: correctly reported that every remaining move made things worse.
#:
#: So each move is tried at a range of amplitudes, interpolating from where the
#: part is now toward where the move would put it. The vocabulary stays coarse
#: and nameable; the amplitude makes it fine. This is what the humanoid's
#: pursue_target does with its magnitudes, and skipping it was my omission.
#: BOTH SIGNS. A negative amount moves AWAY from the named pose, which is what
#: makes this a search rather than a one-way ratchet: with only positive
#: amounts a part can approach each of its named poses and never retreat from
#: one, so the moment every forward move overshoots, every probe scores worse
#: and the search reports that nothing helps. The humanoid's magnitudes have
#: always run -1.0 to 1.0 for this reason; taking only the positive half was my
#: error, and it cost two runs that looked like a coarse vocabulary.
_AMOUNTS = (-1.0, -0.6, -0.3, -0.12, 0.12, 0.3, 0.6, 1.0)


@lru_cache(maxsize=1)
def _vocabulary() -> tuple[dict, dict]:
    """The parts and their named moves, from the manifest that declares them."""
    here = Path(__file__).resolve().parents[1] / "body" / "gripper_parts.v1.json"
    document = json.loads(here.read_text(encoding="utf-8"))
    return document["parts"], document["move_sets"]


@lru_cache(maxsize=1)
def _slots() -> dict[str, int]:
    return {name: index for index, name in enumerate(JOINTS)}


def parts() -> tuple[str, ...]:
    return tuple(_vocabulary()[0])


def moves_for(part: str) -> tuple[str, ...]:
    return tuple(_vocabulary()[1].get(part, {}))


def _pose_for(part: str, move: str, start: np.ndarray, limits: list,
              amount: float = 1.0) -> np.ndarray | None:
    """Where a named move puts the joints, as a fraction of each one's range.

    A move cannot name an out-of-range pose, because it is expressed as a
    fraction of the range rather than as an angle. That is the point of the
    schema and it is why the vocabulary and the limits cannot disagree.
    """
    plan = _vocabulary()[1].get(part, {}).get(move)
    if not plan:
        return None
    slots = _slots()
    trial = start.copy()
    for joint, fraction in plan.items():
        index = slots.get(joint)
        if index is None:
            continue
        low, high = limits[index]
        value = high * float(fraction) if fraction >= 0 else -low * float(fraction)
        # Part of the way there, not all of it, unless asked for all of it.
        blended = start[index] + (value - start[index]) * float(amount)
        trial[index] = float(np.clip(blended, low, high))
    return trial


@lru_cache(maxsize=1)
def packages() -> dict:
    """The named groups of parts the planner may choose between."""
    here = Path(__file__).resolve().parents[1] / "body" / "gripper_parts.v1.json"
    with here.open(encoding="utf-8") as handle:
        return dict(json.load(handle).get("packages", {}))


def _named(using: tuple[str, ...]) -> list[str]:
    """Which PARTS this target allows, from the PACKAGES it names.

    A package name expands to its parts. A bare part name is still honoured,
    because the manifest is the authority on what exists and refusing a real
    part would be pedantry -- but the planner is only ever offered packages.
    Naming nothing means everything, which is the right default: the search
    finding its own way is better than the search being starved.
    """
    known = parts()
    groups = packages()
    wanted: list[str] = []
    for name in using:
        for part in groups.get(name, [name]):
            if part in known and part not in wanted:
                wanted.append(part)
    return wanted or list(known)


def contests(body: Body, seen: Sensed, metrics: tuple[str, ...],
             start_from: np.ndarray | None = None) -> dict[str, list[str]]:
    """For each number, which OTHER numbers get worse when it is driven.

    Measured on this body at this pose, by taking the move that best improves
    each metric and reading what that move did to everything else. Nothing here
    knows what the task is: it is a fact about the mechanism, and it comes out
    different for a bin, a shelf or a doorway.

    It exists because the planner could not see the conflict. Given the whole
    carry it drove "get the block over the bin" alone, then "get it above the
    rim" alone, then the first again -- each undoing the last, three times,
    while using `also` perfectly well on eleven other goals. It was not missing
    the mechanism for protecting a number. It was missing the knowledge that
    these two fight.

    The alternative was a carry_to_bin primitive with the answer written in,
    which works for one bin and teaches the planner nothing.
    """
    from .goals import BEST_AT_ONE, NEEDS_SIGHT, READABLE

    model, data = body.model, body.data
    address = [body.address(name) for name in JOINTS]
    saved_q, saved_v = data.qpos.copy(), data.qvel.copy()
    limits = _limits()
    start = (np.asarray(start_from) if start_from is not None
             else np.asarray([data.qpos[a] for a in address]))

    def read_all():
        out = {}
        for name in metrics:
            fn = READABLE.get(name)
            if fn is None:
                continue
            if name in NEEDS_SIGHT and seen.object_at is None:
                continue
            out[name] = float(fn(body, seen))
        return out

    for slot, value in zip(address, start):
        data.qpos[slot] = value
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    here = read_all()

    def better(name, was, now):
        return (now > was) if name in BEST_AT_ONE else (now < was)

    out: dict[str, list[str]] = {}
    for driven in here:
        best = None
        for part in parts():
            for move in moves_for(part):
                for amount in (-1.0, 1.0):
                    trial = _pose_for(part, move, start, limits, amount)
                    if trial is None or np.allclose(trial, start):
                        continue
                    for slot, value in zip(address, trial):
                        data.qpos[slot] = value
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_camlight(model, data)
                    now = read_all()
                    if driven not in now:
                        continue
                    gain = abs(now[driven] - here[driven])
                    if better(driven, here[driven], now[driven]) and (
                            best is None or gain > best[0]):
                        best = (gain, now)
        if best is None:
            continue
        hurt = [other for other, was in here.items()
                if other != driven and other in best[1]
                and not better(other, was, best[1][other])
                and abs(best[1][other] - was) > 0.01]
        if hurt:
            out[driven] = sorted(hurt)

    data.qpos[:] = saved_q
    data.qvel[:] = saved_v
    mujoco.mj_forward(model, data)
    return out


def changers(body: Body, seen: Sensed,
             metrics: tuple[str, ...],
             start_from: np.ndarray | None = None) -> dict[str, list[str]]:
    """For each number, which parts change it at all. Measured, not assumed.

    Direction-free on purpose: this is a statement about the mechanism, not
    about the current goal. "Which part moves the hand sideways" has one answer
    whatever the hand is trying to do, and it is an answer the planner cannot
    work out from the part names -- which is exactly what went wrong when it
    tried to bring the hand in using the last hinge alone.

    Cheap enough to run on a decision frame and nowhere else.
    """
    from .goals import READABLE

    model, data = body.model, body.data
    address = [body.address(name) for name in JOINTS]
    saved_q, saved_v = data.qpos.copy(), data.qvel.copy()
    limits = _limits()
    start = (np.asarray(start_from) if start_from is not None
             else np.asarray([data.qpos[a] for a in address]))

    def read(name):
        return float(READABLE[name](body, seen))

    for slot, value in zip(address, start):
        data.qpos[slot] = value
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    base = {m: read(m) for m in metrics if m in READABLE}

    out = {m: [] for m in base}
    for group, members in (packages() or {"all": list(parts())}).items():
        touched = {m: 0.0 for m in base}
        for part in members:
            for move in moves_for(part):
                for amount in (-1.0, 1.0):
                    trial = _pose_for(part, move, start, limits, amount)
                    if trial is None or np.allclose(trial, start):
                        continue
                    for slot, value in zip(address, trial):
                        data.qpos[slot] = value
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_camlight(model, data)
                    for m in base:
                        touched[m] = max(touched[m], abs(read(m) - base[m]))
        for m in base:
            if touched[m] > 0.005:
                out[m].append(group)

    data.qpos[:] = saved_q
    data.qvel[:] = saved_v
    mujoco.mj_forward(model, data)
    return out


def movers(body: Body, seen: Sensed, target: NumericTarget,
           spans: dict[str, float],
           start_from: np.ndarray | None = None) -> list[str]:
    """Which parts can actually move this number right now, measured.

    The planner names the parts the search may use, and it can name a set that
    cannot do the job: asked to bring the hand in, it named segment_3 alone --
    the last hinge, which changes where the hand POINTS and not how far out it
    reaches. Only folding segment_2 shortens the arm. The search dutifully
    reported that nothing helped, the arm stayed stretched out past the block,
    and the planner had no way to discover why, so it asked for the same thing
    again on the next call and the one after that.

    So do not guess and do not lecture the model about anatomy: try every part
    and report which ones move the number. This is the same probe pursue()
    already runs, over all parts instead of the named ones, and it is cheap
    because it is only run when a decision is being made.
    """
    model, data = body.model, body.data
    address = [body.address(name) for name in JOINTS]
    saved_q, saved_v = data.qpos.copy(), data.qvel.copy()
    limits = _limits()
    start = (np.asarray(start_from) if start_from is not None
             else np.asarray([data.qpos[a] for a in address]))
    for slot, value in zip(address, start):
        data.qpos[slot] = value
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    here = target.error(body, seen, spans)

    helps = []
    for part in parts():
        best = here
        for move in moves_for(part):
            for amount in _AMOUNTS:
                trial = _pose_for(part, move, start, limits, amount)
                if trial is None or np.allclose(trial, start):
                    continue
                for slot, value in zip(address, trial):
                    data.qpos[slot] = value
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                best = min(best, target.error(body, seen, spans))
        if best < here - 1e-4:
            helps.append(part)

    data.qpos[:] = saved_q
    data.qvel[:] = saved_v
    mujoco.mj_forward(model, data)
    return helps


def pursue(body: Body, seen: Sensed, target: NumericTarget,
           spans: dict[str, float],
           start_from: np.ndarray | None = None
           ) -> tuple[np.ndarray, float, dict[str, Any]]:
    """The joint pose that best serves the target, and what it would achieve.

    Returns the probed qpos for the six joints, the error it reaches, and a note
    saying which control did it -- the note matters because it is what gets fed
    back to the planner next time it is asked to think.
    """
    model, data = body.model, body.data
    address = [body.address(name) for name in JOINTS]
    saved_q = data.qpos.copy()
    saved_v = data.qvel.copy()
    limits = _limits()

    # PROBE FROM WHAT WAS COMMANDED, not from where the body has got to.
    #
    # The controller drives a rate-limited command and the body lags it. Probing
    # from the body's actual pose means every step is measured from somewhere
    # behind the command, so the new target lands short of the old one and drags
    # it backwards -- the command stops advancing and the error settles at a
    # plateau it never leaves. Measured: 1.0 down to 0.31, then back up and
    # stuck at 0.67 for twenty seconds. The search must extend the command, and
    # the command is what it must extend from.
    start = (np.asarray(start_from) if start_from is not None
             else np.asarray([data.qpos[a] for a in address]))
    for slot, value in zip(address, start):
        data.qpos[slot] = value
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    here = target.error(body, seen, spans)
    best = (here, start.copy(), {"part": "-", "move": "hold", "amount": 0.0})

    for part in _named(tuple(target.using)):
        for move in moves_for(part):
            for amount in _AMOUNTS:
                trial = _pose_for(part, move, start, limits, amount)
                if trial is None or np.allclose(trial, start):
                    continue
                for slot, value in zip(address, trial):
                    data.qpos[slot] = value
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                score = target.error(body, seen, spans)
                if score < best[0] - 1e-6:
                    best = (score, trial.copy(),
                            {"part": part, "move": move, "amount": amount})

    data.qpos[:] = saved_q
    data.qvel[:] = saved_v
    mujoco.mj_forward(model, data)
    return best[1], best[0], best[2]
