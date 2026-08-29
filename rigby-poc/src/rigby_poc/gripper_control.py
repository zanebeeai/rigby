"""The same vocabulary, driving torque instead of welded poses.

Every primitive here answers the same question it answered before -- where
should the joints be, and what should the grip be doing -- but the answer is now
a TARGET that a controller tracks with torque, not a pose the body is pinned to.
The difference is the whole point: a commanded position wins its argument with
the object and produces a grip made of overlap; a commanded torque loses that
argument correctly, and the object pushes back.

The phase sequence, the metrics and the refusals are unchanged from the welded
version and from the humanoid hand before that. That is the claim being tested:
that the decision layer is about grasping rather than about a mechanism.

Coordinates are MuJoCo's throughout -- Z-up, no conversions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gripper import spec
from .gripper_torque import JOINTS, Body

#: The eight phases, unchanged in name and order from the welded gripper and,
#: for the first six, from the humanoid hand.
PHASES: tuple[tuple[str, float], ...] = (
    ("move_to", 1.0),
    ("open_grip", 1.0),
    ("move_to", 1.0),
    ("close_grip", 0.4),
    ("close_grip", 1.0),
    ("lift", 0.05),
    ("carry_over", 0.35),
    ("release", 0.5),
)

_ARRIVED_M = 0.12
_ENGULFED_FRACTION = 0.9
_OPEN_DWELL_S = 0.6
_CLEARANCE_M = 0.07
_OVER_TARGET_M = 0.035
_GRIP_CLEARANCE_M = 0.022
#: Newtons of feed-forward squeeze, commanded rather than hoped for.
_ARRIVE_SQUEEZE_N = 2.0
_CARRY_SQUEEZE_N = 12.0
_HOLDING_N = 0.6


def bin_of() -> dict:
    """The bin, in MuJoCo coordinates."""
    document = spec()["scene"]["bin"]
    centre = document["centre"]
    return {
        "centre": np.asarray([centre[0], centre[2], centre[1]], dtype=float),
        "inner": np.asarray(document["inner_half_m"], dtype=float),
        "rim": float(document["rim_height_m"]),
    }


# ---------------------------------------------------------------------------
# The same metrics, read off a torque-driven body.
# ---------------------------------------------------------------------------

def palm_to_object_m(body: Body) -> float:
    offset = np.abs(body.grasp_centre() - body.block()) - body.block_half
    return float(np.linalg.norm(np.maximum(offset, 0.0)))


def object_in_grasp_m(body: Body) -> float:
    left, right = body.pads()
    span = right - left
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        return float(np.linalg.norm(body.block() - left))
    unit = span / length
    along = float(np.clip(float(np.dot(body.block() - left, unit)), 0.0, length))
    return float(np.linalg.norm(body.block() - (left + unit * along)))


def palm_facing(body: Body) -> float:
    face = chosen_face(body)
    if face is None:
        return 0.0
    return float(np.dot(body.approach(), -face[1]))


def chosen_face(body: Body):
    centre, half = body.block(), body.block_half
    here = body.grasp_centre()
    best = None
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            face = centre + normal * half[axis]
            toward = here - face
            size = float(np.linalg.norm(toward))
            if size < 1e-9:
                continue
            score = float(np.dot(toward / size, normal))
            if best is None or score > best[0]:
                best = (score, face, normal)
    return None if best is None else (best[1], best[2])


def object_over_target_m(body: Body) -> float:
    target = bin_of()["centre"]
    block = body.block()
    return float(np.linalg.norm([block[0] - target[0], block[1] - target[1]]))


def object_above_rim_m(body: Body) -> float:
    return float(body.block()[2] - float(body.block_half[2]) - bin_of()["rim"])


def object_in_target(body: Body) -> bool:
    bin_doc = bin_of()
    block = body.block()
    return bool(
        abs(block[0] - bin_doc["centre"][0]) <= bin_doc["inner"][0]
        and abs(block[1] - bin_doc["centre"][1]) <= bin_doc["inner"][2]
        and block[2] <= bin_doc["rim"]
        and block[2] >= bin_doc["centre"][2] - bin_doc["inner"][1])


def holding(body: Body) -> bool:
    forces = body.forces()
    return (forces.get("finger_left", 0.0) >= _HOLDING_N
            and forces.get("finger_right", 0.0) >= _HOLDING_N)


# ---------------------------------------------------------------------------
# Placing the arm: a search over joint targets, honouring the declared range.
# ---------------------------------------------------------------------------

def _limits() -> list[tuple[float, float]]:
    joints = {j["name"]: j for j in spec()["kinematics"]["joints"]}
    out = []
    for name in JOINTS[:4]:
        low, high = joints[name]["range_deg"]
        out.append((float(np.radians(low)), float(np.radians(high))))
    travel = joints["finger_left"]["range_m"]
    out.append((float(travel[0]), float(travel[1])))
    out.append((float(travel[0]), float(travel[1])))
    return out


def _reach_for(body: Body, goal: np.ndarray, square_to: np.ndarray | None,
               support: float) -> np.ndarray | None:
    """Joint angles that put the grasp centre on ``goal``.

    Solved against the model's own kinematics by probing candidate angles
    through mj_kinematics, so the search cannot drift from what the simulation
    will actually do -- which is what a separate analytic chain eventually does.
    """
    import mujoco

    model, data = body.model, body.data
    limits = _limits()
    saved_q = data.qpos.copy()
    saved_v = data.qvel.copy()
    address = [body.address(n) for n in JOINTS]

    def evaluate(angles: np.ndarray) -> float | None:
        for slot, value in zip(address[:4], angles):
            data.qpos[slot] = value
        mujoco.mj_kinematics(model, data)
        here = body.grasp_centre()
        cost = float(np.linalg.norm(here - goal))
        if square_to is not None:
            cost += (1.0 - float(np.dot(body.approach(), -square_to))) * 0.30
        for name in ("shoulder", "seg1", "seg2", "seg3", "plate_geom"):
            if float(body.geom_at(name)[2]) < support:
                return None
        return cost

    best = np.asarray([data.qpos[a] for a in address[:4]])
    here = evaluate(best)
    if here is None:
        here = 1e6
    for _pass in range(7):
        for index in range(4):
            for step in (0.25, 0.09, 0.03, 0.01):
                for direction in (1.0, -1.0):
                    trial = best.copy()
                    trial[index] += step * direction
                    low, high = limits[index]
                    if not (low <= trial[index] <= high):
                        continue
                    value = evaluate(trial)
                    if value is not None and value < here - 1e-5:
                        best, here = trial, value
    data.qpos[:] = saved_q
    data.qvel[:] = saved_v
    mujoco.mj_forward(model, data)
    return best if here < 1e5 else None


@dataclass
class Command:
    """What the controller should track, and how hard to squeeze."""

    target: np.ndarray
    squeeze_n: float = 0.0
    note: str = ""


def decide(body: Body, phase: int, support: float, now: float) -> Command:
    """One phase's command. The refusals are the same ones the hand makes."""
    name, amount = PHASES[phase]
    q = body.q()
    target = q.copy()
    opening_travel = float(np.min(body.block_half)) + _GRIP_CLEARANCE_M

    if name == "move_to":
        forces = body.forces()
        if (max(forces.values(), default=0.0) >= 2.5 or holding(body)
                or float(np.linalg.norm(body.data.cvel[
                    __import__("mujoco").mj_name2id(
                        body.model, __import__("mujoco").mjtObj.mjOBJ_BODY,
                        "block")][3:])) >= 0.02):
            return Command(target, 0.0, "refused: touching, holding or pushing")
        # And do not advance on the object with a closed hand: the pads would
        # arrive where the block is instead of around it.
        if body.opening() < float(np.min(body.block_half)) * 1.5:
            target[4] = target[5] = opening_travel
            return Command(target, 0.0, "opening first")
        face = chosen_face(body)
        if face is None:
            return Command(target)
        centre, normal = face
        lateral = ((body.grasp_centre() - centre)
                   - normal * float(np.dot(body.grasp_centre() - centre, normal)))
        if float(np.linalg.norm(lateral)) > 0.04:
            goal = centre + normal * 0.14
        else:
            goal = body.block()
        found = _reach_for(body, goal, normal, support)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * float(np.clip(amount, 0, 1))
        return Command(target)

    if name == "open_grip":
        if holding(body):
            return Command(target, 0.0, "refused: holding")
        target[4] = target[5] = opening_travel
        return Command(target)

    if name == "close_grip":
        if object_in_grasp_m(body) > float(np.min(body.block_half)) * 1.3 + 0.01:
            return Command(target, 0.0, "refused: object not between the pads")
        # Close ONTO the object and squeeze with a commanded force. The travel
        # target sits at the object's own half-width, so the fingers are not
        # asked to occupy the space the block is in -- the squeeze does the
        # holding, which is what makes this a grip rather than an overlap.
        thickness = float(spec()["kinematics"]["finger"]["thickness_m"])
        target[4] = target[5] = float(np.min(body.block_half)) + thickness
        firm = float(np.clip(amount, 0.0, 1.0))
        squeeze = _ARRIVE_SQUEEZE_N + (_CARRY_SQUEEZE_N - _ARRIVE_SQUEEZE_N) * firm
        return Command(target, squeeze)

    if name == "lift":
        if not holding(body):
            return Command(target, _CARRY_SQUEEZE_N, "squeezing: not holding yet")
        goal = body.grasp_centre() + np.asarray([0.0, 0.0, 0.10])
        found = _reach_for(body, goal, None, support)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * 0.5
        target[4] = target[5] = q[4]
        return Command(target, _CARRY_SQUEEZE_N)

    if name == "carry_over":
        if not holding(body):
            return Command(target, _CARRY_SQUEEZE_N, "refused: nothing held")
        bin_doc = bin_of()
        carry_offset = body.grasp_centre() - body.block()
        goal = np.asarray([
            bin_doc["centre"][0], bin_doc["centre"][1],
            bin_doc["rim"] + _CLEARANCE_M + float(body.block_half[2]),
        ]) + carry_offset
        found = _reach_for(body, goal, None, support)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * float(np.clip(amount, 0, 1))
        target[4] = target[5] = q[4]
        return Command(target, _CARRY_SQUEEZE_N)

    # release
    if object_over_target_m(body) > _OVER_TARGET_M * 2.0:
        return Command(target, _CARRY_SQUEEZE_N, "refused: not over the bin")
    limits = _limits()
    target[4] = target[5] = limits[4][1]
    return Command(target, 0.0)


def advance(body: Body, phase: int, now: float) -> int:
    """The same gates, on the same numbers."""
    if phase == 0 and palm_to_object_m(body) <= _ARRIVED_M:
        return 1
    if phase == 1 and body.opening() >= float(np.min(body.block_half)) * 2.0:
        # Open when it is OPEN, not when a timer says so. A fixed dwell let the
        # approach resume with the pads 1 cm apart around a 6 cm block, so
        # move_to drove closed fingers into the object and jammed them there --
        # and the phase after that waited forever for a grasp that could not
        # form. A gate on elapsed time cannot tell a hand that opened from one
        # that was blocked.
        return 2
    if phase == 2 and object_in_grasp_m(body) <= float(
            np.min(body.block_half)) * _ENGULFED_FRACTION:
        return 3
    if phase == 3 and holding(body):
        return 4
    if phase == 4 and holding(body):
        return 5
    if phase == 5 and object_above_rim_m(body) >= 0.02:
        return 6
    if phase == 6 and object_over_target_m(body) <= _OVER_TARGET_M and holding(body):
        return 7
    return phase
