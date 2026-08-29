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
from .gripper_sense import Sensed
from .gripper_torque import JOINTS, Body

#: The eight phases, unchanged in name and order from the welded gripper and,
#: for the first six, from the humanoid hand.
PHASES: tuple[tuple[str, float], ...] = (
    ("search", 1.0),
    ("move_to", 1.0),
    ("open_grip", 1.0),
    ("move_to", 1.0),
    ("close_grip", 0.4),
    ("close_grip", 1.0),
    ("lift", 0.05),
    ("carry_over", 0.35),
    ("release", 0.5),
)

#: How long the camera dwells on one patch of table before moving to the next.
_SCAN_DWELL_S = 0.75
#: How far above a patch the camera sits while inspecting it, metres. Inside the
#: lens's useful range and far enough back that a patch fills less than the cone.
_SCAN_HEIGHT_M = 0.30

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
# The same metrics -- but read from what the machine can SENSE, not from the
# simulator. Every one of these used to reach into the world for the object's
# true pose and true size. They now take a Sensed, which carries a camera's
# estimate of where the object was last seen and how big it looked, and nothing
# else. A metric that cannot be computed from an observation is a metric a real
# robot cannot have.
# ---------------------------------------------------------------------------

def palm_to_object_m(body: Body, seen: Sensed) -> float:
    if seen.object_at is None or seen.object_size is None:
        return 1.0
    offset = np.abs(body.grasp_centre() - seen.object_at) - seen.object_size
    return float(np.linalg.norm(np.maximum(offset, 0.0)))


def object_in_grasp_m(body: Body, seen: Sensed) -> float:
    if seen.object_at is None:
        return 1.0
    left, right = body.pads()
    span = right - left
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        return float(np.linalg.norm(seen.object_at - left))
    unit = span / length
    along = float(np.clip(float(np.dot(seen.object_at - left, unit)), 0.0, length))
    return float(np.linalg.norm(seen.object_at - (left + unit * along)))


def palm_facing(body: Body, seen: Sensed) -> float:
    face = chosen_face(body, seen)
    if face is None:
        return 0.0
    return float(np.dot(body.approach(), -face[1]))


def chosen_face(body: Body, seen: Sensed):
    if seen.object_at is None or seen.object_size is None:
        return None
    centre, half = seen.object_at, seen.object_size
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


def object_over_target_m(body: Body, seen: Sensed) -> float:
    """How far the HELD object is from over the bin.

    While the object is in the hand its position is known from the hand: the
    grasp centre is where it is, give or take the offset measured when it was
    picked up. That is not a camera reading and does not need to be -- a robot
    holding something knows roughly where it is holding it.
    """
    target = bin_of()["centre"]
    block = body.grasp_centre() if seen.holding() else (
        seen.object_at if seen.object_at is not None else body.grasp_centre())
    return float(np.linalg.norm([block[0] - target[0], block[1] - target[1]]))


def object_above_rim_m(body: Body, seen: Sensed) -> float:
    where = body.grasp_centre() if seen.holding() else (
        seen.object_at if seen.object_at is not None else body.grasp_centre())
    half = seen.object_size[2] if seen.object_size is not None else 0.04
    return float(where[2] - float(half) - bin_of()["rim"])


def object_in_target(body: Body) -> bool:
    """Whether the task succeeded. NOT a control input.

    This one reads the simulator on purpose, because it is the grader rather
    than a sensor: it says whether the run worked, and nothing steers by it.
    Keeping the scoring honest means keeping it OUT of the loop.
    """
    bin_doc = bin_of()
    block = body.block()
    return bool(
        abs(block[0] - bin_doc["centre"][0]) <= bin_doc["inner"][0]
        and abs(block[1] - bin_doc["centre"][1]) <= bin_doc["inner"][2]
        and block[2] <= bin_doc["rim"]
        and block[2] >= bin_doc["centre"][2] - bin_doc["inner"][1])


def holding(seen: Sensed) -> bool:
    """Both fingers meeting resistance while being driven closed.

    No force sensor. Two fingers that will not close further while told to
    close have something between them, which a load cell would report at extra
    cost and no extra information.
    """
    return seen.holding()


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


#: Starting poses for the restarts, in radians: arm folded, arm out, arm high,
#: arm across. Fixed rather than random so a run repeats exactly.
_SEEDS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, -0.6, 1.2, 0.4),
    (0.0, 0.3, -1.0, -0.4),
    (0.8, -0.3, 0.9, 0.2),
    (-0.8, -0.3, 0.9, 0.2),
    (0.0, -1.0, 1.6, 0.6),
)


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

    def descend(seed: np.ndarray, passes: int) -> tuple[np.ndarray, float]:
        best = seed.copy()
        here = evaluate(best)
        if here is None:
            here = 1e6
        for _pass in range(passes):
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
        return best, here

    start = np.asarray([data.qpos[a] for a in address[:4]])
    best, here = descend(start, 7)

    # RESTARTS, and only when the first answer is poor. Coordinate descent moves
    # one joint at a time and accepts only strict improvement, so it cannot
    # cross a ridge that needs two joints turning together -- which is exactly
    # what folding the arm back over a target near the base requires. From the
    # carrying pose it therefore refused to fold, drifted downhill instead, and
    # ploughed the held block twenty centimetres across the table while every
    # frame reported honest progress. The same goal solves to 2 mm from a
    # different starting pose. So try a few, keep the best, and pay for it only
    # on the frames where one start was not enough.
    if here > 0.02:
        for seed in _SEEDS:
            candidate, cost = descend(np.asarray(seed), 5)
            if cost < here:
                best, here = candidate, cost
            if here <= 0.01:
                break

    data.qpos[:] = saved_q
    data.qvel[:] = saved_v
    mujoco.mj_forward(model, data)
    return best if here < 1e5 else None


#: The patches of table the camera visits, in the order it visits them. This is
#: knowledge of the FURNITURE -- where the work surface is and how far across it
#: the arm can reach -- which a machine is entitled to the way it is entitled to
#: know where its own bench is. It contains nothing about the object.
_SCAN_GRID = tuple(
    np.asarray([x, y, 0.0]) for y in (0.30, 0.14, 0.45) for x in (0.0, -0.20, 0.20)
)

_scan_cache: list[np.ndarray] | None = None


def _scan_poses(body: Body, support: float) -> list[np.ndarray]:
    """Arm poses that point the wrist camera down at each patch in turn.

    Solved once. The scan is a fixed property of the bench, so re-deriving it
    every frame would be the same answer at forty times the cost.
    """
    global _scan_cache
    if _scan_cache is not None:
        return _scan_cache
    down = np.asarray([0.0, 0.0, 1.0])
    poses = []
    for patch in _SCAN_GRID:
        at = np.asarray([patch[0], patch[1], support + _SCAN_HEIGHT_M])
        found = _reach_for(body, at, down, support)
        if found is not None:
            poses.append(found)
    _scan_cache = poses
    return poses


@dataclass
class Command:
    """What the controller should track, and how hard to squeeze."""

    target: np.ndarray
    squeeze_n: float = 0.0
    note: str = ""
    #: A refusal means STAY, not "re-aim at wherever I have ended up". Every
    #: refusal used to return the arm's current pose as its target, which under
    #: position welds was a true no-op because the body was always exactly where
    #: it was put. Under torque it is not: the arm sags a millimetre, the next
    #: frame adopts the sag as the goal, and a refusal that repeats for ten
    #: seconds walks the arm down to the table one millimetre at a time while
    #: reporting that it is doing nothing. Holding station means holding the
    #: last command.
    hold_station: bool = False


def decide(body: Body, seen: Sensed, phase: int, support: float,
           now: float) -> Command:
    """One phase's command. The refusals are the same ones the hand makes."""
    name, amount = PHASES[phase]
    q = body.q()
    target = q.copy()

    if name == "search":
        # LOOK FOR IT. With one camera on the wrist the machine does not begin
        # knowing where anything is, and under ground truth that question never
        # came up -- the object's position was simply readable, from the first
        # frame, through the back of the robot's own head. It is not. So the
        # first thing the arm does is sweep the camera across the bench until
        # something turns up, which is what a real one does and what the
        # previous eight phases were quietly excused from.
        poses = _scan_poses(body, support)
        if not poses:
            return Command(target, 0.0, "refused: nowhere to look", hold_station=True)
        target[4] = target[5] = _GRIP_CLEARANCE_M
        at = poses[int(now / _SCAN_DWELL_S) % len(poses)]
        target[:4] = at
        return Command(target, 0.0, "searching")
    # Sized from what the camera estimated, not from what the object is.
    estimate = (float(np.min(seen.object_size)) if seen.object_size is not None
                else 0.03)
    opening_travel = estimate + _GRIP_CLEARANCE_M

    if name == "move_to":
        # The refusals survive losing the sensors, which is the interesting
        # part. "Am I touching something" is now the arm failing to reach where
        # it was sent, and "am I holding something" is two fingers that will not
        # close -- both from encoders. The one refusal that needed the object's
        # velocity is gone: a machine with one wrist camera cannot know that an
        # object it is not looking at is sliding, and pretending otherwise was
        # the cheat.
        if seen.arm_stalled or holding(seen):
            return Command(target, 0.0, "refused: obstructed or already holding", hold_station=True)
        if not seen.object_seen and seen.object_at is None:
            return Command(target, 0.0, "refused: never seen the object", hold_station=True)
        # And do not advance on the object with a closed hand: the pads would
        # arrive where the block is instead of around it.
        if body.opening() < estimate * 1.5:
            target[4] = target[5] = opening_travel
            return Command(target, 0.0, "opening first")
        face = chosen_face(body, seen)
        if face is None:
            return Command(target)
        centre, normal = face
        lateral = ((body.grasp_centre() - centre)
                   - normal * float(np.dot(body.grasp_centre() - centre, normal)))
        if float(np.linalg.norm(lateral)) > 0.04:
            goal = centre + normal * 0.14
        else:
            goal = seen.object_at
        found = _reach_for(body, goal, normal, support)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * float(np.clip(amount, 0, 1))
        return Command(target)

    if name == "open_grip":
        if holding(seen):
            return Command(target, 0.0, "refused: holding", hold_station=True)
        target[4] = target[5] = opening_travel
        return Command(target)

    if name == "close_grip":
        if object_in_grasp_m(body, seen) > estimate * 1.3 + 0.01:
            return Command(target, 0.0, "refused: object not between the pads", hold_station=True)
        # Close ONTO the object and squeeze with a commanded force. The travel
        # target sits at the object's own half-width, so the fingers are not
        # asked to occupy the space the block is in -- the squeeze does the
        # holding, which is what makes this a grip rather than an overlap.
        thickness = float(spec()["kinematics"]["finger"]["thickness_m"])
        target[4] = target[5] = estimate + thickness
        firm = float(np.clip(amount, 0.0, 1.0))
        squeeze = _ARRIVE_SQUEEZE_N + (_CARRY_SQUEEZE_N - _ARRIVE_SQUEEZE_N) * firm
        return Command(target, squeeze)

    if name == "lift":
        if not holding(seen):
            return Command(target, _CARRY_SQUEEZE_N, "squeezing: not holding yet")
        goal = body.grasp_centre() + np.asarray([0.0, 0.0, 0.10])
        found = _reach_for(body, goal, None, support)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * 0.5
        target[4] = target[5] = q[4]
        return Command(target, _CARRY_SQUEEZE_N)

    if name == "carry_over":
        if not holding(seen):
            return Command(target, _CARRY_SQUEEZE_N, "refused: nothing held", hold_station=True)
        bin_doc = bin_of()
        # Drive the GRASP CENTRE to the bin, with no offset. While something is
        # held, where the hand is is where the object is -- that is what closing
        # on it means. The version that corrected by (grasp centre - last seen
        # position) was correcting against a camera fix taken before the pick,
        # frozen because the block ends up nearer than the lens can focus, so
        # the correction grew with every centimetre of lift and walked the goal
        # out of the workspace. A stale reading is worse than no reading when
        # something better is already known.
        goal = np.asarray([
            bin_doc["centre"][0], bin_doc["centre"][1],
            bin_doc["rim"] + _CLEARANCE_M + (
                float(seen.object_size[2]) if seen.object_size is not None
                else 0.04),
        ])
        found = _reach_for(body, goal, None, support)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * float(np.clip(amount, 0, 1))
        target[4] = target[5] = q[4]
        return Command(target, _CARRY_SQUEEZE_N)

    # release
    if object_over_target_m(body, seen) > _OVER_TARGET_M * 2.0:
        return Command(target, _CARRY_SQUEEZE_N, "refused: not over the bin", hold_station=True)
    limits = _limits()
    target[4] = target[5] = limits[4][1]
    return Command(target, 0.0)


def advance(body: Body, seen: Sensed, phase: int, now: float) -> int:
    """The same gates, on the same numbers -- now sensed rather than known."""
    estimate = (float(np.min(seen.object_size)) if seen.object_size is not None
                else 0.03)
    if phase == 0 and seen.object_seen:
        return 1
    if phase == 1 and palm_to_object_m(body, seen) <= _ARRIVED_M:
        return 2
    if phase == 2 and body.opening() >= estimate * 2.0:
        # Open when it is OPEN, not when a timer says so. A fixed dwell let the
        # approach resume with the pads 1 cm apart around a 6 cm block, so
        # move_to drove closed fingers into the object and jammed them there --
        # and the phase after that waited forever for a grasp that could not
        # form. A gate on elapsed time cannot tell a hand that opened from one
        # that was blocked.
        return 3
    if phase == 3 and object_in_grasp_m(body, seen) <= estimate * _ENGULFED_FRACTION:
        return 4
    if phase == 4 and holding(seen):
        return 5
    if phase == 5 and holding(seen):
        return 6
    if phase == 6 and object_above_rim_m(body, seen) >= 0.02:
        return 7
    if phase == 7 and object_over_target_m(body, seen) <= _OVER_TARGET_M             and holding(seen):
        return 8
    return phase
