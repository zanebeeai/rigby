"""What to do, in what order, and when each step is finished.

The phases, the metrics they are judged by, and the refusals. This is the layer
the whole project is about: it names what a body is being ASKED to do in terms
that are true of grasping rather than of this mechanism, which is why the same
sequence drove a nineteen-joint humanoid hand and a two-finger gripper.

Every metric here takes an OBSERVATION rather than reading the world. A metric
that cannot be computed from what the machine can sense is a metric a real robot
cannot have.
"""

from __future__ import annotations

import numpy as np

from ..body.manifest import spec
from ..physics.model import JOINTS, Body
from ..sensing.wrist_camera import Sensed
from .solver import Command, _limits, _reach_for, _shut


#: The eight phases, unchanged in name and order from the welded gripper and,
#: for the first six, from the humanoid hand.
PHASES: tuple[tuple[str, float, str], ...] = (
    ("search", 1.0, "seen"),
    ("inspect", 0.5, "vantage"),
    ("move_to", 1.0, "near"),
    ("open_grip", 1.0, "opened"),
    ("move_to", 1.0, "engulfed"),
    ("close_grip", 0.4, "gripped"),
    ("close_grip", 1.0, "gripped"),
    ("lift", 0.05, "above_rim"),
    ("carry_over", 0.35, "over_target"),
    ("release", 0.5, "done"),
)

#: Each phase names the gate that ends it, rather than the gate being keyed by
#: position in the list. Two separate bugs came from the old arrangement:
#: inserting the search phase renumbered every gate after it, and inserting this
#: one would have done it again. A phase's exit condition belongs to the phase.

#: How far above what it believes it is looking at the hand goes to take its
#: careful look, metres.
_INSPECT_HEIGHT_M = 0.24
#: Close enough to that vantage to call the look a good one.
_VANTAGE_M = 0.05
#: Once a fresh look moves the belief less than this, looking again is not
#: buying anything, metres.
_SETTLED_M = 0.004

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
#: Over what distance the approach ANGLE is allowed to matter, metres.
#:
#: The ranked objective scores preferences at zero until the hand is within
#: `tolerance` of the goal, and the default tolerance is millimetric because
#: that is what "arrived" means for a grasp. But for an APPROACH, arriving
#: pointed the wrong way is not a near miss to be corrected at the end -- the
#: hand has to come in along the right line the whole way, or it arrives having
#: swung through the object. Left at the default the grasp still worked and the
#: pose it grasped from was different enough that the carry afterwards ended up
#: 41 cm from the bin. So the tolerance is the standoff scale here: this is the
#: distance over which "how am I pointed" is part of the task rather than a
#: tie-break.
_AIM_OVER_M = 0.16
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
        found = _reach_for(body, at, down, support, aim="camera")
        if found is not None:
            poses.append(found)
    _scan_cache = poses
    return poses


def decide(body: Body, seen: Sensed, phase: int, support: float,
           now: float) -> Command:
    """One phase's command. The refusals are the same ones the hand makes."""
    name, amount, _gate = PHASES[phase]
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
    # OPEN ALL THE WAY. The opening used to be set to the estimated width plus
    # clearance, which needs the estimate to be right, and from directly above
    # it cannot be: a camera looking straight down at something sees its
    # footprint and NOTHING of its height, so the "height" in that estimate is
    # really a second horizontal extent. Taking the smallest of the three then
    # picked a number that had nothing to do with the width being gripped, and
    # the fingers opened to 5.7 cm around a 6.0 cm block and pressed on top of
    # it. Nothing is lost by opening fully before the approach, and an object
    # too wide for the jaws is a refusal, not a target width.
    opening_travel = _limits()[4][1]

    if name == "inspect":
        # GO AND LOOK PROPERLY. The scan finds the object from wherever the
        # sweep happened to be pointing, which is almost never square to it, and
        # a single oblique look at a quarter of a metre lands 38 mm out with the
        # size half again too big. Committing to that put the hand six
        # centimetres from the block with the block four centimetres off the
        # grip axis, and no amount of careful approaching fixed it, because
        # nothing was ever going to correct the belief.
        #
        # So before reaching for anything, move to directly above where it is
        # believed to be, square on, at the range the estimate is actually good
        # at, and look again. The belief updates while the hand travels, so the
        # vantage refines itself: coarse look, better look, then commit.
        if seen.object_at is None:
            return Command(target, 0.0, "refused: nothing to inspect",
                           hold_station=True)
        goal = np.asarray(seen.object_at) + np.asarray([0.0, 0.0,
                                                        _INSPECT_HEIGHT_M])
        found = _reach_for(body, goal, np.asarray([0.0, 0.0, 1.0]), support,
                           aim="camera")
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * float(np.clip(amount, 0, 1))
        target[4] = target[5] = opening_travel
        return Command(target)

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
        if seen.q[4] < opening_travel - 0.004:
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
        if object_in_grasp_m(body, seen) > max(0.03, body.opening() * 0.7):
            return Command(target, 0.0, "refused: object not between the pads", hold_station=True)
        # Close ONTO the object and squeeze with a commanded force. The travel
        # target sits at the object's own half-width, so the fingers are not
        # asked to occupy the space the block is in -- the squeeze does the
        # holding, which is what makes this a grip rather than an overlap.
        # CLOSE UNTIL SOMETHING STOPS YOU. The fingers used to be sent to the
        # object's estimated half-width, which works only as well as that
        # estimate -- and the estimate comes from a single camera that reads a
        # 30 mm block as anywhere from 33 to 51 mm, so the fingers were being
        # told to stop a centimetre outside the thing they were closing on.
        # Nothing here needs to know how big it is: keep closing, and when the
        # fingers stall, hold them where they stalled and let the squeeze do the
        # gripping. That is what the encoder is for and what a real parallel
        # gripper does.
        if holding(seen):
            target[4], target[5] = q[4], q[5]
        else:
            target[4] = target[5] = _shut()
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
    """Whether this phase is finished, asked of the gate the phase names."""
    gate = PHASES[phase][2]
    done = False

    if gate == "seen":
        done = seen.object_seen
    elif gate == "vantage":
        # Standing where the good look is taken, HAVING TAKEN IT, and having
        # taken enough of them that the answer has stopped moving.
        #
        # Being near the vantage is not enough on its own: the sweep pose that
        # first spots the object already sits within five centimetres of it, so
        # a proximity-only gate was satisfied on the same frame the object was
        # found and the phase passed through in one step without a single extra
        # look. The machine then drove at a 23 mm error and never corrected it,
        # because after that first glance the object was never in frame again.
        if seen.object_at is not None:
            want = np.asarray(seen.object_at) + np.asarray(
                [0.0, 0.0, _INSPECT_HEIGHT_M])
            done = bool(seen.object_seen
                        and seen.looks >= 3
                        and seen.belief_shift <= _SETTLED_M
                        and float(np.linalg.norm(body.camera_pose()[0] - want))
                        <= _VANTAGE_M)
    elif gate == "near":
        done = palm_to_object_m(body, seen) <= _ARRIVED_M
    elif gate == "opened":
        # Open when it is OPEN, not when a timer says so. A fixed dwell let the
        # approach resume with the pads 1 cm apart around a 6 cm block, so
        # move_to drove closed fingers into the object and jammed them there.
        done = seen.q[4] >= _limits()[4][1] - 0.004
    elif gate == "engulfed":
        # Measured against the OPENING, which the encoders know exactly, rather
        # than against the estimated size, which they do not. Scaling this by an
        # inflated estimate let the hand close on the air beside the block.
        done = object_in_grasp_m(body, seen) <= max(0.018, body.opening() * 0.4)
    elif gate == "gripped":
        done = holding(seen)
    elif gate == "above_rim":
        done = object_above_rim_m(body, seen) >= 0.02
    elif gate == "over_target":
        done = (object_over_target_m(body, seen) <= _OVER_TARGET_M
                and holding(seen))

    return phase + 1 if done and phase + 1 < len(PHASES) else phase
