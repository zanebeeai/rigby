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

def _shut() -> float:
    """The travel a fully closed finger reports. Fixed by the mechanism."""
    for joint in spec()["kinematics"]["joints"]:
        if joint["name"] == "finger_left":
            return float(joint["range_m"][0])
    return 0.0


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
               support: float, aim: str = "grasp",
               square_weight: float = 0.30,
               keep_out: tuple | None = None,
               stay_near: float = 0.0,
               tolerance: float = 0.03,
               warm_key: str | None = None,
               max_travel: float | None = None,
               ranked: bool = False) -> np.ndarray | None:
    """Joint angles that put ``aim`` on ``goal`` -- the grasp centre, or the eye.

    Aiming the CAMERA is a different request from aiming the hand, and treating
    them as the same thing was worth about twenty millimetres of error. The lens
    sits five and a half centimetres off the grasp centre, so a pose that puts
    the hand directly over the object leaves the object twenty-two degrees off
    the optical axis -- and a plane-ranged estimate taken from that far off axis
    is exactly the oblique measurement this whole approach is worst at. Looking
    at something means pointing the part that sees.

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

    start = np.asarray([data.qpos[a] for a in address[:4]])
    if warm_key is not None:
        anchor = _ANCHOR.setdefault(warm_key, start.copy())
    else:
        anchor = start

    def evaluate(angles: np.ndarray) -> float | None:
        for slot, value in zip(address[:4], angles):
            data.qpos[slot] = value
        mujoco.mj_kinematics(model, data)
        if aim == "camera":
            # Cameras are placed by mj_camlight, which mj_kinematics does not
            # call, so without this the eye never moves during the search.
            mujoco.mj_camlight(model, data)
            here = body.camera_pose()[0]
        else:
            here = body.grasp_centre()
        # Two objectives, and which one is right depends on the task.
        #
        # RANKED (opt-in): getting to the point is the task; how the hand is
        # angled and how far the arm moved are preferences, scored zero until
        # the hand is within `tolerance` and capped so they can never outvote
        # position. This is what a handle needs -- a preference must not be
        # able to buy position error with itself, and in a plain sum it can.
        #
        # SUMMED (default): everything weighted together. Less principled, and
        # measurably better on the pick-and-place: ranking cost that task 3 of
        # 12 placements, because a grasp approach wants its orientation shaped
        # over the whole reach rather than treated as a tie-break near the end,
        # and the tolerance that expresses this differs per reach in a way that
        # is not yet understood well enough to set from first principles.
        #
        # So the default stays the behaviour that is measured to work, and the
        # ranked objective is asked for where it is needed. An architecture is
        # not better for being tidier than the evidence.
        gap = float(np.linalg.norm(here - goal))
        cost = gap
        preference = 0.0
        if square_to is not None:
            preference += (1.0 - float(np.dot(body.approach(), -square_to))) \
                * square_weight
        if stay_near:
            preference += stay_near * float(np.linalg.norm(angles - start))
        if ranked:
            slack = float(np.clip(1.0 - gap / max(tolerance, 1e-6), 0.0, 1.0))
            cost += slack * min(preference, tolerance * 0.5)
        else:
            cost += preference
        if max_travel is not None and float(
                np.linalg.norm(angles - anchor)) > max_travel:
            # HOW FAR THE ARM MUST MOVE IS A CONSTRAINT, NOT A PREFERENCE.
            # Ranking position first and capping the preferences so they cannot
            # outvote it also made them too weak to reject a gross
            # reconfiguration: the solver returned yaw at its 150 degree limit
            # with the elbow folded back, reaching the handle over its own
            # shoulder, thirteen millimetres from the goal and therefore cheap.
            # Getting into that pose means swinging the whole arm across the
            # workspace, which is not a tie-break against ten millimetres of
            # position -- it is a different manoeuvre, and one no planner would
            # accept to service a single reach. So it is ruled out rather than
            # priced. Bounding travel from the CURRENT pose keeps the admissible
            # set a ball around where the arm already is, so this does not wall
            # the search off the way an obstacle veto would.
            return None
        for name in ("shoulder", "seg1", "seg2", "seg3", "plate_geom"):
            if float(body.geom_at(name)[2]) < support:
                return None
        if keep_out is not None:
            # Solid furniture. The only obstacle the solver knew about was the
            # bench, as a height -- so with a cabinet in the workspace it
            # happily returned poses resting on the cabinet roof, and the arm
            # drove there and jammed. A box is a crude occupancy model and it is
            # the difference between a pose that exists and a pose that can be
            # held.
            # A PENALTY, not a veto. Refusing invalid poses outright walls the
            # search off: coordinate descent only steps to strictly better
            # VALID poses, so a good pose on the far side of the cabinet is
            # unreachable even when it exists, and the solver settles for
            # whatever it can see from where it started. Scoring the depth of
            # the intrusion instead leaves the landscape continuous, so the
            # search can cross the obstacle to get to the answer behind it.
            low, high = keep_out
            for name in ("plate_geom", "left_geom", "right_geom", "seg3"):
                where = body.geom_at(name)
                inside = np.minimum(where - low, high - where)
                if bool(np.all(inside > 0.0)):
                    cost += 3.0 * float(np.min(inside)) + 0.25
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

    def quality(angles: np.ndarray) -> tuple[float, float]:
        """What the pose actually achieves: distance, and how it is angled.

        Asked separately from the cost, because the cost ranks preferences
        BELOW the task and therefore stops reflecting how badly a preference is
        being missed once the task is met. Escalating on cost alone meant a
        pose that arrived facing exactly backwards looked cheap, so no harder
        search was ever run and every approach-angle weight gave the same
        wrong answer.
        """
        for slot, value in zip(address[:4], angles):
            data.qpos[slot] = value
        mujoco.mj_kinematics(model, data)
        if aim == "camera":
            mujoco.mj_camlight(model, data)
            spot = body.camera_pose()[0]
        else:
            spot = body.grasp_centre()
        angle = (float(np.dot(body.approach(), -square_to))
                 if square_to is not None else 1.0)
        return float(np.linalg.norm(spot - goal)), angle

    def rank(angles: np.ndarray) -> tuple[int, float]:
        """Order poses the way the task orders them, not the way a sum does.

        First: does it reach the point at all, within tolerance. Only among
        poses that do is the approach angle compared, and only among poses that
        do not is the remaining distance compared. This is the ordering the
        ranked cost expresses locally, applied to CHOOSING between candidates
        -- and it has to be applied here too, because the ranked cost
        deliberately ignores the approach angle while the hand is still far
        away, which leaves a search that picks its starting point on position
        alone no reason to ever look in a basin where the hand faces the right
        way. It would then refine, perfectly, to the wrong side of the handle.
        """
        gap, facing = quality(angles)
        if gap <= tolerance:
            return (0, -facing)
        # Still short of the goal: order by distance, but not ONLY by distance.
        # Ignoring joint travel entirely here lets the search swap a perfectly
        # good nearby answer for a distant configuration that is a millimetre
        # closer, and the arm then swings across the whole workspace chasing it
        # -- which is what stopped the carry reaching the bin at all, ending it
        # 41 cm away still holding the block. A centimetre of gap is worth about
        # a radian of travel; below that exchange rate, stay where you are.
        travel = float(np.linalg.norm(angles - start))
        return (1, gap + 0.01 * travel)

    def struggling_at(angles: np.ndarray) -> bool:
        """Whether to spend more search on this.

        Under the ranked objective the cost stops reflecting a badly-missed
        preference once the task is met, so escalation has to ask what the pose
        ACHIEVES. Under a plain sum the cost still carries everything, and
        asking it is both cheaper and what the working configuration did.
        """
        if not ranked:
            return here > 0.02
        gap, facing = quality(angles)
        return gap > min(0.012, tolerance / 2.0) or facing < 0.75

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
    if struggling_at(best):
        for seed in _SEEDS:
            candidate, cost = descend(np.asarray(seed), 5)
            if (rank(candidate) < rank(best) if ranked
                    else cost < here - 0.01):
                best, here = candidate, cost
            if not struggling_at(best):
                break

    # STILL STUCK: sweep the whole joint space coarsely and descend from the
    # best square found. Five hand-picked seeds cover five basins, and the arm
    # has four joints -- reaching a handle needs the hand pointing at the door,
    # which lives in a basin none of the seeds are in, so every weighting of
    # position against approach angle returned either the right place facing
    # backwards or the right facing two hundred millimetres away. The sweep is
    # expensive and runs only on the frames where the cheap search failed;
    # once it lands in the right basin, stay_near keeps the next frame there.
    # The whole-joint-space sweep is part of the same opt-in. It was added for
    # the cabinet handle and it is not free: on the pick-and-place -- which
    # never needed it, because that workspace is open and the cheap descent
    # already finds the answer -- it fires whenever the cheap search is merely
    # imperfect and hands back a pose in a different arm configuration, and the
    # placement rate fell from 11/12 to 8/12 and then 6/12 while every isolated
    # measurement of the solver said it had improved. A search that is better at
    # finding the global optimum is not better if the thing it was already
    # finding was the one you wanted.
    if ranked and struggling_at(best):
        grid = [np.linspace(low, high, n) for (low, high), n
                in zip(limits[:4], (9, 7, 9, 7))]
        # WHERE TO LOOK is a different question from WHAT TO ACCEPT, and they
        # want different rules. The grid steps 37 degrees at a time, so no point
        # on it is ever within the tolerance -- judge it by the task hierarchy
        # and every square fails the first test, the comparison falls through to
        # distance alone, and the sweep hands back a starting point in a basin
        # where the hand faces backwards. It then refines, perfectly, to the
        # wrong side of the handle. A blend is the right heuristic at this
        # resolution: it says which region is promising. The hierarchy still
        # decides what comes out at the end.
        coarse, best_blend = None, 9e9
        for a in grid[0]:
            for b in grid[1]:
                for c in grid[2]:
                    for d in grid[3]:
                        trial = np.asarray([a, b, c, d])
                        if evaluate(trial) is None:
                            continue
                        if ranked:
                            gap, facing = quality(trial)
                            blend = gap + 0.6 * (1.0 - facing)
                        else:
                            blend = evaluate(trial) or 9e9
                        if blend < best_blend:
                            coarse, best_blend = trial, blend
        if coarse is not None:
            candidate, cost = descend(coarse, 6)
            if (rank(candidate) < rank(best) if ranked else cost < here):
                best, here = candidate, cost

    data.qpos[:] = saved_q
    data.qvel[:] = saved_v
    mujoco.mj_forward(model, data)
    if here >= 1e5:
        return None
    if warm_key:
        _WARM[warm_key] = best.copy()
    return best


#: The patches of table the camera visits, in the order it visits them. This is
#: knowledge of the FURNITURE -- where the work surface is and how far across it
#: the arm can reach -- which a machine is entitled to the way it is entitled to
#: know where its own bench is. It contains nothing about the object.
_SCAN_GRID = tuple(
    np.asarray([x, y, 0.0]) for y in (0.30, 0.14, 0.45) for x in (0.0, -0.20, 0.20)
)

_scan_cache: list[np.ndarray] | None = None

#: The last pose each named reach settled on. A solver that restarts from
#: wherever the BODY currently is starts, early in a move, a long way from its
#: own previous answer -- so it re-searches from scratch every frame, lands in a
#: different basin whenever the escalation happens to fire, and the target
#: flickers between two arm configurations while the hand never arrives. Keeping
#: the previous answer and trying it first makes the search stable across frames
#: and, because it is nearly always still the best one, cheap.
_WARM: dict[str, np.ndarray] = {}

#: The pose each named reach began from. Travel is budgeted against THIS, not
#: against wherever the arm has since got to. Measuring from the current pose
#: ratchets: the moment the arm starts down a wrong basin the budget re-centres
#: on it, the correct answer falls permanently out of range, and the arm walks
#: steadily further away with every frame reporting that it is within budget.
_ANCHOR: dict[str, np.ndarray] = {}


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
