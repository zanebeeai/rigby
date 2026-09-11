"""Where to put the arm: inverse kinematics, and what it is allowed to trade.

This is a SEARCH, not a formula. It probes candidate joint angles through the
simulation's own kinematics, so the answer cannot drift from what the body will
actually do -- which is what a separately-derived analytic chain eventually
does.

Two things here were worth a day each to learn.

WHAT IT OPTIMISES. There are two objectives available. The default adds
everything up: position error, plus a weighted approach angle, plus a weighted
posture term. It is the one the pick-and-place is measured under. The opt-in
`ranked` objective treats position as the TASK and the rest as PREFERENCES that
score zero until the hand is within tolerance and are capped so they can never
outvote it -- because in a plain sum, any preference can buy position error with
itself, and that bought 0.95 of approach angle for 49 mm of miss on a door
handle. Ranked is better on that problem and measurably worse on the
pick-and-place, so it is asked for rather than assumed.

WHAT IT WILL NOT DO. Two hard constraints rather than priced preferences: no
link may drop below the bench, and the arm may not reconfigure further than a
travel budget to service one reach. The second exists because a capped
preference was too weak to reject yaw at its limit with the elbow folded back,
reaching a handle over the arm's own shoulder.

It has no notion of a PATH. It answers "where should the joints be", the body is
driven at that answer, and the question is asked again next frame. That is the
standing limitation of this layer, and it is why the arm can be given a pose it
cannot reach without dragging itself through the furniture on the way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..body.manifest import spec
from ..physics.model import JOINTS, Body


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
        for name in ("base_hub", "seg1", "seg2", "seg3", "plate_geom"):
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
            # Stop when the answer is GOOD, not merely when it is no longer
            # bad enough to have triggered the search. Entering at 0.02 and
            # leaving at 0.02 means the first seed that clears the entry bar
            # ends the loop, so a better one two seeds later is never tried --
            # and on the two placements furthest from the arm, which need it,
            # that was the difference between a grasp and nothing.
            if (not struggling_at(best)) if ranked else (here <= 0.01):
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
