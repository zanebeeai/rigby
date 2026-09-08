"""Search a few moves ahead, and refuse the ones that hit something.

WHY THIS EXISTS. The greedy search scores 192 single-move probes one step ahead
and takes the best. It has no notion that anything is in the way, so a probe
that rams the plate into a wall scores exactly like one that does not. Measured
in the run of 2026-09-08: the arm carried the block to rim height at the bin's
OUTER edge -- two millimetres outside the opening -- and swung the base
sideways, crushing the block against the wall at 87 N until the contact prised
the jaws open and the block fell out. Every probe along the way scored an
improvement, because every probe was moving the block closer to the bin centre.
Sideways through a wall is closer.

One step is also not enough to see that. The move that starts the collision is
an improvement; only its successors are not. A search that cannot look past its
own next move cannot tell the difference between a route and a wall.

WHY BEAM SEARCH AND NOT A*. A* wants a discrete graph and an admissible
heuristic to a known goal state. Neither exists here: joint angles are
continuous, and the goal is not a configuration but a predicate on a measured
number -- "make palm_to_object_m at most 0.045" -- which has a whole manifold of
solutions and no distance function that is guaranteed not to overestimate.
Beam search over the SAME named-move vocabulary the model already reasons about
needs neither, keeps the parts and moves nameable in the transcript, and gets
the one property that was missing: it can see that a move leads somewhere bad.

WHAT COUNTS AS AN OBSTACLE, and why this is not cheating. Two checks:

  THE ARM against the room -- MuJoCo's own collision detection at the trial
  pose, restricted to the machine's geoms against furniture. A real arm's
  planner has a collision model of itself and its workcell; that is the least
  exotic thing in robotics.

  WHAT IT IS CARRYING against the room -- a box at the grasp centre, of the
  BELIEVED size, tested against the furniture's declared boxes. Both halves
  come from things the machine legitimately has: where it thinks the object is
  and how big it thinks it is, which came from the cameras, and where the
  furniture is, which is calibration. Nothing here reads the block's true pose.
  This is the check that would have caught the crash, because the thing that
  hit the wall was the payload rather than the arm.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from ..physics.model import JOINTS, Body
from ..sensing.gripper_camera import Sensed
from .goals import NumericTarget
from .greedy import _AMOUNTS, _named, _pose_for, moves_for
from .solver import _limits

#: How many moves ahead to look. Three is where seeing the consequence of a
#: move starts to pay: one is blind, two catches the move after the mistake,
#: three catches the mistake.
#:
#: COST, measured rather than projected. Depth 3 with beam 6 expands about 2500
#: poses, and a pose costs 16 us to write and run kinematics, 26 us to score,
#: and 16 us to collision-check -- 41 us once something is being carried, since
#: the payload box is tested against the furniture as well. That is roughly
#: 150 ms a plan while approaching and 350 ms while carrying, so the run loop
#: replans every sixth frame rather than every frame. An earlier version of this
#: comment claimed 110 ms from a microbenchmark that left out the collision
#: check it was there to pay for.
DEPTH = 3
#: How many partial routes to keep at each level.
BEAM = 6
#: Penetration deeper than this counts as hitting something. Solver noise and
#: resting contact sit well under a millimetre; the crash ran to hundreds.
TOUCH_M = 0.001
#: Everything that moves because a joint moved.
_MACHINE = ("plate_geom", "left_geom", "right_geom",
            "base_hub", "seg1", "seg2", "seg3")
#: Furniture is anything static the room declares. The block is deliberately
#: NOT here: the hand is supposed to touch the block, and where it is comes
#: from the belief rather than from this check.
_SKIP = ("block_geom",)


def _geom_ids(model, names) -> set[int]:
    out = set()
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name in names:
            out.add(index)
    return out


def _furniture(model) -> list[tuple[np.ndarray, np.ndarray]]:
    """The static boxes a carried object could be driven into.

    Read from the model's own geometry, which is the declared workcell -- the
    same thing a calibrated cell gives a real planner. Only boxes, because a
    box is the shape an axis-aligned test can answer honestly and the bin, the
    bench and the walls are all boxes.
    """
    out = []
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if not (name.startswith("bin_") or name in ("pedestal",)):
            continue
        if int(model.geom_type[index]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        out.append((np.asarray(model.geom_pos[index], dtype=float),
                    np.asarray(model.geom_size[index], dtype=float)))
    return out


class Foresight:
    """The obstacle model, built once per body and reused every plan."""

    def __init__(self, body: Body):
        self.body = body
        self.model = body.model
        self.data = body.data
        self.mine = _geom_ids(self.model, _MACHINE)
        self.skip = _geom_ids(self.model, _SKIP)
        boxes = _furniture(self.model)
        # STACKED ONCE. Walking five boxes in a Python loop cost 75 us a node,
        # which was most of the search: 90 us with the payload check against
        # 16 us without it. The same arithmetic as one array operation is
        # nearly free, and this runs a few thousand times per plan.
        self.box_pos = (np.stack([pos for pos, _ in boxes])
                        if boxes else np.zeros((0, 3)))
        self.box_half = (np.stack([half for _, half in boxes])
                         if boxes else np.zeros((0, 3)))
        self.address = [body.address(name) for name in JOINTS]

    def hits(self, payload_half: np.ndarray | None) -> float:
        """How far into something the machine is, at whatever pose is set.

        Returns metres of the deepest overlap, zero when clear. Assumes
        kinematics have already been run for the pose under test.
        """
        mujoco.mj_collision(self.model, self.data)
        worst = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self.skip or second in self.skip:
                continue
            touching_me = (first in self.mine) ^ (second in self.mine)
            if not touching_me:
                continue
            worst = max(worst, -float(contact.dist))
        if payload_half is None:
            return worst
        # WHAT IT IS CARRYING, as a box at the grasp centre. The block does not
        # follow the hand when qpos is set directly -- it is a free body, and
        # the trial pose only moves the arm -- so MuJoCo cannot see this one
        # and it is the collision that actually happened.
        if not len(self.box_pos):
            return worst
        centre = np.asarray(self.body.grasp_centre(), dtype=float)
        gaps = np.abs(centre - self.box_pos) - (self.box_half + payload_half)
        return max(worst, float(-gaps.max(axis=1).min()))

    def read(self) -> np.ndarray:
        return np.asarray([self.data.qpos[a] for a in self.address])

    def write(self, pose) -> None:
        for slot, value in zip(self.address, pose):
            self.data.qpos[slot] = value
        mujoco.mj_kinematics(self.model, self.data)


def foresee(body: Body, seen: Sensed, target: NumericTarget,
            spans: dict[str, float], start_from: np.ndarray | None = None,
            depth: int = DEPTH, beam: int = BEAM, sight: Foresight | None = None
            ) -> tuple[np.ndarray, float, dict[str, Any]]:
    """The next move of the best short route that does not hit anything.

    Same shape in and out as greedy.pursue, so it drops into the run loop --
    the pose to command, the error that route reaches, and a note naming the
    part and move so the transcript still reads in the vocabulary the model
    uses.

    A ROUTE IS REJECTED IF ANY STEP ALONG IT COLLIDES, not merely if it ends in
    a collision. Passing through a wall and coming out the far side is not a
    plan. If every route collides -- which happens when the arm is already
    against something -- the least-bad one is taken and the note says so,
    because refusing to move at all when you are already touching a wall is how
    a machine gets stuck against it forever.
    """
    sight = sight or Foresight(body)
    limits = _limits()
    restore = sight.read()
    payload = (np.asarray(seen.object_size, dtype=float)
               if (seen.holding() and seen.object_size is not None) else None)

    start = (np.asarray(start_from, dtype=float) if start_from is not None
             else restore.copy())
    sight.write(start)
    here = target.error(body, seen, spans)
    blocked_now = sight.hits(payload) > TOUCH_M

    parts = _named(tuple(target.using))
    hold = {"part": "-", "move": "hold", "amount": 0.0}
    # (route_score, deepest_overlap, route_end_pose, first_move,
    #  FIRST_POSE, first_score, path)
    #
    # The first pose is carried separately because it is the only one that gets
    # commanded. Plan three, execute one, plan again.
    seed = (here, 0.0, start.copy(), hold, start.copy(), here, [])
    beam_now = [seed]
    best = seed

    for step in range(depth):
        grown = []
        for score, blocked, pose, first, first_pose, first_score, path in beam_now:
            for part in parts:
                for move in moves_for(part):
                    for amount in _AMOUNTS:
                        trial = _pose_for(part, move, pose, limits, amount)
                        if trial is None or np.allclose(trial, pose):
                            continue
                        sight.write(trial)
                        deep = sight.hits(payload)
                        # A route that was already touching something is not
                        # made worse by continuing to touch it, so what counts
                        # is the deepest overlap anywhere along the route.
                        worst = max(blocked, deep)
                        cost = target.error(body, seen, spans)
                        step_note = {"part": part, "move": move,
                                     "amount": float(amount)}
                        grown.append((cost, worst, trial.copy(),
                                      first if path else step_note,
                                      first_pose if path else trial.copy(),
                                      first_score if path else cost,
                                      path + [step_note]))
        if not grown:
            break
        clear = [row for row in grown if row[1] <= TOUCH_M]
        pool = clear if clear else grown
        # Keep the most promising partial routes, preferring clear ones.
        pool.sort(key=lambda row: (row[0], row[1]))
        beam_now = pool[:beam]
        for row in beam_now:
            better = row[0] < best[0] - 1e-6
            safer = best[1] > TOUCH_M >= row[1]
            if (safer or (better and row[1] <= max(TOUCH_M, best[1]))):
                best = row

    sight.write(restore)
    score, worst, _end_pose, first, first_pose, first_score, path = best
    note = dict(first)
    note["route"] = [f"{s['part']}/{s['move']}@{s['amount']:+.2f}"
                     for s in path[:depth]]
    note["clear"] = bool(worst <= TOUCH_M)
    note["deepest_overlap_mm"] = round(worst * 1000.0, 2)
    note["route_reaches"] = round(float(score), 4)
    if blocked_now:
        note["already_touching"] = True
    # THE FIRST MOVE, NOT THE DESTINATION. Returning the route's endpoint threw
    # away the only thing the route was for. What gets commanded is a setpoint
    # the arm slews toward in a straight line through joint space, and a
    # straight line to the end of a three-move route is not that route -- the
    # routes contain reversals, so it is not even close. Measured against
    # greedy from the same pose, the endpoint was a 1.135 rad jump where one
    # move was 0.733, and a chosen route read
    # tilt_down@+0.60, tilt_up@-0.60, nudge_left@-0.12: down then up, executed
    # as a single lunge across the middle. Nothing along that line had been
    # collision-checked either, which is the safety half of the same mistake.
    #
    # Plan three moves, execute one, replan. The lookahead is what makes the
    # first move a good one; it was never meant to be a trajectory to follow
    # open-loop.
    return first_pose, first_score, note
