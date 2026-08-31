"""Open the cabinet, take out what is inside, put it on the bench.

Every task before this one was about an object that was simply lying there. This
one has a mechanism in it, and that changes what a motion primitive has to be.
Reaching for a block is a free choice of path to a free choice of pose; pulling
a door is neither. The handle may travel along exactly one arc, the arc is set
by a hinge the arm cannot see, and any pull that is not tangent to it is a pull
the hinge simply refuses. So ``pull_open`` does not aim at where the handle
should end up -- it aims one small step along the arc, every frame, and lets the
mechanism supply the rest.

The observation is a separate thing from the controller on purpose. The same
phases run off ``look_global``, which reads the simulator, or off the two
cameras, so what changes between the two runs is only what is known, and any
difference in the result is attributable to that and to nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..body.manifest import spec
from .opening import PUSH_M, Mechanism
from .solver import Command, _limits, _reach_for, _shut
from .travel import travel_to
from ..physics.model import Body

#: The order of the work. Each phase names the gate that ends it.
PHASES: tuple[tuple[str, float, str], ...] = (
    ("to_handle", 1.0, "at_handle"),
    ("grip_handle", 1.0, "has_handle"),
    ("pull_open", 1.0, "door_open"),
    ("let_go", 1.0, "released"),
    ("back_off", 1.0, "clear_of_door"),
    ("to_contents", 1.0, "at_contents"),
    ("grip_contents", 1.0, "has_contents"),
    ("withdraw", 1.0, "clear_of_cabinet"),
    ("carry_to_bench", 1.0, "over_spot"),
    ("set_down", 1.0, "done"),
)

#: How far open counts as open, degrees. Past this the opening is clear enough
#: for the hand and what it is carrying to pass through.
_OPEN_ENOUGH_DEG = 78.0
#: One step along the arc per frame, degrees. Small, because the only way to
#: pull a hinge is tangentially and a big step is a chord, not a tangent.
_ARC_STEP_DEG = 2.0
#: Where the hand waits before going in, metres in front of the opening.
_STANDOFF_M = 0.13
#: Close enough to count as arrived, metres. Tight, and it has to be: the
#: handle is a 22 mm bar, so arriving 22 mm off centre puts it against one pad
#: with nothing on the other side, and closing then shoves the bar sideways
#: instead of gripping it -- the left finger runs to its own stop at 12 mm while
#: the right sits at 34 mm, which is a push, not a grasp. The solver lands
#: within 3 mm, so there is no reason to accept more.
_AT_M = 0.011
_CLEAR_M = 0.16
_ON_SPOT_M = 0.05

#: How hard to insist on facing the door squarely. Low, because this arm cannot
#: face an off-axis target squarely at all and pretending otherwise costs
#: position, which is the thing that actually has to be right.
_SQUARE_WEIGHT = 0.30
#: Where the arm waits before and between reaches: high, and in FRONT of the
#: cabinet. Its zero pose points straight out over the top of it, so every reach
#: began by dragging the gripper across the cabinet roof, where it jammed.
READY_AT = np.asarray([0.02, -0.02, 1.14])
#: How much a metre of joint travel is worth against a metre of position error.
#: Enough to stop the solver crossing the room for a millimetre.
_STAY_NEAR = 0.05
#: RETIRED, and why. The travel budget existed to stop the solver answering a
#: reach with a configuration on the far side of the workspace. Once every reach
#: became a short step along a straight line, it could no longer do that -- it is
#: never asked about anywhere distant -- so the budget stopped protecting
#: anything and started doing harm: anchored at the phase's first frame, it
#: capped the TOTAL motion of a phase, and following a door through ninety
#: degrees is one long continuous motion. The arm swung the door to 46 degrees
#: and then stopped dead against its own allowance. Structure beats a number.
#:
#: The most the arm may reconfigure to service one reach, radians summed over
#: the joints. Generous enough for any reach in this workspace and far short of
#: below the 5.1 the over-the-shoulder contortion needs. A narrow gap, and a
#: real one.
_MAX_TRAVEL = 4.7

@dataclass
class Working:
    """What the controller remembers between frames, for ONE run.

    This was two module-level dicts, which meant a second run in the same
    process began believing whatever the first one had concluded -- fine for a
    single task, wrong the moment anything sweeps.
    """

    #: What has been learned about the thing being opened, by moving it.
    mechanism: Mechanism = field(default_factory=Mechanism)
    #: Which approaches have committed to their final run-in. Latched, because a
    #: threshold recomputed every frame is a threshold that chatters.
    committed: dict = field(default_factory=dict)

_HANDLE_SQUEEZE_N = 9.0
_CARRY_SQUEEZE_N = 12.0


def keep_out() -> tuple:
    """The box the gripper may not be inside: the cabinet, slightly inflated."""
    doc = spec()["scene"]["fridge"]
    cx, cy = float(doc["centre"][0]), float(doc["centre"][2])
    ix, iz, iy = (float(doc["inner_half_m"][0]), float(doc["inner_half_m"][1]),
                  float(doc["inner_half_m"][2]))
    wall = float(doc["wall_m"])
    pad = 0.028
    return (np.asarray([cx - ix - wall - pad, cy - iy - wall - pad, 0.72]),
            np.asarray([cx + ix + wall + pad, cy + iy + wall + pad,
                        0.72 + 2 * iz + wall + pad]))


def geometry() -> dict:
    """The cabinet, in MuJoCo coordinates, from the manifest."""
    doc = spec()["scene"]["fridge"]
    return {
        # No hinge, and no handle-relative-to-hinge. Where the door pivots and
        # how far it swings are properties of the mechanism, and the machine
        # discovers those by moving it. What stays is where the FURNITURE is,
        # which is the same class of prior as knowing where the bench is.
        "centre_x": float(doc["centre"][0]),
        "front_y": float(doc["centre"][2]) - float(doc["inner_half_m"][2])
        - float(doc["wall_m"]),
        "shelf_z": 0.72,
        "mid_z": 0.72 + float(doc["inner_half_m"][1]),
        "place": np.asarray([doc["place_target"][0], doc["place_target"][2],
                             doc["place_target"][1]], dtype=float),
        "tolerance": float(doc["place_tolerance_m"]),
    }


@dataclass
class Scene:
    """One observation of the cabinet task."""

    q: np.ndarray
    door_deg: float
    handle_at: np.ndarray
    object_at: np.ndarray | None
    object_size: np.ndarray | None
    contact_left: bool
    contact_right: bool
    grip_effort_n: float
    arm_stalled: bool
    object_seen: bool = True

    def holding(self) -> bool:
        return (self.contact_left and self.contact_right
                and self.grip_effort_n > 1.0)


def _grip_state(body: Body, commanded: np.ndarray, squeeze_n: float,
                latched: dict) -> tuple[bool, bool, bool]:
    """Contact and obstruction from the encoders, as everywhere else."""
    q = np.asarray(body.q())
    qd = np.asarray(body.qd())
    shut = _shut()
    # "I told it to close further, it has not, and it has stopped." All three
    # clauses matter. Without the third -- that the command is actually AHEAD of
    # where the finger is -- the test fires on the first frame of any grip
    # phase, when the squeeze is already commanded and the fingers are simply
    # still open and not yet moving. That latched "holding" onto thin air, the
    # jaws froze where they were, and the arm spent thirty seconds pulling a
    # door handle it was merely standing next to.
    # Squeezed, stopped, and NEITHER fully open NOR fully shut. That last pair
    # is what makes it work without a force sensor: fully open and stationary is
    # a hand that has just arrived and not started closing, fully shut is a hand
    # that closed on nothing, and anything in between that will not move while
    # being squeezed has something in it.
    #
    # Two earlier versions of this test failed in opposite directions. Testing
    # the tracking error's sign fired when the finger was pushed PAST its target
    # by the squeeze, which is what a successful grip looks like. Testing that
    # the command is ahead of the finger fires only while closing, so a settled
    # grip reads empty the moment it succeeds -- and it needs BOTH fingers to
    # read blocked, which an off-centre object breaks, because the two fingers
    # stop at different travels and only the further one looks blocked.
    wide = _limits()[4][1]
    settled = ((squeeze_n > 0.0)
               & (np.abs(np.asarray([qd[4], qd[5]])) < 0.004)
               & (np.asarray([q[4], q[5]]) > shut + 0.004)
               & (np.asarray([q[4], q[5]]) < wide - 0.002))
    left, right = bool(settled[0]), bool(settled[1])
    latched["for"] = latched.get("for", 0) + 1 if (left and right) else 0
    if latched["for"] >= 4 and not latched.get("on"):
        latched["on"] = True
        # Remember WHERE the grip closed. Letting go is judged against this
        # rather than against where the fingers are right now, because where
        # they are right now moves: a door being pulled round pushes the jaws
        # about, and a command that sits a few millimetres outside a finger the
        # object has just shoved inward looks exactly like an instruction to
        # open. It is not. It is the same instruction as last frame, and the
        # finger moved. The door was being dropped at 65 degrees by that.
        latched["at"] = np.asarray(q[4:]).copy()
    held_at = latched.get("at")
    told_to_open = (held_at is not None
                    and bool(np.any(commanded[4:] > held_at + 0.006)))
    if squeeze_n <= 0.0 or told_to_open or float(np.min(q[4:])) <= shut + 0.004:
        latched["on"] = False
        latched["at"] = None

    # Obstruction: behind where it was sent AND stopped. Lag alone is an arm
    # following a moving target, not an arm against something.
    stalled = bool(np.any((np.abs(commanded[:4] - q[:4]) > 0.08)
                          & (np.abs(qd[:4]) < 0.08)))
    return (left or latched["on"], right or latched["on"], stalled)


def look_global(body: Body, commanded: np.ndarray, squeeze_n: float,
                latched: dict) -> Scene:
    """The reference observation: read the simulator and be honest that you did.

    This exists to separate two questions that get confused when a run fails.
    Can the ARM do this at all -- can it hold a handle, follow an arc, and reach
    into a box? And can the PERCEPTION support doing it? Answering the first
    with the second still unbuilt is the only way to know which one to fix.
    """
    left, right, stalled = _grip_state(body, commanded, squeeze_n, latched)
    hinge = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_JOINT,
                              "door_hinge")
    slot = body.model.jnt_qposadr[hinge]
    which = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_GEOM, "handle")
    return Scene(
        q=np.asarray(body.q()),
        door_deg=float(np.degrees(body.data.qpos[slot])),
        handle_at=np.asarray(body.data.geom_xpos[which]),
        object_at=np.asarray(body.block()),
        object_size=np.asarray(body.block_half),
        contact_left=left,
        contact_right=right,
        grip_effort_n=float(squeeze_n),
        arm_stalled=stalled,
    )


def decide(body: Body, seen: Scene, phase: int, support: float,
           now: float, work: Working) -> Command:
    """One phase's command."""
    name, amount, _gate = PHASES[phase]
    place = geometry()
    q = body.q()
    target = q.copy()
    facing = np.asarray([0.0, -1.0, 0.0])

    if name == "to_handle":
        # Stand off in front of the handle, square to the door's face, then go
        # in. Coming straight at it from wherever the arm happens to be puts a
        # finger through the door panel on the way.
        #
        # COMMITTING to the second half is a latch, not a threshold. Choosing
        # between the standoff and the handle by whether the gap exceeds a
        # number means that when the gap sits near that number the goal jumps
        # ten centimetres every frame, and the arm oscillates in front of the
        # handle indefinitely -- which is exactly what it did, for fifty-four
        # seconds, while the jaws rocked between 6.8 and 7.9 cm against the
        # door. Once you have lined up, you go in.
        gap = float(np.linalg.norm(body.grasp_centre() - seen.handle_at))
        if gap <= 0.14:
            work.committed["to_handle"] = True
        goal = (seen.handle_at if work.committed.get("to_handle")
                else seen.handle_at + facing * 0.10)
        found = travel_to(body, goal, facing, support,
                           square_weight=_SQUARE_WEIGHT,
                           keep_out=keep_out(),
                           avoid=work.mechanism.occupies(),
                           stay_near=_STAY_NEAR,
                           warm_key=name, ranked=True)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * float(np.clip(amount, 0, 1))
        target[4] = target[5] = _limits()[4][1]
        return Command(target)

    if name == "grip_handle":
        if seen.holding():
            target[4], target[5] = q[4], q[5]
        else:
            target[4] = target[5] = _shut()
        return Command(target, _HANDLE_SQUEEZE_N)

    if name == "pull_open":
        if not seen.holding():
            return Command(target, _HANDLE_SQUEEZE_N,
                           "refused: lost the handle", hold_station=True)
        # ASK THE MECHANISM WHICH WAY IT GOES.
        #
        # This used to compute where the handle would be at a given door angle,
        # from the hinge position and the swing, both read out of the manifest.
        # It opened exactly one piece of furniture and it did so by being told
        # the answer. Nothing here knows this is a door: it pushes, watches
        # where its own hand actually ended up, and goes that way next. The
        # deflection IS the direction the mechanism permits, and re-asking every
        # frame is what tracks a tangent that rotates.
        hand = body.grasp_centre()
        work.mechanism.start(-body.approach())
        work.mechanism.observe(hand)
        if work.mechanism.barred():
            work.mechanism.cast_wider()
        goal = hand + work.mechanism.aim() * PUSH_M
        found = travel_to(body, goal, None, support, stay_near=_STAY_NEAR,
                           warm_key=name, ranked=True)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * 0.5
        # HOLD EACH FINGER WHERE IT IS, not both where the left one is. An
        # object is never exactly centred in the jaws, so the two fingers stop
        # at different travels -- and commanding both to the left one's position
        # tells the right one to OPEN by the difference. The grip latch reads
        # that, correctly, as the hand being asked to let go, and releases. The
        # door was being dropped at 67 degrees by a line that looks like it
        # freezes the grip.
        target[4], target[5] = q[4], q[5]
        return Command(target, _HANDLE_SQUEEZE_N)

    if name == "let_go":
        target[4] = target[5] = _limits()[4][1]
        return Command(target, 0.0)

    if name == "back_off":
        # Out of the door's way before crossing in front of the opening.
        goal = np.asarray([place["centre_x"], place["front_y"] - _STANDOFF_M,
                           place["mid_z"] + 0.06])
        found = travel_to(body, goal, facing, support,
                           square_weight=_SQUARE_WEIGHT,
                           keep_out=keep_out(),
                           avoid=work.mechanism.occupies(),
                           stay_near=_STAY_NEAR,
                           warm_key=name, ranked=True)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * float(np.clip(amount, 0, 1))
        target[4] = target[5] = _limits()[4][1]
        return Command(target)

    if name == "to_contents":
        if seen.object_at is None:
            return Command(target, 0.0, "refused: cannot see it",
                           hold_station=True)
        # In through the opening, square to it, so the hand does not clip a wall
        # on the way. The last stretch is a straight push along the same axis.
        outside = np.asarray([seen.object_at[0], place["front_y"] - 0.08,
                              seen.object_at[2]])
        lined_up = float(np.linalg.norm(
            (body.grasp_centre() - outside)[[0, 2]])) < 0.03
        goal = seen.object_at if lined_up else outside
        found = travel_to(body, goal, facing, support,
                           square_weight=_SQUARE_WEIGHT,
                           keep_out=keep_out(),
                           avoid=work.mechanism.occupies(),
                           stay_near=_STAY_NEAR,
                           warm_key=name, ranked=True)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * float(np.clip(amount, 0, 1))
        target[4] = target[5] = _limits()[4][1]
        return Command(target)

    if name == "grip_contents":
        if seen.holding():
            target[4], target[5] = q[4], q[5]
        else:
            target[4] = target[5] = _shut()
        return Command(target, _CARRY_SQUEEZE_N)

    if name == "withdraw":
        if not seen.holding():
            return Command(target, _CARRY_SQUEEZE_N, "refused: nothing held",
                           hold_station=True)
        # Straight back out. Anything else swings what is held into a wall.
        here = body.grasp_centre()
        goal = np.asarray([here[0], place["front_y"] - _CLEAR_M, here[2]])
        found = travel_to(body, goal, facing, support,
                           square_weight=_SQUARE_WEIGHT,
                           keep_out=keep_out(),
                           avoid=work.mechanism.occupies(),
                           stay_near=_STAY_NEAR,
                           warm_key=name, ranked=True)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * 0.5
        # HOLD EACH FINGER WHERE IT IS, not both where the left one is. An
        # object is never exactly centred in the jaws, so the two fingers stop
        # at different travels -- and commanding both to the left one's position
        # tells the right one to OPEN by the difference. The grip latch reads
        # that, correctly, as the hand being asked to let go, and releases. The
        # door was being dropped at 67 degrees by a line that looks like it
        # freezes the grip.
        target[4], target[5] = q[4], q[5]
        return Command(target, _CARRY_SQUEEZE_N)

    if name == "carry_to_bench":
        if not seen.holding():
            return Command(target, _CARRY_SQUEEZE_N, "refused: nothing held",
                           hold_station=True)
        goal = place["place"] + np.asarray([0.0, 0.0, 0.12])
        found = travel_to(body, goal, None, support, keep_out=keep_out(),
                           stay_near=_STAY_NEAR, warm_key=name, ranked=True)
        if found is not None:
            target[:4] = q[:4] + (found - q[:4]) * 0.35
        # HOLD EACH FINGER WHERE IT IS, not both where the left one is. An
        # object is never exactly centred in the jaws, so the two fingers stop
        # at different travels -- and commanding both to the left one's position
        # tells the right one to OPEN by the difference. The grip latch reads
        # that, correctly, as the hand being asked to let go, and releases. The
        # door was being dropped at 67 degrees by a line that looks like it
        # freezes the grip.
        target[4], target[5] = q[4], q[5]
        return Command(target, _CARRY_SQUEEZE_N)

    # set_down
    goal = place["place"] + np.asarray([0.0, 0.0, 0.035])
    found = travel_to(body, goal, None, support, keep_out=keep_out(),
                       stay_near=_STAY_NEAR, warm_key=name, ranked=True)
    if found is not None:
        target[:4] = q[:4] + (found - q[:4]) * 0.3
    low = float(np.linalg.norm(body.grasp_centre() - goal)) < 0.04
    target[4] = target[5] = _limits()[4][1] if low else q[4]
    return Command(target, 0.0 if low else _CARRY_SQUEEZE_N)


def advance(body: Body, seen: Scene, phase: int, now: float,
            work: Working) -> int:
    """Whether this phase is finished."""
    gate = PHASES[phase][2]
    place = geometry()
    here = body.grasp_centre()
    done = False

    if gate == "at_handle":
        done = float(np.linalg.norm(here - seen.handle_at)) <= _AT_M
    elif gate == "has_handle":
        done = seen.holding()
    elif gate == "door_open":
        # It moved, and then nothing moved it further. Not an angle: an angle
        # only exists for a hinge, and this has to work for a drawer too.
        done = work.mechanism.opened()
    elif gate == "released":
        done = float(seen.q[4]) >= _limits()[4][1] - 0.006
    elif gate == "clear_of_door":
        done = here[1] <= place["front_y"] - _STANDOFF_M + 0.04
    elif gate == "at_contents":
        done = (seen.object_at is not None
                and float(np.linalg.norm(here - seen.object_at)) <= 0.035)
    elif gate == "has_contents":
        done = seen.holding()
    elif gate == "clear_of_cabinet":
        done = here[1] <= place["front_y"] - _CLEAR_M + 0.03
    elif gate == "over_spot":
        done = (float(np.linalg.norm((here - place["place"])[[0, 1]]))
                <= _ON_SPOT_M and seen.holding())

    return phase + 1 if done and phase + 1 < len(PHASES) else phase


def on_the_bench(body: Body) -> bool:
    """Did it end up where it was supposed to go? The grader, not a sensor."""
    place = geometry()
    at = body.block()
    return bool(float(np.linalg.norm((at - place["place"])[[0, 1]]))
                <= place["tolerance"] and at[2] <= 0.80)
