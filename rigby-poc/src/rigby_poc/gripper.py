"""A three-segment arm with a parallel gripper, driven by the same architecture.

This exists to answer one question: is the vocabulary about GRASPING, or about
the humanoid hand it was written against? Nothing above the solvers changes.
The same metric names, the same four primitives, the same refusals, the same
phase sequence that lifts the block with five fingers.

Below the interface almost everything differs. Four joints instead of a shoulder
chain and nineteen finger bones. Fingers that translate rather than curl, so
there is no thumb, no opposition to arrange, no curl to lock -- and the C-shape
the humanoid had to be taught is this gripper's resting geometry. If one
controller drives both, the abstraction was right.

The kinematics are analytic and the state is six numbers, so nothing here needs
the rig, the GLB, or the anatomical frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_CONFIG = Path(__file__).resolve().parents[2] / "config"


@lru_cache(maxsize=1)
def spec() -> dict[str, Any]:
    return json.loads((_CONFIG / "gripper.v1.json").read_text(encoding="utf-8"))



@lru_cache(maxsize=1)
def joint_limits() -> dict[str, tuple[float, float]]:
    """The declared range of each joint, in radians.

    Read from the manifest rather than assumed, and actually ENFORCED, which it
    was not: the search that places the arm was free coordinate descent with no
    limits at all, so a manifest declaring the wrist at plus or minus 120
    degrees watched it work at -199, folded through itself and intersecting the
    block. A range nothing checks is a comment.
    """
    out: dict[str, tuple[float, float]] = {}
    for joint in spec()["kinematics"]["joints"]:
        if "range_deg" in joint:
            low, high = joint["range_deg"]
            out[joint["name"]] = (float(np.radians(low)), float(np.radians(high)))
    return out


@lru_cache(maxsize=1)
def rate_limits() -> dict[str, float]:
    """Radians per second each joint may move, and metres per second the fingers.

    Separate from the range of motion because they answer different questions.
    A range says where a joint may BE; a rate says how fast it may get there,
    and every pose along a snap is inside the envelope. That is exactly how the
    elbow reached 472 deg/s while never leaving its declared 140 degrees.
    """
    document = spec().get("rate_limits", {})
    out = {
        name: float(np.radians(value))
        for name, value in document.get("joints_deg_per_s", {}).items()
    }
    out["finger"] = float(document.get("finger_m_per_s", 0.07))
    return out


def rate_limited(current: "GripperState", target: "GripperState",
                 dt: float) -> "GripperState":
    """Move toward the target no faster than the manifest allows."""
    limits = rate_limits()
    out = current.copy()
    for name in ("yaw", "lift", "elbow", "wrist", "finger"):
        start = float(getattr(current, name))
        want = float(getattr(target, name))
        ceiling = float(limits.get(name, np.inf)) * dt
        step = float(np.clip(want - start, -ceiling, ceiling))
        setattr(out, name, start + step)
    return out


def within_limits(state: "GripperState") -> bool:
    limits = joint_limits()
    for name in ("yaw", "lift", "elbow", "wrist"):
        low, high = limits.get(name, (-np.inf, np.inf))
        if not (low <= getattr(state, name) <= high):
            return False
    return clears_support(state)


def clears_support(state: "GripperState") -> bool:
    """Is the whole arm above the surface it is bolted to?

    Joint ranges say what each joint may do; they do not say where the arm may
    BE. Every angle can be legal while the linkage they add up to is buried:
    measured, the first elbow reached 30 cm under the table, which reads on
    screen as an arm rising out of the floor. A range of motion is not a
    reachable workspace, and the difference has to be checked separately.
    """
    support = float(spec()["kinematics"].get("support_height_m", -np.inf))
    place = forward(state)
    for point in place["joints"]:
        if float(point[1]) < support - _SUPPORT_MARGIN_M:
            return False
    for key in ("left_pad", "right_pad", "plate"):
        if float(place[key][1]) < support - _SUPPORT_MARGIN_M:
            return False
    return True


#: How far below the surface a link may dip before the pose is refused, metres.
#: Not zero: the pads have to reach an object resting ON the surface, so its
#: own thickness is fair game and the table plane is not a hard ceiling for
#: everything.
_SUPPORT_MARGIN_M = 0.02


def clamp(state: "GripperState") -> "GripperState":
    limits = joint_limits()
    out = state.copy()
    for name in ("yaw", "lift", "elbow", "wrist"):
        low, high = limits.get(name, (-np.inf, np.inf))
        setattr(out, name, float(np.clip(getattr(out, name), low, high)))
    return out

@dataclass
class GripperState:
    """Everything the machine is, as joint values.

    Four angles and two finger openings. The humanoid's equivalent is a
    dictionary of nineteen bone quaternions, and the fact that both drive the
    same controller is the point of this file.
    """

    # HOME is up and back, clear of everything, and re-fitted whenever the
    # mounting moves -- a home pose is a fact about where the base is, not a
    # constant. Mounted flat on the table the first default put the pads BELOW
    # the surface the arm works on; raised onto the pedestal, the same numbers
    # put them 76 cm up and out of reach entirely.
    #
    # Searched against the current mount for a pose standing 34 cm above the
    # table and 41 cm back from the block, with every joint inside its declared
    # range and no link under the surface.
    yaw: float = 0.0
    lift: float = 1.70
    elbow: float = -2.40
    wrist: float = -0.20
    finger: float = 0.05

    def as_array(self) -> np.ndarray:
        return np.asarray([self.yaw, self.lift, self.elbow, self.wrist, self.finger])

    def copy(self) -> "GripperState":
        return GripperState(self.yaw, self.lift, self.elbow, self.wrist, self.finger)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) * cos + sin * cross + (1.0 - cos) * np.outer(axis, axis)


def forward(state: GripperState) -> dict[str, Any]:
    """Where every part of the gripper is, in world metres.

    The equivalent of canonical_positions for the hand: the one function that
    turns joint values into places, which every metric then reads.
    """
    document = spec()
    base = np.asarray(document["kinematics"]["base"]["at"], dtype=float)
    lengths = document["kinematics"]["segments_m"]
    finger = document["kinematics"]["finger"]

    yaw = _rotation(np.array([0.0, 1.0, 0.0]), state.yaw)
    pitch_axis = yaw @ np.array([1.0, 0.0, 0.0])

    # Each segment turns about the pitch axis, in the plane the yaw picked.
    angle = 0.0
    point = base
    joints = [point]
    frame = yaw
    for length, turn in zip(lengths, (state.lift, state.elbow, state.wrist)):
        angle += turn
        frame = _rotation(pitch_axis, angle) @ yaw
        # The arm's own long axis, swung by the accumulated angle.
        direction = frame @ np.array([0.0, 0.0, 1.0])
        point = point + direction * length
        joints.append(point)

    # The plate between the fingers, and the axis they slide along.
    plate = joints[-1]
    approach = frame @ np.array([0.0, 0.0, 1.0])
    across = frame @ np.array([1.0, 0.0, 0.0])
    reach = float(finger["length_m"])
    left_pad = plate + across * state.finger + approach * reach
    right_pad = plate - across * state.finger + approach * reach
    return {
        "base": base,
        "joints": joints,
        "plate": plate,
        "approach": approach,
        "across": across,
        "left_pad": left_pad,
        "right_pad": right_pad,
        # The gap between the GRIPPING SURFACES, not between the pad centres.
        # Each pad is a box, so its inner face is a finger-thickness closer to
        # the middle than its centre is, and reporting centres overstates the
        # aperture by 2.4 cm here: an opening read as 7.9 cm around a 6 cm block
        # was really 5.5 cm and already gripping it. The same error as measuring
        # a fingertip's joint origin instead of the capsule that touches.
        "opening": float(max(
            0.0, np.linalg.norm(left_pad - right_pad)
            - 2.0 * float(finger["thickness_m"]))),
        "pad_centres_m": float(np.linalg.norm(left_pad - right_pad)),
    }


# ---------------------------------------------------------------------------
# The same metrics, by the same names, computed from a different body.
# ---------------------------------------------------------------------------

def palm_to_object_m(state: GripperState, obj: np.ndarray, half: np.ndarray) -> float:
    """Plate to the object's surface. The hand measures a palm pad; same idea."""
    place = forward(state)
    # The point that would do the holding is between the pads, not the plate.
    middle = (place["left_pad"] + place["right_pad"]) / 2.0
    offset = np.abs(middle - obj) - half
    return float(np.linalg.norm(np.maximum(offset, 0.0)))


def palm_facing(state: GripperState, obj: np.ndarray, half: np.ndarray) -> float:
    """+1 when the gripper looks straight at the face it is approaching."""
    place = forward(state)
    face = chosen_face(state, obj, half)
    if face is None:
        return 0.0
    _centre, normal = face
    return float(np.dot(place["approach"], -normal))


def object_in_grasp_m(state: GripperState, obj: np.ndarray) -> float:
    """Object to the line between the finger pads."""
    place = forward(state)
    a, b = place["left_pad"], place["right_pad"]
    span = b - a
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        return float(np.linalg.norm(obj - a))
    unit = span / length
    along = float(np.clip(float(np.dot(obj - a, unit)), 0.0, length))
    return float(np.linalg.norm(obj - (a + unit * along)))


def grip_tip_spread_m(state: GripperState) -> float:
    return forward(state)["opening"]


def object_faces(obj: np.ndarray, half: np.ndarray):
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            yield obj + normal * half[axis], normal


def chosen_face(state: GripperState, obj: np.ndarray, half: np.ndarray):
    """The face the gripper should present itself to: whichever it is in front of."""
    place = forward(state)
    middle = (place["left_pad"] + place["right_pad"]) / 2.0
    best = None
    for centre, normal in object_faces(obj, half):
        toward = middle - centre
        size = float(np.linalg.norm(toward))
        if size < 1e-9:
            continue
        score = float(np.dot(toward / size, normal))
        if best is None or score > best[0]:
            best = (score, centre, normal)
    return None if best is None else (best[1], best[2])


# ---------------------------------------------------------------------------
# The same four primitives.
# ---------------------------------------------------------------------------

#: How far off the face the gripper stops before closing, metres.
_FACE_STANDOFF_M = 0.02
#: Lateral error above which the approach lines up before coming in, metres.
_APPROACH_ALIGN_M = 0.03
#: How far back along the normal that lining up happens, metres.
_APPROACH_LANE_M = 0.10
#: Room left around the object when the grip opens, metres.
_GRIP_CLEARANCE_M = 0.02
#: Contact force at which a finger has arrived and stops advancing, newtons.
_SETTLED_FORCE_N = 0.8
#: The firmest squeeze, newtons.
_SQUEEZE_FORCE_N = 11.0
#: Below this on both fingers a lift becomes a squeeze instead, newtons.
_CARRY_FORCE_N = 7.0
#: Object speed above which the approach is pushing rather than arriving.
_PUSHING_SPEED_M_S = 0.02
#: Force at which the arm stops advancing, newtons.
_ARM_STOP_FORCE_N = 2.5

#: How far inside the object's width the pads are allowed to close, metres.
#: This is compression, and in this actuation model it is where the grip
#: force comes from. Honest about it rather than hidden.
_SKIN_BITE_M = 0.004

#: Radians of shoulder rotation per unit of lift amplitude, per frame.
_LIFT_RATE_RAD = 0.04


def holding(forces: dict[str, float], threshold_n: float = 0.5) -> bool:
    """Both pads loaded. The gripper's version of an opposing pair."""
    return (forces.get("finger_left", 0.0) >= threshold_n
            and forces.get("finger_right", 0.0) >= threshold_n)


def solve_move_to(state: GripperState, obj: np.ndarray, half: np.ndarray,
                  forces: dict[str, float], velocity: np.ndarray,
                  amount: float) -> GripperState | None:
    """Carry the gripper to a face, arriving square, by search over the joints.

    Every refusal the hand makes, made here for the same reasons: do not drive
    an actuator through what it is touching, do not approach what you already
    hold, do not chase something you are pushing.
    """
    if max(forces.values(), default=0.0) >= _ARM_STOP_FORCE_N:
        return None
    if holding(forces):
        return None
    if float(np.linalg.norm(velocity)) >= _PUSHING_SPEED_M_S:
        return None

    face = chosen_face(state, obj, half)
    if face is None:
        return None
    centre, normal = face
    place = forward(state)
    middle = (place["left_pad"] + place["right_pad"]) / 2.0
    lateral = (middle - centre) - normal * float(np.dot(middle - centre, normal))
    if float(np.linalg.norm(lateral)) > _APPROACH_ALIGN_M:
        # Far out along the face's normal, where a gripper swinging across
        # cannot touch anything, and line up there first.
        goal = centre + normal * (_FACE_STANDOFF_M + _APPROACH_LANE_M)
    else:
        # Then bring the pad midpoint to the object's CENTRE, not to a standoff
        # off its surface. A parallel gripper holds by straddling: its pads have
        # to end up either side of the object, so the point between them belongs
        # inside it. Aimed 2 cm off the top face instead, the gripper arrived
        # perfectly square -- palm_facing 1.00, 2.09 cm away -- with the object
        # 6.09 cm below the line between its pads, and stalled there for seven
        # seconds because nothing was between them to close on.
        goal = obj

    def cost(candidate: GripperState) -> float:
        spot = forward(candidate)
        mid = (spot["left_pad"] + spot["right_pad"]) / 2.0
        square = 1.0 - float(np.dot(spot["approach"], -normal))
        return float(np.linalg.norm(mid - goal)) + square * 0.25

    # Coordinate descent over the four angles. Small, analytic, and enough:
    # the arm has four degrees of freedom and one target.
    best = state.copy()
    here = cost(best)
    for _pass in range(6):
        improved = False
        for name in ("yaw", "lift", "elbow", "wrist"):
            for step in (0.20, 0.06, 0.02):
                for direction in (1.0, -1.0):
                    trial = best.copy()
                    setattr(trial, name, getattr(trial, name) + step * direction)
                    # A pose outside the declared range is not a candidate. The
                    # search used to accept them and the arm reached -199 on a
                    # joint declared at +/-120, which is where the folded links
                    # and the segments through the block came from.
                    if not within_limits(trial):
                        continue
                    value = cost(trial)
                    if value < here - 1e-5:
                        best, here, improved = trial, value, True
    if not improved and here >= cost(state):
        return None
    reach = float(np.clip(amount, 0.0, 1.0))
    moved = state.copy()
    for name in ("yaw", "lift", "elbow", "wrist"):
        start = getattr(state, name)
        setattr(moved, name, start + (getattr(best, name) - start) * reach)
    return clamp(moved)


def solve_open_grip(state: GripperState, obj: np.ndarray, half: np.ndarray,
                    forces: dict[str, float], amount: float) -> GripperState | None:
    """Open to the object's width plus clearance -- not as wide as it goes."""
    if holding(forces):
        return None
    document = spec()
    limit = document["kinematics"]["joints"][4]["range_m"][1]
    thickness = float(document["kinematics"]["finger"]["thickness_m"])
    # Travel is measured to the pad centre, the object to its surface, so the
    # thickness has to be added or the gripper opens to the object's width and
    # then finds it cannot admit it.
    wanted = float(np.min(half)) + _GRIP_CLEARANCE_M + thickness
    target = float(np.clip(wanted, 0.0, limit))
    reach = float(np.clip(amount, 0.0, 1.0))
    moved = state.copy()
    moved.finger = state.finger + (target - state.finger) * reach
    return moved


def solve_close_grip(state: GripperState, obj: np.ndarray, half: np.ndarray,
                     forces: dict[str, float], amount: float) -> GripperState | None:
    """Bring the fingers together, stopping at contact and at the object."""
    reach = float(np.clip(amount, 0.0, 1.0))
    hold_to = _SETTLED_FORCE_N + (_SQUEEZE_FORCE_N - _SETTLED_FORCE_N) * reach
    if min(forces.get("finger_left", 0.0),
           forces.get("finger_right", 0.0)) >= hold_to:
        return None
    thickness = float(spec()["kinematics"]["finger"]["thickness_m"])
    # Just INSIDE the object's width, not exactly at it. Closed to the exact
    # width the pads kiss the block and register nothing: a position-driven
    # gripper makes its holding force out of compression, so a floor at the
    # object's own size is a floor at zero grip. The same fact the humanoid
    # hand reports, and the reason both need torque-controlled digits before
    # any of this counts as a real grasp.
    floor = float(np.min(half)) + thickness - _SKIN_BITE_M
    if state.finger <= floor:
        return None
    step = (state.finger - floor) * 0.35 * max(reach, 0.25)
    moved = state.copy()
    moved.finger = max(floor, state.finger - step)
    return moved


# ---------------------------------------------------------------------------
# Placing: the second half of pick AND place.
# ---------------------------------------------------------------------------

def bin_spec() -> dict:
    return spec()["scene"]["bin"]


def object_over_target_m(obj: np.ndarray) -> float:
    """Horizontal distance from the object to the middle of the bin.

    Height is deliberately ignored. Carrying is a problem in the plane; the
    height is what clearing the rim is about, and mixing them into one distance
    makes a gripper that is perfectly placed but too low look the same as one
    that is high and in the wrong county.
    """
    centre = np.asarray(bin_spec()["centre"], dtype=float)
    flat = np.asarray([obj[0] - centre[0], 0.0, obj[2] - centre[2]])
    return float(np.linalg.norm(flat))


def object_above_rim_m(obj: np.ndarray, half: np.ndarray) -> float:
    """How far the object's underside clears the bin's rim.

    Negative means carrying it across would strike the wall.
    """
    rim = float(bin_spec()["rim_height_m"])
    return float(obj[1] - float(half[1]) - rim)


def object_in_target(obj: np.ndarray) -> bool:
    """Is the object inside the bin?"""
    document = bin_spec()
    centre = np.asarray(document["centre"], dtype=float)
    inner = np.asarray(document["inner_half_m"], dtype=float)
    return bool(
        abs(obj[0] - centre[0]) <= inner[0]
        and abs(obj[2] - centre[2]) <= inner[2]
        and obj[1] <= float(document["rim_height_m"])
        and obj[1] >= centre[1] - inner[1]
    )


#: How high above the rim the object is carried before it is let go, metres.
_CLEARANCE_M = 0.06
#: Horizontal error at which the object counts as over the bin, metres.
_OVER_TARGET_M = 0.03


def solve_carry_over(state: GripperState, obj: np.ndarray, half: np.ndarray,
                     forces: dict[str, float], amount: float) -> GripperState | None:
    """Carry what is held to a point above the bin, clear of its rim."""
    if not holding(forces):
        return None
    if (object_over_target_m(obj) <= _OVER_TARGET_M
            and object_above_rim_m(obj, half) >= 0.0):
        return None

    document = bin_spec()
    centre = np.asarray(document["centre"], dtype=float)
    # The pads go where the OBJECT needs to be, offset by however far the object
    # currently sits from them. Steering the gripper to the bin instead leaves
    # the block hanging beside it by exactly that offset.
    place = forward(state)
    middle = (place["left_pad"] + place["right_pad"]) / 2.0
    carry_offset = middle - obj
    goal = np.asarray([
        centre[0],
        float(document["rim_height_m"]) + _CLEARANCE_M + float(half[1]),
        centre[2],
    ]) + carry_offset

    def cost(candidate: GripperState) -> float:
        spot = forward(candidate)
        mid = (spot["left_pad"] + spot["right_pad"]) / 2.0
        return float(np.linalg.norm(mid - goal))

    best = state.copy()
    here = cost(best)
    for _pass in range(6):
        for name in ("yaw", "lift", "elbow", "wrist"):
            for step in (0.14, 0.05, 0.015):
                for direction in (1.0, -1.0):
                    trial = best.copy()
                    setattr(trial, name, getattr(trial, name) + step * direction)
                    if not within_limits(trial):
                        continue
                    value = cost(trial)
                    if value < here - 1e-5:
                        best, here = trial, value
    reach = float(np.clip(amount, 0.0, 1.0))
    moved = state.copy()
    for name in ("yaw", "lift", "elbow", "wrist"):
        start = getattr(state, name)
        setattr(moved, name, start + (getattr(best, name) - start) * reach)
    return clamp(moved)


def solve_release(state: GripperState, obj: np.ndarray, half: np.ndarray,
                  amount: float) -> GripperState | None:
    """Let go, deliberately.

    open_grip refuses while the gripper is holding something, on purpose:
    opening a loaded hand is how a grasp gets lost by accident, and that refusal
    has saved several runs. Releasing is the same motion asked for ON PURPOSE,
    so it is a separate word -- the accident and the intention should not share
    a name, or the guard has to decide which one you meant.
    """
    if object_over_target_m(obj) > _OVER_TARGET_M * 2.0:
        return None
    document = spec()
    limit = document["kinematics"]["joints"][4]["range_m"][1]
    reach = float(np.clip(amount, 0.0, 1.0))
    moved = state.copy()
    moved.finger = min(limit, state.finger + (limit - state.finger) * reach)
    return moved


def solve_lift(state: GripperState, forces: dict[str, float],
               amount: float) -> GripperState | None:
    """Raise the arm, still gripping. Too early, and it squeezes instead."""
    weakest = min(forces.get("finger_left", 0.0), forces.get("finger_right", 0.0))
    if weakest < _CARRY_FORCE_N:
        return None
    reach = min(float(amount), 0.3)
    moved = state.copy()
    # Rotating the shoulder up carries whatever the fingers hold -- slowly. An
    # angle at the base is a much larger motion at the tip: 0.25 rad per unit of
    # amplitude is about 21 degrees a second, which is 0.18 m/s half a metre out,
    # and it shears the block straight out of a 19 N grip. The hand learned the
    # same thing in metres; here it arrives as radians.
    # WHICH WAY IS UP is measured, not assumed. Positive shoulder rotation
    # lowers this arm, and adding it pressed the block into the table while the
    # controller reported a lift phase. Assuming a sign convention is exactly
    # what hid the left hand's failure for a whole session, so the direction is
    # taken from the geometry each time: whichever way raises the pads.
    here = _pad_height(state)
    up = 1.0
    probe = state.copy()
    probe.lift = state.lift + _LIFT_RATE_RAD
    if _pad_height(probe) < here:
        up = -1.0
    moved.lift = state.lift + up * reach * _LIFT_RATE_RAD
    return clamp(moved)


def _pad_height(state: GripperState) -> float:
    place = forward(state)
    return float((place["left_pad"][1] + place["right_pad"][1]) / 2.0)
