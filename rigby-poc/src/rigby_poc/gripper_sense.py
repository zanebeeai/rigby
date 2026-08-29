"""What a cheap gripper could actually know, and nothing else.

The controller had been reading the simulator: the object's true position, its
true half-extents, and a per-pad contact force, in eighteen places. None of
those exist on a real machine of this class. ALOHA (Zhao, Kumar, Levine, Finn,
2023) does battery-slotting and cup-opening at 80-90% with joint positions and
RGB cameras and NOTHING ELSE -- no object pose, no object size, no force or
torque sensing -- which is the standard worth holding to, because every sensor
assumed away here is one that has to exist and be calibrated later.

So this instrument list is deliberately short:

JOINT ENCODERS. Exact, and the only thing that is. Every arm has them.

ONE CAMERA on the wrist, looking along the approach axis. It reports the object
only while the object is in front of it and within range, as an estimate with
error, not as a fact. Out of view, the controller has a memory and knows it is
a memory.

NO FORCE SENSOR. Contact is inferred from the encoders: a finger told to close,
which has stopped moving, and which is not shut, has something between the pads.
Note the shape of that test -- it is NOT "the finger failed to reach its
target". Under a commanded squeeze a blocked finger is driven PAST its position
target and pinned there by the object, so the tracking error goes the other way,
and testing its sign detects contact exactly when there is none. Stopped-and-not-
shut is the signature that survives being pushed. It is free on any encoder,
it is how cheap hardware actually finds an object, and it removes the most
expensive instrument on the list.

GRIP EFFORT from motor command, not from a load cell. What the controller knows
is how hard it is pushing, which is a current reading, rather than what the
object feels, which is not measurable without a sensor in the fingertip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gripper_torque import Body

#: The wrist camera's half-angle and useful range, metres. A short cone, because
#: a wrist camera cannot see what the hand is not pointing at, and that
#: limitation is most of what makes looking a decision rather than a given.
_FOV_DEG = 40.0
_RANGE_M = 0.75
_NEAR_M = 0.03

#: Fractional error on a camera's estimate of where something is and how big it
#: is. Fixed rather than random so a run is reproducible and a failure can be
#: repeated; the point is that the number is WRONG, not that it is noisy.
_POSITION_ERROR = 0.004
_SIZE_ERROR = 0.08

#: A finger moving slower than this, in m/s, has stopped. Encoders differenced
#: between frames, which is what a velocity reading on real hardware is.
_FINGER_STILL = 0.004
#: How far from fully shut a stopped finger must be before its stopping means
#: something is in the way rather than that it simply arrived, metres.
_NOT_SHUT_M = 0.006
#: The arm has stopped short of where it was sent: this much angle behind its
#: command, in radians, AND moving slower than this, in rad/s. Both halves are
#: needed. Lag alone is not obstruction -- an arm slewing at its rate limit sits
#: a couple of degrees behind its target the whole way, which is what following
#: a moving target looks like, and testing lag alone declared every fast reach
#: obstructed. An arm that is behind AND has stopped is against something.
_ARM_STALL_RAD = 0.08
_ARM_STILL = 0.08


def _finger_floor() -> float:
    """The travel a fully shut finger reports. Fixed by the mechanism."""
    from .gripper import spec
    for joint in spec()["kinematics"]["joints"]:
        if joint["name"] == "finger_left":
            return float(joint["range_m"][0])
    return 0.0


@dataclass
class Sensed:
    """One observation. Everything here is obtainable by a real machine."""

    q: np.ndarray
    time_s: float
    #: Where the object is BELIEVED to be, and whether that belief is current.
    object_at: np.ndarray | None
    object_seen: bool
    #: Estimated half-extents, from the camera, once it has had a look.
    object_size: np.ndarray | None
    #: Inferred from the encoders, not measured with a load cell.
    contact_left: bool
    contact_right: bool
    #: How hard the fingers are being driven, in newtons of command.
    grip_effort_n: float
    #: The arm has stopped short of where it was sent.
    arm_stalled: bool

    def holding(self) -> bool:
        """Both pads meeting resistance while the grip is being driven closed.

        The gripper's version of an opposing pair, without a force sensor: two
        fingers that will not close further while being told to close are two
        fingers with something between them.
        """
        return self.contact_left and self.contact_right and self.grip_effort_n > 1.0


@dataclass
class Senses:
    """The camera, and everything the machine remembers between frames.

    Kept as an object because a belief has to persist. The alternative --
    recomputing what is visible each frame and passing the truth through when it
    is not -- is exactly the leak this class exists to stop.
    """

    last_at: np.ndarray | None = None
    last_size: np.ndarray | None = None
    last_seen_s: float | None = None
    frames_seen: int = 0
    #: Whether the fingers are latched onto something. A contact reading is an
    #: instantaneous test and the hand jolts: accelerating into the carry shakes
    #: the fingers past the stopped-moving threshold for a few frames, the grip
    #: reads as empty, and the controller refuses to carry what it is plainly
    #: carrying. Knowing you are holding something is a STATE, entered on
    #: contact and left when you open your hand -- not a fresh measurement each
    #: frame. Real grippers latch for the same reason.
    latched: bool = False

    def look(self, body: Body, now: float) -> tuple[np.ndarray | None, bool,
                                                    np.ndarray | None]:
        """What the camera reports this frame."""
        eye = body.body_at("plate")
        forward = body.approach()
        truth = body.block()
        toward = truth - eye
        distance = float(np.linalg.norm(toward))
        if distance < 1e-9:
            return self.last_at, False, self.last_size
        angle = float(np.degrees(np.arccos(
            np.clip(float(np.dot(toward / distance, forward)), -1.0, 1.0))))
        visible = (angle <= _FOV_DEG and _NEAR_M <= distance <= _RANGE_M)
        if not visible:
            return self.last_at, False, self.last_size

        # An estimate, not the truth. The error is along the viewing direction,
        # because that is where a single camera is worst: bearing is easy and
        # depth is not.
        bias = (toward / distance) * (_POSITION_ERROR * distance / _RANGE_M)
        self.last_at = truth + bias
        self.last_size = np.asarray(body.block_half) * (1.0 + _SIZE_ERROR)
        self.last_seen_s = now
        self.frames_seen += 1
        return self.last_at, True, self.last_size


def sense(body: Body, eyes: Senses, commanded: np.ndarray, squeeze_n: float,
          now: float) -> Sensed:
    """One honest observation of the machine and its world."""
    q = np.asarray(body.q())
    at, seen, size = eyes.look(body, now)

    # Contact from the encoders alone. Being driven shut, having stopped, and
    # not being shut: three readings a stepper controller already has.
    qd = np.asarray(body.qd())
    shut = _finger_floor()
    driving = squeeze_n > 0.0 or bool(np.any(commanded[4:] < q[4:] - 1e-6))
    left = bool(driving and abs(float(qd[4])) < _FINGER_STILL
                and q[4] > shut + _NOT_SHUT_M)
    right = bool(driving and abs(float(qd[5])) < _FINGER_STILL
                 and q[5] > shut + _NOT_SHUT_M)

    # Latch it. Enter on both pads reporting resistance, leave when the hand is
    # told to open or when the fingers have run all the way shut -- meaning
    # whatever was between them is not there any more.
    if left and right:
        eyes.latched = True
    elif not driving or float(np.min(q[4:])) <= shut + _NOT_SHUT_M:
        eyes.latched = False
    behind = np.abs(commanded[:4] - q[:4]) > _ARM_STALL_RAD
    stopped = np.abs(qd[:4]) < _ARM_STILL
    arm_stalled = bool(np.any(behind & stopped))

    return Sensed(
        q=q,
        time_s=now,
        object_at=at,
        object_seen=seen,
        object_size=size,
        contact_left=left or eyes.latched,
        contact_right=right or eyes.latched,
        grip_effort_n=float(squeeze_n),
        arm_stalled=arm_stalled,
    )


def audit(body: Body) -> dict[str, str]:
    """What is readable, and by what instrument. The boundary, in one place."""
    return {
        "joint positions": "encoders, exact",
        "object position": "one wrist camera, only while in view, with error",
        "object size": "the same camera, estimated once seen",
        "contact": "driven shut, stopped, and not shut -- encoders, no force "
                   "sensor",
        "grip effort": "motor command, not a load cell",
        "bin position": "known a priori: it is fixed furniture, not a percept",
        "NOT available": "true object pose, true object size, contact force, "
                         "anything about the object while it is out of view",
    }
