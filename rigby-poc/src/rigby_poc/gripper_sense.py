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

ONE CAMERA on the wrist, looking along the approach axis. It is a real camera --
declared in the model, rendered through, and read as PIXELS. The previous
version computed a field-of-view cone geometrically and then returned the true
position with an error added, which tests the controller against a plausible
error but still reads the answer out of the simulator. This one segments the
image and works the position out from where the object appears, so being wrong
is a property of the view rather than a number chosen here.

Range comes from the GROUND PLANE, not from knowing the object's size: the
bottom edge of the blob is where the object meets the bench, the bench height is
known furniture, so the ray through that pixel meets it at exactly one point.
Size then follows from how large the object appears at that range. This is the
standard monocular trick and it carries the standard monocular error -- the
bottom edge seen from an angle is the object's near-bottom corner, not the point
under its centre, so estimates are biased toward the camera. That bias is real
and the controller has to tolerate it.

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

#: What the camera renders at. Coarse on purpose: a wrist camera is not a
#: measuring instrument, and a metre of range across 150 rows is about the
#: resolution these estimates deserve.
_VIEW_W, _VIEW_H = 200, 150
#: Vertical field of view, matching the camera declared in the model.
_FOV_Y_DEG = 70.0
#: Fewer lit pixels than this is noise, not an object.
_MIN_BLOB_PX = 25
#: The range band a look is worth believing over, metres. There is a FAR limit
#: for the obvious reason and a NEAR one that is not obvious at all: measured,
#: the estimate gets WORSE as the hand closes in, from 2-5 mm at a quarter of a
#: metre to 25-35 mm at eleven centimetres. Close up the object fills the frame,
#: its top face reads as part of the silhouette, and the fingers start to cut
#: into the blob. So the machine looks from a step back, keeps that answer, and
#: does not let the approach spoil it.
_RANGE_M = 0.9
_TRUST_NEAR_M = 0.20

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
    #: How far the last accepted look moved the belief, and how many there have
    #: been. Whether to trust the estimate is itself something the machine can
    #: work out, and it is what says when looking is finished.
    belief_shift: float = 1.0
    looks: int = 0

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
    #: How far the last accepted look moved the belief, metres, and how many
    #: looks have been accepted. A belief that stops moving is a belief worth
    #: acting on; one look is never enough to know that.
    shift: float = 1.0
    looks: int = 0

    def look(self, body: Body, now: float, bench: float
             ) -> tuple[np.ndarray | None, bool, np.ndarray | None]:
        """Find the object in the wrist image, or report that it is not there."""
        image = body.view(_VIEW_W, _VIEW_H)
        red = image[:, :, 0].astype(np.int16)
        green = image[:, :, 1].astype(np.int16)
        blue = image[:, :, 2].astype(np.int16)
        # The object is the only warm thing in the scene: bench, arm, fingers
        # and bin are all blue-grey. Warmth survives the highlight blowing out
        # to white at close range, which a hue test does not.
        warm = (red > green + 12) & (green > blue + 4)
        if int(warm.sum()) < _MIN_BLOB_PX:
            return self.last_at, False, self.last_size

        rows, cols = np.nonzero(warm)
        u0, u1 = float(cols.min()), float(cols.max())
        v0, v1 = float(rows.min()), float(rows.max())
        # Touching the frame edge means the object is cut off, so its apparent
        # size is a lower bound and its bottom edge may not be its bottom.
        if u0 <= 0 or v0 <= 0 or u1 >= _VIEW_W - 1 or v1 >= _VIEW_H - 1:
            return self.last_at, False, self.last_size

        eye, orient = body.camera_pose()
        focal = (_VIEW_H / 2.0) / float(np.tan(np.radians(_FOV_Y_DEG) / 2.0))

        def ray(u: float, v: float) -> np.ndarray:
            local = np.asarray([(u - _VIEW_W / 2.0) / focal,
                                -(v - _VIEW_H / 2.0) / focal, -1.0])
            world = orient @ local
            return world / float(np.linalg.norm(world))

        # Where it is. The ray through the blob's CENTRE, met with the plane
        # the object's centre lies in -- which is the bench raised by half the
        # object's height, and that height is not known until the range is.
        # So solve it by iteration: guess a height, range off that plane,
        # measure the height that range implies, repeat. Three passes is
        # already stable to well under a millimetre.
        #
        # The first version ranged off the BOTTOM edge of the blob instead,
        # which needs no iteration and is wrong by four to seven centimetres:
        # seen from an angle the lowest lit pixel is the object's near-bottom
        # corner, not the point under its centre, so every estimate was pulled
        # toward the camera and every size came out inflated. The controller
        # then closed its fingers on the space beside the block.
        middle = ray((u0 + u1) / 2.0, (v0 + v1) / 2.0)
        if middle[2] > -1e-3:
            return self.last_at, False, self.last_size
        half_tall = 0.03
        centre = None
        for _pass in range(3):
            distance = (bench + half_tall - float(eye[2])) / float(middle[2])
            if not (0.0 < distance <= _RANGE_M):
                return self.last_at, False, self.last_size
            centre = eye + middle * distance
            half_tall = max((v1 - v0) / 2.0 * distance / focal, 0.004)
        if centre is None:
            return self.last_at, False, self.last_size
        if (float(np.linalg.norm(np.asarray(centre) - eye)) < _TRUST_NEAR_M
                and self.last_at is not None):
            # Too close to improve on what is already known. Report it as still
            # in sight, but do not overwrite a better look with a worse one.
            return self.last_at, True, self.last_size
        half_wide = max((u1 - u0) / 2.0 * distance / focal, 0.004)

        self.shift = (float(np.linalg.norm(np.asarray(centre) - self.last_at))
                      if self.last_at is not None else 1.0)
        self.looks += 1
        self.last_at = np.asarray(centre)
        # The camera sees two of the three half-extents. The depth it cannot see
        # is assumed as wide as the width it can, which is what any single view
        # has to do.
        self.last_size = np.asarray([half_wide, half_wide, half_tall])
        self.last_seen_s = now
        self.frames_seen += 1
        return self.last_at, True, self.last_size


def sense(body: Body, eyes: Senses, commanded: np.ndarray, squeeze_n: float,
          now: float, bench: float = 0.72) -> Sensed:
    """One honest observation of the machine and its world."""
    q = np.asarray(body.q())
    at, seen, size = eyes.look(body, now, bench)

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
        belief_shift=float(eyes.shift),
        looks=int(eyes.looks),
    )


def audit(body: Body) -> dict[str, str]:
    """What is readable, and by what instrument. The boundary, in one place."""
    return {
        "joint positions": "encoders, exact",
        "object position": "segmented from the wrist camera image, ranged off "
                           "the bench plane; only while fully in view",
        "object size": "apparent size in the same image at that range",
        "contact": "driven shut, stopped, and not shut -- encoders, no force "
                   "sensor",
        "grip effort": "motor command, not a load cell",
        "bin position": "known a priori: it is fixed furniture, not a percept",
        "NOT available": "true object pose, true object size, contact force, "
                         "anything about the object while it is out of view",
    }
