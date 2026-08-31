"""Opening things that are attached to something.

A door, a drawer, a lid, a lever. What they have in common is not their shape --
it is that once you have hold of them you no longer choose where your hand goes.
The mechanism does. A hinge permits an arc, a slide permits a line, and any pull
that is not along what is permitted is a pull the mechanism simply refuses.

So this does not model doors. It contains no hinge, no axis, no swing limit and
no arc. It discovers what is permitted the only way a machine without a drawing
of the furniture can: it pushes, watches where its own hand actually ended up,
and goes that way next.

    command a step  ->  the mechanism deflects it  ->  the deflection IS the
    direction the mechanism allows  ->  go there, and ask again

That fixed point converges to the constraint's local tangent, and because it is
re-asked every step it tracks a tangent that rotates -- which is what makes the
same code follow an arc and a straight line without being told which it is.

The previous version computed the handle's position from the hinge location and
the door's opening angle, both read out of the manifest. It worked on exactly
one piece of furniture, and it worked by being told the answer.

WHAT IT NEEDS: the hand's own position, which is forward kinematics from the
encoders, and to be holding something. That is all. It never asks what the thing
is, where it pivots, or how far it opens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: A hand movement smaller than this is noise, not the mechanism yielding.
_MOVED_M = 0.0012
#: How much of the newly observed direction to believe each step. Low enough to
#: ride out one noisy frame, high enough to track an arc without lagging it.
_TRUST_NEW = 0.35
#: Frames of no movement before concluding the current direction is barred.
_BARRED_AFTER = 6
#: How far to swing the heading when looking for a direction that yields. Real
#: search, alternating sides and widening -- a hinge tangent can be ninety
#: degrees from the pull that first engaged it.
_CAST_DEG = (18.0, -18.0, 36.0, -36.0, 55.0, -55.0, 75.0, -75.0)
#: Having moved at least this far, a mechanism that stops yielding is open
#: rather than not yet started, metres.
_STARTED_M = 0.012
#: How far to push ahead of the hand along the permitted direction, metres.
#: A goal, not a step: the mechanism decides how much of it actually happens.
PUSH_M = 0.05


@dataclass
class Mechanism:
    """What has been learned about the thing being opened, by moving it.

    Every field is an observation of the machine's OWN hand. Nothing here is
    read from a manifest and nothing describes the mechanism's construction --
    only what it has been seen to permit.
    """

    #: The direction the mechanism has been observed to allow, in the world.
    heading: np.ndarray | None = None
    #: Total distance the held thing has been moved along it.
    opened_m: float = 0.0
    #: Consecutive frames the hand has not moved while being asked to.
    still_for: int = 0
    #: Which cast we are trying, when the current heading stops yielding.
    casting: int = 0
    _was_at: np.ndarray | None = field(default=None, repr=False)

    def start(self, pull: np.ndarray) -> None:
        """Begin, with a first guess at which way to go.

        The guess is not knowledge of the mechanism -- it is the one thing true
        of nearly everything that opens: it comes toward you. If it is wrong the
        casting will find the real direction, at the cost of a second or two.
        """
        if self.heading is None:
            size = float(np.linalg.norm(pull))
            self.heading = pull / size if size > 1e-9 else np.asarray([0.0, -1.0, 0.0])

    def observe(self, hand_at: np.ndarray) -> None:
        """Take in where the hand actually is, and update what is permitted."""
        here = np.asarray(hand_at, dtype=float)
        if self._was_at is None:
            self._was_at = here
            return
        moved = here - self._was_at
        self._was_at = here
        distance = float(np.linalg.norm(moved))

        if distance < _MOVED_M:
            self.still_for += 1
            return

        # It moved: that displacement is the mechanism telling us what it
        # allows. Believe most of the old heading and some of the new one, so a
        # single jittery frame cannot spin the direction round.
        self.still_for = 0
        self.casting = 0
        self.opened_m += distance
        fresh = moved / distance
        if self.heading is None:
            self.heading = fresh
        else:
            blended = (1.0 - _TRUST_NEW) * self.heading + _TRUST_NEW * fresh
            size = float(np.linalg.norm(blended))
            if size > 1e-9:
                self.heading = blended / size

    def barred(self) -> bool:
        """The current direction has stopped yielding."""
        return self.still_for >= _BARRED_AFTER

    def opened(self) -> bool:
        """It moved, and then no direction moved it any further.

        Not "it stopped": a hinge stops the moment you pull square into it, and
        that happens repeatedly on the way round. It is open when it has moved,
        and then every direction in the cast has been tried without it moving
        again -- which is the difference between binding and being against the
        stop, expressed without knowing what kind of stop it is.
        """
        return (self.opened_m >= _STARTED_M
                and self.casting >= len(_CAST_DEG))

    def aim(self, up: np.ndarray | None = None) -> np.ndarray:
        """Which way to push now.

        Normally the direction last observed to work. When that has stopped
        yielding, cast around it -- a hinge's tangent turns as it opens, and a
        pull that engaged the handle at the start can be square into the hinge a
        quarter of a turn later.
        """
        if self.heading is None:
            return np.asarray([0.0, -1.0, 0.0])
        if not self.barred():
            return self.heading
        axis = np.asarray([0.0, 0.0, 1.0]) if up is None else np.asarray(up)
        turn = np.radians(_CAST_DEG[self.casting % len(_CAST_DEG)])
        return _turn_about(self.heading, axis, turn)

    def cast_wider(self) -> None:
        """Give up on this cast and try the next one."""
        self.casting += 1
        self.still_for = 0


def _turn_about(vector: np.ndarray, axis: np.ndarray, radians: float) -> np.ndarray:
    """Rodrigues rotation. The casting turns about vertical for upright things."""
    axis = axis / float(np.linalg.norm(axis))
    return (vector * np.cos(radians)
            + np.cross(axis, vector) * np.sin(radians)
            + axis * float(np.dot(axis, vector)) * (1.0 - np.cos(radians)))
