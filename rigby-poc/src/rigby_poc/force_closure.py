"""Close the hand on what is actually there, not on a pre-computed pose.

Ported from ``rigby-poc-general``'s ``contact/closure.py`` and adapted to the
humanoid. Their statement of the problem is the right one and is worth keeping:

    A grasp cannot be planned open-loop. The object's exact position, the
    friction, the moment the fingers first touch -- none of these are known until
    contact happens, and a trajectory that commands the fingers to a fixed closed
    pose either stops short of the object or drives straight through it.

That is exactly what this repository has been doing. Every grasp attempt so far
picked a per-digit curl by solving geometry against the object's *nominal* pose,
then played that curl open-loop. The object moves the instant it is touched, so
the pose that was solved is not the pose that is met: the measured outcomes were
a hand that missed by 9 cm, and later one that drove 20.6 mm into the block.

Three rules, theirs:

*Drive closure until contact, then hold.* Each digit advances toward its closed
limit at a bounded rate until it reports contact force, then stops. Nothing
commands a final finger position, because nothing knows one.

*Require opposition before declaring a grasp.* Contact on one side is a nudge;
contact on opposing sides is a grip.

*Never weld the object.* The object is held by measured contact force or it is
not held.

One adaptation. Their grippers are URDF models with real actuators, so closure is
a force command on a joint. The humanoid is a rendered rig with no actuators, so
closure here advances a normalized curl and reads the force back out of the
contacts the resulting pose produces. The loop is the same shape; only the
actuator is missing, and the curl stands in for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .models import BonePose, ClipFrame, Hand, HandShape, Quat, SceneObject
from .physics import (
    _body_id,
    _contact_digit,
    _contact_snapshot,
    _embodied_xml,
    _frame_hand_landmarks,
    _hand_segment_pairs,
    _seed_embodied_bodies,
    _set_embodied_mocap,
)
from .primitives import quat_euler

DIGITS = ("thumb", "index", "middle", "ring", "little")
_STEM = {
    "thumb": "Thumb", "index": "Index", "middle": "Middle",
    "ring": "Ring", "little": "Little",
}
_SEGMENTS = {
    "thumb": ("Metacarpal", "Proximal", "Distal"),
    "index": ("Proximal", "Intermediate", "Distal"),
    "middle": ("Proximal", "Intermediate", "Distal"),
    "ring": ("Proximal", "Intermediate", "Distal"),
    "little": ("Proximal", "Intermediate", "Distal"),
}


@dataclass(frozen=True)
class ClosureConfig:
    """Rates and forces, as fractions rather than absolutes where possible.

    ``seat_force_n`` is absolute because the humanoid has no actuator whose
    limit a fraction could be taken of. It is the smallest load that
    distinguishes a contact from solver noise, not a grip strength.
    """

    #: Keep closing until a digit carries THIS much, not until it merely
    #: touches. Measured directly: two plates squeezing the block's side faces
    #: needed about 42 N at 1 mm of squeeze to carry it, and slipped below that.
    #: Seating at 0.25 N -- the first whisper of contact -- stopped every digit
    #: an order of magnitude short of a grip, which is why the hand closed
    #: correctly and the block stayed on the table.
    seat_force_n: float = 8.0
    #: Fast enough to shut fully inside the closure window. At 1.1/s a 0.55 s
    #: window only reaches a curl of 0.6, so the hand never took the shape the
    #: aperture plan was built against and coverage fell from 0.97 to 0.29. The
    #: rate is a ceiling, not a schedule: a digit still stops the moment it
    #: carries load.
    curl_rate_per_s: float = 3.0
    max_curl: float = 1.0
    opposition_force_n: float = 2.0


#: How much of the closure window the thumb gets to itself before the fingers
#: start, as a fraction.
#:
#: A lead, not a handover. Waiting for the thumb to finish before releasing the
#: fingers sounds like the same thing and is not: the thumb travels its full
#: range now, which takes the whole window, so the fingers were released with
#: nothing left and reached a curl of 0.092. On screen that is a hand whose
#: thumb closes correctly and whose four fingers never close at all -- the
#: previous bug wearing the opposite face.
_THUMB_LEAD_FRACTION = 0.3


#: A digit must travel this far before a load counts as having seated it.
#:
#: Without it, "reports force" and "has grasped something" are the same test,
#: and they are not the same thing. The thumb arrives at the close window
#: already touching the block, so it met the seat force at its opening curl of
#: 0.02 and froze there -- for the whole clip. Measured on the compiled
#: armature, the thumb's three bones were byte-identical from t=0.00 to t=3.38
#: while the index travelled from 3 to 62 degrees. On screen that is a hand that
#: closes four fingers around an object its thumb never reaches.
#:
#: A digit loaded before it has travelled is obstructed, not seated. That is an
#: approach failure and it is reported as one rather than being absorbed into a
#: grip that does not exist.
_MIN_TRAVEL_BEFORE_SEAT = 0.15

#: Opening curl. Named because the seating rule is stated relative to it.
_START_CURL = 0.02


@dataclass
class DigitState:
    curl: float
    seated: bool = False
    seated_at_s: float | None = None
    peak_force_n: float = 0.0


@dataclass
class ClosureResult:
    seated_curls: dict[str, float]
    seated_order: tuple[str, ...]
    opposed: bool
    peak_force_n: dict[str, float]
    lift_m: float
    max_penetration_m: float
    detail: str
    frames: list[ClipFrame] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "force_controlled_closure_v1",
            "seated_curls": {k: round(v, 4) for k, v in self.seated_curls.items()},
            "seated_order": list(self.seated_order),
            "opposed": self.opposed,
            "peak_force_n": {k: round(v, 3) for k, v in self.peak_force_n.items()},
            "lift_m": self.lift_m,
            "max_penetration_m": self.max_penetration_m,
            "detail": self.detail,
        }


#: The same numbers ``primitives.hand_pose`` uses. Duplicated rather than
#: imported because they are literals in that function's body, not constants --
#: the class of unsourced value the general package's grounder exists to remove,
#: and worth noting here rather than quietly copying.
_SEGMENT_GAINS = (0.92, 1.12, 0.82)
_THUMB_MAX_RAD = 0.95
_FINGER_MAX_RAD = 1.25
_OPPOSITION_MAX_RAD = 0.75
_SPLAY_MAX_RAD = 0.30


#: Thumb opposition during the approach, measured rather than chosen.
#:
#: Swept over the whole approach against the block, the thumb's shafts clear it
#: by 0.47 cm at -0.8 and 0.91 cm at +1.0, rising monotonically. That looks like
#: clearance at every value until the capsules are accounted for: the thumb
#: segments are 0.75 cm in radius, so a centre-line 0.69 cm out is already 0.6
#: mm inside the surface. Guessing the retracted direction cost exactly that --
#: the thumb's middle shaft struck the block at 11.7 N and it left at 0.68 m/s.
#:
#: Opposition is not what carries the thumb to the object anyway; curl is. Swept
#: alone, curl moves the tip from 6.57 cm clear to 0.32 cm inside, so the thumb
#: can sit at its clearest opposition throughout and still travel in.
_APPROACH_OPPOSITION = 0.9


def digit_rotations(
    hand: Hand, curls: dict[str, float], opposition: float
) -> dict[str, Quat]:
    """Normalized per-digit curl to rig rotations.

    Mirrors ``primitives.hand_pose`` in full -- curl, splay and thumb opposition
    -- because a closure that starts from a *different* open pose than the one
    the aperture plan cleared is not closing on what the plan measured. Omitting
    splay and pinning opposition at its gripping value put the thumb across the
    palm from the first frame, and it seated at its opening curl carrying 134 N:
    buried in the block before the closure began, while all four fingers shut on
    nothing.
    """
    from .primitives import HAND_SHAPES

    open_shape = HAND_SHAPES[HandShape.OPEN]
    side = 1.0 if hand == Hand.LEFT else -1.0
    out: dict[str, Quat] = {}
    for digit in DIGITS:
        curl = float(np.clip(curls.get(digit, 0.0), 0.0, 1.0))
        stem = _STEM[digit]
        splay = float(open_shape.splay[stem])
        for index, segment in enumerate(_SEGMENTS[digit]):
            maximum = _THUMB_MAX_RAD if digit == "thumb" else _FINGER_MAX_RAD
            splay_angle = splay * _SPLAY_MAX_RAD * side if index == 0 else 0.0
            oppose = 0.0
            if digit == "thumb" and index == 0:
                oppose = opposition * _OPPOSITION_MAX_RAD * side
            out[f"{hand.value}{stem}{segment}"] = quat_euler(
                curl * _SEGMENT_GAINS[index] * maximum, oppose, splay_angle
            )
    return out


def close_until_contact(
    block: SceneObject,
    hand: Hand,
    frames: list[ClipFrame],
    *,
    close_window_s: tuple[float, float],
    support_height_m: float,
    opposition: float = 0.85,
    config: ClosureConfig | None = None,
) -> ClosureResult:
    """Advance each digit until it reports contact, then hold it there.

    ``frames`` supplies the arm and wrist motion; the digits in them are ignored
    and replaced by the closure state, because their curls are exactly the
    open-loop guess this is here to remove.
    """
    config = config or ClosureConfig()
    if not frames:
        raise ValueError("closure needs authored frames to ride on")

    from .primitives import HAND_SHAPES

    # Each digit closes toward its FIST value, not toward a uniform 1.0. The
    # aperture plan scores placements against hand_pose(FIST); a closure that
    # drives every digit to the same maximum arrives at a different hand than
    # the one that was chosen, which is why the plan reported three opposition
    # pairs holding while the fingers measured 0.0 N.
    fist = HAND_SHAPES[HandShape.FIST]
    ceilings = {
        digit: float(np.clip(fist.curls[_STEM[digit]], 0.0, config.max_curl))
        for digit in DIGITS
    }
    # The thumb is the exception, and it has to be. Swept over its own curl and
    # opposition from a fixed wrist, the thumb tip travels from 6.57 cm clear of
    # the block to 0.32 cm inside it -- but the authored FIST curl of 0.72 lands
    # it at about 0.78 cm out, permanently short of touching. Held to that
    # ceiling the thumb cannot oppose anything, whatever the fingers do, and the
    # measured result was exactly that: fingers loaded, thumb at 0.0 N. It stops
    # on contact like every other digit; what changes is that it is now allowed
    # to travel far enough to find one.
    ceilings["thumb"] = float(config.max_curl)
    grip_opposition = float(fist.thumb_opposition)

    segment_pairs = _hand_segment_pairs(hand)
    state = {d: DigitState(curl=_START_CURL) for d in DIGITS}

    from .primitives import HAND_SHAPES

    open_opposition = float(HAND_SHAPES[HandShape.OPEN].thumb_opposition)

    def posed(frame: ClipFrame) -> ClipFrame:
        bones = dict(frame.bones)
        # Opposition arrives with the closure, not before it. The thumb swings
        # across the palm as the hand shuts; starting there is what collided.
        # Tied to the THUMB's travel, not the hand's mean. Opposition is the
        # thumb swinging across the palm; averaging it over five digits meant
        # the thumb's own position depended on how far the fingers had shut,
        # which is backwards when the thumb is the one leading.
        progress = float(
            np.clip(state["thumb"].curl / max(ceilings["thumb"], 1e-6), 0.0, 1.0)
        )
        rotations = digit_rotations(
            hand,
            {d: s.curl for d, s in state.items()},
            _APPROACH_OPPOSITION
            + (grip_opposition - _APPROACH_OPPOSITION) * progress,
        )
        for name, rotation in rotations.items():
            bones[name] = BonePose(rotation=rotation)
        return ClipFrame(time_s=frame.time_s, bones=bones, objects=frame.objects)

    # Size the collision bodies from the open hand, as simulate_embodied_grasp
    # does; the geometry does not change with curl.
    opening = [posed(f) for f in frames]
    snapshots = [_frame_hand_landmarks(f, hand) for f in opening]
    lengths: dict[str, list[float]] = {n: [] for n in segment_pairs}
    for landmarks in snapshots:
        for name, (start, end) in segment_pairs.items():
            digit = name.split("_")[1]
            tip = landmarks[f"{hand.value}{digit.title()}Tip"] if end is None else landmarks[end]
            lengths[name].append(float(np.linalg.norm(tip - landmarks[start])))
    halves = {n: max(0.002, float(np.median(v)) * 0.5 - 0.003) for n, v in lengths.items()}
    across = [
        float(np.linalg.norm(l[f"{hand.value}IndexProximal"] - l[f"{hand.value}LittleProximal"]))
        for l in snapshots
    ]
    along = [
        float(np.linalg.norm(l[f"{hand.value}MiddleProximal"] - l[f"{hand.value}Hand"]))
        for l in snapshots
    ]
    palm_half = np.asarray([
        max(0.025, float(np.median(across)) * 0.55),
        max(0.028, float(np.median(along)) * 0.58),
        0.012,
    ])

    model = mujoco.MjModel.from_xml_string(
        _embodied_xml(block, palm_half, halves, support_height_m)
    )
    data = mujoco.MjData(model)
    block_body = _body_id(model, "block")
    _seed_embodied_bodies(model, data, snapshots[0], hand, segment_pairs)
    mujoco.mj_forward(model, data)

    dt = float(model.opt.timestep)
    duration = frames[-1].time_s
    times = np.asarray([f.time_s for f in frames], dtype=float)
    start_z = float(data.xpos[block_body, 2])
    peak_z = start_z
    max_penetration = 0.0
    fingers_released = False
    obstructed: set[str] = set()
    released_at_s: float | None = None
    order: list[str] = []
    out_frames: list[ClipFrame] = []
    next_frame = 0

    for step in range(int(math.ceil(duration / dt)) + 1):
        now = min(step * dt, duration)
        index = int(np.clip(np.searchsorted(times, now), 0, len(frames) - 1))
        frame = frames[index]

        # The thumb leads; the fingers follow once one of them feels the
        # object. The hand does not close as a unit because the wrist cannot
        # place it to: at FIST the four tip spans are 8.6/7.2/5.9/5.9 cm on a
        # 6 cm block, so any single placement leaves some fingers short and
        # drives others through the far face. Closing them together commanded
        # the ring finger 1.57 cm inside the block, and MuJoCo resolved that
        # interpenetration the only way it can -- it threw the block 10.85 cm
        # across the table before a grip existed.
        #
        # So the wrist places the cage, the thumb travels in to meet it, and the
        # fingers shut on what the thumb has pressed against them. Each digit
        # still stops on its own measured load; nothing is commanded to a pose.
        if close_window_s[0] <= now <= close_window_s[1]:
            if not fingers_released:
                # Release on a finger feeling the object, on the thumb seating,
                # or on the thumb running out of travel -- the last so a thumb
                # that finds nothing cannot freeze the whole hand open.
                touched = any(
                    state[d].peak_force_n >= config.opposition_force_n
                    for d in DIGITS[1:]
                )
                lead_over = now >= close_window_s[0] + _THUMB_LEAD_FRACTION * max(
                    close_window_s[1] - close_window_s[0], 1e-6
                )
                if (
                    touched
                    or state["thumb"].seated
                    or state["thumb"].curl >= ceilings["thumb"] - 1e-6
                    or lead_over
                ):
                    fingers_released = True
                    released_at_s = now
            advancing = DIGITS if fingers_released else ("thumb",)
            for digit in advancing:
                digit_state = state[digit]
                if not digit_state.seated:
                    digit_state.curl = min(
                        ceilings[digit], digit_state.curl + config.curl_rate_per_s * dt
                    )

        current = posed(frame)
        _set_embodied_mocap(
            model, data, _frame_hand_landmarks(current, hand), hand, segment_pairs
        )
        mujoco.mj_step(model, data)

        _names, penetration, details = _contact_snapshot(model, data)
        max_penetration = max(max_penetration, penetration)
        peak_z = max(peak_z, float(data.xpos[block_body, 2]))

        for geom, _position, force, _normal in details:
            digit = _contact_digit(geom)
            if digit is None:
                continue
            digit_state = state[digit]
            digit_state.peak_force_n = max(digit_state.peak_force_n, force)
            # Seat on measured load. Nothing commands a final position.
            if not digit_state.seated and force >= config.seat_force_n:
                if digit_state.curl - _START_CURL < _MIN_TRAVEL_BEFORE_SEAT:
                    # Loaded before it moved: the approach put this digit
                    # against the object, and freezing it here is what produced
                    # a thumb that never closed.
                    obstructed.add(digit)
                else:
                    digit_state.seated = True
                    digit_state.seated_at_s = now
                    order.append(digit)

        while next_frame < len(frames) and frames[next_frame].time_s <= now + dt * 0.5:
            out_frames.append(posed(frames[next_frame]))
            next_frame += 1

    while next_frame < len(frames):
        out_frames.append(posed(frames[next_frame]))
        next_frame += 1

    loaded = [d for d, s in state.items() if s.peak_force_n >= config.opposition_force_n]
    opposed = "thumb" in loaded and any(f in loaded for f in DIGITS[1:])
    seated = [d for d, s in state.items() if s.seated]
    detail = (
        f"{len(seated)} digits seated on contact"
        if seated
        else "no digit ever reported contact force"
    )
    if seated and not opposed:
        detail += "; loaded digits are not in opposition"
    if obstructed:
        detail += (
            f"; {'/'.join(sorted(obstructed))} loaded before travelling "
            "-- the approach put them against the object"
        )

    return ClosureResult(
        seated_curls={d: s.curl for d, s in state.items()},
        seated_order=tuple(order),
        opposed=opposed,
        peak_force_n={d: s.peak_force_n for d, s in state.items()},
        lift_m=peak_z - start_z,
        max_penetration_m=max_penetration,
        detail=detail,
        frames=out_frames,
    )
