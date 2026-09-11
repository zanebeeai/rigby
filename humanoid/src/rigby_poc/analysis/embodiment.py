"""Does the hand the renderer draws agree with the physics that moved the object?

Every other check in this package reads one artefact: the compiled clip. This
one reads two things that are *supposed* to describe the same event, and asks
whether they do.

The gap is structural rather than numerical. ``physics.simulate_grasp`` runs a
body named ``cartesian_hand`` -- a palm box on three slide joints with five
sliding pads -- and copies the resulting free-body trajectory into the rendered
frames. Nothing constrains that proxy to sit where the humanoid's arm, wrist and
articulated fingers are, because their pose is compiled independently. A proxy
grasp that closes and lifts perfectly is therefore *not* evidence that the
humanoid the viewer watches touched the object at all.

No existing check can see this. ``analysis.contact`` measures fingertip against
fingertip, ``analysis.objects`` measures the object's own trajectory, and the
physics metrics describe the proxy's internal state. All three can be
simultaneously green while the rendered hand is a hand's length away from a
block that is rising through the air.

The measurement here is deliberately the weakest one that still has teeth. It
does not model the mesh, the skin or the contact manifold. It asks only whether
*any* rendered hand landmark is close enough to the object's surface that a
contact could exist, using half the object's own bounding diagonal as the
contact radius -- the largest distance a point on its surface can be from its
centre. A frame that fails this bound cannot be touching under any hand geometry
whatsoever, so the check reports a lower bound on the divergence rather than an
estimate of it.

Shipped report-only, following the ROM layer's precedent in plan 04b: the bound
is measured and published on every clip before anything gates on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..kinematics import rig_kinematics
from ..models import ClipFrame, Hand, MotionProgram
from ..thresholds import value_of
from .contract import PHYSICS, CheckResult, upper_bound_check

#: Wrist, finger bases and fingertips. Palm-adjacent bones are included because
#: a power grasp loads the palm, not only the tips, and a check that watched
#: fingertips alone would call a correct palm grasp a divergence.
_HAND_LANDMARKS = (
    "Hand",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
)

_FINGERTIPS = ("thumb", "index", "middle", "ring", "little")


def _hand_points(frame: ClipFrame, hands: tuple[Hand, ...]) -> np.ndarray:
    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(frame.bones)
    points: list[np.ndarray] = []
    for hand in hands:
        side = hand.value
        for stem in _HAND_LANDMARKS:
            name = f"{side}{stem}"
            if name in positions:
                points.append(positions[name])
        for tip in kinematics.fingertip_positions(frame.bones, side).values():
            points.append(tip)
    return np.asarray(points, dtype=float)


def carried_object_divergence_metrics(
    frames: list[ClipFrame],
    program: MotionProgram,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Measure rendered-hand proximity across the frames an object is carried.

    "Carried" is defined from the clip itself: a frame where the object sits
    above its own resting height by more than the lift epsilon. Those are
    exactly the frames where something must be holding it, because nothing else
    in the scene can.
    """
    object_id = next(
        (item.object_id for item in program.primitives if item.object_id), None
    )
    if object_id is None or not frames:
        return {}
    # Half the object's bounding diagonal, published by the compiler, which is
    # the only layer that holds the scene. It is geometry rather than an
    # authored tolerance, so it is derived rather than stored in
    # ``thresholds.v1.json`` -- the argument ``rom.v1.json`` makes for rest
    # offsets.
    radius = metrics.get("carried_object_contact_radius_m")
    if not isinstance(radius, (int, float)) or radius <= 0.0:
        return {}
    radius = float(radius)
    if any(object_id not in frame.objects for frame in frames):
        return {}

    hands = tuple(dict.fromkeys(program.hands or [program.hand]))
    lift_epsilon = float(value_of("physics.carried_lift_epsilon_m"))
    rest_height = float(
        np.asarray(frames[0].objects[object_id].translation.as_list())[1]
    )

    carried: list[int] = []
    distances: list[float] = []
    for index, frame in enumerate(frames):
        centre = np.asarray(frame.objects[object_id].translation.as_list())
        if centre[1] - rest_height <= lift_epsilon:
            continue
        points = _hand_points(frame, hands)
        if points.size == 0:
            continue
        carried.append(index)
        distances.append(float(np.min(np.linalg.norm(points - centre, axis=1))))

    if not carried:
        return {
            "carried_object_id": object_id,
            "carried_frame_count": 0,
            "carried_object_contact_radius_m": radius,
            "carried_object_untouched_frame_ratio": 0.0,
        }

    untouched = [
        index for index, distance in zip(carried, distances) if distance > radius
    ]
    return {
        "carried_object_id": object_id,
        "carried_frame_count": len(carried),
        "carried_object_contact_radius_m": radius,
        "carried_object_min_hand_distance_m": float(min(distances)),
        "carried_object_max_hand_distance_m": float(max(distances)),
        "carried_object_untouched_frame_count": len(untouched),
        "carried_object_untouched_frame_ratio": len(untouched) / len(carried),
        "carried_object_untouched_frames": tuple(untouched),
    }


def carried_object_divergence_checks(metrics: dict[str, Any]) -> list[CheckResult]:
    """Publish the divergence as one addressable verdict.

    Skipped rather than passed when no frame carries the object: a clip that
    never lifts anything has not demonstrated agreement, and a check that
    reports ``pass`` for an event that did not occur is the failure mode
    ``docs/testing.md`` calls a guard blind to its own case.
    """
    if not metrics or not metrics.get("carried_frame_count"):
        return [
            CheckResult(
                id="physics.contact.render_plausible",
                layer=PHYSICS,
                status="skip",
                measured=0.0,
                detail=(
                    "no frame lifts the object, so no contact claim was made"
                    if metrics
                    else "the clip has no frames, so nothing was measured"
                ),
            )
        ]
    return [
        upper_bound_check(
            "physics.contact.render_plausible",
            PHYSICS,
            float(metrics["carried_object_untouched_frame_ratio"]),
            float(value_of("physics.render_contact_untouched_ratio_max")),
            frames=tuple(metrics.get("carried_object_untouched_frames", ())),
            detail=(
                "fraction of carried frames where every rendered hand landmark "
                "is farther from the object's centre than half its bounding "
                f"diagonal ({metrics['carried_object_contact_radius_m']:.4f} m); "
                "such a frame cannot be a contact under any hand geometry"
            ),
        )
    ]
