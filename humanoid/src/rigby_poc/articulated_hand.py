"""The hand as a jointed mechanism, driven by torque toward a target.

Everything that has failed in this pipeline traces back to one property of the
old hand: it had no joints. Sixteen rigid bodies were welded to mocap targets and
teleported onto whatever pose the renderer had authored, which means contact
could not influence the hand at all. Commanded to close, a finger closed --
through the object if the authored shape said so. Measured, the ring finger was
driven 1.57 cm inside the block and MuJoCo resolved that interpenetration the
only way it can, by ejecting the block at 0.65 m/s.

No amount of planning fixes that. The aperture search, the seating term, the
palm-clearance test and the force-closure controller were all attempts to choose
a pose so exactly right that a hand which cannot feel would happen to land on the
object without passing through it. That is the wrong question to be answering.

Here the fingers are a real kinematic chain: fifteen hinges with anatomical
limits, driven by position actuators with a bounded force. A finger travels
toward its target until something stops it, and the force the actuator goes on
applying against that obstruction IS the grip. Contact stops being a collision to
resolve and becomes the thing that ends the motion, which is what it is in a
hand.

The wrist is still carried by the arm and still mocap-welded. That is deliberate:
the arm's job is placement, which the renderer already solves, and making it
dynamic would mean modelling shoulder torque to hold a pose against gravity --
a different problem from grasping, and not the one that is failing.

Geometry is measured from the rig at load rather than authored, so the simulated
hand is the same size as the drawn one. That was already true of the old bodies
and is worth keeping true: a physics hand that quietly differs in scale from the
rendered one produces grasps that cannot be reproduced on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .kinematics import rig_kinematics
from .models import Hand

#: Rest-pose chains, proximal first. The thumb carries a metacarpal where the
#: fingers carry a proximal, which is why it is listed rather than derived by a
#: naming rule.
CHAINS = {
    "thumb": ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal"),
    "index": ("IndexProximal", "IndexIntermediate", "IndexDistal"),
    "middle": ("MiddleProximal", "MiddleIntermediate", "MiddleDistal"),
    "ring": ("RingProximal", "RingIntermediate", "RingDistal"),
    "little": ("LittleProximal", "LittleIntermediate", "LittleDistal"),
}

#: Flexion limits per joint, degrees, proximal to distal.
FLEXION_DEG = {
    "thumb": (60.0, 55.0, 80.0),
    "index": (90.0, 100.0, 70.0),
    "middle": (90.0, 100.0, 70.0),
    "ring": (90.0, 100.0, 70.0),
    "little": (90.0, 100.0, 70.0),
}

#: How hard a digit may push. A human finger pad delivers roughly 30 N in a power
#: grip and much less in a pinch, and the thumb delivers more than the others.
#: This is the number that becomes grip force, so it is the one to change if the
#: hand crushes things or drops them.
FORCE_N = {"thumb": 40.0, "index": 30.0, "middle": 30.0, "ring": 25.0, "little": 20.0}

#: Digit capsule radius, matching the old collision proxy so that contact
#: geometry is unchanged and any change in behaviour is the joints, not the shape.
RADIUS_M = 0.0075


@dataclass(frozen=True)
class HandGeometry:
    """Measured lengths and knuckle offsets, in the wrist's frame."""

    lengths: dict[str, tuple[float, ...]]
    origins: dict[str, tuple[float, float, float]]
    palm_half: tuple[float, float, float]


@lru_cache(maxsize=2)
def measure(hand: Hand) -> HandGeometry:
    """Read this rig's hand rather than assuming a shape for it."""
    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions({})
    side = hand.value
    wrist = positions[f"{side}Hand"]
    lengths: dict[str, tuple[float, ...]] = {}
    origins: dict[str, tuple[float, float, float]] = {}
    for finger, bones in CHAINS.items():
        points = [positions[f"{side}{bone}"] for bone in bones]
        tip = positions.get(f"{side}{finger.title()}Tip")
        points.append(tip if tip is not None else points[-1] + (points[-1] - points[-2]))
        lengths[finger] = tuple(
            float(np.linalg.norm(points[index + 1] - points[index]))
            for index in range(len(points) - 1)
        )
        origins[finger] = tuple(float(value) for value in (points[0] - wrist))
    across = np.asarray(origins["index"]) - np.asarray(origins["little"])
    width = float(np.linalg.norm(across))
    depth = float(np.linalg.norm(np.asarray(origins["middle"])))
    return HandGeometry(
        lengths=lengths,
        origins=origins,
        palm_half=(max(0.030, depth * 0.55), max(0.030, width * 0.55), 0.012),
    )


def _finger_xml(hand: Hand, finger: str, geometry: HandGeometry) -> str:
    """One finger as a nested chain of hinges, each with a real limit."""
    side = hand.value
    lengths = geometry.lengths[finger]
    x, y, z = geometry.origins[finger]
    limits = FLEXION_DEG[finger]
    parts = [f'<body name="{side}_{finger}_1" pos="{x:.5f} {y:.5f} {z:.5f}">']
    closing = ""
    for index, limit in enumerate(limits):
        length = lengths[index]
        if index:
            parts.append(
                f'<body name="{side}_{finger}_{index + 1}" '
                f'pos="{lengths[index - 1]:.5f} 0 0">'
            )
            closing += "</body>"
        mass = 0.012 if index == 0 else 0.008
        parts.append(
            f'<joint name="{side}_{finger}_j{index + 1}" type="hinge" '
            f'axis="0 1 0" range="0 {np.radians(limit):.4f}" '
            f'damping="0.02" armature="1e-5"/>'
            f'<geom name="{side}_{finger}_{index + 1}" type="capsule" '
            f'fromto="0 0 0 {length:.5f} 0 0" size="{RADIUS_M:.4f}" '
            f'mass="{mass:.4f}"/>'
        )
    return "".join(parts) + closing + "</body>"


def hand_xml(hand: Hand) -> str:
    """The articulated hand, as a body to place inside a larger model."""
    geometry = measure(hand)
    side = hand.value
    hx, hy, hz = geometry.palm_half
    fingers = "".join(_finger_xml(hand, finger, geometry) for finger in CHAINS)
    return (
        f'<body name="{side}_palm">'
        f'<freejoint name="{side}_palm_free"/>'
        f'<geom name="{side}_palm" type="box" '
        f'size="{hx:.5f} {hy:.5f} {hz:.5f}" mass="0.30"/>'
        f"{fingers}</body>"
    )


def actuator_xml(hand: Hand) -> str:
    """A bounded position servo per joint. The bound is the grip strength.

    ``forcerange`` is what makes this a grasp rather than a command: the target
    may ask for a curl the object refuses to allow, and what the joint delivers
    instead is a measured push against it.
    """
    side = hand.value
    rows: list[str] = []
    for finger, limits in FLEXION_DEG.items():
        force = FORCE_N[finger]
        for index in range(len(limits)):
            joint = f"{side}_{finger}_j{index + 1}"
            rows.append(
                f'<position name="{joint}_act" joint="{joint}" kp="12.0" '
                f'forcerange="-{force:.1f} {force:.1f}" ctrlrange="0 2.0"/>'
            )
    return "".join(rows)


def joint_names(hand: Hand) -> tuple[str, ...]:
    side = hand.value
    return tuple(
        f"{side}_{finger}_j{index + 1}"
        for finger, limits in FLEXION_DEG.items()
        for index in range(len(limits))
    )


def targets_for_curls(hand: Hand, curls: dict[str, float]) -> dict[str, float]:
    """Per-joint target angles in radians, from a per-digit curl in [0, 1].

    The distal joints lead slightly. Driving every joint at the same fraction
    produces a rigid paddle that sweeps an object aside; letting the outer joints
    close first is what makes a hand wrap around one.
    """
    lead = (0.85, 1.0, 1.0)
    out: dict[str, float] = {}
    for finger, limits in FLEXION_DEG.items():
        curl = float(np.clip(curls.get(finger, 0.0), 0.0, 1.0))
        for index, limit in enumerate(limits):
            out[f"{hand.value}_{finger}_j{index + 1}"] = float(
                curl * lead[index] * np.radians(limit)
            )
    return out
