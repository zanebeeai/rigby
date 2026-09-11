"""A closed-loop gaze controller driving the neck and head as one chain.

The previous gaze was a single rotation written onto ``head`` once per phase. It
had two problems and the second is the one that mattered. It aimed open-loop, so
whatever the arm and torso did afterwards, the head never corrected. And it used
only the head bone: head flexion is bounded at 25/-30 degrees in ``rom.v1.json``,
and a block on a table below chest height needs more than that, so the aim stalled
about 23 degrees short of the target no matter what was asked for.

This controller fixes both by treating the neck and head as a two-link chain and
solving each frame against the pose that frame actually has. Error is split
between the two joints in proportion to their remaining range, which is what
lets the pair reach targets neither can reach alone.

Every angle it produces is expressed in the same anatomical DOFs the movement
vocabulary uses -- flexion, abduction, twist -- and clamped to the same
``rom.v1.json`` bounds, so a gaze can no more exceed the envelope than a
``look_down`` primitive can. The controller is a way of *choosing* values in that
vocabulary, not a way around it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .analysis.anatomy.frame import DofAngles, all_frames, compose
from .analysis.anatomy.rom import rom_limit
from .kinematics import rig_kinematics
from .models import BonePose, Quat, Vec3

#: The chain, proximal first. Solved in this order so the neck takes its share
#: before the head is asked for the remainder.
GAZE_CHAIN = ("neck", "head")

#: Canonical forward in the head's local frame.
_FORWARD = np.asarray([0.0, 0.0, 1.0], dtype=float)

#: Below this the aim is as good as the rig can hold and further iteration only
#: trades one axis against another.
_SETTLED_DEG = 1.5

#: Bounded because this runs per frame, per candidate, inside a search. Three
#: passes closes almost all of the reachable error; a fourth is measurably not
#: worth its forward-kinematics call.
_MAX_PASSES = 3


@dataclass(frozen=True)
class GazeResult:
    bones: dict[str, Quat]
    error_deg: float
    initial_error_deg: float
    saturated: bool
    passes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "neck_head_gaze_controller_v1",
            "gaze_error_deg": self.error_deg,
            "initial_gaze_error_deg": self.initial_error_deg,
            "gaze_improvement_deg": self.initial_error_deg - self.error_deg,
            "gaze_saturated": self.saturated,
            "passes": self.passes,
        }


def _limits(bone: str) -> dict[str, tuple[float, float]]:
    return {
        dof: rom_limit(bone, dof).typical_deg
        for dof in ("flexion", "abduction", "twist")
    }


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return float(min(max(value, bounds[0]), bounds[1]))


def gaze_error_deg(
    bones: dict[str, BonePose],
    target: Vec3 | np.ndarray,
) -> float:
    """Angle between where the head points and where the target is."""
    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(bones)
    rotation = kinematics.canonical_world_rotation(bones, "head")
    aim = np.asarray(
        target.as_list() if isinstance(target, Vec3) else target, dtype=float
    )
    to_target = aim - positions["head"]
    distance = float(np.linalg.norm(to_target))
    if distance < 1e-6:
        return 0.0
    forward = rotation @ _FORWARD
    cosine = float(np.clip(np.dot(forward, to_target / distance), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def aim_gaze(
    bones: dict[str, BonePose],
    target: Vec3 | np.ndarray,
    *,
    max_passes: int = _MAX_PASSES,
) -> GazeResult:
    """Drive neck and head so the head's forward axis points at ``target``.

    Closed loop: each pass measures the residual against the pose produced by
    the previous one, so the solution accounts for whatever the rest of the body
    is doing rather than assuming a rest posture.
    """
    kinematics = rig_kinematics()
    frames = all_frames()
    aim = np.asarray(
        target.as_list() if isinstance(target, Vec3) else target, dtype=float
    )
    working = dict(bones)
    angles = {
        bone: {"flexion": 0.0, "abduction": 0.0, "twist": 0.0} for bone in GAZE_CHAIN
    }
    initial = gaze_error_deg(working, aim)
    error = initial
    passes = 0

    for _pass in range(max_passes):
        passes += 1
        positions = kinematics.canonical_positions(working)
        rotation = kinematics.canonical_world_rotation(working, "head")
        to_target = aim - positions["head"]
        distance = float(np.linalg.norm(to_target))
        if distance < 1e-6:
            break
        direction = to_target / distance
        # Required correction, in the head's own frame: pitch is elevation,
        # yaw is bearing. Solving here rather than in world keeps the result
        # expressible as the anatomical DOFs the vocabulary uses.
        local = rotation.T @ direction
        yaw = math.degrees(math.atan2(local[0], local[2]))
        pitch = math.degrees(math.atan2(-local[1], math.hypot(local[0], local[2])))

        # Split by remaining range so the pair reaches what neither can alone.
        for dof, demand in (("flexion", pitch), ("abduction", yaw)):
            headroom = {}
            for bone in GAZE_CHAIN:
                low, high = _limits(bone)[dof]
                current = angles[bone][dof]
                headroom[bone] = (high - current) if demand > 0 else (current - low)
            total = sum(headroom.values())
            if total <= 1e-9:
                continue
            for bone in GAZE_CHAIN:
                share = demand * (headroom[bone] / total)
                angles[bone][dof] = _clamp(
                    angles[bone][dof] + share, _limits(bone)[dof]
                )

        for bone in GAZE_CHAIN:
            xyzw = compose(
                DofAngles(
                    flexion_rad=math.radians(angles[bone]["flexion"]),
                    abduction_rad=math.radians(angles[bone]["abduction"]),
                    twist_rad=math.radians(angles[bone]["twist"]),
                ),
                frames[bone],
            )
            working[bone] = BonePose(
                rotation=Quat(
                    x=float(xyzw[0]), y=float(xyzw[1]),
                    z=float(xyzw[2]), w=float(xyzw[3]),
                )
            )
        error = gaze_error_deg(working, aim)
        if error <= _SETTLED_DEG:
            break

    saturated = any(
        abs(angles[b][d] - _limits(b)[d][0]) < 1e-6
        or abs(angles[b][d] - _limits(b)[d][1]) < 1e-6
        for b in GAZE_CHAIN
        for d in ("flexion", "abduction")
    )
    return GazeResult(
        bones={bone: working[bone].rotation for bone in GAZE_CHAIN},
        error_deg=error,
        initial_error_deg=initial,
        saturated=saturated,
        passes=passes,
    )
