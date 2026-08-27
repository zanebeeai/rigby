"""How fast a joint may move, enforced on the compiled clip.

The range-of-motion envelope bounds where a bone may be. Nothing bounded how
fast it could get there, and the two are not the same constraint: a clip can
step a bone across its entire legal range in a single frame and every pose along
the way is inside the envelope. That is what a jitter is -- a sequence of legal
poses in an illegal order.

It had a measured cost. The hand crossed from preshape to contact fast enough to
strike the block and knock it 7 cm across the table before a single finger had
closed, and because the strike happened between authored keyframes there was
nothing to inspect: no pose in the clip was wrong.

The limiter is a first-order clamp on the per-frame angle change, taken about
the shortest arc between consecutive rotations, with the ceiling read from
``rom.v1.json``'s ``angular_rate_limits`` by the bone's joint class. A bone that
cannot reach its authored target in time lags toward it rather than snapping,
and the lag is reported rather than hidden -- ``unreached_bones`` names anything
still short of its target when the clip ends, so a motion that was quietly
truncated is visible instead of merely slower.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .models import BonePose, ClipFrame, Quat

_CONFIG = Path(__file__).resolve().parents[2] / "config"

#: Below this a residual is a rounding artefact, not an unreached target.
_SETTLED_DEG = 0.5


@dataclass(frozen=True)
class RateLimitReport:
    clamped_frames: int
    peak_before_deg_per_s: float
    peak_after_deg_per_s: float
    unreached_bones: tuple[str, ...]
    worst_residual_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": "angular_rate_limit_v1",
            "clamped_frames": self.clamped_frames,
            "peak_before_deg_per_s": round(self.peak_before_deg_per_s, 2),
            "peak_after_deg_per_s": round(self.peak_after_deg_per_s, 2),
            "unreached_bones": list(self.unreached_bones),
            "worst_residual_deg": round(self.worst_residual_deg, 3),
        }


@lru_cache(maxsize=1)
def rate_limits() -> dict[str, float]:
    """Per-joint-class ceilings, keyed by class name, plus ``__default__``."""
    rom = json.loads((_CONFIG / "rom.v1.json").read_text(encoding="utf-8"))
    section = rom.get("angular_rate_limits")
    if not section:
        raise ValueError("rom.v1.json declares no angular_rate_limits")
    limits = {str(k): float(v) for k, v in section["classes"].items()}
    limits["__default__"] = float(section.get("default_deg_per_s", 180.0))
    return limits


@lru_cache(maxsize=1)
def _bone_classes() -> dict[str, str]:
    """Bone name to joint class, from the body-part manifest."""
    parts = json.loads(
        (_CONFIG / "body_parts.v1.json").read_text(encoding="utf-8")
    )["parts"]
    out: dict[str, str] = {}
    for part in parts.values():
        for bone in part["bones"]:
            out[bone] = part["joint_class"]
    return out


def limit_for(bone: str) -> float:
    limits = rate_limits()
    return limits.get(_bone_classes().get(bone, ""), limits["__default__"])


def _angle_between(first: Quat, second: Quat) -> float:
    a = Rotation.from_quat([first.x, first.y, first.z, first.w])
    b = Rotation.from_quat([second.x, second.y, second.z, second.w])
    return float(np.degrees((b * a.inv()).magnitude()))


def _partial(first: Quat, second: Quat, fraction: float) -> Quat:
    """``fraction`` of the way along the shortest arc from ``first``."""
    rotations = Rotation.from_quat(
        [[first.x, first.y, first.z, first.w], [second.x, second.y, second.z, second.w]]
    )
    xyzw = Slerp([0.0, 1.0], rotations)([float(np.clip(fraction, 0.0, 1.0))]).as_quat()[0]
    return Quat(x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3]))


def limit_angular_rate(
    frames: list[ClipFrame],
) -> tuple[list[ClipFrame], RateLimitReport]:
    """Clamp every bone's per-frame rotation to its joint class's ceiling.

    Object transforms are untouched: they are the simulation's account of where
    things went, not an authored pose, and rate-limiting them would be editing
    the result rather than the command.
    """
    if len(frames) < 2:
        return frames, RateLimitReport(0, 0.0, 0.0, (), 0.0)

    out = [frames[0]]
    previous = dict(frames[0].bones)
    clamped_frames = 0
    peak_before = 0.0
    peak_after = 0.0

    for index in range(1, len(frames)):
        frame = frames[index]
        dt = max(frame.time_s - frames[index - 1].time_s, 1e-6)
        bones: dict[str, BonePose] = {}
        clamped_here = False
        for name, pose in frame.bones.items():
            prior = previous.get(name)
            if prior is None:
                bones[name] = pose
                continue
            delta = _angle_between(prior.rotation, pose.rotation)
            peak_before = max(peak_before, delta / dt)
            ceiling = limit_for(name) * dt
            if delta > ceiling and delta > 1e-9:
                bones[name] = BonePose(
                    rotation=_partial(prior.rotation, pose.rotation, ceiling / delta)
                )
                clamped_here = True
                peak_after = max(peak_after, ceiling / dt)
            else:
                bones[name] = pose
                peak_after = max(peak_after, delta / dt)
        clamped_frames += int(clamped_here)
        previous = bones
        out.append(
            ClipFrame(time_s=frame.time_s, bones=bones, objects=frame.objects)
        )

    # What the clip asked for last, against what the limiter actually delivered.
    unreached: list[tuple[str, float]] = []
    for name, pose in frames[-1].bones.items():
        final = out[-1].bones.get(name)
        if final is None:
            continue
        residual = _angle_between(final.rotation, pose.rotation)
        if residual > _SETTLED_DEG:
            unreached.append((name, residual))
    unreached.sort(key=lambda item: -item[1])

    return out, RateLimitReport(
        clamped_frames=clamped_frames,
        peak_before_deg_per_s=peak_before,
        peak_after_deg_per_s=peak_after,
        unreached_bones=tuple(name for name, _r in unreached),
        worst_residual_deg=unreached[0][1] if unreached else 0.0,
    )
