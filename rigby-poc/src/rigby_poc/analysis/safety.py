"""Clip-level safety metrics: NaNs, discontinuities, joint limits, root drift.

Moved verbatim from ``compiler._safety_metrics``. Every compile path calls it,
so it is the one check that applies to every intent.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..models import ClipFrame
from .contract import CONTRACT, CheckResult, count_check
from .rig import rig_profile


def safety_metrics(
    frames: list[ClipFrame],
    *,
    allow_root_motion: bool = False,
) -> dict[str, Any]:
    if not frames:
        return {
            "joint_limit_violations": 0,
            "root_drift_m": 0.0,
            "foot_drift_m": 0.0,
            "nan_count": 0,
            "discontinuities": 0,
            "max_frame_rotation_delta_rad": 0.0,
            "quaternion_norm_max_error": 0.0,
            "safety_derivation": "no frames",
        }
    profile = rig_profile()
    all_quats = np.asarray(
        [pose.rotation.as_list() for frame in frames for pose in frame.bones.values()], dtype=float
    )
    all_positions = np.asarray(
        [
            pose.position.as_list()
            for frame in frames
            for pose in frame.bones.values()
            if pose.position is not None
        ],
        dtype=float,
    )
    nan_count = int(np.count_nonzero(~np.isfinite(all_quats)))
    if all_positions.size:
        nan_count += int(np.count_nonzero(~np.isfinite(all_positions)))
    norm_error = float(np.max(np.abs(np.linalg.norm(all_quats, axis=1) - 1.0)))
    max_delta = 0.0
    discontinuities = 0
    for previous, current in zip(frames, frames[1:]):
        frame_max = 0.0
        for key in previous.bones:
            a = np.asarray(previous.bones[key].rotation.as_list())
            b = np.asarray(current.bones[key].rotation.as_list())
            delta = 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))
            frame_max = max(frame_max, delta)
        max_delta = max(max_delta, frame_max)
        discontinuities += int(frame_max > 0.35)
    joint_violations = 0
    for canonical, bounds in profile.get("joint_limits_rad", {}).items():
        for frame in frames:
            quat = frame.bones[canonical].rotation
            angle = 2.0 * math.acos(float(np.clip(abs(quat.w), 0.0, 1.0)))
            if angle > max(abs(bounds[0]), abs(bounds[1])) + 1e-6:
                joint_violations += 1
    fixed_keys = ("hips", "leftFoot", "rightFoot", "leftToes", "rightToes")
    fixed_delta = max(
        2.0 * math.acos(float(np.clip(abs(frame.bones[key].rotation.w), 0.0, 1.0)))
        for frame in frames
        for key in fixed_keys
    )
    hips_positions = np.asarray(
        [
            frame.bones["hips"].position.as_list()
            if frame.bones["hips"].position is not None
            else [0.0, 0.0, 0.0]
            for frame in frames
        ],
        dtype=float,
    )
    root_drift = max(
        (float(np.linalg.norm(position - hips_positions[0])) for position in hips_positions),
        default=0.0,
    )
    if not allow_root_motion and root_drift > 1e-6:
        joint_violations += 1
    return {
        "joint_limit_violations": joint_violations,
        "root_drift_m": root_drift,
        "foot_drift_m": 0.0,
        "fixed_root_foot_max_rotation_delta_rad": fixed_delta,
        "nan_count": nan_count,
        "discontinuities": discontinuities,
        "max_frame_rotation_delta_rad": max_delta,
        "quaternion_norm_max_error": norm_error,
        "safety_derivation": (
            "computed over every frame/local delta quaternion and authored hips translation; "
            + ("root motion is explicitly enabled" if allow_root_motion else "root motion must remain fixed")
        ),
    }


def safety_checks(metrics: dict[str, Any]) -> list[CheckResult]:
    """Report the safety metrics as individually addressable checks."""

    return [
        count_check(
            "contract.clip.non_finite_transforms",
            CONTRACT,
            int(metrics.get("nan_count", 0)),
            scale=1,
            detail="clip contains non-finite transforms",
        ),
        count_check(
            "contract.clip.joint_limit_violations",
            CONTRACT,
            int(metrics.get("joint_limit_violations", 0)),
            detail="clip exceeds a joint limit",
        ),
        count_check(
            "contract.clip.rotational_discontinuities",
            CONTRACT,
            int(metrics.get("discontinuities", 0)),
            detail="clip contains rotational discontinuities",
        ),
    ]
