"""Backwards-compatible home for the gesture quality metrics.

The implementation moved to :mod:`rigby_poc.analysis.gesture` in PR 02a so the
checks can run against any finished clip without going through the compiler.
This module re-exports it so existing call sites keep working unchanged.
"""

from __future__ import annotations

from .analysis.gesture import (
    QUALITY_REFERENCE,
    _angle,
    _angular_kinematics,
    _arm_self_collision,
    _full_hand_visible,
    _inside_torso,
    _oscillation_summary,
    arm_landmarks,
    evaluate_gesture_structure,
    gesture_structure_checks,
    quality_reference,
    shake_joint_oscillation_metrics,
    swing_twist_angles,
)


__all__ = [
    "QUALITY_REFERENCE",
    "_angle",
    "_angular_kinematics",
    "_arm_self_collision",
    "_full_hand_visible",
    "_inside_torso",
    "_oscillation_summary",
    "arm_landmarks",
    "evaluate_gesture_structure",
    "gesture_structure_checks",
    "quality_reference",
    "shake_joint_oscillation_metrics",
    "swing_twist_angles",
]
