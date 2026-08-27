"""Deterministic motion timing, compilation, and task-space refinement."""

from .accents import MotionAccentSpec
from .compiler import compile_motion_program
from .errors import MotionCompilationError, MotionFailureReason
from .refinement import (
    CollisionDistanceObjective,
    HardJointAnchor,
    PinchDistanceObjective,
    SiteOrientationObjective,
    SitePositionObjective,
    refine_joint_window,
)
from .rotations import slerp, squad
from .timing import MonotoneTimeLaw, PhaseRetimer, QuinticSegment, TimingProfile
from .trajectory import CandidateTrajectoryV1, ContactPlateauV1

__all__ = [
    "CandidateTrajectoryV1",
    "CollisionDistanceObjective",
    "ContactPlateauV1",
    "HardJointAnchor",
    "MonotoneTimeLaw",
    "MotionCompilationError",
    "MotionAccentSpec",
    "MotionFailureReason",
    "PhaseRetimer",
    "PinchDistanceObjective",
    "QuinticSegment",
    "SitePositionObjective",
    "SiteOrientationObjective",
    "TimingProfile",
    "compile_motion_program",
    "refine_joint_window",
    "slerp",
    "squad",
]
