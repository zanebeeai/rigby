from __future__ import annotations

from enum import StrEnum
from typing import Any

from rigby_core.errors import FailureCode, RigbyV2Error


class MotionFailureReason(StrEnum):
    INVALID_PHASE_LAYOUT = "invalid_phase_layout"
    UNRESOLVED_TRACK_OWNERSHIP = "unresolved_track_ownership"
    UNSUPPORTED_TRACK = "unsupported_track"
    UNKNOWN_DOF = "unknown_dof"
    INVALID_MODEL = "invalid_model"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    IK_INFEASIBLE = "ik_infeasible"


class MotionCompilationError(RigbyV2Error):
    """A deterministic, typed motion infeasibility.

    All motion compiler failures map onto the public ``infeasible`` code while
    retaining a stable, motion-specific reason for repair and evaluation.
    """

    def __init__(
        self,
        reason: MotionFailureReason,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_details = {"motion_reason": reason.value, **(details or {})}
        super().__init__(FailureCode.INFEASIBLE, message, details=full_details)
        self.reason = reason
