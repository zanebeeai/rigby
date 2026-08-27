from __future__ import annotations

from enum import StrEnum
from typing import Any

from rigby_core.errors import FailureCode, RigbyV2Error


class EvidenceFailureReason(StrEnum):
    INVALID_TRACE = "invalid_trace"
    RENDER_BACKEND_UNAVAILABLE = "render_backend_unavailable"
    VIDEO_BACKEND_UNAVAILABLE = "video_backend_unavailable"
    RENDER_FAILED = "render_failed"


class EvidenceError(RigbyV2Error):
    def __init__(
        self,
        reason: EvidenceFailureReason,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        if reason in {
            EvidenceFailureReason.RENDER_BACKEND_UNAVAILABLE,
            EvidenceFailureReason.VIDEO_BACKEND_UNAVAILABLE,
        }:
            code = FailureCode.UNSUPPORTED
        elif reason is EvidenceFailureReason.INVALID_TRACE:
            code = FailureCode.INVALID_CONTRACT
        else:
            code = FailureCode.INTERNAL_ERROR
        super().__init__(
            code,
            message,
            details={"evidence_reason": reason.value, **(details or {})},
        )
        self.reason = reason
