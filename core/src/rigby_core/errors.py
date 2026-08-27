from __future__ import annotations

from enum import StrEnum
from typing import Any


class FailureCode(StrEnum):
    """Stable machine-readable failure vocabulary for the v2 pipeline."""

    INVALID_CONTRACT = "invalid_contract"
    UNSUPPORTED = "unsupported"
    UNSUPPORTED_ASSET = "unsupported_asset"
    INFEASIBLE = "infeasible"
    ARTIFACT_INTEGRITY = "artifact_integrity"
    SIMULATION_FAILED = "simulation_failed"
    DETERMINISTIC_GATE_FAILED = "deterministic_gate_failed"
    ALL_CANDIDATES_REJECTED = "all_candidates_rejected"
    CANCELLED = "cancelled"
    WORKER_LOST = "worker_lost"
    INTERNAL_ERROR = "internal_error"


class RigbyV2Error(RuntimeError):
    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class ArtifactIntegrityError(RigbyV2Error):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(FailureCode.ARTIFACT_INTEGRITY, message, details=details)


class JobNotFoundError(KeyError):
    pass


class JobLeaseError(RuntimeError):
    pass


class InvalidJobTransitionError(RuntimeError):
    pass
