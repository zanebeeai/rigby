"""Authoritative MuJoCo trace evidence capture."""

from .errors import EvidenceError, EvidenceFailureReason
from .renderer import (
    EVIDENCE_FPS,
    EvidenceRenderConfig,
    MujocoEvidenceRenderer,
    anonymous_id_for,
    trace_archive_bytes,
)

__all__ = [
    "EVIDENCE_FPS",
    "EvidenceError",
    "EvidenceFailureReason",
    "EvidenceRenderConfig",
    "MujocoEvidenceRenderer",
    "anonymous_id_for",
    "trace_archive_bytes",
]
