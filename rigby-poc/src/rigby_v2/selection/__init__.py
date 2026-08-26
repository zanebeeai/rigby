"""Five-way proposal generation and anonymous selection orchestration."""

from .generation import (
    CandidateCompiler,
    CandidateGenerationError,
    CandidateGenerationReason,
    DeterministicCandidateCompiler,
    generate_candidate_set,
    structural_fingerprint,
    variation_for_attempt,
)
from .orchestrator import (
    AnonymousEvidenceView,
    AnonymousJudge,
    BestOfFiveOrchestrator,
    EvidenceRepairer,
)
from .vlm_judge import (
    OpenAIAnonymousJudge,
    TimelineSheet,
    VlmJudgeError,
    build_timeline_sheet,
)

__all__ = [
    "AnonymousEvidenceView",
    "AnonymousJudge",
    "BestOfFiveOrchestrator",
    "CandidateCompiler",
    "CandidateGenerationError",
    "CandidateGenerationReason",
    "DeterministicCandidateCompiler",
    "EvidenceRepairer",
    "OpenAIAnonymousJudge",
    "TimelineSheet",
    "VlmJudgeError",
    "build_timeline_sheet",
    "generate_candidate_set",
    "structural_fingerprint",
    "variation_for_attempt",
]
