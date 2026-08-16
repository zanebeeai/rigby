"""Typed on-disk format for the golden corpus.

Every model forbids unknown fields, so a typo in a hand-edited case file is a load
error rather than a silently ignored key.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator
from rigby_poc.models import BodyAction, Contract, Intent

from .hashing import PORTABLE_PLATFORM_KEY

CASE_ID_PATTERN = r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"

MANIFEST_SCHEMA_VERSION = "1.0"
EXPECTED_SCHEMA_VERSION = "1.0"


class Family(StrEnum):
    """Coarse grouping used by the corpus coverage table in plan 03 section 3.4."""

    GESTURE = "gesture"
    STRIKE = "strike"
    COMPOSITE = "composite"
    GRASP = "grasp"
    OBJECT_INTERACTION = "object_interaction"
    FULL_BODY = "full_body"
    SEQUENCE = "sequence"
    KNOWN_BAD = "known_bad"


class DeterminismClass(StrEnum):
    """Whether a case's motion is bit-identical everywhere, or only per platform."""

    PORTABLE = "portable"
    PLATFORM_DEPENDENT = "platform_dependent"


class StoragePolicy(StrEnum):
    """How the corpus stores a case's motion. See plan 03 section 6.3."""

    #: Store the program and recompile at test time.  The recompile is the test.
    PROGRAMS_ONLY = "programs_only"
    #: Store the program *and* a gzipped slim clip.  Reserved for 03b.
    PROGRAM_AND_CLIP = "program_and_clip"


class BlessEnvironment(Contract):
    """Where a case was last blessed. Informational; never asserted against."""

    platform_key: str
    python_version: str
    compiler_version: str
    blessed_at: str


class ExpectedResult(Contract):
    """The recorded outcome of compiling one case."""

    schema_version: str = EXPECTED_SCHEMA_VERSION
    case_id: str = Field(pattern=CASE_ID_PATTERN)
    determinism_class: DeterminismClass = DeterminismClass.PORTABLE
    #: Keyed by :data:`~evals.corpus.hashing.PORTABLE_PLATFORM_KEY` for portable
    #: cases, or by :func:`~evals.corpus.hashing.platform_key` for the rest.
    motion_sha256: dict[str, str] = Field(min_length=1)
    metrics_sha256: str
    observables_sha256: str
    success: bool
    structural_valid: bool
    fps: Annotated[int, Field(ge=1)]
    frame_count: Annotated[int, Field(ge=0)]
    duration_s: Annotated[float, Field(ge=0.0)]
    contact_count: Annotated[int, Field(ge=0)]
    environment: BlessEnvironment

    @model_validator(mode="after")
    def hashes_match_determinism_class(self) -> "ExpectedResult":
        keys = set(self.motion_sha256)
        if self.determinism_class == DeterminismClass.PORTABLE:
            if keys != {PORTABLE_PLATFORM_KEY}:
                raise ValueError(
                    f"a portable case records exactly one {PORTABLE_PLATFORM_KEY!r} hash"
                )
        elif PORTABLE_PLATFORM_KEY in keys:
            raise ValueError(
                f"a platform-dependent case cannot claim a {PORTABLE_PLATFORM_KEY!r} hash"
            )
        return self

    def resolve_motion_sha256(self, key: str) -> str | None:
        """The hash to assert against on the platform identified by ``key``.

        Returns ``None`` when this platform has never been blessed, which callers
        must report as a skip rather than a failure -- see plan 03 section 6.1.
        """
        if self.determinism_class == DeterminismClass.PORTABLE:
            return self.motion_sha256[PORTABLE_PLATFORM_KEY]
        return self.motion_sha256.get(key)


class CaseEntry(Contract):
    """One row of the manifest."""

    id: str = Field(pattern=CASE_ID_PATTERN)
    family: Family
    intent: Intent
    #: Body actions the case's program contains, empty for non-full-body cases.
    body_actions: list[BodyAction] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    #: The prompt the program was planned from, for humans reading the corpus.
    #: Never replanned at test time -- the program is the source of truth.
    source_prompt: str = Field(min_length=1, max_length=500)
    #: Seed the program carried before :mod:`evals.corpus.freeze` pinned it.
    #: ``None`` for cases planned offline, which never had a response-id seed.
    source_seed: int | None = None
    expected_structural_valid: bool = True
    storage: StoragePolicy = StoragePolicy.PROGRAMS_ONLY
    notes: str | None = Field(default=None, max_length=300)


class CoverageAxis(Contract):
    """Which enum members the corpus covers, and why the rest are absent.

    A member that appears in neither list is a hole: adding an enum member without
    a case makes ``tests/test_corpus_coverage.py`` fail, which is the point.
    """

    covered: list[str] = Field(default_factory=list)
    deferred: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def members_are_declared_once(self) -> "CoverageAxis":
        overlap = set(self.covered) & set(self.deferred)
        if overlap:
            raise ValueError(f"declared both covered and deferred: {sorted(overlap)}")
        if len(set(self.covered)) != len(self.covered):
            raise ValueError("duplicate entries in covered")
        empty = sorted(name for name, reason in self.deferred.items() if not reason.strip())
        if empty:
            raise ValueError(f"deferred members need a reason: {empty}")
        return self


class Coverage(Contract):
    intent: CoverageAxis
    body_action: CoverageAxis


class CorpusManifest(Contract):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    #: The compiler the corpus was last blessed against.  A mismatch is not an
    #: error -- the hashes are the assertion -- but it explains one.
    compiler_version: str
    storage_policy: StoragePolicy = StoragePolicy.PROGRAMS_ONLY
    #: Empty only while a corpus is being bootstrapped by :mod:`evals.corpus.freeze`.
    cases: list[CaseEntry] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=lambda: Coverage(
        intent=CoverageAxis(), body_action=CoverageAxis()
    ))

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> "CorpusManifest":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        if ids != sorted(ids):
            raise ValueError("cases must be listed in id order")
        return self

    def entry(self, case_id: str) -> CaseEntry | None:
        return next((case for case in self.cases if case.id == case_id), None)
