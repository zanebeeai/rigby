"""Typed on-disk format for the golden corpus.

Every model forbids unknown fields, so a typo in a hand-edited case file is a load
error rather than a silently ignored key.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar

from pydantic import Field, model_validator

from rigby_poc.models import (
    BodyAction,
    Contract,
    HandShape,
    Intent,
    ObjectAction,
    StrikeType,
)

from .gates import StructuralGate
from .hashing import PORTABLE_PLATFORM_KEY

CASE_ID_PATTERN = r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"

MANIFEST_SCHEMA_VERSION = "1.1"
EXPECTED_SCHEMA_VERSION = "1.1"


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
    """Whether a case's motion is bit-identical everywhere, or only per platform.

    **No case is** :attr:`PORTABLE` **any more, and the enum member is kept rather
    than deleted so the distinction stays sayable.**  03a used it to mean "does not
    touch MuJoCo".  Windows CI measured that the compiler is not bit-reproducible
    across instruction sets either -- last-few-ulp differences in libm and FMA
    contraction between arm64 and x86-64 -- so the member now means what it says,
    bit-identical on every platform, and nothing qualifies.  Pinned by
    ``tests/test_corpus_determinism.py``.  See :func:`evals.corpus.hashing.platform_key`.
    """

    PORTABLE = "portable"
    PLATFORM_DEPENDENT = "platform_dependent"


class StoragePolicy(StrEnum):
    """How the corpus stores a case's motion. See plan 03 section 6.3."""

    #: Store the program and recompile at test time.  The recompile is the test.
    #: Unused after 03b -- see :attr:`PROGRAM_AND_CLIP`.
    PROGRAMS_ONLY = "programs_only"
    #: Store the program *and* a gzipped slim clip.  Every case, after 03b.
    #:
    #: 03a resolved section 6.3 as programs-only, on the stated grounds that
    #: "determinism is not in doubt".  Windows CI disproved that premise: the
    #: compiler is not bit-reproducible across instruction sets.  With every case
    #: now platform-keyed, an unblessed platform would otherwise *skip* every case
    #: and verify nothing, so the committed clip is what it compares against
    #: instead -- weaker than a hash, and reported as weaker, but not nothing.
    #: The hash still comes from a recompile, never from the stored clip, so a
    #: clip can never become a second source of truth.
    PROGRAM_AND_CLIP = "program_and_clip"


class BlessEnvironment(Contract):
    """Where a case was last blessed. Informational; never asserted against."""

    platform_key: str
    python_version: str
    compiler_version: str
    blessed_at: str


class ExpectedResult(Contract):
    """The recorded outcome of compiling one case.

    All three digests are platform-keyed maps.  ``metrics_sha256`` and
    ``observables_sha256`` were bare strings in 03a, which asserted cross-platform
    bit-equality of derived floats -- the claim Windows CI disproved most sharply,
    since ``max_angular_jerk_rad_s3`` differs by tens of ulps between arm64 and
    x86-64.
    """

    schema_version: str = EXPECTED_SCHEMA_VERSION
    case_id: str = Field(pattern=CASE_ID_PATTERN)
    determinism_class: DeterminismClass = DeterminismClass.PLATFORM_DEPENDENT
    #: ``True`` when compiling this case enters the MuJoCo solver, which decides
    #: whether the MuJoCo version is part of the platform key.
    solver_used: bool = False
    #: Keyed by :data:`~evals.corpus.hashing.PORTABLE_PLATFORM_KEY` for portable
    #: cases, or by :func:`~evals.corpus.hashing.platform_key` for the rest.
    motion_sha256: dict[str, str] = Field(min_length=1)
    metrics_sha256: dict[str, str] = Field(min_length=1)
    observables_sha256: dict[str, str] = Field(min_length=1)
    success: bool
    structural_valid: bool
    fps: Annotated[int, Field(ge=1)]
    frame_count: Annotated[int, Field(ge=0)]
    duration_s: Annotated[float, Field(ge=0.0)]
    contact_count: Annotated[int, Field(ge=0)]
    environment: BlessEnvironment

    #: The digest fields, so callers iterate rather than repeat the three names.
    DIGESTS: ClassVar[tuple[str, ...]] = (
        "motion_sha256",
        "metrics_sha256",
        "observables_sha256",
    )

    @model_validator(mode="after")
    def hashes_match_determinism_class(self) -> ExpectedResult:
        for name in self.DIGESTS:
            keys = set(getattr(self, name))
            if self.determinism_class == DeterminismClass.PORTABLE:
                if keys != {PORTABLE_PLATFORM_KEY}:
                    raise ValueError(
                        f"a portable case records exactly one "
                        f"{PORTABLE_PLATFORM_KEY!r} {name}"
                    )
            elif PORTABLE_PLATFORM_KEY in keys:
                raise ValueError(
                    f"a platform-dependent case cannot claim a "
                    f"{PORTABLE_PLATFORM_KEY!r} {name}"
                )
        motion_keys = set(self.motion_sha256)
        for name in self.DIGESTS[1:]:
            if set(getattr(self, name)) != motion_keys:
                raise ValueError(
                    f"{name} is blessed for {sorted(getattr(self, name))} but "
                    f"motion_sha256 for {sorted(motion_keys)}; a platform is blessed "
                    f"for all three digests or for none"
                )
        return self

    def resolve(self, name: str, key: str) -> str | None:
        """One digest on the platform identified by ``key``.

        Returns ``None`` when this platform has never been blessed, which callers
        must report as a skip or a clip fallback rather than a failure -- see plan
        03 section 6.1.
        """
        digests: dict[str, str] = getattr(self, name)
        if self.determinism_class == DeterminismClass.PORTABLE:
            return digests[PORTABLE_PLATFORM_KEY]
        return digests.get(key)

    def resolve_motion_sha256(self, key: str) -> str | None:
        return self.resolve("motion_sha256", key)


class CaseEntry(Contract):
    """One row of the manifest."""

    id: str = Field(pattern=CASE_ID_PATTERN)
    family: Family
    intent: Intent
    #: Body actions the case's program contains, empty for non-full-body cases.
    body_actions: list[BodyAction] = Field(default_factory=list)
    #: Object actions the program performs, sequence steps included.  Denormalised
    #: here, like ``body_actions``, so ``rebuild_coverage`` stays a pure function of
    #: the manifest rather than having to reopen every program on disk.
    object_actions: list[ObjectAction] = Field(default_factory=list)
    strike_types: list[StrikeType] = Field(default_factory=list)
    hand_shapes: list[HandShape] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    #: The prompt the program was planned from, for humans reading the corpus.
    #: Never replanned at test time -- the program is the source of truth.
    source_prompt: str = Field(min_length=1, max_length=500)
    #: Seed the program carried before :mod:`evals.corpus.freeze` pinned it.
    #: ``None`` for cases planned offline, which never had a response-id seed.
    source_seed: int | None = None
    expected_structural_valid: bool = True
    #: Gates this case exists to trip, named rather than matched as a substring.
    #: Non-empty only for :attr:`Family.KNOWN_BAD`; see :mod:`evals.corpus.gates`.
    must_fail: list[StructuralGate] = Field(default_factory=list)
    storage: StoragePolicy = StoragePolicy.PROGRAM_AND_CLIP
    notes: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def known_bad_cases_name_the_gate_they_fail(self) -> CaseEntry:
        """A known-bad case that names no gate cannot be asserted on.

        The point of the known-bad family is to catch a check that has stopped
        firing.  A case that only records ``expected_structural_valid: false``
        cannot do that: it stays green when the wrong gate fails.
        """
        if self.family == Family.KNOWN_BAD:
            if self.expected_structural_valid:
                raise ValueError("a known-bad case cannot expect to be structurally valid")
            if not self.must_fail and self.intent != Intent.UNSUPPORTED:
                raise ValueError(
                    f"known-bad case {self.id!r} names no gate in must_fail, so nothing "
                    f"asserts on it; only an Intent.UNSUPPORTED case may omit it, where "
                    f"the planner's refusal is the check"
                )
        elif self.must_fail:
            raise ValueError(
                f"only a known_bad case may declare must_fail; {self.id!r} is {self.family}"
            )
        if len(set(self.must_fail)) != len(self.must_fail):
            raise ValueError("duplicate entries in must_fail")
        return self


class CoverageAxis(Contract):
    """Which enum members the corpus covers, and why the rest are absent.

    A member that appears in neither list is a hole: adding an enum member without
    a case makes ``tests/test_corpus_coverage.py`` fail, which is the point.
    """

    covered: list[str] = Field(default_factory=list)
    deferred: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def members_are_declared_once(self) -> CoverageAxis:
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
    """Declared coverage, one axis per enum the corpus is meant to span.

    Adding an axis here makes a newly added member of that enum a build failure
    rather than an oversight, which is the whole mechanism -- see plan 03 section
    3.4.  ``object_action``, ``strike_type`` and ``hand_shape`` were added in 03b:
    the corpus consumes all three and 03a guarded neither.
    """

    intent: CoverageAxis
    body_action: CoverageAxis
    object_action: CoverageAxis = Field(default_factory=CoverageAxis)
    strike_type: CoverageAxis = Field(default_factory=CoverageAxis)
    hand_shape: CoverageAxis = Field(default_factory=CoverageAxis)

    def axes(self) -> dict[str, CoverageAxis]:
        """Every axis by name, so callers iterate rather than list fields."""
        return {name: getattr(self, name) for name in type(self).model_fields}


class CorpusManifest(Contract):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    #: The compiler the corpus was last blessed against.  A mismatch is not an
    #: error -- the hashes are the assertion -- but it explains one.
    compiler_version: str
    storage_policy: StoragePolicy = StoragePolicy.PROGRAM_AND_CLIP
    #: Empty only while a corpus is being bootstrapped by :mod:`evals.corpus.freeze`.
    cases: list[CaseEntry] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=lambda: Coverage(
        intent=CoverageAxis(), body_action=CoverageAxis()
    ))

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> CorpusManifest:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        if ids != sorted(ids):
            raise ValueError("cases must be listed in id order")
        return self

    def entry(self, case_id: str) -> CaseEntry | None:
        return next((case for case in self.cases if case.id == case_id), None)
