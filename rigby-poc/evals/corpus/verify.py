"""Compare a case's recorded expectations against a fresh compile.

Kept separate from the CLI so tests assert on structured verdicts rather than on
printed text, and so nothing that verifies can also write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from rigby_poc.models import ClipResult

from .hashing import platform_key
from .loader import CorpusCase, compile_case, observe
from .models import DeterminismClass, ExpectedResult

#: Fields excluded from the diff.  ``environment`` records where a case was blessed,
#: which legitimately differs per machine; ``schema_version`` is format identity.
VOLATILE_FIELDS = frozenset({"environment", "schema_version", "case_id"})


class Verdict(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    #: The case is platform-dependent and this platform has never been blessed.
    #: Callers must report this as a skip, never as a pass or a failure.
    UNBLESSED_PLATFORM = "unblessed_platform"


@dataclass(frozen=True)
class FieldDifference:
    field: str
    expected: object
    actual: object

    def render(self) -> str:
        return f"    {self.field}\n      expected  {self.expected!r}\n      actual    {self.actual!r}"


@dataclass(frozen=True)
class CaseComparison:
    case_id: str
    verdict: Verdict
    differences: list[FieldDifference] = field(default_factory=list)
    recorded: ExpectedResult | None = None
    observed: ExpectedResult | None = None
    platform: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is not Verdict.MISMATCH

    def render(self) -> str:
        if self.verdict is Verdict.UNBLESSED_PLATFORM:
            return (
                f"  ? {self.case_id}: platform-dependent, no hash blessed for "
                f"{self.platform!r} (see plan 03 section 6.1)"
            )
        if self.verdict is Verdict.MATCH:
            return f"  = {self.case_id}"
        lines = [f"  ! {self.case_id}"]
        lines.extend(difference.render() for difference in self.differences)
        return "\n".join(lines)


def diff_expected(
    recorded: ExpectedResult,
    observed: ExpectedResult,
    *,
    platform: str,
) -> tuple[Verdict, list[FieldDifference]]:
    if recorded.determinism_class != observed.determinism_class:
        return Verdict.MISMATCH, [
            FieldDifference(
                "determinism_class",
                recorded.determinism_class.value,
                observed.determinism_class.value,
            )
        ]

    differences: list[FieldDifference] = []
    expected_motion = recorded.resolve_motion_sha256(platform)
    if expected_motion is None:
        return Verdict.UNBLESSED_PLATFORM, []
    actual_motion = observed.resolve_motion_sha256(platform)
    if expected_motion != actual_motion:
        differences.append(
            FieldDifference("motion_sha256", expected_motion, actual_motion)
        )

    for name in recorded.__class__.model_fields:
        if name in VOLATILE_FIELDS or name in {"motion_sha256", "determinism_class"}:
            continue
        expected_value = getattr(recorded, name)
        actual_value = getattr(observed, name)
        if expected_value != actual_value:
            differences.append(FieldDifference(name, expected_value, actual_value))

    return (Verdict.MISMATCH if differences else Verdict.MATCH), differences


def compare_case(case: CorpusCase, clip: ClipResult | None = None) -> CaseComparison:
    clip = compile_case(case) if clip is None else clip
    observed = observe(case.id, clip, case.expected.determinism_class)
    key = platform_key()
    verdict, differences = diff_expected(case.expected, observed, platform=key)
    return CaseComparison(
        case_id=case.id,
        verdict=verdict,
        differences=differences,
        recorded=case.expected,
        observed=observed,
        platform=key,
    )


def rebless(case: CorpusCase, observed: ExpectedResult) -> ExpectedResult:
    """Merge a fresh observation into a case's recorded expectations.

    Platform-dependent cases keep the hashes other platforms contributed; only this
    platform's entry is replaced.  Without that, whoever runs ``bless`` second would
    silently delete the first developer's Windows hash.
    """
    if observed.determinism_class == DeterminismClass.PORTABLE:
        return observed
    merged = dict(case.expected.motion_sha256)
    merged.update(observed.motion_sha256)
    return observed.model_copy(update={"motion_sha256": merged})
