"""Compare a case's recorded expectations against a fresh compile.

Kept separate from the CLI so tests assert on structured verdicts rather than on
printed text, and so nothing that verifies can also write.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from rigby_poc.models import ClipResult

from .hashing import platform_key
from .loader import (
    CorpusCase,
    compile_case,
    observe,
    read_slim_clip,
    slim_clip_payload,
)
from .models import DeterminismClass, ExpectedResult

#: Absolute tolerance for the per-platform clip fallback, on quaternion components
#: (dimensionless, |q| = 1) and on positions (metres).
#:
#: **PROVISIONAL.**  It cannot be derived on one machine: the quantity it bounds is
#: MuJoCo's cross-platform solver drift, and every value measured here is drift
#: within one platform, which is exactly zero.  1e-3 sits three orders above the
#: 1e-6 the slim clip is rounded to and well below any motion difference a viewer
#: could see -- a 1e-3 quaternion component is under 0.12 degrees of rotation.  The
#: comparison reports its observed maximum deviation either way, so the developer
#: who first runs a MuJoCo case on a second platform learns the real number and can
#: replace this with a measured one.  See plan 03 section 6.1.
SLIM_CLIP_TOLERANCE = 1e-3

#: Fields excluded from the diff.  ``environment`` records where a case was blessed,
#: which legitimately differs per machine; ``schema_version`` is format identity.
VOLATILE_FIELDS = frozenset({"environment", "schema_version", "case_id"})


class Verdict(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    #: The case is platform-dependent, this platform has never been blessed, and
    #: the case stores no clip to fall back to.  Callers report this as a skip,
    #: never as a pass or a failure.
    UNBLESSED_PLATFORM = "unblessed_platform"
    #: The case is platform-dependent and unblessed here, but its committed slim
    #: clip matches within :data:`SLIM_CLIP_TOLERANCE`.  Weaker than ``MATCH`` and
    #: reported as such -- but not nothing, which is what a bare skip would be.
    TOLERANCE_MATCH = "tolerance_match"


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
    tolerance: "ToleranceReport | None" = None

    @property
    def ok(self) -> bool:
        return self.verdict is not Verdict.MISMATCH

    def render(self) -> str:
        if self.verdict is Verdict.UNBLESSED_PLATFORM:
            return (
                f"  ? {self.case_id}: platform-dependent, no hash blessed for "
                f"{self.platform!r} (see plan 03 section 6.1)"
            )
        if self.verdict is Verdict.TOLERANCE_MATCH:
            assert self.tolerance is not None
            return (
                f"  ~ {self.case_id}: no hash blessed for {self.platform!r}; slim clip "
                f"matches within {self.tolerance.tolerance} "
                f"(max deviation {self.tolerance.max_deviation:.3e} at "
                f"{self.tolerance.at or '<none>'})"
            )
        if self.verdict is Verdict.MATCH:
            return f"  = {self.case_id}"
        if self.tolerance is not None:
            return (
                f"  ! {self.case_id}: no hash blessed for {self.platform!r} and the "
                f"committed slim clip drifted by {self.tolerance.max_deviation:.3e} "
                f"at {self.tolerance.at}, over the PROVISIONAL "
                f"{self.tolerance.tolerance} tolerance. Either MuJoCo differs more "
                f"across platforms than assumed -- in which case this number is the "
                f"first measurement of it and the tolerance should be replaced with "
                f"it -- or the motion genuinely changed. Bless this platform to "
                f"decide: python -m evals.corpus bless --case {self.case_id} --write"
            )
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
    if recorded.resolve("motion_sha256", platform) is None:
        return Verdict.UNBLESSED_PLATFORM, []
    for name in recorded.DIGESTS:
        expected_digest = recorded.resolve(name, platform)
        actual_digest = observed.resolve(name, platform)
        if expected_digest != actual_digest:
            differences.append(FieldDifference(name, expected_digest, actual_digest))

    for name in recorded.__class__.model_fields:
        if (
            name in VOLATILE_FIELDS
            or name in recorded.DIGESTS
            or name == "determinism_class"
        ):
            continue
        expected_value = getattr(recorded, name)
        actual_value = getattr(observed, name)
        if expected_value != actual_value:
            differences.append(FieldDifference(name, expected_value, actual_value))

    return (Verdict.MISMATCH if differences else Verdict.MATCH), differences


def _walk_numbers(value: object, other: object, path: str = "") -> Iterator[tuple[str, float, float]]:
    """Yield every aligned numeric leaf of two JSON-shaped structures.

    A structural mismatch -- a different key set, a different length, a number
    against a null -- raises, because that is a motion *shape* change and no
    tolerance should absorb it.
    """
    if isinstance(value, dict) and isinstance(other, dict):
        if set(value) != set(other):
            raise SlimClipShapeError(f"{path or '<root>'}: key sets differ")
        for key in sorted(value):
            yield from _walk_numbers(value[key], other[key], f"{path}.{key}")
    elif isinstance(value, list) and isinstance(other, list):
        if len(value) != len(other):
            raise SlimClipShapeError(f"{path or '<root>'}: {len(value)} vs {len(other)}")
        for index, (left, right) in enumerate(zip(value, other)):
            yield from _walk_numbers(left, right, f"{path}[{index}]")
    elif isinstance(value, bool) or isinstance(other, bool):
        if value != other:
            raise SlimClipShapeError(f"{path}: {value!r} vs {other!r}")
    elif isinstance(value, (int, float)) and isinstance(other, (int, float)):
        yield path, float(value), float(other)
    elif value != other:
        raise SlimClipShapeError(f"{path}: {value!r} vs {other!r}")


class SlimClipShapeError(ValueError):
    """Two clips differ in structure, not just in value."""


@dataclass(frozen=True)
class ToleranceReport:
    """How far a fresh compile drifted from the committed slim clip."""

    max_deviation: float
    at: str
    tolerance: float
    leaves: int

    @property
    def within(self) -> bool:
        return self.max_deviation <= self.tolerance


def compare_slim_clip(
    case: CorpusCase, clip: ClipResult, tolerance: float = SLIM_CLIP_TOLERANCE
) -> ToleranceReport:
    """Compare a fresh compile against the case's committed slim clip.

    Only used where a hash comparison is impossible: a platform-dependent case on a
    platform nobody has blessed.  It is a weaker check than the hash and is always
    reported as such.
    """
    committed = read_slim_clip(case.slim_clip_path)
    fresh = slim_clip_payload(clip)
    worst, where, count = 0.0, "", 0
    for path, left, right in _walk_numbers(committed, fresh):
        count += 1
        deviation = abs(left - right)
        if deviation > worst:
            worst, where = deviation, path
    return ToleranceReport(
        max_deviation=worst, at=where, tolerance=tolerance, leaves=count
    )


def compare_case(case: CorpusCase, clip: ClipResult | None = None) -> CaseComparison:
    clip = compile_case(case) if clip is None else clip
    observed = observe(
        case.id,
        clip,
        case.expected.determinism_class,
        solver_used=case.expected.solver_used,
    )
    key = platform_key(case.expected.solver_used)
    verdict, differences = diff_expected(case.expected, observed, platform=key)
    tolerance: ToleranceReport | None = None
    if verdict is Verdict.UNBLESSED_PLATFORM and case.slim_clip_path.is_file():
        # No hash to assert against here, but the committed clip still says whether
        # this platform produced the same motion to within solver drift.  Skipping
        # instead would give this platform zero coverage of the whole MuJoCo path.
        tolerance = compare_slim_clip(case, clip)
        verdict = Verdict.TOLERANCE_MATCH if tolerance.within else Verdict.MISMATCH
        if not tolerance.within:
            differences = [
                FieldDifference(
                    f"slim clip at {tolerance.at}",
                    f"<= {tolerance.tolerance}",
                    tolerance.max_deviation,
                )
            ]
    return CaseComparison(
        case_id=case.id,
        verdict=verdict,
        differences=differences,
        recorded=case.expected,
        observed=observed,
        platform=key,
        tolerance=tolerance,
    )


def rebless(case: CorpusCase, observed: ExpectedResult) -> ExpectedResult:
    """Merge a fresh observation into a case's recorded expectations.

    Platform-dependent cases keep the hashes other platforms contributed; only this
    platform's entry is replaced.  Without that, whoever runs ``bless`` second would
    silently delete the first developer's Windows hash.
    """
    if observed.determinism_class == DeterminismClass.PORTABLE:
        return observed
    merged: dict[str, dict[str, str]] = {}
    for name in observed.DIGESTS:
        digests = dict(getattr(case.expected, name))
        digests.update(getattr(observed, name))
        merged[name] = digests
    return observed.model_copy(update=merged)
