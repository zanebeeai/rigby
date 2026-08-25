"""``MutationSpec``: a named, graded, deterministic perturbation of a compiled clip.

Under a no-human-evaluation policy this is the primary source of labelled data for
the whole suite (plan 06 section 1.1), so the contract matters more than the
mechanics.  Four rules, each written because breaking it produces a plausible,
well-formed, monotonic result that measures the harness rather than the check.

**1. A mutated clip does not announce that it is mutated.**  ``apply`` never sets
``success=False``, never attaches a ``Failure``, and never writes its own id into
``metrics``.  ``evals/corruptions.py`` does all three, and
``evals/calibrate_judge.py:224`` then scores ``parsed["accept"] and structural_valid``
-- a conjunction whose deterministic term the corruption was built to force false.
The recorded outcome was therefore not a function of the grader's verdict at all.
Provenance belongs in the driver's record, keyed by ``(case, spec)``.  Plan 06
section 6.5.

**2. Applicability is a property of the pair, not of the spec.**  A spec whose target
bone is static in the chosen case perturbs an authored constant: in
``fullbody-walk-forward`` the right elbow sits at 42.23 degrees in *every* frame,
standard deviation 0.000.  The median corpus case moves 18 of 52 bones and the
quietest moves 4.  :meth:`MutationSpec.applies_to` answers per pair, and a driver
must record a skip rather than scoring an unapplied mutation as undetected.

**3. Anatomical deltas are added in DOF coordinates, never post-multiplied.**
``existing * delta`` is exact only when the bone carries nothing but the target DOF.
Measured by lane ``anatomy``: on ``fullbody-dance`` frame 0, a 30-degree abduction
injection delivered a 43.2-degree change.  See :mod:`evals.mutations.inject`.

**4. A delta is signed against the excursion it is perturbing.**  Injecting +30
degrees into a bone already at -41.8 *reduces* the peak by 30 -- correct arithmetic,
useless measurement, and in a sweep it reads as a detection failure at high severity,
which is the shape of a real finding.

None of rules 2 to 4 is caught by ``test_mutation_severity_monotonic``: each produces
a curve that rises with severity, just wrongly.  They are pinned by instrument tests
instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from rigby_poc.models import ClipResult

from .family import MutationFamily, Tier

#: Severity is a normalised 0..1 coordinate, monotonic within a family, so that
#: families with different natural units share one axis in the report.  The
#: *physical* magnitude lives in ``params``; this is the ordering.
Severity = float


@dataclass(frozen=True)
class Applicability:
    """Whether a spec can be applied to a clip, and what the result would mean.

    A reason is required for the negative case.  A driver that records
    ``skipped: true`` with no reason cannot tell "no detector exists here" from "the
    harness declined", and those are different findings.

    ``static_target`` is the third state, and getting it wrong is subtle.  A spec
    whose target bone never moves in this clip *is* applicable -- the base holds one
    pose and the mutated clip holds a different one, so a grader that can see the
    bone separates them perfectly.  But that is a **capability** result, not a
    detection threshold: there is no real motion for a threshold to sit inside, so
    the severity axis is meaningless and the curve is a step.  Scoring it alongside
    threshold results inflates the sweep; skipping it discards a real capability
    measurement.  So it is applied, and tagged, and reported separately.
    """

    ok: bool
    reason: str = ""
    #: The mutation lands on a bone this clip never moves.  Applicable, but the
    #: result is a capability measurement rather than a point on a threshold curve.
    static_target: bool = False

    def __bool__(self) -> bool:
        return self.ok


APPLICABLE = Applicability(True)


@dataclass(frozen=True)
class MutationSpec:
    """One graded perturbation, and the checks it should trip."""

    id: str
    family: MutationFamily
    #: Check ids this mutation should trip, e.g. ``("anatomy.elbow.off_axis",)``.
    #: Empty for the ``semantic`` family, which has no deterministic oracle -- see
    #: plan 06 section 6.1, and note that such a spec must still declare a
    #: measurable program-level difference via ``asserts``.
    targets: tuple[str, ...]
    #: 0..1, monotonic within a family.
    severity: Severity
    tier: Tier
    params: Mapping[str, Any] = field(default_factory=dict)
    #: Deterministic seed.  Mutations are pure: same seed, byte-identical output.
    seed: int = 0
    #: Applied to a deep copy of the clip.  Set by the family module.
    transform: Callable[[ClipResult, "MutationSpec"], ClipResult] | None = None
    #: Answers whether this spec can meaningfully be applied to a given clip.
    #: The default accepts everything; families that need a moving bone, a named
    #: phase, or an attached object override it.
    guard: Callable[[ClipResult, "MutationSpec"], Applicability] | None = None
    #: For ``semantic`` mutations: a program-level difference that must be
    #: independently observable, so the negative is valid without a deterministic
    #: oracle.  Plan 06 section 6.1.
    asserts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"{self.id}: severity {self.severity} is outside 0..1")
        if self.family is MutationFamily.SEMANTIC and not self.asserts:
            raise ValueError(
                f"{self.id}: a semantic mutation has no deterministic oracle, so it "
                f"must declare a measurable program-level difference in `asserts` "
                f"or it is not a valid negative (plan 06 section 6.1)"
            )
        if self.family is not MutationFamily.SEMANTIC and not self.targets:
            raise ValueError(
                f"{self.id}: a mutation that targets no check cannot be scored; "
                f"declare its targets, or make it a semantic mutation with asserts"
            )

    def applies_to(self, clip: ClipResult) -> Applicability:
        """Whether applying this spec to this clip would measure anything."""
        if not clip.frames:
            return Applicability(False, "the clip has no frames")
        if self.guard is None:
            return APPLICABLE
        return self.guard(clip, self)

    def apply(self, clip: ClipResult) -> ClipResult:
        """Return a mutated deep copy.  Raises if the spec does not apply.

        Raising rather than returning the clip unchanged is deliberate: a silently
        unmutated clip scored as a mutation is a manufactured false negative, and it
        looks exactly like a genuine gap in check coverage.
        """
        verdict = self.applies_to(clip)
        if not verdict:
            raise NotApplicable(f"{self.id}: {verdict.reason}")
        if self.transform is None:
            raise NotApplicable(f"{self.id}: has no transform")
        return self.transform(clip.model_copy(deep=True), self)

    def at(self, severity: Severity, tier: Tier) -> "MutationSpec":
        """A copy of this spec at another point on the same axis."""
        return replace(self, id=f"{self.id}@{severity:g}", severity=severity, tier=tier)


class NotApplicable(RuntimeError):
    """A spec was applied to a clip it cannot meaningfully perturb."""
