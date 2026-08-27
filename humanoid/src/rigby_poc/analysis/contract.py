"""The check contract every analysis check reports through.

A check reports a graded ``severity`` rather than a boolean so downstream
ranking (candidate selection, mutation detection) can order failures without a
human. ``severity`` is 0.0 for a passing or skipped check and rises towards 1.0
as the measurement gets further past its threshold.

``severity`` therefore ranks only the failing half. :class:`CheckResult`
forbids a non-zero severity on anything but a fail, so a score folded from
severities alone cannot order two clips that both pass -- which is exactly what
plan 10 §6's selection regret asks, since it ranks candidates for one prompt and
most candidates pass most checks. ``headroom`` is the signed extension: negative
and equal to ``-severity`` on a fail, positive on a pass and measuring how much
room was left, ``None`` when nothing was measured.

**Direction is known here and nowhere else.** A consumer holding a finished
:class:`CheckResult` cannot recover which side of the bound is bad: ``measured``
may be a dict and ``threshold`` may be ``None``. Measured over the 47-case
corpus, 7,348 of 7,636 verdicts (96.2%) carry a dict ``measured`` or no
threshold, so a downstream fold that infers direction from those two fields sees
3.8% of the check surface. The bound helpers below know the direction because
they are the bound; every other construction site has to say so explicitly, and
``__post_init__`` refuses a result that does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover - the analyzers import this, not the reverse
    from .context import AnalysisContext


CheckStatus = Literal["pass", "fail", "skip"]

CONTRACT = "contract"
ANATOMY = "anatomy"
PHYSICS = "physics"
SIGNAL = "signal"

LAYERS = (CONTRACT, ANATOMY, PHYSICS, SIGNAL)


class Analyzer(Protocol):
    """Compute the metric keys one action contributes to a clip."""

    def __call__(self, ctx: "AnalysisContext") -> dict[str, Any]: ...


@dataclass(frozen=True)
class AnalyzerEntry:
    """One action's registration in the analyzer registry.

    ``analyzer`` is ``None`` while the action's metric block still lives in the
    compiler. ``owner`` says which module currently computes it, and ``plan``
    names the PR that moves it here.

    It lives beside :class:`CheckResult` rather than in ``registry`` so the
    analyzer modules can declare their own entries without importing the
    registry that collects them.
    """

    analyzer: Analyzer | None
    owner: str
    plan: str
    note: str = ""

    @property
    def ported(self) -> bool:
        return self.analyzer is not None


@dataclass(frozen=True)
class CheckResult:
    """One addressable verdict about one clip."""

    id: str
    layer: str
    status: CheckStatus
    measured: float | dict[str, Any]
    threshold: float | None = None
    severity: float = 0.0
    #: Signed room against the bound, in ``[-1, 1]``, or ``None`` when nothing
    #: was measured. Negative failing (and always exactly ``-severity``), zero
    #: at the bound, positive passing. See the module docstring for why this
    #: cannot be derived downstream.
    headroom: float | None = None
    frames: tuple[int, ...] = field(default_factory=tuple)
    detail: str = ""

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"unknown check layer: {self.layer}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be within [0, 1]: {self.severity}")
        if self.status != "fail" and self.severity != 0.0:
            raise ValueError("only a failing check may carry a non-zero severity")
        self._check_headroom()

    def _check_headroom(self) -> None:
        """Tie ``headroom`` to ``status`` so a new check site cannot default it.

        A plain ``headroom: float | None = None`` field would let any future
        construction site omit the direction and still produce a valid-looking
        result, which ``docs/testing.md`` calls an absence becoming a value: the
        composite would fold it as "not measured" and read as though that check
        had abstained. The rules are enforced rather than documented for the
        same reason the ROM layer skips an unmeasured bone instead of passing it.
        """

        if self.status == "skip":
            if self.headroom is not None:
                raise ValueError(
                    "a skipped check measured nothing, so it has no headroom: "
                    f"{self.id} carries {self.headroom}"
                )
            return
        if self.headroom is None:
            raise ValueError(
                f"{self.id} reports status={self.status!r} without a headroom. "
                "Direction is only known where the check is built -- use one of "
                "the bound helpers in this module, or pass headroom explicitly."
            )
        if not -1.0 <= self.headroom <= 1.0:
            raise ValueError(f"headroom must be within [-1, 1]: {self.headroom}")
        if self.status == "fail" and self.headroom != -self.severity:
            raise ValueError(
                f"{self.id} fails with severity {self.severity} but headroom "
                f"{self.headroom}; a failure's headroom is exactly -severity"
            )
        if self.status == "pass" and self.headroom < 0.0:
            raise ValueError(
                f"{self.id} passes but reports negative headroom {self.headroom}"
            )

    @property
    def failed(self) -> bool:
        return self.status == "fail"


def saturating_severity(excess: float, scale: float) -> float:
    """Map a positive threshold overshoot onto (0, 1].

    ``scale`` is the overshoot treated as fully egregious. Anything at or past
    it saturates at 1.0; the mapping is linear below that. A non-positive
    overshoot is not a failure and returns 0.0.
    """

    if excess <= 0.0:
        return 0.0
    if scale <= 0.0:
        return 1.0
    return min(1.0, excess / scale)


def saturating_margin(margin: float, scale: float) -> float:
    """Map a non-negative distance from a bound onto [0, 1].

    The passing-side mirror of :func:`saturating_severity`, and deliberately the
    same ``scale``: a check that treats a 45-degree overshoot as fully egregious
    treats 45 degrees of remaining room as fully comfortable, so the two halves
    of ``headroom`` are one ruler rather than two. A negative margin means the
    check passed only because of its tolerance, and clamps to 0.0 -- at the
    bound, which is where it is.
    """

    if margin <= 0.0:
        return 0.0
    if scale <= 0.0:
        return 1.0
    return min(1.0, margin / scale)


def binary_check(
    check_id: str,
    layer: str,
    *,
    passed: bool,
    measured: float | dict[str, Any],
    threshold: float | None = None,
    severity: float = 1.0,
    frames: tuple[int, ...] = (),
    detail: str = "",
) -> CheckResult:
    """A check whose subject is categorical, so its headroom has no middle.

    Ordering, sequence and exchange assertions are satisfied or they are not;
    there is no measurement that is "nearly in the right order". The passing
    headroom is therefore 1.0 rather than a fabricated fraction, and that is a
    claim worth making explicitly at the call site rather than by writing the
    literal there -- a bare ``headroom=1.0`` next to a raw ``CheckResult`` reads
    like a placeholder, and the next reader cannot tell a categorical check from
    a continuous one somebody did not finish.
    """

    return CheckResult(
        id=check_id,
        layer=layer,
        status="pass" if passed else "fail",
        measured=measured,
        threshold=threshold,
        severity=0.0 if passed else severity,
        headroom=1.0 if passed else -severity,
        frames=frames,
        detail=detail,
    )


def upper_bound_check(
    check_id: str,
    layer: str,
    measured: float,
    threshold: float,
    *,
    tolerance: float = 0.0,
    scale: float | None = None,
    frames: tuple[int, ...] = (),
    detail: str = "",
) -> CheckResult:
    """A check that fails when ``measured`` rises above ``threshold``."""

    failed = measured > threshold + tolerance
    span = scale if scale is not None else max(abs(threshold), 1e-9)
    severity = saturating_severity(measured - threshold, span) if failed else 0.0
    return CheckResult(
        id=check_id,
        layer=layer,
        status="fail" if failed else "pass",
        measured=measured,
        threshold=threshold,
        severity=severity,
        headroom=(
            -severity if failed else saturating_margin(threshold - measured, span)
        ),
        frames=frames,
        detail=detail,
    )


def lower_bound_check(
    check_id: str,
    layer: str,
    measured: float,
    threshold: float,
    *,
    tolerance: float = 0.0,
    scale: float | None = None,
    frames: tuple[int, ...] = (),
    detail: str = "",
) -> CheckResult:
    """A check that fails when ``measured`` falls below ``threshold``."""

    failed = measured < threshold - tolerance
    span = scale if scale is not None else max(abs(threshold), 1e-9)
    severity = saturating_severity(threshold - measured, span) if failed else 0.0
    return CheckResult(
        id=check_id,
        layer=layer,
        status="fail" if failed else "pass",
        measured=measured,
        threshold=threshold,
        severity=severity,
        headroom=(
            -severity if failed else saturating_margin(measured - threshold, span)
        ),
        frames=frames,
        detail=detail,
    )


def count_check(
    check_id: str,
    layer: str,
    count: int,
    *,
    scale: int = 8,
    frames: tuple[int, ...] = (),
    detail: str = "",
) -> CheckResult:
    """A check that fails as soon as a violation count is non-zero."""

    severity = saturating_severity(float(count), float(scale)) if count else 0.0
    return CheckResult(
        id=check_id,
        layer=layer,
        status="fail" if count else "pass",
        measured=float(count),
        threshold=0.0,
        # Zero violations is the whole of the passing side -- there is no
        # "further below zero" -- so a clean count check is categorically clean.
        severity=severity,
        headroom=1.0 if not count else -severity,
        frames=frames,
        detail=detail,
    )


def skipped(check_id: str, layer: str, detail: str = "") -> CheckResult:
    return CheckResult(
        id=check_id,
        layer=layer,
        status="skip",
        measured=0.0,
        threshold=None,
        severity=0.0,
        detail=detail,
    )
