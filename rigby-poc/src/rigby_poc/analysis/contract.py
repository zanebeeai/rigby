"""The check contract every analysis check reports through.

A check reports a graded ``severity`` rather than a boolean so downstream
ranking (candidate selection, mutation detection) can order failures without a
human. ``severity`` is 0.0 for a passing or skipped check and rises towards 1.0
as the measurement gets further past its threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


CheckStatus = Literal["pass", "fail", "skip"]

CONTRACT = "contract"
ANATOMY = "anatomy"
PHYSICS = "physics"
SIGNAL = "signal"

LAYERS = (CONTRACT, ANATOMY, PHYSICS, SIGNAL)


@dataclass(frozen=True)
class CheckResult:
    """One addressable verdict about one clip."""

    id: str
    layer: str
    status: CheckStatus
    measured: float | dict[str, Any]
    threshold: float | None = None
    severity: float = 0.0
    frames: tuple[int, ...] = field(default_factory=tuple)
    detail: str = ""

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"unknown check layer: {self.layer}")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be within [0, 1]: {self.severity}")
        if self.status != "fail" and self.severity != 0.0:
            raise ValueError("only a failing check may carry a non-zero severity")

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
    return CheckResult(
        id=check_id,
        layer=layer,
        status="fail" if failed else "pass",
        measured=measured,
        threshold=threshold,
        severity=(
            saturating_severity(
                measured - threshold,
                scale if scale is not None else max(abs(threshold), 1e-9),
            )
            if failed
            else 0.0
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
    return CheckResult(
        id=check_id,
        layer=layer,
        status="fail" if failed else "pass",
        measured=measured,
        threshold=threshold,
        severity=(
            saturating_severity(
                threshold - measured,
                scale if scale is not None else max(abs(threshold), 1e-9),
            )
            if failed
            else 0.0
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

    return CheckResult(
        id=check_id,
        layer=layer,
        status="fail" if count else "pass",
        measured=float(count),
        threshold=0.0,
        severity=saturating_severity(float(count), float(scale)) if count else 0.0,
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
