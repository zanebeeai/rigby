"""Per-DOF range-of-motion measurement over a finished clip.

Plan 04 §3.5 and §3.6. Every check lands here in **report-only** mode: it
computes and records its measurement, and reports ``status="pass"`` regardless.
Enforcement is 04c, turned on per DOF after the distribution review, because
switching 44 bones' worth of new limits on at once would reject a large
fraction of currently-shipping motion with no way to separate a genuine
anatomical violation from a mis-derived axis.

A violation is a record, not a boolean. ``integral_deg_s`` is what a gate
should read and ``peak_deg`` is what a human should triage on: a 2-degree
single-frame blip and a 40-degree sustained excursion are not the same event,
and the check this replaces cannot tell them apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...models import ClipFrame
from ..contract import ANATOMY, CheckResult
from ..rig import PROJECT_ROOT
from .frame import all_frames, decompose_series
from .neutral import rest_relative

ROM_FILE = PROJECT_ROOT / "config" / "rom.v1.json"

DOFS = ("flexion", "abduction", "twist")
Band = Literal["within_typical", "beyond_typical", "beyond_max"]

#: Source kinds and the fields each one must carry. Mirrors the schema lane
#: `infra` shipped for ``thresholds.v1.json`` so the repo has one convention
#: rather than two.
REQUIRED_SOURCE_FIELDS: dict[str, frozenset[str]] = {
    "measured": frozenset({"n", "derived_from", "date", "reference_frame"}),
    "invariant": frozenset({"rationale"}),
    "external": frozenset({"cites"}),
    "provisional": frozenset({"rationale"}),
}


class RomError(LookupError):
    """A range-of-motion limit was requested that the config does not define."""


@dataclass(frozen=True)
class DofLimit:
    """One ``(bone, dof)`` entry, in anatomical degrees."""

    bone: str
    dof: str
    typical_deg: tuple[float, float]
    max_deg: tuple[float, float]
    hard_assert: bool
    rest_offset_deg: float | None
    source: dict[str, Any]

    @property
    def has_neutral(self) -> bool:
        return self.rest_offset_deg is not None

    def to_rest_relative(self, anatomical_deg: float) -> float:
        """An anatomical bound expressed in the rig's rest-relative frame.

        The offset is defined as the delta taking **rest to neutral**, so a
        rest-relative angle ``r`` has anatomical value ``a = r - offset`` and the
        inverse used here is ``r = a + offset``. Getting this backwards puts
        every bound on the wrong side of the rest pose by twice the offset,
        which for the shoulder is 179 degrees; the first smoke test of this
        module reported a knee sitting at its own rest pose as beyond_max.

        For a bone with no anatomical neutral the two frames coincide by fiat,
        because the authored bound was written rest-relatively in the first
        place -- which is why those entries are ``provisional``.
        """

        return rest_relative(anatomical_deg, self.rest_offset_deg)

    def band_of(self, measured_deg: float) -> Band:
        low, high = (self.to_rest_relative(v) for v in self.max_deg)
        if measured_deg < low or measured_deg > high:
            return "beyond_max"
        low, high = (self.to_rest_relative(v) for v in self.typical_deg)
        if measured_deg < low or measured_deg > high:
            return "beyond_typical"
        return "within_typical"

    def excess_deg(self, measured_deg: float, band: Band) -> float:
        """How far past the band's own bound a measurement sits. 0 inside."""

        bounds = self.max_deg if band == "beyond_max" else self.typical_deg
        low, high = (self.to_rest_relative(v) for v in bounds)
        if measured_deg > high:
            return measured_deg - high
        if measured_deg < low:
            return low - measured_deg
        return 0.0


@dataclass(frozen=True)
class RomViolation:
    """Plan §3.5's record. Reported whether or not the DOF is enforced."""

    bone: str
    dof: str
    peak_deg: float
    frames: int
    integral_deg_s: float
    band: Band
    threshold_deg: float
    hard_assert: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def _document() -> dict[str, Any]:
    return json.loads(Path(ROM_FILE).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def rom_limits() -> dict[tuple[str, str], DofLimit]:
    """Every ``(bone, dof)`` limit, keyed by the pair."""

    limits: dict[tuple[str, str], DofLimit] = {}
    for bone, entry in _document()["limits"].items():
        for dof, value in entry["dofs"].items():
            limits[(bone, dof)] = DofLimit(
                bone=bone,
                dof=dof,
                typical_deg=tuple(value["typical_deg"]),
                max_deg=tuple(value["max_deg"]),
                hard_assert=bool(value["hard_assert"]),
                rest_offset_deg=value["rest_offset_deg"],
                source=value["source"],
            )
    return limits


def rom_limit(bone: str, dof: str) -> DofLimit:
    try:
        return rom_limits()[(bone, dof)]
    except KeyError as error:
        raise RomError(
            f"no range-of-motion limit for ({bone!r}, {dof!r}); add it to "
            f"{Path(ROM_FILE).name} rather than hardcoding a bound"
        ) from error


def enforceability(bone: str) -> str:
    """``"generation"`` or ``"mutation_only"``.

    ``mutation_only`` marks a bone ``compiler.py`` never assigns a rotation to on
    any path, so its limit cannot fire from generated motion however the corpus
    grows. Recorded so those entries cannot be mistaken for ordinary limits that
    happen never to have fired.
    """

    return _document()["limits"][bone]["enforceable"]


def decompose_clip(frames: Iterable[ClipFrame]) -> dict[str, np.ndarray]:
    """Resolve every bone onto its anatomical frame, as ``(n, 3)`` degree arrays.

    Columns are flexion, abduction, twist. Batched per bone rather than per
    frame: see :func:`~rigby_poc.analysis.anatomy.frame.decompose_series`.
    """

    derived = all_frames()
    collected = list(frames)
    if not collected:
        return {}
    stacked: dict[str, np.ndarray] = {}
    for bone, frame in derived.items():
        quaternions = np.asarray(
            [
                item.bones[bone].rotation.as_list()
                for item in collected
                if bone in item.bones
            ],
            dtype=float,
        )
        if quaternions.size == 0:
            continue
        flexion, abduction, twist = decompose_series(quaternions, frame)
        stacked[bone] = np.degrees(np.column_stack((flexion, abduction, twist)))
    return stacked


def bone_violations(bone: str, angles: np.ndarray, *, fps: float) -> list[RomViolation]:
    """Every DOF of one bone that leaves its typical band, worst band first.

    ``angles`` is the ``(n, 3)`` degree array :func:`decompose_clip` returns.
    """

    violations: list[RomViolation] = []
    for index, dof in enumerate(DOFS):
        limit = rom_limit(bone, dof)
        values = [float(value) for value in np.asarray(angles)[:, index]]
        bands = [limit.band_of(value) for value in values]
        worst = (
            "beyond_max"
            if "beyond_max" in bands
            else "beyond_typical"
            if "beyond_typical" in bands
            else "within_typical"
        )
        if worst == "within_typical":
            continue
        offending = [
            (value, limit.excess_deg(value, worst))
            for value, band in zip(values, bands)
            if band == worst
        ]
        peak_value = max(offending, key=lambda pair: pair[1])[0]
        integral = sum(excess for _, excess in offending) / max(fps, 1e-9)
        bounds = limit.max_deg if worst == "beyond_max" else limit.typical_deg
        threshold = limit.to_rest_relative(bounds[1] if peak_value > 0 else bounds[0])
        violations.append(
            RomViolation(
                bone=bone,
                dof=dof,
                peak_deg=round(peak_value, 4),
                frames=len(offending),
                integral_deg_s=round(integral, 6),
                band=worst,
                threshold_deg=round(threshold, 4),
                hard_assert=limit.hard_assert,
            )
        )
    return violations


def rom_violations(frames: list[ClipFrame], *, fps: float) -> list[RomViolation]:
    """Every DOF excursion in a clip, worst first."""

    series = decompose_clip(frames)
    found = [
        violation
        for bone, angles in series.items()
        for violation in bone_violations(bone, angles, fps=fps)
    ]
    found.sort(key=lambda item: (item.band != "beyond_max", -item.integral_deg_s))
    return found


def rom_checks(frames: list[ClipFrame], *, fps: float) -> list[CheckResult]:
    """Report-only range-of-motion checks. Plan §3.6.

    Every result is ``status="pass"``. The measurement is carried in
    ``measured`` so the distribution review has something to read, and 04c turns
    enforcement on per DOF once that review says which limits are trustworthy.
    """

    violations = rom_violations(frames, fps=fps)
    by_key = {(item.bone, item.dof): item for item in violations}
    results: list[CheckResult] = []
    for (bone, dof), limit in sorted(rom_limits().items()):
        violation = by_key.get((bone, dof))
        results.append(
            CheckResult(
                id=f"anatomy.rom.{bone}.{dof}",
                layer=ANATOMY,
                status="pass",
                measured=(
                    violation.as_dict()
                    if violation
                    else {"bone": bone, "dof": dof, "band": "within_typical"}
                ),
                threshold=limit.to_rest_relative(limit.max_deg[1]),
                severity=0.0,
                detail=(
                    "report-only (plan 04 §3.6); enforcement lands per DOF in 04c"
                    + ("; hard_assert DOF" if limit.hard_assert else "")
                    + (
                        "; mutation-only bone, cannot fire from generated motion"
                        if enforceability(bone) == "mutation_only"
                        else ""
                    )
                ),
            )
        )
    return results


__all__ = [
    "DOFS",
    "REQUIRED_SOURCE_FIELDS",
    "DofLimit",
    "RomError",
    "RomViolation",
    "bone_violations",
    "decompose_clip",
    "enforceability",
    "rom_checks",
    "rom_limit",
    "rom_limits",
    "rom_violations",
]
