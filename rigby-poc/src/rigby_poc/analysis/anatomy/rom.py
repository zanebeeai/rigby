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
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...kinematics import rig_kinematics
from ...models import ClipFrame
from ..contract import (
    ANATOMY,
    CheckResult,
    saturating_margin,
    saturating_severity,
    skipped,
)
from ..rig import PROJECT_ROOT
from .frame import all_frames, decompose_series
from .neutral import rest_offset, rest_relative

ROM_FILE = PROJECT_ROOT / "config" / "rom.v1.json"

DOFS = ("flexion", "abduction", "twist")

#: Degrees past a bound treated as fully egregious when ordering failures.
#:
#: This is an arbitrary constant and that costs nothing, because severity only
#: **ranks** failures -- it never decides one. The gate is the band. Stated
#: explicitly so nobody later mistakes it for a threshold: the largest excess in
#: the corpus is roughly 91 degrees, so 45 puts the worst elbow at saturation and
#: leaves the rest ordered.
SEVERITY_SCALE_DEG = 45.0
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
    enforced: bool
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

        For a bone with no anatomical neutral there is no conversion to perform:
        the authored bound was written rest-relatively in the first place, which
        is why those entries are ``provisional``. That branch is explicit rather
        than folded into an ``or 0.0``, so a ``None`` arriving for any other
        reason raises instead of being silently treated as zero.
        """

        if self.rest_offset_deg is None:
            return anatomical_deg
        return rest_relative(anatomical_deg, self.rest_offset_deg)

    def band_of(self, measured_deg: float) -> Band:
        low, high = (self.to_rest_relative(v) for v in self.max_deg)
        if measured_deg < low or measured_deg > high:
            return "beyond_max"
        low, high = (self.to_rest_relative(v) for v in self.typical_deg)
        if measured_deg < low or measured_deg > high:
            return "beyond_typical"
        return "within_typical"

    def room_deg(self, measured_deg: "float | np.ndarray") -> float:
        """Degrees of room left before the **max** bound. Negative once past it.

        Over a series, the worst frame decides: the room a DOF left is the
        smallest room any single frame left.

        The mirror of :meth:`excess_deg`, and the passing half of the same
        ruler. ``excess_deg`` answers "how far past" and returns 0.0 inside the
        band, which makes every in-band measurement look identical -- fine for
        ordering failures, and the reason a severity-derived score cannot order
        two clips that both pass. This answers "how much further it could have
        gone", so two clean DOFs are still distinguishable.

        Measured against ``max_deg`` rather than ``typical_deg`` deliberately.
        ``typical_deg`` is the narrower band and 46 of 47 corpus cases sit
        outside it on at least one DOF, so a typical-relative margin would be
        negative nearly everywhere and would clamp to zero -- reintroducing the
        flat-across-passing-clips problem this exists to remove.
        """

        low, high = (self.to_rest_relative(v) for v in self.max_deg)
        values = np.asarray(measured_deg, dtype=float)
        return float(np.min(np.minimum(high - values, values - low)))

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
    asset_sha256: str
    """Which skeleton this verdict was measured against.

    On the record rather than in an assertion: a violation that cannot say which
    rig it came from is a guard; one that can is an artifact.
    """

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
        # Derived here rather than read from the file. Storing it made the
        # committed JSON architecture-dependent: several offsets sit within
        # 1e-5 degrees of a three-decimal rounding boundary, which is inside
        # cross-architecture float drift, and the byte-equality test went red on
        # Windows while passing on macOS.
        offset = rest_offset(bone)
        for dof, value in entry["dofs"].items():
            limits[(bone, dof)] = DofLimit(
                bone=bone,
                dof=dof,
                typical_deg=tuple(value["typical_deg"]),
                max_deg=tuple(value["max_deg"]),
                hard_assert=bool(value["hard_assert"]),
                enforced=bool(value["enforced"]),
                rest_offset_deg=(
                    None
                    if offset is None
                    else math.degrees(getattr(offset, f"{dof}_rad"))
                ),
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
                asset_sha256=rig_kinematics().asset_sha256,
            )
        )
    return violations


def rom_violations(frames: list[ClipFrame], *, fps: float) -> list[RomViolation]:
    """Every DOF excursion in a clip, worst first."""

    return violations_of_series(decompose_clip(frames), fps=fps)


def violations_of_series(
    series: dict[str, np.ndarray], *, fps: float
) -> list[RomViolation]:
    """:func:`rom_violations` over an already-decomposed clip.

    Split out so :func:`rom_checks` can decompose once. It previously called
    ``decompose_clip`` for the measured-bone set and then ``rom_violations``,
    which decomposes again -- two full passes over every bone of every frame per
    call, for one clip's worth of angles. Same output, one pass.
    """

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

    series = decompose_clip(frames)
    measured_bones = set(series)
    violations = violations_of_series(series, fps=fps)
    by_key = {(item.bone, item.dof): item for item in violations}
    results: list[CheckResult] = []
    for (bone, dof), limit in sorted(rom_limits().items()):
        violation = by_key.get((bone, dof))
        if bone not in measured_bones:
            # NOT the same as "measured and within range". A bone absent from
            # the clip has no measurement, and saying `within_typical` here
            # would put a value in the output that is present and not derived
            # from what it claims to describe -- the shape lanes `infra`,
            # `analysis` and `groundtruth` each hit separately in
            # `_base_metrics` defaults, stale post-mutation reads and the
            # NOT_MEASURED gate state. An empty clip used to yield 156 checks
            # all claiming `within_typical`.
            results.append(
                skipped(
                    f"anatomy.rom.{bone}.{dof}",
                    ANATOMY,
                    detail=f"{bone} carries no pose in this clip; nothing was measured",
                )
            )
            continue
        breached = violation is not None and violation.band == "beyond_max"
        failing = breached and limit.enforced
        severity = (
            saturating_severity(
                abs(violation.peak_deg) - abs(violation.threshold_deg),
                SEVERITY_SCALE_DEG,
            )
            if failing
            else 0.0
        )
        # The worst frame decides, so the room is the smallest room any frame
        # left. A DOF past an *unenforced* max bound clamps to 0.0 rather than
        # going negative: the verdict is a pass by design in that case, and
        # "no room left" is the honest reading of a bound nothing enforces.
        room = limit.room_deg(series[bone][:, DOFS.index(dof)])
        results.append(
            CheckResult(
                id=f"anatomy.rom.{bone}.{dof}",
                layer=ANATOMY,
                status="fail" if failing else "pass",
                measured=(
                    violation.as_dict()
                    if violation
                    else {"bone": bone, "dof": dof, "band": "within_typical"}
                ),
                threshold=limit.to_rest_relative(limit.max_deg[1]),
                severity=severity,
                headroom=(
                    -severity
                    if failing
                    else saturating_margin(room, SEVERITY_SCALE_DEG)
                ),
                frames=(),
                detail=(
                    (
                        "beyond max"
                        if failing
                        else "report-only: "
                        + (
                            "the bound is provisional"
                            if limit.source["kind"] == "provisional"
                            else "mutation-only bone"
                            if enforceability(bone) == "mutation_only"
                            else "root orientation, not a joint angle"
                        )
                    )
                    + ("; hard_assert DOF" if limit.hard_assert else "")
                ),
            )
        )
    return results


def assert_evidence_matches_the_analysed_rig(rendered_sha256: str) -> None:
    """Fail loudly when the evidence depicts a different skeleton than the frames.

    ``rendered_sha256`` is ``render_provenance.asset_sha256`` from an evidence
    manifest -- the hash the **browser** computed over the bytes it actually
    parsed. Comparing that against this process's own parse is parse against
    parse. Comparing either against the declared hash in ``assets/manifest.json``
    is not, and stays green in the one case that matters: a GLB swapped together
    with its manifest, where capture passes while this process still serves the
    old skeleton from its cache.

    Plan 04 §6.1d, and lane `capture`'s 05.
    """

    parsed = rig_kinematics().asset_sha256
    if rendered_sha256 != parsed:
        raise RomError(
            "the evidence depicts a different skeleton than the one these "
            f"measurements were taken against: rendered {rendered_sha256}, "
            f"analysed {parsed}. Every anatomical frame and range-of-motion "
            "verdict in this process derives from the second."
        )


__all__ = [
    "DOFS",
    "REQUIRED_SOURCE_FIELDS",
    "DofLimit",
    "RomError",
    "RomViolation",
    "assert_evidence_matches_the_analysed_rig",
    "bone_violations",
    "decompose_clip",
    "violations_of_series",
    "enforceability",
    "rom_checks",
    "rom_limit",
    "rom_limits",
    "rom_violations",
]
