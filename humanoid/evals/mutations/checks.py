"""The check ids a mutation may target, derived from the analyzer rather than the plan.

A ``MutationSpec`` declares ``targets``: the deterministic checks it should trip.
Plan 06 section 5 makes those declarations load-bearing -- the detection matrix
asserts "the targeted check fires at high severity and non-targeted checks do not",
and section 6.1 says a mutation whose target never fires is "either a broken mutation
or a broken check, and the matrix says which by showing whether *anything* fired."

**There is a third case, and it is the one 06a shipped.**  A target id that no check
ever emits produces exactly the same matrix cell as "no detector exists": nothing
fired, on every case, at every severity.  06a took its four target ids from plan 06
section 3.3 -- ``anatomy.wrist.swing``, ``anatomy.wrist.twist``, ``signal.min_jerk``
and ``signal.sparc`` -- and measured over all 47 corpus cases,
:func:`rigby_poc.analysis.validate` emits none of the four.  ``min_jerk`` and
``sparc`` appear nowhere in the repository outside that one declaration.  All 28
ported severe specs therefore pointed at checks that cannot fire, and the matrix
would have read as a total detection failure by the deterministic layer.

That is the section 6.1 class in its worse form: not a visible gap, but a result.
So the registry here is **measured, not declared** -- :data:`FIXED_CHECK_IDS` is
pinned by ``test_mutation_check_registry.py`` against what the analyzer actually
emits over the corpus, so it cannot drift back into a plan-derived wish list.

**Two axes are report-only today and say so.**  ``anatomy.rom.*`` is emitted by
:func:`~rigby_poc.analysis.anatomy.rom.rom_checks` with ``status="pass"`` on every
result until lane ``anatomy``'s 04c turns enforcement on per DOF.  A detection
matrix that reads ``status`` would score every ROM mutation as undetected -- a
manufactured false negative of precisely the shape plan 06 section 6.1 warns about.
The observable signal is the ``band`` inside ``measured``, which is live now, so
:func:`rom_detected` reads that and :data:`REPORT_ONLY_PREFIXES` records why.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rigby_poc.analysis.anatomy.rom import rom_limits

from .spec import MutationSpec

#: Every check id :func:`rigby_poc.analysis.validate` emitted across all 47 corpus
#: cases, with the number of cases that emitted it.  Measured, not transcribed: the
#: counts are here because they are the honest denominator for any detection rate
#: quoted against one of these axes.  Only the three ``contract.clip.*`` checks are
#: emitted by every case; the anatomy and signal axes exist on the 14 gesture-path
#: cases alone, which is the same non-emission the cross-cutting CONDUCTOR line on
#: L3 gate item 5 records.
EMITTED_BY_CASES: Mapping[str, int] = {
    "anatomy.arm.self_collision": 14,
    "anatomy.forearm.twist": 14,
    "anatomy.travel_wheel.axis_error": 2,
    "anatomy.travel_wheel.forearm_separation": 2,
    "anatomy.travel_wheel.frontal_axis_error": 2,
    "anatomy.travel_wheel.hand_separation": 2,
    "anatomy.wrist.swing_twist_limit": 14,
    "contract.camera.active_hand_visibility": 14,
    "contract.clip.joint_limit_violations": 47,
    "contract.clip.non_finite_transforms": 47,
    "contract.clip.root_drift": 47,
    "contract.clip.rotational_discontinuities": 47,
    "physics.contact.foot_skate": 47,
    "physics.contact.ground_support": 47,
    "physics.ground.penetration": 47,
    "signal.activity.dead_limb": 47,
    "signal.smoothness.sparc": 47,
    "signal.angular.acceleration": 14,
    "signal.angular.jerk": 14,
    "signal.angular.velocity": 14,
    "signal.semantic_cycle.excursion": 2,
    "signal.semantic_cycle.reversals": 2,
    "signal.travel_wheel.cross_body_fraction": 2,
    "signal.travel_wheel.opposite_elbow_distance": 2,
    "signal.travel_wheel.order_exchange": 2,
}

#: The ids above, as a set.  ``anatomy.rom.*`` is *not* here: it is generated per
#: ``(bone, dof)`` by :func:`rom_check_ids` from the committed ROM document, so a
#: bone entering or leaving ``config/rom.v1.json`` moves the registry with it.
FIXED_CHECK_IDS: frozenset[str] = frozenset(EMITTED_BY_CASES)

#: Prefixes whose checks were emitted but could not reach ``status="fail"``.
#:
#: **Empty since 04c, and deliberately kept rather than deleted.**  It held one entry,
#: ``anatomy.rom.``, on the grounds that every ROM result was ``status="pass"`` until
#: enforcement landed.  04c landed it, and the property stopped being true of the
#: prefix: **82 of 156 DOFs are enforced and 74 are not**, so "report-only" is now a
#: fact about a ``(bone, dof)`` pair and a prefix map cannot express it.  Leaving the
#: old entry in place would have been a documented, structurally-wrong claim about
#: 82 live gates -- the class this repository has catalogued six times.
#:
#: :func:`report_only_reason` therefore resolves ROM ids through
#: :attr:`~rigby_poc.analysis.anatomy.rom.DofLimit.enforced` rather than through this
#: map.  The map remains for a future axis whose whole prefix is genuinely report-only.
REPORT_ONLY_PREFIXES: Mapping[str, str] = {}


def rom_check_ids() -> frozenset[str]:
    """``anatomy.rom.<bone>.<dof>`` for every limit in the committed ROM document."""
    return frozenset(f"anatomy.rom.{bone}.{dof}" for bone, dof in rom_limits())


def known_check_ids() -> frozenset[str]:
    """Every id a mutation may name as a target, across both detector namespaces.

    ``structural.*`` ids are not check ids -- they name gates that append a string to
    ``metrics["structural_failures"]`` and emit no ``CheckResult``. They are included
    here because a *target* is a claim about what a mutation should trip, and the
    contact and balance families trip gates in that namespace exclusively. Excluding
    them would have forced those families to declare no target at all, which
    :meth:`MutationSpec.__post_init__` refuses, or a check id that cannot fire.
    """
    from .structural import structural_gate_ids

    return FIXED_CHECK_IDS | rom_check_ids() | structural_gate_ids()


def report_only_reason(check_id: str) -> str:
    """Why ``check_id`` cannot reach ``fail`` today, or ``""`` if it can.

    ROM ids are resolved per ``(bone, dof)`` against the committed limit, because
    since 04c enforcement is per DOF: 82 of 156 gate and 74 remain report-only, so
    ``anatomy.rom.rightLowerLeg.flexion`` can fail while
    ``anatomy.rom.leftThumbProximal.twist`` cannot.  Answering at the prefix would be
    wrong for whichever side of the 82/74 split it picked.
    """
    for prefix, reason in REPORT_ONLY_PREFIXES.items():
        if check_id.startswith(prefix):
            return reason
    bone_dof = _rom_bone_dof(check_id)
    if bone_dof is None:
        return ""
    limit = rom_limits().get(bone_dof)
    if limit is None or limit.enforced:
        return ""
    return (
        f"{bone_dof[0]}.{bone_dof[1]} carries a report-only bound (04c enforces 82 of "
        f"156 DOFs; this is one of the 74 that stay status=pass). The live signal is "
        f"the `band` inside `measured`, which distinguishes beyond_typical and "
        f"beyond_max where `status` never will"
    )


def _rom_bone_dof(check_id: str) -> tuple[str, str] | None:
    """``("rightLowerLeg", "flexion")`` for a ROM id, else ``None``."""
    if not check_id.startswith("anatomy.rom."):
        return None
    parts = check_id.split(".")
    if len(parts) != 4:
        return None
    return parts[2], parts[3]


def unknown_targets(specs: Iterable[MutationSpec]) -> dict[str, tuple[str, ...]]:
    """Spec id -> the targets it declares that no check emits.

    Returned rather than raised so a caller can report the whole set at once; the
    two call sites that must refuse use :func:`require_known_targets`.
    """
    known = known_check_ids()
    found: dict[str, tuple[str, ...]] = {}
    for spec in specs:
        missing = tuple(target for target in spec.targets if target not in known)
        if missing:
            found[spec.id] = missing
    return found


def require_known_targets(specs: Iterable[MutationSpec]) -> None:
    """Raise unless every declared target is a check id something emits."""
    collected = list(specs)
    missing = unknown_targets(collected)
    if missing:
        detail = "; ".join(
            f"{spec_id} -> {', '.join(targets)}"
            for spec_id, targets in sorted(missing.items())
        )
        raise ValueError(
            f"mutation targets name checks nothing emits: {detail}. A target no check "
            f"emits produces the same detection-matrix cell as 'no detector exists', "
            f"so it reads as a measured failure of the deterministic layer rather "
            f"than as a broken declaration (plan 06 section 6.1)"
        )


def rom_detected(measured: Any) -> bool:
    """Whether a ``anatomy.rom.*`` result records an excursion out of the typical band.

    ``rom_checks`` reports ``status="pass"`` unconditionally until 04c, so the status
    carries no detection signal and reading it would score every ROM mutation as
    undetected.  ``band`` is live now: ``within_typical`` is clean, and both
    ``beyond_typical`` and ``beyond_max`` are excursions.

    A result with no ``band`` at all -- a bone the clip carries no pose for -- is
    **not** detection and is not cleanliness either; the caller must have excluded it
    as unmeasured before asking.  Returning ``False`` here would fold "not measured"
    into "measured negative", which four lanes hit independently this push.
    """
    if not isinstance(measured, Mapping):
        raise ValueError(
            f"a rom check result carries a mapping in `measured`, got {type(measured).__name__}"
        )
    band = measured.get("band")
    if band is None:
        raise ValueError(
            f"{measured.get('bone')!r}/{measured.get('dof')!r} has no band: the bone "
            f"carries no pose in this clip, so it was not measured. Exclude it as "
            f"unmeasured rather than scoring it as undetected"
        )
    return str(band) != "within_typical"


__all__ = [
    "EMITTED_BY_CASES",
    "FIXED_CHECK_IDS",
    "REPORT_ONLY_PREFIXES",
    "known_check_ids",
    "report_only_reason",
    "require_known_targets",
    "rom_check_ids",
    "rom_detected",
    "unknown_targets",
]
