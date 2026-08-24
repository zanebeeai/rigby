"""Which deterministic structural gate a case trips, derived from its metrics.

The compiler reports its verdict as ``structural_valid`` plus a list of free-text
``structural_failures``.  That is fine for a human reading a report and useless as a
test assertion: a known-bad case has to name the gate it was built to fail, and a
sentence is not a name.  This module names the gates and scores them from the
numeric metrics the clip already carries, so a case declares
``must_fail: [angular_jerk]`` rather than matching a substring that a reworded
message would break.

Two properties are load-bearing, and ``tests/test_corpus_known_bad.py`` pins both.

**Nothing here is a second definition of validity.**  Every gate this module calls
``FAILED`` must appear in the compiler's own ``structural_failures``.  The converse
does not hold and is not claimed: the compiler has many path-specific failures
(``dance does not visibly lift the left foot``, ``crawl does not retain its
requested root travel``) that are not expressible as a metric against a bound.

``nan_count`` and ``discontinuities`` are deliberately **not** gates here, despite
being universal metrics.  Measured in 03b: ``_compile_full_body``,
``_compile_object_handoff`` and ``_compile_sequence`` fail a clip on a non-zero
``discontinuities``; the gesture/strike/composite path and the non-handoff
``_compile_object_interaction`` path do not.  Nothing in the metrics distinguishes
those two groups, so this module cannot say whether a breach is enforced, and a gate
that cannot report its own enforcement is worse than an absent one.  The
``contract.clip.rotational_discontinuities`` check in ``analysis/safety.py`` is the
right instrument for that, and plan 06's detection matrix should use it.

**A gate is only enforced on some compile paths, and this module says so rather
than assuming.**  Measured in 03b across all seven executable intents: only
`gesture`, `strike` and `composite` compare the calibrated ceilings in
``config/motion_quality_reference.json`` against anything.  `full_body` computes
``max_angular_velocity_rad_s``, ``max_angular_acceleration_rad_s2`` and
``max_angular_jerk_rad_s3`` and never compares them to a limit; `grab`,
`object_interaction` and `sequence` do not emit the anatomical metrics at all.
Scoring an unenforced gate as ``PASSED`` would report deterministic coverage that
does not exist, so :class:`GateOutcome` keeps the two apart.  This is the "which
families have no detector" question plan 06 section 6.1 asks, answered from metrics
rather than from reading the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rigby_poc.quality import quality_reference

#: Suffix of the metrics key the calibrated paths record their limits under.
#: ``analysis/gesture.py`` writes it in the same dict where it builds
#: ``structural_failures``, so its presence *is* the enforcement signal -- there is
#: no separate list of enforcing intents to keep in sync.  Composite prefixes it per
#: arm (``left_quality_limits``), hence the suffix match.
QUALITY_LIMITS_SUFFIX = "quality_limits"


class StructuralGate(StrEnum):
    """One bound a compiled clip can breach."""

    # Calibrated ceilings from config/motion_quality_reference.json.
    WRIST_SWING = "wrist_swing"
    WRIST_TWIST = "wrist_twist"
    FOREARM_TWIST = "forearm_twist"
    SELF_COLLISION = "self_collision"
    HAND_VISIBILITY = "hand_visibility"
    ANGULAR_VELOCITY = "angular_velocity"
    ANGULAR_ACCELERATION = "angular_acceleration"
    ANGULAR_JERK = "angular_jerk"
    # Path-native bounds: the clip carries both the value and its reference, and
    # the path that emits the pair always acts on it.
    OBJECT_STEP = "object_step"
    CARRIED_OBJECT_STEP = "carried_object_step"
    STATEFUL_LANDING_HEIGHT = "stateful_landing_height"


class Enforcement(StrEnum):
    """Whether a compile path acts on a gate when the clip breaches it."""

    #: Enforced only where the path recorded a ``quality_limits`` block.
    CALIBRATED = "calibrated"
    #: Enforced by every path that emits the metric at all.  Only used for gates
    #: whose bound is a *sibling metric*: a path emits ``object_max_step_reference_m``
    #: precisely because it compares against it, so emitting the pair is the
    #: enforcement signal, exactly as ``quality_limits`` is for the calibrated ones.
    ALWAYS = "always"


class GateOutcome(StrEnum):
    """What one gate did on one clip.

    The three non-``FAILED`` outcomes are deliberately not collapsed: a suite that
    counts ``UNGATED`` and ``NOT_MEASURED`` as passes reports coverage it does not
    have.
    """

    #: Measured, enforced, within the bound.
    PASSED = "passed"
    #: Measured, enforced, over the bound.  Appears in ``structural_failures``.
    FAILED = "failed"
    #: Measured, over the bound, and the compile path does not act on it.
    BREACHED_UNGATED = "breached_ungated"
    #: Measured, within the bound, and the path would not have acted anyway.
    UNGATED = "ungated"
    #: The compile path does not emit this gate's metric.
    NOT_MEASURED = "not_measured"


#: Outcomes meaning the clip is over its bound, enforced or not.
BREACHING_OUTCOMES = frozenset({GateOutcome.FAILED, GateOutcome.BREACHED_UNGATED})
#: Outcomes meaning the gate exists for this clip and would have caught a breach.
ENFORCED_OUTCOMES = frozenset({GateOutcome.PASSED, GateOutcome.FAILED})


@dataclass(frozen=True)
class GateSpec:
    """How one gate reads its value and its bound out of ``clip.metrics``."""

    metric: str
    enforcement: Enforcement = Enforcement.CALIBRATED
    #: Key under ``hard_limits``.
    limit_key: str | None = None
    #: Sibling metric carrying this clip's own reference value.
    limit_metric: str | None = None
    #: Bound for a gate that has neither.
    literal_limit: float = 0.0
    #: ``True`` when the metric must stay at or below the bound.
    is_ceiling: bool = True

    def bound(self, metrics: dict[str, Any], limits: dict[str, Any]) -> float | None:
        """This clip's bound, or ``None`` when the clip does not carry one."""
        if self.limit_key is not None:
            return float(limits[self.limit_key])
        if self.limit_metric is not None:
            raw = metrics.get(self.limit_metric)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
            return float(raw)
        return self.literal_limit


GATE_SPECS: dict[StructuralGate, GateSpec] = {
    StructuralGate.WRIST_SWING: GateSpec("max_wrist_swing_rad", limit_key="wrist_swing_rad"),
    StructuralGate.WRIST_TWIST: GateSpec("max_wrist_twist_rad", limit_key="wrist_twist_rad"),
    StructuralGate.FOREARM_TWIST: GateSpec(
        "max_forearm_twist_rad", limit_key="forearm_twist_rad"
    ),
    StructuralGate.SELF_COLLISION: GateSpec("self_collision_frames"),
    StructuralGate.HAND_VISIBILITY: GateSpec(
        "active_hand_visibility_fraction",
        limit_key="minimum_active_hand_visibility_fraction",
        is_ceiling=False,
    ),
    StructuralGate.ANGULAR_VELOCITY: GateSpec(
        "max_angular_velocity_rad_s", limit_key="angular_velocity_rad_s"
    ),
    StructuralGate.ANGULAR_ACCELERATION: GateSpec(
        "max_angular_acceleration_rad_s2", limit_key="angular_acceleration_rad_s2"
    ),
    StructuralGate.ANGULAR_JERK: GateSpec(
        "max_angular_jerk_rad_s3", limit_key="angular_jerk_rad_s3"
    ),
    StructuralGate.OBJECT_STEP: GateSpec(
        "object_max_step_m",
        Enforcement.ALWAYS,
        limit_metric="object_max_step_reference_m",
    ),
    StructuralGate.CARRIED_OBJECT_STEP: GateSpec(
        "carried_object_max_step_m",
        Enforcement.ALWAYS,
        limit_metric="carried_object_max_step_reference_m",
    ),
    StructuralGate.STATEFUL_LANDING_HEIGHT: GateSpec(
        "stateful_object_landing_height_error_m",
        Enforcement.ALWAYS,
        literal_limit=0.01,
    ),
}

#: The tolerance ``analysis/gesture.py`` compares with, so this module and the
#: compiler agree on a value sitting exactly on its bound.
TOLERANCE = 1e-8


def enforces_quality_limits(metrics: dict[str, Any]) -> bool:
    """Whether the compile path that produced ``metrics`` gates on the ceilings."""
    return any(key.endswith(QUALITY_LIMITS_SUFFIX) for key in metrics)


def resolve_limits(metrics: dict[str, Any]) -> dict[str, Any]:
    """The ceilings this clip was judged against, preferring the ones it recorded.

    Reading the clip's own block rather than the config file means a change to the
    committed calibration moves this module and the compiler together, and keeps
    plan 08's threshold consolidation from having to list this file as a call site.
    """
    for key, value in metrics.items():
        if key.endswith(QUALITY_LIMITS_SUFFIX) and isinstance(value, dict):
            return value
    return quality_reference()["hard_limits"]


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def evaluate_gates(
    metrics: dict[str, Any], limits: dict[str, Any] | None = None
) -> dict[StructuralGate, GateOutcome]:
    """Score every gate against one clip's metrics."""
    resolved = limits if limits is not None else resolve_limits(metrics)
    calibrated = enforces_quality_limits(metrics)
    outcomes: dict[StructuralGate, GateOutcome] = {}
    for gate, spec in GATE_SPECS.items():
        value = _numeric(metrics.get(spec.metric))
        bound = spec.bound(metrics, resolved) if value is not None else None
        if value is None or bound is None:
            outcomes[gate] = GateOutcome.NOT_MEASURED
            continue
        over = (
            value > bound + TOLERANCE if spec.is_ceiling else value < bound - TOLERANCE
        )
        enforced = spec.enforcement == Enforcement.ALWAYS or calibrated
        if enforced:
            outcomes[gate] = GateOutcome.FAILED if over else GateOutcome.PASSED
        else:
            outcomes[gate] = GateOutcome.BREACHED_UNGATED if over else GateOutcome.UNGATED
    return outcomes


def _gates_with(
    metrics: dict[str, Any],
    limits: dict[str, Any] | None,
    wanted: frozenset[GateOutcome],
) -> list[StructuralGate]:
    outcomes = evaluate_gates(metrics, limits)
    return [gate for gate in GATE_SPECS if outcomes[gate] in wanted]


def failed_gates(
    metrics: dict[str, Any], limits: dict[str, Any] | None = None
) -> list[StructuralGate]:
    """Gates this clip trips *and* the compile path acts on."""
    return _gates_with(metrics, limits, frozenset({GateOutcome.FAILED}))


def breached_gates(
    metrics: dict[str, Any], limits: dict[str, Any] | None = None
) -> list[StructuralGate]:
    """Gates this clip is over, whether or not the compile path acts on it."""
    return _gates_with(metrics, limits, BREACHING_OUTCOMES)


def enforced_gates(
    metrics: dict[str, Any], limits: dict[str, Any] | None = None
) -> list[StructuralGate]:
    """Gates that would have failed this clip had it breached them."""
    return _gates_with(metrics, limits, ENFORCED_OUTCOMES)
