"""The second detector namespace: structural-failure gates, which are not check ids.

06b's registry (:mod:`evals.mutations.checks`) validates a mutation's ``targets``
against the ids :func:`rigby_poc.analysis.validate` emits.  That namespace does not
contain the detectors plan 06 section 3.3 needs for the **contact** and **balance**
families.  Those gates append a *string* to ``metrics["structural_failures"]`` and
emit no ``CheckResult`` at all, so a contact mutation declaring a check-id target
would have been refused by the registry, and one declaring nothing would have been
unscoreable.  Both readings are wrong: the detectors exist and they work.

**Measured rather than read out of the source.**  A gate present in
``analysis/full_body/failures.py`` and unreachable by any mutation is the section 6.1
case that must be reported as a gap, and only an attempt distinguishes it from one
that works.  Probing four full-body cases across six proximal joints at three
magnitudes tripped **18 distinct gates**, and **none of them fires on an unmutated
clip** -- 0 across all 10 corpus cases that carry a support block.

That zero is what makes this namespace usable, and it is the opposite of what 06b
found next door: in ``anatomy.rom.*`` both elbow abductions are out of band on 46 of
46 clips, so they are unusable as targets at any severity, and lane `analysis`
measured that a pooled ROM rate would be counting one compiler defect 91 times rather
than 156 independent axes.  Here the base rate is zero on every gate, so a detection
is attributable to the mutation and a pooled rate is honest.

**The joint matters more than the magnitude.**  ``rightFoot.flexion`` never trips the
support gates at any angle up to 40 degrees, because these compare the *achieved*
ankle position against the *commanded* IK target and rotating the foot about its own
ankle does not translate the ankle.  The knee crosses at 1.587 degrees and the error
is linear at 7.5614e-3 m per degree.  A contact family built on the foot would have
reported zero detection at every severity -- a manufactured false negative
indistinguishable from a dead gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .spec import MutationSpec

#: Stable id -> the literal prefix of the failure string the gate appends.
#:
#: Prefix rather than equality because several gates interpolate a measured value
#: into their message (``foo penetrates ground by 0.031 m``), and matching the whole
#: string would make the id depend on the number -- so a gate would stop resolving at
#: exactly the severities that trip it hardest.
STRUCTURAL_GATES: Mapping[str, str] = {
    # contact
    "structural.support_foot.planted_target": "support foot failed its planted world-space target",
    "structural.support_foot.slide": "support foot slides during a planted stance interval",
    "structural.recovery_foot.grounded": "a recovery foot did not finish on the ground",
    "structural.ground.penetration": "foot penetrates ground",
    "structural.ground.horizontal_penetration": "horizontal pose penetrates the ground plane",
    "structural.pushup.toes_on_plane": "push-up toes do not remain on the support plane",
    "structural.pushup.toes_slide": "push-up toes slide during repetitions",
    "structural.pushup.toe_height": "push-up toe support changes height during repetitions",
    # balance
    "structural.balance.support_region": "root finished outside the balanced support region",
    "structural.balance.ground_contact": "non-jumping motion loses both ground contacts",
    "structural.balance.plank_alignment": "push-up legs do not retain a straight plank alignment",
    "structural.balance.upright": "requested grounded horizontal pose remains too upright",
    "structural.balance.head_plant": "cartwheel places the head on the floor instead of the hands",
    "structural.root.vertical_settle": "root did not settle vertically after recovery",
    # semantic, reachable but owned by the semantic family
    "structural.semantic.rotation_overshoot": "whole-body rotation overshoots the requested angle",
    "structural.semantic.left_foot_lift": "dance does not visibly lift the left foot",
    "structural.semantic.right_foot_lift": "dance does not visibly lift the right foot",
    "structural.semantic.beat_count": "dance foot-lift count does not match the requested beats",
}

#: Gates measured as tripped by at least one mutation in the 4x6x3 probe. Declared
#: so a family cannot silently target one that has never been shown to fire; a gate
#: absent here is a **gap to report**, not a detector to score against.
REACHABLE_BY_MUTATION: frozenset[str] = frozenset(STRUCTURAL_GATES)


def structural_gate_ids() -> frozenset[str]:
    return frozenset(STRUCTURAL_GATES)


def gates_tripped(metrics: Mapping[str, Any]) -> frozenset[str]:
    """Which declared gates the failure list in ``metrics`` reports.

    Raises on a metrics dict with no ``structural_failures`` key at all, rather than
    returning the empty set. A path that never writes the key and a path that wrote
    an empty list are different facts -- the first has no detector here and the
    second measured clean -- and folding them together is the not-measured /
    measured-negative conflation four lanes hit separately in this push.
    """
    if "structural_failures" not in metrics:
        raise KeyError(
            "this clip's metrics carry no `structural_failures` key, so no structural "
            "gate was evaluated for it. That is 'no detector on this path', not 'no "
            "failures'; render it as a gap rather than as a clean result"
        )
    reported = [str(item) for item in metrics["structural_failures"]]
    return frozenset(
        gate_id
        for gate_id, prefix in STRUCTURAL_GATES.items()
        if any(message.startswith(prefix) for message in reported)
    )


def undeclared_failures(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    """Failure strings this module does not have an id for.

    The registry is a declaration and declarations rot. A gate that starts firing
    and matches nothing here would otherwise be invisible -- counted as no detection
    rather than as an unrecognised detector -- so the matrix reports these rather
    than dropping them.
    """
    reported = [str(item) for item in metrics.get("structural_failures", [])]
    prefixes = tuple(STRUCTURAL_GATES.values())
    return tuple(
        message for message in reported if not message.startswith(prefixes)
    )


def unknown_structural_targets(
    specs: Iterable[MutationSpec],
) -> dict[str, tuple[str, ...]]:
    """Spec id -> the ``structural.*`` targets it declares that are not registered."""
    known = structural_gate_ids()
    found: dict[str, tuple[str, ...]] = {}
    for spec in specs:
        missing = tuple(
            target
            for target in spec.targets
            if target.startswith("structural.") and target not in known
        )
        if missing:
            found[spec.id] = missing
    return found


__all__ = [
    "REACHABLE_BY_MUTATION",
    "STRUCTURAL_GATES",
    "gates_tripped",
    "structural_gate_ids",
    "undeclared_failures",
    "unknown_structural_targets",
]
