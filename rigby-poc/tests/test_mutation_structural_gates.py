"""The structural-failure namespace, pinned against what mutation can actually trip.

06b's registry covers the ids `analysis.validate()` emits. The contact and balance
detectors plan 06 section 3.3 needs are not in it: they append strings to
`metrics["structural_failures"]` and emit no `CheckResult`. This file pins the two
properties that make that namespace usable as ground truth, and both are measured
rather than asserted from the source.
"""

from __future__ import annotations

import math

import pytest

from evals.corpus import load_corpus
from evals.corpus.loader import compile_case
from evals.mutations.checks import known_check_ids
from evals.mutations.inject import add_dof
from evals.mutations.structural import (
    STRUCTURAL_GATES,
    gates_tripped,
    structural_gate_ids,
    undeclared_failures,
    unknown_structural_targets,
)
from rigby_poc.analysis import analyze

#: Compiles the full-body corpus.
pytestmark = pytest.mark.medium

#: Proximal joints, chosen because they displace the ankle. `rightFoot` is absent
#: from *this* list only: it cannot translate its own ankle, so it trips no
#: ankle-position gate at any angle up to 40 deg -- but it does trip the
#: foot-geometry gates, which is why the axis is per gate rather than global. See
#: `test_the_right_axis_is_per_gate_and_not_global`.
PROBE_JOINTS = (
    ("rightLowerLeg", "flexion"),
    ("rightUpperLeg", "flexion"),
    ("hips", "flexion"),
)
PROBE_DEGREES = (10, 40)


@pytest.fixture(scope="module")
def full_body_cases():
    """Corpus cases whose path evaluates structural gates at all."""
    found = []
    for case in load_corpus():
        try:
            clip = compile_case(case)
        except Exception:
            continue
        if not clip.frames or "support_constraints" not in clip.metrics:
            continue
        found.append((case, clip))
    return found


@pytest.fixture(scope="module")
def base_metrics(full_body_cases):
    out = []
    for case, clip in full_body_cases:
        request = case.compile_request()
        out.append((case.id, analyze(clip, request.program, request.scene)))
    return out


def test_the_probe_saw_real_full_body_cases(full_body_cases) -> None:
    """Instrument test. Every assertion below passes vacuously on an empty list."""
    assert len(full_body_cases) >= 10


def test_no_declared_gate_fires_on_an_unmutated_clip(base_metrics) -> None:
    """The property that makes this namespace usable as ground truth.

    Measured 0 across every corpus case that evaluates these gates, which is the
    opposite of the neighbouring `anatomy.rom.*` namespace where both elbow
    abductions are out of band on 46 of 46 clips and are therefore unusable as
    targets at any severity. A zero base rate means a detection is attributable to
    the mutation rather than to the bound.
    """
    dirty = {
        case_id: sorted(gates_tripped(metrics))
        for case_id, metrics in base_metrics
        if gates_tripped(metrics)
    }
    assert not dirty, (
        f"these gates fire before any mutation: {dirty}. A gate with a non-zero base "
        f"rate cannot attribute a detection to the mutation and must be stratified "
        f"out rather than pooled."
    )


def test_every_declared_gate_id_resolves_in_the_registry() -> None:
    assert structural_gate_ids() <= known_check_ids()
    assert len(structural_gate_ids()) == len(STRUCTURAL_GATES)


def test_a_structural_target_outside_the_registry_is_refused() -> None:
    from evals.mutations.family import MutationFamily, Tier
    from evals.mutations.spec import MutationSpec

    spec = MutationSpec(
        id="contact.invented",
        family=MutationFamily.CONTACT,
        targets=("structural.support_foot.no_such_gate",),
        severity=1.0,
        tier=Tier.SEVERE,
    )
    assert unknown_structural_targets([spec]) == {
        "contact.invented": ("structural.support_foot.no_such_gate",)
    }


def test_mutation_trips_the_contact_and_balance_gates(full_body_cases) -> None:
    """The gates are reachable, which is what separates a detector from a gap.

    A gate present in `analysis/full_body/failures.py` and unreachable by any
    mutation is plan 06 section 6.1's "no detector exists" case. Only an attempt
    distinguishes the two, so this attempts rather than reads the source.
    """
    tripped: set[str] = set()
    for case, clip in full_body_cases[:4]:
        request = case.compile_request()
        for bone, dof in PROBE_JOINTS:
            for degrees in PROBE_DEGREES:
                mutated = add_dof(
                    clip.model_copy(deep=True),
                    bone,
                    dof,
                    math.radians(degrees),
                    sign_from_clip=False,
                )
                metrics = analyze(mutated, request.program, request.scene)
                tripped |= gates_tripped(metrics)

    # The two families plan 06 needs, named individually rather than by count: a
    # count would stay green if the support gates went dark and unrelated ones
    # started firing.
    assert "structural.support_foot.planted_target" in tripped
    assert "structural.balance.support_region" in tripped
    assert len(tripped) >= 5, sorted(tripped)


#: Gates that read the ankle's *position* against its commanded IK target. Rotating
#: the foot about its own ankle cannot move these, however large the angle.
ANKLE_POSITION_GATES = frozenset(
    {
        "structural.support_foot.planted_target",
        "structural.support_foot.slide",
    }
)


def test_the_right_axis_is_per_gate_and_not_global(full_body_cases) -> None:
    """The axis choice is per gate, and the first draft of this test got it wrong.

    Measured earlier: injecting up to 40 degrees into ``rightFoot.flexion`` leaves
    ``max_support_foot_target_error_m`` identical to four significant figures, because
    those gates compare the *achieved ankle position* against the *commanded* IK
    target and rotating the foot about the ankle does not translate the ankle. From
    that this file first asserted the foot "trips nothing" -- and that is false. The
    same injection does trip ``ground.penetration``, ``pushup.toes_on_plane`` and
    ``recovery_foot.grounded``, because rotating the foot swings the sole and the toes
    into the floor.

    So a single global "best joint" does not exist: the foot is a **dead axis for the
    ankle-position gates and a live one for the foot-geometry gates**. A contact family
    that picked one joint for the whole family would have manufactured false negatives
    on whichever half it did not suit. Both halves are asserted, because a test of only
    the dead half would have passed on the wrong claim -- as this one did.
    """
    case, clip = full_body_cases[0]
    request = case.compile_request()
    mutated = add_dof(
        clip.model_copy(deep=True),
        "rightFoot",
        "flexion",
        math.radians(40),
        sign_from_clip=False,
    )
    tripped = gates_tripped(analyze(mutated, request.program, request.scene))

    assert not (tripped & ANKLE_POSITION_GATES), (
        "the foot cannot translate its own ankle, so these must stay clean"
    )
    assert tripped, "the foot moves the sole and toes; it is not a dead axis outright"
    assert "structural.ground.penetration" in tripped


def test_an_unrecognised_failure_string_is_reported_not_dropped() -> None:
    metrics = {"structural_failures": ["a gate nobody registered", "foot penetrates ground by 0.03 m"]}
    assert undeclared_failures(metrics) == ("a gate nobody registered",)
    assert gates_tripped(metrics) == {"structural.ground.penetration"}


def test_a_path_that_evaluates_no_gates_raises_rather_than_reading_clean() -> None:
    # "No detector on this path" and "measured, no failures" are different facts.
    with pytest.raises(KeyError, match="no detector on this path"):
        gates_tripped({"some_other_metric": 1.0})
