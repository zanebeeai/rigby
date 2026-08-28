"""Each known-bad case must fail the gate it was built to fail, and only that gate.

A corpus of only-valid clips cannot detect a check that has stopped firing: every
case passes before and after the check is disabled.  These six cases are the other
half.  Each names its gate in ``must_fail`` (plan 03 section 3.4), so deleting the
visibility check turns ``knownbad-gesture-out-of-view`` red rather than leaving the
suite uniformly green.

The gate names come from :mod:`evals.corpus.gates`, which scores bounds off the
numeric metrics.  That module is not a second definition of validity, and
``test_no_gate_fails_a_clip_the_compiler_calls_valid`` is what holds it to that.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evals.corpus import load_corpus
from evals.corpus.gates import (
    BREACHING_OUTCOMES,
    ENFORCED_OUTCOMES,
    GATE_SPECS,
    GateOutcome,
    StructuralGate,
    breached_gates,
    enforced_gates,
    evaluate_gates,
    failed_gates,
)
from evals.corpus.models import Family
from rigby_poc.models import Intent

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium

CASES = {case.id: case for case in load_corpus()}


@pytest.fixture(scope="module")
def metrics_by_case(compile_whole_corpus) -> dict[str, dict]:
    """``case id -> compiler metrics`` for the whole corpus.

    This was a module-level dict comprehension, so pytest paid all 47 compiles at
    *collection* -- on every invocation, including ``-m fast``, which then
    deselected every test in this file. That was 12.9s of the fast tier's 14.6s
    collection and it is what pushed CI's 90s ceiling to 131s. A fixture is only
    built when something in this file actually runs.
    """

    return {case_id: clip.metrics for case_id, clip in compile_whole_corpus().items()}

CASE_IDS = sorted(CASES)
KNOWN_BAD_IDS = sorted(
    case_id for case_id, case in CASES.items() if case.entry.family is Family.KNOWN_BAD
)
GOOD_IDS = sorted(set(CASE_IDS) - set(KNOWN_BAD_IDS))


def test_the_corpus_has_known_bad_cases_at_all() -> None:
    assert len(KNOWN_BAD_IDS) >= 4, KNOWN_BAD_IDS


@pytest.mark.parametrize("case_id", KNOWN_BAD_IDS)
def test_known_bad_case_fails_exactly_the_gates_it_names(case_id: str, metrics_by_case) -> None:
    case = CASES[case_id]
    metrics = metrics_by_case[case_id]
    if case.entry.intent is Intent.UNSUPPORTED:
        pytest.skip("scored by test_an_unsupported_case_produces_no_motion instead")
    observed = failed_gates(metrics)
    declared = list(case.entry.must_fail)
    assert observed == declared, (
        f"{case_id} was built to fail {[gate.value for gate in declared]} but failed "
        f"{[gate.value for gate in observed]}.\n"
        f"compiler said: {metrics.get('structural_failures')}"
    )


@pytest.mark.parametrize("case_id", KNOWN_BAD_IDS)
def test_known_bad_case_is_actually_rejected(case_id: str, metrics_by_case) -> None:
    """The gate has to reach the compiler's own verdict, not just this scorer."""
    assert metrics_by_case[case_id].get("structural_valid") is not True
    assert CASES[case_id].expected.structural_valid is False


@pytest.mark.parametrize("case_id", KNOWN_BAD_IDS)
def test_every_other_applicable_gate_still_passes(case_id: str, metrics_by_case) -> None:
    """A known-bad case must isolate its defect.

    A case that trips five gates at once cannot tell you which check stopped
    firing, so it is not a regression test for any of them.
    """
    case = CASES[case_id]
    outcomes = evaluate_gates(metrics_by_case[case_id])
    declared = set(case.entry.must_fail)
    collateral = sorted(
        gate.value
        for gate, outcome in outcomes.items()
        if gate not in declared and outcome in BREACHING_OUTCOMES
    )
    assert not collateral, f"{case_id} also breaches {collateral}, which it does not declare"


@pytest.mark.parametrize("case_id", GOOD_IDS)
def test_no_gate_fails_a_clip_the_compiler_calls_valid(case_id: str, metrics_by_case) -> None:
    """``gates.py`` must never be a second, disagreeing definition of validity.

    One direction only: everything this module calls ``FAILED`` has to be something
    the compiler also rejected.  The converse is false and not claimed -- the
    compiler has many path-specific failures that are not a metric against a bound.
    """
    metrics = metrics_by_case[case_id]
    observed = failed_gates(metrics)
    if metrics.get("structural_valid") is True:
        assert not observed, (
            f"{case_id}: gates.py says {[gate.value for gate in observed]} failed, "
            f"but the compiler passed the clip. One of them is wrong."
        )


def test_an_unsupported_case_produces_no_motion(metrics_by_case) -> None:
    """``Intent.UNSUPPORTED`` has no gate to trip -- the refusal is the check."""
    case = CASES["knownbad-eigenvalues-unsupported"]
    assert case.program.intent is Intent.UNSUPPORTED
    assert case.program.unsupported_reason
    assert case.expected.frame_count == 0
    assert case.expected.success is False
    assert not failed_gates(metrics_by_case[case.id])


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_an_unenforced_gate_is_never_reported_as_passing(case_id: str, metrics_by_case) -> None:
    """Full-body clips carry the kinematic metrics and no path compares them.

    Counting those as passes would report anatomical coverage the deterministic
    layer does not have, which is exactly plan 06 section 6.1's risk.
    """
    outcomes = evaluate_gates(metrics_by_case[case_id])
    for gate, outcome in outcomes.items():
        if outcome in ENFORCED_OUTCOMES:
            assert gate in set(enforced_gates(metrics_by_case[case_id]))
        else:
            assert outcome in {
                GateOutcome.UNGATED,
                GateOutcome.BREACHED_UNGATED,
                GateOutcome.NOT_MEASURED,
            }


def test_the_full_body_path_enforces_no_calibrated_ceiling(metrics_by_case) -> None:
    """Pins the 03b finding, so a future change that adds the check is noticed.

    ``_compile_full_body`` computes ``max_angular_velocity_rad_s``,
    ``max_angular_acceleration_rad_s2`` and ``max_angular_jerk_rad_s3`` and never
    compares them to ``hard_limits``.  **5 of 10 committed full-body cases (n = 10,
    the whole full-body family in this corpus) are over the acceleration and jerk
    ceilings -- by up to 2.9x and 3.4x -- while reporting** ``structural_valid:
    true``.  Two of the five are plain ``walk`` and ``run``.  The rate is 0.50 with
    n = 10 and is a property of this corpus, not an estimate of Rigby's output.

    The finding is not that the compiler is lax.  It is that the ceilings were
    derived from six hand-picked hang-ten *gesture* clips and are the wrong scale
    for whole-body motion, which is why plan 08 section 6.2 needs per-family values.
    Recorded here so that enforcing them without deriving them first turns this red
    rather than rejecting most of what Rigby generates.

    One caveat before anyone compares these numbers across families: **the bone set
    fed to the angular pass is per compile path, not per action.**  Whole body
    measures ten bones including both feet; sequence swaps the feet for both
    forearms; object interaction uses three bones of the active arm; gesture and
    strike take theirs from ``evaluate_gesture_structure``.  A whole-body
    acceleration figure and a sequence one are therefore not measured over the same
    skeleton, and an apparent inconsistency between two paths is not necessarily a
    bug in either.  Confirmed by lane ``analysis`` against
    ``analysis/full_body/failures.py`` during 02b.
    """
    breaching = {}
    for case_id, case in CASES.items():
        if case.program.intent is not Intent.FULL_BODY:
            continue
        metrics = metrics_by_case[case_id]
        assert not enforced_gates(metrics), (
            f"{case_id} now enforces {enforced_gates(metrics)}. If that is "
            f"intentional, this test and plan 06 section 6.1 both need updating."
        )
        outcomes = evaluate_gates(metrics)
        over = [
            gate.value
            for gate, outcome in outcomes.items()
            if outcome is GateOutcome.BREACHED_UNGATED
        ]
        if over:
            breaching[case_id] = over
    full_body_n = sum(
        1 for case in CASES.values() if case.program.intent is Intent.FULL_BODY
    )
    assert full_body_n == 10, full_body_n
    assert breaching == {
        "fullbody-burpee-cycle": ["angular_acceleration", "angular_jerk"],
        "fullbody-cartwheel": ["angular_acceleration", "angular_jerk"],
        "fullbody-kick-right": ["angular_acceleration", "angular_jerk"],
        "fullbody-run-forward": ["angular_acceleration", "angular_jerk"],
        "fullbody-walk-forward": ["angular_acceleration", "angular_jerk"],
    }, breaching


def test_a_known_bad_case_cannot_omit_its_gate() -> None:
    entry = CASES["knownbad-strike-hyperfast"].entry
    with pytest.raises(ValidationError, match="names no gate in must_fail"):
        entry.model_copy(update={"must_fail": []}).model_validate(
            entry.model_dump(mode="json") | {"must_fail": []}
        )


def test_a_known_bad_case_cannot_expect_to_be_valid() -> None:
    entry = CASES["knownbad-strike-hyperfast"].entry
    with pytest.raises(ValidationError, match="cannot expect to be structurally valid"):
        entry.model_validate(
            entry.model_dump(mode="json") | {"expected_structural_valid": True}
        )


def test_only_a_known_bad_case_may_declare_must_fail() -> None:
    entry = CASES["strike-jab-left"].entry
    with pytest.raises(ValidationError, match="only a known_bad case"):
        entry.model_validate(
            entry.model_dump(mode="json")
            | {"must_fail": [StructuralGate.ANGULAR_JERK.value]}
        )


def test_every_gate_is_exercised_by_some_case(metrics_by_case) -> None:
    """A gate no case measures is a gate nobody knows is still wired up.

    Reported rather than asserted to zero: three gates are unreachable by any
    compiled program, which is a finding about the compiler, not a gap to paper
    over.  See ``test_three_gates_cannot_be_tripped_by_any_compiled_program``.
    """
    measured = {
        gate
        for metrics in metrics_by_case.values()
        for gate, outcome in evaluate_gates(metrics).items()
        if outcome is not GateOutcome.NOT_MEASURED
    }
    assert measured == set(GATE_SPECS), sorted(
        gate.value for gate in set(GATE_SPECS) - measured
    )


#: Gates no compiled program can trip.  Not an aspiration -- a measurement, and the
#: single strongest argument for plan 06's metamorphic role.
UNREACHABLE_GATES = frozenset(
    {
        StructuralGate.WRIST_SWING,
        StructuralGate.WRIST_TWIST,
        StructuralGate.SELF_COLLISION,
    }
)


def test_three_gates_cannot_be_tripped_by_any_compiled_program(metrics_by_case) -> None:
    """Wrist swing, wrist twist and self-collision are unfalsifiable from generation.

    Measured over 517 compiles -- all 47 cases against 11 parameter-override
    configurations including every extreme wrist corner -- ``max_wrist_swing_rad``
    peaks at 98.0% of its limit and ``max_wrist_twist_rad`` at 94.1%.  Those are
    clamps, not coincidences.  ``self_collision_frames`` is 0 in all 517.

    So three of the eleven gates have **no compiled program that can exercise them**,
    and the only way to regression-test that they still fire is to inject a defect
    directly into a clip.  That is plan 06 section 1.4's argument arriving as a
    measurement rather than an assertion, and plan 06 section 6.1 records it.

    This test uses the corpus as blessed rather than re-running the sweep, which
    would add ninety seconds to the suite for a number that does not change.  Going
    red here means a gate became reachable, which is good news and wants the
    ``UNREACHABLE_GATES`` set narrowed.
    """
    tripped = {
        gate
        for metrics in metrics_by_case.values()
        for gate in breached_gates(metrics)
    }
    assert not (tripped & UNREACHABLE_GATES), sorted(
        gate.value for gate in tripped & UNREACHABLE_GATES
    )
