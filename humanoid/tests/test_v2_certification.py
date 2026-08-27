from __future__ import annotations

from dataclasses import replace

from rigby_v2.certification import (
    ArticulatedCompletionPredicate,
    CandidateCertificationRequest,
    CertificationEngine,
    CertificationOutcome,
    CertificationPolicy,
    ContactPair,
    ExportValidation,
    GateCode,
    GraspPredicate,
    HoldPredicate,
    JointComparator,
    LiftPredicate,
    ReleasePredicate,
    SupportFoot,
)
from rigby_v2.simulation import NativeMujocoRuntime

from test_v2_balance_controller import standing_request
import pytest

pytestmark = pytest.mark.medium


SUPPORT_FEET = (
    SupportFoot("left_foot", frozenset({"left_foot_geom"})),
    SupportFoot("right_foot", frozenset({"right_foot_geom"})),
)
GROUND_CONTACTS = frozenset(
    {
        ContactPair.of("floor", "left_foot_geom"),
        ContactPair.of("floor", "right_foot_geom"),
    }
)


def _policy(**changes) -> CertificationPolicy:
    return replace(
        CertificationPolicy(
            pelvis_body="pelvis",
            torso_body="pelvis",
            support_feet=SUPPORT_FEET,
            allowed_contact_pairs=GROUND_CONTACTS,
        ),
        **changes,
    )


def _candidate(*, policy: CertificationPolicy | None = None, predicates=(), export=True):
    return CandidateCertificationRequest(
        simulation=standing_request(),
        policy=policy or _policy(),
        predicates=tuple(predicates),
        export_reimport=(lambda result: ExportValidation(export, "round trip mismatch")),
    )


def test_certification_uses_three_actual_runs_and_export_reimport_gate() -> None:
    predicates = (
        # These exercise the generic measured interfaces on stable physical
        # bodies and contacts; domain packs supply their real object names.
        GraspPredicate(
            object_geoms=frozenset({"floor"}),
            effector_geoms=frozenset({"left_foot_geom", "right_foot_geom"}),
            min_distinct_effectors=2,
        ),
        HoldPredicate(
            object_body="pelvis",
            start_s=0.05,
            end_s=0.15,
            min_height_above_initial_m=-0.01,
            max_vertical_drift_m=0.02,
        ),
        ReleasePredicate(
            object_geoms=frozenset({"never_contacted_object"}),
            effector_geoms=frozenset({"left_foot_geom"}),
            start_s=0.1,
        ),
        ArticulatedCompletionPredicate(
            joint_name="shoulder",
            target=-1.0,
            tolerance=0.0,
            comparator=JointComparator.AT_LEAST,
        ),
    )
    result = CertificationEngine().certify(_candidate(predicates=predicates))

    assert result.outcome is CertificationOutcome.CERTIFIED
    assert len(result.simulation_runs) == 3
    assert all(run.completed for run in result.simulation_runs)
    assert all(evaluation.passed for evaluation in result.predicate_evaluations)
    assert {evaluation.name for evaluation in result.predicate_evaluations} == {
        "grasp",
        "hold",
        "release",
        "articulated_completion",
    }


def test_limits_penetration_contacts_and_object_state_are_hard_gates() -> None:
    velocity_result = CertificationEngine().certify(
        _candidate(policy=_policy(default_joint_velocity_limit=1e-6))
    )
    assert velocity_result.outcome is CertificationOutcome.INFEASIBLE
    assert any(v.code is GateCode.VELOCITY_LIMIT for v in velocity_result.violations)

    effort_result = CertificationEngine().certify(
        _candidate(
            policy=_policy(
                default_actuator_effort_limit=1e-8,
                default_joint_effort_limit=1e-8,
                default_joint_power_limit=1e-8,
            )
        )
    )
    assert effort_result.outcome is CertificationOutcome.INFEASIBLE
    assert {
        GateCode.ACTUATOR_EFFORT_LIMIT,
        GateCode.JOINT_EFFORT_LIMIT,
        GateCode.JOINT_POWER_LIMIT,
    }.issubset({violation.code for violation in effort_result.violations})

    contact_result = CertificationEngine().certify(
        _candidate(policy=_policy(allowed_contact_pairs=frozenset()))
    )
    assert contact_result.outcome is CertificationOutcome.INFEASIBLE
    assert any(v.code is GateCode.UNEXPECTED_CONTACT for v in contact_result.violations)

    order_result = CertificationEngine().certify(
        _candidate(
            policy=_policy(
                expected_contact_order=(ContactPair.of("missing", "object"),)
            )
        )
    )
    assert order_result.outcome is CertificationOutcome.INFEASIBLE
    assert any(v.code is GateCode.CONTACT_ORDER for v in order_result.violations)

    object_result = CertificationEngine().certify(
        _candidate(predicates=(LiftPredicate("pelvis", min_rise_m=1.0),))
    )
    assert object_result.outcome is CertificationOutcome.INFEASIBLE
    assert any(v.code is GateCode.OBJECT_PREDICATE for v in object_result.violations)


def test_penetration_fall_pelvis_foot_drift_and_slip_are_measured() -> None:
    assert CertificationPolicy().max_penetration_m == 0.002
    baseline = NativeMujocoRuntime().simulate(standing_request())
    first_frame = next(frame for frame in baseline.trace.contacts if frame.contacts)
    frame_index = baseline.trace.contacts.index(first_frame)
    penetrated_contact = replace(first_frame.contacts[0], distance_m=-0.003)
    penetrated_frame = replace(
        first_frame,
        contacts=(penetrated_contact, *first_frame.contacts[1:]),
    )
    frames = list(baseline.trace.contacts)
    frames[frame_index] = penetrated_frame
    penetrated = replace(baseline, trace=replace(baseline.trace, contacts=tuple(frames)))

    class PenetratedRuntime:
        def simulate(self, request):
            del request
            return penetrated

    penetration_result = CertificationEngine(PenetratedRuntime()).certify(_candidate())  # type: ignore[arg-type]
    assert penetration_result.outcome is CertificationOutcome.INFEASIBLE
    assert any(v.code is GateCode.PENETRATION for v in penetration_result.violations)

    balance_result = CertificationEngine().certify(
        _candidate(
            policy=_policy(
                max_pelvis_drop_m=1e-9,
                max_pelvis_drift_m=1e-9,
                max_torso_tilt_deg=1e-9,
                max_foot_drift_m=1e-9,
                max_foot_slip_m=1e-9,
            )
        )
    )
    codes = {violation.code for violation in balance_result.violations}
    assert balance_result.outcome is CertificationOutcome.INFEASIBLE
    assert GateCode.FALL in codes
    assert GateCode.PELVIS_DRIFT in codes
    assert GateCode.FOOT_DRIFT in codes
    assert GateCode.FOOT_SLIP in codes


def test_missing_object_capability_is_typed_unsupported() -> None:
    result = CertificationEngine().certify(
        _candidate(predicates=(LiftPredicate("missing_object", min_rise_m=0.1),))
    )

    assert result.outcome is CertificationOutcome.UNSUPPORTED
    assert result.violations[0].code is GateCode.UNSUPPORTED_PREDICATE


def test_hidden_state_mutation_and_failed_export_are_rejected() -> None:
    baseline = NativeMujocoRuntime().simulate(standing_request())
    forged = replace(
        baseline,
        diagnostics={**baseline.diagnostics, "qpos_writes_after_initialization": 1},
    )

    class ForgedRuntime:
        def simulate(self, request):
            del request
            return forged

    forged_result = CertificationEngine(ForgedRuntime()).certify(_candidate())  # type: ignore[arg-type]
    assert forged_result.outcome is CertificationOutcome.INFEASIBLE
    assert any(v.code is GateCode.TELEPORT for v in forged_result.violations)

    export_result = CertificationEngine().certify(_candidate(export=False))
    assert export_result.outcome is CertificationOutcome.SIMULATION_FAILED
    assert export_result.violations[0].code is GateCode.EXPORT_REIMPORT


def test_weld_and_mocap_shortcuts_are_never_certified() -> None:
    request = standing_request()
    welded_xml = request.model_xml.replace(
        "</mujoco>",
        '<equality><weld body1="pelvis" body2="upper_arm"/></equality></mujoco>',
    )
    welded = CertificationEngine().certify(
        replace(_candidate(), simulation=replace(request, model_xml=welded_xml))
    )
    assert welded.outcome is CertificationOutcome.INFEASIBLE
    assert any(v.code is GateCode.HIDDEN_WELD for v in welded.violations)

    mocap_xml = request.model_xml.replace(
        "</worldbody>",
        '<body name="forbidden_mocap" mocap="true" pos="0 0 2">'
        '<geom type="sphere" size="0.01"/></body></worldbody>',
    )
    mocap = CertificationEngine().certify(
        replace(_candidate(), simulation=replace(request, model_xml=mocap_xml))
    )
    assert mocap.outcome is CertificationOutcome.INFEASIBLE
    assert any(v.code is GateCode.MOCAP_BODY for v in mocap.violations)


def test_repeat_disagreement_and_all_candidates_rejected_are_typed() -> None:
    baseline = NativeMujocoRuntime().simulate(standing_request())
    changed_trace = replace(baseline.trace, ctrl=baseline.trace.ctrl + 1e-4)
    changed = replace(baseline, trace=changed_trace)

    class AlternatingRuntime:
        def __init__(self):
            self.index = 0

        def simulate(self, request):
            del request
            self.index += 1
            return baseline if self.index % 2 else changed

    disagreement = CertificationEngine(AlternatingRuntime()).certify(_candidate())  # type: ignore[arg-type]
    assert disagreement.outcome is CertificationOutcome.SIMULATION_FAILED
    assert disagreement.violations[0].code is GateCode.REPEAT_DISAGREEMENT

    rejected = CertificationEngine().certify_candidates(
        (
            _candidate(policy=_policy(default_joint_velocity_limit=1e-6)),
            _candidate(predicates=(LiftPredicate("pelvis", min_rise_m=1.0),)),
        )
    )
    assert rejected.outcome is CertificationOutcome.ALL_CANDIDATES_REJECTED
    assert rejected.selected_candidate_id is None
    assert len(rejected.candidates) == 2


def test_missing_export_validator_and_invalid_model_return_typed_results() -> None:
    no_export = replace(_candidate(), export_reimport=None)
    unsupported = CertificationEngine().certify(no_export)
    assert unsupported.outcome is CertificationOutcome.UNSUPPORTED
    assert unsupported.violations[0].code is GateCode.EXPORT_REIMPORT

    bad_simulation = replace(standing_request(), model_xml="<not-mujoco>")
    failed = CertificationEngine().certify(
        replace(_candidate(), simulation=bad_simulation)
    )
    assert failed.outcome is CertificationOutcome.SIMULATION_FAILED
