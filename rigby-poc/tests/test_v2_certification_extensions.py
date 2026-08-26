from __future__ import annotations

from dataclasses import replace

from rigby_v2.certification import (
    CandidateCertificationRequest,
    CertificationEngine,
    CertificationOutcome,
    CertificationPolicy,
    ExportValidation,
    GateCode,
)
from rigby_v2.contracts import ContactEdgeV2, CoordinateSystem
from rigby_v2.simulation import NativeMujocoRuntime

from test_v2_balance_controller import standing_request


def _candidate(policy: CertificationPolicy) -> CandidateCertificationRequest:
    return CandidateCertificationRequest(
        simulation=standing_request(),
        policy=replace(policy, pelvis_body="pelvis", torso_body="pelvis"),
        export_reimport=lambda result: ExportValidation(result.completed),
    )


def test_measured_acceleration_is_a_hard_gate() -> None:
    result = CertificationEngine().certify(
        _candidate(CertificationPolicy(default_joint_acceleration_limit=1e-9))
    )
    assert result.outcome is CertificationOutcome.INFEASIBLE
    assert any(
        violation.code is GateCode.ACCELERATION_LIMIT
        for violation in result.violations
    )


def test_measured_joint_range_excursion_is_a_hard_gate() -> None:
    baseline = NativeMujocoRuntime().simulate(standing_request())
    changed_qpos = baseline.trace.qpos.copy()
    # The test model's first scalar joint is shoulder at qpos address 7 with
    # a declared upper bound of 1.2 radians.
    changed_qpos[-1, 7] = 1.3
    forged = replace(baseline, trace=replace(baseline.trace, qpos=changed_qpos))

    class ForgedRuntime:
        def simulate(self, request):
            del request
            return forged

    result = CertificationEngine(ForgedRuntime()).certify(  # type: ignore[arg-type]
        _candidate(CertificationPolicy())
    )
    assert result.outcome is CertificationOutcome.INFEASIBLE
    assert any(
        violation.code is GateCode.JOINT_POSITION_LIMIT
        for violation in result.violations
    )


def test_authored_contact_window_break_penetration_and_slip_are_hard_gates() -> None:
    edge = ContactEdgeV2(
        contact_id="left-support",
        body_a="floor",
        body_b="left_foot_geom",
        start_s=0.0,
        end_s=0.2,
        max_slip_m=0.1,
    )
    certified = CertificationEngine().certify(
        replace(_candidate(CertificationPolicy()), planned_contacts=(edge,))
    )
    assert certified.outcome is CertificationOutcome.CERTIFIED

    persistent = CertificationEngine().certify(
        replace(
            _candidate(CertificationPolicy()),
            planned_contacts=(edge.model_copy(update={"end_s": 0.1}),),
        )
    )
    assert persistent.outcome is CertificationOutcome.INFEASIBLE
    assert any(item.code is GateCode.CONTACT_WINDOW for item in persistent.violations)

    tight = CertificationEngine().certify(
        replace(
            _candidate(CertificationPolicy()),
            planned_contacts=(
                edge.model_copy(
                    update={"max_penetration_m": 1e-9, "max_slip_m": 1e-9}
                ),
            ),
        )
    )
    codes = {item.code for item in tight.violations}
    assert GateCode.PENETRATION in codes
    assert GateCode.CONTACT_SLIP in codes

    baseline = NativeMujocoRuntime().simulate(standing_request())
    broken_frames = tuple(
        replace(
            frame,
            contacts=tuple(
                contact
                for contact in frame.contacts
                if frame.time_s <= 0.1
                or "left_foot_geom" not in {contact.geom1_name, contact.geom2_name}
            ),
        )
        for frame in baseline.trace.contacts
    )
    broken = replace(
        baseline, trace=replace(baseline.trace, contacts=broken_frames)
    )

    class BrokenRuntime:
        def simulate(self, request):
            del request
            return broken

    broken_result = CertificationEngine(BrokenRuntime()).certify(  # type: ignore[arg-type]
        replace(_candidate(CertificationPolicy()), planned_contacts=(edge,))
    )
    assert broken_result.outcome is CertificationOutcome.INFEASIBLE
    assert any(item.code is GateCode.CONTACT_BREAK for item in broken_result.violations)

    dropout_index = len(baseline.trace.contacts) // 2
    dropout_frames = tuple(
        replace(
            frame,
            contacts=tuple(
                contact
                for contact in frame.contacts
                if index != dropout_index
                or "left_foot_geom" not in {contact.geom1_name, contact.geom2_name}
            ),
        )
        for index, frame in enumerate(baseline.trace.contacts)
    )
    dropout = replace(
        baseline, trace=replace(baseline.trace, contacts=dropout_frames)
    )

    class DropoutRuntime:
        def simulate(self, request):
            del request
            return dropout

    dropout_result = CertificationEngine(DropoutRuntime()).certify(  # type: ignore[arg-type]
        replace(_candidate(CertificationPolicy()), planned_contacts=(edge,))
    )
    assert any(item.code is GateCode.CONTACT_DROPOUT for item in dropout_result.violations)


def test_certification_boundary_rejects_rotation_asset_and_coordinate_mismatch() -> None:
    baseline = NativeMujocoRuntime().simulate(standing_request())
    qpos = baseline.trace.qpos.copy()
    qpos[:, 3:7] *= 2.0
    nonunit = replace(baseline, trace=replace(baseline.trace, qpos=qpos))

    class NonUnitRuntime:
        def simulate(self, request):
            del request
            return nonunit

    rotation = CertificationEngine(NonUnitRuntime()).certify(  # type: ignore[arg-type]
        _candidate(CertificationPolicy())
    )
    assert rotation.outcome is CertificationOutcome.INFEASIBLE
    assert any(
        item.code is GateCode.ROTATION_NORMALIZATION
        for item in rotation.violations
    )

    asset = CertificationEngine().certify(
        replace(
            _candidate(CertificationPolicy()),
            expected_model_hash="0" * 64,
            rig_asset_hash="1" * 64,
            scene_rig_asset_hash="2" * 64,
        )
    )
    assert asset.outcome is CertificationOutcome.INFEASIBLE
    assert asset.simulation_runs == ()
    assert any(item.code is GateCode.ASSET_BINDING for item in asset.violations)

    coordinates = CertificationEngine().certify(
        replace(
            _candidate(CertificationPolicy()),
            rig_coordinate_system=CoordinateSystem(),
            scene_coordinate_system=CoordinateSystem(
                up_axis="+Y", forward_axis="-Z"
            ),
        )
    )
    assert coordinates.outcome is CertificationOutcome.INFEASIBLE
    assert any(
        item.code is GateCode.COORDINATE_BINDING
        for item in coordinates.violations
    )
