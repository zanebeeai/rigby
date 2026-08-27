from __future__ import annotations

from pathlib import Path

import mujoco

from rigby_v2.acceptance.production_grasp_place import (
    build_production_grasp_place_case,
    run_authoritative_diagnostic,
)
from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.simulation import SimulationStatus
import pytest

pytestmark = pytest.mark.medium


def test_production_grasp_program_compiles_but_failed_trace_is_not_certified(
    tmp_path: Path,
) -> None:
    case = build_production_grasp_place_case(
        ContentAddressedArtifactStore(tmp_path / "artifacts")
    )
    program = case.proposal.program
    assert program.metadata["task_space_sample_hz"] == 1
    assert {track.target for track in program.tracks} == {
        "left_palm",
        "left_arm_joints",
        "left_hand_joints",
        "left_thumb_tip",
        "left_index_tip",
    }
    assert tuple(phase.kind.value for phase in program.phases) == (
        "setup",
        "action",
        "action",
        "hold",
        "hold",
        "release",
        "recovery",
    )
    assert tuple(contact.contact_id for contact in program.contacts) == (
        "left-opposed-digits-block",
    )

    model = case.executor.scene_model
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free")
    block = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj__block__free")
    assert model.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE
    assert model.jnt_type[block] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nmocap == 0
    assert not any(
        model.eq_type[index] == mujoco.mjtEq.mjEQ_WELD
        for index in range(model.neq)
    )
    assert all(
        not (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[actuator, 0]),
            )
            or ""
        ).startswith("obj__")
        for actuator in range(model.nu)
    )

    result, diagnostic = run_authoritative_diagnostic(case)
    assert result.status is SimulationStatus.COMPLETED
    assert diagnostic.production_compile_succeeded
    assert result.diagnostics["qpos_writes_after_initialization"] == 0
    # The redesigned path creates opposing thumb/index contacts but the free
    # block still fails the 120 mm physical-lift gate.
    assert diagnostic.max_simultaneous_digit_contacts >= 2
    assert any("thumb" in name for name in diagnostic.contacting_digits)
    assert any("index" in name for name in diagnostic.contacting_digits)
    assert diagnostic.object_rise_m < 0.12
    assert diagnostic.max_penetration_m > 0.0
