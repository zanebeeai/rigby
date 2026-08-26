from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rigby_v2.acceptance.production_container_lid import (
    build_production_container_lid_case,
    enriched_request,
)
from rigby_v2.motion.errors import MotionCompilationError, MotionFailureReason


def test_calibrated_container_lid_is_passive_and_authors_contact_lifecycle(
    tmp_path,
) -> None:
    case = build_production_container_lid_case(tmp_path / "artifacts")
    model = case.executor.scene_model
    program = case.proposal.program

    hinge_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "obj__container__lid_hinge"
    )
    assert hinge_id >= 0
    np.testing.assert_allclose(model.jnt_axis[hinge_id], (-1.0, 0.0, 0.0))
    assert model.dof_frictionloss[int(model.jnt_dofadr[hinge_id])] == pytest.approx(0.4)

    actuator_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(model.nu)
    }
    assert not any(name and "container" in name for name in actuator_names)
    assert program.metadata["object_target_channels"] == 0
    assert case.proposal.compile_attempt == 2
    assert {contact.contact_id for contact in program.contacts} == {
        "left-base-stabilize",
        "right-knob-open",
    }
    assert all(
        not name.startswith("obj__")
        for track in program.tracks
        for keyframe in track.keyframes
        for name in keyframe.joint_values
    )


def test_final_bounded_trajectory_fails_closed_at_typed_production_ik(tmp_path) -> None:
    case = build_production_container_lid_case(tmp_path / "artifacts")

    with pytest.raises(MotionCompilationError) as captured:
        enriched_request(case)

    error = captured.value
    assert error.reason is MotionFailureReason.IK_INFEASIBLE
    violations = error.details["violations"]
    position_errors = [
        float(item["error_m"]) for item in violations if "error_m" in item
    ]
    orientation_errors = [
        float(item["error_rad"]) for item in violations if "error_rad" in item
    ]
    assert max(position_errors) > 0.05
    assert max(orientation_errors) > 0.05
