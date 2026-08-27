from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from rigby_v2.acceptance.production_hand_tool import (
    build_production_hand_tool_case,
    build_production_hand_tool_program,
)
from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_core.motion import MotionCompilationError, MotionFailureReason

pytestmark = pytest.mark.medium


@pytest.mark.parametrize("authored_attempt", (1, 2))
def test_two_bounded_production_hand_tool_attempts_fail_as_typed_ik_rejections(
    tmp_path: Path,
    authored_attempt: int,
) -> None:
    case = build_production_hand_tool_case(
        ContentAddressedArtifactStore(tmp_path / f"attempt-{authored_attempt}"),
        authored_attempt=authored_attempt,
    )
    program = case.proposal.program
    assert program.metadata["authored_attempt"] == authored_attempt
    assert {track.target for track in program.tracks} == {
        "left_palm",
        "left_hand_joints",
    }
    assert {edge.body_a for edge in program.contacts} == {
        "left_thumb_distal",
        "left_index_distal",
    }
    assert all(edge.body_b == "obj__hammer__body__handle" for edge in program.contacts)

    model = case.executor.scene_model
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free")
    hammer = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "obj__hammer__free"
    )
    assert model.jnt_type[pelvis] == mujoco.mjtJoint.mjJNT_FREE
    assert model.jnt_type[hammer] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nmocap == 0 and model.neq == 0
    assert all(
        not (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[index, 0]),
            )
            or ""
        ).startswith("obj__")
        for index in range(model.nu)
    )

    with pytest.raises(MotionCompilationError) as failure:
        case.executor._prepare_certification_request(case.proposal)  # noqa: SLF001
    assert failure.value.reason is MotionFailureReason.IK_INFEASIBLE
    assert failure.value.details["violations"]
    assert all(
        item["objective"] == "left_palm"
        for item in failure.value.details["violations"]
    )
    # Compile rejection occurs before simulation, evidence, or certification.
    assert case.executor._certification_results == {}  # noqa: SLF001


def test_hand_tool_authoring_is_hard_bounded_to_two_attempts() -> None:
    with pytest.raises(ValueError, match="bounded to two"):
        build_production_hand_tool_program(
            rig_id="canonical-human-medium",
            scene_id="rigby-hand-tool-medium",
            authored_attempt=3,
        )
