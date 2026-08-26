from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_v2.acceptance.container_bimanual import (
    _configured_pack_source,
    _neutral_request,
    build_container_lid_acceptance,
    build_two_handed_object_acceptance,
)
from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.motion import MotionCompilationError, MotionFailureReason
from rigby_v2.rigging import stage_canonical_rig
from rigby_v2.scenes import compile_scene, load_object_pack, stage_object_pack_scene
from rigby_v2.simulation import NativeMujocoRuntime


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or ""


def test_declared_handle_geometry_preserves_mass_metadata_and_passive_objects() -> None:
    bar = load_object_pack("two_handed_object")
    container = load_object_pack("container_lid")
    bar_object = bar.objects[0]
    container_object = container.objects[0]

    assert sum(part.mass_kg for part in bar_object.parts) == pytest.approx(3.5)
    assert sum(part.mass_kg for part in container_object.parts) == pytest.approx(1.5)
    assert {
        "left_forward_handle",
        "right_forward_handle",
    } <= {part.part_id for part in bar_object.parts}
    assert "lid_knob" in {part.part_id for part in container_object.parts}

    bar_scene = compile_scene(bar)
    container_scene = compile_scene(container)
    assert bar_scene.affordances["carry_bar.left_forward_grip"]["site"] == (
        "obj__carry_bar__left_forward_grasp"
    )
    assert bar_scene.affordances["carry_bar.right_forward_grip"]["site"] == (
        "obj__carry_bar__right_forward_grasp"
    )
    assert container_scene.affordances["container.open_lid"]["site"] == (
        "obj__container__lid_knob"
    )

    for model in (bar_scene.load_model(), container_scene.load_model()):
        assert model.nmocap == 0
        assert model.neq == 0
        assert all(
            not _name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(model.actuator_trnid[actuator_id, 0]),
            ).startswith("obj__")
            for actuator_id in range(model.nu)
        )
        assert sum(
            _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id).startswith("obj__")
            and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            for joint_id in range(model.njnt)
        ) == 1


def test_new_handle_geometry_has_no_neutral_self_motion(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    translations: dict[str, float] = {}
    for pack_id, joint_name in (
        ("two_handed_object", "obj__carry_bar__free"),
        ("container_lid", "obj__container__free"),
    ):
        scene = stage_object_pack_scene(
            pack_id,
            profile="medium",
            rig_reference=rig.reference,
            artifacts=artifacts,
        )
        source, model = _configured_pack_source(scene.compiled.xml, pack_id)
        result = NativeMujocoRuntime().simulate(
            _neutral_request(source, model, 1.0, f"neutral-{pack_id}")
        )
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        address = int(model.jnt_qposadr[joint_id])
        xyz = result.trace.qpos[:, address : address + 3]
        translations[pack_id] = float(
            np.max(np.linalg.norm(xyz - xyz[0], axis=1))
        )
        assert result.diagnostics["qpos_writes_after_initialization"] == 0
        assert max(
            (
                max(0.0, -contact.distance_m)
                for frame in result.trace.contacts
                for contact in frame.contacts
            ),
            default=0.0,
        ) <= 0.002

    # The passive cradle permits less than 3 mm of gravity settling; the
    # container's broad support is effectively stationary.
    assert translations["two_handed_object"] <= 0.0031
    assert translations["container_lid"] <= 0.0002


@pytest.mark.parametrize(
    "builder",
    (build_two_handed_object_acceptance, build_container_lid_acceptance),
)
def test_production_task_space_plans_fail_closed_when_collision_infeasible(
    builder,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / builder.__name__)
    with pytest.raises(MotionCompilationError) as failure:
        builder(artifacts)
    assert failure.value.reason is MotionFailureReason.IK_INFEASIBLE
    assert failure.value.details["motion_reason"] == "ik_infeasible"
    assert failure.value.details["violations"]
