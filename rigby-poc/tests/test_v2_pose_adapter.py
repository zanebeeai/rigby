from __future__ import annotations

from dataclasses import fields, replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rigby_v2.rigging import (
    GltfPosition,
    PoseAdapterError,
    VisualBoneLocalTransform,
    VisualPhysicalPoseAdapter,
    VisualSkeletonPose,
)


def _representative_pose(adapter: VisualPhysicalPoseAdapter, variant: int = 0) -> np.ndarray:
    model = adapter.model
    qpos = np.asarray(model.qpos0, dtype=float).copy()
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free")
    root_address = int(model.jnt_qposadr[root_id])
    qpos[root_address : root_address + 3] += np.asarray(
        [0.045, -0.035, 0.025] if variant == 0 else [-0.03, 0.025, 0.04]
    )
    xyzw = Rotation.from_euler(
        "xyz", [5, -7, 9] if variant == 0 else [-8, 4, -6], degrees=True
    ).as_quat()
    qpos[root_address + 3 : root_address + 7] = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        lower, upper = model.jnt_range[joint_id]
        fraction = (0.30 if joint_id % 2 else 0.60) if variant == 0 else (
            0.68 if joint_id % 3 else 0.38
        )
        # Staying near the middle of each physical interval avoids Euler
        # singularities while exercising every mapped joint and every digit.
        qpos[int(model.jnt_qposadr[joint_id])] = 0.55 * (
            lower + fraction * (upper - lower)
        )
    return qpos


@pytest.mark.parametrize("profile", ("small", "medium", "large"))
@pytest.mark.parametrize("variant", (0, 1))
def test_quantitative_pose_roundtrip_meets_world_space_acceptance(
    profile: str,
    variant: int,
) -> None:
    adapter = VisualPhysicalPoseAdapter(profile)
    source = _representative_pose(adapter, variant)
    result = adapter.round_trip(source)
    diagnostics = result.diagnostics
    assert diagnostics.profile == profile
    assert len(diagnostics.fingertip_errors_m) == 10
    assert set(diagnostics.wrist_foot_errors_m) == {
        "leftWrist",
        "rightWrist",
        "leftFoot",
        "rightFoot",
    }
    assert diagnostics.max_fingertip_error_m < 0.008
    assert diagnostics.max_wrist_foot_error_m < 0.015
    assert diagnostics.max_local_rotation_error_rad < 1e-6
    assert diagnostics.passes_acceptance
    np.testing.assert_allclose(result.reconstructed_qpos, source, atol=1e-7)


def test_inverse_genuinely_reconstructs_without_retaining_source_qpos() -> None:
    adapter = VisualPhysicalPoseAdapter("medium")
    source = _representative_pose(adapter)
    expected = source.copy()
    visual = adapter.physical_to_visual(source)
    assert "qpos" not in {field.name for field in fields(visual)}
    assert "qpos" not in {field.name for field in fields(next(iter(visual.bone_local.values())))}
    # Destroy the caller's input after forward conversion.  Reconstruction can
    # only use immutable visual bone-local transforms from this point onward.
    source[:] = adapter.model.qpos0
    reconstructed = adapter.visual_to_physical(visual)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-7)


def test_forward_pose_contains_audited_gltf_local_delta_conventions() -> None:
    adapter = VisualPhysicalPoseAdapter("large")
    visual = adapter.physical_to_visual(_representative_pose(adapter))
    assert visual.profile == "large"
    assert set(visual.bone_local) == set(adapter.manifest.skeleton_to_dofs)
    hips = visual.bone_local["hips"]
    assert hips.translation.FRAME == "gltf_y_up_z_forward"
    assert hips.rotation.FRAME == "gltf_y_up_z_forward"
    assert hips.rotation.ORDER == "xyzw"
    assert np.linalg.norm(hips.translation.as_array()) > 0


def test_inverse_rejects_missing_bone_and_nonphysical_local_translation() -> None:
    adapter = VisualPhysicalPoseAdapter("medium")
    visual = adapter.physical_to_visual(_representative_pose(adapter))
    missing = dict(visual.bone_local)
    missing.pop("leftIndexDistal")
    with pytest.raises(PoseAdapterError, match="bone mismatch"):
        adapter.visual_to_physical(VisualSkeletonPose(profile="medium", bone_local=missing))

    translated = dict(visual.bone_local)
    original = translated["leftIndexDistal"]
    translated["leftIndexDistal"] = replace(
        original,
        translation=GltfPosition(0.001, 0.0, 0.0),
    )
    with pytest.raises(PoseAdapterError, match="physical joints are rotational"):
        adapter.visual_to_physical(
            VisualSkeletonPose(profile="medium", bone_local=translated)
        )


def test_profile_mismatch_and_joint_limit_violations_are_rejected() -> None:
    adapter = VisualPhysicalPoseAdapter("small")
    source = _representative_pose(adapter)
    visual = adapter.physical_to_visual(source)
    with pytest.raises(PoseAdapterError, match="does not match"):
        adapter.visual_to_physical(
            VisualSkeletonPose(profile="large", bone_local=visual.bone_local)
        )

    joint_id = mujoco.mj_name2id(
        adapter.model, mujoco.mjtObj.mjOBJ_JOINT, "left_index_pip"
    )
    source[int(adapter.model.jnt_qposadr[joint_id])] = (
        adapter.model.jnt_range[joint_id, 1] + 0.1
    )
    with pytest.raises(PoseAdapterError, match="outside its physical limits"):
        adapter.physical_to_visual(source)


def test_visual_pose_mapping_is_immutable() -> None:
    adapter = VisualPhysicalPoseAdapter()
    visual = adapter.physical_to_visual(adapter.model.qpos0)
    with pytest.raises(TypeError):
        visual.bone_local["hips"] = VisualBoneLocalTransform(  # type: ignore[index]
            translation=GltfPosition(0, 0, 0),
            rotation=visual.bone_local["hips"].rotation,
        )
