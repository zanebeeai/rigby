from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rigby_v2.rigging.coordinates import (
    GltfDirection,
    GltfPosition,
    GltfQuaternionXYZW,
    GltfTransform,
    MujocoDirection,
    MujocoPosition,
    MujocoQuaternionWXYZ,
    convert_gltf_matrix_to_mujoco,
    convert_mujoco_matrix_to_gltf,
    gltf_direction_to_mujoco,
    gltf_position_to_mujoco,
    gltf_quaternion_to_mujoco,
    gltf_transform_to_mujoco,
    mujoco_direction_to_gltf,
    mujoco_position_to_gltf,
    mujoco_quaternion_to_gltf,
    mujoco_transform_to_gltf,
)
import pytest

pytestmark = pytest.mark.fast


def _gltf_rotation_matrix(value: GltfQuaternionXYZW) -> np.ndarray:
    return Rotation.from_quat(value.as_tuple()).as_matrix()


def test_audited_axis_mapping() -> None:
    assert gltf_direction_to_mujoco(GltfDirection(1, 0, 0)).as_tuple() == (1.0, 0.0, 0.0)
    assert gltf_direction_to_mujoco(GltfDirection(0, 1, 0)).as_tuple() == (0.0, 0.0, 1.0)
    assert gltf_direction_to_mujoco(GltfDirection(0, 0, 1)).as_tuple() == (0.0, -1.0, 0.0)
    assert GltfDirection.FRAME == "gltf_y_up_z_forward"
    assert MujocoDirection.FRAME == "mujoco_z_up_negative_y_forward"


def test_position_and_direction_roundtrip_without_conflating_types() -> None:
    position = GltfPosition(0.27, 1.42, -0.31)
    direction = GltfDirection(-0.4, 0.2, 0.89)
    converted_position = gltf_position_to_mujoco(position)
    converted_direction = gltf_direction_to_mujoco(direction)
    assert isinstance(converted_position, MujocoPosition)
    assert isinstance(converted_direction, MujocoDirection)
    np.testing.assert_allclose(
        mujoco_position_to_gltf(converted_position).as_array(), position.as_array(), atol=1e-12
    )
    np.testing.assert_allclose(
        mujoco_direction_to_gltf(converted_direction).as_array(), direction.as_array(), atol=1e-12
    )
    assert np.linalg.norm(converted_direction.as_array()) == np.linalg.norm(direction.as_array())


def test_quaternion_basis_and_storage_order_roundtrip() -> None:
    source_xyzw = Rotation.from_euler("xyz", [31, -47, 123], degrees=True).as_quat()
    source = GltfQuaternionXYZW(*map(float, source_xyzw))
    converted = gltf_quaternion_to_mujoco(source)
    assert isinstance(converted, MujocoQuaternionWXYZ)
    assert converted.ORDER == "wxyz"
    roundtripped = mujoco_quaternion_to_gltf(converted)
    np.testing.assert_allclose(
        _gltf_rotation_matrix(roundtripped), _gltf_rotation_matrix(source), atol=1e-12
    )


def test_transform_and_matrix_conversions_agree_and_roundtrip() -> None:
    source = GltfTransform(
        translation=GltfPosition(0.2, 1.3, 0.45),
        rotation=GltfQuaternionXYZW(
            *map(float, Rotation.from_euler("zyx", [20, -30, 42], degrees=True).as_quat())
        ),
    )
    typed = gltf_transform_to_mujoco(source)
    matrix = convert_gltf_matrix_to_mujoco(source.matrix())
    np.testing.assert_allclose(typed.matrix(), matrix, atol=1e-12)

    roundtripped = mujoco_transform_to_gltf(typed)
    np.testing.assert_allclose(roundtripped.matrix(), source.matrix(), atol=1e-12)
    np.testing.assert_allclose(
        convert_mujoco_matrix_to_gltf(matrix), source.matrix(), atol=1e-12
    )


def test_quaternion_types_normalize_and_reject_zero() -> None:
    assert GltfQuaternionXYZW(0, 0, 0, 2).as_tuple() == (0.0, 0.0, 0.0, 1.0)
    assert MujocoQuaternionWXYZ(2, 0, 0, 0).as_tuple() == (1.0, 0.0, 0.0, 0.0)
    with np.testing.assert_raises(ValueError):
        GltfQuaternionXYZW(0, 0, 0, 0)

