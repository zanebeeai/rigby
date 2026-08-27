"""Audited coordinate conversions between Rigby's authoring and physics frames.

glTF uses a right-handed frame with +Y up and +Z forward.  Rigby's MuJoCo
models use +Z up and -Y forward.  The basis change is a +90 degree rotation
about X.  Public quaternion types also make the storage order explicit:
glTF is XYZW while MuJoCo is WXYZ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


_GLTF_TO_MUJOCO = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=float,
)
_MUJOCO_TO_GLTF = _GLTF_TO_MUJOCO.T


def _vec3(value: Iterable[float], *, label: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain three finite values")
    return array


def _quat4(value: Iterable[float], *, label: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=float)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain four finite values")
    norm = float(np.linalg.norm(array))
    if norm < 1e-12:
        raise ValueError(f"{label} must not be the zero quaternion")
    return array / norm


@dataclass(frozen=True, slots=True)
class _TypedVector:
    x: float
    y: float
    z: float

    FRAME: ClassVar[str]
    KIND: ClassVar[str]

    def __post_init__(self) -> None:
        values = self.as_array()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{type(self).__name__} must contain finite values")

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z], dtype=float)

    def as_tuple(self) -> tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))


@dataclass(frozen=True, slots=True)
class GltfPosition(_TypedVector):
    FRAME: ClassVar[str] = "gltf_y_up_z_forward"
    KIND: ClassVar[str] = "position"


@dataclass(frozen=True, slots=True)
class MujocoPosition(_TypedVector):
    FRAME: ClassVar[str] = "mujoco_z_up_negative_y_forward"
    KIND: ClassVar[str] = "position"


@dataclass(frozen=True, slots=True)
class GltfDirection(_TypedVector):
    FRAME: ClassVar[str] = "gltf_y_up_z_forward"
    KIND: ClassVar[str] = "direction"


@dataclass(frozen=True, slots=True)
class MujocoDirection(_TypedVector):
    FRAME: ClassVar[str] = "mujoco_z_up_negative_y_forward"
    KIND: ClassVar[str] = "direction"


@dataclass(frozen=True, slots=True)
class GltfQuaternionXYZW:
    x: float
    y: float
    z: float
    w: float
    FRAME: ClassVar[str] = "gltf_y_up_z_forward"
    ORDER: ClassVar[str] = "xyzw"

    def __post_init__(self) -> None:
        normalized = _quat4(self.as_tuple(), label=type(self).__name__)
        object.__setattr__(self, "x", float(normalized[0]))
        object.__setattr__(self, "y", float(normalized[1]))
        object.__setattr__(self, "z", float(normalized[2]))
        object.__setattr__(self, "w", float(normalized[3]))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)


@dataclass(frozen=True, slots=True)
class MujocoQuaternionWXYZ:
    w: float
    x: float
    y: float
    z: float
    FRAME: ClassVar[str] = "mujoco_z_up_negative_y_forward"
    ORDER: ClassVar[str] = "wxyz"

    def __post_init__(self) -> None:
        # scipy consumes XYZW, while the public constructor remains WXYZ.
        normalized_xyzw = _quat4(
            (self.x, self.y, self.z, self.w), label=type(self).__name__
        )
        object.__setattr__(self, "w", float(normalized_xyzw[3]))
        object.__setattr__(self, "x", float(normalized_xyzw[0]))
        object.__setattr__(self, "y", float(normalized_xyzw[1]))
        object.__setattr__(self, "z", float(normalized_xyzw[2]))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)


@dataclass(frozen=True, slots=True)
class GltfTransform:
    translation: GltfPosition
    rotation: GltfQuaternionXYZW

    def matrix(self) -> np.ndarray:
        value = np.eye(4, dtype=float)
        value[:3, :3] = Rotation.from_quat(self.rotation.as_tuple()).as_matrix()
        value[:3, 3] = self.translation.as_array()
        return value


@dataclass(frozen=True, slots=True)
class MujocoTransform:
    translation: MujocoPosition
    rotation: MujocoQuaternionWXYZ

    def matrix(self) -> np.ndarray:
        value = np.eye(4, dtype=float)
        quat = self.rotation
        value[:3, :3] = Rotation.from_quat((quat.x, quat.y, quat.z, quat.w)).as_matrix()
        value[:3, 3] = self.translation.as_array()
        return value


def gltf_position_to_mujoco(value: GltfPosition) -> MujocoPosition:
    result = _GLTF_TO_MUJOCO @ value.as_array()
    return MujocoPosition(*map(float, result))


def mujoco_position_to_gltf(value: MujocoPosition) -> GltfPosition:
    result = _MUJOCO_TO_GLTF @ value.as_array()
    return GltfPosition(*map(float, result))


def gltf_direction_to_mujoco(value: GltfDirection) -> MujocoDirection:
    result = _GLTF_TO_MUJOCO @ value.as_array()
    return MujocoDirection(*map(float, result))


def mujoco_direction_to_gltf(value: MujocoDirection) -> GltfDirection:
    result = _MUJOCO_TO_GLTF @ value.as_array()
    return GltfDirection(*map(float, result))


def gltf_quaternion_to_mujoco(value: GltfQuaternionXYZW) -> MujocoQuaternionWXYZ:
    source = Rotation.from_quat(value.as_tuple()).as_matrix()
    converted = _GLTF_TO_MUJOCO @ source @ _MUJOCO_TO_GLTF
    x, y, z, w = Rotation.from_matrix(converted).as_quat()
    return MujocoQuaternionWXYZ(w=float(w), x=float(x), y=float(y), z=float(z))


def mujoco_quaternion_to_gltf(value: MujocoQuaternionWXYZ) -> GltfQuaternionXYZW:
    source = Rotation.from_quat((value.x, value.y, value.z, value.w)).as_matrix()
    converted = _MUJOCO_TO_GLTF @ source @ _GLTF_TO_MUJOCO
    x, y, z, w = Rotation.from_matrix(converted).as_quat()
    return GltfQuaternionXYZW(x=float(x), y=float(y), z=float(z), w=float(w))


def gltf_transform_to_mujoco(value: GltfTransform) -> MujocoTransform:
    return MujocoTransform(
        translation=gltf_position_to_mujoco(value.translation),
        rotation=gltf_quaternion_to_mujoco(value.rotation),
    )


def mujoco_transform_to_gltf(value: MujocoTransform) -> GltfTransform:
    return GltfTransform(
        translation=mujoco_position_to_gltf(value.translation),
        rotation=mujoco_quaternion_to_gltf(value.rotation),
    )


def convert_gltf_matrix_to_mujoco(matrix: np.ndarray) -> np.ndarray:
    """Convert a rigid 4x4 glTF transform to the MuJoCo basis."""

    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("transform must be a finite 4x4 matrix")
    basis = np.eye(4, dtype=float)
    basis[:3, :3] = _GLTF_TO_MUJOCO
    return basis @ value @ basis.T


def convert_mujoco_matrix_to_gltf(matrix: np.ndarray) -> np.ndarray:
    """Convert a rigid 4x4 MuJoCo transform to the glTF basis."""

    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("transform must be a finite 4x4 matrix")
    basis = np.eye(4, dtype=float)
    basis[:3, :3] = _MUJOCO_TO_GLTF
    return basis @ value @ basis.T

