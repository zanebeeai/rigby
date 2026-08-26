"""Audited visual-skeleton ↔ MuJoCo physical-DOF pose conversion.

The visual representation contains glTF-basis bone-local *delta* transforms.
Forward conversion is measured from MuJoCo's actual kinematics.  Inverse
conversion solves each body's ordered hinge chain from those transforms; the
visual pose deliberately contains no source qpos that could be returned as a
shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .canonical_human import load_canonical_human
from .coordinates import (
    GltfPosition,
    GltfQuaternionXYZW,
    MujocoPosition,
    MujocoQuaternionWXYZ,
    gltf_position_to_mujoco,
    gltf_quaternion_to_mujoco,
    mujoco_position_to_gltf,
    mujoco_quaternion_to_gltf,
)
from .mapping import RigManifest, load_rig_manifest


@dataclass(frozen=True, slots=True)
class VisualBoneLocalTransform:
    """A glTF-basis local delta from the canonical bone's rest transform."""

    translation: GltfPosition
    rotation: GltfQuaternionXYZW


@dataclass(frozen=True, slots=True)
class VisualSkeletonPose:
    profile: str
    bone_local: Mapping[str, VisualBoneLocalTransform]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bone_local", MappingProxyType(dict(self.bone_local)))


@dataclass(frozen=True, slots=True)
class PoseRoundTripDiagnostics:
    profile: str
    fingertip_errors_m: Mapping[str, float]
    wrist_foot_errors_m: Mapping[str, float]
    max_fingertip_error_m: float
    max_wrist_foot_error_m: float
    max_local_rotation_error_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fingertip_errors_m", MappingProxyType(dict(self.fingertip_errors_m))
        )
        object.__setattr__(
            self, "wrist_foot_errors_m", MappingProxyType(dict(self.wrist_foot_errors_m))
        )

    @property
    def passes_acceptance(self) -> bool:
        return self.max_fingertip_error_m < 0.008 and self.max_wrist_foot_error_m < 0.015


@dataclass(frozen=True, slots=True)
class PoseRoundTripResult:
    reconstructed_qpos: tuple[float, ...]
    diagnostics: PoseRoundTripDiagnostics


class PoseAdapterError(ValueError):
    pass


def _body_local_transform(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    position = np.asarray(data.xpos[body_id], dtype=float)
    parent_id = int(model.body_parentid[body_id])
    if parent_id == 0:
        return position.copy(), rotation.copy()
    parent_rotation = np.asarray(data.xmat[parent_id], dtype=float).reshape(3, 3)
    parent_position = np.asarray(data.xpos[parent_id], dtype=float)
    return parent_rotation.T @ (position - parent_position), parent_rotation.T @ rotation


def _matrix_to_mujoco_quaternion(matrix: np.ndarray) -> MujocoQuaternionWXYZ:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return MujocoQuaternionWXYZ(w=float(w), x=float(x), y=float(y), z=float(z))


class VisualPhysicalPoseAdapter:
    """Bidirectional adapter tied to one canonical human size profile."""

    def __init__(
        self,
        profile: str = "medium",
        *,
        manifest: RigManifest | None = None,
    ) -> None:
        self.profile = profile
        self.manifest = manifest or load_rig_manifest()
        self.model = load_canonical_human(profile, manifest=self.manifest)
        self._bone_body_ids = self._resolve_bone_bodies()
        self._bone_joint_ids = self._resolve_bone_joints()
        self._rest = mujoco.MjData(self.model)
        self._rest.qpos[:] = self.model.qpos0
        mujoco.mj_forward(self.model, self._rest)
        self._rest_local = {
            bone: _body_local_transform(self.model, self._rest, body_id)
            for bone, body_id in self._bone_body_ids.items()
        }

    def _resolve_bone_joints(self) -> dict[str, tuple[int, ...]]:
        result: dict[str, tuple[int, ...]] = {}
        for bone, names in self.manifest.skeleton_to_dofs.items():
            ids = tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in names
            )
            if any(joint_id < 0 for joint_id in ids):
                raise PoseAdapterError(f"manifest maps {bone!r} to an unknown joint")
            result[bone] = ids
        return result

    def _resolve_bone_bodies(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for bone, joint_names in self.manifest.skeleton_to_dofs.items():
            if joint_names:
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[0]
                )
                if joint_id < 0:
                    raise PoseAdapterError(f"manifest maps {bone!r} to an unknown joint")
                result[bone] = int(self.model.jnt_bodyid[joint_id])
            elif bone == "head":
                result[bone] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, "head"
                )
        if any(body_id < 0 for body_id in result.values()):
            raise PoseAdapterError("manifest contains an unknown visual body")
        return result

    def _validate_qpos(self, qpos: np.ndarray) -> np.ndarray:
        value = np.asarray(qpos, dtype=float)
        if value.shape != (self.model.nq,) or not np.all(np.isfinite(value)):
            raise PoseAdapterError(f"physical pose must contain {self.model.nq} finite qpos values")
        root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.manifest.free_root_joint
        )
        root_address = int(self.model.jnt_qposadr[root_id])
        if not np.isclose(np.linalg.norm(value[root_address + 3 : root_address + 7]), 1.0, atol=1e-6):
            raise PoseAdapterError("free-root quaternion must be normalized WXYZ")
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            address = int(self.model.jnt_qposadr[joint_id])
            if self.model.jnt_limited[joint_id]:
                lower, upper = self.model.jnt_range[joint_id]
                if value[address] < lower - 1e-9 or value[address] > upper + 1e-9:
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    raise PoseAdapterError(f"joint {name!r} is outside its physical limits")
        return value

    def physical_to_visual(self, qpos: np.ndarray) -> VisualSkeletonPose:
        value = self._validate_qpos(qpos)
        data = mujoco.MjData(self.model)
        data.qpos[:] = value
        mujoco.mj_forward(self.model, data)
        transforms: dict[str, VisualBoneLocalTransform] = {}
        for bone, body_id in self._bone_body_ids.items():
            rest_position, rest_rotation = self._rest_local[bone]
            position, rotation = _body_local_transform(self.model, data, body_id)
            delta_position = position - rest_position
            delta_rotation = rest_rotation.T @ rotation
            gltf_position = mujoco_position_to_gltf(
                MujocoPosition(*map(float, delta_position))
            )
            gltf_rotation = mujoco_quaternion_to_gltf(
                _matrix_to_mujoco_quaternion(delta_rotation)
            )
            transforms[bone] = VisualBoneLocalTransform(
                translation=gltf_position,
                rotation=gltf_rotation,
            )
        return VisualSkeletonPose(profile=self.profile, bone_local=transforms)

    def visual_to_physical(self, pose: VisualSkeletonPose) -> np.ndarray:
        if pose.profile != self.profile:
            raise PoseAdapterError(
                f"visual pose profile {pose.profile!r} does not match adapter profile {self.profile!r}"
            )
        required = set(self._bone_body_ids)
        if set(pose.bone_local) != required:
            missing = sorted(required - set(pose.bone_local))
            extra = sorted(set(pose.bone_local) - required)
            raise PoseAdapterError(f"visual pose bone mismatch; missing={missing}, extra={extra}")

        output = np.asarray(self.model.qpos0, dtype=float).copy()
        root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.manifest.free_root_joint
        )
        root_address = int(self.model.jnt_qposadr[root_id])
        root_transform = pose.bone_local["hips"]
        root_translation = gltf_position_to_mujoco(root_transform.translation).as_array()
        output[root_address : root_address + 3] += root_translation
        root_rotation = gltf_quaternion_to_mujoco(root_transform.rotation)
        output[root_address + 3 : root_address + 7] = root_rotation.as_tuple()

        for bone, joint_ids in self._bone_joint_ids.items():
            if not joint_ids or self.model.jnt_type[joint_ids[0]] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            transform = pose.bone_local[bone]
            if np.linalg.norm(transform.translation.as_array()) > 1e-7:
                raise PoseAdapterError(
                    f"visual bone {bone!r} has translation but its physical joints are rotational"
                )
            target_quaternion = gltf_quaternion_to_mujoco(transform.rotation)
            target = Rotation.from_quat(
                (
                    target_quaternion.x,
                    target_quaternion.y,
                    target_quaternion.z,
                    target_quaternion.w,
                )
            ).as_matrix()
            axes = [np.asarray(self.model.jnt_axis[joint_id], dtype=float) for joint_id in joint_ids]
            lower = np.asarray([self.model.jnt_range[joint_id, 0] for joint_id in joint_ids])
            upper = np.asarray([self.model.jnt_range[joint_id, 1] for joint_id in joint_ids])
            rest_values = np.asarray(
                [
                    self.model.qpos0[int(self.model.jnt_qposadr[joint_id])]
                    for joint_id in joint_ids
                ]
            )

            def residual(values: np.ndarray) -> np.ndarray:
                predicted = np.eye(3, dtype=float)
                for axis, value, rest_value in zip(
                    axes, values, rest_values, strict=True
                ):
                    predicted = predicted @ Rotation.from_rotvec(
                        axis * (value - rest_value)
                    ).as_matrix()
                return Rotation.from_matrix(target.T @ predicted).as_rotvec()

            solved = least_squares(
                residual,
                x0=np.clip(rest_values, lower, upper),
                bounds=(lower, upper),
                ftol=1e-13,
                xtol=1e-13,
                gtol=1e-13,
                max_nfev=200,
            )
            if not solved.success or np.linalg.norm(residual(solved.x)) > 1e-7:
                raise PoseAdapterError(
                    f"visual rotation for bone {bone!r} is infeasible within physical joint limits"
                )
            for joint_id, value in zip(joint_ids, solved.x, strict=True):
                output[int(self.model.jnt_qposadr[joint_id])] = float(value)
        return self._validate_qpos(output)

    def round_trip(self, source_qpos: np.ndarray) -> PoseRoundTripResult:
        source = self._validate_qpos(source_qpos)
        visual = self.physical_to_visual(source)
        reconstructed = self.visual_to_physical(visual)
        diagnostics = self.compare_world_kinematics(source, reconstructed)
        return PoseRoundTripResult(
            reconstructed_qpos=tuple(map(float, reconstructed)),
            diagnostics=diagnostics,
        )

    def compare_world_kinematics(
        self,
        expected_qpos: np.ndarray,
        actual_qpos: np.ndarray,
    ) -> PoseRoundTripDiagnostics:
        expected = mujoco.MjData(self.model)
        actual = mujoco.MjData(self.model)
        expected.qpos[:] = self._validate_qpos(expected_qpos)
        actual.qpos[:] = self._validate_qpos(actual_qpos)
        mujoco.mj_forward(self.model, expected)
        mujoco.mj_forward(self.model, actual)

        fingertip_errors: dict[str, float] = {}
        for semantic, site_name in self.manifest.semantic_sites.items():
            if not semantic.endswith("Tip"):
                continue
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            fingertip_errors[semantic] = float(
                np.linalg.norm(expected.site_xpos[site_id] - actual.site_xpos[site_id])
            )

        wrist_foot_errors: dict[str, float] = {}
        for label, body_name in (("leftWrist", "left_hand"), ("rightWrist", "right_hand")):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            wrist_foot_errors[label] = float(
                np.linalg.norm(expected.xpos[body_id] - actual.xpos[body_id])
            )
        for semantic in ("leftFoot", "rightFoot"):
            site_name = self.manifest.semantic_sites[semantic]
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            wrist_foot_errors[semantic] = float(
                np.linalg.norm(expected.site_xpos[site_id] - actual.site_xpos[site_id])
            )

        max_rotation_error = 0.0
        for body_id in self._bone_body_ids.values():
            _, expected_rotation = _body_local_transform(self.model, expected, body_id)
            _, actual_rotation = _body_local_transform(self.model, actual, body_id)
            error = np.linalg.norm(
                Rotation.from_matrix(expected_rotation.T @ actual_rotation).as_rotvec()
            )
            max_rotation_error = max(max_rotation_error, float(error))
        return PoseRoundTripDiagnostics(
            profile=self.profile,
            fingertip_errors_m=fingertip_errors,
            wrist_foot_errors_m=wrist_foot_errors,
            max_fingertip_error_m=max(fingertip_errors.values(), default=0.0),
            max_wrist_foot_error_m=max(wrist_foot_errors.values(), default=0.0),
            max_local_rotation_error_rad=max_rotation_error,
        )
