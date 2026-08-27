"""Named-joint adapters between the canonical rig and compiled task scenes."""

from __future__ import annotations

import mujoco
import numpy as np

from .controller import StandingControlConfig


def _joint_widths(joint_type: int) -> tuple[int, int]:
    kind = mujoco.mjtJoint(joint_type)
    if kind is mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if kind is mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def map_generalized_state(
    source_model: mujoco.MjModel,
    target_model: mujoco.MjModel,
    source_qpos: np.ndarray,
    source_qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map rig state by joint name while preserving target-only object state."""

    positions = np.broadcast_to(
        target_model.qpos0, (len(source_qpos), target_model.nq)
    ).copy()
    velocities = np.zeros((len(source_qvel), target_model.nv), dtype=np.float64)
    for source_joint in range(source_model.njnt):
        name = mujoco.mj_id2name(
            source_model, mujoco.mjtObj.mjOBJ_JOINT, source_joint
        )
        if name is None:
            raise ValueError("source rig contains an unnamed joint")
        target_joint = mujoco.mj_name2id(
            target_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if target_joint < 0:
            raise ValueError(f"compiled scene is missing rig joint {name!r}")
        joint_type = int(source_model.jnt_type[source_joint])
        if joint_type != int(target_model.jnt_type[target_joint]):
            raise ValueError(f"compiled scene changed the type of rig joint {name!r}")
        qpos_width, qvel_width = _joint_widths(joint_type)
        source_qpos_adr = int(source_model.jnt_qposadr[source_joint])
        target_qpos_adr = int(target_model.jnt_qposadr[target_joint])
        source_qvel_adr = int(source_model.jnt_dofadr[source_joint])
        target_qvel_adr = int(target_model.jnt_dofadr[target_joint])
        positions[:, target_qpos_adr : target_qpos_adr + qpos_width] = source_qpos[
            :, source_qpos_adr : source_qpos_adr + qpos_width
        ]
        velocities[:, target_qvel_adr : target_qvel_adr + qvel_width] = source_qvel[
            :, source_qvel_adr : source_qvel_adr + qvel_width
        ]
    return positions, velocities


def project_qpos_to_rig(
    scene_model: mujoco.MjModel,
    rig_model: mujoco.MjModel,
    scene_qpos: np.ndarray,
) -> np.ndarray:
    projected = np.broadcast_to(
        rig_model.qpos0, (len(scene_qpos), rig_model.nq)
    ).copy()
    for rig_joint in range(rig_model.njnt):
        name = mujoco.mj_id2name(
            rig_model, mujoco.mjtObj.mjOBJ_JOINT, rig_joint
        )
        if name is None:
            raise ValueError("rig contains an unnamed joint")
        scene_joint = mujoco.mj_name2id(
            scene_model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if scene_joint < 0:
            raise ValueError(f"compiled scene is missing rig joint {name!r}")
        joint_type = int(rig_model.jnt_type[rig_joint])
        if joint_type != int(scene_model.jnt_type[scene_joint]):
            raise ValueError(f"compiled scene changed the type of rig joint {name!r}")
        qpos_width, _ = _joint_widths(joint_type)
        rig_address = int(rig_model.jnt_qposadr[rig_joint])
        scene_address = int(scene_model.jnt_qposadr[scene_joint])
        projected[:, rig_address : rig_address + qpos_width] = scene_qpos[
            :, scene_address : scene_address + qpos_width
        ]
    return projected


def canonical_standing_config(model: mujoco.MjModel) -> StandingControlConfig:
    prefixes = (
        "spine_",
        "chest_",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    correction_joints = tuple(
        name
        for joint_id in range(model.njnt)
        if (
            name := mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
        )
        and name.startswith(prefixes)
    )
    return StandingControlConfig(
        support_foot_bodies=("left_foot", "right_foot"),
        pelvis_body="pelvis",
        torso_body="torso",
        correction_joint_names=correction_joints,
        max_generalized_correction=30.0,
    )

