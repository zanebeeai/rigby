from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_v2.rigging.canonical_human import load_canonical_human
from rigby_v2.rigging.mapping import load_rig_manifest

pytestmark = pytest.mark.medium


def _names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> set[str]:
    return {
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, object_type, index)) is not None
    }


def test_all_three_profiles_compile_and_scale_geometry_and_mass() -> None:
    models = [load_canonical_human(name) for name in ("small", "medium", "large")]
    assert all(model.nbody >= 48 for model in models)
    masses = [float(mujoco.mj_getTotalmass(model)) for model in models]
    extents = [float(model.stat.extent) for model in models]
    assert masses[0] < masses[1] < masses[2]
    assert extents[0] < extents[1] < extents[2]
    assert np.all(models[1].body_mass[1:] > 0.0)
    assert masses == [
        pytest.approx(58.0, abs=1.0),
        pytest.approx(74.0, abs=1.0),
        pytest.approx(93.0, abs=1.0),
    ]


def test_model_has_free_pelvis_full_body_and_actuated_nonfree_joints() -> None:
    model = load_canonical_human()
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    body_names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    required_bodies = {
        "pelvis",
        "lumbar",
        "torso",
        "neck",
        "head",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_thigh",
        "left_shin",
        "left_foot",
        "right_thigh",
        "right_shin",
        "right_foot",
    }
    assert required_bodies <= body_names
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free")
    assert root_id >= 0
    assert model.jnt_type[root_id] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nu == model.njnt - 1
    actuated_joint_ids = set(map(int, model.actuator_trnid[:, 0]))
    assert actuated_joint_ids == set(range(model.njnt)) - {root_id}
    assert np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
    assert np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
    assert "left_shoulder_flex" in joint_names
    assert "right_ankle_roll" in joint_names


def test_both_hands_have_five_articulated_digits_and_semantic_sites() -> None:
    model = load_canonical_human()
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    site_names = _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    for side in ("left", "right"):
        assert {
            f"{side}_thumb_opposition",
            f"{side}_thumb_mcp",
            f"{side}_thumb_pip",
            f"{side}_thumb_dip",
        } <= joint_names
        for digit in ("index", "middle", "ring", "little"):
            assert {
                f"{side}_{digit}_mcp",
                f"{side}_{digit}_pip",
                f"{side}_{digit}_dip",
            } <= joint_names
        assert f"{side}_palm" in site_names
        assert f"{side}_foot_sole" in site_names
        for digit in ("thumb", "index", "middle", "ring", "little"):
            assert f"{side}_{digit}_tip" in site_names


def test_manifest_mapping_resolves_every_joint_and_semantic_site() -> None:
    manifest = load_rig_manifest()
    model = load_canonical_human(manifest=manifest)
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    site_names = _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    mapped = {
        joint for values in manifest.skeleton_to_dofs.values() for joint in values
    }
    assert mapped == joint_names
    assert set(manifest.profiles) == {"small", "medium", "large"}
    assert set(manifest.semantic_sites.values()) <= site_names
    assert manifest.source_frame == "gltf_y_up_z_forward_xyzw"
    assert manifest.canonical_frame == "mujoco_z_up_negative_y_forward_wxyz"
    source_profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "config"
            / "rig_profiles"
            / "mesh2motion-human-vrm1.json"
        ).read_text(encoding="utf-8")
    )
    covered_visual_bones = set(manifest.skeleton_to_dofs) | set(
        manifest.fixed_visual_bones
    )
    assert set(source_profile["required_bones"]) <= covered_visual_bones
    assert set(manifest.fixed_visual_bones) == {
        "upperChest",
        "leftShoulder",
        "rightShoulder",
        "leftToes",
        "rightToes",
    }


def test_collision_geometry_is_primitive_and_adjacent_pairs_are_filtered() -> None:
    model = load_canonical_human()
    geom_types = {mujoco.mjtGeom(int(value)) for value in model.geom_type}
    assert geom_types <= {
        mujoco.mjtGeom.mjGEOM_PLANE,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_BOX,
    }
    assert model.nexclude >= 40
    assert np.all(model.geom_contype[1:] != 0)
    assert np.all(model.geom_conaffinity[1:] != 0)


def test_neutral_pose_can_step_with_finite_state() -> None:
    model = load_canonical_human()
    data = mujoco.MjData(model)
    for _ in range(8):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
