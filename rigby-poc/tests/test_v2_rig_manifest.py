from __future__ import annotations

from pathlib import Path

import mujoco

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.rigging.manifest_builder import stage_canonical_rig
import pytest

pytestmark = pytest.mark.medium


def test_staged_rig_manifest_binds_physics_visual_and_mapping(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    staged = stage_canonical_rig("medium", artifacts=artifacts)
    manifest = staged.manifest

    assert artifacts.exists(staged.reference, verify=True)
    assert artifacts.exists(manifest.mjcf, verify=True)
    assert manifest.visual_asset is not None
    assert artifacts.exists(manifest.visual_asset, verify=True)
    assert manifest.body_size_profile == "medium"
    assert len(manifest.dofs) == 67
    assert len(manifest.actuator_order) == 67
    assert len(manifest.bodies) >= 47
    assert all(body.mass_kg > 0 for body in manifest.bodies)
    assert all(
        min(
            body.diagonal_inertia_kg_m2.x,
            body.diagonal_inertia_kg_m2.y,
            body.diagonal_inertia_kg_m2.z,
        )
        > 0
        for body in manifest.bodies
    )
    assert len(manifest.colliders) >= 47
    assert len(manifest.adjacent_collision_exclusions) >= 40
    assert manifest.visual_skeleton_map["leftIndexDistal"] == ("left_index_dip",)
    assert manifest.visual_body_map["leftIndexDistal"] == "left_index_distal"
    assert manifest.visual_node_map["leftIndexDistal"] == "index_03_l"
    assert set(manifest.fixed_visual_bones) == {
        "upperChest",
        "leftShoulder",
        "rightShoulder",
        "leftToes",
        "rightToes",
    }
    assert manifest.visual_node_map["upperChest"] == "spine_03"
    assert "upperChest" not in manifest.visual_body_map

    model = mujoco.MjModel.from_xml_string(artifacts.read_bytes(manifest.mjcf).decode())
    assert model.nq == len(manifest.rest_qpos)
    assert model.nu == len(manifest.actuator_order)
