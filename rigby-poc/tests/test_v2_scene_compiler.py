from __future__ import annotations

import hashlib
import io
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from rigby_v2.errors import ArtifactIntegrityError
from rigby_v2.scenes import compile_scene, load_compiled_scene, load_object_pack


PACKS = (
    "grasp_place_block",
    "drawer",
    "lever_button",
    "hand_tool",
    "container_lid",
    "two_handed_object",
)


@pytest.mark.parametrize("pack_id", PACKS)
def test_certified_pack_loads_compiles_and_steps(pack_id: str) -> None:
    pack = load_object_pack(pack_id)
    compiled, model = load_compiled_scene(pack_id)
    assert pack.pack_id == compiled.pack_id == pack_id
    assert compiled.sha256 == compile_scene(pack_id).sha256
    assert model.nbody >= 49
    assert model.ngeom >= 49
    assert compiled.affordances
    assert compiled.state_predicates
    assert "<include" not in compiled.xml
    assert "C:\\" not in compiled.xml
    assert "../" not in compiled.xml
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_composed_mjcf_keeps_canonical_human_and_namespaces_scene_assets() -> None:
    compiled, model = load_compiled_scene("container_lid")
    body_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(model.nbody)
    }
    assert {"pelvis", "left_hand", "right_hand"} <= body_names
    assert {"obj__container__base", "obj__container__lid"} <= body_names
    joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    assert "pelvis_free" in joint_names
    assert "obj__container__free" in joint_names
    assert "obj__container__lid_hinge" in joint_names
    assert compiled.semantic_sites["container.lid_handle"] == "obj__container__lid_handle"


def test_articulated_pack_metadata_targets_real_joints_and_sites() -> None:
    for pack_id in ("drawer", "lever_button", "container_lid"):
        compiled, model = load_compiled_scene(pack_id)
        joint_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        }
        site_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index)
            for index in range(model.nsite)
        }
        for predicate in compiled.state_predicates:
            assert predicate["target"] in joint_names | site_names
        for affordance in compiled.affordances.values():
            assert affordance["site"] in site_names


def test_compiled_xml_is_human_readable_and_contains_machine_metadata() -> None:
    compiled = compile_scene("grasp_place_block")
    root = ET.fromstring(compiled.xml)
    assert "\n  <" in compiled.xml
    text = root.find("./custom/text[@name='rigby_scene_metadata']")
    assert text is not None
    assert "grasp_place_block" in text.attrib["data"]
    assert root.find("worldbody/body[@name='pelvis']") is not None
    assert root.find("worldbody/body[@name='obj__block__body']") is not None


def test_mjcf_roundtrips_through_mjspec_and_mjz() -> None:
    compiled = compile_scene("container_lid")
    source_spec = mujoco.MjSpec.from_string(compiled.xml, assets=compiled.assets)
    source_model = source_spec.compile()
    archived_spec = mujoco.from_zip(io.BytesIO(compiled.mjz_bytes))
    archived_model = archived_spec.compile()
    assert archived_model.nbody == source_model.nbody
    assert archived_model.njnt == source_model.njnt
    assert archived_model.ngeom == source_model.ngeom
    assert archived_model.nsite == source_model.nsite
    assert compiled.sha256 == compiled.mjz_sha256
    assert compiled.xml_sha256 != compiled.mjz_sha256


def test_native_mjz_is_deterministic_and_uses_fixed_zip_metadata() -> None:
    first = compile_scene("lever_button")
    second = compile_scene("lever_button")
    assert first.mjz_bytes == second.mjz_bytes
    assert first.mjz_sha256 == second.mjz_sha256
    with zipfile.ZipFile(io.BytesIO(first.mjz_bytes)) as archive:
        members = archive.infolist()
        assert len(members) == 1
        assert members[0].filename.endswith(".xml")
        # MuJoCo's native writer intentionally fixes ZIP timestamps, avoiding
        # wall-clock nondeterminism in an otherwise identical scene artifact.
        assert members[0].date_time == (1980, 1, 1, 0, 0, 0)


def test_load_model_uses_validated_mjz_not_loose_xml_or_asset_map() -> None:
    compiled = compile_scene("grasp_place_block")
    detached = replace(compiled, xml="<not-used/>", assets={})
    assert detached.load_model().nbody >= 49
    tampered = replace(compiled, mjz_bytes=compiled.mjz_bytes + b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        tampered.load_model()


def test_mjz_members_are_relative_and_contain_no_external_paths() -> None:
    compiled = compile_scene("two_handed_object")
    with zipfile.ZipFile(io.BytesIO(compiled.mjz_bytes)) as archive:
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            assert not normalized.startswith("/")
            assert "../" not in normalized
            assert ":/" not in normalized
            if name.endswith(".xml"):
                archived_xml = archive.read(name).decode("utf-8")
                assert "C:\\" not in archived_xml
                assert "../" not in archived_xml


def test_load_rejects_hash_valid_but_path_escaping_mjz() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("../scene.xml", "<mujoco/>")
    value = stream.getvalue()
    digest = hashlib.sha256(value).hexdigest()
    compiled = compile_scene("grasp_place_block")
    malicious = replace(
        compiled,
        mjz_bytes=value,
        mjz_sha256=digest,
        sha256=digest,
    )
    with pytest.raises(ArtifactIntegrityError, match="unsafe member path"):
        malicious.load_model()


def test_dynamic_pack_collisions_are_primitive_and_have_explicit_mass_and_friction() -> None:
    allowed = {
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    }
    for pack_id in ("grasp_place_block", "hand_tool", "container_lid", "two_handed_object"):
        pack = load_object_pack(pack_id)
        compiled, model = load_compiled_scene(pack_id)
        assert not compiled.assets
        for obj in pack.objects:
            for part in obj.parts:
                assert part.mass_kg > 0
                assert all(geom.type != "convex_mesh" for geom in part.geoms)
        object_geom_ids = [
            index
            for index in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith("obj__")
        ]
        assert object_geom_ids
        assert {mujoco.mjtGeom(int(model.geom_type[index])) for index in object_geom_ids} <= allowed
        assert np.all(model.geom_friction[object_geom_ids, 0] > 0)


def test_six_packs_cover_required_interaction_families() -> None:
    coverage = {
        "grasp_place_block": {"grasp", "place"},
        "drawer": {"pull"},
        "lever_button": {"rotate", "push"},
        "hand_tool": {"grasp", "strike"},
        "container_lid": {"open"},
        "two_handed_object": {"lift", "grasp"},
    }
    for pack_id, required in coverage.items():
        pack = load_object_pack(pack_id)
        actual = {affordance.kind for obj in pack.objects for affordance in obj.affordances}
        assert required <= actual
