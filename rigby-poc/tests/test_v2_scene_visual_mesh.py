from __future__ import annotations

import hashlib
import io
import json
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import pytest

from rigby_v2.errors import FailureCode
from rigby_v2.scenes import SceneAssetError, compile_scene, load_object_pack


# A closed cube with an inward top-center vertex. It is intentionally concave
# and is valid render topology, but is not an authored convex collision piece.
CONCAVE_OBJ = b"""\
v -0.1 -0.1 0
v 0.1 -0.1 0
v 0.1 0.1 0
v -0.1 0.1 0
v -0.1 -0.1 0.1
v 0.1 -0.1 0.1
v 0.1 0.1 0.1
v -0.1 0.1 0.1
v 0 0 0.04
f 1 3 2
f 1 4 3
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
f 5 6 9
f 6 7 9
f 7 8 9
f 8 5 9
"""


def _tetra_stl() -> bytes:
    vertices = (
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.0, 0.1, 0.0),
        (0.0, 0.0, 0.1),
    )
    faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    output = bytearray(b"Rigby render-only STL".ljust(80, b"\0"))
    output.extend(struct.pack("<I", len(faces)))
    for face in faces:
        values = (0.0, 0.0, 0.0) + tuple(
            coordinate for index in face for coordinate in vertices[index]
        )
        output.extend(struct.pack("<12fH", *values, 0))
    return bytes(output)


TETRA_STL = _tetra_stl()


def _pack_data(asset_path: str, digest: str) -> dict:
    return {
        "schema_version": "1.0",
        "units": "meters_degrees_kilograms",
        "pack_id": "visual_pack",
        "description": "A primitive collider with a separate detailed render mesh.",
        "supports": [],
        "objects": [
            {
                "object_id": "object",
                "pos": [0, 0, 1],
                "dynamic": True,
                "articulated": False,
                "mass_kg": 1.0,
                "material": {
                    "name": "test",
                    "friction": [0.8, 0.02, 0.002],
                    "rgba": [0.2, 0.4, 0.7, 1.0],
                },
                "parts": [
                    {
                        "part_id": "root",
                        "mass_kg": 1.0,
                        "collision_kind": "primitive",
                        "geoms": [
                            {
                                "name": "collision_box",
                                "type": "box",
                                "size": [0.1, 0.1, 0.1],
                            }
                        ],
                        "visual_meshes": [
                            {
                                "name": "detailed_surface",
                                "render_only": True,
                                "asset_path": asset_path,
                                "asset_sha256": digest,
                            }
                        ],
                    }
                ],
                "sites": [],
                "affordances": [],
                "state_predicates": [],
            }
        ],
    }


def _write_pack(root: Path, data: dict, payload: bytes = CONCAVE_OBJ) -> Path:
    reference = data["objects"][0]["parts"][0]["visual_meshes"][0]
    relative = Path(*reference["asset_path"].split("/"))
    if ".." not in relative.parts and not relative.is_absolute():
        asset = root / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(payload)
    (root / "visual_pack.json").write_text(json.dumps(data), encoding="utf-8")
    return root / relative


def _content_address(payload: bytes, filename: str = "concave.obj") -> tuple[str, str]:
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256/{digest}/{filename}", digest


def test_raw_concave_visual_mesh_is_embedded_and_strictly_noncolliding(
    tmp_path: Path,
) -> None:
    asset_path, digest = _content_address(CONCAVE_OBJ)
    loose_asset = _write_pack(tmp_path, _pack_data(asset_path, digest))

    compiled = compile_scene("visual_pack", pack_root=tmp_path)
    model = compiled.load_model()
    assert compiled.render_only_geoms == (
        "obj__object__root__visual__detailed_surface",
    )
    render_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        compiled.render_only_geoms[0],
    )
    collision_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "obj__object__root__collision_box"
    )
    assert render_id >= 0 and collision_id >= 0
    assert int(model.geom_type[render_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    assert int(model.geom_contype[render_id]) == 0
    assert int(model.geom_conaffinity[render_id]) == 0
    assert int(model.geom_group[render_id]) == 2
    assert int(model.geom_contype[collision_id]) == 1
    assert int(model.geom_conaffinity[collision_id]) == 1
    object_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "obj__object__root"
    )
    assert float(model.body_mass[object_body_id]) == pytest.approx(1.0)
    xml_root = ET.fromstring(compiled.xml)
    render_xml = xml_root.find(
        ".//geom[@name='obj__object__root__visual__detailed_surface']"
    )
    assert render_xml is not None
    assert render_xml.attrib["mass"] == "0"
    assert render_xml.attrib["contype"] == "0"
    assert render_xml.attrib["conaffinity"] == "0"

    with zipfile.ZipFile(io.BytesIO(compiled.mjz_bytes)) as archive:
        assert archive.read(asset_path) == CONCAVE_OBJ

    # The authoritative MJZ remains self-contained after every loose source is gone.
    loose_asset.unlink()
    compiled.assets.clear()
    detached = compiled.load_model()
    detached_render_id = mujoco.mj_name2id(
        detached, mujoco.mjtObj.mjOBJ_GEOM, compiled.render_only_geoms[0]
    )
    assert detached_render_id >= 0
    assert int(detached.geom_contype[detached_render_id]) == 0
    assert int(detached.geom_conaffinity[detached_render_id]) == 0


def test_content_addressed_stl_uses_the_same_render_only_path(tmp_path: Path) -> None:
    asset_path, digest = _content_address(TETRA_STL, "surface.stl")
    _write_pack(tmp_path, _pack_data(asset_path, digest), TETRA_STL)

    compiled = compile_scene("visual_pack", pack_root=tmp_path)
    model = compiled.load_model()
    render_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, compiled.render_only_geoms[0]
    )
    assert render_id >= 0
    assert int(model.geom_contype[render_id]) == 0
    assert int(model.geom_conaffinity[render_id]) == 0
    with zipfile.ZipFile(io.BytesIO(compiled.mjz_bytes)) as archive:
        assert archive.read(asset_path) == TETRA_STL


def test_render_only_mesh_cannot_be_reclassified_as_collision(tmp_path: Path) -> None:
    asset_path, digest = _content_address(CONCAVE_OBJ)
    data = _pack_data(asset_path, digest)
    part = data["objects"][0]["parts"][0]
    part["collision_kind"] = "convex_decomposition"
    part["geoms"] = [
        {
            "name": "claimed_convex",
            "type": "convex_mesh",
            "asset_path": asset_path,
            "asset_sha256": digest,
        }
    ]
    _write_pack(tmp_path, data)

    with pytest.raises(SceneAssetError, match="cannot also be classified") as raised:
        compile_scene("visual_pack", pack_root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET


@pytest.mark.parametrize(
    ("field", "value"),
    (("render_only", False), ("contype", 1), ("collision", True)),
)
def test_visual_mesh_contract_cannot_enable_physics(
    tmp_path: Path, field: str, value: object
) -> None:
    asset_path, digest = _content_address(CONCAVE_OBJ)
    data = _pack_data(asset_path, digest)
    data["objects"][0]["parts"][0]["visual_meshes"][0][field] = value
    _write_pack(tmp_path, data)

    with pytest.raises(SceneAssetError) as raised:
        load_object_pack("visual_pack", root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../concave.obj",
        "/absolute/concave.obj",
        "sha256/" + "a" * 64 + "/../concave.obj",
        "sha256\\" + "a" * 64 + "\\concave.obj",
        "sha256/" + "a" * 64 + "/nested/concave.obj",
    ),
)
def test_visual_mesh_path_escape_is_typed_unsupported(
    tmp_path: Path, unsafe_path: str
) -> None:
    data = _pack_data(unsafe_path, "a" * 64)
    (tmp_path / "visual_pack.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SceneAssetError) as raised:
        compile_scene("visual_pack", pack_root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET


def test_visual_obj_external_reference_is_rejected_before_compile(
    tmp_path: Path,
) -> None:
    hostile = b"mtllib C:/outside/material.mtl\n" + CONCAVE_OBJ
    asset_path, digest = _content_address(hostile)
    _write_pack(tmp_path, _pack_data(asset_path, digest), hostile)

    with pytest.raises(SceneAssetError, match="external reference") as raised:
        compile_scene("visual_pack", pack_root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET


def test_visual_meshes_share_aggregate_asset_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rigby_v2.scenes.loader as loader

    asset_path, digest = _content_address(CONCAVE_OBJ)
    _write_pack(tmp_path, _pack_data(asset_path, digest))
    monkeypatch.setattr(loader, "MAX_TOTAL_MESH_ASSET_BYTES", len(CONCAVE_OBJ) - 1)

    with pytest.raises(SceneAssetError, match="aggregate") as raised:
        compile_scene("visual_pack", pack_root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET
