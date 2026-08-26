from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from rigby_v2.errors import ArtifactIntegrityError, FailureCode
from rigby_v2.scenes import (
    SceneAssetError,
    compile_scene,
    load_compiled_scene,
    load_object_pack,
    resolve_content_asset,
)
from rigby_v2.scenes.models import ObjectPackSpec


def _valid_pack() -> dict:
    return {
        "schema_version": "1.0",
        "units": "meters_degrees_kilograms",
        "pack_id": "test_pack",
        "description": "A test object pack.",
        "supports": [],
        "objects": [
            {
                "object_id": "object",
                "pos": [0, 0, 1],
                "dynamic": True,
                "articulated": False,
                "mass_kg": 1.0,
                "material": {"name": "test", "friction": [0.8, 0.02, 0.002]},
                "parts": [
                    {
                        "part_id": "root",
                        "mass_kg": 1.0,
                        "collision_kind": "primitive",
                        "geoms": [
                            {"name": "shape", "type": "box", "size": [0.1, 0.1, 0.1]}
                        ],
                    }
                ],
                "sites": [
                    {
                        "name": "center",
                        "part": "root",
                        "kind": "grasp",
                        "pos": [0, 0, 0],
                    }
                ],
                "affordances": [
                    {
                        "name": "grasp",
                        "kind": "grasp",
                        "site": "center",
                        "allowed_effectors": ["right_palm"],
                    }
                ],
                "state_predicates": [
                    {
                        "name": "raised",
                        "target": "center",
                        "operator": "ge",
                        "values": [0.1],
                        "units": "meters",
                    }
                ],
            }
        ],
    }


def _write_pack(root: Path, data: dict) -> None:
    (root / f"{data['pack_id']}.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    "identifier", ("../canonical_human", "C:/escape", "bad\\path", "UPPER")
)
def test_pack_identifier_cannot_escape_or_select_arbitrary_files(
    tmp_path: Path, identifier: str
) -> None:
    with pytest.raises(SceneAssetError) as raised:
        load_object_pack(identifier, root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET


def test_articulated_object_without_joint_metadata_is_typed_unsupported_asset(
    tmp_path: Path,
) -> None:
    data = _valid_pack()
    data["objects"][0]["articulated"] = True
    _write_pack(tmp_path, data)
    with pytest.raises(SceneAssetError) as raised:
        load_object_pack("test_pack", root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET
    assert "invalid" in raised.value.message
    assert "joint metadata" in str(raised.value.details["error"])


@pytest.mark.parametrize(
    ("field", "value"),
    (("mass_kg", 0.0), ("mass_kg", -1.0)),
)
def test_dynamic_mass_is_strictly_positive(
    tmp_path: Path, field: str, value: float
) -> None:
    data = _valid_pack()
    data["objects"][0][field] = value
    data["objects"][0]["parts"][0][field] = value
    _write_pack(tmp_path, data)
    with pytest.raises(SceneAssetError) as raised:
        load_object_pack("test_pack", root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET


@pytest.mark.parametrize(
    "friction", ([0, 0, 0], [-0.1, 0.02, 0.002], [50, 0.02, 0.002])
)
def test_material_friction_is_bounded_and_physical(
    tmp_path: Path, friction: list[float]
) -> None:
    data = _valid_pack()
    data["objects"][0]["material"]["friction"] = friction
    _write_pack(tmp_path, data)
    with pytest.raises(SceneAssetError):
        load_object_pack("test_pack", root=tmp_path)


def test_raw_or_undeclared_mesh_collision_is_rejected(tmp_path: Path) -> None:
    data = _valid_pack()
    part = data["objects"][0]["parts"][0]
    part["geoms"] = [{"name": "mesh", "type": "convex_mesh"}]
    _write_pack(tmp_path, data)
    with pytest.raises(SceneAssetError) as raised:
        load_object_pack("test_pack", root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET


@pytest.mark.parametrize(
    "path",
    (
        "../piece.obj",
        "/absolute/piece.obj",
        "sha256/" + "a" * 64 + "/../piece.obj",
        "sha256\\" + "a" * 64 + "\\piece.obj",
        "sha256/" + "a" * 64 + "/nested/piece.obj",
    ),
)
def test_content_asset_path_must_be_exact_relative_address(
    tmp_path: Path, path: str
) -> None:
    with pytest.raises(SceneAssetError):
        resolve_content_asset(tmp_path, path, "a" * 64)


def test_content_asset_hash_is_verified_before_use(tmp_path: Path) -> None:
    content = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    digest = hashlib.sha256(content).hexdigest()
    directory = tmp_path / "sha256" / digest
    directory.mkdir(parents=True)
    path = directory / "piece.obj"
    path.write_bytes(content)
    assert (
        resolve_content_asset(tmp_path, f"sha256/{digest}/piece.obj", digest)
        == path.resolve()
    )
    path.write_bytes(content + b"# tampered")
    with pytest.raises(SceneAssetError, match="does not match"):
        resolve_content_asset(tmp_path, f"sha256/{digest}/piece.obj", digest)


def test_explicit_content_addressed_convex_piece_can_compile(tmp_path: Path) -> None:
    content = (
        b"v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
        b"f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
    )
    digest = hashlib.sha256(content).hexdigest()
    directory = tmp_path / "sha256" / digest
    directory.mkdir(parents=True)
    (directory / "tetra.obj").write_bytes(content)
    data = _valid_pack()
    part = data["objects"][0]["parts"][0]
    part["collision_kind"] = "convex_decomposition"
    part["geoms"] = [
        {
            "name": "convex_piece_0",
            "type": "convex_mesh",
            "asset_path": f"sha256/{digest}/tetra.obj",
            "asset_sha256": digest,
        }
    ]
    _write_pack(tmp_path, data)
    compiled, model = load_compiled_scene("test_pack", pack_root=tmp_path)
    repeated, _ = load_compiled_scene("test_pack", pack_root=tmp_path)
    assert compiled.assets == {f"sha256/{digest}/tetra.obj": content}
    assert compiled.mjz_bytes == repeated.mjz_bytes
    assert model.nmesh == 1
    with zipfile.ZipFile(io.BytesIO(compiled.mjz_bytes)) as archive:
        assert f"sha256/{digest}/tetra.obj" in archive.namelist()
        assert archive.read(f"sha256/{digest}/tetra.obj") == content
    # Loading is independent of the original filesystem and loose asset map.
    (directory / "tetra.obj").unlink()
    compiled.assets.clear()
    assert compiled.load_model().nmesh == 1


def test_unknown_manifest_fields_are_rejected(tmp_path: Path) -> None:
    data = _valid_pack()
    data["objects"][0]["plugin"] = "arbitrary_library.dll"
    _write_pack(tmp_path, data)
    with pytest.raises(SceneAssetError) as raised:
        load_object_pack("test_pack", root=tmp_path)
    assert "extra_forbidden" in str(raised.value.details["error"])


@pytest.mark.parametrize("units", (None, "centimeters_degrees_kilograms"))
def test_pack_requires_the_explicit_certified_unit_contract(
    tmp_path: Path, units: str | None
) -> None:
    data = _valid_pack()
    if units is None:
        data.pop("units")
    else:
        data["units"] = units
    _write_pack(tmp_path, data)
    with pytest.raises(SceneAssetError) as raised:
        load_object_pack("test_pack", root=tmp_path)
    assert raised.value.code is FailureCode.UNSUPPORTED_ASSET


@pytest.mark.parametrize(
    ("target", "value"),
    (
        ("object_position", [1_000_000.0, 0.0, 0.0]),
        ("collision_size", [1_000_000.0, 1_000_000.0, 1_000_000.0]),
    ),
)
def test_scene_dimensions_have_certified_meter_bounds(
    tmp_path: Path, target: str, value: list[float]
) -> None:
    data = _valid_pack()
    if target == "object_position":
        data["objects"][0]["pos"] = value
    else:
        data["objects"][0]["parts"][0]["geoms"][0]["size"] = value
    _write_pack(tmp_path, data)
    with pytest.raises(SceneAssetError):
        load_object_pack("test_pack", root=tmp_path)


@pytest.mark.parametrize("missing", ("sites", "affordances", "state_predicates"))
def test_articulated_pack_requires_complete_semantic_metadata(missing: str) -> None:
    data = load_object_pack("drawer").model_dump(mode="json")
    data["objects"][0][missing] = []
    with pytest.raises(
        ValueError, match="semantic sites, affordances, and state predicates"
    ):
        ObjectPackSpec.model_validate(data)


def test_pack_aggregate_geometry_budget_cannot_be_bypassed_in_memory() -> None:
    data = _valid_pack()
    obj = data["objects"][0]
    obj["mass_kg"] = 5.0
    children = []
    for part_index in range(4):
        children.append(
            {
                "part_id": f"child_{part_index}",
                "parent": "root",
                "mass_kg": 1.0,
                "collision_kind": "primitive",
                "geoms": [
                    {
                        "name": f"g_{part_index}_{geom_index}",
                        "type": "box",
                        "size": [0.001, 0.001, 0.001],
                    }
                    for geom_index in range(64)
                ],
            }
        )
    obj["parts"].extend(children)
    with pytest.raises(ValueError, match="aggregate complexity"):
        ObjectPackSpec.model_validate(data)


def test_obj_payload_cannot_reference_external_materials(tmp_path: Path) -> None:
    content = (
        b"mtllib C:/outside/material.mtl\n"
        b"v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
        b"f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
    )
    digest = hashlib.sha256(content).hexdigest()
    directory = tmp_path / "sha256" / digest
    directory.mkdir(parents=True)
    (directory / "tetra.obj").write_bytes(content)
    data = _valid_pack()
    part = data["objects"][0]["parts"][0]
    part["collision_kind"] = "convex_decomposition"
    part["geoms"] = [
        {
            "name": "piece",
            "type": "convex_mesh",
            "asset_path": f"sha256/{digest}/tetra.obj",
            "asset_sha256": digest,
        }
    ]
    _write_pack(tmp_path, data)
    with pytest.raises(SceneAssetError, match="external reference"):
        load_compiled_scene("test_pack", pack_root=tmp_path)


def _archive(*members: tuple[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("xml", "message"),
    (
        (
            b"<mujoco><extension><plugin plugin='unsafe'/></extension></mujoco>",
            "plugins",
        ),
        (
            b"<mujoco><asset><mesh name='m' file='C:/outside.obj'/></asset></mujoco>",
            "non-embedded asset",
        ),
    ),
)
def test_hash_valid_mjz_rejects_plugins_and_external_assets(
    xml: bytes, message: str
) -> None:
    value = _archive(("scene.xml", xml))
    digest = hashlib.sha256(value).hexdigest()
    hostile = replace(
        compile_scene("grasp_place_block"),
        mjz_bytes=value,
        mjz_sha256=digest,
        sha256=digest,
    )
    with pytest.raises(ArtifactIntegrityError, match=message):
        hostile.load_model()


def test_hash_valid_mjz_rejects_duplicate_members() -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        value = _archive(
            ("scene.xml", b"<mujoco/>"),
            ("asset.obj", b"one"),
            ("asset.obj", b"two"),
        )
    digest = hashlib.sha256(value).hexdigest()
    hostile = replace(
        compile_scene("grasp_place_block"),
        mjz_bytes=value,
        mjz_sha256=digest,
        sha256=digest,
    )
    with pytest.raises(ArtifactIntegrityError, match="duplicate"):
        hostile.load_model()


def test_hash_valid_mjz_enforces_uncompressed_budget(monkeypatch) -> None:
    import rigby_v2.scenes.compiler as compiler

    baseline = compile_scene("grasp_place_block")
    monkeypatch.setattr(compiler, "MAX_MJZ_UNCOMPRESSED_BYTES", 32)
    value = _archive(("scene.xml", b"<mujoco>" + b" " * 64 + b"</mujoco>"))
    digest = hashlib.sha256(value).hexdigest()
    hostile = replace(
        baseline,
        mjz_bytes=value,
        mjz_sha256=digest,
        sha256=digest,
    )
    with pytest.raises(ArtifactIntegrityError, match="uncompressed size"):
        hostile.load_model()
