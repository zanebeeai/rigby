from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .models import ClipResult, SceneManifest


GLTF_COMPONENT_FLOAT = 5126
GLTF_COMPONENT_UNSIGNED_SHORT = 5123


def _read_glb(path: Path) -> tuple[dict[str, Any], bytearray]:
    raw = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("asset is not a glTF 2.0 binary")
    offset = 12
    document: dict[str, Any] | None = None
    binary = bytearray()
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        payload = raw[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \x00"))
        elif kind == 0x004E4942:
            binary.extend(payload)
        offset += 8 + length
    if document is None:
        raise ValueError("GLB does not contain JSON")
    return document, binary


def _append(binary: bytearray, data: bytes) -> tuple[int, int]:
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    binary.extend(data)
    return offset, len(data)


def _add_accessor(
    document: dict[str, Any],
    binary: bytearray,
    array: np.ndarray,
    accessor_type: str,
    component_type: int,
    target: int | None = None,
    include_bounds: bool = False,
) -> int:
    array = np.ascontiguousarray(array)
    offset, byte_length = _append(binary, array.tobytes())
    view: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": byte_length}
    if target is not None:
        view["target"] = target
    document.setdefault("bufferViews", []).append(view)
    accessor: dict[str, Any] = {
        "bufferView": len(document["bufferViews"]) - 1,
        "componentType": component_type,
        "count": int(len(array)),
        "type": accessor_type,
    }
    if include_bounds:
        values = array if array.ndim > 1 else array.reshape(-1, 1)
        accessor["min"] = [float(v) for v in values.min(axis=0)]
        accessor["max"] = [float(v) for v in values.max(axis=0)]
    document.setdefault("accessors", []).append(accessor)
    return len(document["accessors"]) - 1


def _compose(rest: list[float] | None, delta: list[float]) -> list[float]:
    rest_q = rest or [0.0, 0.0, 0.0, 1.0]
    result = Rotation.from_quat(rest_q) * Rotation.from_quat(delta)
    return [float(v) for v in result.as_quat()]


def export_glb(
    clip: ClipResult,
    scene: SceneManifest,
    output_path: Path,
    asset_path: Path,
    rig_profile: dict[str, Any],
) -> Path:
    """Append local rest*delta animation and animated block geometry to the source GLB."""
    document, binary = _read_glb(asset_path)
    document.setdefault("asset", {}).setdefault("generator", "Rigby POC")
    document["asset"]["generator"] = "Rigby POC 0.1.0"
    document.setdefault("extras", {})["rigby"] = {
        "schema_version": clip.schema_version,
        "rotation_contract": "source_rest_local_xyzw * clip_local_delta_xyzw",
        "translation_contract": (
            "source_rest_local_xyz + app-space root delta transformed as [x,-z,y]"
        ),
        "provenance": clip.provenance.model_dump(mode="json"),
        "metrics": clip.metrics,
    }
    document.setdefault("buffers", [{"byteLength": 0}])
    nodes = document.setdefault("nodes", [])
    node_by_name = {node.get("name"): index for index, node in enumerate(nodes)}
    times = np.asarray([frame.time_s for frame in clip.frames], dtype=np.float32)
    time_accessor = _add_accessor(
        document, binary, times, "SCALAR", GLTF_COMPONENT_FLOAT, include_bounds=True
    )
    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []

    for canonical, source_name in rig_profile["bone_map"].items():
        node_index = node_by_name.get(source_name)
        if node_index is None or not clip.frames:
            continue
        rest = nodes[node_index].get("rotation")
        rotations = np.asarray(
            [
                _compose(rest, frame.bones.get(canonical).rotation.as_list())
                if canonical in frame.bones
                else (rest or [0.0, 0.0, 0.0, 1.0])
                for frame in clip.frames
            ],
            dtype=np.float32,
        )
        rotation_accessor = _add_accessor(document, binary, rotations, "VEC4", GLTF_COMPONENT_FLOAT)
        samplers.append({"input": time_accessor, "output": rotation_accessor, "interpolation": "LINEAR"})
        channels.append(
            {
                "sampler": len(samplers) - 1,
                "target": {"node": node_index, "path": "rotation"},
            }
        )
        if any(
            canonical in frame.bones
            and frame.bones[canonical].position is not None
            for frame in clip.frames
        ):
            rest_translation = np.asarray(
                nodes[node_index].get("translation", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
            translations = np.asarray(
                [
                    rest_translation
                    + np.asarray(
                        (
                            [
                                frame.bones[canonical].position.x,
                                -frame.bones[canonical].position.z,
                                frame.bones[canonical].position.y,
                            ]
                            if canonical == "hips"
                            and canonical in frame.bones
                            and frame.bones[canonical].position is not None
                            else frame.bones[canonical].position.as_list()
                            if canonical in frame.bones
                            and frame.bones[canonical].position is not None
                            else [0.0, 0.0, 0.0]
                        ),
                        dtype=np.float32,
                    )
                    for frame in clip.frames
                ],
                dtype=np.float32,
            )
            translation_accessor = _add_accessor(
                document,
                binary,
                translations,
                "VEC3",
                GLTF_COMPONENT_FLOAT,
            )
            samplers.append(
                {
                    "input": time_accessor,
                    "output": translation_accessor,
                    "interpolation": "LINEAR",
                }
            )
            channels.append(
                {
                    "sampler": len(samplers) - 1,
                    "target": {"node": node_index, "path": "translation"},
                }
            )

    vertices = np.asarray(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    indices = np.asarray(
        [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
         2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
        dtype=np.uint16,
    )
    position_accessor = _add_accessor(
        document, binary, vertices, "VEC3", GLTF_COMPONENT_FLOAT, target=34962, include_bounds=True
    )
    index_accessor = _add_accessor(
        document, binary, indices, "SCALAR", GLTF_COMPONENT_UNSIGNED_SHORT, target=34963
    )
    document.setdefault("materials", []).append(
        {"name": "RigbyBlock", "pbrMetallicRoughness": {"baseColorFactor": [0.18, 0.48, 0.9, 1.0]}}
    )
    document.setdefault("meshes", []).append(
        {
            "name": "RigbyBlockMesh",
            "primitives": [
                {
                    "attributes": {"POSITION": position_accessor},
                    "indices": index_accessor,
                    "material": len(document["materials"]) - 1,
                }
            ],
        }
    )
    for item in scene.objects:
        node_index = len(nodes)
        nodes.append(
            {
                "name": f"rigby_object_{item.id}",
                "mesh": len(document["meshes"]) - 1,
                "scale": item.dimensions_m.as_list(),
                "extras": {"object_id": item.id, "kind": item.kind},
            }
        )
        document["scenes"][document.get("scene", 0)].setdefault("nodes", []).append(node_index)
        translations = np.asarray(
            [frame.objects[item.id].translation.as_list() for frame in clip.frames], dtype=np.float32
        )
        rotations = np.asarray(
            [frame.objects[item.id].rotation.as_list() for frame in clip.frames], dtype=np.float32
        )
        trans_accessor = _add_accessor(document, binary, translations, "VEC3", GLTF_COMPONENT_FLOAT)
        rot_accessor = _add_accessor(document, binary, rotations, "VEC4", GLTF_COMPONENT_FLOAT)
        for path, accessor in (("translation", trans_accessor), ("rotation", rot_accessor)):
            samplers.append({"input": time_accessor, "output": accessor, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": node_index, "path": path}})

    document.setdefault("animations", []).append(
        {"name": "RigbyGeneratedMotion", "samplers": samplers, "channels": channels}
    )
    document["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    length = 12 + 8 + len(json_bytes) + 8 + len(binary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, length))
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(binary), 0x004E4942))
        handle.write(binary)
    return output_path
