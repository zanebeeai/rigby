from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_core.contracts import ArtifactRefV1, RigAssetManifestV1
from .coordinates import (
    MujocoPosition,
    MujocoQuaternionWXYZ,
    mujoco_position_to_gltf,
    mujoco_quaternion_to_gltf,
)


GLTF_COMPONENT_FLOAT = 5126


@dataclass(frozen=True, slots=True)
class ExportedSimulation:
    reference: ArtifactRefV1
    frame_count: int
    source_trace_sha256: str


def _read_glb(raw: bytes) -> tuple[dict[str, Any], bytearray]:
    if len(raw) < 20:
        raise ValueError("GLB is truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(raw):
        raise ValueError("Asset is not a complete glTF 2.0 binary")
    offset = 12
    document: dict[str, Any] | None = None
    binary = bytearray()
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        payload = raw[offset + 8 : offset + 8 + length]
        if len(payload) != length:
            raise ValueError("GLB chunk is truncated")
        if kind == 0x4E4F534A:
            document = json.loads(payload.rstrip(b" \x00"))
        elif kind == 0x004E4942:
            binary.extend(payload)
        offset += 8 + length
    if document is None:
        raise ValueError("GLB does not contain a JSON chunk")
    return document, binary


def _append(binary: bytearray, payload: bytes) -> tuple[int, int]:
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    binary.extend(payload)
    return offset, len(payload)


def _add_float_accessor(
    document: dict[str, Any],
    binary: bytearray,
    values: np.ndarray,
    accessor_type: str,
    *,
    include_bounds: bool = False,
) -> int:
    array = np.ascontiguousarray(values, dtype=np.float32)
    byte_offset, byte_length = _append(binary, array.tobytes())
    document.setdefault("bufferViews", []).append(
        {"buffer": 0, "byteOffset": byte_offset, "byteLength": byte_length}
    )
    accessor: dict[str, Any] = {
        "bufferView": len(document["bufferViews"]) - 1,
        "componentType": GLTF_COMPONENT_FLOAT,
        "count": int(len(array)),
        "type": accessor_type,
    }
    if include_bounds:
        shaped = array if array.ndim > 1 else array.reshape(-1, 1)
        accessor["min"] = [float(value) for value in shaped.min(axis=0)]
        accessor["max"] = [float(value) for value in shaped.max(axis=0)]
    document.setdefault("accessors", []).append(accessor)
    return len(document["accessors"]) - 1


def _write_glb(document: dict[str, Any], binary: bytearray) -> bytes:
    document.setdefault("buffers", [{"byteLength": 0}])
    if len(document["buffers"]) != 1:
        raise ValueError("Rigby v2 export requires a single-buffer source GLB")
    document["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    output = bytearray(struct.pack("<4sII", b"glTF", 2, total))
    output.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    output.extend(json_bytes)
    output.extend(struct.pack("<II", len(binary), 0x004E4942))
    output.extend(binary)
    return bytes(output)


def _body_local_rotation(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    world = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    parent_id = int(model.body_parentid[body_id])
    if parent_id == 0:
        return world
    parent = np.asarray(data.xmat[parent_id], dtype=float).reshape(3, 3)
    return parent.T @ world


def _continuous_quaternions(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    for index in range(1, len(output)):
        if float(np.dot(output[index - 1], output[index])) < 0.0:
            output[index] *= -1.0
    return output


def _validated_visual_bindings(
    document: dict[str, Any],
    model: mujoco.MjModel,
    rig: RigAssetManifestV1,
) -> tuple[dict[str, int], dict[str, int]]:
    """Resolve every required simulated/fixed bone without silent omission."""

    nodes = document.get("nodes", [])
    named_nodes: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        name = node.get("name")
        if isinstance(name, str):
            named_nodes.setdefault(name, []).append(index)
    simulated = set(rig.visual_skeleton_map)
    fixed = set(rig.fixed_visual_bones)
    if not simulated:
        raise ValueError("Rig manifest has no required visual skeleton mapping")
    required = simulated | fixed
    missing_mapping = sorted(required - set(rig.visual_node_map))
    if missing_mapping:
        raise ValueError(
            f"Rig manifest omitted required visual bone mappings: {missing_mapping}"
        )
    missing_body = sorted(simulated - set(rig.visual_body_map))
    if missing_body:
        raise ValueError(
            f"Rig manifest omitted required physical visual bodies: {missing_body}"
        )
    duplicate_or_missing_nodes = {
        bone: rig.visual_node_map[bone]
        for bone in sorted(required)
        if len(named_nodes.get(rig.visual_node_map[bone], ())) != 1
    }
    if duplicate_or_missing_nodes:
        raise ValueError(
            "Required visual bones must resolve to exactly one source GLB node: "
            f"{duplicate_or_missing_nodes}"
        )
    node_ids = {
        bone: named_nodes[rig.visual_node_map[bone]][0] for bone in sorted(required)
    }
    body_ids = {
        bone: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, rig.visual_body_map[bone]
        )
        for bone in sorted(simulated)
    }
    unknown_bodies = sorted(bone for bone, body_id in body_ids.items() if body_id < 0)
    if unknown_bodies:
        raise ValueError(
            f"Required visual bones map to unknown MuJoCo bodies: {unknown_bodies}"
        )
    return node_ids, body_ids


def export_simulated_glb_bytes(
    *,
    source_glb: bytes,
    model_xml: str,
    rig: RigAssetManifestV1,
    times_s: np.ndarray,
    qpos: np.ndarray,
    physics_hz: int = 240,
    export_fps: int = 30,
) -> bytes:
    """Export actual simulated states as a glTF skeletal animation."""

    model = mujoco.MjModel.from_xml_string(model_xml)
    times = np.asarray(times_s, dtype=np.float64)
    positions = np.asarray(qpos, dtype=np.float64)
    if times.ndim != 1 or positions.shape != (len(times), model.nq):
        raise ValueError("Trace times/qpos do not match the compiled model")
    if len(times) < 2 or np.any(np.diff(times) <= 0) or np.any(~np.isfinite(positions)):
        raise ValueError("Trace must contain finite, strictly increasing samples")
    if physics_hz <= 0 or export_fps <= 0 or physics_hz % export_fps:
        raise ValueError("export_fps must divide physics_hz")
    stride = physics_hz // export_fps
    indices = list(range(0, len(times), stride))
    if indices[-1] != len(times) - 1:
        indices.append(len(times) - 1)
    sampled_times = times[indices]
    sampled_qpos = positions[indices]

    document, binary = _read_glb(source_glb)
    nodes = document.get("nodes", [])
    visual_node_ids, body_ids = _validated_visual_bindings(document, model, rig)
    time_accessor = _add_float_accessor(
        document, binary, sampled_times, "SCALAR", include_bounds=True
    )
    samplers: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []

    rest = mujoco.MjData(model)
    rest.qpos[:] = np.asarray(rig.rest_qpos, dtype=np.float64)
    mujoco.mj_forward(model, rest)
    frame_data = mujoco.MjData(model)
    rest_local = {
        bone: _body_local_rotation(model, rest, body_id)
        for bone, body_id in body_ids.items()
    }
    physical_frames: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    for frame in sampled_qpos:
        frame_data.qpos[:] = frame
        mujoco.mj_forward(model, frame_data)
        physical_frames.append(
            (
                np.asarray(frame_data.xpos[pelvis_id], dtype=float).copy(),
                {
                    bone: _body_local_rotation(model, frame_data, body_id)
                    for bone, body_id in body_ids.items()
                },
            )
        )

    for bone, source_node in rig.visual_node_map.items():
        body_id = body_ids.get(bone, -1)
        node_id = visual_node_ids.get(bone)
        if bone in rig.fixed_visual_bones:
            continue
        if body_id < 0 or node_id is None or bone not in rest_local:
            # _validated_visual_bindings should make this unreachable; keeping
            # the explicit guard prevents future refactors from restoring a
            # silent partial export.
            raise ValueError(f"Required visual bone {bone!r} is not exportable")
        source_rest = np.asarray(
            nodes[node_id].get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=float
        )
        rotations: list[np.ndarray] = []
        for _, local_by_bone in physical_frames:
            delta = rest_local[bone].T @ local_by_bone[bone]
            x, y, z, w = Rotation.from_matrix(delta).as_quat()
            gltf_delta = mujoco_quaternion_to_gltf(
                MujocoQuaternionWXYZ(w=float(w), x=float(x), y=float(y), z=float(z))
            )
            composed = Rotation.from_quat(source_rest) * Rotation.from_quat(
                gltf_delta.as_tuple()
            )
            rotations.append(composed.as_quat())
        rotation_array = _continuous_quaternions(np.asarray(rotations, dtype=np.float32))
        accessor = _add_float_accessor(document, binary, rotation_array, "VEC4")
        samplers.append(
            {"input": time_accessor, "output": accessor, "interpolation": "LINEAR"}
        )
        channels.append(
            {"sampler": len(samplers) - 1, "target": {"node": node_id, "path": "rotation"}}
        )

        if bone == "hips":
            source_translation = np.asarray(
                nodes[node_id].get("translation", [0.0, 0.0, 0.0]), dtype=np.float32
            )
            pelvis_rest = np.asarray(rest.xpos[pelvis_id], dtype=float)
            translations = []
            for pelvis_position, _ in physical_frames:
                delta = pelvis_position - pelvis_rest
                converted = mujoco_position_to_gltf(MujocoPosition(*map(float, delta)))
                translations.append(source_translation + converted.as_array())
            accessor = _add_float_accessor(
                document, binary, np.asarray(translations), "VEC3"
            )
            samplers.append(
                {"input": time_accessor, "output": accessor, "interpolation": "LINEAR"}
            )
            channels.append(
                {
                    "sampler": len(samplers) - 1,
                    "target": {"node": node_id, "path": "translation"},
                }
            )

    if not channels:
        raise ValueError("Rig manifest did not map any physical bodies to source GLB nodes")
    trace_hash = hashlib.sha256(
        np.ascontiguousarray(positions).tobytes()
        + np.ascontiguousarray(times).tobytes()
    ).hexdigest()
    document.setdefault("animations", []).append(
        {
            "name": "RigbyV2ActualSimulation",
            "samplers": samplers,
            "channels": channels,
            "extras": {
                "source": "native_mujoco_actual_state",
                "physics_hz": physics_hz,
                "export_fps": export_fps,
                "trace_sha256": trace_hash,
                "coordinate_contract": "mujoco_z_up_negative_y_forward_to_gltf_y_up_z_forward",
            },
        }
    )
    document.setdefault("asset", {})["generator"] = "Rigby v2"
    return _write_glb(document, binary)


def export_trace_to_artifact(
    *,
    artifacts: ContentAddressedArtifactStore,
    source_glb_ref: ArtifactRefV1,
    model_xml: str,
    rig: RigAssetManifestV1,
    times_s: np.ndarray,
    qpos: np.ndarray,
    filename: str = "animation.glb",
) -> ExportedSimulation:
    payload = export_simulated_glb_bytes(
        source_glb=artifacts.read_bytes(source_glb_ref),
        model_xml=model_xml,
        rig=rig,
        times_s=times_s,
        qpos=qpos,
    )
    reference = artifacts.put_bytes(
        payload, media_type="model/gltf-binary", filename=filename
    )
    source_hash = hashlib.sha256(
        np.ascontiguousarray(qpos).tobytes()
        + np.ascontiguousarray(times_s).tobytes()
    ).hexdigest()
    return ExportedSimulation(
        reference=reference,
        frame_count=(
            1
            + (len(times_s) - 1) // 8
            + int((len(times_s) - 1) % 8 != 0)
        ),
        source_trace_sha256=source_hash,
    )


def write_exported_glb(payload: bytes, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path
