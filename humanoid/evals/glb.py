from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any


COMPONENT_FORMATS = {5126: "f"}
TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


@dataclass
class GlbCheck:
    valid: bool
    has_provenance: bool
    compared_samples: int
    max_translation_error_m: float | None
    max_rotation_error: float | None
    failures: list[str]


def _parse_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < 20:
        raise ValueError("GLB is shorter than its header")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        raise ValueError("invalid GLB 2.0 header")
    offset = 12
    document: dict[str, Any] | None = None
    binary = b""
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + length]
        offset += length
        if kind == 0x4E4F534A:
            document = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
        elif kind == 0x004E4942:
            binary = chunk
    if document is None:
        raise ValueError("GLB does not contain a JSON chunk")
    return document, binary


def _accessor(document: dict[str, Any], binary: bytes, index: int) -> list[list[float]]:
    accessor = document["accessors"][index]
    if accessor.get("sparse"):
        raise ValueError("sparse accessors are not supported by the independent verifier")
    component_type = accessor["componentType"]
    fmt = COMPONENT_FORMATS.get(component_type)
    width = TYPE_WIDTHS.get(accessor["type"])
    if fmt is None or width is None:
        raise ValueError("animation accessor is not FLOAT SCALAR/VEC2/VEC3/VEC4")
    view = document["bufferViews"][accessor["bufferView"]]
    item_size = struct.calcsize("<" + fmt * width)
    stride = int(view.get("byteStride", item_size))
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    output: list[list[float]] = []
    for item in range(int(accessor["count"])):
        output.append(list(struct.unpack_from("<" + fmt * width, binary, start + item * stride)))
    return output


def _clip_frames(clip: dict[str, Any]) -> list[dict[str, Any]]:
    frames = clip.get("frames")
    if isinstance(frames, list):
        return [frame for frame in frames if isinstance(frame, dict)]
    animation = clip.get("animation")
    if isinstance(animation, dict) and isinstance(animation.get("frames"), list):
        return [frame for frame in animation["frames"] if isinstance(frame, dict)]
    return []


def _frame_value(frame: dict[str, Any], node: str, path: str) -> list[float] | None:
    sources = [frame.get("bones"), frame.get("nodes"), frame.get("objects"), frame.get("transforms")]
    aliases = [path]
    if path == "translation":
        aliases += ["position", "position_m"]
    elif path == "rotation":
        aliases += ["quaternion"]
    for source in sources:
        if not isinstance(source, dict) or node not in source:
            continue
        transform = source[node]
        if not isinstance(transform, dict):
            continue
        for alias in aliases:
            value = transform.get(alias)
            if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                return [float(x) for x in value]
            if isinstance(value, dict):
                components = ("x", "y", "z", "w") if path == "rotation" else ("x", "y", "z")
                if all(isinstance(value.get(component), (int, float)) for component in components):
                    return [float(value[component]) for component in components]
    return None


def _distance(left: list[float], right: list[float], rotation: bool) -> float:
    direct = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    if not rotation:
        return direct
    negated = math.sqrt(sum((a + b) ** 2 for a, b in zip(left, right)))
    return min(direct, negated)


def _quat_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def check_glb(
    data: bytes,
    clip: dict[str, Any],
    *,
    translation_tolerance: float,
    rotation_tolerance: float,
    node_aliases: dict[str, str] | None = None,
) -> GlbCheck:
    failures: list[str] = []
    try:
        document, binary = _parse_glb(data)
    except (KeyError, ValueError, struct.error, json.JSONDecodeError) as exc:
        return GlbCheck(False, False, 0, None, None, [str(exc)])
    extras = document.get("extras", {})
    rigby_metadata = extras.get("rigby") if isinstance(extras, dict) else None
    provenance = rigby_metadata.get("provenance") if isinstance(rigby_metadata, dict) else None
    required_provenance = {
        "rig_id", "rig_asset", "compiler_version", "physics_engine", "physics_version",
        "planner_provider", "planner_model", "model_calls", "seed", "physics_model",
        "coordinate_frames",
    }
    has_provenance = isinstance(provenance, dict) and required_provenance <= set(provenance)
    if not has_provenance:
        missing = sorted(required_provenance - set(provenance or {}))
        failures.append(f"GLB metadata has incomplete Rigby provenance; missing {missing}")
    frames = _clip_frames(clip)
    if not frames:
        failures.append("clip has no raw frames for independent round-trip comparison")
        return GlbCheck(True, has_provenance, 0, None, None, failures)
    nodes = document.get("nodes", [])
    compared = 0
    max_translation = 0.0
    max_rotation = 0.0
    try:
        for animation in document.get("animations", []):
            for channel in animation.get("channels", []):
                target = channel.get("target", {})
                path = target.get("path")
                if path not in {"translation", "rotation"}:
                    continue
                node_index = target.get("node")
                if not isinstance(node_index, int) or node_index >= len(nodes):
                    continue
                node_name = nodes[node_index].get("name")
                if not isinstance(node_name, str):
                    continue
                clip_node = (node_aliases or {}).get(node_name, node_name)
                if node_name.startswith("rigby_object_"):
                    clip_node = node_name.removeprefix("rigby_object_")
                sampler = animation["samplers"][channel["sampler"]]
                values = _accessor(document, binary, sampler["output"])
                if len(values) != len(frames):
                    failures.append(
                        f"{node_name} {path} has {len(values)} samples; clip has {len(frames)} frames"
                    )
                    continue
                for frame, exported in zip(frames, values):
                    expected = _frame_value(frame, clip_node, path)
                    if expected is None or len(expected) != len(exported):
                        continue
                    if path == "rotation" and node_name in (node_aliases or {}):
                        expected = _quat_multiply(
                            [float(x) for x in nodes[node_index].get("rotation", [0.0, 0.0, 0.0, 1.0])],
                            expected,
                        )
                    elif path == "translation" and node_name in (node_aliases or {}):
                        delta = expected
                        translation_contract = (
                            rigby_metadata.get("translation_contract")
                            if isinstance(rigby_metadata, dict)
                            else None
                        )
                        if (
                            clip_node == "hips"
                            and isinstance(translation_contract, str)
                            and "[x,-z,y]" in translation_contract.replace(" ", "")
                        ):
                            delta = [expected[0], -expected[2], expected[1]]
                        rest_translation = [
                            float(x)
                            for x in nodes[node_index].get(
                                "translation", [0.0, 0.0, 0.0]
                            )
                        ]
                        expected = [
                            rest + offset
                            for rest, offset in zip(
                                rest_translation, delta, strict=True
                            )
                        ]
                    error = _distance(expected, exported, path == "rotation")
                    compared += 1
                    if path == "rotation":
                        max_rotation = max(max_rotation, error)
                    else:
                        max_translation = max(max_translation, error)
    except (KeyError, IndexError, ValueError, struct.error) as exc:
        failures.append(f"animation decode failed: {exc}")
    if compared == 0:
        failures.append("no named animation samples matched raw clip transforms")
    if max_translation > translation_tolerance:
        failures.append(f"translation error {max_translation:.8f} exceeds tolerance")
    if max_rotation > rotation_tolerance:
        failures.append(f"rotation error {max_rotation:.8f} exceeds tolerance")
    return GlbCheck(
        valid=True,
        has_provenance=has_provenance,
        compared_samples=compared,
        max_translation_error_m=max_translation if compared else None,
        max_rotation_error=max_rotation if compared else None,
        failures=failures,
    )
