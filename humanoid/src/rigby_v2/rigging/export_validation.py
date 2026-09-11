"""Export a simulated trace, reimport its GLB channels, and audit every frame."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from rigby_v2.certification.models import ExportValidation
from rigby_core.contracts import RigAssetManifestV1

from .coordinates import GltfPosition, GltfQuaternionXYZW
from .exporter import (
    _read_glb,
    _validated_visual_bindings,
    export_simulated_glb_bytes,
)
from .pose_adapter import (
    VisualBoneLocalTransform,
    VisualPhysicalPoseAdapter,
    VisualSkeletonPose,
)


_TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_FIXED_BONE_HIERARCHY: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        "upperChest": ("chest", ("leftShoulder", "rightShoulder")),
        "leftShoulder": ("upperChest", ("leftUpperArm",)),
        "rightShoulder": ("upperChest", ("rightUpperArm",)),
        "leftToes": ("leftFoot", ()),
        "rightToes": ("rightFoot", ()),
    }
)


@dataclass(frozen=True, slots=True)
class FixedVisualBoneAudit:
    """Evidence that a non-physical visual bone truthfully inherits its parent."""

    bone: str
    node_name: str
    parent_bone: str
    parent_node_name: str
    required_child_bones: tuple[str, ...]
    required_child_node_names: tuple[str, ...]
    independently_animated: bool


@dataclass(frozen=True, slots=True)
class GlbFrameRoundTripMetrics:
    """Physical endpoint discrepancies for one reimported GLB keyframe."""

    frame_index: int
    source_sample_index: int
    time_s: float
    fingertip_errors_m: Mapping[str, float]
    wrist_foot_errors_m: Mapping[str, float]
    max_fingertip_error_m: float
    max_wrist_foot_error_m: float
    max_local_rotation_error_rad: float
    passes_acceptance: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingertip_errors_m",
            MappingProxyType(dict(self.fingertip_errors_m)),
        )
        object.__setattr__(
            self,
            "wrist_foot_errors_m",
            MappingProxyType(dict(self.wrist_foot_errors_m)),
        )


@dataclass(frozen=True, slots=True)
class GlbRoundTripAudit:
    """Immutable evidence from a real export, GLB parse, and all-frame reimport."""

    profile: str
    exported_frame_count: int
    exported_glb_sha256: str
    frames: tuple[GlbFrameRoundTripMetrics, ...]
    fixed_bones: tuple[FixedVisualBoneAudit, ...]
    max_fingertip_error_m: float
    max_wrist_foot_error_m: float
    passes_acceptance: bool


def _float_accessor(
    document: dict[str, Any], binary: bytearray, accessor_id: int
) -> np.ndarray:
    accessor = document["accessors"][accessor_id]
    if accessor["componentType"] != 5126 or accessor["type"] not in _TYPE_WIDTH:
        raise ValueError("Exported animation accessor is not float data")
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("byteStride") is not None:
        raise ValueError("Strided animation accessors are unsupported")
    width = _TYPE_WIDTH[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    values = np.frombuffer(binary, dtype="<f4", count=count * width, offset=offset)
    return values.reshape(count, width).astype(np.float64)


def _parent_nodes(document: dict[str, Any]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for parent_id, node in enumerate(document.get("nodes", [])):
        for child in node.get("children", []):
            child_id = int(child)
            if child_id in parents:
                raise ValueError(f"Visual node {child_id} has multiple parents")
            parents[child_id] = parent_id
    return parents


def _audit_fixed_visual_bones(
    *,
    document: dict[str, Any],
    rig: RigAssetManifestV1,
    node_ids: Mapping[str, int],
    animated_node_ids: set[int],
) -> tuple[FixedVisualBoneAudit, ...]:
    declared = set(rig.fixed_visual_bones)
    expected = set(_FIXED_BONE_HIERARCHY)
    if declared != expected:
        raise ValueError(
            "Fixed inherited visual-bone declaration is incomplete; "
            f"missing={sorted(expected - declared)}, extra={sorted(declared - expected)}"
        )
    parents = _parent_nodes(document)
    nodes = document["nodes"]
    audits: list[FixedVisualBoneAudit] = []
    for bone, (parent_bone, child_bones) in _FIXED_BONE_HIERARCHY.items():
        node_id = node_ids[bone]
        expected_parent_id = node_ids[parent_bone]
        actual_parent_id = parents.get(node_id)
        if actual_parent_id != expected_parent_id:
            actual_name = (
                nodes[actual_parent_id].get("name")
                if actual_parent_id is not None
                else None
            )
            raise ValueError(
                f"Fixed visual bone {bone!r} does not inherit directly from "
                f"{parent_bone!r}; actual parent={actual_name!r}"
            )
        actual_children = {int(child) for child in nodes[node_id].get("children", [])}
        missing_children = [
            child for child in child_bones if node_ids[child] not in actual_children
        ]
        if missing_children:
            raise ValueError(
                f"Fixed visual bone {bone!r} is missing required inherited child "
                f"links: {missing_children}"
            )
        if node_id in animated_node_ids:
            raise ValueError(
                f"Fixed visual bone {bone!r} was independently animated despite "
                "having no physical DOF"
            )
        audits.append(
            FixedVisualBoneAudit(
                bone=bone,
                node_name=rig.visual_node_map[bone],
                parent_bone=parent_bone,
                parent_node_name=rig.visual_node_map[parent_bone],
                required_child_bones=child_bones,
                required_child_node_names=tuple(
                    rig.visual_node_map[child] for child in child_bones
                ),
                independently_animated=False,
            )
        )
    return tuple(audits)


def audit_simulated_glb_round_trip(
    *,
    source_glb: bytes,
    model_xml: str,
    rig: RigAssetManifestV1,
    times_s: np.ndarray,
    qpos: np.ndarray,
    physics_hz: int = 240,
    export_fps: int = 30,
) -> GlbRoundTripAudit:
    """Export and reimport every emitted GLB frame through the real asset path."""

    times = np.asarray(times_s, dtype=np.float64)
    positions = np.asarray(qpos, dtype=np.float64)
    exported = export_simulated_glb_bytes(
        source_glb=source_glb,
        model_xml=model_xml,
        rig=rig,
        times_s=times,
        qpos=positions,
        physics_hz=physics_hz,
        export_fps=export_fps,
    )
    source_document, _ = _read_glb(source_glb)
    document, binary = _read_glb(exported)
    model = mujoco.MjModel.from_xml_string(model_xml)
    node_ids, _ = _validated_visual_bindings(source_document, model, rig)
    if len(document.get("nodes", [])) != len(source_document.get("nodes", [])):
        raise ValueError("Export unexpectedly changed the source visual hierarchy")
    animation = document["animations"][-1]
    if animation.get("name") != "RigbyV2ActualSimulation":
        raise ValueError("Expected Rigby v2 animation was not reimported")

    channels: dict[tuple[int, str], np.ndarray] = {}
    channel_times: np.ndarray | None = None
    animated_node_ids: set[int] = set()
    for channel in animation["channels"]:
        sampler = animation["samplers"][channel["sampler"]]
        target = channel["target"]
        key = (int(target["node"]), str(target["path"]))
        if key in channels:
            raise ValueError(f"Export has duplicate animation channel {key}")
        values = _float_accessor(document, binary, int(sampler["output"]))
        sample_times = _float_accessor(
            document, binary, int(sampler["input"])
        ).reshape(-1)
        if len(values) != len(sample_times):
            raise ValueError(f"Animation channel {key} has mismatched times/values")
        if channel_times is None:
            channel_times = sample_times
        elif not np.array_equal(channel_times, sample_times):
            raise ValueError("Exported animation channels do not share exact key times")
        channels[key] = values
        animated_node_ids.add(key[0])
    if channel_times is None or len(channel_times) == 0:
        raise ValueError("Exported animation has no keyframes")
    if np.any(~np.isfinite(channel_times)) or np.any(np.diff(channel_times) <= 0.0):
        raise ValueError("Exported animation key times are not finite and increasing")

    if physics_hz <= 0 or export_fps <= 0 or physics_hz % export_fps:
        raise ValueError("export_fps must divide physics_hz")
    stride = physics_hz // export_fps
    source_indices = list(range(0, len(times), stride))
    if source_indices[-1] != len(times) - 1:
        source_indices.append(len(times) - 1)
    expected_times = times[source_indices]
    if len(channel_times) != len(expected_times) or not np.allclose(
        channel_times, expected_times, rtol=0.0, atol=5e-7
    ):
        raise ValueError("Reimported GLB key times do not match exported trace samples")

    profile = rig.body_size_profile
    if profile not in {"small", "medium", "large"}:
        raise ValueError("Custom profiles require a registered export adapter")
    adapter = VisualPhysicalPoseAdapter(profile)
    adapter_bones = set(adapter.manifest.skeleton_to_dofs)
    if set(rig.visual_skeleton_map) != adapter_bones:
        raise ValueError(
            "Rig visual skeleton does not exactly match the inverse adapter; "
            f"missing={sorted(adapter_bones - set(rig.visual_skeleton_map))}, "
            f"extra={sorted(set(rig.visual_skeleton_map) - adapter_bones)}"
        )
    fixed_audit = _audit_fixed_visual_bones(
        document=document,
        rig=rig,
        node_ids=node_ids,
        animated_node_ids=animated_node_ids,
    )

    source_nodes = source_document["nodes"]
    for bone in adapter.manifest.skeleton_to_dofs:
        node_id = node_ids[bone]
        rotations = channels.get((node_id, "rotation"))
        if rotations is None:
            raise ValueError(f"Export omitted visual rotation channel for {bone}")
    hips_id = node_ids["hips"]
    if (hips_id, "translation") not in channels:
        raise ValueError("Export omitted visual root translation channel")

    frame_metrics: list[GlbFrameRoundTripMetrics] = []
    for frame_index, (source_index, frame_time) in enumerate(
        zip(source_indices, channel_times, strict=True)
    ):
        transforms: dict[str, VisualBoneLocalTransform] = {}
        for bone in adapter.manifest.skeleton_to_dofs:
            node_id = node_ids[bone]
            source_rest = np.asarray(
                source_nodes[node_id].get("rotation", [0.0, 0.0, 0.0, 1.0]),
                dtype=np.float64,
            )
            rotation = channels[(node_id, "rotation")][frame_index]
            if not np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-5):
                raise ValueError(f"Reimported quaternion for {bone!r} is not normalized")
            delta = Rotation.from_quat(source_rest).inv() * Rotation.from_quat(
                rotation
            )
            x, y, z, w = delta.as_quat()
            translation = np.zeros(3, dtype=np.float64)
            if bone == "hips":
                source_translation = np.asarray(
                    source_nodes[node_id].get("translation", [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                )
                translation = (
                    channels[(node_id, "translation")][frame_index]
                    - source_translation
                )
            transforms[bone] = VisualBoneLocalTransform(
                translation=GltfPosition(*map(float, translation)),
                rotation=GltfQuaternionXYZW(
                    x=float(x), y=float(y), z=float(z), w=float(w)
                ),
            )
        reconstructed = adapter.visual_to_physical(
            VisualSkeletonPose(profile=profile, bone_local=transforms)
        )
        diagnostics = adapter.compare_world_kinematics(
            positions[source_index], reconstructed
        )
        metrics = GlbFrameRoundTripMetrics(
            frame_index=frame_index,
            source_sample_index=source_index,
            time_s=float(frame_time),
            fingertip_errors_m=diagnostics.fingertip_errors_m,
            wrist_foot_errors_m=diagnostics.wrist_foot_errors_m,
            max_fingertip_error_m=diagnostics.max_fingertip_error_m,
            max_wrist_foot_error_m=diagnostics.max_wrist_foot_error_m,
            max_local_rotation_error_rad=diagnostics.max_local_rotation_error_rad,
            passes_acceptance=diagnostics.passes_acceptance,
        )
        if not metrics.passes_acceptance:
            raise ValueError(
                f"Export/reimport frame {frame_index} at {frame_time:.6g}s exceeded "
                "endpoint thresholds: "
                f"fingertip={metrics.max_fingertip_error_m:.6g} m, "
                f"wrist/foot={metrics.max_wrist_foot_error_m:.6g} m"
            )
        frame_metrics.append(metrics)

    max_fingertip = max(
        (frame.max_fingertip_error_m for frame in frame_metrics), default=0.0
    )
    max_wrist_foot = max(
        (frame.max_wrist_foot_error_m for frame in frame_metrics), default=0.0
    )
    return GlbRoundTripAudit(
        profile=profile,
        exported_frame_count=len(frame_metrics),
        exported_glb_sha256=hashlib.sha256(exported).hexdigest(),
        frames=tuple(frame_metrics),
        fixed_bones=fixed_audit,
        max_fingertip_error_m=max_fingertip,
        max_wrist_foot_error_m=max_wrist_foot,
        passes_acceptance=all(frame.passes_acceptance for frame in frame_metrics),
    )


def validate_simulated_glb_round_trip(
    *,
    source_glb: bytes,
    model_xml: str,
    rig: RigAssetManifestV1,
    times_s: np.ndarray,
    qpos: np.ndarray,
) -> ExportValidation:
    """Compatibility wrapper returning certification's compact validation type."""

    try:
        audit = audit_simulated_glb_round_trip(
            source_glb=source_glb,
            model_xml=model_xml,
            rig=rig,
            times_s=times_s,
            qpos=qpos,
        )
        return ExportValidation(
            True,
            f"{audit.exported_frame_count} GLB frames; "
            f"max fingertip={audit.max_fingertip_error_m:.6g} m; "
            f"max wrist/foot={audit.max_wrist_foot_error_m:.6g} m",
        )
    except Exception as error:
        return ExportValidation(False, f"{type(error).__name__}: {error}")
