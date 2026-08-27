from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.rigging.exporter import _read_glb, export_trace_to_artifact
from rigby_v2.rigging.manifest_builder import stage_canonical_rig
from rigby_v2.simulation import (
    LinearKeyframeTrajectory,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
)
import pytest

pytestmark = pytest.mark.medium


def _accessor(document: dict, binary: bytearray, accessor_id: int) -> np.ndarray:
    accessor = document["accessors"][accessor_id]
    view = document["bufferViews"][accessor["bufferView"]]
    components = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    count = int(accessor["count"])
    return np.frombuffer(binary, dtype="<f4", count=count * components, offset=offset).reshape(
        count, components
    )


def test_actual_simulation_exports_to_visual_skeleton(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    staged = stage_canonical_rig("medium", artifacts=artifacts)
    assert staged.manifest.visual_asset is not None
    model_xml = artifacts.read_bytes(staged.manifest.mjcf).decode()
    model = mujoco.MjModel.from_xml_string(model_xml)
    start = model.qpos0.copy()
    target = start.copy()
    shoulder_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_flex"
    )
    shoulder_qpos = int(model.jnt_qposadr[shoulder_id])
    target[shoulder_qpos] = 0.35
    zeros = np.zeros(model.nv)
    trajectory = LinearKeyframeTrajectory(
        times_s=np.asarray([0.0, 0.1]),
        qpos=np.vstack((start, target)),
        qvel=np.vstack((zeros, zeros)),
    )
    simulated = NativeMujocoRuntime().simulate(
        SimulationRequest(
            request_id="export",
            model_xml=model_xml,
            trajectory=trajectory,
            config=SimulationConfig(duration_s=0.1),
            initial_qpos=start,
        )
    )
    exported = export_trace_to_artifact(
        artifacts=artifacts,
        source_glb_ref=staged.manifest.visual_asset,
        model_xml=model_xml,
        rig=staged.manifest,
        times_s=simulated.trace.times_s,
        qpos=simulated.trace.qpos,
    )

    document, binary = _read_glb(artifacts.read_bytes(exported.reference))
    animation = document["animations"][-1]
    assert animation["name"] == "RigbyV2ActualSimulation"
    assert animation["extras"]["source"] == "native_mujoco_actual_state"
    node_ids = {node.get("name"): index for index, node in enumerate(document["nodes"])}
    upper_arm_id = node_ids["upperarm_l"]
    rotation_channel = next(
        channel
        for channel in animation["channels"]
        if channel["target"] == {"node": upper_arm_id, "path": "rotation"}
    )
    sampler = animation["samplers"][rotation_channel["sampler"]]
    rotations = _accessor(document, binary, sampler["output"])
    assert rotations.shape[0] == exported.frame_count
    assert not np.allclose(rotations[0], rotations[-1])
    assert np.allclose(np.linalg.norm(rotations, axis=1), 1.0, atol=1e-5)
    assert artifacts.exists(exported.reference, verify=True)

