from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.rigging.export_validation import audit_simulated_glb_round_trip
from rigby_v2.rigging.manifest_builder import stage_canonical_rig


EXPECTED_FIXED_HIERARCHY = {
    "upperChest": ("chest", ("leftShoulder", "rightShoulder")),
    "leftShoulder": ("upperChest", ("leftUpperArm",)),
    "rightShoulder": ("upperChest", ("rightUpperArm",)),
    "leftToes": ("leftFoot", ()),
    "rightToes": ("rightFoot", ()),
}


def _representative_trace(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Five exported poses exercising root, body, wrist, and every finger joint."""

    rest = np.asarray(model.qpos0, dtype=np.float64)
    keyframes = [rest.copy() for _ in range(5)]
    for key_index in range(1, 4):
        frame = keyframes[key_index]
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            address = int(model.jnt_qposadr[joint_id])
            lower, upper = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
            span = float(upper - lower)
            direction = 1.0 if (joint_id + key_index) % 2 else -1.0
            magnitude = (0.055 + 0.015 * key_index) * span
            frame[address] = np.clip(
                rest[address] + direction * magnitude,
                lower + 0.02 * span,
                upper - 0.02 * span,
            )
        frame[:3] += np.asarray(
            [0.008 * key_index, (-1.0) ** key_index * 0.006, 0.005 * key_index]
        )

    samples: list[np.ndarray] = []
    for left, right in zip(keyframes[:-1], keyframes[1:], strict=True):
        for step in range(8):
            amount = step / 8.0
            samples.append((1.0 - amount) * left + amount * right)
    samples.append(keyframes[-1])
    qpos = np.asarray(samples, dtype=np.float64)
    times = np.arange(len(qpos), dtype=np.float64) / 240.0

    nonfree_addresses = [
        int(model.jnt_qposadr[joint_id])
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    assert all(np.ptp(qpos[:, address]) > 1e-6 for address in nonfree_addresses)
    return times, qpos


@pytest.mark.parametrize("profile", ["small", "medium", "large"])
def test_real_glb_multiframe_round_trip_all_profiles(
    profile: str, tmp_path: Path
) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / profile / "artifacts")
    staged = stage_canonical_rig(profile, artifacts=artifacts)
    assert staged.manifest.visual_asset is not None
    model_xml = artifacts.read_bytes(staged.manifest.mjcf).decode("utf-8")
    source_glb = artifacts.read_bytes(staged.manifest.visual_asset)
    model = mujoco.MjModel.from_xml_string(model_xml)
    times, qpos = _representative_trace(model)

    audit = audit_simulated_glb_round_trip(
        source_glb=source_glb,
        model_xml=model_xml,
        rig=staged.manifest,
        times_s=times,
        qpos=qpos,
    )

    assert audit.profile == profile
    assert audit.exported_frame_count == 5
    assert audit.passes_acceptance
    assert len(audit.exported_glb_sha256) == 64
    assert audit.max_fingertip_error_m < 0.008
    assert audit.max_wrist_foot_error_m < 0.015
    assert [frame.source_sample_index for frame in audit.frames] == [0, 8, 16, 24, 32]
    for frame in audit.frames:
        assert len(frame.fingertip_errors_m) == 10
        assert set(frame.wrist_foot_errors_m) == {
            "leftWrist",
            "rightWrist",
            "leftFoot",
            "rightFoot",
        }
        assert frame.max_fingertip_error_m < 0.008
        assert frame.max_wrist_foot_error_m < 0.015

    fixed = {item.bone: item for item in audit.fixed_bones}
    assert set(fixed) == set(EXPECTED_FIXED_HIERARCHY)
    for bone, (parent, children) in EXPECTED_FIXED_HIERARCHY.items():
        assert fixed[bone].parent_bone == parent
        assert fixed[bone].required_child_bones == children
        assert fixed[bone].independently_animated is False


@pytest.mark.parametrize("missing_bone", ["leftIndexDistal", "upperChest"])
def test_glb_export_fails_closed_on_unmapped_required_visual_bone(
    missing_bone: str, tmp_path: Path
) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / missing_bone / "artifacts")
    staged = stage_canonical_rig("medium", artifacts=artifacts)
    assert staged.manifest.visual_asset is not None
    model_xml = artifacts.read_bytes(staged.manifest.mjcf).decode("utf-8")
    source_glb = artifacts.read_bytes(staged.manifest.visual_asset)
    model = mujoco.MjModel.from_xml_string(model_xml)
    times, qpos = _representative_trace(model)
    incomplete_map = dict(staged.manifest.visual_node_map)
    del incomplete_map[missing_bone]
    incomplete = staged.manifest.model_copy(update={"visual_node_map": incomplete_map})

    with pytest.raises(ValueError, match="omitted required visual bone mappings"):
        audit_simulated_glb_round_trip(
            source_glb=source_glb,
            model_xml=model_xml,
            rig=incomplete,
            times_s=times,
            qpos=qpos,
        )
