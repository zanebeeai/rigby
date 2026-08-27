from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.contracts import (
    ContactEdgeV2,
    CoordinateFrame,
    InterpolationKind,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    Quaternion,
    QuaternionConvention,
    QuaternionMeaning,
    QuaternionOrder,
    Vec3,
)
from rigby_v2.flywheel.schemas import PathFamily, SemanticPlanV1
from rigby_v2.hashing import content_hash
from rigby_v2.motion import TimingProfile, compile_motion_program
from rigby_v2.rigging import stage_canonical_rig
from rigby_v2.selection import (
    DeterministicCandidateCompiler,
    generate_candidate_set,
    variation_for_attempt,
)
import pytest

pytestmark = pytest.mark.medium


_CONVENTION = QuaternionConvention(
    order=QuaternionOrder.WXYZ,
    frame=CoordinateFrame.WORLD,
    meaning=QuaternionMeaning.ABSOLUTE,
)


def _site_pose(
    model: mujoco.MjModel, qpos: np.ndarray, site_name: str
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    rotation = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(rotation, data.site_xmat[site_id])
    return data.site_xpos[site_id].copy(), rotation


def _keyframe(
    time_s: float,
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    hard: bool,
) -> MotionKeyframeV2:
    return MotionKeyframeV2(
        time_s=time_s,
        position=Vec3(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        ),
        rotation=Quaternion(
            values=tuple(float(value) for value in rotation),
            convention=_CONVENTION,
        ),
        hard=hard,
    )


def _canonical_task_plan(
    model: mujoco.MjModel, rig_id: str
) -> SemanticPlanV1:
    rest = model.qpos0.copy()
    reached = rest.copy()
    for joint_name, value in (
        ("right_shoulder_flex", 0.10),
        ("right_elbow_flex", 0.18),
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        reached[int(model.jnt_qposadr[joint_id])] = value
    start_position, start_rotation = _site_pose(model, rest, "right_palm")
    end_position, end_rotation = _site_pose(model, reached, "right_palm")
    program = MotionProgramV2(
        program_id="canonical-palm-task-route",
        source_text="Reach with the right palm, hold contact, then settle.",
        duration_s=1.2,
        rig_id=rig_id,
        scene_id="selection-task-space-fixture",
        seed=17,
        phases=(
            MotionPhaseV2(
                phase_id="reach",
                kind=PhaseKind.ACTION,
                start_s=0.0,
                end_s=1.2,
                energy=0.5,
            ),
        ),
        tracks=(
            MotionTrackV2(
                track_id="right-palm-route",
                target="right_palm",
                owner="right_arm",
                interpolation=InterpolationKind.SLERP,
                keyframes=(
                    _keyframe(
                        0.0, start_position, start_rotation, hard=True
                    ),
                    _keyframe(1.2, end_position, end_rotation, hard=True),
                ),
            ),
        ),
        contacts=(
            ContactEdgeV2(
                contact_id="palm-button",
                body_a="right_hand",
                body_b="button",
                start_s=0.4,
                end_s=0.8,
            ),
        ),
    )
    return SemanticPlanV1(
        plan_id="canonical-palm-selection-plan",
        prompt=program.source_text,
        base_program=program,
        retrieval_release="release-N",
    )


def _route_signature(program: MotionProgramV2) -> str:
    return content_hash(
        [
            keyframe.model_dump(mode="json")
            for keyframe in program.tracks[0].keyframes
        ]
    )


def _anchor_signature(program: MotionProgramV2, time_s: float) -> str:
    keyframe = next(
        keyframe
        for keyframe in program.tracks[0].keyframes
        if abs(keyframe.time_s - time_s) <= 1e-10
    )
    return content_hash(keyframe.model_dump(mode="json"))


def test_all_six_axes_change_task_space_motion_without_moving_anchors(
    tmp_path: Path,
) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    model = mujoco.MjModel.from_xml_string(
        artifacts.read_bytes(rig.manifest.mjcf).decode("utf-8")
    )
    plan = _canonical_task_plan(model, rig.manifest.rig_id)
    compiler = DeterministicCandidateCompiler()
    baseline_variation = variation_for_attempt(plan, 1)
    baseline = compiler.compile(plan, baseline_variation, 1)
    isolated = {
        "retrieval_seed": baseline_variation.model_copy(
            update={"retrieval_seed": baseline_variation.retrieval_seed + 1}
        ),
        "path_family": baseline_variation.model_copy(
            update={"path_family": PathFamily.ARC}
        ),
        "timing_style": baseline_variation.model_copy(
            update={"timing_style": "relaxed"}
        ),
        "energy": baseline_variation.model_copy(
            update={"energy": baseline_variation.energy + 0.1}
        ),
        "body_participation": baseline_variation.model_copy(
            update={"body_participation": ("whole_body", "hands")}
        ),
        "contact_strategy": baseline_variation.model_copy(
            update={"contact_strategy": "task_normal_aligned"}
        ),
    }

    for variation in isolated.values():
        varied = compiler.compile(plan, variation, 1)
        assert _route_signature(varied) != _route_signature(baseline)
        for anchor_s in (0.0, 0.4, 0.8, 1.2):
            assert _anchor_signature(varied, anchor_s) == _anchor_signature(
                baseline, anchor_s
            )


def test_canonical_task_plan_compiles_exactly_five_physical_trajectory_routes(
    tmp_path: Path,
) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    model = mujoco.MjModel.from_xml_string(
        artifacts.read_bytes(rig.manifest.mjcf).decode("utf-8")
    )
    plan = _canonical_task_plan(model, rig.manifest.rig_id)
    generated = generate_candidate_set(plan)

    assert len(generated.candidates) == 5
    assert generated.total_compile_attempts == 5
    assert generated.repair_rounds == 0
    assert {candidate.variation.path_family for candidate in generated.candidates} == set(
        PathFamily
    )
    compiled_fingerprints: set[str] = set()
    contact_motion_fingerprints: set[str] = set()
    for proposal in generated.candidates:
        # The broad same-root declaration isolates candidate-route compilation;
        # collision objective behavior has its own focused tests.
        compiled = compile_motion_program(
            proposal.program,
            model,
            rig.manifest,
            sample_hz=5,
            timing_profile=TimingProfile(proposal.variation.timing_style),
            allowed_contact_pairs=(("pelvis", "pelvis"),),
        )
        compiled_fingerprints.add(
            content_hash(
                {
                    "times_s": compiled.times_s.tolist(),
                    "qpos": np.round(compiled.qpos, decimals=10).tolist(),
                }
            )
        )
        contact_samples = (
            (compiled.times_s > 0.4 + 1e-10)
            & (compiled.times_s < 0.8 - 1e-10)
        )
        assert np.any(contact_samples)
        contact_motion_fingerprints.add(
            content_hash(
                np.round(compiled.qpos[contact_samples], decimals=10).tolist()
            )
        )
        assert tuple(
            (plateau.contact_id, plateau.start_s, plateau.end_s)
            for plateau in compiled.contact_plateaus
        ) == (("palm-button", 0.4, 0.8),)

    assert len(compiled_fingerprints) == 5
    assert len(contact_motion_fingerprints) == 5

