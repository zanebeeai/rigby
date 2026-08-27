from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rigby_v2.contracts import (
    ArtifactRefV1,
    ContactEdgeV2,
    CoordinateFrame,
    DofSpecV1,
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
    RigAssetManifestV1,
    TrackOwnership,
)
from rigby_v2.motion import (
    MotionAccentSpec,
    MotionCompilationError,
    MotionFailureReason,
    TimingProfile,
    compile_motion_program,
)

pytestmark = pytest.mark.medium


MODEL_XML = r"""
<mujoco model="motion_compiler_test">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="pelvis_geom" type="sphere" size=".08" mass="2"/>
      <body name="upper" pos="0 0 0">
        <joint name="shoulder" axis="0 0 1" range="-1.5 1.5"/>
        <geom name="upper_geom" type="capsule" size=".03" fromto="0 0 0 .3 0 0" mass="1"/>
        <body name="forearm" pos=".3 0 0">
          <joint name="elbow" axis="0 0 1" range="-1.5 1.5"/>
          <geom name="forearm_geom" type="capsule" size=".025" fromto="0 0 0 .25 0 0" mass=".7"/>
          <site name="fingertip" pos=".25 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="shoulder"/><motor joint="elbow"/></actuator>
</mujoco>
"""


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MODEL_XML)


def _rig() -> RigAssetManifestV1:
    return RigAssetManifestV1(
        rig_id="test-rig",
        mjcf=ArtifactRefV1(sha256="a" * 64, size_bytes=len(MODEL_XML)),
        dofs=(
            DofSpecV1(
                name="shoulder",
                joint="shoulder",
                minimum=-1.5,
                maximum=1.5,
                velocity_limit=50,
                effort_limit=40,
            ),
            DofSpecV1(
                name="elbow",
                joint="elbow",
                minimum=-1.5,
                maximum=1.5,
                velocity_limit=50,
                effort_limit=40,
            ),
        ),
        actuator_order=("shoulder", "elbow"),
        rest_qpos=tuple(float(value) for value in _model().qpos0),
    )


def _track(
    track_id: str,
    joint: str,
    start: float,
    end: float,
    *,
    ownership: TrackOwnership = TrackOwnership.EXCLUSIVE,
    priority: int = 0,
    times: tuple[float, float] = (0.0, 1.0),
) -> MotionTrackV2:
    return MotionTrackV2(
        track_id=track_id,
        target=joint,
        owner=track_id,
        ownership=ownership,
        priority=priority,
        keyframes=(
            MotionKeyframeV2(time_s=times[0], joint_values={joint: start}),
            MotionKeyframeV2(time_s=times[1], joint_values={joint: end}),
        ),
    )


def _program(
    tracks: tuple[MotionTrackV2, ...],
    *,
    contacts: tuple[ContactEdgeV2, ...] = (),
) -> MotionProgramV2:
    return MotionProgramV2(
        program_id="program",
        source_text="move test arm",
        duration_s=1,
        rig_id="test-rig",
        scene_id="test-scene",
        seed=1,
        phases=(
            MotionPhaseV2(
                phase_id="action", kind=PhaseKind.ACTION, start_s=0, end_s=1, energy=0.5
            ),
        ),
        tracks=tracks,
        contacts=contacts,
    )


def _qpos_address(model: mujoco.MjModel, joint: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return int(model.jnt_qposadr[joint_id])


def test_equal_priority_exclusive_overlap_is_typed_infeasible() -> None:
    program = _program((_track("one", "shoulder", 0, 1), _track("two", "shoulder", 0, -1)))
    with pytest.raises(MotionCompilationError) as failure:
        compile_motion_program(program, _model(), _rig())
    assert failure.value.reason is MotionFailureReason.UNRESOLVED_TRACK_OWNERSHIP


def test_priority_resolves_exclusive_and_additive_tracks_compose() -> None:
    model = _model()
    program = _program(
        (
            _track("low", "shoulder", 0, -1, priority=0),
            _track("high", "shoulder", 0, 1, priority=10),
            _track(
                "style",
                "shoulder",
                0,
                0.2,
                ownership=TrackOwnership.ADDITIVE,
            ),
            _track("elbow", "elbow", 0, -0.4),
        )
    )
    candidate = compile_motion_program(
        program, model, _rig(), timing_profile=TimingProfile.NEUTRAL
    )
    shoulder = _qpos_address(model, "shoulder")
    elbow = _qpos_address(model, "elbow")
    assert candidate.qpos[120, shoulder] == pytest.approx(0.6)
    assert candidate.qpos[120, elbow] == pytest.approx(-0.2)
    assert candidate.sample(0.503).qpos.shape == (model.nq,)
    assert candidate.sample(0.503).qvel.shape == (model.nv,)
    assert candidate.candidate_id == compile_motion_program(
        program, model, _rig(), timing_profile=TimingProfile.NEUTRAL
    ).candidate_id


def test_contact_plateau_remains_exact_through_energetic_retiming() -> None:
    model = _model()
    plateau = MotionTrackV2(
        track_id="contact-hold",
        target="shoulder",
        owner="arm",
        keyframes=tuple(
            MotionKeyframeV2(time_s=time, joint_values={"shoulder": value})
            for time, value in ((0, 0), (0.4, 1), (0.6, 1), (1, 0))
        ),
    )
    contact = ContactEdgeV2(
        contact_id="press",
        body_a="fingertip",
        body_b="button",
        start_s=0.4,
        end_s=0.6,
    )
    candidate = compile_motion_program(
        _program((plateau,), contacts=(contact,)),
        model,
        _rig(),
        timing_profile=TimingProfile.ENERGETIC,
    )
    shoulder = _qpos_address(model, "shoulder")
    mask = (candidate.times_s >= 0.4) & (candidate.times_s <= 0.6)
    assert np.allclose(candidate.qpos[mask, shoulder], 1.0, atol=1e-12)
    assert candidate.contact_plateaus[0].contact_id == "press"


def test_bounded_overshoot_and_rebound_move_joint_trajectory_but_not_anchors() -> None:
    model = _model()
    track = _track("accented", "shoulder", 0.0, 1.0)
    program = _program((track,))
    plain = compile_motion_program(program, model, _rig())
    accented = compile_motion_program(
        program,
        model,
        _rig(),
        accents=(MotionAccentSpec("accented", 0.12, 0.08),),
    )
    address = _qpos_address(model, "shoulder")
    difference = accented.qpos[:, address] - plain.qpos[:, address]
    assert np.max(difference) > 1e-3
    assert np.min(difference) < -1e-4
    assert accented.qpos[0, address] == plain.qpos[0, address]
    assert accented.qpos[-1, address] == plain.qpos[-1, address]
    assert np.max(np.abs(accented.qvel[:, model.jnt_dofadr[1]])) <= 50.0 + 1e-8


def test_unknown_dof_and_composed_limit_violation_are_typed() -> None:
    with pytest.raises(MotionCompilationError) as unknown:
        compile_motion_program(_program((_track("bad", "missing", 0, 1),)), _model(), _rig())
    assert unknown.value.reason is MotionFailureReason.UNKNOWN_DOF

    program = _program(
        (
            _track("base", "shoulder", 0, 1.4),
            _track(
                "add",
                "shoulder",
                0,
                0.3,
                ownership=TrackOwnership.ADDITIVE,
            ),
        )
    )
    with pytest.raises(MotionCompilationError) as limit:
        compile_motion_program(program, _model(), _rig())
    assert limit.value.reason is MotionFailureReason.JOINT_LIMIT_VIOLATION


@pytest.mark.parametrize(
    "interpolation", (InterpolationKind.SLERP, InterpolationKind.SQUAD)
)
def test_task_space_quintic_position_and_spherical_rotation_drive_real_ik(
    interpolation: InterpolationKind,
) -> None:
    model = _model()
    goal_data = mujoco.MjData(model)
    goal_data.qpos[_qpos_address(model, "shoulder")] = 0.25
    goal_data.qpos[_qpos_address(model, "elbow")] = -0.40
    mujoco.mj_forward(model, goal_data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fingertip")
    goal_position = goal_data.site_xpos[site_id].copy()
    convention = QuaternionConvention(
        order=QuaternionOrder.WXYZ,
        frame=CoordinateFrame.WORLD,
        meaning=QuaternionMeaning.ABSOLUTE,
    )
    goal_angle = -0.15
    task_track = MotionTrackV2(
        track_id="fingertip-world-pose",
        target="fingertip",
        owner="right-arm",
        interpolation=interpolation,
        keyframes=(
            MotionKeyframeV2(
                time_s=0.0,
                position={"x": 0.55, "y": 0.0, "z": 1.0},
                rotation=Quaternion(values=(1.0, 0.0, 0.0, 0.0), convention=convention),
            ),
            MotionKeyframeV2(
                time_s=1.0,
                position={
                    "x": float(goal_position[0]),
                    "y": float(goal_position[1]),
                    "z": float(goal_position[2]),
                },
                rotation=Quaternion(
                    values=(
                        float(np.cos(goal_angle / 2.0)),
                        0.0,
                        0.0,
                        float(np.sin(goal_angle / 2.0)),
                    ),
                    convention=convention,
                ),
            ),
        ),
    )
    candidate = compile_motion_program(
        _program((task_track,)), model, _rig(), sample_hz=12
    )

    assert abs(candidate.qpos[-1, _qpos_address(model, "shoulder")]) > 0.05
    actual = mujoco.MjData(model)
    actual.qpos[:] = candidate.qpos[-1]
    mujoco.mj_forward(model, actual)
    assert np.linalg.norm(actual.site_xpos[site_id] - goal_position) < 0.005
    target_rotation = np.asarray(
        [
            [np.cos(goal_angle), -np.sin(goal_angle), 0.0],
            [np.sin(goal_angle), np.cos(goal_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.linalg.norm(
        actual.site_xmat[site_id].reshape(3, 3) - target_rotation
    ) < 0.06
    assert not np.array_equal(candidate.qpos[0], candidate.qpos[-1])

    accented = compile_motion_program(
        _program((task_track,)),
        model,
        _rig(),
        sample_hz=12,
        accents=(MotionAccentSpec(task_track.track_id, 0.03, 0.02),),
    )
    assert not np.allclose(accented.qpos[1:-1], candidate.qpos[1:-1])
    assert np.allclose(accented.qpos[[0, -1]], candidate.qpos[[0, -1]], atol=2e-3)


def test_task_space_rotation_with_non_spherical_interpolation_is_typed_not_skipped() -> None:
    convention = QuaternionConvention(
        order=QuaternionOrder.WXYZ,
        frame=CoordinateFrame.WORLD,
        meaning=QuaternionMeaning.ABSOLUTE,
    )
    track = MotionTrackV2(
        track_id="bad-rotation",
        target="fingertip",
        owner="arm",
        interpolation=InterpolationKind.QUINTIC,
        keyframes=(
            MotionKeyframeV2(
                time_s=0.0,
                rotation=Quaternion(values=(1.0, 0.0, 0.0, 0.0), convention=convention),
            ),
            MotionKeyframeV2(
                time_s=1.0,
                rotation=Quaternion(values=(1.0, 0.0, 0.0, 0.0), convention=convention),
            ),
        ),
    )
    with pytest.raises(MotionCompilationError) as failure:
        compile_motion_program(_program((track,)), _model(), _rig(), sample_hz=12)
    assert failure.value.reason is MotionFailureReason.UNSUPPORTED_TRACK


def test_task_space_near_collisions_are_automatic_and_declared_contacts_are_excluded() -> None:
    blocked_xml = MODEL_XML.replace(
        "</worldbody>",
        '<geom name="blocking_obstacle" type="sphere" pos=".45 0 1" size=".08"/></worldbody>',
    )
    model = mujoco.MjModel.from_xml_string(blocked_xml)
    track = MotionTrackV2(
        track_id="blocked-reach",
        target="fingertip",
        owner="arm",
        keyframes=(
            MotionKeyframeV2(time_s=0.0, position={"x": 0.55, "y": 0.0, "z": 1.0}),
            MotionKeyframeV2(time_s=1.0, position={"x": 0.55, "y": 0.0, "z": 1.0}),
        ),
    )
    with pytest.raises(MotionCompilationError) as collision:
        compile_motion_program(_program((track,)), model, _rig(), sample_hz=8)
    assert collision.value.reason is MotionFailureReason.IK_INFEASIBLE

    declared = ContactEdgeV2(
        contact_id="authored-touch",
        body_a="forearm_geom",
        body_b="blocking_obstacle",
        start_s=0.0,
        end_s=1.0,
    )
    allowed = compile_motion_program(
        _program((track,), contacts=(declared,)), model, _rig(), sample_hz=8
    )
    assert len(allowed.times_s) == 9
