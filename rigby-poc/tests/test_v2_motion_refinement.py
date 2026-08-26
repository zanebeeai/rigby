from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rigby_v2.motion import (
    CandidateTrajectoryV1,
    CollisionDistanceObjective,
    HardJointAnchor,
    MotionCompilationError,
    MotionFailureReason,
    PinchDistanceObjective,
    SitePositionObjective,
    refine_joint_window,
)


HAND_XML = r"""
<mujoco model="joint_window_refinement">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size=".06" mass="2"/>
      <body name="upper_arm">
        <joint name="shoulder" axis="0 0 1" range="-1.2 1.2"/>
        <geom type="capsule" size=".025" fromto="0 0 0 .25 0 0" mass=".5"/>
        <body name="forearm" pos=".25 0 0">
          <joint name="elbow" axis="0 0 1" range="-1.2 1.2"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .22 0 0" mass=".4"/>
          <body name="hand" pos=".22 0 0">
            <joint name="wrist" axis="0 0 1" range="-.8 .8"/>
            <geom type="box" size=".07 .05 .02" pos=".07 0 0" mass=".2"/>
            <body name="thumb" pos=".12 -.045 0">
              <joint name="thumb_flex" axis="0 0 1" range="0 1.2"/>
              <geom name="thumb_geom" type="capsule" size=".012" fromto="0 0 0 .10 0 0" mass=".04"/>
              <site name="thumb_tip" pos=".10 0 0"/>
            </body>
            <body name="index" pos=".12 .045 0">
              <joint name="index_flex" axis="0 0 1" range="-1.2 0"/>
              <geom name="index_geom" type="capsule" size=".012" fromto="0 0 0 .10 0 0" mass=".04"/>
              <site name="index_tip" pos=".10 0 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="shoulder"/><motor joint="elbow"/><motor joint="wrist"/>
    <motor joint="thumb_flex"/><motor joint="index_flex"/>
  </actuator>
</mujoco>
"""


JOINTS = ("shoulder", "elbow", "wrist", "thumb_flex", "index_flex")


def _joint_qpos(model: mujoco.MjModel, name: str) -> int:
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[identifier])


def _candidate(model: mujoco.MjModel) -> CandidateTrajectoryV1:
    qpos = np.repeat(model.qpos0[None, :], 3, axis=0)
    return CandidateTrajectoryV1(
        candidate_id="neutral",
        program_hash="a" * 64,
        rig_hash="b" * 64,
        times_s=np.asarray([0.0, 0.5, 1.0]),
        qpos=qpos,
        qvel=np.zeros((3, model.nv)),
        qacc=np.zeros((3, model.nv)),
        quaternion_qpos_adrs=(3,),
    )


def _goal_sites(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, float]:
    data = mujoco.MjData(model)
    goal = {"shoulder": 0.25, "elbow": -0.4, "wrist": 0.2, "thumb_flex": 0.8, "index_flex": -0.8}
    for name, value in goal.items():
        data.qpos[_joint_qpos(model, name)] = value
    mujoco.mj_forward(model, data)
    thumb = data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "thumb_tip")
    ].copy()
    index = data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "index_tip")
    ].copy()
    return thumb, index, float(np.linalg.norm(thumb - index))


def test_joint_refinement_hits_fingertips_and_pinch_with_hard_anchor() -> None:
    model = mujoco.MjModel.from_xml_string(HAND_XML)
    candidate = _candidate(model)
    thumb, index, pinch = _goal_sites(model)
    refined = refine_joint_window(
        model,
        candidate,
        JOINTS,
        0,
        2,
        site_objectives=(
            SitePositionObjective("thumb_tip", (1,), np.asarray([thumb]), weight=30),
            SitePositionObjective("index_tip", (1,), np.asarray([index]), weight=30),
        ),
        pinch_objectives=(
            PinchDistanceObjective("thumb_tip", "index_tip", (1,), pinch, weight=20),
        ),
        hard_anchors=(HardJointAnchor(1, "wrist", 0.2),),
        temporal_weight=1e-5,
        deviation_weight=1e-8,
    )
    assert refined.qpos[1, _joint_qpos(model, "wrist")] == pytest.approx(0.2, abs=1e-12)
    assert np.array_equal(refined.qpos[0], candidate.qpos[0])
    assert np.array_equal(refined.qpos[2], candidate.qpos[2])

    data = mujoco.MjData(model)
    data.qpos[:] = refined.qpos[1]
    mujoco.mj_forward(model, data)
    thumb_actual = data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "thumb_tip")
    ]
    index_actual = data.site_xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "index_tip")
    ]
    assert np.linalg.norm(thumb_actual - thumb) < 0.003
    assert np.linalg.norm(index_actual - index) < 0.003
    assert abs(np.linalg.norm(thumb_actual - index_actual) - pinch) < 0.002
    assert np.all(np.isfinite(refined.qvel))
    assert refined.candidate_id != candidate.candidate_id


def test_unreachable_fingertip_target_returns_typed_infeasible() -> None:
    model = mujoco.MjModel.from_xml_string(HAND_XML)
    with pytest.raises(MotionCompilationError) as failure:
        refine_joint_window(
            model,
            _candidate(model),
            JOINTS,
            0,
            2,
            site_objectives=(
                SitePositionObjective(
                    "thumb_tip", (1,), np.asarray([[100.0, 100.0, 100.0]]), weight=10
                ),
            ),
            max_nfev=30,
        )
    assert failure.value.reason is MotionFailureReason.IK_INFEASIBLE
    assert failure.value.code.value == "infeasible"


def test_unreachable_collision_clearance_is_typed_infeasible() -> None:
    model = mujoco.MjModel.from_xml_string(HAND_XML)
    with pytest.raises(MotionCompilationError) as failure:
        refine_joint_window(
            model,
            _candidate(model),
            JOINTS,
            0,
            2,
            collision_objectives=(
                CollisionDistanceObjective(
                    "thumb_geom",
                    "index_geom",
                    (1,),
                    minimum_distance_m=0.5,
                    weight=50,
                    tolerance_m=0.001,
                ),
            ),
            temporal_weight=0.0,
            deviation_weight=0.0,
            max_nfev=30,
        )
    assert failure.value.reason is MotionFailureReason.IK_INFEASIBLE
    assert "collision" in str(failure.value.details)
