from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import os

import mujoco
import numpy as np

from rigby_v2.simulation import (
    PHYSICS_HZ,
    ConstantTarget,
    ControlTarget,
    LinearKeyframeTrajectory,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationFailureCode,
    SimulationRequest,
    SimulationStatus,
    repeat_replay,
)
import pytest

pytestmark = pytest.mark.medium


# A deliberately small articulated biped used only as a runtime smoke model.
# The pelvis is a true free root; legs, feet, arm, and forearm are physical
# bodies.  Nothing is welded, mocap-driven, or represented by a Cartesian proxy.
ARTICULATED_BIPED_XML = r"""
<mujoco model="rigby_v2_articulated_biped_smoke">
  <compiler angle="radian" autolimits="true" inertiafromgeom="true"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" iterations="80" tolerance="1e-10"/>
  <default>
    <joint damping="2" armature="0.02"/>
    <geom condim="3" friction="0.9 0.01 0.001" solref="0.01 1"/>
    <motor ctrllimited="true" ctrlrange="-80 80"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="pelvis" pos="0 0 0.61">
      <freejoint name="root"/>
      <geom name="pelvis_geom" type="box" size="0.18 0.10 0.09" mass="5"/>
      <geom name="torso_geom" type="box" size="0.16 0.09 0.20"
            pos="0 0 0.25" mass="8"/>
      <body name="left_thigh" pos="0.10 0 0">
        <joint name="left_hip" type="hinge" axis="0 1 0" range="-0.7 0.7"/>
        <geom name="left_thigh_geom" type="capsule" size="0.045"
              fromto="0 0 0 0 0 -0.28" mass="1.8"/>
        <body name="left_shin" pos="0 0 -0.28">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="-0.1 1.4"/>
          <geom name="left_shin_geom" type="capsule" size="0.04"
                fromto="0 0 0 0 0 -0.28" mass="1.2"/>
          <body name="left_foot" pos="0 0 -0.28">
            <geom name="left_foot_geom" type="box" size="0.075 0.14 0.025"
                  pos="0 0.05 -0.025" mass="0.6"/>
          </body>
        </body>
      </body>
      <body name="right_thigh" pos="-0.10 0 0">
        <joint name="right_hip" type="hinge" axis="0 1 0" range="-0.7 0.7"/>
        <geom name="right_thigh_geom" type="capsule" size="0.045"
              fromto="0 0 0 0 0 -0.28" mass="1.8"/>
        <body name="right_shin" pos="0 0 -0.28">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="-0.1 1.4"/>
          <geom name="right_shin_geom" type="capsule" size="0.04"
                fromto="0 0 0 0 0 -0.28" mass="1.2"/>
          <body name="right_foot" pos="0 0 -0.28">
            <geom name="right_foot_geom" type="box" size="0.075 0.14 0.025"
                  pos="0 0.05 -0.025" mass="0.6"/>
          </body>
        </body>
      </body>
      <body name="upper_arm" pos="0.18 0 0.36">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom name="upper_arm_geom" type="capsule" size="0.035"
              fromto="0 0 0 0.24 0 0" mass="0.8"/>
        <body name="forearm" pos="0.24 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom name="forearm_geom" type="capsule" size="0.03"
                fromto="0 0 0 0.22 0 0" mass="0.55"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="left_hip_motor" joint="left_hip"/>
    <motor name="left_knee_motor" joint="left_knee"/>
    <motor name="right_hip_motor" joint="right_hip"/>
    <motor name="right_knee_motor" joint="right_knee"/>
    <motor name="shoulder_motor" joint="shoulder"/>
    <motor name="elbow_motor" joint="elbow"/>
  </actuator>
</mujoco>
"""


def _joint_qpos_adr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id])


def _request(*, request_id: str = "smoke", shoulder_target: float = 0.28) -> SimulationRequest:
    model = mujoco.MjModel.from_xml_string(ARTICULATED_BIPED_XML)
    qpos_start = model.qpos0.copy()
    qpos_end = qpos_start.copy()
    qpos_end[_joint_qpos_adr(model, "shoulder")] = shoulder_target
    qvel_start = np.zeros(model.nv, dtype=np.float64)
    qvel_end = np.zeros(model.nv, dtype=np.float64)
    trajectory = LinearKeyframeTrajectory(
        times_s=np.asarray([0.0, 0.20]),
        qpos=np.vstack((qpos_start, qpos_end)),
        qvel=np.vstack((qvel_start, qvel_end)),
    )
    return SimulationRequest(
        model_xml=ARTICULATED_BIPED_XML,
        trajectory=trajectory,
        config=SimulationConfig(duration_s=0.20),
        initial_qpos=qpos_start,
        request_id=request_id,
    )


def test_native_runtime_records_actual_free_root_state_controls_and_contacts() -> None:
    request = _request()
    result = NativeMujocoRuntime().simulate(request)

    assert result.status is SimulationStatus.COMPLETED
    assert result.failure is None
    assert result.metrics is not None and result.metrics.finite
    assert result.metrics.physics_steps == 48
    assert result.diagnostics["physics_hz"] == PHYSICS_HZ
    assert result.diagnostics["qpos_writes_after_initialization"] == 0
    assert result.diagnostics["task_success_evaluated"] is False
    assert not hasattr(result, "success")
    assert result.trace.qpos.shape == (49, 13)
    assert result.trace.qvel.shape == (49, 12)
    assert result.trace.ctrl.shape == (49, 6)
    assert np.allclose(np.diff(result.trace.times_s), 1.0 / PHYSICS_HZ, atol=1e-14)
    assert np.max(np.abs(result.trace.ctrl)) <= 80.0
    assert result.metrics.contact_sample_count > 0

    model = mujoco.MjModel.from_xml_string(ARTICULATED_BIPED_XML)
    shoulder_adr = _joint_qpos_adr(model, "shoulder")
    # Actual integration lags the requested keyframe; the trace is not a copy
    # of the planned qpos and the articulated joint genuinely moves.
    requested_first = 0.28 * (1.0 / PHYSICS_HZ) / 0.20
    assert not np.isclose(result.trace.qpos[1, shoulder_adr], requested_first)
    assert abs(result.trace.qpos[-1, shoulder_adr]) > 0.01


def test_repeat_replay_compares_state_control_and_contact_traces() -> None:
    report = repeat_replay(NativeMujocoRuntime(), _request(), runs=3)

    assert report.deterministic
    assert len(report.runs) == 3
    assert len(report.comparisons) == 2
    assert all(comparison.matches for comparison in report.comparisons)
    assert all(comparison.max_qpos_error == 0.0 for comparison in report.comparisons)
    assert all(comparison.contact_topology_matches for comparison in report.comparisons)


def test_batch_uses_independent_traces_and_preserves_input_order() -> None:
    requests = [
        _request(request_id="candidate-0", shoulder_target=0.15),
        _request(request_id="candidate-1", shoulder_target=0.30),
        _request(request_id="candidate-2", shoulder_target=-0.20),
    ]
    results = NativeMujocoRuntime().simulate_batch(requests, max_workers=3)

    assert [result.request_id for result in results] == [request.request_id for request in requests]
    assert all(result.completed for result in results)
    assert all(
        result.diagnostics["execution_backend"] == "closed_loop_process_compatible"
        for result in results
    )
    assert all(result.diagnostics["process_id"] != os.getpid() for result in results)
    assert not np.shares_memory(results[0].trace.qpos, results[1].trace.qpos)
    assert not np.array_equal(results[0].trace.qpos, results[2].trace.qpos)


def test_precomputed_controls_use_one_native_multithreaded_rollout() -> None:
    base = _request()
    steps = int(base.config.duration_s * PHYSICS_HZ)
    requests = []
    for index, shoulder_control in enumerate((5.0, -5.0, 2.5)):
        controls = np.zeros((steps, 6), dtype=np.float64)
        controls[:, 4] = shoulder_control
        requests.append(
            replace(
                base,
                request_id=f"open-loop-{index}",
                precomputed_ctrl=controls,
            )
        )

    results = NativeMujocoRuntime().simulate_batch(requests, max_workers=3)

    assert [item.request_id for item in results] == [item.request_id for item in requests]
    assert all(item.completed for item in results)
    assert all(item.diagnostics["execution_backend"] == "native_mujoco_rollout" for item in results)
    assert all(item.diagnostics["rollout_batch_size"] == 3 for item in results)
    assert all(item.diagnostics["rollout_threads"] == 3 for item in results)
    assert all(item.diagnostics["process_id"] == os.getpid() for item in results)
    assert all(item.trace.qpos.shape[0] == steps + 1 for item in results)
    assert not np.array_equal(results[0].trace.qpos, results[1].trace.qpos)


def test_invalid_frequency_and_missing_free_root_return_typed_failures() -> None:
    bad_frequency = replace(
        _request(), config=SimulationConfig(duration_s=0.1, physics_hz=120)
    )
    frequency_result = NativeMujocoRuntime().simulate(bad_frequency)
    assert frequency_result.status is SimulationStatus.FAILED
    assert frequency_result.failure is not None
    assert frequency_result.failure.code is SimulationFailureCode.INVALID_REQUEST

    anchored_xml = """
    <mujoco>
      <worldbody><body><joint name="joint"/><geom type="capsule" size=".03"
        fromto="0 0 0 0 0 .2" mass="1"/></body></worldbody>
      <actuator><motor joint="joint"/></actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(anchored_xml)
    target = ControlTarget.stationary(model.qpos0, model.nv)
    missing_root_result = NativeMujocoRuntime().simulate(
        SimulationRequest(
            model_xml=anchored_xml,
            trajectory=lambda time_s: target,  # type: ignore[arg-type]
            config=SimulationConfig(duration_s=0.1),
        )
    )
    assert missing_root_result.failure is not None
    assert missing_root_result.failure.code is SimulationFailureCode.MISSING_FREE_ROOT


def test_state_dependent_actuator_is_rejected_instead_of_miscontrolled() -> None:
    xml = """
    <mujoco>
      <worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
        <body><joint name="joint"/><geom type="capsule" size=".03"
          fromto="0 0 0 0 0 .2" mass="1"/></body>
      </body></worldbody>
      <actuator><position joint="joint" kp="10"/></actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    target = ControlTarget.stationary(model.qpos0, model.nv)

    class Provider:
        def sample(self, time_s: float) -> ControlTarget:
            del time_s
            return target

    result = NativeMujocoRuntime().simulate(
        SimulationRequest(
            model_xml=xml,
            trajectory=Provider(),
            config=SimulationConfig(duration_s=0.05),
        )
    )
    assert result.status is SimulationStatus.FAILED
    assert result.failure is not None
    assert result.failure.code is SimulationFailureCode.UNSUPPORTED_ACTUATOR


def test_self_contained_mjz_with_embedded_mesh_is_authoritative_model_source() -> None:
    xml = """
    <mujoco model="embedded_mesh_mjz">
      <asset><mesh name="tetra" file="tetra.obj"/></asset>
      <worldbody>
        <geom name="embedded_obstacle" type="mesh" mesh="tetra" pos="2 0 0"/>
        <body name="pelvis" pos="0 0 1"><freejoint name="root"/>
          <geom type="sphere" size="0.1" mass="1"/>
          <body name="arm"><joint name="joint"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="joint" ctrllimited="true" ctrlrange="-5 5"/></actuator>
    </mujoco>
    """
    tetra_obj = b"""v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""
    spec = mujoco.MjSpec.from_string(xml, assets={"tetra.obj": tetra_obj})
    model = spec.compile()
    archive = BytesIO()
    spec.to_zip(archive)
    payload = archive.getvalue()
    request = SimulationRequest(
        model_xml=None,
        model_mjz=payload,
        trajectory=ConstantTarget(ControlTarget.stationary(model.qpos0, model.nv)),
        config=SimulationConfig(duration_s=0.05),
        request_id="mjz-mesh",
    )

    result = NativeMujocoRuntime().simulate(request)

    assert result.completed
    assert result.model_hash == sha256(payload).hexdigest()
    assert result.diagnostics["model_source_format"] == "mjz"
    assert result.trace.qpos.shape[1] == model.nq


def test_multiple_or_missing_authoritative_sources_are_typed_invalid_requests() -> None:
    request = _request()
    both = replace(request, model_mjz=b"not-used-because-source-is-ambiguous")
    result = NativeMujocoRuntime().simulate(both)

    assert result.status is SimulationStatus.FAILED
    assert result.failure is not None
    assert result.failure.code is SimulationFailureCode.INVALID_REQUEST
