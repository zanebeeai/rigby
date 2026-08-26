from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from rigby_v2.simulation import (
    ConstantTarget,
    ControlTarget,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
    StandingControlConfig,
    repeat_replay,
)

from test_v2_simulation_runtime import ARTICULATED_BIPED_XML, _request


SUPPORT_CONFIG = StandingControlConfig(
    support_foot_bodies=("left_foot", "right_foot"),
    pelvis_body="pelvis",
    torso_body="pelvis",
)


def standing_request(*, duration_s: float = 0.20, request_id: str = "standing"):
    return replace(
        _request(request_id=request_id),
        config=SimulationConfig(duration_s=duration_s, standing=SUPPORT_CONFIG),
    )


def _body_positions(xml: str, qpos_trace: np.ndarray, body_name: str) -> np.ndarray:
    model = mujoco.MjModel.from_xml_string(xml)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    data = mujoco.MjData(model)
    positions = []
    for qpos in qpos_trace:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[body_id].copy())
    return np.asarray(positions)


def test_standing_controller_keeps_a_perturbed_free_root_supported() -> None:
    model = mujoco.MjModel.from_xml_string(ARTICULATED_BIPED_XML)
    initial_velocity = np.zeros(model.nv)
    initial_velocity[0] = 0.4
    initial_velocity[4] = 0.4
    request = replace(
        standing_request(duration_s=0.50),
        initial_qvel=initial_velocity,
    )

    result = NativeMujocoRuntime().simulate(request)

    assert result.completed
    assert result.diagnostics["controller"] == "whole_body_standing_plus_inverse_dynamics_pd"
    assert result.diagnostics["standing_control"] is True
    assert result.diagnostics["qpos_writes_after_initialization"] == 0
    assert result.metrics is not None
    assert result.metrics.root_translation_drift_m < 0.10
    assert result.trace.qpos[-1, 2] > 0.55
    assert np.max(np.abs(result.trace.ctrl)) <= 80.0
    # The free-root generalized coordinates have no actuator effort. Contact
    # and articulated joints, rather than a hidden base actuator, support it.
    assert np.array_equal(result.trace.generalized_effort[:, :6], np.zeros((121, 6)))

    for foot_name in ("left_foot", "right_foot"):
        positions = _body_positions(ARTICULATED_BIPED_XML, result.trace.qpos, foot_name)
        assert np.max(np.linalg.norm(positions - positions[0], axis=1)) < 0.10


def test_standing_tasks_are_layered_over_pd_and_replay_exactly() -> None:
    standing = standing_request()
    base = replace(standing, config=SimulationConfig(duration_s=0.20))
    runtime = NativeMujocoRuntime()

    standing_result = runtime.simulate(standing)
    base_result = runtime.simulate(base)

    assert standing_result.completed and base_result.completed
    assert not np.array_equal(standing_result.trace.ctrl, base_result.trace.ctrl)
    assert standing_result.metrics is not None
    assert standing_result.metrics.max_actuator_force > 0
    assert standing_result.metrics.max_generalized_effort > 0
    assert standing_result.metrics.max_joint_power > 0
    assert repeat_replay(runtime, standing, runs=3).deterministic


def test_named_humanoid_root_coexists_with_a_free_manipulated_object() -> None:
    xml = ARTICULATED_BIPED_XML.replace(
        "</worldbody>",
        '<body name="free_object" pos="0.5 0 0.2"><freejoint name="object_free"/>'
        '<geom name="object_geom" type="box" size="0.04 0.04 0.04" mass="0.2"/>'
        "</body></worldbody>",
    )
    model = mujoco.MjModel.from_xml_string(xml)
    target = ConstantTarget(ControlTarget.stationary(model.qpos0, model.nv))
    request = SimulationRequest(
        model_xml=xml,
        trajectory=target,
        config=SimulationConfig(duration_s=0.05, free_root_joint_name="root"),
        request_id="free-object",
    )

    result = NativeMujocoRuntime().simulate(request)

    assert result.completed
    assert result.trace.qpos.shape[1] == model.nq
    assert result.diagnostics["free_root_qpos_adr"] == 0
