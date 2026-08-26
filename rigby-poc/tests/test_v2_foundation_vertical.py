from __future__ import annotations

import mujoco

from rigby_v2.rigging.canonical_human import xml_for_profile
from rigby_v2.simulation import (
    ConstantTarget,
    ControlTarget,
    NativeMujocoRuntime,
    SimulationConfig,
    SimulationRequest,
    SimulationStatus,
)


def test_canonical_human_executes_in_authoritative_runtime() -> None:
    xml = xml_for_profile("medium")
    model = mujoco.MjModel.from_xml_string(xml)
    target = ControlTarget.stationary(model.qpos0, model.nv)

    result = NativeMujocoRuntime().simulate(
        SimulationRequest(
            request_id="canonical-human-smoke",
            model_xml=xml,
            trajectory=ConstantTarget(target),
            config=SimulationConfig(duration_s=0.05),
            initial_qpos=model.qpos0,
        )
    )

    assert result.status is SimulationStatus.COMPLETED
    assert result.failure is None
    assert result.metrics is not None and result.metrics.finite
    assert result.trace.qpos.shape == (13, model.nq)
    assert result.trace.qvel.shape == (13, model.nv)
    assert result.trace.ctrl.shape == (13, model.nu)
    assert result.diagnostics["qpos_writes_after_initialization"] == 0
    assert result.diagnostics["task_success_evaluated"] is False

