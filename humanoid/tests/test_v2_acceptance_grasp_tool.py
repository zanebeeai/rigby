from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import mujoco
import numpy as np
import pytest

from rigby_v2.acceptance.grasp_tool import (
    AcceptanceGate,
    PackTask,
    build_attempt,
    evaluate_attempt,
    run_acceptance,
)
from rigby_v2.simulation import NativeMujocoRuntime

pytestmark = pytest.mark.medium


@pytest.mark.parametrize("task", tuple(PackTask))
def test_attempt_models_keep_the_free_root_and_have_no_shortcuts(task: PackTask) -> None:
    attempt = build_attempt(task)
    assert attempt.request.model_mjz is not None
    model = mujoco.MjSpec.from_zip(BytesIO(attempt.request.model_mjz)).compile()

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free")
    assert root >= 0 and model.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE
    assert model.nmocap == 0
    assert not any(model.eq_type[index] == mujoco.mjtEq.mjEQ_WELD for index in range(model.neq))
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        assert not name.startswith("obj__")


def test_neutral_or_lift_only_trace_cannot_false_positive_place() -> None:
    attempt = build_attempt(PackTask.GRASP_PLACE_BLOCK)
    runtime = NativeMujocoRuntime()
    run = runtime.simulate(attempt.request)
    neutral_qpos = np.repeat(run.trace.qpos[:1], len(run.trace.qpos), axis=0)
    neutral_contacts = tuple(
        replace(frame, contacts=()) for frame in run.trace.contacts
    )
    neutral = replace(
        run,
        trace=replace(run.trace, qpos=neutral_qpos, contacts=neutral_contacts),
    )

    report = evaluate_attempt(attempt, (neutral, neutral, neutral))
    by_gate = {item.gate: item for item in report.measurements}

    assert not report.accepted
    assert not by_gate[AcceptanceGate.SELECTED_STATE].passed
    assert not by_gate[AcceptanceGate.GRASP].passed
    assert not by_gate[AcceptanceGate.PLACE_AND_RELEASE].passed


@pytest.mark.parametrize("task", tuple(PackTask))
def test_real_pack_probe_is_repeatable_and_fails_closed_with_measured_diagnostics(
    task: PackTask,
) -> None:
    report = run_acceptance(task)
    by_gate = {item.gate: item for item in report.measurements}

    assert len(report.runs) == 3
    assert by_gate[AcceptanceGate.STRUCTURE].passed
    assert by_gate[AcceptanceGate.EXPORT_REIMPORT].passed
    assert by_gate[AcceptanceGate.THREE_EXACT_REPEATS].passed
    assert all(run.diagnostics["qpos_writes_after_initialization"] == 0 for run in report.runs)
    # Balance and fixture clearance are real, but neither task is accepted
    # unless the measured object lifecycle also succeeds.
    assert not report.accepted
    assert by_gate[AcceptanceGate.GLOBAL_PHYSICS].passed
    if task is PackTask.GRASP_PLACE_BLOCK:
        assert by_gate[AcceptanceGate.GRASP].passed
        assert not by_gate[AcceptanceGate.CONTACT_LIFECYCLE].passed
        assert not by_gate[AcceptanceGate.PLACE_AND_RELEASE].passed
    else:
        assert by_gate[AcceptanceGate.GRASP].passed
        assert not by_gate[AcceptanceGate.SELECTED_STATE].passed
        assert not by_gate[AcceptanceGate.CONTACT_LIFECYCLE].passed
        assert not by_gate[AcceptanceGate.TOOL_OUTCOME].passed
