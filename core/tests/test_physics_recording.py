from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rigby_core.simulation.recording import PhysicsRecord, PhysicsRecorder, replay_physics


XML = """<mujoco><option timestep="0.005"/>
<worldbody><geom type="plane" size="2 2 .1"/>
<body pos="0 0 .4"><joint name="slide" type="slide" axis="0 0 1"/>
<geom type="sphere" size=".04" mass="1"/></body></worldbody>
<actuator><general joint="slide" dyntype="filter" dynprm=".02"/></actuator>
<sensor><jointpos joint="slide"/><jointvel joint="slide"/></sensor></mujoco>"""


def make_record():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.3
    # Start partway through contact, with nonzero actuator and warmstart state.
    for _ in range(90):
        mujoco.mj_step(model, data)
    recorder = PhysicsRecorder(model)
    for i in range(161):
        action = np.array([4.0 + 3.0 * np.sin(i / 12.0)])
        data.qfrc_applied[:] = 0.5 if 50 <= i < 80 else 0.0
        before = (data.time, data.qpos.copy(), data.qacc_warmstart.copy(), data.ctrl.copy())
        recorder.capture(data, action, control_time_s=i * model.opt.timestep)
        assert data.time == before[0]
        for observed, expected in zip((data.qpos, data.qacc_warmstart, data.ctrl), before[1:]):
            np.testing.assert_array_equal(observed, expected)
        if i < 160:
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
    return model, recorder.finish()


def test_contact_actuator_and_warmstart_replay_from_one_initial_state(tmp_path):
    model, record = make_record()
    assert len(record.arrays["contact_geom"]) > 0
    assert np.max(np.abs(record.arrays["contact_wrench"])) > 0
    path = tmp_path / "model.mjb"
    mujoco.mj_saveModel(model, str(path))
    restored_model = mujoco.MjModel.from_binary_path(str(path))
    restored = PhysicsRecord.from_bytes(record.to_bytes())
    assert restored.content_hash() == record.content_hash()
    replay = replay_physics(restored_model, restored)
    assert replay["agrees"], replay
    assert replay["integrated_steps"] == 160
    assert replay["max_state_error"] == 0.0
    assert replay_physics(
        restored_model, restored,
        controller=lambda data, i: np.array([4.0 + 3.0 * np.sin(i / 12.0)]),
    )["agrees"]


def test_changed_action_is_not_hidden_by_resetting_intermediate_states():
    model, record = make_record()
    changed = PhysicsRecord({k: v.copy() for k, v in record.arrays.items()})
    changed.arrays["action"][10, 0] += 100.0
    result = replay_physics(model, changed)
    assert not result["agrees"]
    assert result["first_state_difference"] == 11
    assert changed.content_hash() != record.content_hash()


def test_recomputed_controller_must_match_recorded_commands():
    model, record = make_record()
    result = replay_physics(model, record, controller=lambda data, i: np.zeros(model.nu))
    assert not result["agrees"]
    assert result["first_action_difference"] == 0


def test_incompatible_timestep_and_callback_are_refused():
    model, record = make_record()
    model.opt.timestep *= 2
    with pytest.raises(ValueError, match="timestep"):
        replay_physics(model, record)
    try:
        mujoco.set_mjcb_control(lambda model, data: None)
        with pytest.raises(ValueError, match="external mjcb_control"):
            PhysicsRecorder(model)
    finally:
        mujoco.set_mjcb_control(None)


def test_initial_state_is_required_and_duplicate_times_are_refused():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    recorder = PhysicsRecorder(model)
    with pytest.raises(ValueError, match="initial state"):
        recorder.finish()
    recorder.capture(data, np.zeros(model.nu), control_time_s=0.0)
    with pytest.raises(ValueError, match="strictly increase"):
        recorder.capture(data, np.zeros(model.nu), control_time_s=0.0)
    assert replay_physics(model, recorder.finish())["integrated_steps"] == 0
