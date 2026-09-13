"""Real simulator probes of certification's clock and observation timestamps."""

from types import SimpleNamespace
import math

import mujoco
import numpy as np
import pytest
from rigby_core.motion.trajectory import CandidateTrajectoryV1

from rigby_general.gates.certify import simulate
from rigby_general.gates.control import ComputedTorqueController, ControllerConfig


def fixture(timestep, *, wall=False):
    obstacle = '<body name="obstacle" pos=".010 0 0"><geom type="box" size=".001 .01 .01"/></body>' if wall else ''
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep="{timestep}" gravity="0 0 0" integrator="Euler"/>
      <worldbody><body name="anchor"><body name="rail">
        <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-1 1"/>
        <geom type="sphere" size=".002" mass="1"/><site name="tip" size=".001"/>
      </body></body>{obstacle}</worldbody>
      <actuator><motor joint="slide"/></actuator></mujoco>''')
    # Only the three fields simulate reads; this test isolates the real physics
    # clock and has no dependence on morphology inference or benchmark labels.
    manifest = SimpleNamespace(rest_qpos=(0.,), morphology=SimpleNamespace(base_body="anchor"),
                               adjacent_collision_exclusions=())
    return model, manifest


def linear_reference(duration, *, speed=.1, start=0.):
    return CandidateTrajectoryV1(candidate_id="clock-probe", program_hash="p", rig_hash="r",
        times_s=np.array([start, duration]), qpos=np.array([[0.], [speed*duration]]),
        qvel=np.full((2, 1), speed), qacc=np.zeros((2, 1)))


def independent_rollout(model, duration, speed):
    """Drive real physics using its own clock, independent of simulate's loop."""
    controller = ComputedTorqueController(model, ControllerConfig())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    times, qpos, qvel, controls = [], [], [], []
    from rigby_core.simulation.controller import ControlTarget
    while True:
        now = float(data.time)
        target = ControlTarget(qpos=np.array([speed*min(now, duration)]),
                               qvel=np.array([speed]), qacc=np.zeros(1))
        command = controller.compute(data, target)
        times.append(now);qpos.append(data.qpos.copy());qvel.append(data.qvel.copy());controls.append(command.copy())
        if now >= duration or math.isclose(now, duration, rel_tol=0, abs_tol=1e-14):
            break
        data.ctrl[:] = command
        mujoco.mj_step(model, data)
    return np.asarray(times), np.asarray(qpos), np.asarray(qvel), np.asarray(controls)


@pytest.mark.parametrize("timestep,duration", [(.002, .02), (.002, .021), (1/240, .05), (.003, .02), (.01, .07)])
def test_trace_and_control_follow_native_physics_time(timestep, duration):
    model, manifest = fixture(timestep)
    reference = linear_reference(duration)
    before = reference.content_hash()
    trace = simulate(model, manifest, reference, site_name="tip")
    times, qpos, qvel, controls = independent_rollout(model, duration, .1)
    assert np.array_equal(trace.times_s, times)
    assert np.allclose(trace.qpos, qpos, atol=1e-15, rtol=0)
    assert np.allclose(trace.qvel, qvel, atol=1e-14, rtol=0)
    assert np.allclose(trace.ctrl, controls, atol=1e-12, rtol=0)
    assert trace.times_s[-1] >= duration-1e-14
    assert trace.times_s[-1] < duration+timestep-1e-14
    assert model.opt.timestep == timestep and reference.content_hash() == before
    expected_error = np.abs(np.minimum(times, duration)*.1 - trace.qpos[:, 0])
    assert np.allclose(trace.tracking_error_m, expected_error, atol=1e-15, rtol=0)
    repeat = simulate(model, manifest, reference, site_name="tip")
    assert trace.content_hash() == repeat.content_hash()


def test_final_tick_contact_is_measured_after_integration():
    timestep = 1/240
    model, manifest = fixture(timestep, wall=True)
    trace = simulate(model, manifest, linear_reference(2*timestep, speed=1), site_name="tip")
    assert trace.qpos[-2, 0] < .007 < trace.qpos[-1, 0]
    # The original mj_step cache was still at the previous, non-contact pose.
    assert ("obstacle", "rail") in trace.unexpected_contacts


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_physics_timestep_is_refused(value):
    model, manifest = fixture(.002)
    model.opt.timestep = value
    with pytest.raises(ValueError, match="timestep"):
        simulate(model, manifest, linear_reference(.02))


def test_nonzero_reference_origin_is_refused():
    model, manifest = fixture(.002)
    with pytest.raises(ValueError, match="time zero"):
        simulate(model, manifest, linear_reference(.02, start=.01))
