"""An undeclared torque limit, and a gate that can actually fail.

Both of these were silent for a long time and both produced confident wrong
answers rather than errors, which is the kind of bug a test has to hold down
once it is found.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rigby_general.gates.certify import GatePolicy
from rigby_general.gates.control import (
    ComputedTorqueController,
    DEFAULT_NATURAL_FREQUENCY_HZ,
)
from rigby_general.morphology import measure
from rigby_general.pipeline import ingest_robot


UNDECLARED = """<?xml version="1.0"?>
<robot name="undeclared">
  <link name="base">
    <inertial>
      <mass value="4.0"/><origin xyz="0 0 0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><geometry><box size="0.16 0.16 0.10"/></geometry></collision>
  </link>
  <link name="upper">
    <inertial>
      <mass value="9.0"/><origin xyz="0 0 0.30"/>
      <inertia ixx="0.30" iyy="0.30" izz="0.02" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><origin xyz="0 0 0.30"/>
      <geometry><box size="0.08 0.08 0.60"/></geometry>
    </collision>
  </link>
  <link name="fore">
    <inertial>
      <mass value="5.0"/><origin xyz="0 0 0.24"/>
      <inertia ixx="0.12" iyy="0.12" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><origin xyz="0 0 0.24"/>
      <geometry><box size="0.06 0.06 0.48"/></geometry>
    </collision>
  </link>
  <link name="column">
    <inertial>
      <mass value="3.0"/><origin xyz="0 0 0.05"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision><origin xyz="0 0 0.05"/>
      <geometry><box size="0.10 0.10 0.10"/></geometry>
    </collision>
  </link>
  <joint name="yaw" type="revolute">
    <parent link="base"/><child link="column"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" effort="0" velocity="1.5"/>
  </joint>
  <joint name="shoulder" type="revolute">
    <parent link="column"/><child link="upper"/>
    <origin xyz="0 0 0.10"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="0" velocity="1.5"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="fore"/>
    <origin xyz="0 0 0.60"/><axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="2.4" effort="0" velocity="1.5"/>
  </joint>
</robot>
"""

DECLARED = UNDECLARED.replace('effort="0"', 'effort="180"')


def _ingest(tmp_path, text, robot_id):
    source = tmp_path / f"{robot_id}.urdf"
    source.write_text(text, encoding="utf-8")
    return ingest_robot(source, robot_id=robot_id)


def test_undeclared_effort_is_recorded_as_unknown(tmp_path):
    """effort="0" is a refusal to state a limit, not a limit of zero."""

    robot = _ingest(tmp_path, UNDECLARED, "undeclared")
    joints = {joint.name: joint for joint in robot.morphology.joints}
    assert not joints["shoulder"].effort_declared
    assert not joints["elbow"].effort_declared


def test_declared_effort_is_taken_as_given(tmp_path):
    """What the source states outranks anything derivable from the body."""

    robot = _ingest(tmp_path, DECLARED, "declared")
    joints = {joint.name: joint for joint in robot.morphology.joints}
    assert joints["shoulder"].effort_declared
    assert joints["shoulder"].effort_limit == pytest.approx(180.0, rel=1e-6)


def test_undeclared_joints_compile_unlimited(tmp_path):
    """A guessed clamp builds an arm that sags; leave it unlimited instead."""

    robot = _ingest(tmp_path, UNDECLARED, "undeclared")
    model = mujoco.MjModel.from_xml_string(robot.mjcf_xml)
    assert not any(model.actuator_forcelimited)

    declared = _ingest(tmp_path, DECLARED, "declared")
    limited = mujoco.MjModel.from_xml_string(declared.mjcf_xml)
    assert all(limited.actuator_forcelimited)


def test_measured_floor_holds_the_arm_up(tmp_path):
    """The reported floor is a lower bound, so it must clear gravity at rest."""

    robot = _ingest(tmp_path, UNDECLARED, "undeclared")
    model = mujoco.MjModel.from_xml_string(robot.mjcf_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gravity = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, gravity)

    floors = {joint.name: joint.effort_limit for joint in robot.morphology.joints}
    for index, joint in enumerate(robot.morphology.joints):
        dof = int(model.jnt_dofadr[index])
        assert floors[joint.name] >= abs(float(gravity[dof]))


def test_the_effort_gate_reads_the_demand_not_the_clamp(tmp_path):
    """The controller clips before returning, so ``ctrl`` can never exceed the limit.

    Comparing the applied control against the actuator's own limit is a test
    that cannot fail. What makes saturation visible is the demand kept before
    the clamp, and this pins that the two actually differ when it matters.
    """

    robot = _ingest(tmp_path, DECLARED, "declared")
    model = mujoco.MjModel.from_xml_string(robot.mjcf_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    controller = ComputedTorqueController(model)

    class _Target:
        qpos = np.full(model.nq, 1.5)
        qvel = np.zeros(model.nv)
        qacc = np.zeros(model.nv)

    applied = controller.compute(data, _Target())
    demand = controller.last_demand

    limit = float(model.actuator_forcerange[0][1])
    assert abs(float(applied[0])) <= limit + 1e-9
    assert abs(float(demand[0])) > limit
    assert abs(float(demand[0])) > abs(float(applied[0]))


def test_gate_policy_and_measurement_agree_on_the_controller(tmp_path):
    """measure.py mirrors two numbers it must not import. Keep them in step."""

    assert measure.EFFORT_RAMP_SECONDS > 0.0
    assert GatePolicy().tracking_error_fraction == pytest.approx(0.035)
    assert DEFAULT_NATURAL_FREQUENCY_HZ == pytest.approx(14.0)
