"""An evaluator observes actual physics; actor encoders cannot include objects."""

from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest
from rigby_core.observations import ObservationBoundaryError, PolicyWorker
from rigby_core.hashing import content_hash

from rigby_general.benchmark.contracts import BenchmarkWorldV1, RootGoalV1, RegionV1, SensorPolicyV1
from rigby_general.benchmark.evaluator import IndependentEvaluator
from rigby_general.benchmark.sensors import SensorAdapter
from rigby_general.benchmark.world import (
    BenchmarkBodyManifestV1, BenchmarkRefusal, compile_world, model_digest, resolved_world_manifest,
)
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import load_environment

ROOT = Path(__file__).resolve().parents[1]


def fixture_world(*, around_initial_object=False, sensors=None):
    low, high = ((0.13, -0.04, 0.04), (0.2, 0.04, 0.1)) if around_initial_object else ((0.25, -0.05, 0.03), (0.35, 0.05, 0.15))
    return BenchmarkWorldV1(environment=load_environment("desk_bench"),
                            sensor_policy=sensors or SensorPolicyV1(rgb=False),
                            goal=RootGoalV1(object_names=("cube",), target=RegionV1(minimum_m=low, maximum_m=high)))


def integration_state(model, data):
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def test_two_second_dwell_requires_stable_released_whole_object():
    c = compile_world(fixture_world(around_initial_object=True))
    data = mujoco.MjData(c.model)
    evaluator = IndependentEvaluator(c, expected_world_sha256=c.world_sha256)
    first = evaluator.evaluate(data)
    assert first.conditions_met and not first.root_success
    for _ in range(1100):
        mujoco.mj_step(c.model, data)
        result = evaluator.evaluate(data)
        if data.time < 2:
            assert not result.root_success
    assert result.root_success and result.dwell_elapsed_s >= 2
    assert result.objects[0]["whole_geometry_inside"] and result.objects[0]["released"]


def test_dwell_resets_when_the_object_exits_and_duplicate_time_is_rejected():
    c = compile_world(fixture_world(around_initial_object=True))
    data = mujoco.MjData(c.model)
    evaluator = IndependentEvaluator(c, expected_world_sha256=c.world_sha256)
    evaluator.evaluate(data)
    for _ in range(600): mujoco.mj_step(c.model, data)
    assert not evaluator.evaluate(data).root_success
    # An explicit test perturbation; never an action in a scored policy.
    data.qpos[0] = 0.5
    mujoco.mj_step(c.model, data)
    moved = evaluator.evaluate(data)
    assert not moved.conditions_met and moved.dwell_elapsed_s == 0
    with pytest.raises(BenchmarkRefusal) as error:
        evaluator.evaluate(data)
    assert error.value.code == "invalid_evaluation_clock"


def test_center_inside_does_not_count_as_whole_object_inside():
    world = fixture_world(around_initial_object=True)
    raw = world.model_dump(mode="json")
    raw["goal"]["target"]["maximum_m"][0] = 0.17  # Center=.165; object extends to .177.
    c = compile_world(BenchmarkWorldV1.model_validate(raw))
    result = IndependentEvaluator(c, expected_world_sha256=c.world_sha256).evaluate(mujoco.MjData(c.model))
    assert not result.objects[0]["whole_geometry_inside"]


def test_skipped_physics_intervals_cannot_prove_dwell():
    c = compile_world(fixture_world(around_initial_object=True))
    data = mujoco.MjData(c.model)
    evaluator = IndependentEvaluator(c, expected_world_sha256=c.world_sha256)
    evaluator.evaluate(data)
    for _ in range(1100): mujoco.mj_step(c.model, data)
    snapshot = evaluator.evaluate(data)
    assert snapshot.conditions_met and snapshot.dwell_elapsed_s == 0 and not snapshot.root_success


def test_rgb_encoders_and_truth_stay_separate_in_actual_simulator(tmp_path):
    robot = ingest_robot(ROOT / "assets/general/zoo/zoo_compact_arm/robot.urdf")
    c = compile_world(fixture_world(sensors=SensorPolicyV1(rgb=True, width=32, height=24)),
                      body=BenchmarkBodyManifestV1(robot=robot.manifest), robot_xml=robot.finalized.mjcf_xml)
    data = mujoco.MjData(c.model)
    # World free qpos precedes robot qpos; inject a visibly distinct object value.
    data.qpos[0] = 0.173
    for name, rest in zip(robot.manifest.actuator_order, robot.manifest.rest_qpos):
        joint = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_JOINT, "robot/"+name)
        data.qpos[c.model.jnt_qposadr[joint]] = rest
    before = integration_state(c.model, data)
    truth = IndependentEvaluator(c, expected_world_sha256=c.world_sha256).evaluate(data).truth
    module = tmp_path / "sensor_policy_fixture.py"
    module.write_text('from rigby_core.observations import PolicyAction\n'
                      'def act(packet):\n'
                      '    assert not hasattr(packet, "object_poses")\n'
                      '    assert not hasattr(packet, "success_labels")\n'
                      '    positions = [r.values[0] for r in packet.readings if r.channel.endswith("_position")]\n'
                      '    assert 0.173 not in positions\n'
                      '    assert len(next(r for r in packet.readings if r.channel=="rgb").values)==24*32*3\n'
                      '    return PolicyAction(values=tuple(0.0 for _ in positions))\n', encoding="utf-8", newline="\n")
    with SensorAdapter(c) as sensor:
        packet = sensor.observe(data, sequence=0)
        with PolicyWorker(policy="sensor_policy_fixture:act", declaration=sensor.declaration,
                          action_size=c.model.nu, policy_paths=(tmp_path,), timeout_s=10) as worker:
            assert worker.act(packet).values == (0.0,)*c.model.nu
            with pytest.raises(ObservationBoundaryError):
                worker.act(truth)
    assert np.array_equal(before, integration_state(c.model, data)), "Inspection must not modify live integration state"
    assert truth.object_poses[0].position_xyz[0] == 0.173


def test_fully_observed_baseline_cannot_masquerade_as_sensor_only():
    robot = ingest_robot(ROOT / "assets/general/zoo/zoo_compact_arm/robot.urdf")
    c = compile_world(fixture_world(sensors=SensorPolicyV1(rgb=False, mode="fully_observed_diagnostic")),
                      body=BenchmarkBodyManifestV1(robot=robot.manifest), robot_xml=robot.finalized.mjcf_xml)
    data = mujoco.MjData(c.model)
    truth = IndependentEvaluator(c, expected_world_sha256=c.world_sha256).evaluate(data).truth
    with SensorAdapter(c) as sensor:
        with pytest.raises(BenchmarkRefusal): sensor.observe(data, sequence=0)
        diagnostic = sensor.observe_diagnostic(data, sequence=0, truth=truth)
        assert diagnostic.access == "fully_observed_diagnostic"
        assert diagnostic.evaluator_truth.object_poses
        with pytest.raises(ObservationBoundaryError): sensor.projector.validate(diagnostic)


@pytest.mark.parametrize("margin, expected_held", [(0.01, True), (0.0, False)])
def test_positive_distance_robot_contact_force_cannot_count_as_release(margin, expected_held):
    raw = fixture_world(around_initial_object=True).model_dump(mode="json")
    raw["physics"]["gravity_mps2"] = [0, 0, 0]
    raw["goal"]["dwell_s"] = 0.01
    c = compile_world(BenchmarkWorldV1.model_validate(raw))
    root = ET.fromstring(c.xml)
    worldbody = root.find("worldbody")
    cube = c.world.environment.objects[0]
    for sign in (-1, 1):
        position = np.array(cube.position_m)
        position[0] += sign * (cube.size_m[0] + 0.01 + 0.005)
        body = ET.SubElement(worldbody, "body", name=f"robot/holder{sign}", pos=" ".join(map(str, position)))
        ET.SubElement(body, "geom", type="sphere", size=".01", margin=str(margin), priority="2")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    manifest = resolved_world_manifest(model, c.world)
    # A deliberately constructed evaluator fixture, not an admission/world-hash
    # bypass: the test model's complete identity is pinned before evaluation.
    c = replace(c, model=model, xml=xml, manifest=manifest,
                world_sha256=content_hash(manifest), compiled_model_sha256=model_digest(model))
    data = mujoco.MjData(model)
    evaluator = IndependentEvaluator(c, expected_world_sha256=c.world_sha256)
    for _ in range(10):
        assessment = evaluator.evaluate(data)
        row = assessment.objects[0]
        assert row["whole_geometry_inside"]
        assert row["released"] is not expected_held
        if expected_held:
            assert row["maximum_robot_normal_force_n"] > 4.0
            assert not assessment.root_success
        mujoco.mj_step(model, data)
    assert assessment.root_success is not expected_held
