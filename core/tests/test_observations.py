"""Exercise the sensor/evaluator boundary, including actual child access attempts."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from rigby_core.observations import (
    EvaluatorTruth, FullyObservedDiagnostic, ObjectPose, ObservationBoundaryError,
    ObservationProjector, PolicyAction, PolicyObservation, PolicyWorker,
    PolicyWorkerError, PolicyWorkerTimeout, PredicateValue, SensorChannel,
    SensorDeclaration, SensorReading,
)


def _declaration() -> SensorDeclaration:
    return SensorDeclaration(channels=(
        SensorChannel(name="encoders", kind="joint_position", shape=(2,),
                      unit="radian", entities=("axis_a", "axis_b")),
        SensorChannel(name="velocity", kind="joint_velocity", shape=(2,),
                      unit="radian_per_second"),
        SensorChannel(name="camera", kind="rgb", shape=(1, 2, 3), unit="uint8", frame="camera_frame"),
        SensorChannel(name="depth", kind="depth", shape=(1, 2), unit="meter"),
        SensorChannel(name="touch", kind="contact", shape=(2,), unit="binary"),
        SensorChannel(name="force", kind="contact", shape=(1,), unit="newton"),
    ))


def _readings() -> dict:
    return {
        "encoders": [0.25, -0.5], "velocity": [0.0, 0.25],
        "camera": [0, 127, 255, 12, 34, 56], "depth": [0.3, 1.2],
        "touch": [0, 1], "force": [2.5],
    }


def _packet(sequence=0, time_s=0.0):
    return ObservationProjector(_declaration()).project(
        sequence=sequence, time_s=time_s, readings=_readings(),
    )


def _truth():
    return EvaluatorTruth(
        time_s=0.0,
        object_poses=(ObjectPose(object_id="hidden_object", position_xyz=(42.0, 43.0, 44.0),
                               orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),),
        success_labels=(PredicateValue(name="task_success", satisfied=True),),
        simulator_state=(91.0, 92.0),
    )


@pytest.fixture
def policy_module(tmp_path, monkeypatch):
    # Parent-only assignments below are NOT part of this source module. A fork,
    # pickle/closure handoff, or same-interpreter policy call would see them.
    source = '''
import os
import sys
import time
from rigby_core.observations import PolicyAction

def act(packet):
    print("policy logging is not a JSON protocol message")
    return PolicyAction(values=(packet.readings[0].values[0] * -2, packet.time_s))

def probe(packet):
    checks = []
    for name in ("object_poses", "success_labels", "evaluator_truth", "model", "data"):
        try:
            getattr(packet, name)
            checks.append(0)
        except AttributeError:
            checks.append(1)
    raw = packet.model_dump()
    for name in ("object_poses", "success_labels", "simulator_state"):
        try:
            raw[name]
            checks.append(0)
        except KeyError:
            checks.append(1)
    checks.append(int("EVALUATOR_TRUTH" not in globals()))
    checks.append(int(not hasattr(sys.modules["__main__"], "PARENT_ONLY_TRUTH")))
    checks.append(int("RIGBY_TEST_SECRET" not in os.environ))
    checks.append(int(all(x.channel not in ("object_poses", "success_labels") for x in packet.readings)))
    return PolicyAction(values=tuple(checks))

def mutation_probe(packet):
    blocked = 0
    try:
        packet.sequence = 900
    except Exception:
        blocked += 1
    try:
        packet.readings[0].values[0] = 900
    except TypeError:
        blocked += 1
    # Even an intentional Python-level frozen-model bypass affects only this
    # child's deserialized copy, never the simulator or parent's packet.
    object.__setattr__(packet.readings[0], "values", (999.0, 999.0))
    return PolicyAction(values=(blocked, packet.readings[0].values[0]))

def slow(packet):
    time.sleep(20)
    return PolicyAction(values=(0,))

def crash(packet):
    os._exit(17)

def wrong_type(packet):
    return {"values": [1, 2]}

def wrong_size(packet):
    return PolicyAction(values=(1,))

def nonfinite(packet):
    return PolicyAction.model_construct(values=(float("nan"), 0))

def raises(packet):
    raise RuntimeError("deliberate test failure")

def writes_malformed_protocol(packet):
    os.write(1, b"not-json\\n")
    return PolicyAction(values=(1, 2))

def stop_reading_input(packet):
    class NeverReads:
        def readline(self, *args):
            time.sleep(20)
            return ""
    sys.stdin = NeverReads()
    return PolicyAction(values=(1, 2))
'''
    module_name = "boundary_test_policy"
    (tmp_path / f"{module_name}.py").write_text(source, encoding="utf-8", newline="\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "EVALUATOR_TRUTH", _truth(), raising=False)
    monkeypatch.setattr(sys.modules["__main__"], "PARENT_ONLY_TRUTH", _truth(), raising=False)
    monkeypatch.setenv("RIGBY_TEST_SECRET", "parent-only-test-value")
    yield tmp_path, module
    sys.modules.pop(module_name, None)


def _worker(policy_module, function="act", action_size=2, timeout_s=10.0):
    path, _ = policy_module
    return PolicyWorker(
        policy=f"boundary_test_policy:{function}", declaration=_declaration(),
        action_size=action_size, policy_paths=(path,), timeout_s=timeout_s,
    )


def test_declared_channels_roundtrip_without_aliasing_or_simulator_objects():
    source = _readings()
    projector = ObservationProjector(_declaration())
    packet = projector.project(sequence=3, time_s=0.25, readings=source)
    source["encoders"][0] = 300
    assert packet.readings[0].values == (0.25, -0.5)
    assert projector.from_json(packet.canonical_json()) == packet
    assert packet.declaration_sha256 == _declaration().content_hash()
    assert packet.access == "declared_sensors"
    with pytest.raises(ValidationError):
        packet.sequence = 4
    with pytest.raises(TypeError):
        packet.readings[0].values[0] = 4
    with pytest.raises(ValidationError):
        packet.readings[0].values = (4, 4)
    assert set(json.loads(packet.canonical_json())) == {
        "schema", "access", "sequence", "time_s", "declaration_sha256", "readings",
    }


@pytest.mark.parametrize("name", ["object_poses", "success_labels", "simulator_state", "extra_camera"])
def test_projection_rejects_extra_fields_instead_of_sanitizing(name):
    readings = _readings()
    readings[name] = [1.0]
    with pytest.raises(ObservationBoundaryError, match="exactly"):
        ObservationProjector(_declaration()).project(sequence=0, time_s=0.0, readings=readings)


def test_projection_rejects_missing_channels_and_privileged_packet_types():
    projector = ObservationProjector(_declaration())
    readings = _readings()
    del readings["depth"]
    with pytest.raises(ObservationBoundaryError):
        projector.project(sequence=0, time_s=0.0, readings=readings)
    diagnostic = FullyObservedDiagnostic(observation=_packet(), evaluator_truth=_truth())
    for privileged in (_truth(), diagnostic):
        with pytest.raises(ObservationBoundaryError):
            projector.validate(privileged)
        with pytest.raises(ObservationBoundaryError):
            projector.project(sequence=0, time_s=0.0, readings=privileged)
        with pytest.raises(ValidationError):
            projector.from_json(privileged.canonical_json())
    assert diagnostic.access == "fully_observed_diagnostic"
    assert _truth().access == "evaluator_only"


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(success_labels={"task_success": True}),
    lambda x: x.update(access="fully_observed_diagnostic"),
    lambda x: x.update(declaration_sha256="0" * 64),
    lambda x: x.update(sequence=True),
    lambda x: x.update(time_s=-1.0),
    lambda x: x["readings"][0].update(object_poses=[42, 43, 44]),
    lambda x: x["readings"][0].update(channel="unknown_sensor"),
    lambda x: x["readings"].reverse(),
    lambda x: x["readings"].append(x["readings"][0]),
])
def test_wire_contract_rejects_malformed_privileged_and_mismatched_packets(mutation):
    payload = json.loads(_packet().canonical_json())
    mutation(payload)
    with pytest.raises((ValidationError, ObservationBoundaryError)):
        ObservationProjector(_declaration()).from_json(json.dumps(payload))


@pytest.mark.parametrize("name,values", [
    ("encoders", [0.0]), ("encoders", [float("nan"), 0.0]),
    ("encoders", [float("inf"), 0.0]), ("velocity", [False, 0.0]),
    ("camera", [0, 1, 256, 0, 0, 0]), ("camera", [0, 1, 2.5, 0, 0, 0]),
    ("depth", [-0.1, 0.5]), ("force", [-2.0]),
    ("touch", [0, 2]), ("touch", [0, 1.0]),
])
def test_sensor_kind_shape_range_and_finite_validation(name, values):
    readings = _readings()
    readings[name] = values
    with pytest.raises((ValidationError, ObservationBoundaryError, ValueError)):
        ObservationProjector(_declaration()).project(sequence=0, time_s=0.0, readings=readings)


@pytest.mark.parametrize("channel", [
    {"name": "object_poses", "kind": "depth", "shape": (1, 1), "unit": "meter"},
    {"name": "x", "kind": "rgb", "shape": (1, 1, 4), "unit": "uint8"},
    {"name": "x", "kind": "rgb", "shape": (1, 1, 3), "unit": "meter"},
    {"name": "x", "kind": "joint_position", "shape": (0,), "unit": "radian"},
    {"name": "x", "kind": "joint_position", "shape": (True,), "unit": "radian"},
    {"name": "x", "kind": "joint_position", "shape": (2,), "unit": "radian", "entities": ("one",)},
])
def test_invalid_sensor_declarations_fail_before_policy_start(channel):
    with pytest.raises(ValidationError):
        SensorChannel(**channel)


def test_duplicate_sensor_names_and_invalid_evaluator_truth_are_rejected():
    channel = _declaration().channels[0]
    with pytest.raises(ValidationError):
        SensorDeclaration(channels=(channel, channel))
    with pytest.raises(ValidationError):
        ObjectPose(object_id="x", position_xyz=(0, 0, 0), orientation_wxyz=(0, 0, 0, 0))
    with pytest.raises(ValidationError):
        EvaluatorTruth(time_s=0.0, simulator_state=(float("inf"),))
    with pytest.raises(ValidationError):
        EvaluatorTruth(time_s=0.0, object_poses=(_truth().object_poses[0],) * 2)


def test_duplicate_json_keys_nonfinite_wire_and_validation_bypasses_are_rejected():
    projector = ObservationProjector(_declaration())
    valid = _packet()
    with pytest.raises(ObservationBoundaryError, match="Duplicate"):
        projector.from_json(valid.canonical_json().replace('"sequence":0', '"sequence":0,"sequence":1'))
    for nonfinite in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ObservationBoundaryError, match="Nonfinite"):
            projector.from_json(valid.canonical_json().replace('"time_s":0.0', f'"time_s":{nonfinite}'))
    with pytest.raises(ValidationError):
        projector.validate(valid.model_copy(update={"sequence": -10}))
    forged = valid.model_copy(update={"readings": (SensorReading(channel="object_poses", values=(42,)),)})
    with pytest.raises(ObservationBoundaryError):
        projector.validate(forged)
    # Pydantic's model_copy intentionally bypasses validation; its ordinary
    # serializer can silently omit unknown keys. The boundary must refuse them.
    with pytest.raises(ObservationBoundaryError, match="Undeclared"):
        projector.validate(valid.model_copy(update={"success_labels": (True,)}))
    with pytest.raises(ObservationBoundaryError, match="Undeclared"):
        projector.validate(valid.model_copy(update={"readings": (
            valid.readings[0].model_copy(update={"object_poses": (42, 43, 44)}),
            *valid.readings[1:],
        )}))


def test_clean_worker_really_cannot_read_parent_evaluator_poses_labels_or_globals(policy_module):
    _, module = policy_module
    # Establish that the privileged values genuinely exist in the parent and
    # that an in-process policy would see them. The child must fail those reads.
    assert module.EVALUATOR_TRUTH.object_poses[0].position_xyz == (42.0, 43.0, 44.0)
    assert module.EVALUATOR_TRUTH.success_labels[0].satisfied is True
    local = module.probe(_packet())
    assert local.values[8:11] == (0, 0, 0)
    worker = _worker(policy_module, "probe", action_size=12)
    with worker:
        assert worker.is_alive and worker.pid is not None
        action = worker.act(_packet())
        assert action.values == (1,) * 12
    assert worker.closed and not worker.is_alive and worker.exit_code is not None


def test_worker_executes_policy_with_only_sensor_values_and_separate_copies(policy_module):
    packet = _packet()
    before = packet.canonical_json()
    with _worker(policy_module) as worker:
        assert worker.act(packet).values == (-0.5, 0.0)
        assert worker.act(_packet(sequence=1, time_s=0.125)).values == (-0.5, 0.125)
        with pytest.raises(ObservationBoundaryError, match="stale"):
            worker.act(_packet(sequence=1, time_s=0.2))
        with pytest.raises(ObservationBoundaryError, match="stale"):
            worker.act(_packet(sequence=2, time_s=0.01))
        assert worker.is_alive
        assert worker.act(_packet(sequence=2, time_s=0.25)).values == (-0.5, 0.25)
    with _worker(policy_module, "mutation_probe") as worker:
        assert worker.act(packet).values == (2, 999.0)
    assert packet.canonical_json() == before


def test_worker_refuses_truth_before_transmission_and_does_not_sanitize_it(policy_module):
    with _worker(policy_module) as worker:
        for privileged in (_truth(), FullyObservedDiagnostic(observation=_packet(), evaluator_truth=_truth())):
            with pytest.raises(ObservationBoundaryError):
                worker.act(privileged)
        assert worker.act(_packet()).values == (-0.5, 0.0)


@pytest.mark.parametrize("function", [
    "crash", "wrong_type", "wrong_size", "nonfinite", "raises", "writes_malformed_protocol",
])
def test_worker_failure_is_observable_and_process_is_reaped(policy_module, function):
    worker = _worker(policy_module, function)
    with pytest.raises(PolicyWorkerError):
        with worker:
            worker.act(_packet())
    assert worker.closed and not worker.is_alive and worker.exit_code is not None
    worker.close()
    with pytest.raises(PolicyWorkerError, match="not running"):
        worker.act(_packet())


def test_worker_timeout_kills_and_reaps_hung_policy(policy_module):
    worker = _worker(policy_module, "slow", action_size=1, timeout_s=10)
    with worker:
        worker.timeout_s = 0.1
        with pytest.raises(PolicyWorkerTimeout, match="deadline"):
            worker.act(_packet())
        assert worker.closed and not worker.is_alive and worker.exit_code is not None


def test_worker_startup_failure_and_context_exception_cleanup(policy_module):
    worker = _worker(policy_module, "missing_function")
    with pytest.raises(PolicyWorkerError):
        with worker:
            pytest.fail("missing policy unexpectedly initialized")
    assert worker.closed and not worker.is_alive and worker.exit_code is not None
    second = _worker(policy_module)
    with pytest.raises(RuntimeError, match="parent exception"):
        with second:
            raise RuntimeError("parent exception")
    assert second.closed and not second.is_alive and second.exit_code is not None
    with pytest.raises(PolicyWorkerError, match="only once"):
        second.__enter__()


def test_startup_timeout_reaps_child_and_removes_private_working_directory(tmp_path):
    (tmp_path / "slow_start_policy.py").write_text(
        "import time\ntime.sleep(20)\ndef act(packet): return None\n",
        encoding="utf-8", newline="\n",
    )
    worker = PolicyWorker(policy="slow_start_policy:act", declaration=_declaration(),
                          action_size=1, policy_paths=(tmp_path,), timeout_s=0.1)
    with pytest.raises(PolicyWorkerTimeout):
        with worker:
            pytest.fail("slow startup unexpectedly finished")
    assert worker.closed and not worker.is_alive and worker.exit_code is not None
    assert worker._temporary is not None and not Path(worker._temporary.name).exists()
    assert worker._reader is not None and not worker._reader.is_alive()
    assert worker._writer is not None and not worker._writer.is_alive()


def test_deadline_also_bounds_a_blocked_large_pipe_write(policy_module):
    path, _ = policy_module
    declaration = SensorDeclaration(channels=(SensorChannel(
        name="encoders", kind="joint_position", shape=(65536,), unit="radian",
    ),))
    projector = ObservationProjector(declaration)
    packet = projector.project(sequence=0, time_s=0.0, readings={"encoders": (0.0,) * 65536})
    worker = PolicyWorker(policy="boundary_test_policy:stop_reading_input",
                          declaration=declaration, action_size=2, policy_paths=(path,))
    with worker:
        assert worker.act(packet).values == (1, 2)
        worker.timeout_s = 0.2
        with pytest.raises(PolicyWorkerTimeout, match="input deadline"):
            worker.act(packet.model_copy(update={"sequence": 1}))
        assert worker.closed and not worker.is_alive and worker.exit_code is not None
        assert worker._writer is not None and not worker._writer.is_alive()
        assert worker._reader is not None and not worker._reader.is_alive()


@pytest.mark.parametrize("overrides", [
    {"policy": "not a policy"}, {"action_size": 0}, {"action_size": True},
    {"timeout_s": 0}, {"timeout_s": math.inf}, {"timeout_s": True},
])
def test_invalid_worker_setup_rejected_without_process(overrides):
    arguments = dict(policy="some_module:act", declaration=_declaration(), action_size=2)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        PolicyWorker(**arguments)
