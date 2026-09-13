"""The production entry point applies the registered modes before policy steps."""

import json
from pathlib import Path

import pytest
from rigby_core.hashing import hash_file

from rigby_general.benchmark.runner import run_trial
from rigby_general.benchmark.contracts import BenchmarkProtocolV1
from rigby_general.benchmark.protocol import register_protocol
from rigby_general.benchmark.world import BenchmarkRefusal

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "assets/general/research-protocols/transfer-bench-v1-r1"
BODY = ROOT / "assets/general/zoo/zoo_compact_arm/robot.urdf"


def trial(tmp_path, **kwargs):
    return run_trial(registration=REGISTRATION, expected_registration_sha256=hash_file(REGISTRATION / "registration.json"),
                     robot_source=BODY, split_id="development", seed=0,
                     policy="rigby_general.benchmark.policies:zero_command_probe", destination=tmp_path / "trial",
                     probe_steps=2, **kwargs)


def test_actual_policy_probe_keeps_modes_and_truth_separate(tmp_path):
    strict_dir, normalized_dir = tmp_path / "strict", tmp_path / "normalized"
    strict = trial(strict_dir, mode="strict_fixed_world")
    normalized = trial(normalized_dir, mode="capability_normalized")
    assert strict["status"] == normalized["status"] == "protocol_probe_complete"
    assert not strict["scored"] and not normalized["scored"]
    assert not strict["root_success"] and not normalized["root_success"]
    assert strict["world_sha256"] != normalized["world_sha256"]
    for directory in (strict_dir, normalized_dir):
        root = directory / "trial"
        result = json.loads((root / "result.json").read_bytes())
        policy_packet = json.loads((root / "initial-policy-observation.json").read_bytes())
        truth = json.loads((root / "final-evaluator-truth.json").read_bytes())
        assert result["worker_closed_before_privileged_output"]
        assert policy_packet["access"] == "declared_sensors"
        assert not {"object_poses", "success_labels", "simulator_state"} & policy_packet.keys()
        assert truth["access"] == "evaluator_only" and truth["object_poses"]
        assert result["physical_steps"] == 2 and result["elapsed_simulation_s"] == pytest.approx(0.004)
    with pytest.raises(FileExistsError): trial(strict_dir, mode="strict_fixed_world")


def test_wrong_registration_digest_prevents_any_policy_execution(tmp_path):
    with pytest.raises(BenchmarkRefusal) as error:
        run_trial(registration=REGISTRATION, expected_registration_sha256="0"*64, robot_source=BODY,
                  split_id="development", seed=0, mode="strict_fixed_world",
                  policy="rigby_general.benchmark.policies:zero_command_probe", destination=tmp_path / "must-not-exist", probe_steps=2)
    assert error.value.code == "registration_changed"
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("cap, success, elapsed", [(0.0095, False, 0.008), (0.01, True, 0.01)])
def test_success_must_occur_within_the_registered_simulation_budget(tmp_path, cap, success, elapsed):
    raw = json.loads((REGISTRATION / "benchmark-protocol.v1.json").read_bytes())
    raw["world"]["environment"]["objects"][0]["position_m"] = [0.8, 0, 0.2]
    raw["world"]["physics"]["gravity_mps2"] = [0, 0, 0]
    raw["world"]["sensor_policy"]["rgb"] = False
    raw["world"]["goal"]["target"] = {"minimum_m": [0.75, -0.05, 0.15], "maximum_m": [0.85, 0.05, 0.25]}
    raw["world"]["goal"]["dwell_s"] = 0.009
    raw["budget"]["maximum_simulation_s"] = cap
    registration = tmp_path / "registration"
    register_protocol(BenchmarkProtocolV1.model_validate(raw), registration)
    result = run_trial(registration=registration, expected_registration_sha256=hash_file(registration / "registration.json"),
                       robot_source=BODY, split_id="development", seed=0, mode="strict_fixed_world",
                       policy="rigby_general.benchmark.policies:zero_command_probe", destination=tmp_path / "trial")
    assert result["root_success"] is success
    assert result["status"] == ("root_success" if success else "simulation_budget_exhausted")
    assert result["elapsed_simulation_s"] == pytest.approx(elapsed)
    assert result["elapsed_simulation_s"] <= cap + 1e-12
    assert not result["scored"]
