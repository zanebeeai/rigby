"""Pre-registration is tied to externally pinned content and disjoint seeds."""

import json

import pytest
from pydantic import ValidationError
from rigby_core.hashing import content_hash, hash_file

from rigby_general.benchmark.contracts import (
    BenchmarkProtocolV1, BenchmarkWorldV1, RootGoalV1, RegionV1, PublicSplitV1,
)
from rigby_general.benchmark.protocol import load_registration, public_variant, register_protocol, split_hashes
from rigby_general.benchmark.world import BenchmarkRefusal, compile_world
from rigby_general.scenes.environment import load_environment


def protocol_fixture():
    return BenchmarkProtocolV1(protocol_id="public-engineering-fixture", world=BenchmarkWorldV1(
        environment=load_environment("desk_bench"), goal=RootGoalV1(object_names=("cube",),
        target=RegionV1(minimum_m=(0.25, -0.05, 0.03), maximum_m=(0.35, 0.05, 0.15)))),
        splits=(PublicSplitV1(split_id="development", purpose="development", seeds=(0, 1)),
                PublicSplitV1(split_id="engineering", purpose="engineering_test", seeds=(2, 3))))


def test_registration_precedes_execution_and_tampering_invalidates_it(tmp_path):
    directory = tmp_path / "registration"
    protocol = protocol_fixture()
    record = register_protocol(protocol, directory)
    digest = hash_file(directory / "registration.json")
    assert not record["scored_runs_started"] and not record["existing_sealed_manifests_read"]
    assert load_registration(directory, expected_registration_sha256=digest) == protocol
    with pytest.raises(FileExistsError): register_protocol(protocol, directory)
    path = directory / "benchmark-protocol.v1.json"
    changed = json.loads(path.read_bytes())
    changed["world"]["goal"]["dwell_s"] = 0.1
    path.write_text(json.dumps(changed), encoding="utf-8", newline="\n")
    with pytest.raises(BenchmarkRefusal) as error:
        load_registration(directory, expected_registration_sha256=digest)
    assert error.value.code == "protocol_changed"


def test_seed_and_split_changes_require_a_new_registration():
    protocol = protocol_fixture()
    first, draw = public_variant(protocol, split_id="engineering", seed=2)
    repeated, repeated_draw = public_variant(protocol, split_id="engineering", seed=2)
    assert first == repeated and draw == repeated_draw
    assert compile_world(first).world_sha256 == compile_world(repeated).world_sha256
    other, _ = public_variant(protocol, split_id="engineering", seed=3)
    assert content_hash(other) != content_hash(first)
    assert draw["split_sha256"] == split_hashes(protocol)["engineering"]
    with pytest.raises(BenchmarkRefusal): public_variant(protocol, split_id="development", seed=2)
    raw = protocol.model_dump(mode="json")
    raw["splits"][1]["seeds"][0] = 0
    with pytest.raises(ValidationError, match="disjoint"):
        BenchmarkProtocolV1.model_validate(raw)


@pytest.mark.parametrize("mutation", ["short_budget", "weaken_release", "nonfinite_world", "missing_goal_object", "overlap_ids"])
def test_invalid_goal_and_protocol_contracts_refuse(mutation):
    raw = protocol_fixture().model_dump(mode="json")
    if mutation == "short_budget": raw["budget"]["maximum_simulation_s"] = 1
    elif mutation == "weaken_release": raw["world"]["goal"]["require_release"] = False
    elif mutation == "nonfinite_world": raw["world"]["environment"]["objects"][0]["position_m"][0] = float("nan")
    elif mutation == "missing_goal_object": raw["world"]["goal"]["object_names"] = ["unknown"]
    else: raw["splits"][1]["split_id"] = raw["splits"][0]["split_id"]
    with pytest.raises((ValidationError, ValueError)):
        BenchmarkProtocolV1.model_validate(raw)
