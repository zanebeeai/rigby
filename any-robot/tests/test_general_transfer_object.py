"""The closed-loop TransferObject skill on physics: recoveries from sensed evidence, retries bounded, one record.

Three episodes of the jaw arm in the G10 world: nominal, an induced slip
early in the carry, and a temporary occlusion during the placement
verification. Pinned: the skill completes only when the stably-placed
conditional decides pass from the sensors; a lost hold stops the carry
where it is and the next acquisition plans to where the camera then sees
the object, still, not to where the world was authored to put it; an
occluded verification is undecided and re-observed within the fallback
budget, never passed; every leaf continues one physics record that
replays; the object's recorded motion is continuous (nothing teleported
it); and no loop exceeds its budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rigby_core.skills import Belief, Interrupt, Verdict, execute
from rigby_core.skills.examples import RETRY_BUDGET, transfer_object_library
from rigby_core.simulation.recording import replay_physics

from rigby_general.contact.placement import PlacementGoal
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.sensing import load_policy
from rigby_general.skills import TransferObjectRuntime, TransferObjectSession, disturbance_named, g10_world, predicates_for_transfer
from rigby_general.skills.transfer_runtime import SimulationClock


ROOT = Path(__file__).resolve().parents[1]
G06 = ROOT / "assets" / "general" / "research-protocols" / "g06-transfer-v1"
POLICY = ROOT / "assets" / "general" / "research-protocols" / "g09-conditionals-v1" / "policy.json"
SOURCE = ROOT / "assets" / "general" / "zoo" / "zoo_jaw_arm" / "robot.urdf"


@pytest.fixture(scope="module")
def world():
    env = g10_world(EnvironmentV1.model_validate_json((G06 / "environment.json").read_bytes()))
    raw = json.loads((G06 / "goal.json").read_bytes())
    goal = PlacementGoal(region_minimum_m=tuple(raw["region_minimum_m"]), region_maximum_m=tuple(raw["region_maximum_m"]), dwell_s=raw["dwell_s"],
                         maximum_linear_speed_mps=raw["maximum_linear_speed_mps"], maximum_angular_speed_radps=raw["maximum_angular_speed_radps"])
    return env, goal, load_policy(POLICY)


def run(world, kind: str):
    env, goal, policy = world
    session = TransferObjectSession.open("zoo_jaw_arm", SOURCE, env, goal, policy, disturbance=disturbance_named(kind), seed_label=f"test-{kind}")
    library = transfer_object_library()
    tree = library.expand("transfer_object", {"object": "cube", "destination": "platform", "effector": session.effector.chain_id})
    interrupt = Interrupt()
    runtime = TransferObjectRuntime(session, interrupt)
    record = execute(tree, library, runtime, predicates_for_transfer(), clock=SimulationClock(session), interrupt=interrupt, belief=Belief())
    return session, runtime, record


@pytest.fixture(scope="module")
def nominal(world):
    return run(world, "nominal")


@pytest.fixture(scope="module")
def slip(world):
    return run(world, "slip")


@pytest.fixture(scope="module")
def occlusion(world):
    return run(world, "occlusion")


def leaves(runtime, name: str):
    return [c for c in runtime.calls if c["leaf"] == name]


def test_nominal_completes_on_sensor_verdicts_within_the_cap(nominal) -> None:
    session, runtime, record = nominal
    assert record.verdict is Verdict.SUCCESS
    assert session.time_s <= 120.0
    placement = leaves(runtime, "verify_placement")[-1]["trail"][-1]
    assert placement["decision"] == "pass" and "camera:front" in placement["sensors"]
    hold = leaves(runtime, "verify_hold")[-1]["trail"][-1]
    assert hold["decision"] == "pass" and "contact:palm" in hold["sensors"]
    assert record.belief["placed:cube:platform"] is True and record.belief["held:cube"] is False


def test_acquire_plans_to_the_sensed_position_not_the_authored_one(nominal, world) -> None:
    session, runtime, _ = nominal
    env, _, _ = world
    planned = np.asarray(leaves(runtime, "acquire")[0]["planned_to"])
    authored = np.asarray(env.objects[0].position_m)
    assert np.linalg.norm(planned - authored) < 0.02, "the sensed position is the authored one to within the camera's noise"
    assert not np.array_equal(planned, authored), "and it is the sensor's reading, not the authoring"


def test_a_lost_hold_stops_the_carry_and_the_next_attempt_plans_to_where_the_object_fell(slip) -> None:
    session, runtime, record = slip
    assert session.disturbance.log and session.disturbance.log[0]["kind"] == "slip"
    carries = leaves(runtime, "transport")
    assert carries[0]["verdict"] == "failure" and carries[0]["reason"] == "hold_lost"
    assert any(e["event"] == "hold_lost" for e in session.events)
    acquisitions = leaves(runtime, "acquire")
    assert len(acquisitions) >= 2
    first, second = np.asarray(acquisitions[0]["planned_to"]), np.asarray(acquisitions[1]["planned_to"])
    assert np.linalg.norm(second - first) > 0.004, "the object moved when it fell, and the plan followed the sensor"
    # Where the object lands after a slip is physics; whether the skill
    # recovers is the campaign's question. What is pinned here is honesty:
    # a success is claimed only with the placement verified from sensors,
    # and no loop exceeds its budget on the way to whatever verdict.
    assert record.verdict in (Verdict.SUCCESS, Verdict.FAILURE, Verdict.UNKNOWN)
    assert (record.verdict is Verdict.SUCCESS) == bool(record.belief.get("placed:cube:platform", False))
    if record.verdict is Verdict.SUCCESS:
        assert leaves(runtime, "verify_placement")[-1]["trail"][-1]["decision"] == "pass"
    for node in record.root.walk():
        if node.skill_id in ("place_until_placed", "acquire_until_held"):
            assert node.evidence.get("attempts", node.attempts) <= RETRY_BUDGET


def test_an_occluded_verification_is_undecided_then_decided_never_passed_blind(occlusion) -> None:
    session, runtime, record = occlusion
    trail = leaves(runtime, "verify_placement")[0]["trail"]
    assert trail[0]["decision"] == "unknown" and trail[0]["reason"] == "occluded:camera:front"
    assert trail[-1]["decision"] == "pass"
    assert len(trail) - 1 <= session.conditionals["stably_placed"].fallback.budget
    assert record.verdict is Verdict.SUCCESS


def test_every_episode_is_one_replayable_record_and_the_object_was_never_teleported(nominal, slip, occlusion) -> None:
    for session, _, _ in (nominal, slip, occlusion):
        physical = session.recorder.finish()
        assert replay_physics(session.model, physical)["agrees"]
        address = session.model.jnt_qposadr[session.model.joint("scene_block_free").id]
        positions = physical.arrays["qpos"][:, address: address + 3]
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        assert float(steps.max()) < 0.01, "no recorded step moves the object more than free fall could at this timestep"
        assert float(np.diff(physical.arrays["time_s"]).min()) > 0.0


def test_predicates_never_established_are_false_not_unknown() -> None:
    predicates = predicates_for_transfer()
    empty = Belief()
    assert predicates["object_held"](empty, ("cube",)) is False
    assert predicates["object_placed"](empty, ("cube", "platform")) is False
    assert predicates["object_in_reach"](empty, ("cube",)) is False
