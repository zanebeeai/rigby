"""The neutral skill tree bound to a body on physics.

The core executor never sees a body; this binds its leaves to the G06
transfer primitive in the registered fixed world and runs the five-deep
example tree on real bodies. Pinned: a selector whose first alternative the
guard refuses before motion falls through to the second and the tree
succeeds on one continuous physics record; an interruption from outside
stops the running transfer where it is and every node above it records it;
and a body whose every alternative is refused fails typed without moving.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rigby_core.skills import Belief, Interrupt, NodeKind, Verdict, execute
from rigby_core.skills.examples import clear_bench_library

from rigby_general.contact.placement import PlacementGoal
from rigby_general.scenes.environment import EnvironmentV1
from rigby_general.skills import BodySession, TransferRuntime, predicates_for
from rigby_general.skills.transfer_runtime import SimulationClock


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"
PROTOCOL = Path(__file__).resolve().parents[1] / "assets" / "general" / "research-protocols" / "g06-transfer-v1"


@pytest.fixture(scope="module")
def fixed_world():
    env = EnvironmentV1.model_validate_json((PROTOCOL / "environment.json").read_bytes())
    raw = json.loads((PROTOCOL / "goal.json").read_bytes())
    goal = PlacementGoal(region_minimum_m=tuple(raw["region_minimum_m"]), region_maximum_m=tuple(raw["region_maximum_m"]), dwell_s=raw["dwell_s"],
                         maximum_linear_speed_mps=raw["maximum_linear_speed_mps"], maximum_angular_speed_radps=raw["maximum_angular_speed_radps"])
    return env, goal


def run_tree(zoo_id: str, fixed_world, *, effector: str, alternate: str, stop_at_s: float | None = None):
    env, goal = fixed_world
    session = BodySession.open(zoo_id, ZOO_ROOT / zoo_id / "robot.urdf", env, goal)
    library = clear_bench_library()
    tree = library.expand("clear_bench", {"object": "cube", "destination": "platform", "effector": effector, "alternate": alternate})
    interrupt = Interrupt()
    runtime = TransferRuntime(session, interrupt, stop_at_s=stop_at_s)
    record = execute(tree, library, runtime, predicates_for(session), clock=SimulationClock(session), interrupt=interrupt, belief=Belief())
    return session, runtime, tree, record


def test_a_refused_first_alternative_falls_through_to_one_that_transfers(fixed_world) -> None:
    env, goal = fixed_world
    probe = BodySession.open("zoo_dual_arm", ZOO_ROOT / "zoo_dual_arm" / "robot.urdf", env, goal)
    chains = sorted(probe.effectors)
    assert len(chains) == 2, "the bimanual body has two grasping effectors"
    session, runtime, tree, record = run_tree("zoo_dual_arm", fixed_world, effector=chains[1], alternate=chains[0])
    assert tree.max_depth == 5
    assert record.verdict is Verdict.SUCCESS, record.root.reason
    selector = record.record("0.1.0.0")
    first, second = selector.children
    assert first.verdict is Verdict.FAILURE and first.reason in ("self_collision_path", "unreachable_path"), "refused by the guard, typed"
    assert runtime.calls[1]["executed"] is False, "a refusal moves nothing"
    assert second.verdict is Verdict.SUCCESS and runtime.calls[2]["executed"] and runtime.calls[2]["certified"]
    assert record.record("0.1.0.1").verdict is Verdict.SUCCESS, "the verifying observation read the evaluator's dwell"
    assert record.record("0.1").reason == "until_after_1"
    assert record.belief["placed:cube:platform"] is True
    times = session.recorder.rows["time_s"]
    assert len(times) > 0 and np.all(np.diff(np.asarray(times)) > 0), "one continuous physics record"
    assert record.ended_s == pytest.approx(float(times[-1]), abs=1e-9), "the executor's clock is the physics clock"


def test_an_interruption_from_outside_stops_the_transfer_where_it_is(fixed_world) -> None:
    env, goal = fixed_world
    probe = BodySession.open("zoo_jaw_arm", ZOO_ROOT / "zoo_jaw_arm" / "robot.urdf", env, goal)
    only = sorted(probe.effectors)[0]
    session, runtime, tree, record = run_tree("zoo_jaw_arm", fixed_world, effector=only, alternate=only, stop_at_s=6.0)
    assert record.verdict is Verdict.INTERRUPTED and record.interrupted
    leaf = record.record("0.1.0.0.0")
    assert leaf.verdict is Verdict.INTERRUPTED and leaf.reason == "interrupted"
    assert runtime.calls[1]["interrupted"] and runtime.calls[1]["phases"][-1] in ("lift", "hold", "close", "descend", "turn", "approach")
    for node_id in ("0.1.0.0", "0.1.0", "0.1", "0"):
        assert record.record(node_id).verdict is Verdict.INTERRUPTED, node_id
    assert len(runtime.calls) == 2, "nothing runs after the interruption"
    last_result = session.last_result
    assert last_result is not None and last_result.interrupted and last_result.failed_gate == "interrupted"
    assert 6.0 <= last_result.final_time_s < 6.01, "stopped on the physics step the request arrived"
    assert record.ended_s == pytest.approx(last_result.final_time_s)


def test_a_second_leaf_continues_the_world_the_first_left_on_one_replayable_record(fixed_world) -> None:
    """The multifinger hand loses the cube on its first attempt; the tree's
    second alternative and second loop attempt start from the arm where it
    stands and the cube where it went, on the same physics clock, and the
    recorded controls replay to the recorded states across every boundary."""

    from rigby_core.simulation.recording import replay_physics

    env, goal = fixed_world
    probe = BodySession.open("zoo_hand_arm", ZOO_ROOT / "zoo_hand_arm" / "robot.urdf", env, goal)
    only = sorted(probe.effectors)[0]
    session, runtime, tree, record = run_tree("zoo_hand_arm", fixed_world, effector=only, alternate=only)
    assert record.verdict is Verdict.FAILURE and record.record("0.1").reason in ("budget_exhausted", "no_progress")
    executed = [call for call in runtime.calls if "effector" in call and call["executed"]]
    assert len(executed) >= 2, "at least two leaves moved the body"
    results = [r for _, r in session.results if r.executed]
    for earlier, later in zip(results, results[1:]):
        assert later.times_s[0] == pytest.approx(earlier.final_time_s), "the clock continues"
        assert np.allclose(later.qpos[0], earlier.qpos[-1]), "the world continues"
    times = np.asarray(session.recorder.rows["time_s"])
    assert np.all(np.diff(times) > 0)
    assert replay_physics(session.model, session.recorder.finish())["agrees"]


def test_a_body_refused_on_every_alternative_fails_typed_without_moving(fixed_world) -> None:
    env, goal = fixed_world
    probe = BodySession.open("zoo_compact_arm", ZOO_ROOT / "zoo_compact_arm" / "robot.urdf", env, goal)
    only = sorted(probe.effectors)[0]
    session, runtime, tree, record = run_tree("zoo_compact_arm", fixed_world, effector=only, alternate=only)
    assert record.verdict is Verdict.FAILURE
    # The inventory sees the cube outside the compact arm's measured envelope,
    # so after the first attempt the loop's progress predicate fails it: a
    # truer reason than spending the budget on refusals.
    assert record.record("0.1").reason == "no_progress"
    assert record.belief["reach:cube"] is False
    leaves = [r for r in record.root.walk() if r.kind is NodeKind.PRIMITIVE]
    assert len(leaves) == 2 and all(r.verdict is Verdict.FAILURE and r.reason == "unreachable_path" for r in leaves)
    assert not session.recorder.rows["time_s"], "nothing moved"
    assert all(not call.get("executed", True) for call in runtime.calls if "effector" in call)
