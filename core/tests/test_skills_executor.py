"""Success, failure, unknown and interruption at every node kind, recorded.

The runtime here is a scripted toy: it answers each leaf from a table, on a
clock only the leaves advance. What is pinned is the executor's semantics --
which verdict each composite reaches from its children's, what an
undecidable predicate does, when a budget or a timeout stops a loop, how an
interruption propagates -- and that every one of those decisions is in the
record with the predicate or leaf that made it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from rigby_core.skills import (
    Belief,
    ChildRefV1,
    ExecutionRecordV1,
    Interrupt,
    LeafOutcome,
    LoopSpecV1,
    MonotonicClock,
    NodeKind,
    ObservationSpecV1,
    OnUnknown,
    PredicateSpecV1,
    RecoveryV1,
    ResourceClaimV1,
    SkillDefinitionV1,
    SkillExecutionError,
    SkillLibraryV1,
    TerminationRuleV1,
    Verdict,
    execute,
)
from rigby_core.skills.examples import ARGUMENTS, A, P, clear_bench_library, with_recovery


@dataclass
class ScriptedRuntime:
    """Leaf answers keyed by (skill, effector) for primitives and by skill for
    observations; ``None`` for an observation means nothing could be seen."""

    primitives: dict[tuple[str, str], tuple[Verdict, bool]] = field(default_factory=dict)
    observations: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    leaf_seconds: float = 10.0
    stop_on_call: int | None = None
    interrupt: Interrupt | None = None
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run_primitive(self, context) -> LeafOutcome:
        node = context.node
        self.calls.append(("primitive", node.node_id, node.arguments.get("effector", "")))
        if self.stop_on_call is not None and len(self.calls) >= self.stop_on_call and self.interrupt is not None:
            self.interrupt.request("stopped from outside")
        context.clock.advance(self.leaf_seconds)
        if context.should_stop():
            return LeafOutcome(Verdict.INTERRUPTED, "stopped mid-motion")
        verdict, placed = self.primitives.get((node.skill_id, node.arguments.get("effector", "")), (Verdict.FAILURE, False))
        facts = {} if placed is None else {"placed:cube:platform": placed}
        return LeafOutcome(verdict, "" if verdict is Verdict.SUCCESS else "grasp_not_achieved", facts=facts, evidence={"effector": node.arguments.get("effector")})

    def observe(self, context):
        node = context.node
        self.calls.append(("observe", node.node_id))
        context.clock.advance(0.5)
        if node.skill_id in self.observations:
            answer = self.observations[node.skill_id]
            return None if answer is None else dict(answer)
        if node.skill_id == "observe_inventory":
            return {"known:cube": True, "reach:cube": True, "placed:cube:platform": False}
        return {"placed:cube:platform": context.belief.get("placed:cube:platform")}


PREDICATES = {
    "object_known": lambda belief, args: belief.get("known:" + args[0]),
    "object_placed": lambda belief, args: belief.get("placed:" + args[0] + ":" + args[1]),
    "object_in_reach": lambda belief, args: belief.get("reach:" + args[0]),
}


def run(runtime, library=None, interrupt=None, clock=None):
    library = library or clear_bench_library()
    tree = library.expand("clear_bench", ARGUMENTS)
    return execute(tree, library, runtime, PREDICATES, clock=clock or MonotonicClock(), interrupt=interrupt)


def library_where_only_the_observation_decides_placement(**loop_termination) -> SkillLibraryV1:
    """The transfer leaf claims no effect of its own, so the placement fact
    comes only from the observation that follows it."""

    payload = json.loads(clear_bench_library().model_dump_json())
    for definition in payload["skills"]:
        if definition["skill_id"] == "transfer":
            definition["effects"] = []
        if definition["skill_id"] == "place_until_done" and loop_termination:
            definition["termination"] = loop_termination
    return SkillLibraryV1.model_validate(payload)


# --------------------------------------------------------------------------
# the example, end to end
# --------------------------------------------------------------------------


def test_the_first_alternative_carries_the_whole_tree() -> None:
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.SUCCESS, True)})
    record = run(runtime)
    assert record.verdict is Verdict.SUCCESS
    assert [c[0:2] for c in runtime.calls] == [("observe", "0.0"), ("primitive", "0.1.0.0.0"), ("observe", "0.1.0.1")]
    assert record.record("0.1.0.0").reason == "alternative:0.1.0.0.0"
    assert record.record("0.1").reason == "until_after_1" and record.record("0.1").evidence == {"attempts": 1}
    assert all(check.answer is True for check in record.root.effects)
    assert record.ended_s == pytest.approx(11.0)


def test_a_selector_falls_through_to_the_next_alternative() -> None:
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.FAILURE, False), ("transfer", "secondary"): (Verdict.SUCCESS, True)})
    record = run(runtime)
    assert record.verdict is Verdict.SUCCESS
    selector = record.record("0.1.0.0")
    assert [c.verdict for c in selector.children] == [Verdict.FAILURE, Verdict.SUCCESS]
    assert selector.children[0].reason == "grasp_not_achieved" and selector.reason == "alternative:0.1.0.0.1"


def test_a_loop_spends_its_budget_and_fails_typed() -> None:
    runtime = ScriptedRuntime()
    record = run(runtime)
    assert record.verdict is Verdict.FAILURE
    loop = record.record("0.1")
    assert loop.reason == "budget_exhausted" and loop.attempts == 1 and len(loop.children) == 2
    assert [c.answer for c in loop.loop_checks if c.predicate == "object_placed"] == [False, False, False]
    assert record.root.reason == "child_failed:0.1"
    assert sum(1 for c in runtime.calls if c[0] == "primitive") == 4, "two alternatives, twice"


def test_a_loop_stops_when_progress_is_lost() -> None:
    runtime = ScriptedRuntime(observations={"observe_inventory": {"known:cube": True, "reach:cube": False, "placed:cube:platform": False}})
    record = run(runtime)
    assert record.record("0.1").reason == "no_progress"
    assert record.verdict is Verdict.FAILURE


# --------------------------------------------------------------------------
# unknown is neither success nor failure
# --------------------------------------------------------------------------


def test_an_observation_that_sees_nothing_leaves_the_tree_undecided() -> None:
    library = library_where_only_the_observation_decides_placement()
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.SUCCESS, None)}, observations={"observe_placement": None})
    record = run(runtime, library)
    assert record.record("0.1.0.0.0").verdict is Verdict.SUCCESS, "the leaf did what it was asked"
    assert record.record("0.1.0.1").verdict is Verdict.UNKNOWN and record.record("0.1.0.1").reason == "nothing_observed"
    assert record.record("0.1.0").verdict is Verdict.UNKNOWN, "a sequence propagates undecided"
    assert record.record("0.1").verdict is Verdict.UNKNOWN and record.record("0.1").reason == "child_undecided:0.1.0"
    assert record.verdict is Verdict.UNKNOWN
    assert sum(1 for c in runtime.calls if c[0] == "primitive") == 1, "an undecided attempt is not repeated"


def test_a_termination_rule_may_count_unknown_against_the_node() -> None:
    library = library_where_only_the_observation_decides_placement(require_effects=True, on_unknown="fail")
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.SUCCESS, None)}, observations={"observe_placement": None})
    record = run(runtime, library)
    loop = record.record("0.1")
    assert loop.verdict is Verdict.FAILURE and loop.reason == "budget_exhausted"
    assert [c.verdict for c in loop.children] == [Verdict.UNKNOWN, Verdict.UNKNOWN], "undecided attempts counted against the budget"


def test_an_observation_whose_effect_cannot_be_checked_is_undecided() -> None:
    runtime = ScriptedRuntime(observations={"observe_inventory": {"reach:cube": True, "placed:cube:platform": False}})
    record = run(runtime)
    inventory = record.record("0.0")
    assert inventory.verdict is Verdict.UNKNOWN and inventory.reason == "effects_undecidable"
    assert inventory.effects[0].predicate == "object_known" and inventory.effects[0].answer is None
    assert record.verdict is Verdict.UNKNOWN and runtime.calls == [("observe", "0.0")]


def test_an_undecidable_initiation_is_recorded_as_such() -> None:
    payload = json.loads(clear_bench_library().model_dump_json())
    for definition in payload["skills"]:
        if definition["skill_id"] == "observe_inventory":
            definition["effects"] = []
    library = SkillLibraryV1.model_validate(payload)
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.SUCCESS, True)}, observations={"observe_inventory": {"reach:cube": True, "placed:cube:platform": False}})
    record = run(runtime, library)
    loop = record.record("0.1")
    assert loop.verdict is Verdict.UNKNOWN and loop.reason == "initiation_undecidable"
    assert loop.initiation[0].predicate == "object_known" and loop.initiation[0].answer is None
    assert not any(c[0] == "primitive" for c in runtime.calls)


# --------------------------------------------------------------------------
# effects, invariants, recovery, timeouts, interruption
# --------------------------------------------------------------------------


def test_a_leaf_that_claims_success_without_its_effect_fails() -> None:
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.SUCCESS, False), ("transfer", "secondary"): (Verdict.SUCCESS, False)})
    record = run(runtime)
    transfer = record.record("0.1.0.0.0")
    assert transfer.verdict is Verdict.FAILURE and transfer.reason == "effects_unmet"
    assert transfer.effects[0].predicate == "object_placed" and transfer.effects[0].answer is False


def test_a_recovery_runs_between_bounded_attempts() -> None:
    library = with_recovery(clear_bench_library())
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.FAILURE, False), ("transfer", "secondary"): (Verdict.SUCCESS, True)})
    record = run(runtime, library)
    first = record.record("0.1.0.0.0")
    assert first.attempts == 2 and first.verdict is Verdict.FAILURE
    assert [r.skill_id for r in first.recoveries] == ["observe_inventory"] and first.recoveries[0].verdict is Verdict.SUCCESS
    assert record.verdict is Verdict.SUCCESS


def test_a_timeout_fails_the_node_that_overran() -> None:
    payload = json.loads(clear_bench_library().model_dump_json())
    for definition in payload["skills"]:
        if definition["skill_id"] == "transfer":
            definition["timeout_s"] = 5.0
    library = SkillLibraryV1.model_validate(payload)
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.SUCCESS, True)}, leaf_seconds=10.0)
    record = run(runtime, library)
    assert record.record("0.1.0.0.0").reason == "timeout" and record.record("0.1.0.0.0").verdict is Verdict.FAILURE


def test_an_interruption_stops_the_tree_where_it_was() -> None:
    interrupt = Interrupt()
    runtime = ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.SUCCESS, True)}, stop_on_call=2, interrupt=interrupt)
    record = run(runtime, interrupt=interrupt)
    assert record.verdict is Verdict.INTERRUPTED and record.interrupted and record.interrupt_reason == "stopped from outside"
    assert record.record("0.1.0.0.0").verdict is Verdict.INTERRUPTED and record.record("0.1.0.0.0").reason == "stopped mid-motion"
    for node_id in ("0.1.0.0", "0.1.0", "0.1", "0"):
        assert record.record(node_id).verdict is Verdict.INTERRUPTED, node_id
    assert not any(c[1] == "0.1.0.0.1" for c in runtime.calls), "no alternative is tried after an interruption"
    assert not any(c[1] == "0.1.0.1" for c in runtime.calls)


def test_an_interruption_before_a_node_starts_is_recorded_without_running_it() -> None:
    interrupt = Interrupt()
    interrupt.request("before start")
    runtime = ScriptedRuntime()
    record = run(runtime, interrupt=interrupt)
    assert record.verdict is Verdict.INTERRUPTED and runtime.calls == []
    assert record.root.reason == "before start"


# --------------------------------------------------------------------------
# every kind, all four verdicts
# --------------------------------------------------------------------------


def small_library(kind: NodeKind, *, loop=None, on_unknown=OnUnknown.PROPAGATE) -> SkillLibraryV1:
    children = (ChildRefV1(skill="leaf_a", rank=0), ChildRefV1(skill="leaf_b", rank=1)) if kind is not NodeKind.REPEAT_UNTIL else (ChildRefV1(skill="leaf_a"),)
    resources = (ResourceClaimV1(resource="slot:a"), ResourceClaimV1(resource="slot:b"))
    return SkillLibraryV1(
        library_id="small", predicates=(PredicateSpecV1(name="done"),),
        skills=(
            SkillDefinitionV1(skill_id="root", kind=kind, timeout_s=100.0, resources=resources, children=children, loop=loop,
                              termination=TerminationRuleV1(require_effects=False, on_unknown=on_unknown)),
            SkillDefinitionV1(skill_id="leaf_a", kind=NodeKind.PRIMITIVE, timeout_s=50.0, controller="c", resources=(ResourceClaimV1(resource="slot:a"),),
                              termination=TerminationRuleV1(require_effects=False)),
            SkillDefinitionV1(skill_id="leaf_b", kind=NodeKind.PRIMITIVE, timeout_s=50.0, controller="c", resources=(ResourceClaimV1(resource="slot:b"),),
                              termination=TerminationRuleV1(require_effects=False)),
        ),
    )


@dataclass
class TableRuntime:
    answers: dict[str, list[Verdict]]
    interrupt: Interrupt | None = None
    done_after: int | None = None
    calls: int = 0

    def run_primitive(self, context) -> LeafOutcome:
        self.calls += 1
        context.clock.advance(1.0)
        answers = self.answers[context.node.skill_id]
        verdict = answers.pop(0) if len(answers) > 1 else answers[0]
        if verdict is Verdict.INTERRUPTED and self.interrupt is not None:
            self.interrupt.request("table")
        facts = {"done": True} if self.done_after is not None and self.calls >= self.done_after else {}
        return LeafOutcome(verdict, verdict.value, facts=facts)

    def observe(self, context):
        return None


def run_small(kind, answers, *, loop=None, interrupt=None, done_after=None, on_unknown=OnUnknown.PROPAGATE, done=None):
    library = small_library(kind, loop=loop, on_unknown=on_unknown)
    tree = library.expand("root")
    runtime = TableRuntime({k: list(v) for k, v in answers.items()}, interrupt=interrupt, done_after=done_after)
    predicates = {"done": done or (lambda belief, args: belief.get("done", False))}
    return execute(tree, library, runtime, predicates, interrupt=interrupt), runtime


S, F, U, I = Verdict.SUCCESS, Verdict.FAILURE, Verdict.UNKNOWN, Verdict.INTERRUPTED


@pytest.mark.parametrize("a, b, expected, calls", [
    (S, S, S, 2), (S, F, F, 2), (F, S, F, 1), (S, U, U, 2), (U, S, U, 1), (S, I, I, 2), (I, S, I, 1),
])
def test_sequence_verdicts(a, b, expected, calls) -> None:
    interrupt = Interrupt()
    record, runtime = run_small(NodeKind.SEQUENCE, {"leaf_a": [a], "leaf_b": [b]}, interrupt=interrupt)
    assert record.verdict is expected and runtime.calls == calls


@pytest.mark.parametrize("a, b, expected, calls", [
    (S, F, S, 1), (F, S, S, 2), (F, F, F, 2), (U, F, U, 2), (F, U, U, 2), (U, S, S, 2), (I, S, I, 1), (F, I, I, 2),
])
def test_selector_verdicts(a, b, expected, calls) -> None:
    interrupt = Interrupt()
    record, runtime = run_small(NodeKind.SELECTOR, {"leaf_a": [a], "leaf_b": [b]}, interrupt=interrupt)
    assert record.verdict is expected and runtime.calls == calls


@pytest.mark.parametrize("a, b, expected", [(S, S, S), (S, F, F), (F, S, F), (S, U, U), (U, F, F), (I, S, I)])
def test_parallel_verdicts(a, b, expected) -> None:
    interrupt = Interrupt()
    record, runtime = run_small(NodeKind.PARALLEL, {"leaf_a": [a], "leaf_b": [b]}, interrupt=interrupt)
    assert record.verdict is expected
    assert record.root.children[0].resources == ("slot:a",)
    if expected is I:
        assert len(record.root.children) == 1, "nothing starts after an interruption"
    else:
        assert record.root.children[1].resources == ("slot:b",)


def test_repeat_until_success_failure_unknown_and_interruption() -> None:
    loop = LoopSpecV1(until=P("done"), max_attempts=3)
    record, runtime = run_small(NodeKind.REPEAT_UNTIL, {"leaf_a": [F, F, S]}, loop=loop, done_after=2)
    assert record.verdict is S and record.root.reason == "until_after_2" and runtime.calls == 2
    undecidable = lambda belief, args: None  # noqa: E731
    record, runtime = run_small(NodeKind.REPEAT_UNTIL, {"leaf_a": [F]}, loop=loop, done=undecidable)
    assert record.verdict is U and record.root.reason == "until_undecidable" and runtime.calls == 0, "an undecidable predicate never spends an attempt"
    record, runtime = run_small(NodeKind.REPEAT_UNTIL, {"leaf_a": [F]}, loop=loop, on_unknown=OnUnknown.FAIL, done=undecidable)
    assert record.verdict is F and record.root.reason == "until_undecidable"
    interrupt = Interrupt()
    record, runtime = run_small(NodeKind.REPEAT_UNTIL, {"leaf_a": [I]}, loop=loop, interrupt=interrupt)
    assert record.verdict is I and runtime.calls == 1


def test_a_loop_whose_predicate_is_decidable_but_false_spends_its_budget() -> None:
    library = small_library(NodeKind.REPEAT_UNTIL, loop=LoopSpecV1(until=P("done"), max_attempts=3))
    tree = library.expand("root")
    runtime = TableRuntime({"leaf_a": [F]})
    record = execute(tree, library, runtime, {"done": lambda belief, args: False})
    assert record.verdict is F and record.root.reason == "budget_exhausted" and runtime.calls == 3


def test_observe_success_unknown_and_interruption() -> None:
    library = SkillLibraryV1(library_id="o", predicates=(PredicateSpecV1(name="seen"),), skills=(
        SkillDefinitionV1(skill_id="look", kind=NodeKind.OBSERVE, timeout_s=5.0, effects=(P("seen"),), observation=ObservationSpecV1(evidence=("thing",), source="test")),
    ))
    tree = library.expand("look")
    predicates = {"seen": lambda belief, args: belief.get("seen")}

    class Eyes:
        def __init__(self, answer, interrupt=None):
            self.answer, self.interrupt = answer, interrupt

        def run_primitive(self, context):
            raise AssertionError("not a primitive")

        def observe(self, context):
            if self.interrupt is not None:
                self.interrupt.request("blinked")
            return self.answer

    assert execute(tree, library, Eyes({"seen": True}), predicates).verdict is S
    unknown = execute(tree, library, Eyes(None), predicates)
    assert unknown.verdict is U and unknown.root.reason == "nothing_observed"
    assert execute(tree, library, Eyes({"seen": False}), predicates).verdict is F
    interrupt = Interrupt()
    assert execute(tree, library, Eyes({"seen": True}, interrupt), predicates, interrupt=interrupt).verdict is I


def test_the_tree_is_refused_before_running_if_a_predicate_has_no_evaluator() -> None:
    library = clear_bench_library()
    tree = library.expand("clear_bench", ARGUMENTS)
    with pytest.raises(SkillExecutionError, match="object_in_reach"):
        execute(tree, library, ScriptedRuntime(), {k: v for k, v in PREDICATES.items() if k != "object_in_reach"})
    with pytest.raises(SkillExecutionError, match="different library"):
        execute(tree, with_recovery(library), ScriptedRuntime(), PREDICATES)


def test_the_record_round_trips_and_refers_to_the_tree_by_hash() -> None:
    library = clear_bench_library()
    tree = library.expand("clear_bench", ARGUMENTS)
    record = execute(tree, library, ScriptedRuntime(primitives={("transfer", "primary"): (Verdict.SUCCESS, True)}), PREDICATES)
    assert record.tree_sha256 == tree.content_hash()
    again = ExecutionRecordV1.model_validate_json(record.model_dump_json())
    assert again == record and again.content_hash() == record.content_hash()
    assert {n.node_id for n in again.root.walk()} <= {n.node_id for n in tree.root.walk()}
