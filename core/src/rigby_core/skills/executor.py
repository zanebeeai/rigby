"""Execute an expanded task tree against a body-supplied runtime.

The executor owns the semantics of the composite kinds and of the contract's
checks; the runtime owns the leaves. A leaf is asked to run a controller or
to observe, is given the belief, a deadline on the runtime's own clock and
a way to learn that it should stop, and answers with a verdict and the facts
it established. Predicates are functions of the belief that answer true,
false or -- when the belief cannot decide -- nothing at all. The executor
never invents an answer: an undecidable initiation, effect or loop
predicate leaves the node undecided unless its termination rule says that
counts against it, and a leaf that could not observe says so.

Every node's run is recorded: what was checked and what it answered, the
verdict and the reason, on the clock, with the children under it, so a
failure anywhere in a deep tree is traceable to the predicate or the leaf
that decided it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pydantic import Field

from ..contracts import Contract
from .contract import (
    NodeKind,
    OnUnknown,
    PredicateRefV1,
    SkillLibraryV1,
    TaskNodeV1,
    TaskTreeV1,
    Verdict,
    substitute,
)


class SkillExecutionError(RuntimeError):
    """The tree cannot be run at all: a predicate has no evaluator."""


class Clock(Protocol):
    def now(self) -> float: ...


@dataclass
class MonotonicClock:
    """A clock the executor advances only through the runtime's leaves."""

    time_s: float = 0.0

    def now(self) -> float:
        return self.time_s

    def advance(self, seconds: float) -> None:
        self.time_s += seconds


@dataclass
class Interrupt:
    """Raised from outside; checked before every node and offered to leaves."""

    requested: bool = False
    reason: str = ""

    def request(self, reason: str = "interrupted") -> None:
        self.requested = True
        self.reason = reason


@dataclass
class Belief:
    """What the executor currently holds true, by fact name."""

    facts: dict[str, Any] = field(default_factory=dict)

    def knows(self, name: str) -> bool:
        return name in self.facts

    def get(self, name: str, default: Any = None) -> Any:
        return self.facts.get(name, default)

    def update(self, facts: dict[str, Any]) -> None:
        self.facts.update(facts)


@dataclass(frozen=True)
class LeafContext:
    belief: Belief
    clock: Clock
    deadline_s: float
    should_stop: Callable[[], bool]
    node: TaskNodeV1


@dataclass(frozen=True)
class LeafOutcome:
    verdict: Verdict
    reason: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


class LeafRuntime(Protocol):
    def run_primitive(self, context: LeafContext) -> LeafOutcome: ...

    def observe(self, context: LeafContext) -> dict[str, Any] | None:
        """The facts observed, or ``None`` when nothing could be observed."""
        ...


PredicateEvaluator = Callable[[Belief, tuple[str, ...]], bool | None]


class CheckRecordV1(Contract):
    predicate: str
    arguments: tuple[str, ...] = ()
    negate: bool = False
    answer: bool | None = None


class NodeRecordV1(Contract):
    node_id: str
    skill_id: str
    kind: NodeKind
    verdict: Verdict
    reason: str = ""
    started_s: float
    ended_s: float
    attempts: int = Field(ge=1)
    initiation: tuple[CheckRecordV1, ...] = ()
    invariants: tuple[CheckRecordV1, ...] = ()
    effects: tuple[CheckRecordV1, ...] = ()
    loop_checks: tuple[CheckRecordV1, ...] = ()
    resources: tuple[str, ...] = ()
    evidence: dict[str, Any] = {}
    children: tuple[NodeRecordV1, ...] = ()
    recoveries: tuple[NodeRecordV1, ...] = ()

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()
        for recovery in self.recoveries:
            yield from recovery.walk()


class ExecutionRecordV1(Contract):
    schema_version: str = "1.0"
    tree_sha256: str = Field(min_length=64, max_length=64)
    library_id: str
    verdict: Verdict
    interrupted: bool = False
    interrupt_reason: str = ""
    started_s: float
    ended_s: float
    root: NodeRecordV1
    belief: dict[str, Any] = {}

    def record(self, node_id: str) -> NodeRecordV1:
        for candidate in self.root.walk():
            if candidate.node_id == node_id:
                return candidate
        raise KeyError(node_id)


def execute(
    tree: TaskTreeV1,
    library: SkillLibraryV1,
    runtime: LeafRuntime,
    predicates: dict[str, PredicateEvaluator],
    *,
    clock: Clock | None = None,
    interrupt: Interrupt | None = None,
    belief: Belief | None = None,
) -> ExecutionRecordV1:
    """Run ``tree`` and return its record. Raises before running anything if
    a predicate the tree uses has no evaluator."""

    if library.content_hash() != tree.library_sha256:
        raise SkillExecutionError("the tree was expanded from a different library")
    needed = {ref.name for node in tree.root.walk() for ref in library.skill(node.skill_id).predicate_refs}
    missing = sorted(needed - set(predicates))
    if missing:
        raise SkillExecutionError(f"no evaluator for predicate(s) {missing}")
    runner = _Runner(library, runtime, predicates, clock or MonotonicClock(), interrupt or Interrupt(), belief or Belief())
    started = runner.clock.now()
    root = runner.run(tree.root, parent_deadline=float("inf"))
    return ExecutionRecordV1(tree_sha256=tree.content_hash(), library_id=tree.library_id, verdict=root.verdict,
                             interrupted=runner.interrupt.requested, interrupt_reason=runner.interrupt.reason,
                             started_s=started, ended_s=runner.clock.now(), root=root, belief=dict(runner.belief.facts))


@dataclass
class _Runner:
    library: SkillLibraryV1
    runtime: LeafRuntime
    predicates: dict[str, PredicateEvaluator]
    clock: Clock
    interrupt: Interrupt
    belief: Belief
    holders: dict[str, str] = field(default_factory=dict)
    """Which running node owns each exclusive resource."""
    stack: list[str] = field(default_factory=list)
    """The node ids currently executing, root first."""

    # -- predicates -------------------------------------------------------

    def check(self, refs: tuple[PredicateRefV1, ...], node: TaskNodeV1) -> tuple[CheckRecordV1, ...]:
        records = []
        for ref in refs:
            arguments = tuple(substitute(a, node.arguments) if a.startswith("$") else a for a in ref.arguments)
            answer = self.predicates[ref.name](self.belief, arguments)
            if answer is not None and ref.negate:
                answer = not answer
            records.append(CheckRecordV1(predicate=ref.name, arguments=arguments, negate=ref.negate, answer=answer))
        return tuple(records)

    @staticmethod
    def settle(records: tuple[CheckRecordV1, ...], on_unknown: OnUnknown) -> Verdict:
        answers = [r.answer for r in records]
        if any(a is False for a in answers):
            return Verdict.FAILURE
        if any(a is None for a in answers):
            return Verdict.FAILURE if on_unknown is OnUnknown.FAIL else Verdict.UNKNOWN
        return Verdict.SUCCESS

    # -- nodes ------------------------------------------------------------

    def run(self, node: TaskNodeV1, *, parent_deadline: float) -> NodeRecordV1:
        definition = self.library.skill(node.skill_id)
        started = self.clock.now()
        deadline = min(parent_deadline, started + definition.timeout_s)
        if self.interrupt.requested:
            return NodeRecordV1(node_id=node.node_id, skill_id=node.skill_id, kind=node.kind, verdict=Verdict.INTERRUPTED,
                                reason=self.interrupt.reason, started_s=started, ended_s=started, attempts=1, resources=node.resources)
        initiation = self.check(definition.initiation, node)
        verdict = self.settle(initiation, definition.termination.on_unknown)
        if verdict is not Verdict.SUCCESS:
            return NodeRecordV1(node_id=node.node_id, skill_id=node.skill_id, kind=node.kind, verdict=verdict,
                                reason="initiation_unmet" if verdict is Verdict.FAILURE else "initiation_undecidable",
                                started_s=started, ended_s=self.clock.now(), attempts=1, initiation=initiation, resources=node.resources)
        # A node may own what an ancestor owns -- that is how ownership is
        # delegated -- but not what some other running node holds.
        conflict = [r for r in node.resources if r in self.holders and self.holders[r] not in self.stack]
        if conflict:
            return NodeRecordV1(node_id=node.node_id, skill_id=node.skill_id, kind=node.kind, verdict=Verdict.FAILURE,
                                reason=f"resource_conflict:{conflict[0]}", started_s=started, ended_s=self.clock.now(), attempts=1,
                                initiation=initiation, resources=node.resources)
        newly_owned = list(dict.fromkeys(r for r in node.resources if r not in self.holders))
        for resource in newly_owned:
            self.holders[resource] = node.node_id
        self.stack.append(node.node_id)
        try:
            attempts = 0
            recoveries: list[NodeRecordV1] = []
            children: tuple[NodeRecordV1, ...] = ()
            loop_checks: tuple[CheckRecordV1, ...] = ()
            evidence: dict[str, Any] = {}
            while True:
                attempts += 1
                verdict, reason, children, loop_checks, evidence = self.run_kind(node, definition, deadline)
                if verdict is Verdict.SUCCESS and self.clock.now() > deadline:
                    verdict, reason = Verdict.FAILURE, "timeout"
                if verdict is Verdict.FAILURE and attempts < definition.recovery.max_attempts and not self.interrupt.requested:
                    if node.recovery is not None:
                        recovery = self.run(node.recovery, parent_deadline=deadline)
                        recoveries.append(recovery)
                        if recovery.verdict is not Verdict.SUCCESS:
                            reason = f"recovery_{recovery.verdict.value}:{reason}"
                            break
                    continue
                break
            invariants = self.check(definition.invariants, node) if verdict is Verdict.SUCCESS else ()
            if verdict is Verdict.SUCCESS and any(r.answer is False for r in invariants):
                verdict, reason = Verdict.FAILURE, "invariant_violated"
            elif verdict is Verdict.SUCCESS and any(r.answer is None for r in invariants):
                verdict = Verdict.FAILURE if definition.termination.on_unknown is OnUnknown.FAIL else Verdict.UNKNOWN
                reason = "invariant_undecidable"
            effects = self.check(definition.effects, node) if verdict is Verdict.SUCCESS and definition.termination.require_effects else ()
            if verdict is Verdict.SUCCESS and effects:
                settled = self.settle(effects, definition.termination.on_unknown)
                if settled is not Verdict.SUCCESS:
                    verdict = settled
                    reason = "effects_unmet" if settled is Verdict.FAILURE else "effects_undecidable"
            return NodeRecordV1(node_id=node.node_id, skill_id=node.skill_id, kind=node.kind, verdict=verdict, reason=reason,
                                started_s=started, ended_s=self.clock.now(), attempts=attempts, initiation=initiation, invariants=invariants,
                                effects=effects, loop_checks=loop_checks, resources=node.resources, evidence=evidence,
                                children=children, recoveries=tuple(recoveries))
        finally:
            self.stack.pop()
            for resource in newly_owned:
                del self.holders[resource]

    def run_kind(self, node: TaskNodeV1, definition, deadline: float):
        kind = node.kind
        if kind is NodeKind.PRIMITIVE:
            return self.run_primitive(node, deadline)
        if kind is NodeKind.OBSERVE:
            return self.run_observe(node, deadline)
        if kind is NodeKind.SEQUENCE:
            return self.run_sequence(node, definition, deadline)
        if kind is NodeKind.SELECTOR:
            return self.run_selector(node, deadline)
        if kind is NodeKind.PARALLEL:
            return self.run_parallel(node, deadline)
        if kind is NodeKind.REPEAT_UNTIL:
            return self.run_loop(node, definition, deadline)
        raise SkillExecutionError(f"unknown node kind {kind}")

    def context(self, node: TaskNodeV1, deadline: float) -> LeafContext:
        return LeafContext(belief=self.belief, clock=self.clock, deadline_s=deadline, should_stop=lambda: self.interrupt.requested, node=node)

    def run_primitive(self, node: TaskNodeV1, deadline: float):
        outcome = self.runtime.run_primitive(self.context(node, deadline))
        self.belief.update(outcome.facts)
        verdict, reason = outcome.verdict, outcome.reason
        if verdict is Verdict.SUCCESS and self.clock.now() > deadline:
            verdict, reason = Verdict.FAILURE, "timeout"
        if self.interrupt.requested and verdict is not Verdict.INTERRUPTED:
            verdict, reason = Verdict.INTERRUPTED, self.interrupt.reason
        return verdict, reason, (), (), dict(outcome.evidence)

    def run_observe(self, node: TaskNodeV1, deadline: float):
        facts = self.runtime.observe(self.context(node, deadline))
        if self.interrupt.requested:
            return Verdict.INTERRUPTED, self.interrupt.reason, (), (), {}
        if facts is None:
            return Verdict.UNKNOWN, "nothing_observed", (), (), {}
        self.belief.update(facts)
        if self.clock.now() > deadline:
            return Verdict.FAILURE, "timeout", (), (), {"observed": sorted(facts)}
        return Verdict.SUCCESS, "", (), (), {"observed": sorted(facts)}

    def run_sequence(self, node: TaskNodeV1, definition, deadline: float):
        records = []
        for child in node.children:
            if self.clock.now() > deadline:
                return Verdict.FAILURE, "timeout", tuple(records), (), {}
            record = self.run(child, parent_deadline=deadline)
            records.append(record)
            if record.verdict is Verdict.SUCCESS:
                between = self.check(definition.invariants, node)
                if any(r.answer is False for r in between):
                    return Verdict.FAILURE, "invariant_violated", tuple(records), (), {}
                continue
            if record.verdict is Verdict.INTERRUPTED:
                return Verdict.INTERRUPTED, record.reason, tuple(records), (), {}
            if record.verdict is Verdict.UNKNOWN and definition.termination.on_unknown is OnUnknown.PROPAGATE:
                return Verdict.UNKNOWN, f"child_unknown:{child.node_id}", tuple(records), (), {}
            return Verdict.FAILURE, f"child_failed:{child.node_id}", tuple(records), (), {}
        return Verdict.SUCCESS, "", tuple(records), (), {}

    def run_selector(self, node: TaskNodeV1, deadline: float):
        records = []
        undecided = False
        for child in node.children:
            if self.clock.now() > deadline:
                return Verdict.FAILURE, "timeout", tuple(records), (), {}
            record = self.run(child, parent_deadline=deadline)
            records.append(record)
            if record.verdict is Verdict.SUCCESS:
                return Verdict.SUCCESS, f"alternative:{child.node_id}", tuple(records), (), {}
            if record.verdict is Verdict.INTERRUPTED:
                return Verdict.INTERRUPTED, record.reason, tuple(records), (), {}
            undecided |= record.verdict is Verdict.UNKNOWN
        if undecided:
            return Verdict.UNKNOWN, "no_alternative_decided", tuple(records), (), {}
        return Verdict.FAILURE, "no_alternative_succeeded", tuple(records), (), {}

    def run_parallel(self, node: TaskNodeV1, deadline: float):
        # Children own disjoint resources by construction; the executor runs
        # them one after another and reports the conjunction, which is the
        # verdict a concurrent run would have to reach on the same facts.
        records = []
        for child in node.children:
            record = self.run(child, parent_deadline=deadline)
            records.append(record)
            if record.verdict is Verdict.INTERRUPTED:
                return Verdict.INTERRUPTED, record.reason, tuple(records), (), {}
        verdicts = {r.verdict for r in records}
        if verdicts == {Verdict.SUCCESS}:
            return Verdict.SUCCESS, "", tuple(records), (), {}
        if Verdict.FAILURE in verdicts:
            failed = next(r for r in records if r.verdict is Verdict.FAILURE)
            return Verdict.FAILURE, f"child_failed:{failed.node_id}", tuple(records), (), {}
        return Verdict.UNKNOWN, "child_unknown", tuple(records), (), {}

    def run_loop(self, node: TaskNodeV1, definition, deadline: float):
        loop = definition.loop
        child = node.children[0]
        records = []
        checks = []
        attempts = 0
        while True:
            until = self.check((loop.until,), node)
            checks.extend(until)
            answer = until[0].answer
            if answer is True:
                return Verdict.SUCCESS, f"until_after_{attempts}", tuple(records), tuple(checks), {"attempts": attempts}
            if answer is None:
                if definition.termination.on_unknown is OnUnknown.FAIL:
                    return Verdict.FAILURE, "until_undecidable", tuple(records), tuple(checks), {"attempts": attempts}
                return Verdict.UNKNOWN, "until_undecidable", tuple(records), tuple(checks), {"attempts": attempts}
            if attempts >= loop.max_attempts:
                return Verdict.FAILURE, "budget_exhausted", tuple(records), tuple(checks), {"attempts": attempts}
            if self.clock.now() > deadline:
                return Verdict.FAILURE, "timeout", tuple(records), tuple(checks), {"attempts": attempts}
            if self.interrupt.requested:
                return Verdict.INTERRUPTED, self.interrupt.reason, tuple(records), tuple(checks), {"attempts": attempts}
            attempts += 1
            record = self.run(child, parent_deadline=deadline)
            records.append(record)
            if record.verdict is Verdict.INTERRUPTED:
                return Verdict.INTERRUPTED, record.reason, tuple(records), tuple(checks), {"attempts": attempts}
            if record.verdict is Verdict.UNKNOWN and definition.termination.on_unknown is OnUnknown.PROPAGATE:
                # An attempt whose outcome could not be decided leaves the
                # loop undecided too: repeating it would be guessing.
                return Verdict.UNKNOWN, f"child_undecided:{child.node_id}", tuple(records), tuple(checks), {"attempts": attempts}
            if loop.progress is not None:
                progress = self.check((loop.progress,), node)
                checks.extend(progress)
                if progress[0].answer is False:
                    return Verdict.FAILURE, "no_progress", tuple(records), tuple(checks), {"attempts": attempts}
                if progress[0].answer is None and definition.termination.on_unknown is OnUnknown.FAIL:
                    return Verdict.FAILURE, "progress_undecidable", tuple(records), tuple(checks), {"attempts": attempts}
