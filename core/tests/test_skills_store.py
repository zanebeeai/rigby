"""The skill store: promotion as a gate, validity as an exact verdict, invalidation by dimension, restart from observation.

No body, no physics. A certificate is issued against the shared transfer
library under a fabricated context, and the store's rules are exercised on
it: an outcome that fails, or a validation set that shares a development
set, leaves a candidate; retrieval ranks by similarity but chooses only a
valid promoted certificate; each dimension of the context invalidates the
certificates that differ on it and no other; revalidation issues a new
version that supersedes the old only when its own outcome passes; the store
round-trips through disk. The checkpointing runtime stops a scripted tree
after exactly the requested leaf and hands over the world state; a restart
with an empty belief re-observes and completes or stops with a reason.
"""

from __future__ import annotations

from typing import Any

import pytest

from rigby_core.hashing import content_hash
from rigby_core.skills import (
    Belief,
    CertificateStatus,
    CertificateV1,
    CheckpointingRuntime,
    ContextDimension,
    ContextFacetV1,
    CostV1,
    ExecutionContextV1,
    Interrupt,
    LeafContext,
    LeafOutcome,
    RangeV1,
    SkillStoreV1,
    ValidationOutcomeV1,
    ValidationSetV1,
    Verdict,
    compare,
    execute,
    safe_boundaries,
)
from rigby_core.skills.examples import transfer_object_library


DEVELOPMENT = ("g10-transfer-v1:nominal-seeds-0-99",)
ARGS = {"object": "cube", "destination": "platform", "effector": "chain_palm"}


def facet(dimension: ContextDimension, name: str, **detail: Any) -> ContextFacetV1:
    return ContextFacetV1.of(dimension, name, {"name": name, **detail})


def context(*, body="zoo_jaw_arm", controller="ct-8hz", sensors="front_overhead_contact", geometry="cube-15mm", friction="1.0-1.8", schema="episode/1", values=None) -> ExecutionContextV1:
    return ExecutionContextV1(
        facets=(facet(ContextDimension.BODY, body), facet(ContextDimension.CONTROLLER, controller), facet(ContextDimension.SENSORS, sensors),
                facet(ContextDimension.GEOMETRY, geometry), facet(ContextDimension.FRICTION, friction), facet(ContextDimension.EVIDENCE_SCHEMA, schema)),
        ranges=(RangeV1(quantity="object_friction", low=1.0, high=1.8, units="coefficient"), RangeV1(quantity="object_distance_m", low=0.55, high=0.80, units="m")),
        values=values or {"object_friction": 1.4, "object_distance_m": 0.667},
    )


def validation_set(set_id="g11-validation-jaw", threshold=11, independent_of=DEVELOPMENT) -> ValidationSetV1:
    return ValidationSetV1(set_id=set_id, body="zoo_jaw_arm", episodes=tuple(f"zoo_jaw_arm-validation-{i:04d}" for i in range(1000, 1012)),
                           draws_sha256=content_hash({"seeds": list(range(1000, 1012))}), independent_of=independent_of, threshold=threshold)


def outcome(set_id="g11-validation-jaw", successes=12, false_completions=0, threshold=11) -> ValidationOutcomeV1:
    return ValidationOutcomeV1(set_id=set_id, episodes=12, successes=successes, false_completions=false_completions, threshold=threshold,
                               evidence=tuple(content_hash({"episode": i}) for i in range(3)), rows_sha256=content_hash({"rows": successes}))


COST = CostV1(physics_s=300.0, wall_s=900.0, episodes=12, generation_calls=0, dollars=0.0)


@pytest.fixture
def library():
    return transfer_object_library()


@pytest.fixture
def store(library):
    return SkillStoreV1(store_id="test").add_library(library)


def promoted(store, library, ctx=None):
    candidate = CertificateV1.candidate(skill_id="transfer_object", library=library, context=ctx or context(), validation=outcome(), cost=COST)
    store, certificate = store.promote(candidate, validation_set(), development_sets=DEVELOPMENT)
    assert certificate.status is CertificateStatus.PROMOTED
    return store, certificate


# -- promotion is a gate ------------------------------------------------------------------


def test_promotion_needs_a_passing_independent_set(store, library):
    store, certificate = promoted(store, library)
    assert certificate.version == 1 and certificate.definition_sha256 == library.skill("transfer_object").content_hash()
    assert store.history[-1].event == "promoted"


def test_a_failing_outcome_stays_a_candidate(store, library):
    candidate = CertificateV1.candidate(skill_id="transfer_object", library=library, context=context(), validation=outcome(successes=10), cost=COST)
    store, decided = store.promote(candidate, validation_set(), development_sets=DEVELOPMENT)
    assert decided.status is CertificateStatus.CANDIDATE and "validation_failed:10/12<11" in decided.reason
    assert store.history[-1].event == "rejected"
    assert store.retrieve("transfer_object", context()).chosen is None


def test_a_false_completion_blocks_promotion(store, library):
    candidate = CertificateV1.candidate(skill_id="transfer_object", library=library, context=context(), validation=outcome(successes=12, false_completions=1), cost=COST)
    _, decided = store.promote(candidate, validation_set(), development_sets=DEVELOPMENT)
    assert decided.status is CertificateStatus.CANDIDATE and "false_completions=1" in decided.reason


def test_a_set_that_shares_a_development_set_cannot_promote(store, library):
    dependent = validation_set(independent_of=("something-else",))
    candidate = CertificateV1.candidate(skill_id="transfer_object", library=library, context=context(), validation=outcome(), cost=COST)
    _, decided = store.promote(candidate, dependent, development_sets=DEVELOPMENT)
    assert decided.status is CertificateStatus.CANDIDATE and "not_independent" in decided.reason


def test_promotion_needs_the_library_in_the_store(library):
    empty = SkillStoreV1(store_id="empty")
    candidate = CertificateV1.candidate(skill_id="transfer_object", library=library, context=context(), validation=outcome(), cost=COST)
    with pytest.raises(KeyError):
        empty.promote(candidate, validation_set(), development_sets=DEVELOPMENT)


# -- validity is exact; similarity is only a ranking --------------------------------------------


def test_compare_is_exact_on_facets_and_inside_ranges():
    verdict = compare(context(), context())
    assert verdict.valid and verdict.similarity == 1.0
    changed = compare(context(), context(controller="ct-4hz"))
    assert not changed.valid and changed.similarity == pytest.approx(5 / 6)
    assert [d.dimension for d in changed.differences] == [ContextDimension.CONTROLLER]
    out_of_range = compare(context(), context(values={"object_friction": 0.6, "object_distance_m": 0.667}))
    assert not out_of_range.valid and out_of_range.similarity == 1.0
    assert out_of_range.differences[0].kind == "range" and out_of_range.differences[0].quantity == "object_friction"
    missing = compare(context(), ExecutionContextV1(facets=context().facets[:3], values={}))
    assert not missing.valid and {d.kind for d in missing.differences} == {"missing"}


def test_retrieval_never_chooses_by_similarity(store, library):
    store, certificate = promoted(store, library)
    nearly = store.retrieve("transfer_object", context(sensors="front_contact"))
    assert nearly.matches[0].verdict.similarity == pytest.approx(5 / 6)
    assert nearly.chosen is None and "sensors" in nearly.reason
    exact = store.retrieve("transfer_object", context())
    assert exact.chosen == certificate.certificate_id
    other = store.retrieve("transfer_object", context(values={"object_friction": 1.4, "object_distance_m": 0.9}))
    assert other.chosen is None and "object_distance_m" in other.reason
    assert store.retrieve("acquire", context()).reason == "no certificate for this skill"


# -- invalidation by dimension, revalidation by outcome ---------------------------------------------


@pytest.mark.parametrize("dimension, new", [
    (ContextDimension.GEOMETRY, "cube-20mm"), (ContextDimension.CONTROLLER, "ct-4hz"), (ContextDimension.SENSORS, "front_contact"),
    (ContextDimension.FRICTION, "0.5-0.9"), (ContextDimension.EVIDENCE_SCHEMA, "episode/2"), (ContextDimension.BODY, "zoo_long_arm"),
])
def test_each_dimension_invalidates_only_the_certificates_that_differ(store, library, dimension, new):
    store, first = promoted(store, library)
    kwargs = {ContextDimension.GEOMETRY: "geometry", ContextDimension.CONTROLLER: "controller", ContextDimension.SENSORS: "sensors",
              ContextDimension.FRICTION: "friction", ContextDimension.EVIDENCE_SCHEMA: "schema", ContextDimension.BODY: "body"}
    already = context(**{kwargs[dimension]: new})
    store, second = promoted(store, library, already)
    store, affected = store.invalidate(dimension, facet(dimension, new), reason="test")
    assert affected == (first.certificate_id,)
    assert store.certificate(first.certificate_id).status is CertificateStatus.INVALIDATED
    assert store.certificate(first.certificate_id).invalidated_by == (dimension,)
    assert store.certificate(second.certificate_id).status is CertificateStatus.PROMOTED
    assert store.retrieve("transfer_object", context()).chosen is None
    assert store.retrieve("transfer_object", already).chosen == second.certificate_id


def test_revalidation_issues_a_superseding_version_only_on_a_pass(store, library):
    store, first = promoted(store, library)
    store, _ = store.invalidate(ContextDimension.GEOMETRY, facet(ContextDimension.GEOMETRY, "cube-20mm"))
    new_context = context(geometry="cube-20mm")
    failed_set = validation_set(set_id="g11-revalidation-geometry")
    store, failed = store.revalidate(first.certificate_id, library=library, context=new_context, validation=outcome(set_id="g11-revalidation-geometry", successes=9), validation_set=failed_set,
                                     cost=COST, development_sets=DEVELOPMENT)
    assert failed.status is CertificateStatus.CANDIDATE and failed.version == 2 and failed.supersedes == first.certificate_id
    assert store.certificate(first.certificate_id).status is CertificateStatus.INVALIDATED
    assert store.history[-1].event == "revalidation_failed"
    assert store.retrieve("transfer_object", new_context).chosen is None
    store, passed = store.revalidate(first.certificate_id, library=library, context=new_context, validation=outcome(set_id="g11-revalidation-geometry", successes=12), validation_set=failed_set,
                                     cost=COST, development_sets=DEVELOPMENT)
    assert passed.status is CertificateStatus.PROMOTED and passed.version == 2
    assert store.certificate(first.certificate_id).status is CertificateStatus.SUPERSEDED
    assert store.retrieve("transfer_object", new_context).chosen == passed.certificate_id
    assert store.history[-1].event == "revalidated"


def test_revalidation_needs_an_invalidated_certificate(store, library):
    store, first = promoted(store, library)
    with pytest.raises(ValueError):
        store.revalidate(first.certificate_id, library=library, context=context(), validation=outcome(), validation_set=validation_set(), cost=COST, development_sets=DEVELOPMENT)


# -- persistence -----------------------------------------------------------------------------------------


def test_store_round_trips_through_disk(store, library, tmp_path):
    store, first = promoted(store, library)
    store, _ = store.invalidate(ContextDimension.FRICTION, facet(ContextDimension.FRICTION, "0.5-0.9"))
    digests = store.save(tmp_path / "store")
    loaded = SkillStoreV1.load(tmp_path / "store")
    assert loaded.content_hash() == store.content_hash()
    assert loaded.certificate(first.certificate_id).status is CertificateStatus.INVALIDATED
    assert set(digests) == {f"certificates/{first.certificate_id}.json", f"libraries/{library.content_hash()}.json"}
    assert loaded.library(library.content_hash()).content_hash() == library.content_hash()


# -- restart -----------------------------------------------------------------------------------------------


class World:
    """A scripted body: the cube is where the last leaf left it."""

    def __init__(self) -> None:
        self.held = False
        self.placed = False
        self.seen = True
        self.time_s = 0.0

    def state(self) -> dict[str, Any]:
        return {"held": self.held, "placed": self.placed, "seen": self.seen, "time_s": self.time_s}


class ScriptedRuntime:
    def __init__(self, world: World, clock) -> None:
        self.world = world
        self.clock = clock
        self.leaves: list[str] = []

    def run_primitive(self, context: LeafContext) -> LeafOutcome:
        leaf = context.node.skill_id
        self.leaves.append(leaf)
        self.clock.advance(1.0)
        self.world.time_s = self.clock.now()
        if leaf == "acquire":
            self.world.held = True
        elif leaf == "release":
            self.world.held = False
            self.world.placed = True
        return LeafOutcome(Verdict.SUCCESS, facts={"held:cube": False} if leaf == "release" else {})

    def observe(self, context: LeafContext) -> dict[str, Any] | None:
        leaf = context.node.skill_id
        self.leaves.append(leaf)
        self.clock.advance(0.5)
        self.world.time_s = self.clock.now()
        if leaf == "observe_object":
            return {"known:cube": True, "pose:cube": [0.0, 0.6, 0.27], "reach:cube": True} if self.world.seen else None
        if leaf == "verify_hold":
            return {"held:cube": self.world.held}
        if leaf == "verify_placement":
            return {"placed:cube:platform": self.world.placed, "held:cube": False}
        return None

    def reconstruct(self) -> dict[str, Any]:
        """What a restarted executor may believe: only what the world shows now."""

        facts: dict[str, Any] = {}
        if self.world.held:
            facts["held:cube"] = True
        if self.world.placed:
            facts["placed:cube:platform"] = True
        if self.world.seen:
            facts.update({"known:cube": True, "pose:cube": [0.0, 0.6, 0.27], "reach:cube": True})
        return facts


def predicates():
    return {
        "object_known": lambda belief, args: bool(belief.get(f"known:{args[0]}", False)),
        "object_held": lambda belief, args: bool(belief.get(f"held:{args[0]}", False)),
        "object_placed": lambda belief, args: bool(belief.get(f"placed:{args[0]}:{args[1]}", False)),
        "object_in_reach": lambda belief, args: bool(belief.get(f"reach:{args[0]}", False)),
    }


def test_safe_boundaries_are_the_leaves_in_order(library):
    tree = library.expand("transfer_object", ARGS)
    boundaries = safe_boundaries(tree)
    assert [tree.node(b).skill_id for b in boundaries] == ["observe_object", "acquire", "verify_hold", "transport", "release", "verify_placement"]


@pytest.mark.parametrize("stop_after", [1, 2, 3, 4, 5, 6])
def test_checkpoint_after_each_leaf_then_restart_completes(library, stop_after):
    from rigby_core.skills import MonotonicClock

    tree = library.expand("transfer_object", ARGS)
    world = World()
    clock = MonotonicClock()
    interrupt = Interrupt()
    inner = ScriptedRuntime(world, clock)
    runtime = CheckpointingRuntime(inner, interrupt, stop_after, world_state=world.state, tree_sha256=tree.content_hash(), library_sha256=library.content_hash())
    record = execute(tree, library, runtime, predicates(), clock=clock, interrupt=interrupt)
    checkpoint = runtime.checkpoint
    assert checkpoint is not None and checkpoint.boundary_index == stop_after and len(inner.leaves) == stop_after
    # The executor sees the request the moment the leaf returns, so even the
    # last leaf's completion leaves the run marked interrupted: nothing after
    # a checkpoint is trusted, not even a success the checkpoint followed.
    assert record.verdict is Verdict.INTERRUPTED and record.interrupt_reason == checkpoint.reason
    assert checkpoint.world_state["time_s"] == pytest.approx(clock.now())
    # restart: a fresh executor, an empty belief, facts from the world alone
    restarted = ScriptedRuntime(world, clock)
    facts = restarted.reconstruct()
    record2 = execute(tree, library, restarted, predicates(), clock=clock, interrupt=Interrupt(), belief=Belief(dict(facts)))
    assert record2.verdict is Verdict.SUCCESS
    assert world.placed
    if stop_after >= 5:
        assert restarted.leaves == [], "a placed cube completes the tree before any leaf runs"
    elif stop_after >= 2:
        assert "acquire" not in restarted.leaves, "a held cube is not acquired again"


def test_restart_stops_explicitly_when_the_world_offers_nothing(library):
    from rigby_core.skills import MonotonicClock

    tree = library.expand("transfer_object", ARGS)
    world = World()
    world.seen = False
    clock = MonotonicClock()
    restarted = ScriptedRuntime(world, clock)
    record = execute(tree, library, restarted, predicates(), clock=clock, interrupt=Interrupt(), belief=Belief(dict(restarted.reconstruct())))
    # Nothing to see means nothing to act on: the observation returns nothing,
    # the loop propagates the undecided child, and the tree stops with the
    # reason on record rather than guessing where the object went.
    assert record.verdict in (Verdict.FAILURE, Verdict.UNKNOWN) and record.root.reason
    assert not world.placed
