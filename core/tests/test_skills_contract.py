"""The neutral skill contract refuses what would make a tree meaningless.

One contract for leaves and composites; a library that is a DAG with
bounded loops; expansion into a task tree at least five deep; persisted
definitions that round-trip without changing meaning; and no limb, digit or
joint anywhere in the schema.
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from rigby_core.skills import (
    ArgumentSpecV1,
    ArgumentType,
    ChildRefV1,
    ExecutionRecordV1,
    LoopSpecV1,
    NodeKind,
    ObservationSpecV1,
    PredicateRefV1,
    PredicateSpecV1,
    RecoveryV1,
    ResourceClaimV1,
    ResourceMode,
    SkillDefinitionV1,
    SkillLibraryV1,
    TaskTreeV1,
    parse_argument,
)
from rigby_core.skills.examples import ARGUMENTS, A, P, clear_bench_library, with_recovery


def leaf(skill_id: str, **overrides) -> SkillDefinitionV1:
    fields = dict(skill_id=skill_id, kind=NodeKind.PRIMITIVE, timeout_s=1.0, controller="test.controller")
    fields.update(overrides)
    return SkillDefinitionV1(**fields)


def library(*skills: SkillDefinitionV1, predicates=()) -> SkillLibraryV1:
    return SkillLibraryV1(library_id="t", predicates=tuple(predicates), skills=tuple(skills))


# --------------------------------------------------------------------------
# one contract, every kind
# --------------------------------------------------------------------------


def test_every_node_kind_is_the_same_contract() -> None:
    lib = clear_bench_library()
    kinds = {definition.kind for definition in lib.skills}
    assert kinds == {NodeKind.SEQUENCE, NodeKind.SELECTOR, NodeKind.REPEAT_UNTIL, NodeKind.OBSERVE, NodeKind.PRIMITIVE}
    for definition in lib.skills:
        assert isinstance(definition, SkillDefinitionV1)
        for name in ("arguments", "requirements", "initiation", "invariants", "effects", "termination", "timeout_s", "recovery", "resources", "validity"):
            assert hasattr(definition, name), name


def test_the_shape_of_each_kind_is_enforced() -> None:
    with pytest.raises(ValidationError, match="names its controller"):
        SkillDefinitionV1(skill_id="p", kind=NodeKind.PRIMITIVE, timeout_s=1.0)
    with pytest.raises(ValidationError, match="says what it establishes"):
        SkillDefinitionV1(skill_id="o", kind=NodeKind.OBSERVE, timeout_s=1.0)
    with pytest.raises(ValidationError, match="exactly one child"):
        SkillDefinitionV1(skill_id="r", kind=NodeKind.REPEAT_UNTIL, timeout_s=1.0, loop=LoopSpecV1(until=P("x"), max_attempts=1),
                          children=(ChildRefV1(skill="a"), ChildRefV1(skill="b")))
    with pytest.raises(ValidationError, match="at least 2"):
        SkillDefinitionV1(skill_id="q", kind=NodeKind.PARALLEL, timeout_s=1.0, children=(ChildRefV1(skill="a"),))
    with pytest.raises(ValidationError, match="distinct ranks"):
        SkillDefinitionV1(skill_id="s", kind=NodeKind.SELECTOR, timeout_s=1.0, children=(ChildRefV1(skill="a", rank=0), ChildRefV1(skill="b", rank=0)))
    with pytest.raises(ValidationError, match="finite timeout"):
        leaf("p", timeout_s=float("inf"))
    with pytest.raises(ValidationError):
        leaf("p", timeout_s=0.0)


def test_a_predicate_may_only_bind_declared_arguments() -> None:
    with pytest.raises(ValidationError, match=r"undeclared argument \$thing"):
        leaf("p", effects=(P("held", "$thing"),))
    with pytest.raises(ValidationError, match=r"undeclared argument \$thing"):
        leaf("p", resources=(ResourceClaimV1(resource="object:$thing"),))


# --------------------------------------------------------------------------
# what the library refuses
# --------------------------------------------------------------------------


def test_a_cycle_among_definitions_is_refused() -> None:
    with pytest.raises(ValidationError, match="cycle: a -> b -> a"):
        library(
            SkillDefinitionV1(skill_id="a", kind=NodeKind.SEQUENCE, timeout_s=1.0, children=(ChildRefV1(skill="b"),)),
            SkillDefinitionV1(skill_id="b", kind=NodeKind.SEQUENCE, timeout_s=1.0, children=(ChildRefV1(skill="a"),)),
        )


def test_a_cycle_through_a_recovery_is_refused() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        library(
            SkillDefinitionV1(skill_id="a", kind=NodeKind.SEQUENCE, timeout_s=1.0, children=(ChildRefV1(skill="b"),)),
            leaf("b", recovery=RecoveryV1(skill="a", max_attempts=2)),
        )
    with pytest.raises(ValidationError, match="its own recovery"):
        leaf("b", recovery=RecoveryV1(skill="b", max_attempts=2))


def test_unbounded_retries_are_refused() -> None:
    with pytest.raises(ValidationError):
        RecoveryV1(skill=None, max_attempts=0)
    with pytest.raises(ValidationError):
        RecoveryV1(skill=None, max_attempts=17)
    with pytest.raises(ValidationError):
        LoopSpecV1(until=P("x"), max_attempts=0)
    with pytest.raises(ValidationError):
        LoopSpecV1(until=P("x"), max_attempts=65)
    payload = json.loads(clear_bench_library().model_dump_json())
    loop = next(s for s in payload["skills"] if s["skill_id"] == "place_until_done")
    del loop["loop"]["max_attempts"]
    with pytest.raises(ValidationError):
        SkillLibraryV1.model_validate(payload)
    loop["loop"]["max_attempts"] = None
    with pytest.raises(ValidationError):
        SkillLibraryV1.model_validate(payload)


def test_an_unknown_predicate_is_refused() -> None:
    with pytest.raises(ValidationError, match="unknown predicate 'held'"):
        library(leaf("p", effects=(P("held"),)))
    with pytest.raises(ValidationError, match="takes 1 argument"):
        library(leaf("p", effects=(P("held"),)), predicates=(PredicateSpecV1(name="held", parameters=("object",)),))


def test_an_unknown_child_or_unbound_argument_is_refused() -> None:
    with pytest.raises(ValidationError, match="unknown child skill 'ghost'"):
        library(SkillDefinitionV1(skill_id="a", kind=NodeKind.SEQUENCE, timeout_s=1.0, children=(ChildRefV1(skill="ghost"),)))
    with pytest.raises(ValidationError, match="argument 'object' is unbound"):
        library(SkillDefinitionV1(skill_id="a", kind=NodeKind.SEQUENCE, timeout_s=1.0, children=(ChildRefV1(skill="b"),)),
                leaf("b", arguments=(A("object"),)))
    with pytest.raises(ValidationError, match="is not an integer"):
        library(SkillDefinitionV1(skill_id="a", kind=NodeKind.SEQUENCE, timeout_s=1.0, children=(ChildRefV1(skill="b", bindings={"count": "many"}),)),
                leaf("b", arguments=(ArgumentSpecV1(name="count", type=ArgumentType.INTEGER),)))


def test_incompatible_resource_ownership_is_refused() -> None:
    with pytest.raises(ValidationError, match="owns 'object:cube', which its parent does not"):
        library(SkillDefinitionV1(skill_id="a", kind=NodeKind.SEQUENCE, timeout_s=1.0, children=(ChildRefV1(skill="b"),)),
                leaf("b", resources=(ResourceClaimV1(resource="object:cube"),)))
    with pytest.raises(ValidationError, match="exclusively, which its parent only shares"):
        library(SkillDefinitionV1(skill_id="a", kind=NodeKind.SEQUENCE, timeout_s=1.0, resources=(ResourceClaimV1(resource="camera", mode=ResourceMode.SHARED),),
                                  children=(ChildRefV1(skill="b"),)),
                leaf("b", resources=(ResourceClaimV1(resource="camera"),)))
    with pytest.raises(ValidationError, match="parallel children b and c both own 'effector:left' exclusively"):
        library(SkillDefinitionV1(skill_id="a", kind=NodeKind.PARALLEL, timeout_s=1.0, resources=(ResourceClaimV1(resource="effector:left"),),
                                  children=(ChildRefV1(skill="b"), ChildRefV1(skill="c"))),
                leaf("b", resources=(ResourceClaimV1(resource="effector:left"),)),
                leaf("c", resources=(ResourceClaimV1(resource="effector:left"),)))


def test_parallel_children_with_disjoint_ownership_are_accepted() -> None:
    lib = library(
        SkillDefinitionV1(skill_id="a", kind=NodeKind.PARALLEL, timeout_s=1.0, arguments=(A("left"), A("right")),
                          resources=(ResourceClaimV1(resource="effector:$left"), ResourceClaimV1(resource="effector:$right")),
                          children=(ChildRefV1(skill="b", bindings={"effector": "$left"}), ChildRefV1(skill="b", bindings={"effector": "$right"}))),
        leaf("b", arguments=(A("effector"),), resources=(ResourceClaimV1(resource="effector:$effector"),)),
    )
    tree = lib.expand("a", {"left": "one", "right": "two"})
    assert [c.resources for c in tree.root.children] == [("effector:one",), ("effector:two",)]
    with pytest.raises(ValidationError, match="both own 'effector:\\$same' exclusively"):
        library(
            SkillDefinitionV1(skill_id="a", kind=NodeKind.PARALLEL, timeout_s=1.0, arguments=(A("same"),), resources=(ResourceClaimV1(resource="effector:$same"),),
                              children=(ChildRefV1(skill="b", bindings={"effector": "$same"}), ChildRefV1(skill="b", bindings={"effector": "$same"}))),
            leaf("b", arguments=(A("effector"),), resources=(ResourceClaimV1(resource="effector:$effector"),)),
        )


# --------------------------------------------------------------------------
# expansion
# --------------------------------------------------------------------------


def test_the_example_expands_at_least_five_deep_with_loops_kept_as_loops() -> None:
    lib = clear_bench_library()
    tree = lib.expand("clear_bench", ARGUMENTS)
    assert tree.max_depth == 5 and lib.definition_depth("clear_bench") == 5
    kinds = [node.kind for node in tree.root.walk()]
    assert set(kinds) == {NodeKind.SEQUENCE, NodeKind.OBSERVE, NodeKind.REPEAT_UNTIL, NodeKind.SELECTOR, NodeKind.PRIMITIVE}
    loop = tree.node("0.1")
    assert loop.kind is NodeKind.REPEAT_UNTIL and len(loop.children) == 1, "a loop is one child with a budget, never unrolled"
    leaves = [n for n in tree.root.walk() if n.kind is NodeKind.PRIMITIVE]
    assert [n.arguments["effector"] for n in leaves] == ["primary", "secondary"], "selector alternatives in rank order"
    assert tree.node("0.1.0.0.0").resources == ("effector:primary", "object:cube")
    assert tree.library_sha256 == lib.content_hash()


def test_expansion_refuses_a_missing_or_ill_typed_root_argument() -> None:
    lib = clear_bench_library()
    with pytest.raises(ValueError, match="argument 'alternate' is unbound"):
        lib.expand("clear_bench", {k: v for k, v in ARGUMENTS.items() if k != "alternate"})
    with pytest.raises(ValueError, match="no argument 'colour'"):
        lib.expand("clear_bench", {**ARGUMENTS, "colour": "red"})


def test_a_recovery_is_expanded_under_the_node_it_recovers() -> None:
    tree = with_recovery(clear_bench_library()).expand("clear_bench", ARGUMENTS)
    leaf_node = tree.node("0.1.0.0.0")
    assert leaf_node.recovery is not None and leaf_node.recovery.skill_id == "observe_inventory"
    assert leaf_node.recovery.node_id == "0.1.0.0.0.r" and leaf_node.recovery.arguments == {"object": "cube"}


# --------------------------------------------------------------------------
# persistence and neutrality
# --------------------------------------------------------------------------


def test_persisted_definitions_round_trip_without_changing_meaning(tmp_path) -> None:
    lib = clear_bench_library()
    path = tmp_path / "library.json"
    path.write_text(lib.model_dump_json(indent=2), encoding="utf-8")
    again = SkillLibraryV1.model_validate_json(path.read_text(encoding="utf-8"))
    assert again == lib
    assert again.content_hash() == lib.content_hash()
    tree, tree_again = lib.expand("clear_bench", ARGUMENTS), again.expand("clear_bench", ARGUMENTS)
    assert tree == tree_again and tree.content_hash() == tree_again.content_hash()
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(tree.model_dump_json(), encoding="utf-8")
    assert TaskTreeV1.model_validate_json(tree_path.read_text(encoding="utf-8")).content_hash() == tree.content_hash()


def test_the_contract_names_no_limb_digit_or_joint() -> None:
    forbidden = re.compile(r"thumb|finger|digit|palm|wrist|elbow|shoulder|\barms?\b|\bhands?\b|\bleg|knee|jaw|gripper", re.IGNORECASE)
    for model in (SkillLibraryV1, TaskTreeV1, ExecutionRecordV1):
        schema = json.dumps(model.model_json_schema())
        assert not forbidden.search(schema), f"{model.__name__}: {forbidden.search(schema).group(0)}"
    example = clear_bench_library().model_dump_json()
    assert not forbidden.search(example)


def test_arguments_are_parsed_by_their_declared_type() -> None:
    assert parse_argument(ArgumentType.INTEGER, "3") == 3
    assert parse_argument(ArgumentType.NUMBER, "0.5") == 0.5
    assert parse_argument(ArgumentType.BOOLEAN, "true") is True
    assert parse_argument(ArgumentType.IDENTIFIER, "zoo_dual_arm") == "zoo_dual_arm"
    for kind, value in ((ArgumentType.INTEGER, "3.5"), (ArgumentType.NUMBER, "nan"), (ArgumentType.BOOLEAN, "yes"), (ArgumentType.IDENTIFIER, "9 lives")):
        with pytest.raises(ValueError):
            parse_argument(kind, value)
