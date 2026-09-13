"""A nested example library shared by the skill tests and the G07 evidence.

Clearing a bench: observe what is there, then, within a bounded number of
attempts, transfer the object with whichever of two manipulators can, and
verify the placement with an observation the transfer cannot fake. Five
levels deep from the root to the transfer leaf. Nothing in it names a limb:
``primary`` and ``secondary`` are whatever the body binds them to.
"""

from __future__ import annotations

from rigby_core.skills import (
    ArgumentSpecV1,
    ArgumentType,
    ChildRefV1,
    LoopSpecV1,
    NodeKind,
    ObservationSpecV1,
    PredicateRefV1,
    PredicateSpecV1,
    RangeV1,
    RecoveryV1,
    ResourceClaimV1,
    ResourceMode,
    SkillDefinitionV1,
    SkillLibraryV1,
    TerminationRuleV1,
    ValidityContextV1,
)


def P(name: str, *arguments: str, negate: bool = False) -> PredicateRefV1:
    return PredicateRefV1(name=name, arguments=tuple(arguments), negate=negate)


def A(name: str) -> ArgumentSpecV1:
    return ArgumentSpecV1(name=name, type=ArgumentType.IDENTIFIER)


FOUR = (A("object"), A("destination"), A("effector"), A("alternate"))
OWN_ALL = (
    ResourceClaimV1(resource="effector:$effector"),
    ResourceClaimV1(resource="effector:$alternate"),
    ResourceClaimV1(resource="object:$object"),
    ResourceClaimV1(resource="perception", mode=ResourceMode.SHARED),
)
BIND_ALL = {"object": "$object", "destination": "$destination", "effector": "$effector", "alternate": "$alternate"}
ARGUMENTS = {"object": "cube", "destination": "platform", "effector": "primary", "alternate": "secondary"}


def clear_bench_library() -> SkillLibraryV1:
    return SkillLibraryV1(
        library_id="clear_bench_v1",
        description="Observe, then transfer with any manipulator and verify, within a bounded number of attempts.",
        predicates=(
            PredicateSpecV1(name="object_known", parameters=("object",), description="the object's pose is in the belief"),
            PredicateSpecV1(name="object_placed", parameters=("object", "destination"), description="an independent evaluator saw the object resting, released, inside the destination"),
            PredicateSpecV1(name="object_in_reach", parameters=("object",), description="the object is inside the body's measured envelope"),
        ),
        skills=(
            SkillDefinitionV1(
                skill_id="clear_bench", kind=NodeKind.SEQUENCE, arguments=FOUR,
                description="Observe the bench, then place the object within a bounded number of attempts.",
                effects=(P("object_placed", "$object", "$destination"),), timeout_s=120.0, resources=OWN_ALL,
                validity=ValidityContextV1(environments=("g06_transfer_v1",), operating_range=(RangeV1(quantity="object_span", low=0.019, high=0.046, units="m"),)),
                children=(ChildRefV1(skill="observe_inventory", bindings={"object": "$object"}), ChildRefV1(skill="place_until_done", bindings=BIND_ALL)),
            ),
            SkillDefinitionV1(
                skill_id="observe_inventory", kind=NodeKind.OBSERVE, arguments=(A("object"),),
                effects=(P("object_known", "$object"),), timeout_s=5.0,
                resources=(ResourceClaimV1(resource="perception", mode=ResourceMode.SHARED),),
                observation=ObservationSpecV1(evidence=("object_pose",), source="scene_inventory"),
            ),
            SkillDefinitionV1(
                skill_id="place_until_done", kind=NodeKind.REPEAT_UNTIL, arguments=FOUR,
                initiation=(P("object_known", "$object"),), effects=(P("object_placed", "$object", "$destination"),), timeout_s=100.0, resources=OWN_ALL,
                loop=LoopSpecV1(until=P("object_placed", "$object", "$destination"), progress=P("object_in_reach", "$object"), max_attempts=2),
                children=(ChildRefV1(skill="transfer_and_verify", bindings=BIND_ALL),),
            ),
            SkillDefinitionV1(
                skill_id="transfer_and_verify", kind=NodeKind.SEQUENCE, arguments=FOUR, timeout_s=60.0,
                termination=TerminationRuleV1(require_effects=False), resources=OWN_ALL,
                children=(ChildRefV1(skill="transfer_with_any_effector", bindings=BIND_ALL), ChildRefV1(skill="observe_placement", bindings={"object": "$object", "destination": "$destination"})),
            ),
            SkillDefinitionV1(
                skill_id="transfer_with_any_effector", kind=NodeKind.SELECTOR, arguments=FOUR, timeout_s=50.0,
                termination=TerminationRuleV1(require_effects=False),
                resources=(ResourceClaimV1(resource="effector:$effector"), ResourceClaimV1(resource="effector:$alternate"), ResourceClaimV1(resource="object:$object")),
                children=(
                    ChildRefV1(skill="transfer", bindings={"object": "$object", "destination": "$destination", "effector": "$effector"}, rank=0),
                    ChildRefV1(skill="transfer", bindings={"object": "$object", "destination": "$destination", "effector": "$alternate"}, rank=1),
                ),
            ),
            SkillDefinitionV1(
                skill_id="transfer", kind=NodeKind.PRIMITIVE, arguments=(A("object"), A("destination"), A("effector")),
                requirements=("grasping_effector",), initiation=(P("object_known", "$object"),),
                effects=(P("object_placed", "$object", "$destination"),), timeout_s=40.0, controller="contact.transfer",
                resources=(ResourceClaimV1(resource="effector:$effector"), ResourceClaimV1(resource="object:$object")),
            ),
            SkillDefinitionV1(
                skill_id="observe_placement", kind=NodeKind.OBSERVE, arguments=(A("object"), A("destination")), timeout_s=5.0,
                resources=(ResourceClaimV1(resource="perception", mode=ResourceMode.SHARED),),
                observation=ObservationSpecV1(evidence=("object_placed",), source="placement_evaluator"),
            ),
        ),
    )


def with_recovery(library: SkillLibraryV1) -> SkillLibraryV1:
    """The same library with the transfer leaf allowed two attempts, a
    re-observation between them."""

    payload = library.model_dump(mode="json")
    for definition in payload["skills"]:
        if definition["skill_id"] == "transfer":
            definition["recovery"] = RecoveryV1(skill="observe_inventory", max_attempts=2).model_dump(mode="json")
    return SkillLibraryV1.model_validate(payload)
