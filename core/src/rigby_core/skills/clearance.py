"""ClearWorkArea: a generated tree over the shared TransferObject subskills, one grounding per object.

A work area with several objects is cleared by transferring each to its own
cell of the destination and then looking at the area. Nothing here is
written per instance: the library is generated from the list of objects
and cells, every object is transferred by the same ``transfer_object``
definition (the G10 skill, unchanged, with its two bounded loops and its
sensor-decided verifications), and only the bindings differ. The tree is
deep by construction -- root, the clearing loop, a pass, a per-object
selector, the transfer, its loops, its leaves -- which is what recursion
buys: one definition per level, reused across objects.

Recovery lives at three levels. A leaf's verification may be re-observed
within its budget (G09); a subgoal may be retried within its loop (G10); and
an object whose transfer fails is passed over and returned to on the next
pass, until the pass budget is spent. Every pass ends with a look at the
work area and the cells from the declared cameras, and what they see
decides which placements stand: a cube knocked out of its cell by a later
placement loses its ``placed`` fact and the next pass transfers it again.
The root succeeds only when every object is observed in its cell and the
area is observed clear: no child's success counts toward an unfinished
root, and no stale belief does either.

``observe_each_pass=False`` keeps the first structure (one observation after
the clearing loop, the loop closed on the transfers' own verdicts); it is
what the baseline evidence ran, and it is kept only for that comparison.

The *flat* variant is the same leaves in the same order with the same
retry budget, but no recovery structure: each loop makes one attempt, a
failed object ends the pass, and a leaf that fails is retried in place
without re-observing. It is what a script of the same primitives would do,
and it is the comparison the catalog asks for.
"""

from __future__ import annotations

from .contract import (
    ChildRefV1,
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
    TerminationRuleV1,
)
from .examples import A, BIND_THREE, OWN_THREE, PERCEPTION, RETRY_BUDGET, THREE, transfer_object_library


PASS_BUDGET = 2
"""How many passes over the objects the clearing loop may make."""
SECONDS_PER_OBJECT = 90.0
"""The root's cap, per object: three quarters of the single-object cap, since a pass over placed objects costs nothing."""
LEAVES = ("observe_object", "acquire", "verify_hold", "transport", "release", "verify_placement")


def episode_cap_s(count: int) -> float:
    return SECONDS_PER_OBJECT * count


def clear_work_area_library(objects: tuple[str, ...], cells: tuple[str, ...], *, flat: bool = False, retry_budget: int = RETRY_BUDGET, pass_budget: int = PASS_BUDGET,
                            observe_each_pass: bool = True) -> SkillLibraryV1:
    """The clearance library for these objects and cells; ``flat`` removes
    the recovery structure and keeps the leaves and their retry budget;
    ``observe_each_pass`` closes the clearing loop on the cameras."""

    if len(objects) != len(cells) or not objects:
        raise ValueError("one cell per object, and at least one object")
    if len(set(objects)) != len(objects) or len(set(cells)) != len(cells):
        raise ValueError("objects and cells must be distinct")
    base = transfer_object_library()
    cap = episode_cap_s(len(objects))
    skills: list[SkillDefinitionV1] = []
    for definition in base.skills:
        if not flat:
            skills.append(definition)
            continue
        payload = definition.model_dump(mode="json")
        if definition.kind is NodeKind.REPEAT_UNTIL:
            payload["loop"]["max_attempts"] = 1
        if definition.skill_id in LEAVES:
            payload["recovery"] = RecoveryV1(max_attempts=retry_budget).model_dump(mode="json")
        skills.append(SkillDefinitionV1.model_validate(payload))
    own = (ResourceClaimV1(resource="effector:$effector"), *[ResourceClaimV1(resource=f"object:{name}") for name in objects], ResourceClaimV1(resource="perception", mode=ResourceMode.SHARED))
    effector_only = (A("effector"),)
    skills.append(SkillDefinitionV1(
        skill_id="stand_by", kind=NodeKind.PRIMITIVE, arguments=THREE, timeout_s=10.0, controller="contact.transfer.stand_by",
        description="Hold still where the arm is; the alternative a pass takes when an object's transfer fails, so the pass goes on to the next object.",
        termination=TerminationRuleV1(require_effects=False), resources=OWN_THREE,
    ))
    skills.append(SkillDefinitionV1(
        skill_id="try_transfer", kind=NodeKind.SELECTOR, arguments=THREE, timeout_s=cap, termination=TerminationRuleV1(require_effects=False), resources=OWN_THREE,
        description="Transfer the object; failing that, stand by and leave it for the next pass.",
        children=(ChildRefV1(skill="transfer_object", bindings=BIND_THREE, rank=0), ChildRefV1(skill="stand_by", bindings=BIND_THREE, rank=1)),
    ))
    skills.append(SkillDefinitionV1(
        skill_id="clear_pass", kind=NodeKind.SEQUENCE, arguments=effector_only, timeout_s=cap, termination=TerminationRuleV1(require_effects=False), resources=own,
        description="One pass over every object, each transferred to its own cell by the shared transfer skill.",
        children=tuple(ChildRefV1(skill="transfer_object" if flat else "try_transfer", bindings={"object": name, "destination": cell, "effector": "$effector"}) for name, cell in zip(objects, cells)),
    ))
    if observe_each_pass:
        skills.append(SkillDefinitionV1(
            skill_id="observe_work_area", kind=NodeKind.OBSERVE, arguments=(), timeout_s=15.0, resources=PERCEPTION,
            description="Look at the work area and the cells from the declared cameras: is any object still in the area, and does every object seen stand in its cell?",
            observation=ObservationSpecV1(evidence=("work_area_clear", "objects_in_cells"), source="camera"),
        ))
        skills.append(SkillDefinitionV1(
            skill_id="clear_pass_and_check", kind=NodeKind.SEQUENCE, arguments=effector_only, timeout_s=cap, termination=TerminationRuleV1(require_effects=False), resources=own,
            description="One pass over every object, then a look at the work area and the cells: what the cameras see decides which placements stand.",
            children=(ChildRefV1(skill="clear_pass", bindings={"effector": "$effector"}), ChildRefV1(skill="observe_work_area", bindings={})),
        ))
        skills.append(SkillDefinitionV1(
            skill_id="clear_until_clear", kind=NodeKind.REPEAT_UNTIL, arguments=effector_only, timeout_s=cap, resources=own,
            description="Pass over the objects and look, until every object is seen in its cell and the area is seen clear, within the pass budget.",
            loop=LoopSpecV1(until=PredicateRefV1(name="area_cleared"), max_attempts=1 if flat else pass_budget),
            children=(ChildRefV1(skill="clear_pass_and_check", bindings={"effector": "$effector"}),),
        ))
        skills.append(SkillDefinitionV1(
            skill_id="clear_work_area", kind=NodeKind.SEQUENCE, arguments=effector_only, timeout_s=cap, resources=own,
            description="Clear the work area: every object observed in its cell, and the area observed clear.",
            effects=(PredicateRefV1(name="all_objects_placed"), PredicateRefV1(name="work_area_clear")),
            children=(ChildRefV1(skill="clear_until_clear", bindings={"effector": "$effector"}),),
        ))
    else:
        skills.append(SkillDefinitionV1(
            skill_id="clear_until_clear", kind=NodeKind.REPEAT_UNTIL, arguments=effector_only, timeout_s=cap, resources=own,
            description="Pass over the objects until every one is placed, within the pass budget.",
            loop=LoopSpecV1(until=PredicateRefV1(name="all_objects_placed"), max_attempts=1 if flat else pass_budget),
            children=(ChildRefV1(skill="clear_pass", bindings={"effector": "$effector"}),),
        ))
        skills.append(SkillDefinitionV1(
            skill_id="observe_work_area", kind=NodeKind.OBSERVE, arguments=(), timeout_s=15.0, resources=PERCEPTION,
            description="Look at the work area from the declared cameras: is any object still in it?",
            observation=ObservationSpecV1(evidence=("work_area_clear",), source="camera"),
        ))
        skills.append(SkillDefinitionV1(
            skill_id="clear_work_area", kind=NodeKind.SEQUENCE, arguments=effector_only, timeout_s=cap, resources=own,
            description="Clear the work area: every object placed in its cell, and the area observed clear.",
            effects=(PredicateRefV1(name="all_objects_placed"), PredicateRefV1(name="work_area_clear")),
            children=(ChildRefV1(skill="clear_until_clear", bindings={"effector": "$effector"}), ChildRefV1(skill="observe_work_area", bindings={})),
        ))
    predicates = (*base.predicates,
                  PredicateSpecV1(name="all_objects_placed", parameters=(), description="every designated object's placement in its own cell stands: decided pass from the declared sensors and not since seen elsewhere"),
                  PredicateSpecV1(name="work_area_clear", parameters=(), description="the declared cameras saw no object left in the work area"),
                  PredicateSpecV1(name="area_cleared", parameters=(), description="every designated object's placement stands and the area was seen clear: the clearing loop's exit"))
    version = "v2" if observe_each_pass else "v1"
    return SkillLibraryV1(library_id=f"clear_work_area_{version}_{'flat' if flat else 'tree'}_{len(objects)}",
                          description=f"ClearWorkArea over {len(objects)} objects: {'the same leaves with no recovery structure' if flat else 'the generated tree with recovery at three levels'}"
                                      + ("; every pass ends with a look at the area and the cells." if observe_each_pass else "; one look at the area after the clearing loop."),
                          predicates=predicates, skills=tuple(skills))


def object_names(count: int) -> tuple[str, ...]:
    return tuple(f"cube_{index:02d}" for index in range(1, count + 1))


def cell_names(count: int) -> tuple[str, ...]:
    return tuple(f"cell_{index:02d}" for index in range(1, count + 1))


__all__ = ["LEAVES", "PASS_BUDGET", "SECONDS_PER_OBJECT", "cell_names", "clear_work_area_library", "episode_cap_s", "object_names"]
