"""The generated ClearWorkArea library: deep by construction, one grounding per object, and a flat twin with equal retries.

No body, no physics. For 3, 5 and 10 objects the library validates, expands
from its root with the shared transfer definition bound once per object,
reaches at least depth 4 (it reaches 11), and reuses every transfer
subskill across objects. The flat variant keeps the same leaves in the
same order with the same retry budget and no loop that could try twice. A
scripted run shows a pass passing over a failed object and the next pass
returning to it; a placement the cameras later see undone losing its fact
and the next pass transferring the cube again, where the flat twin fails;
and the root failing when the area is not clear even though every child
that ran succeeded. The first structure (one observation after the loop)
is kept behind a flag for the baseline evidence and checked here too.
"""

from __future__ import annotations

from typing import Any

import pytest

from rigby_core.skills import Belief, Interrupt, LeafContext, LeafOutcome, MonotonicClock, NodeKind, Verdict, execute
from rigby_core.skills.clearance import LEAVES, cell_names, clear_work_area_library, episode_cap_s, object_names


@pytest.mark.parametrize("count", [3, 5, 10])
def test_library_expands_deep_with_one_transfer_per_object(count):
    objects, cells = object_names(count), cell_names(count)
    library = clear_work_area_library(objects, cells)
    tree = library.expand("clear_work_area", {"effector": "palm"})
    assert tree.max_depth >= 4 and tree.max_depth == 11
    transfers = [node for node in tree.root.walk() if node.skill_id == "transfer_object"]
    assert [node.arguments["object"] for node in transfers] == list(objects)
    assert [node.arguments["destination"] for node in transfers] == list(cells)
    assert len({node.skill_id for node in tree.root.walk() if node.kind in (NodeKind.PRIMITIVE, NodeKind.OBSERVE)}) == len(LEAVES) + 2
    definitions = {s.skill_id for s in library.skills}
    assert len(definitions) == 11 + 7, "eleven shared transfer definitions and seven clearance definitions, whatever the count"
    assert library.skill("clear_work_area").timeout_s == episode_cap_s(count)
    assert [c.skill for c in library.skill("clear_pass_and_check").children] == ["clear_pass", "observe_work_area"], "every pass ends with a look"
    assert library.skill("clear_until_clear").loop.until.name == "area_cleared"


def test_legacy_structure_is_kept_for_the_baseline():
    objects, cells = object_names(3), cell_names(3)
    library = clear_work_area_library(objects, cells, observe_each_pass=False)
    tree = library.expand("clear_work_area", {"effector": "palm"})
    assert library.library_id.startswith("clear_work_area_v1_") and tree.max_depth == 10 and len(library.skills) == 11 + 6
    assert [c.skill for c in library.skill("clear_work_area").children] == ["clear_until_clear", "observe_work_area"], "one look after the loop, the loop closed on the transfers' verdicts"
    assert library.skill("clear_until_clear").loop.until.name == "all_objects_placed"
    assert clear_work_area_library(objects, cells).library_id.startswith("clear_work_area_v2_")


def test_flat_variant_keeps_the_leaves_and_the_retry_budget_without_loops():
    objects, cells = object_names(3), cell_names(3)
    tree_library = clear_work_area_library(objects, cells)
    flat_library = clear_work_area_library(objects, cells, flat=True)
    tree = tree_library.expand("clear_work_area", {"effector": "palm"})
    flat = flat_library.expand("clear_work_area", {"effector": "palm"})
    leaves = lambda t: [n.skill_id for n in t.root.walk() if n.kind in (NodeKind.PRIMITIVE, NodeKind.OBSERVE) and n.skill_id != "stand_by"]
    assert leaves(tree) == leaves(flat)
    for definition in flat_library.skills:
        if definition.kind is NodeKind.REPEAT_UNTIL:
            assert definition.loop.max_attempts == 1
        if definition.skill_id in LEAVES:
            assert definition.recovery.max_attempts == tree_library.skill("place_until_placed").loop.max_attempts
    for definition in tree_library.skills:
        if definition.skill_id in LEAVES:
            assert definition.recovery.max_attempts == 1
    assert "try_transfer" not in {n.skill_id for n in flat.root.walk()}
    assert "try_transfer" in {n.skill_id for n in tree.root.walk()}


def test_library_refuses_mismatched_or_repeated_names():
    with pytest.raises(ValueError):
        clear_work_area_library(("a", "b"), ("c",))
    with pytest.raises(ValueError):
        clear_work_area_library(("a", "a"), ("c", "d"))


class ScriptedArea:
    """Objects that transfer on demand, except one that fails on the first pass."""

    def __init__(self, objects, fails_first: str | None, area_clear_at_end: bool = True, knocked_out_after_first_look: str | None = None) -> None:
        self.objects = objects
        self.fails_first = fails_first
        self.area_clear = area_clear_at_end
        self.knocked = knocked_out_after_first_look
        self.looks = 0
        self.attempts: dict[str, int] = {o: 0 for o in objects}
        self.placed: set[str] = set()
        self.leaves: list[tuple[str, str]] = []

    def run_primitive(self, context: LeafContext) -> LeafOutcome:
        node = context.node
        obj = node.arguments.get("object", "")
        self.leaves.append((node.skill_id, obj))
        context.clock.advance(1.0)
        if node.skill_id == "acquire":
            self.attempts[obj] += 1
            # Nine failures outlast the transfer's own recovery (three acquisitions
            # per placement attempt, three placement attempts) and leave the
            # object to the next pass; three exhaust a flat leaf's budget.
            if obj == self.fails_first and self.attempts[obj] <= 9:
                return LeafOutcome(Verdict.FAILURE, "grasp_not_achieved")
            return LeafOutcome(Verdict.SUCCESS)
        if node.skill_id == "release":
            self.placed.add(obj)
            return LeafOutcome(Verdict.SUCCESS, facts={f"held:{obj}": False})
        return LeafOutcome(Verdict.SUCCESS)

    def observe(self, context: LeafContext) -> dict[str, Any] | None:
        node = context.node
        obj = node.arguments.get("object", "")
        self.leaves.append((node.skill_id, obj))
        context.clock.advance(0.5)
        if node.skill_id == "observe_object":
            return {f"known:{obj}": True, f"pose:{obj}": [0, 0, 0], f"reach:{obj}": True}
        if node.skill_id == "verify_hold":
            return {f"held:{obj}": True}
        if node.skill_id == "verify_placement":
            return {f"placed:{obj}:{node.arguments['destination']}": obj in self.placed, f"held:{obj}": False}
        if node.skill_id == "observe_work_area":
            self.looks += 1
            if self.looks == 1 and self.knocked is not None:
                # the cameras see a cube a later placement knocked out of its cell
                self.placed.discard(self.knocked)
            facts = {"work_area_clear": self.area_clear and len(self.placed) == len(self.objects)}
            facts.update({f"placed:{o}:{c}": o in self.placed for o, c in zip(self.objects, self.cells)})
            return facts
        return None


def predicates(objects, cells):
    pairs = dict(zip(objects, cells))
    return {
        "object_known": lambda b, a: bool(b.get(f"known:{a[0]}", False)),
        "object_held": lambda b, a: bool(b.get(f"held:{a[0]}", False)),
        "object_placed": lambda b, a: bool(b.get(f"placed:{a[0]}:{a[1]}", False)),
        "object_in_reach": lambda b, a: bool(b.get(f"reach:{a[0]}", False)),
        "all_objects_placed": lambda b, a: all(b.get(f"placed:{o}:{c}", False) for o, c in pairs.items()),
        "work_area_clear": lambda b, a: bool(b.get("work_area_clear", False)),
        "area_cleared": lambda b, a: all(b.get(f"placed:{o}:{c}", False) for o, c in pairs.items()) and bool(b.get("work_area_clear", False)),
    }


def test_a_failed_object_is_passed_over_and_returned_to_on_the_next_pass():
    objects, cells = object_names(3), cell_names(3)
    library = clear_work_area_library(objects, cells)
    tree = library.expand("clear_work_area", {"effector": "palm"})
    area = ScriptedArea(objects, fails_first="cube_02")
    area.cells = cells
    record = execute(tree, library, area, predicates(objects, cells), clock=MonotonicClock(), interrupt=Interrupt(), belief=Belief())
    assert record.verdict is Verdict.SUCCESS
    order = [obj for leaf, obj in area.leaves if leaf == "release"]
    assert order == ["cube_01", "cube_03", "cube_02"], "the failed object is left for the second pass"
    passes = next(n for n in record.root.walk() if n.skill_id == "clear_until_clear").evidence["attempts"]
    assert passes == 2


def test_flat_run_stops_at_the_first_failed_object():
    objects, cells = object_names(3), cell_names(3)
    library = clear_work_area_library(objects, cells, flat=True)
    tree = library.expand("clear_work_area", {"effector": "palm"})
    area = ScriptedArea(objects, fails_first="cube_02")
    area.cells = cells
    record = execute(tree, library, area, predicates(objects, cells), clock=MonotonicClock(), interrupt=Interrupt(), belief=Belief())
    assert record.verdict is Verdict.FAILURE
    assert [obj for leaf, obj in area.leaves if leaf == "release"] == ["cube_01"]
    assert area.attempts["cube_02"] == 3, "the leaf retried within the same budget, without re-observing"


def test_root_fails_when_the_area_is_not_clear_even_if_every_transfer_succeeded():
    objects, cells = object_names(3), cell_names(3)
    library = clear_work_area_library(objects, cells)
    tree = library.expand("clear_work_area", {"effector": "palm"})
    area = ScriptedArea(objects, fails_first=None, area_clear_at_end=False)
    area.cells = cells
    record = execute(tree, library, area, predicates(objects, cells), clock=MonotonicClock(), interrupt=Interrupt(), belief=Belief())
    assert record.verdict is Verdict.FAILURE and record.root.reason == "child_failed:0.0", "the clearing loop spends its passes looking at an area that never clears"
    assert next(n for n in record.root.walk() if n.skill_id == "clear_until_clear").evidence["attempts"] == 2
    assert len(area.placed) == 3 and all(record.belief.get(f"placed:{o}:{c}") for o, c in zip(objects, cells)), "every transfer succeeded; the root still does not"


def test_a_placement_the_cameras_see_undone_is_transferred_again_on_the_next_pass():
    objects, cells = object_names(3), cell_names(3)
    library = clear_work_area_library(objects, cells)
    tree = library.expand("clear_work_area", {"effector": "palm"})
    area = ScriptedArea(objects, fails_first=None, knocked_out_after_first_look="cube_01")
    area.cells = cells
    record = execute(tree, library, area, predicates(objects, cells), clock=MonotonicClock(), interrupt=Interrupt(), belief=Belief())
    assert record.verdict is Verdict.SUCCESS
    assert [obj for leaf, obj in area.leaves if leaf == "release"] == ["cube_01", "cube_02", "cube_03", "cube_01"], "the knocked-out cube is transferred again, the others are not touched"
    assert area.looks == 2 and next(n for n in record.root.walk() if n.skill_id == "clear_until_clear").evidence["attempts"] == 2
    # the flat twin looks once, sees the cube out of its cell, and has no second pass
    flat_library = clear_work_area_library(objects, cells, flat=True)
    flat_area = ScriptedArea(objects, fails_first=None, knocked_out_after_first_look="cube_01")
    flat_area.cells = cells
    flat_record = execute(flat_library.expand("clear_work_area", {"effector": "palm"}), flat_library, flat_area, predicates(objects, cells), clock=MonotonicClock(), interrupt=Interrupt(), belief=Belief())
    assert flat_record.verdict is Verdict.FAILURE and flat_area.looks == 1
    assert [obj for leaf, obj in flat_area.leaves if leaf == "release"] == ["cube_01", "cube_02", "cube_03"]
    # the legacy structure trusts the verdicts: it never looks at the cells, so the same knock goes unseen and the root reports success
    legacy_library = clear_work_area_library(objects, cells, observe_each_pass=False)
    legacy_area = ScriptedArea(objects, fails_first=None, knocked_out_after_first_look="cube_01")
    legacy_area.cells = cells
    legacy_record = execute(legacy_library.expand("clear_work_area", {"effector": "palm"}), legacy_library, legacy_area, predicates(objects, cells), clock=MonotonicClock(), interrupt=Interrupt(), belief=Belief())
    assert legacy_record.verdict is Verdict.FAILURE, "with the placed facts reported by the look, even the legacy root cannot claim its effects; on physics the legacy look reported only the area"
