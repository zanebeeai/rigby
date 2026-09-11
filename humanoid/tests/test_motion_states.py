"""The state controller: what each state solves, checks, and looks at.

Two claims are under test. That the catalog is internally coherent -- no state
both solves and holds the same region, every gate is reachable, the plan is
ordered. And that scoping is *safe*: it may never empty the repair set, because
a filter that turns a fixable candidate into an unfixable one costs more than
the simulations it saves.
"""

from __future__ import annotations

import pytest

from rigby_poc.embodied_inspector import scope_proposals_to_active_regions
from rigby_poc.motion_states import (
    GAZE_TARGETS,
    MotionStateError,
    every_gate,
    gates_for,
    plan,
    state_catalog,
    state_for,
    states,
)

#: no compile, no corpus, no browser -- see docs/testing.md
pytestmark = pytest.mark.fast


def test_the_plan_is_ordered_and_covers_the_grab_lifecycle() -> None:
    """'move arm to box, grab box, move arm up', made explicit and ordered."""
    sequence = plan("grab")
    assert [s.name for s in sequence] == [
        "reach", "preshape", "contact", "close", "lift", "hold", "recover",
    ]
    assert [s.order for s in sequence] == list(range(len(sequence)))
    assert plan("gesture") == (), "only grab is controlled today"


def test_no_state_both_solves_and_holds_the_same_region() -> None:
    for name, state in states().items():
        overlap = set(state.active_regions) & set(state.inert_regions)
        assert not overlap, f"{name} lists {sorted(overlap)} as active and inert"


def test_every_state_explains_its_scoping_decision() -> None:
    """A gate that stops firing must be reviewable later.

    Scoping decides which checks run at all, so an unexplained one is a check
    silently switched off with no record of why.
    """
    for name, state in states().items():
        assert len(state.rationale) > 40, f"{name} has no usable rationale"
        assert state.gaze in GAZE_TARGETS


def test_the_digits_are_solved_when_closing_and_held_while_lifting() -> None:
    """The specific correctness claim, not just a cost claim.

    During lift the digits are inert *because they are holding the object*.
    Re-solving them there changes the grip currently carrying the block.
    """
    close = state_for("close")
    lift = state_for("lift")
    assert close is not None and lift is not None
    assert close.solves("right_digits")
    assert not lift.solves("right_digits")
    assert "digits" in lift.inert_regions
    assert lift.solves("left_arm"), "sides are generic in the catalog"


def test_the_head_watches_the_grasp_while_it_is_being_formed() -> None:
    assert state_for("close").gaze == "grasp"
    assert state_for("contact").gaze == "object"
    assert state_for("recover").gaze == "forward"


def test_gates_are_scoped_to_states_that_can_judge_them() -> None:
    assert gates_for("reach") == (), "nothing is in contact yet"
    assert "object_socket_alignment" in gates_for("contact")
    assert "digit_chain_and_opposition" in gates_for("close")
    assert "embodied_contact_dynamics" in gates_for("hold")
    # Every gate a state names must be one some state actually runs.
    assert every_gate() >= {
        "object_socket_alignment",
        "palm_surface_orientation",
        "digit_chain_and_opposition",
        "embodied_contact_dynamics",
    }


def test_scoping_drops_repairs_aimed_at_an_inert_region() -> None:
    inspection = {
        "first_failure": {"id": "embodied_contact_dynamics", "body_region": "right_digits"}
    }
    proposals = [
        ("program-a", {"target_body_region": "right_digits"}),
        ("program-b", {"target_body_region": "right_arm"}),
    ]
    kept = scope_proposals_to_active_regions(proposals, inspection)
    assert [c["target_body_region"] for _p, c in kept] == ["right_arm"], (
        "lift holds the digits; the arm is what it solves"
    )


def test_scoping_never_empties_the_repair_set() -> None:
    """The safety property. An empty set turns fixable into unfixable."""
    inspection = {
        "first_failure": {"id": "embodied_contact_dynamics", "body_region": "right_digits"}
    }
    only_inert = [("program-a", {"target_body_region": "right_digits"})]
    assert scope_proposals_to_active_regions(only_inert, inspection) == only_inert


def test_an_unknown_gate_leaves_proposals_untouched() -> None:
    """No state claims the gate, so scoping has no opinion and must not act."""
    proposals = [("p", {"target_body_region": "right_digits"})]
    for inspection in (
        {},
        {"first_failure": None},
        {"first_failure": {"id": "architecture_route", "body_region": "whole_body"}},
    ):
        assert scope_proposals_to_active_regions(proposals, inspection) == proposals


def test_a_catalog_that_contradicts_itself_is_rejected() -> None:
    state_catalog.cache_clear()
    states.cache_clear()
    try:
        catalog = state_catalog()
        assert catalog["states"], "fixture must load a real catalog"
    finally:
        state_catalog.cache_clear()
        states.cache_clear()
    assert issubclass(MotionStateError, ValueError)
