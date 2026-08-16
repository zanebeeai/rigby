"""Every action in the primitive vocabulary must be accounted for.

The failure this guards against is silent: someone adds a ``BodyAction`` or an
``ObjectAction``, the compiler grows a branch for it, and the deterministic
layer never learns it exists. Registering with ``analyzer=None`` is allowed —
that says "declared, ported in PR 02b" — but leaving the enum member out
entirely is not.
"""

from __future__ import annotations

import pytest

from rigby_poc.analysis import registry
from rigby_poc.models import BodyAction, ObjectAction


def test_every_body_action_has_a_registry_entry() -> None:
    missing = [action.name for action in BodyAction if action not in registry.BODY_ANALYZERS]

    assert not missing, (
        f"BodyAction members with no analyzer entry: {missing}. Add them to "
        "rigby_poc.analysis.registry.BODY_ANALYZERS."
    )


def test_every_object_action_has_a_registry_entry() -> None:
    missing = [
        action.name for action in ObjectAction if action not in registry.OBJECT_ANALYZERS
    ]

    assert not missing, (
        f"ObjectAction members with no analyzer entry: {missing}. Add them to "
        "rigby_poc.analysis.registry.OBJECT_ANALYZERS."
    )


def test_unregistered_actions_reports_nothing() -> None:
    assert registry.unregistered_actions() == []


def test_the_registry_holds_no_entry_for_a_removed_action() -> None:
    body_names = {action for action in BodyAction}
    object_names = {action for action in ObjectAction}

    assert set(registry.BODY_ANALYZERS) == body_names
    assert set(registry.OBJECT_ANALYZERS) == object_names


def test_a_new_enum_member_without_an_analyzer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must actually fire, not merely be present.

    A synthetic enum member stands in for a future action; the coverage check
    has to notice it. Without this, a registry that silently accepted anything
    would still pass every assertion above.
    """

    class ExtendedBodyAction(str):
        name = "SOMERSAULT"

    invented = ExtendedBodyAction("somersault")
    monkeypatch.setattr(
        registry, "BodyAction", list(BodyAction) + [invented], raising=True
    )

    assert registry.unregistered_actions() == ["BodyAction.SOMERSAULT"]


def test_deferred_actions_names_the_pr_that_ports_each_one() -> None:
    deferred = registry.deferred_actions()

    assert deferred, "nothing is deferred; update this test when 02b-02e land"
    for action, entry in registry.BODY_ANALYZERS.items():
        if not entry.ported:
            assert entry.plan in {"02b", "02c", "02d", "02e"}, action
            assert entry.owner.startswith("compiler."), action
    for action, entry in registry.OBJECT_ANALYZERS.items():
        if not entry.ported:
            assert entry.plan in {"02b", "02c", "02d", "02e"}, action
            assert entry.owner.startswith("compiler."), action
