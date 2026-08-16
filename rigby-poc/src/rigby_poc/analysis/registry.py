"""Action-keyed registry of clip analyzers.

Every ``BodyAction`` and every ``ObjectAction`` must appear here. An action
whose metric block still lives inside ``compiler.py`` registers with
``analyzer=None`` and the plan PR that will port it — it is *declared*, just not
extracted yet. What the registry forbids is an action nobody has accounted for:
:mod:`tests.test_registry_coverage` fails the suite the moment a new enum member
is added without an entry, which is what stops the deterministic layer silently
falling behind the primitive vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..models import BodyAction, ObjectAction
from .context import AnalysisContext


class Analyzer(Protocol):
    """Compute the metric keys one action contributes to a clip."""

    def __call__(self, ctx: AnalysisContext) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AnalyzerEntry:
    """One action's registration.

    ``analyzer`` is ``None`` while the action's metric block still lives in the
    compiler. ``owner`` says which module currently computes it, and ``plan``
    names the PR that moves it here.
    """

    analyzer: Analyzer | None
    owner: str
    plan: str
    note: str = ""

    @property
    def ported(self) -> bool:
        return self.analyzer is not None


BODY_ANALYZERS: dict[BodyAction, AnalyzerEntry] = {
    BodyAction.HOLD: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.STEP: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.WALK: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.RUN: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.TURN: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.CROUCH: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.JUMP: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.KICK: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.DANCE: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.CLIMB: AnalyzerEntry(
        None,
        "compiler._compile_full_body",
        "02b",
        note="needs the commanded climb support targets persisted first",
    ),
    BodyAction.ROTATE: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
    BodyAction.POSE: AnalyzerEntry(None, "compiler._compile_full_body", "02b"),
}

OBJECT_ANALYZERS: dict[ObjectAction, AnalyzerEntry] = {
    ObjectAction.THROW: AnalyzerEntry(None, "compiler._compile_object_interaction", "02d"),
    ObjectAction.CATCH: AnalyzerEntry(None, "compiler._compile_object_interaction", "02d"),
    ObjectAction.PUSH: AnalyzerEntry(None, "compiler._compile_object_interaction", "02d"),
    ObjectAction.PULL: AnalyzerEntry(None, "compiler._compile_object_interaction", "02d"),
    ObjectAction.ROLL: AnalyzerEntry(
        None,
        "compiler._compile_object_interaction",
        "02d",
        note="rolling_angle_rad is an accumulator; expect a documented tolerance",
    ),
    ObjectAction.SPIN: AnalyzerEntry(
        None,
        "compiler._compile_object_interaction",
        "02d",
        note="support_spin_angle_rad is an accumulator; expect a documented tolerance",
    ),
    ObjectAction.PLACE: AnalyzerEntry(None, "compiler._compile_object_interaction", "02d"),
    ObjectAction.DROP: AnalyzerEntry(None, "compiler._compile_object_interaction", "02d"),
    ObjectAction.HANDOFF: AnalyzerEntry(None, "compiler._compile_object_handoff", "02d"),
}


def body_analyzer(action: BodyAction) -> AnalyzerEntry:
    try:
        return BODY_ANALYZERS[action]
    except KeyError:  # pragma: no cover - guarded by test_registry_coverage
        raise KeyError(f"no analyzer registered for BodyAction.{action.name}") from None


def object_analyzer(action: ObjectAction) -> AnalyzerEntry:
    try:
        return OBJECT_ANALYZERS[action]
    except KeyError:  # pragma: no cover - guarded by test_registry_coverage
        raise KeyError(
            f"no analyzer registered for ObjectAction.{action.name}"
        ) from None


def unregistered_actions() -> list[str]:
    """Enum members with no registry entry. Empty is the only healthy answer."""

    missing = [
        f"BodyAction.{action.name}"
        for action in BodyAction
        if action not in BODY_ANALYZERS
    ]
    missing.extend(
        f"ObjectAction.{action.name}"
        for action in ObjectAction
        if action not in OBJECT_ANALYZERS
    )
    return missing


def deferred_actions() -> list[str]:
    """Registered actions whose metric block still lives in the compiler."""

    deferred = [
        f"BodyAction.{action.name}"
        for action, entry in BODY_ANALYZERS.items()
        if not entry.ported
    ]
    deferred.extend(
        f"ObjectAction.{action.name}"
        for action, entry in OBJECT_ANALYZERS.items()
        if not entry.ported
    )
    return deferred


def register_body_analyzer(
    action: BodyAction, analyzer: Analyzer, *, owner: str = "analysis"
) -> None:
    BODY_ANALYZERS[action] = AnalyzerEntry(analyzer, owner, "ported")


def register_object_analyzer(
    action: ObjectAction, analyzer: Analyzer, *, owner: str = "analysis"
) -> None:
    OBJECT_ANALYZERS[action] = AnalyzerEntry(analyzer, owner, "ported")


__all__ = [
    "Analyzer",
    "AnalyzerEntry",
    "BODY_ANALYZERS",
    "OBJECT_ANALYZERS",
    "body_analyzer",
    "deferred_actions",
    "object_analyzer",
    "register_body_analyzer",
    "register_object_analyzer",
    "unregistered_actions",
]
