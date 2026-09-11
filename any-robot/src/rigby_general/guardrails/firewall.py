"""Refuse requests outside the system's scope, before anything tries to read them.

A recognizer keyed on spatial vocabulary has a specific weakness, and the audit
found it: "drive across the room and open the door" contains the word *across*,
and a sweep is a motion that goes across things, so the schema recognizer
happily produced one. The request has nothing to do with sweeping. A fixed-base
arm cannot drive anywhere, and nothing in the closed class means "open a door".

The fix is the same one v2 uses -- ``guardrails/firewall.py`` there rejects on
capability before the planner is reached -- and the ordering is the point.
Recognizing first and filtering after would mean the filter has to understand the
recognizer's output; filtering first means it only has to understand the request.

Each rule names what the system cannot do rather than listing forbidden words, so
a refusal explains itself: "this is a fixed-base arm; it cannot travel" is
actionable, and "unsupported" is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class FirewallRule(StrEnum):
    LOCOMOTION = "capability.locomotion.v1"
    """Travel to somewhere else. A bolted-down arm has no somewhere else."""

    DEFORMABLE_OR_FLUID = "capability.deformable-fluid.v1"
    """Cloth, rope, liquid, dough. The simulator models rigid bodies."""

    MULTI_AGENT = "capability.single-character.v1"
    """Two robots interacting. One model, one robot."""

    DIRECT_EXECUTION = "capability.simulation-only.v1"
    """Commanding real hardware. This authors motion for models."""

    COGNITIVE = "capability.motion-only.v1"
    """Asking the system to explain or decide rather than to move."""

    RELEASE = "capability.no-release-schema.v1"
    """Letting go of a held object.

    The closed class can express taking hold of something and carrying it, and
    cannot express putting it down again. That asymmetry is a real gap in the
    vocabulary rather than a property of any robot, so the refusal names the
    missing schema instead of blaming the body.

    It is a firewall rule rather than a planner miss because the generic motion
    verbs happily swallow these phrasings: "let go" contains *go* and "put it
    down" contains *put*, and both fall through to a reach that certifies. A
    wrong motion reported as certified is worse than no motion at all."""

    HELD_OBJECT = "capability.no-carried-state.v1"
    """Constraining a later motion by something still being held.

    A segment chain carries no notion of an object staying grasped across a
    boundary, so "reach upwards while holding the block" reads as a bare reach
    and the holding is silently dropped."""


_RULES: tuple[tuple[FirewallRule, re.Pattern[str], str], ...] = (
    (
        FirewallRule.RELEASE,
        re.compile(
            r"\b(let go( of)?|release|drop|ungrasp|un-?grip|open the (gripper|jaw|hand)"
            r"|set (it|the \w+) down|put (it|the \w+) down|place (it|the \w+) down"
            r"|let (it|the \w+) go)\b",
            re.IGNORECASE,
        ),
        "the closed class has no release schema: this system can express taking "
        "hold of an object and carrying it, but not putting it down again",
    ),
    (
        FirewallRule.HELD_OBJECT,
        re.compile(
            r"\bwhile (still )?(holding|carrying|gripping|grasping)\b"
            r"|\b(holding|carrying) (it|the \w+),? (and|then)\b"
            r"|\bwith the \w+ in (its|the) (grip|gripper|hand|jaws)\b",
            re.IGNORECASE,
        ),
        "a segment carries no notion of an object still being held, so a motion "
        "conditioned on holding one cannot be expressed without silently dropping "
        "that condition",
    ),
    (
        FirewallRule.LOCOMOTION,
        re.compile(
            r"\b(drive|walk|roll over to|navigate|travel|wander|patrol)\b"
            r"|\b(go|move|come|get)\s+(to|into|over to|across)\s+the\s+"
            r"(room|kitchen|hall|corridor|door|garden|street|building|other side)\b"
            r"|\bacross the (room|floor|hall|yard|street)\b",
            re.IGNORECASE,
        ),
        "this is a fixed-base arm; it is bolted down and cannot travel anywhere",
    ),
    (
        FirewallRule.DEFORMABLE_OR_FLUID,
        re.compile(
            r"\b(swim|pour|splash|fold (the )?(cloth|towel|laundry|shirt)|knead|"
            r"dough|rope|cable tie|liquid|water|sand|fabric)\b",
            re.IGNORECASE,
        ),
        "the simulation models rigid bodies only; deformables and fluids are not "
        "represented",
    ),
    (
        FirewallRule.MULTI_AGENT,
        re.compile(
            r"\b(the other robot|another robot|each other|both robots|"
            r"hand it to (me|the human|the person))\b",
            re.IGNORECASE,
        ),
        "only one robot is loaded; interaction between agents is not represented",
    ),
    (
        FirewallRule.DIRECT_EXECUTION,
        re.compile(
            r"\b(real robot|actual robot|physical robot|real hardware|"
            r"on the actual|send (it )?to the robot|execute on)\b",
            re.IGNORECASE,
        ),
        "this authors motion for a model; it does not command hardware",
    ),
    (
        FirewallRule.COGNITIVE,
        re.compile(
            r"\b(explain|describe|tell me (why|how)|what do you think|"
            r"your reasoning|decide whether|prove|solve)\b",
            re.IGNORECASE,
        ),
        "this turns requests into motion; it does not answer questions",
    ),
)


@dataclass(frozen=True, slots=True)
class FirewallDecision:
    allowed: bool
    rule: FirewallRule | None = None
    reason: str = ""
    matched: str = ""

    def as_details(self) -> dict[str, str]:
        return {
            "rule": self.rule.value if self.rule else "",
            "matched": self.matched,
        }


ALLOWED = FirewallDecision(allowed=True)


def screen(prompt: str) -> FirewallDecision:
    """Check a request against every rule, before any recognizer sees it."""

    for rule, pattern, reason in _RULES:
        match = pattern.search(prompt)
        if match:
            return FirewallDecision(
                allowed=False,
                rule=rule,
                reason=reason,
                matched=match.group(0),
            )
    return ALLOWED
