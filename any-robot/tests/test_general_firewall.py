"""Capability screening, and the line it has to walk.

A recognizer keyed on spatial vocabulary finds a sweep in "drive across the
room", because the word is there. The firewall's job is to catch that without
also catching "sweep across the bench" -- and the second half is the hard half,
since over-refusing is just as much a failure as under-refusing, only quieter.
"""

from __future__ import annotations

import pytest

from rigby_general.guardrails import FirewallRule, screen


@pytest.mark.parametrize(
    "prompt, rule",
    [
        ("drive across the room and open the door", FirewallRule.LOCOMOTION),
        ("walk over to the kitchen", FirewallRule.LOCOMOTION),
        ("go to the door", FirewallRule.LOCOMOTION),
        ("swim to the bottom of the pool", FirewallRule.DEFORMABLE_OR_FLUID),
        ("fold the towel neatly", FirewallRule.DEFORMABLE_OR_FLUID),
        ("pour the water out", FirewallRule.DEFORMABLE_OR_FLUID),
        ("hand it to me when you are done", FirewallRule.MULTI_AGENT),
        ("pass it to the other robot", FirewallRule.MULTI_AGENT),
        ("run this on the real robot", FirewallRule.DIRECT_EXECUTION),
        ("explain your reasoning", FirewallRule.COGNITIVE),
        ("what do you think you should do next", FirewallRule.COGNITIVE),
        ("solve the halting problem", FirewallRule.COGNITIVE),
    ],
)
def test_out_of_scope_requests_are_refused_by_the_right_rule(
    prompt: str, rule: FirewallRule
) -> None:
    decision = screen(prompt)
    assert not decision.allowed
    assert decision.rule is rule
    assert decision.reason


@pytest.mark.parametrize(
    "prompt",
    [
        "reach out as far as you can",
        "sweep slowly across in front of you",
        "sweep across the workspace, then hold still",
        "wave three times",
        "trace a big circle",
        "lower the tool down toward the table",
        "reach out quickly, just a little",
        "reach out as far as you can and then come back",
        "move back a little",
        "hold still",
        "extend gently, halfway",
        "withdraw and come back",
    ],
)
def test_legitimate_motion_requests_pass(prompt: str) -> None:
    """Over-refusing is a failure too, and a quieter one.

    Every prompt here is one the demo set or the audit actually uses, so a rule
    that widened far enough to catch them would show up as a broken demo rather
    than as a caught mistake.
    """

    assert screen(prompt).allowed


def test_a_refusal_explains_the_capability_not_the_word() -> None:
    decision = screen("drive across the room")
    assert "fixed-base" in decision.reason
    assert decision.matched
