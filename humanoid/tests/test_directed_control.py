"""The directed control path, driven by a script instead of a model.

The corpus exercises the phase controller, which squeezes through its own
`decide()` and never touches the jaw states, the direct acts, the wake logic or
the thresholds. All of that lives in the directed loop, and until now the only
way to test any of it was to spend model calls -- so it was tested by watching
runs and reading transcripts afterwards, which is how a goal that silenced the
planner for forty-three seconds got as far as it did.

A scripted planner is not a model and cannot tell us whether the model chooses
well. It can tell us that a machine which is told the right things does the
right thing, which is the half that should never regress quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigby_poc.gripper.decision.goals import NumericTarget
from rigby_poc.gripper.runs.directed import run

pytestmark = pytest.mark.medium


class _Script:
    """A planner that follows a fixed list of (time, jaws, goal) steps."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.at = 0
        self.jaws = "open"
        self.doing = ""
        self.held = None
        self.spans = {}
        self.calls = 0
        self.transcript = []
        self.plan = []
        self.step = 0
        self.imagination = {}
        self.imagination_confidence = 0.0
        self.seen_states = []

    def due(self, body, seen, now):
        return self.at < len(self.steps) and now >= self.steps[self.at][0]

    def ask(self, body, seen, now, note=""):
        _when, jaws, goal = self.steps[self.at]
        self.at += 1
        self.jaws = jaws
        self.seen_states.append(jaws)
        if goal is None:
            self.held = None
            return None
        metric, value = goal
        self.held = NumericTarget(metric=metric, value=value, set_at_s=now,
                                  using=("move",))
        self.spans = self.held.spans(body, seen)
        return self.held

    def observe(self, *a, **k):
        pass

    def build_imagination(self, *a, **k):
        return {}

    def aim(self, *a, **k):
        return {}


def test_the_jaws_follow_their_state_through_a_whole_run(tmp_path):
    """open, close, hold -- and the arm moving underneath all three."""
    script = _Script([
        (0.0, "open", ("hand_pointing_down", 1.0)),
        (2.5, "close", ("hand_pointing_down", 1.0)),
        (4.5, "hold", ("hand_z_m", 1.0)),
    ])
    got = run(seconds=6.0, planner=script, name="test-jaw-states",
              verbose=False, watch=False,
              output_path=tmp_path / "clip.json")
    assert script.seen_states == ["open", "close", "hold"]
    assert got["deepest_penetration_mm"] < 5.0


def test_a_planner_with_no_goal_still_drives_the_jaws(tmp_path):
    """The jaws are a state, not a consequence of having something to chase.

    This failed when the jaw command sat after the loop's `target is None`
    check: a planner with nothing to drive left the hand frozen wherever the
    fingers happened to be.
    """
    script = _Script([(0.0, "open", None)])
    run(seconds=2.5, planner=script, name="test-jaws-no-goal",
        verbose=False, watch=False, output_path=tmp_path / "clip.json")
    # Reaching here at all means the loop stepped physics without a target;
    # before the fix it spun without ever calling mj_step.
    assert script.seen_states == ["open"]


def test_the_search_cannot_override_the_jaw_state(tmp_path):
    """The arm is the search's; the fingers are not.

    Every earlier version let the two disagree about the same joints.
    """
    script = _Script([(0.0, "close", ("hand_y_m", 0.30))])
    got = run(seconds=3.0, planner=script, name="test-jaws-vs-search",
              verbose=False, watch=False, output_path=tmp_path / "clip.json")
    assert got is not None


def test_a_run_without_the_optional_hooks_still_works(tmp_path):
    """A planner only has to decide. Everything else is instrumentation.

    Requiring `observe` broke a test double that implemented the real contract,
    which is a sign the contract had grown rather than that the double was
    wrong.
    """

    class _Bare:
        jaws = "open"
        doing = ""
        held = None
        spans: dict = {}
        calls = 0
        transcript: list = []

        def due(self, body, seen, now):
            return False

        def ask(self, body, seen, now, note=""):
            return None

    run(seconds=1.0, planner=_Bare(), name="test-bare-planner",
        verbose=False, watch=False, output_path=tmp_path / "clip.json")
