"""The jaws as three states, pinned.

This file used to test `squeeze_for`, a rule that read the current goal and
worked out what the plan wanted the jaws to do. That rule was wrong four times
and each rewrite lost a case the previous version handled, so it was replaced by
three explicit states. The cases below are the same failures, restated against
the thing that replaced it -- they are what the states have to get right, and
every one of them is a run that actually went wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigby_poc.gripper.decision.grip import GRIP_N, STATES, jaw_command
from rigby_poc.gripper.physics.model import make

# Pure units: one body build, no pipeline, no subprocess.
pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def body():
    return make(np.asarray([0.025, 0.025, 0.03]),
                np.asarray([0.0, 0.30, 0.76]), table_top=0.72)


def _limits(body):
    return body.model.jnt_range[body.model.joint("finger_left").id]


def test_open_drives_the_jaws_wide_with_no_grip(body):
    """The failure that started this: a grip that could not be released.

    Both pads read 3 N against the block for forty seconds while the planner
    asked three times for 86 mm, because `holding` was left in charge of the
    rule. A state cannot be overruled by what is being held.
    """
    _low, high = _limits(body)
    out, squeeze = jaw_command("open", body, np.zeros(6))
    assert out[4] == pytest.approx(high)
    assert out[5] == pytest.approx(high)
    assert squeeze == 0.0


def test_close_drives_them_shut_with_grip(body):
    low, _high = _limits(body)
    out, squeeze = jaw_command("close", body, np.zeros(6))
    assert out[4] == pytest.approx(low)
    assert squeeze == GRIP_N


def test_hold_leaves_the_fingers_exactly_where_they_are(body):
    """A carry must not need a decision every frame to remain a carry."""
    commanded = np.array([0.1, 0.2, 0.3, 0.4, 0.031, 0.031])
    out, squeeze = jaw_command("hold", body, commanded)
    assert out[4] == pytest.approx(0.031)
    assert out[5] == pytest.approx(0.031)
    assert squeeze == GRIP_N


def test_the_arm_is_never_touched(body):
    """Only the fingers. The arm belongs to the search.

    The old rule and the search could disagree about the same joints; this is
    the boundary that makes that impossible.
    """
    commanded = np.array([0.11, 0.22, 0.33, 0.44, 0.01, 0.01])
    for state in STATES:
        out, _ = jaw_command(state, body, commanded)
        assert np.allclose(out[:4], commanded[:4])


def test_a_state_does_not_depend_on_the_goal(body):
    """The whole point.

    Every earlier version inferred the jaws from whatever number was being
    driven, so an alignment goal that said nothing about the hand clamped it
    shut. jaw_command cannot see a goal at all.
    """
    out_a, sq_a = jaw_command("open", body, np.zeros(6))
    out_b, sq_b = jaw_command("open", body, np.zeros(6))
    assert np.allclose(out_a, out_b) and sq_a == sq_b


def test_an_unknown_state_holds_rather_than_flinging_open(body):
    """A typo must not drop what is being carried."""
    commanded = np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.02])
    out, squeeze = jaw_command("wibble", body, commanded)
    assert out[4] == pytest.approx(0.02)
    assert squeeze == GRIP_N
