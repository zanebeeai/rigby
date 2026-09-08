"""Looking a few moves ahead, and refusing the ones that hit something.

Every case here is the run of 2026-09-08 restated as a test. That run carried
the block to rim height at the bin's OUTER edge -- two millimetres outside the
opening -- and swung the base sideways, crushing the block against the wall at
87 N until the contact prised the jaws open and the block fell out. The greedy
search scored every one of those probes as an improvement, correctly: sideways
through a wall is closer to the bin centre.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigby_poc.gripper.decision.goals import NumericTarget
from rigby_poc.gripper.decision.greedy import pursue
from rigby_poc.gripper.decision.lookahead import (
    BEAM, DEPTH, TOUCH_M, Foresight, foresee,
)
from rigby_poc.gripper.physics.model import make
from rigby_poc.gripper.sensing.gripper_camera import Senses, sense

pytestmark = pytest.mark.medium

#: What the cameras believed the block was, in the run that crashed.
BELIEVED = np.asarray([0.036, 0.036, 0.041])


@pytest.fixture(scope="module")
def scene():
    body = make(np.asarray([0.025, 0.025, 0.03]),
                np.asarray([0.0, 0.30, 0.76]), table_top=0.72)
    eyes = Senses()
    seen = sense(body, eyes, np.asarray(body.q()), 0.0, 0.0, 0.72)
    return body, seen, Foresight(body)


def _overlap(sight, centre, half=BELIEVED):
    gaps = np.abs(np.asarray(centre) - sight.box_pos) - (sight.box_half + half)
    return float(-gaps.max(axis=1).min())


def test_the_crash_spot_is_refused(scene):
    """The exact place the block was destroyed, and it is not clear.

    Block at x=0.377 at rim height. The bin's opening spans x 0.225 to 0.375,
    so those two millimetres put it over the WALL. Every number the old planner
    could read said the carry was going fine.
    """
    _body, _seen, sight = scene
    assert _overlap(sight, [0.377, 0.153, 1.019]) > TOUCH_M


def test_the_route_that_works_is_allowed(scene):
    """Over the opening and above the rim has to stay available.

    A collision check that refuses the correct approach as well as the wrong
    one has not made the machine safer, it has made it unable to do the task.
    """
    _body, _seen, sight = scene
    assert _overlap(sight, [0.300, 0.020, 1.060]) < 0.0


def test_the_bin_walls_are_solid_to_what_is_carried(scene):
    """Dead centre of a wall is the least ambiguous case there is."""
    _body, _seen, sight = scene
    assert _overlap(sight, [0.387, 0.020, 0.940]) > 0.04


def test_open_bench_is_clear(scene):
    _body, _seen, sight = scene
    assert _overlap(sight, [0.0, 0.30, 0.76]) < -0.1


def test_the_route_reaches_further_than_one_move_can(scene):
    """Three moves ahead beats one, on the goal's own terms.

    Measured on the bench pose: one move gets hand_z_m to 0.1076, the route
    gets it to 0.0028.
    """
    body, seen, sight = scene
    target = NumericTarget(metric="hand_z_m", value=1.0, set_at_s=0.0,
                           using=("move",))
    spans = target.spans(body, seen)
    _pose_g, greedy_error, _g = pursue(body, seen, target, spans)
    _pose_f, _first_error, note = foresee(body, seen, target, spans,
                                          sight=sight)
    assert note["route_reaches"] < greedy_error
    assert note["clear"] is True


def test_it_commands_one_move_and_not_the_destination(scene):
    """What gets returned is the FIRST step, which is the whole contract.

    Returning the route's endpoint threw away the route: the arm slews to a
    setpoint in a straight line, and a straight line to the end of a three-move
    route is not that route -- the routes contain reversals, one of them
    literally tilt_down@+0.60 then tilt_up@-0.60. Measured, the endpoint was a
    1.135 rad jump where one move was 0.733, and nothing along that line had
    been collision-checked.
    """
    body, seen, sight = scene
    start = np.asarray(body.q())
    for metric, value in (("palm_to_object_m", 0.045),
                          ("pointing_at_object", 0.9),
                          ("hand_x_m", 0.30)):
        target = NumericTarget(metric=metric, value=value, set_at_s=0.0,
                               using=("move",))
        spans = target.spans(body, seen)
        greedy_pose, _e, _g = pursue(body, seen, target, spans,
                                     start_from=start)
        look_pose, _f, _n = foresee(body, seen, target, spans,
                                    start_from=start, sight=sight)
        one_move = float(np.abs(greedy_pose[:4] - start[:4]).max())
        commanded = float(np.abs(look_pose[:4] - start[:4]).max())
        # One named move at one amplitude, from the same vocabulary greedy
        # draws from -- so it cannot be dramatically further than greedy's.
        assert commanded <= one_move * 1.5 + 1e-9, (metric, commanded, one_move)


def test_it_reports_the_route_it_chose(scene):
    """A route nobody can read is a route nobody can review."""
    body, seen, sight = scene
    target = NumericTarget(metric="palm_to_object_m", value=0.045,
                           set_at_s=0.0, using=("move",))
    spans = target.spans(body, seen)
    _pose, _error, note = foresee(body, seen, target, spans, sight=sight)
    assert 1 <= len(note["route"]) <= DEPTH
    assert all("/" in step and "@" in step for step in note["route"])
    assert note["part"] in ("base", "segment_1", "segment_2", "segment_3", "-")


def test_it_still_moves_when_already_touching_something(scene):
    """Refusing to move because you are against a wall is how you stay there.

    If every route collides, the least-bad one is taken and the note says so.
    Without this the arm freezes the moment it brushes anything.
    """
    body, seen, sight = scene
    target = NumericTarget(metric="hand_z_m", value=0.60, set_at_s=0.0,
                           using=("move",))
    spans = target.spans(body, seen)
    pose, _error, note = foresee(body, seen, target, spans, sight=sight)
    assert pose is not None
    assert "clear" in note and "deepest_overlap_mm" in note


def test_the_search_is_the_size_it_claims(scene):
    """Depth and beam are what the cost was measured against."""
    assert DEPTH == 3
    assert BEAM == 6
