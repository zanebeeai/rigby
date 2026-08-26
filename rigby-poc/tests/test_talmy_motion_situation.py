"""A command partitioned into Talmy's components, and the contour it specifies.

The property under test is not "the parser understands English". It is that a
command either resolves to a stated Figure/Path/Motion or resolves to nothing,
and that when it resolves, the contour it produces is a specification the object
can be measured against. A parser that guesses is worse than one that declines,
because a wrong specification silently becomes a wrong optimisation target.
"""

from __future__ import annotations

import numpy as np
import pytest

from rigby_poc.models import default_scene
from rigby_poc.talmy import (
    MOTION_VERBS,
    MotionSituation,
    TalmyError,
    contour_error,
    interpret,
    path_catalog,
    path_contour,
)

#: no compile, no corpus, no browser -- see docs/testing.md
pytestmark = pytest.mark.fast


def test_the_path_vocabulary_is_closed_and_well_formed() -> None:
    """Talmy's claim about PATH is that the set is closed (1975, p. 181).

    That is the property worth encoding: FIGURE and GROUND come from open sets
    of noun phrases, PATH does not. A catalog that grew a morpheme per prompt
    would have abandoned the idea it is named after.
    """
    catalog = path_catalog()
    paths = catalog["paths"]
    assert 5 <= len(paths) <= 24, "a closed set, not a growing list"
    for name, spec in paths.items():
        assert spec["motion"] in MOTION_VERBS
        assert spec["aliases"], f"{name} is unreachable from any command"
        times = [float(w["at"]) for w in spec["contour"]["waypoints"]]
        assert times == sorted(times)
        assert times[0] == 0.0 and times[-1] == 1.0


def test_commands_partition_into_figure_path_and_motion() -> None:
    scene = default_scene()
    cases = {
        "pick up box": ("block", "OFF", "MOVE", None, "GRASPED"),
        "hold the block": ("block", "AT", "BEL", None, "GRASPED"),
        "lower the block": ("block", "DOWN", "MOVE", None, None),
        "put the block on the ladder": ("block", "ONTO", "MOVE", "ladder", None),
        "move the block to the hurdle": ("block", "TO", "MOVE", "hurdle", None),
    }
    for text, (figure, path, motion, ground, manner) in cases.items():
        situation = interpret(text, scene)
        assert situation is not None, f"{text!r} did not resolve"
        assert situation.figure == figure
        assert situation.path == path
        assert situation.motion == motion
        assert situation.ground == ground
        assert situation.manner == manner


def test_a_located_state_uses_the_other_deep_verb() -> None:
    """BEL, not MOVE. Talmy treats a located state as the limiting case."""
    situation = interpret("hold the block", default_scene())
    assert situation is not None
    assert situation.motion == "BEL"
    assert situation.as_notation() == "N(block) V(BEL+GRASPED) Pl(AT)"


def test_an_unresolvable_command_returns_nothing_rather_than_a_guess() -> None:
    scene = default_scene()
    assert interpret("wave hello", scene) is None, "no path morpheme"
    assert interpret("pick up the anvil", scene) is None, "figure not in the scene"
    # ONTO requires a Ground and there is none named.
    assert interpret("put it on", scene) is None


def test_the_contour_of_a_pickup_rises_clear_of_its_support() -> None:
    scene = default_scene()
    situation = interpret("pick up box", scene)
    assert situation is not None
    contour = path_contour(situation, scene)
    start, end = contour.points[0], contour.points[-1]
    assert end[1] - start[1] == pytest.approx(0.12, abs=1e-9)
    assert np.allclose(end[[0, 2]], start[[0, 2]]), "a pickup does not translate sideways"
    # Tangents point along travel, which for OFF is world up.
    _point, tangent = contour.sample(0.5)
    assert tangent[1] > 0.9


def test_onto_resolves_through_a_declared_support_socket() -> None:
    """The bounding box is the wrong answer for anything that is not a plinth.

    Before this, "put the block on the ladder" targeted 2.14 m -- the top of a
    2.1 m tall bounding box -- rather than a rung the ladder actually declares.
    """
    scene = default_scene()
    situation = interpret("put the block on the ladder", scene)
    assert situation is not None
    contour = path_contour(situation, scene)
    ladder = scene.object_by_id("ladder")
    assert ladder is not None
    box_top = ladder.transform.translation.y + ladder.dimensions_m.y / 2
    assert contour.points[-1][1] < box_top, "resolved to the bounding box, not a socket"
    highest_rung = max(
        s.transform.translation.y for s in ladder.sockets if s.supports_body_weight
    )
    assert contour.points[-1][1] == pytest.approx(
        ladder.transform.translation.y + highest_rung + 0.04, abs=1e-6
    )


def test_a_site_has_position_and_no_direction_of_travel() -> None:
    scene = default_scene()
    situation = interpret("hold the block", scene)
    assert situation is not None
    contour = path_contour(situation, scene)
    _point, tangent = contour.sample(0.5)
    assert np.allclose(tangent, 0.0), "a located site has no travel direction"


def test_contour_error_scores_a_trajectory_against_its_specification() -> None:
    """The loss a primitive-fitting search would minimise.

    Checked against two synthetic trajectories rather than a compiled clip, so
    the assertion is about the metric and not about today's compiler.
    """
    from rigby_poc.models import ClipFrame, Transform, Vec3

    scene = default_scene()
    situation = interpret("pick up box", scene)
    assert situation is not None
    contour = path_contour(situation, scene)
    start = contour.points[0]

    def clip_from(displacements):
        return [
            ClipFrame(
                time_s=index / 30.0,
                bones={},
                objects={
                    "block": Transform(
                        translation=Vec3(
                            x=float(start[0] + d[0]),
                            y=float(start[1] + d[1]),
                            z=float(start[2] + d[2]),
                        )
                    )
                },
            )
            for index, d in enumerate(displacements)
        ]

    steps = 31
    following = clip_from(
        [contour.sample(i / (steps - 1))[0] - start for i in range(steps)]
    )
    stationary = clip_from([np.zeros(3) for _ in range(steps)])

    good = contour_error(contour, following)
    bad = contour_error(contour, stationary)

    assert good["contour_mean_error_m"] < 1e-6
    assert good["contour_endpoint_error_m"] < 1e-6
    # An object that never moved is exactly the specified displacement away.
    assert bad["contour_endpoint_error_m"] == pytest.approx(0.12, abs=1e-6)
    assert bad["contour_mean_error_m"] > good["contour_mean_error_m"]
    assert good["contour_specified_displacement_m"] == pytest.approx(0.12, abs=1e-9)
    assert bad["contour_actual_displacement_m"] == pytest.approx(0.0, abs=1e-9)


def test_an_unknown_figure_raises_rather_than_producing_a_contour() -> None:
    scene = default_scene()
    bogus = MotionSituation(figure="anvil", path="OFF", motion="MOVE")
    with pytest.raises(TalmyError):
        path_contour(bogus, scene)
