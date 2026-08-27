"""The hand as a jointed mechanism, and the seating rule it depends on.

Two properties. That the articulated hand is a well-formed mechanism with one
actuator per joint and geometry taken from this rig rather than assumed. And
that a digit loaded before it has travelled is reported as obstructed rather
than counted as seated -- the distinction whose absence froze the thumb.
"""

from __future__ import annotations

import pytest

from rigby_poc.models import Hand

#: no compile, no corpus, no browser -- see docs/testing.md
pytestmark = pytest.mark.fast


def test_the_hand_is_measured_from_this_rig_not_assumed() -> None:
    """A physics hand that differs in size from the drawn one is unreproducible."""
    from rigby_poc.articulated_hand import CHAINS, measure

    geometry = measure(Hand.RIGHT)
    assert set(geometry.lengths) == set(CHAINS)
    for finger, lengths in geometry.lengths.items():
        assert len(lengths) >= 3, finger
        for length in lengths:
            # Human phalanges, loosely. A zero here means a bone was missing and
            # the chain silently collapsed to a point.
            assert 0.015 < length < 0.070, f"{finger} has a {length:.3f} m segment"
    # The knuckles must be spread across the palm, not stacked.
    across = {round(geometry.origins[f][2], 3) for f in ("index", "middle", "ring", "little")}
    assert len(across) == 4, "finger roots are not distinct"


def test_every_joint_has_exactly_one_actuator() -> None:
    """An unactuated joint is a finger nothing can close."""
    mujoco = pytest.importorskip("mujoco")
    from rigby_poc.articulated_hand import actuator_xml, hand_xml, joint_names

    xml = (
        f"<mujoco><worldbody>{hand_xml(Hand.RIGHT)}</worldbody>"
        f"<actuator>{actuator_xml(Hand.RIGHT)}</actuator></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    names = joint_names(Hand.RIGHT)
    assert len(names) == 15, "five digits, three joints each"
    assert model.nu == len(names)
    for name in names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act") >= 0
    # Bounded force is what makes this a grip rather than a command.
    for index in range(model.nu):
        low, high = model.actuator_forcerange[index]
        assert low < 0.0 < high, "an unbounded actuator cannot report grip force"


def test_curl_maps_monotonically_onto_joint_targets() -> None:
    from rigby_poc.articulated_hand import targets_for_curls

    closed = targets_for_curls(Hand.RIGHT, {d: 1.0 for d in ("thumb", "index", "middle", "ring", "little")})
    half = targets_for_curls(Hand.RIGHT, {d: 0.5 for d in ("thumb", "index", "middle", "ring", "little")})
    opened = targets_for_curls(Hand.RIGHT, {})
    for joint, angle in closed.items():
        assert opened[joint] == 0.0
        assert 0.0 < half[joint] < angle


def test_a_digit_loaded_before_it_travels_is_obstructed_not_seated() -> None:
    """The distinction that unfroze the thumb.

    Seating on any load made "reports force" and "has grasped" the same test.
    They are not: the thumb reached the closure window already touching the
    block, met the seat force at its opening curl, and froze -- its three bones
    were byte-identical from t=0.00 to t=3.38 while the index travelled 3 to 62
    degrees. On screen, four fingers closing around an object the thumb never
    reached.
    """
    from rigby_poc.force_closure import _MIN_TRAVEL_BEFORE_SEAT, _START_CURL

    assert _MIN_TRAVEL_BEFORE_SEAT > 0.0, (
        "with no travel requirement, an obstructed digit is indistinguishable "
        "from a seated one"
    )
    # The threshold has to leave real closing room below the FIST curls, or a
    # digit that legitimately seats early would be misreported as obstructed.
    from rigby_poc.models import HandShape
    from rigby_poc.primitives import HAND_SHAPES

    for stem, curl in HAND_SHAPES[HandShape.FIST].curls.items():
        assert curl - _START_CURL > _MIN_TRAVEL_BEFORE_SEAT, (
            f"{stem} closes less than the travel required to count as seated"
        )
