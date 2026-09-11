"""Super primitives: a grouped action whose every step is judged by a sensor.

The property worth testing is not that ``grab`` produces nice angles. It is that
a step can only pass on evidence a sensor actually produced -- a load that
existed, or a target the head could actually see. A hand that forms a perfect
shape around nothing must fail, because that is the failure this repository
shipped for months.

``unverified`` is tested as a distinct outcome rather than folded into pass or
fail. It is the honest answer when no sensor could have observed a step, and
collapsing it either way is what makes a gate lie.
"""

from __future__ import annotations

import pytest

from rigby_poc.models import ContactEvent, Hand, Vec3
from rigby_poc.super_primitives import (
    SENSOR_KINDS,
    SuperPrimitiveError,
    catalog,
    evaluate,
    expand,
    names,
    resolve,
    spec_for,
)

#: no compile, no corpus, no browser -- see docs/testing.md
pytestmark = pytest.mark.fast


def _contact(t: float, digit: str, force: float) -> ContactEvent:
    return ContactEvent(
        time_s=t, hand=Hand.RIGHT, object_id="block", digit=digit,
        position=Vec3(x=0.0, y=1.05, z=0.29), normal_force_n=force,
    )


def test_the_catalog_is_coherent_and_every_step_explains_itself() -> None:
    document = catalog()["super_primitives"]
    assert set(names()) == set(document)
    for name, spec in document.items():
        times = [float(s["at"]) for s in spec["steps"]]
        assert times == sorted(times) and times[0] == 0.0 and times[-1] == 1.0
        for step in spec["steps"]:
            assert step["sensor"]["kind"] in SENSOR_KINDS
            assert len(step["note"]) > 30, f"{name}.{step['label']} is unexplained"


def test_commands_resolve_to_a_super_primitive_or_to_nothing() -> None:
    assert resolve("pick up box") == "grab"
    assert resolve("pinch the block") == "pinch"
    assert resolve("let go of it") == "release"
    assert resolve("wave hello") is None


def test_a_super_primitive_expands_through_the_movement_vocabulary() -> None:
    """It cannot express a pose the vocabulary could not, so it cannot leave ROM."""
    for name in ("grab", "pinch"):
        rotations = expand(name, Hand.RIGHT, 1.0)
        assert len(rotations) == 14, "three segments per digit, minus the shared bases"
        assert all(bone.startswith("right") for bone in rotations)
    assert expand("grab", Hand.LEFT, 0.5), "both hands resolve"


def test_a_grasp_with_no_load_anywhere_fails_at_the_thumb() -> None:
    """The shape may be perfect; without a load it is not a grasp.

    This is the exact case the pipeline used to report as success.
    """
    verdict = evaluate(
        "grab", contacts=[], frames=[], duration_s=4.0,
        object_position=Vec3(x=0.0, y=1.05, z=0.29),
    )
    assert verdict.passed is False
    assert verdict.first_failure == "thumb_anchor"


def test_the_thumb_must_be_loaded_before_the_fingers_close() -> None:
    """The ordering the grasp depends on, asserted as an ordering.

    Fingers reaching a free object before the thumb push it out of the hand, so
    the same set of contacts in the wrong order is a different outcome.
    """
    duration = 4.0
    steps = {s["label"]: float(s["at"]) * duration for s in spec_for("grab")["steps"]}
    in_order = [
        _contact(steps["thumb_anchor"] - 0.1, "thumb", 1.2),
        # A real contact persists and keeps re-reporting; the anchor is still
        # loaded while the fingers close, which is what makes it an anchor.
        _contact(steps["close_fingers"] - 0.2, "thumb", 1.3),
        _contact(steps["close_fingers"] - 0.1, "index", 0.9),
        _contact(steps["close_fingers"] - 0.1, "middle", 0.9),
        _contact(steps["secure"] - 0.1, "thumb", 1.5),
        _contact(steps["secure"] - 0.1, "index", 1.1),
        _contact(steps["secure"] - 0.1, "middle", 1.0),
    ]
    good = evaluate("grab", contacts=in_order, frames=[], duration_s=duration)
    assert good.first_failure is None
    assert good.passed

    # Same contacts, thumb arriving only after the fingers have closed.
    late_thumb = [
        _contact(steps["close_fingers"] - 0.1, "index", 0.9),
        _contact(steps["close_fingers"] - 0.1, "middle", 0.9),
        _contact(steps["secure"] - 0.1, "thumb", 1.5),
        _contact(steps["secure"] - 0.1, "index", 1.1),
        _contact(steps["secure"] - 0.1, "middle", 1.0),
    ]
    bad = evaluate("grab", contacts=late_thumb, frames=[], duration_s=duration)
    assert bad.first_failure == "thumb_anchor", "late thumb must not pass"


def test_opposition_is_required_not_merely_several_loaded_digits() -> None:
    duration = 4.0
    steps = {s["label"]: float(s["at"]) * duration for s in spec_for("grab")["steps"]}
    same_side = [
        _contact(steps["thumb_anchor"] - 0.1, "thumb", 1.2),
        _contact(steps["close_fingers"] - 0.2, "thumb", 1.3),
        _contact(steps["close_fingers"] - 0.1, "index", 0.9),
        _contact(steps["close_fingers"] - 0.1, "middle", 0.9),
        # secure window: fingers only, no thumb -- loaded but not opposed
        _contact(steps["secure"] - 0.1, "index", 1.1),
        _contact(steps["secure"] - 0.1, "middle", 1.0),
        _contact(steps["secure"] - 0.1, "ring", 1.0),
    ]
    verdict = evaluate("grab", contacts=same_side, frames=[], duration_s=duration)
    assert verdict.first_failure == "secure"
    secure = next(s for s in verdict.steps if s.label == "secure")
    assert secure.measured["opposed"] is False


def test_a_pinch_needs_both_pads_and_a_poke_is_not_a_pinch() -> None:
    duration = 3.0
    steps = {s["label"]: float(s["at"]) * duration for s in spec_for("pinch")["steps"]}
    one_pad = [
        _contact(steps["pad_contact"] - 0.1, "index", 0.9),
        _contact(steps["secure"] - 0.1, "index", 0.9),
    ]
    verdict = evaluate("pinch", contacts=one_pad, frames=[], duration_s=duration)
    assert verdict.first_failure == "pad_contact"
    assert "thumb" in verdict.steps[2].detail


def test_release_passes_on_the_absence_of_force() -> None:
    """The inverse criterion. Letting go is verified by nothing being loaded."""
    duration = 2.0
    steps = {s["label"]: float(s["at"]) * duration for s in spec_for("release")["steps"]}
    let_go = [_contact(0.05, "thumb", 1.0)]
    verdict = evaluate("release", contacts=let_go, frames=[], duration_s=duration)
    assert verdict.first_failure is None, "held, then clear"
    assert verdict.passed

    still_holding = [_contact(0.05, "thumb", 1.0), _contact(steps["clear"] - 0.1, "thumb", 0.9)]
    stuck = evaluate("release", contacts=still_holding, frames=[], duration_s=duration)
    assert stuck.first_failure == "clear"


def test_a_step_no_sensor_could_observe_is_unverified_not_passed() -> None:
    verdict = evaluate(
        "grab", contacts=[], frames=[], duration_s=4.0, object_position=None
    )
    positioning = verdict.steps[0]
    assert positioning.status == "unverified"
    # Vision with no target to watch is also unverified, never a silent pass.
    vision = next(s for s in verdict.steps if s.sensor == "vision")
    assert vision.status == "unverified"
    assert "no target" in vision.detail


def test_an_unknown_super_primitive_raises() -> None:
    with pytest.raises(SuperPrimitiveError):
        spec_for("teleport")
    with pytest.raises(SuperPrimitiveError):
        expand("teleport", Hand.RIGHT, 1.0)
