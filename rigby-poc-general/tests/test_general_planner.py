"""Recognising motion in plain language, and refusing what is not there.

The planner fills Talmy's slots and nothing else. What these tests check is that
each slot is filled from the right cues, that the result stays body-neutral, and
that the two ways of saying no -- I do not know that phrase, and your robot
cannot do that -- stay distinguishable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rigby_general.errors import GeneralFailureCode, RigbyGeneralError
from rigby_general.pipeline import ingest_robot
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.schema.inventory import afforded_entries, load_inventory
from rigby_general.schema.program import Remove


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"


@pytest.fixture(scope="module")
def inventory():
    return load_inventory()


@pytest.fixture(scope="module")
def planner(inventory):
    return OfflineSchemaPlanner(inventory)


@pytest.fixture(scope="module")
def robots():
    return {
        robot_id: ingest_robot(ZOO_ROOT / robot_id / "robot.urdf", robot_id=robot_id)
        for robot_id in ("zoo_jaw_arm", "zoo_dual_arm", "zoo_tool_arm")
    }


def afforded_for(inventory, robots, robot_id: str):
    return afforded_entries(inventory, robots[robot_id].morphology)


def plan(planner, inventory, robots, prompt: str, robot_id: str = "zoo_jaw_arm"):
    return planner.plan(prompt, afforded=afforded_for(inventory, robots, robot_id))


# --------------------------------------------------------------------------
# The Path slot
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("reach out in front of you", "path:to.point.neutral.straight"),
        ("sweep across the workspace", "path:via.line.neutral.straight"),
        ("wave at me", "path:via.point.neutral.oscillating"),
        ("trace a circle", "path:via.axis.neutral.circular"),
        ("withdraw and come back", "path:from.point.neutral.straight"),
        ("lower it down toward the table", "path:to.surface.neutral.straight"),
        ("lift it off the table", "path:from.surface.neutral.straight"),
        ("hold still", "stative:hold"),
    ],
)
def test_surface_forms_map_onto_path_schemas(
    planner, inventory, robots, prompt: str, expected: str
) -> None:
    program = plan(planner, inventory, robots, prompt)
    assert program.segments[0].motion_schema.canonical_key == expected


def test_a_more_specific_cue_beats_a_general_one(planner, inventory, robots) -> None:
    """"Wave" contains no motion word a general pattern would catch first.

    Ordering matters: ``move`` and ``go`` appear in almost every request, so if
    they were tested first every gesture in the language would become a reach.
    """

    program = plan(planner, inventory, robots, "move your hand in a wave")
    assert program.segments[0].motion_schema.contour.value == "oscillating"


# --------------------------------------------------------------------------
# The Region slot
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("reach out as far as you can", Remove.DISTAL),
        ("reach way out there", Remove.DISTAL),
        ("reach out halfway", Remove.MEDIAL),
        ("reach just a little", Remove.PROXIMAL),
        ("reach right up against it", Remove.ADJACENT),
    ],
)
def test_distance_words_map_onto_degrees_of_remove(
    planner, inventory, robots, prompt: str, expected: Remove
) -> None:
    program = plan(planner, inventory, robots, prompt)
    assert program.segments[0].region.remove is expected


def test_an_unmarked_reach_is_a_comfortable_one(planner, inventory, robots) -> None:
    """Language leaves distance unstated far more often than not."""

    program = plan(planner, inventory, robots, "reach out")
    assert program.segments[0].region.remove is Remove.MEDIAL


# --------------------------------------------------------------------------
# The Manner slot
# --------------------------------------------------------------------------


def test_pace_words_become_ordinals(planner, inventory, robots) -> None:
    quick = plan(planner, inventory, robots, "reach out quickly")
    slow = plan(planner, inventory, robots, "reach out slowly")
    assert quick.segments[0].manner.speed > 0
    assert slow.segments[0].manner.speed < 0


def test_size_words_become_amplitude(planner, inventory, robots) -> None:
    big = plan(planner, inventory, robots, "wave with a big sweeping motion")
    small = plan(planner, inventory, robots, "wave with a small tight motion")
    assert big.segments[0].manner.amplitude > small.segments[0].manner.amplitude


def test_a_stated_count_is_taken_exactly(planner, inventory, robots) -> None:
    """Cardinality is the one thing language does commit to precisely."""

    assert plan(planner, inventory, robots, "wave three times").segments[
        0
    ].manner.repetition_count == 3
    assert plan(planner, inventory, robots, "wave 7 times").segments[
        0
    ].manner.repetition_count == 7
    assert plan(planner, inventory, robots, "wave twice").segments[
        0
    ].manner.repetition_count == 2


def test_an_unstated_count_stays_an_ordinal(planner, inventory, robots) -> None:
    assert plan(planner, inventory, robots, "wave at me").segments[
        0
    ].manner.repetition_count is None


# --------------------------------------------------------------------------
# Segment chains
# --------------------------------------------------------------------------


def test_sequence_markers_split_a_prompt_into_segments(
    planner, inventory, robots
) -> None:
    """Tversky and Lee: a route is a chain, and its joints are marked."""

    program = plan(
        planner, inventory, robots, "reach out as far as you can and then come back"
    )
    assert len(program.segments) == 2
    assert program.links
    assert program.links[0].relation.value == "sequence"


def test_a_three_step_request_produces_three_segments(
    planner, inventory, robots
) -> None:
    program = plan(
        planner,
        inventory,
        robots,
        "reach out, then wave twice, then withdraw",
    )
    assert len(program.segments) == 3


# --------------------------------------------------------------------------
# Body neutrality
# --------------------------------------------------------------------------


def test_one_prompt_reads_identically_on_every_body(
    planner, inventory, robots
) -> None:
    """Requirement ``schema_invariance``, measured at the planner."""

    prompt = "reach out as far as you can and then come back"
    hashes = {
        plan(planner, inventory, robots, prompt, robot_id).role_normalized_hash()
        for robot_id in robots
    }
    assert len(hashes) == 1


def test_a_plan_carries_no_metric_value(planner, inventory, robots) -> None:
    from rigby_general.schema.program import MotionSchemaProgramV1, assert_metric_free

    assert_metric_free(MotionSchemaProgramV1)
    program = plan(planner, inventory, robots, "reach out quickly as far as you can")
    assert "0." not in program.model_dump_json().replace('"0.0"', "")


# --------------------------------------------------------------------------
# Refusing
# --------------------------------------------------------------------------


def test_unknown_wording_is_refused(planner, inventory, robots) -> None:
    with pytest.raises(RigbyGeneralError) as caught:
        plan(planner, inventory, robots, "make me a sandwich")
    assert caught.value.code is GeneralFailureCode.UNAFFORDED_SCHEMA


def test_an_unafforded_schema_names_itself(planner, inventory, robots) -> None:
    """A one-armed robot asked for a handover should hear *why*, not "no".

    This is the reason the two refusals are kept apart: the caller can tell an
    unrecognised phrase from a recognised one this body cannot perform, and only
    the second is worth suggesting a different robot for.
    """

    with pytest.raises(RigbyGeneralError) as caught:
        plan(planner, inventory, robots, "hand it over to the other arm", "zoo_jaw_arm")
    assert "hand_across" in str(caught.value)


def test_the_same_request_is_accepted_by_a_two_armed_robot(
    planner, inventory, robots
) -> None:
    program = plan(
        planner, inventory, robots, "hand it over to the other arm", "zoo_dual_arm"
    )
    assert program.segments[0].motion_schema.canonical_key.startswith("path:to.point")


def test_an_empty_prompt_is_refused(planner, inventory, robots) -> None:
    with pytest.raises(RigbyGeneralError, match="empty"):
        plan(planner, inventory, robots, "   ")


def test_the_trace_records_which_cue_fired(planner, inventory, robots) -> None:
    """A wrong reading should be traceable to the cue that produced it."""

    plan(planner, inventory, robots, "sweep slowly across the bench")
    trace = planner.last_trace
    assert trace and trace[0].entry_id == "traverse_line"
    assert trace[0].manner.get("speed", 0) < 0
