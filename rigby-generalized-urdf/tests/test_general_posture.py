"""Postures: a shape stated in counts and poles, resolved against one body.

The schema layer could say where an effector goes and how it was turned, and
nothing about what its members were doing -- so a five-digit hand that can
plainly hold a peace sign had no way to be asked for one. These pin the two
properties that make the answer general rather than hand-shaped: the posture
never names a member, and the same posture lands differently on every body or
refuses with a measurement.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_general.bake.enumerate import enumerate_bindings
from rigby_general.grounding import ground
from rigby_general.pipeline import ingest_robot
from rigby_general.planner.schema_planner import OfflineSchemaPlanner
from rigby_general.schema.inventory import (
    Requirement,
    afforded_entries,
    capabilities_of,
    load_inventory,
)
from rigby_general.schema.program import (
    Flexion,
    MotionSchemaProgramV1,
    PostureV1,
    Stative,
    assert_metric_free,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "general"


def _load(robot_id: str, family: str):
    """Ingest from the robot's own directory, so relative meshes resolve."""

    source = ASSETS / family / robot_id / "robot.urdf"
    if not source.is_file():
        pytest.skip(f"{robot_id} is not present in this checkout")
    cwd = os.getcwd()
    os.chdir(source.parent)
    try:
        return ingest_robot(Path("robot.urdf"), robot_id=robot_id)
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def inventory():
    return load_inventory()


def test_a_posture_carries_no_magnitude():
    """The whole IR stays metric-free with postures in it."""

    assert_metric_free(MotionSchemaProgramV1)


def test_a_posture_never_names_a_member():
    """Selection is a count and a measured side, not a digit's name."""

    posture = PostureV1(selected_count=2)
    dumped = posture.model_dump(mode="python")
    words = " ".join(str(value) for value in dumped.values()).lower()
    for name in ("index", "middle", "ring", "little", "thumb", "finger"):
        assert name not in words


def test_members_are_ordered_by_where_they_are_not_by_name():
    """The uHand's digits come out in anatomical order from geometry alone.

    Alphabetically these sort index, little, middle, ring. Any ordering that
    produced that would make "two fingers up" the index and the little finger,
    which is not a gesture anyone makes.
    """

    robot = _load("uhand2", "irl")
    effector = robot.morphology.effectors[0]
    opposed = effector.opposition_groups[0]
    assert opposed == (
        "index_distal",
        "middle_distal",
        "ring_distal",
        "little_distal",
    )
    assert opposed != tuple(sorted(opposed))


def test_a_hand_affords_postures_and_a_bare_tool_does_not():
    robot = _load("uhand2", "irl")
    assert Requirement.ARTICULATED_EFFECTOR in capabilities_of(robot.morphology)

    tool = _load("zoo_tool_arm", "zoo")
    assert Requirement.ARTICULATED_EFFECTOR not in capabilities_of(tool.morphology)


def test_the_peace_sign_extends_exactly_the_two_leading_members(inventory):
    """The gesture, measured: two digits straight and the rest curled.

    Which end of a joint's range straightens it is measured per member, because
    it is not derivable from the grip: on this hand the digits reach furthest at
    the lower limit while the grip also closes toward it.
    """

    robot = _load("uhand2", "irl")
    model = mujoco.MjModel.from_xml_string(robot.mjcf_xml)
    afforded = afforded_entries(inventory, robot.morphology)
    program = OfflineSchemaPlanner(inventory).plan(
        "throw up a peace sign", afforded=afforded
    )
    segment = program.segments[0]
    assert segment.motion_schema.stative is Stative.CONFIGURE
    assert segment.posture is not None
    assert segment.posture.selected_count == 2

    grounded = ground(program, robot.manifest, model, inventory)
    posture_frame = next(
        frame
        for track in grounded.program.tracks
        for frame in track.keyframes
        if frame.hard and any(name.endswith("_flex") for name in frame.joint_values)
    )
    degrees = {
        name: float(np.degrees(value))
        for name, value in posture_frame.joint_values.items()
        if name.endswith("_flex")
    }
    assert degrees["index_flex"] < 15.0
    assert degrees["middle_flex"] < 15.0
    for curled in ("ring_flex", "little_flex", "thumb_flex"):
        assert degrees[curled] > 75.0


def test_the_same_gesture_refuses_on_a_body_with_too_few_members(inventory):
    """A two-jaw gripper has nowhere to put a third finger, and says so."""

    robot = _load("zoo_jaw_arm", "zoo")
    model = mujoco.MjModel.from_xml_string(robot.mjcf_xml)
    afforded = afforded_entries(inventory, robot.morphology)
    program = OfflineSchemaPlanner(inventory).plan(
        "throw up a peace sign", afforded=afforded
    )
    with pytest.raises(Exception) as raised:
        ground(program, robot.manifest, model, inventory)
    assert "opposed members" in str(raised.value)


def test_the_library_bakes_one_posture_per_shape_the_body_can_hold(inventory):
    """How many postures a body has is read off the body, not configured."""

    hand = _load("uhand2", "irl")
    jaw = _load("zoo_jaw_arm", "zoo")

    def postures(robot) -> int:
        return sum(
            1
            for candidate in enumerate_bindings(inventory, robot.morphology)
            if candidate.entry_id == "configure_effector"
        )

    assert postures(hand) > postures(jaw) > 0


def test_a_posture_only_rides_on_the_schema_that_takes_one():
    """Every other schema leaves the effector's members alone."""

    entry = load_inventory().by_id("configure_effector")
    assert entry.takes_posture
    assert not load_inventory().by_id("hold_still").takes_posture


def test_open_and_closed_land_on_opposite_ends_of_the_same_travel(inventory):
    """The two poles have to actually differ, and by the member's whole range."""

    robot = _load("uhand2", "irl")
    model = mujoco.MjModel.from_xml_string(robot.mjcf_xml)
    from rigby_general.grounding.grounder import _posture_targets

    chain = robot.morphology.effectors[0].chain_id
    opened = _posture_targets(
        PostureV1(selected_count=0, remainder=Flexion.EXTENDED, opposing=Flexion.EXTENDED),
        robot.manifest,
        model,
        chain,
    )
    closed = _posture_targets(
        PostureV1(selected_count=0, remainder=Flexion.FLEXED, opposing=Flexion.FLEXED),
        robot.manifest,
        model,
        chain,
    )
    assert set(opened) == set(closed)
    for name in opened:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        span = abs(float(np.diff(model.jnt_range[joint])[0]))
        assert abs(opened[name] - closed[name]) > 0.8 * span
