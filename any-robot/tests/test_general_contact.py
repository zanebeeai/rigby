"""Contact: the scene, the closure loop, and the gates that keep a grasp honest.

These tests pin what is actually working, which is not all of it. The scene, the
force-controlled closure and the gates behave correctly; reliable grasping across
every gripper in the zoo does not yet, and the tests say which is which rather
than testing only the parts that pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rigby_general.contact import ClosureController, GripState, attempt_grasp
from rigby_general.errors import RigbyGeneralError
from rigby_general.grounding.grounder import figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes import build_grasp_scene


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"

# The one gripper that currently completes a certified pick. Kept as a named
# constant so the day another joins it, this file needs one line changed rather
# than a rewrite.
CERTIFIED_GRASPER = "zoo_compact_arm"


def setup_robot(robot_id: str):
    robot = ingest_robot(ZOO_ROOT / robot_id / "robot.urdf", robot_id=robot_id)
    effector = robot.morphology.grasping_effectors[0]
    chain = next(
        item for item in robot.morphology.chains if item.chain_id == effector.chain_id
    )
    frame = build_workspace_frame(
        robot.finalized.model,
        robot.morphology,
        chain,
        figure_site=figure_site_for(robot.manifest, effector.chain_id),
    )
    scene = build_grasp_scene(robot.manifest, robot.mjcf_xml, effector, frame)
    return robot, effector, frame, scene


@pytest.fixture(scope="module")
def jaw():
    return setup_robot("zoo_hand_arm")


# --------------------------------------------------------------------------
# The scene is measured, not authored
# --------------------------------------------------------------------------


def test_the_block_is_sized_from_the_measured_aperture(jaw) -> None:
    _, effector, _, scene = jaw
    assert effector.max_aperture_m is not None
    assert 0.3 < (2 * scene.block_half_extent_m) / effector.max_aperture_m < 0.8


def test_a_bigger_gripper_gets_a_bigger_block() -> None:
    """A scene in absolute metres is the same mistake as a motion in them."""

    _, small_effector, _, small = setup_robot("zoo_compact_arm")
    _, large_effector, _, large = setup_robot("zoo_long_arm")

    assert large.block_half_extent_m > small.block_half_extent_m * 2
    ratio_small = 2 * small.block_half_extent_m / small_effector.max_aperture_m
    ratio_large = 2 * large.block_half_extent_m / large_effector.max_aperture_m
    assert pytest.approx(ratio_small, abs=1e-6) == ratio_large


def test_the_block_never_sits_below_the_robot(jaw) -> None:
    _, _, frame, scene = jaw
    assert scene.block_position_m[2] > 0.0


def test_the_block_mass_respects_the_measured_payload(jaw) -> None:
    robot, _, _, scene = jaw
    assert scene.block_mass_kg <= robot.morphology.scale.payload_kg


def test_a_robot_without_a_gripper_cannot_be_given_a_grasp_scene() -> None:
    robot = ingest_robot(ZOO_ROOT / "zoo_tool_arm" / "robot.urdf", robot_id="zoo_tool_arm")
    effector = robot.morphology.effectors[0]
    chain = robot.morphology.chains[0]
    frame = build_workspace_frame(
        robot.finalized.model,
        robot.morphology,
        chain,
        figure_site=figure_site_for(robot.manifest, effector.chain_id),
    )
    with pytest.raises(RigbyGeneralError, match="no gripper"):
        build_grasp_scene(robot.manifest, robot.mjcf_xml, effector, frame)


# --------------------------------------------------------------------------
# Closure commands force, not position
# --------------------------------------------------------------------------


def test_closure_commands_a_squeeze_toward_the_closed_end(jaw) -> None:
    robot, effector, _, scene = jaw
    closure = ClosureController(
        scene.model,
        robot.manifest,
        effector,
        object_geoms=frozenset({"scene_block_geom"}),
    )
    closure.state = GripState.CLOSING
    commands = closure.force_commands()

    assert set(commands) == set(effector.grip_joints)
    assert all(abs(value) > 0.0 for value in commands.values())


def test_releasing_reverses_the_command(jaw) -> None:
    robot, effector, _, scene = jaw
    closure = ClosureController(
        scene.model,
        robot.manifest,
        effector,
        object_geoms=frozenset({"scene_block_geom"}),
    )
    closure.state = GripState.CLOSING
    squeeze = closure.force_commands()
    closure.state = GripState.RELEASING
    release = closure.force_commands()

    for name in squeeze:
        assert np.sign(squeeze[name]) == -np.sign(release[name])


def test_opposition_needs_both_sides(jaw) -> None:
    """Contact on one side is a nudge; contact on both is a grip."""

    robot, effector, _, scene = jaw
    closure = ClosureController(
        scene.model,
        robot.manifest,
        effector,
        object_geoms=frozenset({"scene_block_geom"}),
    )
    groups = effector.opposition_groups
    assert len(groups) == 2
    assert not closure._opposition_satisfied(tuple(groups[0]))
    assert closure._opposition_satisfied(tuple(groups[0]) + tuple(groups[1]))


# --------------------------------------------------------------------------
# The gates
# --------------------------------------------------------------------------


def test_a_certified_grasp_holds_a_real_object() -> None:
    """No weld, no attachment: the block is held by measured contact or not held.

    Currently one gripper in the zoo gets all the way through. The others fail on
    identifiable gates -- see ``scripts/grasp_report.py`` -- and that is recorded
    rather than smoothed over.
    """

    robot, effector, frame, scene = setup_robot(CERTIFIED_GRASPER)
    result = attempt_grasp(robot.manifest, scene, effector, frame)

    assert result.certified, [violation.code for violation in result.violations]
    assert result.opposition_achieved
    assert result.lift_height_m > 1.6 * scene.block_half_extent_m
    assert scene.model.neq == 0, "a grasp proven with an equality constraint proves nothing"


def test_the_gates_actually_reject_a_failed_grasp() -> None:
    """The suite would be worthless if every attempt certified.

    Two of the five grippers currently fail, and the gates name which claim
    failed rather than reporting a generic error. The robot named here is one of
    them; the compact and jaw arms both used to be and now certify, which is the
    direction this is supposed to move in.
    """

    robot, effector, frame, scene = setup_robot("zoo_hand_arm")
    result = attempt_grasp(robot.manifest, scene, effector, frame)

    assert not result.certified
    assert result.failed_gate in {
        "object_not_lifted",
        "object_dropped",
        "excessive_penetration",
        "grasp_not_achieved",
    }
    assert result.violations[0].measured != result.violations[0].limit


def test_a_grasp_result_carries_its_measurements() -> None:
    robot, effector, frame, scene = setup_robot(CERTIFIED_GRASPER)
    result = attempt_grasp(robot.manifest, scene, effector, frame)

    assert result.qpos.shape[0] > 100
    assert result.times_s[-1] > 0.0
    assert result.peak_force_n > 0.0
