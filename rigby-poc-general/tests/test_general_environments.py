"""Worlds authored before the robot, and the refusals that only they can produce.

The derived grasp scene sizes its block from the gripper's aperture and places it
at a fraction of the measured reach. That makes one probe meaningful across nine
bodies, and it also means the block is reachable by construction: the probe can
never demonstrate a robot failing to reach something, because it never puts
anything out of reach.

These environments are authored in absolute metres and know nothing about which
robot will arrive, so "too far", "too wide" and "too close" become answerable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rigby_general.grounding.grounder import figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes import (
    admit_environment,
    admit_object,
    available_environments,
    build_environment_model,
    load_environment,
    object_qpos_address,
)


ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "assets" / "general" / "zoo"

SMALL = "zoo_compact_arm"   # 32 mm jaw, 245 mm reach
LARGE = "zoo_long_arm"      # 132 mm jaw, 1348 mm reach


@pytest.fixture(scope="module")
def arms():
    out = {}
    for robot_id in (SMALL, LARGE):
        robot = ingest_robot(ZOO / robot_id / "robot.urdf", robot_id=robot_id)
        effector = robot.morphology.grasping_effectors[0]
        chain = next(
            item
            for item in robot.morphology.chains
            if item.chain_id == effector.chain_id
        )
        frame = build_workspace_frame(
            robot.finalized.model,
            robot.morphology,
            chain,
            figure_site=figure_site_for(robot.manifest, effector.chain_id),
        )
        out[robot_id] = (robot, effector, frame)
    return out


def test_every_environment_loads_and_is_sealed_in_metres() -> None:
    names = available_environments()
    assert names, "no environments authored"
    for name in names:
        environment = load_environment(name)
        assert environment.environment_id == name
        assert environment.objects, f"{name} has nothing to manipulate"
        for item in environment.objects:
            assert item.mass_kg > 0.0
            assert item.span_m > 0.0


def test_an_environment_mentions_no_robot() -> None:
    """The whole claim. A world that adapts to the arm proves nothing about it."""

    directory = ROOT / "assets" / "general" / "environments"
    forbidden = {p.name for p in ZOO.iterdir() if p.is_dir()} | {"panda", "iiwa7"}
    for path in directory.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        for robot_id in forbidden:
            assert robot_id not in text, f"{path.name} names {robot_id}"


def test_the_same_object_is_out_of_reach_for_one_arm_and_not_another(arms) -> None:
    """The refusal a derived scene cannot produce."""

    pallet = load_environment("far_pallet")
    crate = pallet.objects[0]

    small_robot, small_effector, small_frame = arms[SMALL]
    large_robot, large_effector, large_frame = arms[LARGE]

    small = admit_object(small_robot.manifest, small_effector, small_frame, crate)
    large = admit_object(large_robot.manifest, large_effector, large_frame, crate)

    assert not small.admitted
    assert small.code == "object_too_wide"
    assert large.admitted, "the long arm should manage a metre and a 90 mm crate"


def test_a_long_arm_can_be_refused_for_being_too_close(arms) -> None:
    """The reach shell's near surface, in the one place it decides an outcome.

    An arm cannot fold its effector onto its own shoulder. The desk bench sits
    inside that hole for the largest arm here, so the biggest robot in the fleet
    is the one that cannot use the smallest world.
    """

    bench = load_environment("desk_bench")
    robot, effector, frame = arms[LARGE]
    admission = admit_object(robot.manifest, effector, frame, bench.objects[0])

    assert not admission.admitted
    assert admission.code == "object_inside_reach_hole"
    # Checked on the object's own bearing, not on the level one: the bench sits
    # below the shoulder, and the near surface is a different radius down there.
    azimuth, elevation = frame.bearing_of(
        __import__("numpy").asarray(bench.objects[0].position_m, dtype=float)
    )
    assert admission.distance_m < frame.inner_reach(azimuth, elevation)


def test_the_small_arm_reaches_the_small_world(arms) -> None:
    bench = load_environment("desk_bench")
    robot, effector, frame = arms[SMALL]
    admission = admit_object(robot.manifest, effector, frame, bench.objects[0])
    assert admission.admitted, admission.reason


def test_the_robot_moves_into_the_world_and_the_world_stays_put(arms) -> None:
    """Compiling must place the objects where the file says, to the millimetre."""

    import numpy as np

    table = load_environment("work_table")
    robot, _, _ = arms[LARGE]
    model, _ = build_environment_model(robot.manifest, robot.mjcf_xml, table)

    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for item in table.objects:
        address = object_qpos_address(model, item.name)
        placed = np.asarray(data.qpos[address : address + 3], dtype=float)
        assert placed == pytest.approx(np.asarray(item.position_m), abs=1e-6), item.name


def test_admission_covers_every_object_in_the_environment(arms) -> None:
    table = load_environment("work_table")
    robot, effector, frame = arms[LARGE]
    rows = admit_environment(robot.manifest, effector, frame, table)
    assert len(rows) == len(table.objects)
    assert {row.object_name for row in rows} == {item.name for item in table.objects}
