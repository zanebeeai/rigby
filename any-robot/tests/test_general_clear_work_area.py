"""ClearWorkArea on a body: the world, the chain of sessions, the carried state, the observed area.

The clearance world puts every start slot inside the work area and every
cell on the enlarged platform with margin, and the support of a cube on
the table is the table, not the floor under it. On the jaw arm a
two-object clearance runs as two sessions on one clock, the second opened
on the state the first left (the first cube where it was placed, the arm
where it stood), and the root succeeds with the area observed clear by the
cameras; the oracle agrees from the final poses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rigby_core.skills import Belief, Interrupt, Verdict, execute
from rigby_core.skills.clearance import clear_work_area_library
from rigby_general.pipeline import ingest_robot
from rigby_general.sensing import load_policy
from rigby_general.skills.transfer_object import TransferObjectSession
from rigby_general.skills.clear_work_area import CELL_HALF_M, LAYOUT, LAYOUT_V1, LAYOUT_V2, ChainClock, ClearWorkAreaRuntime, build_clearance_world, final_object_poses, in_work_area, oracle_clear, predicates_for_clearance
from rigby_general.skills.transfer_object import support_fixture


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g10_corpus as g10  # noqa: E402


@pytest.fixture(scope="module")
def base():
    return g10.environment()


def test_world_slots_lie_in_the_work_area_and_cells_on_the_platform_with_margin(base):
    for count in (3, 5, 10):
        world = build_clearance_world(base, count)
        assert len(world.objects) == count and set(world.cells.values()) <= set(world.offsets)
        platform = next(f for f in world.environment.fixtures if f.name == "platform")
        for item in world.environment.objects:
            assert in_work_area(item.position_m)
            assert support_fixture(world.environment, item.position_m).name == "table"
        for cell in world.cells.values():
            goal = world.goal_for(cell)
            low, high = np.asarray(goal.region_minimum_m), np.asarray(goal.region_maximum_m)
            assert low[0] >= platform.position_m[0] - platform.size_m[0] + 0.02 and high[0] <= platform.position_m[0] + platform.size_m[0] - 0.02
            assert low[1] >= platform.position_m[1] - platform.size_m[1] + 0.02 and high[1] <= platform.position_m[1] + platform.size_m[1] - 0.02
            assert high[0] - low[0] == pytest.approx(2 * CELL_HALF_M)
    jittered = build_clearance_world(base, 2, draw={"objects": [{"translation_m": [0.01, -0.01, 0.0], "mass_multiplier": 1.1, "friction_multiplier": 0.9}, {"translation_m": [0.0, 0.0, 0.0]}]})
    plain = build_clearance_world(base, 2)
    assert jittered.environment.objects[0].position_m[0] == pytest.approx(plain.environment.objects[0].position_m[0] + 0.01)
    assert jittered.environment.objects[0].mass_kg == pytest.approx(plain.environment.objects[0].mass_kg * 1.1)


def test_two_object_clearance_runs_as_a_chain_on_one_clock(base):
    body = "zoo_jaw_arm"
    source = ROOT / "assets/general/zoo" / body / "robot.urdf"
    robot = ingest_robot(source, robot_id=body)
    world = build_clearance_world(base, 2)
    library = clear_work_area_library(world.objects, tuple(world.cells[o] for o in world.objects))
    interrupt = Interrupt()
    runtime = ClearWorkAreaRuntime(body=body, source=source, robot=robot, world=world, policy=load_policy(g10.G09 / "policy.json"), interrupt=interrupt, seed_label="test")
    tree = library.expand("clear_work_area", {"effector": robot.morphology.grasping_effectors[0].chain_id})
    record = execute(tree, library, runtime, predicates_for_clearance(world), clock=ChainClock(runtime), interrupt=interrupt, belief=Belief())
    poses = final_object_poses(runtime)
    assert record.verdict is Verdict.SUCCESS, (record.verdict, record.root.reason)
    assert [s.object_name for s in runtime.segments] == ["cube_01", "cube_02"]
    first, second = runtime.segments
    assert second.started_s == pytest.approx(first.ended_s) and second.ended_s > second.started_s
    assert record.ended_s == pytest.approx(runtime.clock_s)
    # the second session began on the first's final state: the first cube already on the platform
    placed = poses["cube_01"][:3]
    goal = world.goal_for(world.cells["cube_01"])
    assert np.all(placed >= np.asarray(goal.region_minimum_m)) and np.all(placed <= np.asarray(goal.region_maximum_m))
    oracle = oracle_clear(world, poses)
    assert oracle["root_success"] and oracle["placed"] == 2
    assert runtime.observations and runtime.observations[-1]["clear"] is True
    assert record.belief.get("work_area_clear") is True
    assert runtime.observations[-1]["in_cells"] == {"cube_01": True, "cube_02": True}, "the look decides the placements from the cameras"
    assert record.belief.get("placed:cube_01:cell_01") is True and record.belief.get("placed:cube_02:cell_02") is True


def test_layout_v2_clears_the_widest_jaw_and_lies_within_every_body_reach(base):
    """Twelve centimetres between neighbours against the long arm's 17.2 cm
    open jaw; every slot and cell within 95% of each enabled body's
    directional reach; nothing the robot stands on at rest."""

    assert LAYOUT is LAYOUT_V2 and LAYOUT_V2.pitch_m >= 0.12 > LAYOUT_V1.pitch_m
    positions = list(LAYOUT_V2.slots_m) + [(LAYOUT_V2.platform_centre_m[0] + dx, LAYOUT_V2.platform_centre_m[1] + dy) for dx, dy in LAYOUT_V2.cell_offsets_m]
    for i, a in enumerate(positions):
        for b in positions[i + 1:]:
            assert np.hypot(a[0] - b[0], a[1] - b[1]) >= LAYOUT_V2.pitch_m - 1e-9, (a, b)
    policy = load_policy(g10.G09 / "policy.json")
    world = build_clearance_world(base, 10)
    for body in ("zoo_dual_arm", "zoo_jaw_arm", "zoo_long_arm"):
        source = ROOT / "assets/general/zoo" / body / "robot.urdf"
        robot = ingest_robot(source, robot_id=body)
        session = TransferObjectSession.open(body, source, world.environment, world.goal_for("cell_01"), policy, object_name="cube_01", destination_fixture="platform",
                                             destination_offset_m=world.offsets["cell_01"], robot=robot)
        frame = session.sensing.frames[robot.morphology.grasping_effectors[0].chain_id]
        top = float(next(f for f in world.environment.fixtures if f.name == "platform").position_m[2] + LAYOUT_V2.platform_half_m[2])
        points = [np.array(o.position_m) for o in world.environment.objects] + [np.array([world.goal_for(c).region_minimum_m[0] + CELL_HALF_M, world.goal_for(c).region_minimum_m[1] + CELL_HALF_M, top + 0.015]) for c in world.cells.values()]
        for point in points:
            azimuth, elevation = frame.bearing_of(point)
            distance = float(np.linalg.norm(point - np.asarray(frame.origin)))
            assert distance <= 0.95 * float(frame.directional_reach(azimuth, elevation)) and distance >= float(frame.inner_reach(azimuth, elevation)) + 0.03, (body, point)
        # at rest nothing of the robot touches a cube
        data = mujoco.MjData(session.model)
        data.qpos[:] = session.current_qpos()
        mujoco.mj_forward(session.model, data)
        cubes = {g for g in range(session.model.ngeom) if "cube" in (mujoco.mj_id2name(session.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") or "scene_block" in (mujoco.mj_id2name(session.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")}
        fixtures = {g for g in range(session.model.ngeom) if "env_fixture" in (mujoco.mj_id2name(session.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")}
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 in cubes or contact.geom2 in cubes:
                other = contact.geom2 if contact.geom1 in cubes else contact.geom1
                assert other in fixtures or other in cubes, (body, mujoco.mj_id2name(session.model, mujoco.mjtObj.mjOBJ_GEOM, other))


def test_the_arm_guard_leaves_the_closure_own_fingers_to_the_closure(base):
    """Two fingers parted by grip joints alone are not the arm planner's to
    keep apart: closed on nothing they stand inside one another, and a guard
    that counted them refused every later path in the chain. Every other
    pair the morphology lists is still guarded."""

    from rigby_general.grounding import grounder
    from rigby_general.grounding.grounder import _collision_guard

    body = "zoo_jaw_arm"
    source = ROOT / "assets/general/zoo" / body / "robot.urdf"
    robot = ingest_robot(source, robot_id=body)
    model, manifest = robot.finalized.model, robot.manifest
    name_of = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    grip = {j for e in manifest.morphology.grasping_effectors for j in e.grip_joints}
    assert grip, "the jaw arm declares its grip joints"
    finger_bodies = {name_of(int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])) for j in grip}
    assert len(finger_bodies) == 2
    guarded = {tuple(sorted((name_of(a), name_of(b)))) for a, b in _collision_guard(manifest, model).pairs}
    assert tuple(sorted(finger_bodies)) not in guarded, "the closure's own pair is left to the closure"
    assert any(set(pair) & finger_bodies and not set(pair) <= finger_bodies for pair in guarded), "a finger against the rest of the arm is still guarded"
    grounder.GUARD_LEAVES_CLOSURE_PAIRS = False
    try:
        legacy = {tuple(sorted((name_of(a), name_of(b)))) for a, b in _collision_guard(manifest, model).pairs}
    finally:
        grounder.GUARD_LEAVES_CLOSURE_PAIRS = True
    assert tuple(sorted(finger_bodies)) in legacy and legacy - guarded == {tuple(sorted(finger_bodies))}, "the first guard counted exactly that one pair more"


def test_the_closure_joints_hold_their_limits_in_every_compiled_scene(base):
    """A lost hold snapped the jaw shut and the soft default limits let the
    fingers cross; the transfer's scenes now give every declared grip joint
    a limit constraint that holds, and a scene recompiled with the shutter
    keeps it. The toggle reproduces the soft limits for the comparison."""

    from rigby_general.contact import closure as closure_module
    from rigby_general.skills.transfer_object import with_shutter

    body = "zoo_jaw_arm"
    source = ROOT / "assets/general/zoo" / body / "robot.urdf"
    robot = ingest_robot(source, robot_id=body)
    world = build_clearance_world(base, 1)
    policy = load_policy(g10.G09 / "policy.json")
    session = TransferObjectSession.open(body, source, world.environment, world.goal_for("cell_01"), policy, object_name="cube_01", destination_fixture="platform", destination_offset_m=world.offsets["cell_01"], robot=robot)
    model = session.model
    grip = [j for e in robot.manifest.morphology.grasping_effectors for j in e.grip_joints]
    assert grip
    for name in grip:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert model.jnt_limited[joint] and model.jnt_solref[joint][0] == pytest.approx(closure_module.LIMIT_TIMECONST_S) and model.jnt_solref[joint][0] >= 2 * model.opt.timestep
        assert tuple(model.jnt_solimp[joint]) == pytest.approx(closure_module.LIMIT_SOLIMP)
    arm = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith("joint_")]
    assert arm and all(model.jnt_solref[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][0] == pytest.approx(0.02) for n in arm), "the arm's own joints keep the compiler's limits"
    closure_module.CLOSURE_LIMITS_HOLD = False
    try:
        soft = TransferObjectSession.open(body, source, world.environment, world.goal_for("cell_01"), policy, object_name="cube_01", destination_fixture="platform", destination_offset_m=world.offsets["cell_01"], robot=robot)
    finally:
        closure_module.CLOSURE_LIMITS_HOLD = True
    assert all(soft.model.jnt_solref[mujoco.mj_name2id(soft.model, mujoco.mjtObj.mjOBJ_JOINT, n)][0] == pytest.approx(0.02) for n in grip), "the toggle reproduces the soft limits"
