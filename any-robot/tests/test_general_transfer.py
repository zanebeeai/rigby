"""A transfer is proven phase by phase, on the physics clock, by an evaluator
the controller cannot read.

The grasp probe stopped at the lift. G06 asks for the whole family -- approach,
oppose, lift, hold, carry, set down, let go -- on one fixed fixture chosen
before any controller ran, with the outcome decided by an independent
placement evaluator. What is pinned here: the evaluator's predicates on a
hand-authored world where the answers are known; the scene policy that
refuses a welded, actuated or non-colliding object; the restart seeds and the
finger standoff that let a straight-up rest pose reach a low bench; the
facing objective that brings a hand vertical when solved with the point
rather than after it; the registered fixture's own integrity; and the three
kinds of outcome on real bodies -- a certified transfer, a typed runtime
failure, and a typed refusal before anything moves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from rigby_core.simulation.recording import PhysicsRecorder, replay_physics

from rigby_general.contact.grasp import _hand_facing, _scene_rest_qpos
from rigby_general.contact.placement import PlacementEvaluator, PlacementGoal
from rigby_general.contact.transfer import (
    LIFT_REQUIRED_FRACTION,
    PHASES,
    PLACE_STANDOFF_M,
    TransferResult,
    attempt_transfer,
    collision_policy,
    grasp_standoff_m,
    restart_seeds,
    transfer_scene_from_environment,
)
from rigby_general.contracts import SiteSemantic
from rigby_general.grounding import ik
from rigby_general.grounding.grounder import _collision_guard, figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.scenes.environment import EnvironmentV1


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"
PROTOCOL = Path(__file__).resolve().parents[1] / "assets" / "general" / "research-protocols" / "g06-transfer-v1"
CERTIFIED_BODY = "zoo_jaw_arm"
FAILING_BODY = "zoo_hand_arm"
INFEASIBLE_BODY = "zoo_compact_arm"


# --------------------------------------------------------------------------
# The independent placement evaluator, on a world where the answers are known
# --------------------------------------------------------------------------

PLATFORM_TOP_M = 0.11
CUBE_HALF_M = 0.015
TOY_WORLD = f"""
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="env_fixture_platform" pos="0 0 0.1">
      <geom name="env_fixture_platform_geom" type="box" size="0.05 0.05 0.01"/>
    </body>
    <body name="scene_block" pos="0 0 {PLATFORM_TOP_M + CUBE_HALF_M}">
      <freejoint name="scene_block_free"/>
      <geom name="scene_block_geom" type="box" size="{CUBE_HALF_M} {CUBE_HALF_M} {CUBE_HALF_M}" mass="0.01"/>
    </body>
    <body name="finger" pos="0 0 0.3" mocap="true">
      <geom name="finger_geom" type="box" size="0.005 0.005 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def toy_goal(margin_m: float = 0.02, dwell_s: float = 2.0) -> PlacementGoal:
    return PlacementGoal(
        region_minimum_m=(-margin_m, -margin_m, PLATFORM_TOP_M - 0.001),
        region_maximum_m=(margin_m, margin_m, PLATFORM_TOP_M + 0.1),
        dwell_s=dwell_s,
    )


def toy_world() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(TOY_WORLD)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def run_evaluator(model, data, evaluator, seconds: float):
    steps = int(round(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        evaluator.assess(data)


def test_a_goal_needs_a_region_with_extent_and_positive_limits() -> None:
    with pytest.raises(ValueError):
        PlacementGoal(region_minimum_m=(0, 0, 0), region_maximum_m=(1, 0, 1))
    with pytest.raises(ValueError):
        PlacementGoal(region_minimum_m=(0, 0, 0), region_maximum_m=(1, 1, 1), dwell_s=0.0)


def test_scaling_a_goal_scales_the_region_and_keeps_the_stillness() -> None:
    goal = toy_goal()
    scaled = goal.scaled(2.0)
    assert scaled.region_minimum_m == pytest.approx(tuple(2 * v for v in goal.region_minimum_m))
    assert scaled.region_maximum_m == pytest.approx(tuple(2 * v for v in goal.region_maximum_m))
    assert scaled.dwell_s == goal.dwell_s
    assert scaled.maximum_linear_speed_mps == goal.maximum_linear_speed_mps


def test_a_cube_resting_inside_the_region_succeeds_once_the_dwell_is_measured() -> None:
    model, data = toy_world()
    evaluator = PlacementEvaluator(model, toy_goal(dwell_s=2.0), object_geom="scene_block_geom", object_joint="scene_block_free")
    run_evaluator(model, data, evaluator, 1.9)
    assert evaluator.last is not None and evaluator.last.whole_geometry_inside and evaluator.last.released
    assert not evaluator.success, "two seconds of stillness were required and 1.9 were shown"
    run_evaluator(model, data, evaluator, 0.3)
    assert evaluator.success
    assert evaluator.dwell_s >= 2.0
    assert evaluator.last.maximum_robot_normal_force_n == 0.0, "the platform is the world, not the robot"


def test_a_cube_outside_the_region_never_qualifies() -> None:
    model, data = toy_world()
    tight = PlacementGoal(region_minimum_m=(0.05, -0.02, PLATFORM_TOP_M - 0.001), region_maximum_m=(0.1, 0.02, PLATFORM_TOP_M + 0.1))
    evaluator = PlacementEvaluator(model, tight, object_geom="scene_block_geom", object_joint="scene_block_free")
    run_evaluator(model, data, evaluator, 2.5)
    assert not evaluator.success
    assert evaluator.last is not None and not evaluator.last.whole_geometry_inside
    assert evaluator.dwell_s == 0.0


def test_containment_is_of_the_whole_rotated_geometry() -> None:
    model, data = toy_world()
    address = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scene_block_free")])
    half_turn = np.pi / 4
    data.qpos[address + 3: address + 7] = [np.cos(half_turn / 2), 0.0, 0.0, np.sin(half_turn / 2)]
    mujoco.mj_forward(model, data)
    diagonal = CUBE_HALF_M * np.sqrt(2.0)
    snug = PlacementEvaluator(model, toy_goal(margin_m=diagonal - 0.001), object_geom="scene_block_geom", object_joint="scene_block_free")
    roomy = PlacementEvaluator(model, toy_goal(margin_m=diagonal + 0.001), object_geom="scene_block_geom", object_joint="scene_block_free")
    assert not snug.assess(data).whole_geometry_inside, "a cube turned 45 degrees is wider than its side"
    assert roomy.assess(data).whole_geometry_inside


def test_a_cube_a_robot_geom_is_touching_is_not_released() -> None:
    model, data = toy_world()
    finger_bottom_on_cube_top = PLATFORM_TOP_M + 2 * CUBE_HALF_M + 0.02 - 0.0005
    data.mocap_pos[0] = [0.0, 0.0, finger_bottom_on_cube_top]
    mujoco.mj_forward(model, data)
    evaluator = PlacementEvaluator(model, toy_goal(), object_geom="scene_block_geom", object_joint="scene_block_free")
    run_evaluator(model, data, evaluator, 0.5)
    assert evaluator.last is not None
    assert not evaluator.last.released
    assert evaluator.last.whole_geometry_inside
    assert not evaluator.last.conditions_met
    assert evaluator.dwell_s == 0.0


def test_a_gap_in_observation_restarts_the_dwell() -> None:
    model, data = toy_world()
    evaluator = PlacementEvaluator(model, toy_goal(dwell_s=2.0), object_geom="scene_block_geom", object_joint="scene_block_free")
    run_evaluator(model, data, evaluator, 1.0)
    assert evaluator.dwell_s == pytest.approx(1.0, abs=2 * model.opt.timestep)
    for _ in range(int(round(0.5 / model.opt.timestep))):
        mujoco.mj_step(model, data)  # unobserved
    sample = evaluator.assess(data)
    assert sample.conditions_met
    assert evaluator.dwell_s == 0.0, "the evaluator vouches only for what it watched"


# --------------------------------------------------------------------------
# The registered fixture
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registration() -> dict:
    return json.loads((PROTOCOL / "registration.json").read_bytes())


@pytest.fixture(scope="module")
def fixed_world() -> tuple[EnvironmentV1, PlacementGoal]:
    env = EnvironmentV1.model_validate_json((PROTOCOL / "environment.json").read_bytes())
    raw = json.loads((PROTOCOL / "goal.json").read_bytes())
    goal = PlacementGoal(
        region_minimum_m=tuple(raw["region_minimum_m"]), region_maximum_m=tuple(raw["region_maximum_m"]), dwell_s=raw["dwell_s"],
        maximum_linear_speed_mps=raw["maximum_linear_speed_mps"], maximum_angular_speed_radps=raw["maximum_angular_speed_radps"],
    )
    return env, goal


def test_the_registered_files_match_their_registration(registration) -> None:
    for name, digest in registration["files"].items():
        assert hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest() == digest, name
    files = json.dumps(registration["files"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert hashlib.sha256(files.encode("utf-8")).hexdigest() == registration["registration_sha256"]


def test_the_feasibility_map_names_a_checkable_reason_for_every_class(registration) -> None:
    feasibility = {row["zoo_id"]: row for row in json.loads((PROTOCOL / "feasibility-map.json").read_bytes())["bodies"]}
    compact = feasibility[INFEASIBLE_BODY]
    assert compact["class"] == "infeasible"
    assert "necessary condition violated: aperture_spans_cube" in compact["reasons"]
    assert compact["conditions"]["aperture_spans_cube"]["aperture_m"] < compact["conditions"]["aperture_spans_cube"]["span_m"] + 0.004
    for zoo_id in registration["feasible_bodies"]:
        row = feasibility[zoo_id]
        assert row["class"] == "feasible"
        assert row["witness_configurations"]["source"] and row["witness_configurations"]["destination"], zoo_id
        assert all(row["conditions"][name]["ok"] for name in ("aperture_spans_cube", "payload_carries_cube", "points_inside_measured_envelope", "fixture_clear_of_body_at_rest"))
    for zoo_id in registration["unsupported_by_structure"]:
        assert feasibility[zoo_id]["reasons"] == ["no grasping effector"]
    roster = json.loads((PROTOCOL / "roster.json").read_bytes())
    draws = [body["seeds"] for body in roster["bodies"]]
    assert all(d == draws[0] for d in draws), "the perturbation draws are identical across bodies"
    assert len(draws[0]) == roster["draw"]["trials_per_body"] == 100


# --------------------------------------------------------------------------
# Real bodies in the fixed world
# --------------------------------------------------------------------------


def setup_body(zoo_id: str, env: EnvironmentV1, goal: PlacementGoal):
    source = ZOO_ROOT / zoo_id / "robot.urdf"
    robot = ingest_robot(source, robot_id=zoo_id)
    effector = robot.morphology.grasping_effectors[0]
    chain = next(c for c in robot.morphology.chains if c.chain_id == effector.chain_id)
    frame = build_workspace_frame(robot.finalized.model, robot.morphology, chain, figure_site=figure_site_for(robot.manifest, effector.chain_id))
    scene = transfer_scene_from_environment(robot.manifest, robot.mjcf_xml, env, object_name="cube", destination_fixture="platform", goal=goal, asset_root=source.parent)
    return robot, effector, frame, scene


@pytest.fixture(scope="module")
def jaw(fixed_world):
    return setup_body(CERTIFIED_BODY, *fixed_world)


@pytest.fixture(scope="module")
def certified(jaw) -> tuple[TransferResult, PhysicsRecorder]:
    robot, effector, frame, scene = jaw
    recorder = PhysicsRecorder(scene.model)
    return attempt_transfer(robot.manifest, scene, effector, frame, recorder=recorder), recorder


def test_the_authored_world_has_nothing_that_could_fake_a_hold(jaw) -> None:
    _, _, _, scene = jaw
    policy = collision_policy(scene.model)
    assert policy["equality_constraints"] == 0
    assert policy["object_actuators"] == 0
    assert policy["contact_exclusions_with_object"] == []
    assert policy["object_geoms_collidable"]
    assert policy["object_state_writes"] == 0


def test_a_welded_object_is_refused_before_anything_moves(jaw) -> None:
    robot, effector, frame, scene = jaw
    xml = scene.scene.xml.replace("</mujoco>", '<equality><weld body1="scene_block" body2="world"/></equality></mujoco>')
    welded = replace(scene, scene=replace(scene.scene, model=mujoco.MjModel.from_xml_string(xml), xml=xml))
    assert collision_policy(welded.model)["equality_constraints"] == 1
    result = attempt_transfer(robot.manifest, welded, effector, frame)
    assert not result.certified
    assert result.violations[0].code == "hidden_weld"
    assert result.phases == () and result.duration_s == 0.0 and len(result.times_s) == 0


def test_restart_seeds_begin_at_rest_and_end_at_mid_range(jaw) -> None:
    robot, effector, frame, scene = jaw
    model = scene.model
    joints = ik.chain_joint_names(model, frame.figure_site, exclude=frozenset(effector.grip_joints))
    rest = _scene_rest_qpos(model, robot.manifest)
    seeds = restart_seeds(model, frame, joints, rest, np.asarray(scene.scene.block_position_m))
    labels = [label for label, _ in seeds]
    assert labels[0] == "rest" and labels[-1] == "midpoint"
    assert np.array_equal(seeds[0][1], rest)
    addresses = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in joints]
    ranges = [model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joints]
    assert seeds[-1][1][addresses] == pytest.approx([0.5 * (lo + hi) for lo, hi in ranges])
    if "turned" in labels:
        turned = seeds[labels.index("turned")][1]
        changed = np.flatnonzero(turned != rest)
        assert len(changed) == 1 and changed[0] in addresses, "turning to face the target spends exactly one joint"


def test_the_grasp_standoff_keeps_the_fingers_out_of_the_bench(jaw) -> None:
    """The witness configuration the map recorded puts the grasp point on the
    cube with the standoff applied; at that configuration no finger is inside
    the bench."""

    robot, effector, frame, scene = jaw
    model = scene.model
    site = next(s.name for s in robot.manifest.morphology.sites if s.semantic is SiteSemantic.GRASP_POINT and s.name.startswith(effector.chain_id))
    standoff = grasp_standoff_m(model, robot.manifest, effector, site, scene.scene.block_half_extent_m)
    assert standoff >= PLACE_STANDOFF_M
    feasibility = {row["zoo_id"]: row for row in json.loads((PROTOCOL / "feasibility-map.json").read_bytes())["bodies"]}
    witness = feasibility[CERTIFIED_BODY]
    assert witness["conditions"]["kinematic_witness"]["source"]["standoff_m"] == pytest.approx(standoff)
    data = mujoco.MjData(model)
    data.qpos[:] = witness["witness_configurations"]["source"][-1]
    mujoco.mj_forward(model, data)
    member_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in effector.member_bodies}
    for index in range(data.ncon):
        contact = data.contact[index]
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""}
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if bodies & member_bodies and any(n.startswith("env_fixture_") for n in names):
            assert float(contact.dist) > -0.001, names


def test_facing_solved_with_the_point_brings_the_hand_vertical(jaw) -> None:
    """Spent in the null space after the point was placed, the facing went
    unmet by more than a right angle on this five-axis arm; solved together
    with the point it is met exactly, at the same position."""

    robot, effector, frame, scene = jaw
    model = scene.model
    site = next(s.name for s in robot.manifest.morphology.sites if s.semantic is SiteSemantic.GRASP_POINT and s.name.startswith(effector.chain_id))
    joints = ik.chain_joint_names(model, frame.figure_site, exclude=frozenset(effector.grip_joints))
    rest = _scene_rest_qpos(model, robot.manifest)
    facing = _hand_facing(robot.manifest, effector)
    assert facing is not None
    down = -np.asarray(frame.up, dtype=float)
    standoff = grasp_standoff_m(model, robot.manifest, effector, site, scene.scene.block_half_extent_m)
    at = np.asarray(scene.scene.block_position_m) + np.array([0.0, 0.0, standoff])
    targets = np.asarray([at + np.array([0.0, 0.0, 0.10]), at])
    guard = _collision_guard(robot.manifest, model)

    def angle_from_vertical(qpos: np.ndarray) -> float:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        rotation = np.array(data.site_xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)]).reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip((rotation @ facing) @ down, -1.0, 1.0))))

    with_facing = ik.solve_site_path(model, site, joints, targets, seed_qpos=rest, guard=guard, facing=(facing, down))
    without = ik.solve_site_path(model, site, joints, targets, seed_qpos=rest, guard=guard)
    assert with_facing.worst_residual_m <= ik.DEFAULT_TOLERANCE_M
    assert angle_from_vertical(with_facing.qpos[-1]) < 10.0
    assert angle_from_vertical(without.qpos[-1]) > 45.0


def test_the_turn_holds_the_point_and_brings_the_hand_vertical_before_the_descent(jaw) -> None:
    """The approach reaches the hover however the straight line from home
    leads, 60 degrees off vertical on this arm; the turn then swings the
    hand to vertical with the grasp point held, a few degrees per row, so
    no finger sweeps through the cube below. Solved in one step, the joint
    blend between the two configurations carried the fingers of the long
    arm fifty millimetres through the cube in its normalized world."""

    from rigby_general.contact.transfer import _joint_path, _path, _grasp_offset_m

    robot, effector, frame, scene = jaw
    model = scene.model
    site = next(s.name for s in robot.manifest.morphology.sites if s.semantic is SiteSemantic.GRASP_POINT and s.name.startswith(effector.chain_id))
    joints = ik.chain_joint_names(model, frame.figure_site, exclude=frozenset(effector.grip_joints))
    rest = _scene_rest_qpos(model, robot.manifest)
    facing = _hand_facing(robot.manifest, effector)
    down = -np.asarray(frame.up, dtype=float)
    data = mujoco.MjData(model)
    data.qpos[:] = rest
    mujoco.mj_kinematics(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    home = np.array(data.site_xpos[site_id])
    points, spans = _path(scene, frame, home, _grasp_offset_m(robot.manifest, effector), grasp_standoff_m(model, robot.manifest, effector, site, scene.scene.block_half_extent_m))
    assert spans["turn"] == (1, 2) and np.array_equal(points[1], points[2])
    path, marks, _ = _joint_path(model, site, joints, points, restart_seeds(model, frame, joints, rest, points[3]), _collision_guard(robot.manifest, model), 6, facing=(facing, down))

    def observe(qpos):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        rotation = np.array(data.site_xmat[site_id]).reshape(3, 3)
        return np.array(data.site_xpos[site_id]), float(np.degrees(np.arccos(np.clip((rotation @ facing) @ down, -1.0, 1.0))))

    start, end = marks[spans["turn"][0]], marks[spans["turn"][1]]
    positions, angles = zip(*(observe(path[row]) for row in range(start, end + 1)))
    assert angles[0] > 30.0, "the approach alone leaves the hand well off vertical"
    assert angles[-1] < 5.0
    assert all(later <= earlier + 1.0 for earlier, later in zip(angles, angles[1:])), angles
    assert max(float(np.linalg.norm(p - points[1])) for p in positions) <= ik.DEFAULT_TOLERANCE_M
    assert max(np.diff(angles) * -1.0) < 25.0, "no single row turns more than a fraction of the whole"


def test_a_transfer_is_certified_phase_by_phase_on_the_physics_clock(certified) -> None:
    result, _ = certified
    assert result.certified, [v.code for v in result.violations]
    assert tuple(p.name for p in result.phases) == PHASES
    assert all(p.end_s >= p.start_s for p in result.phases)
    assert all(later.start_s == earlier.end_s for earlier, later in zip(result.phases, result.phases[1:]))
    assert result.opposition_achieved and result.placement_success and result.placed_inside and result.released
    assert result.lift_height_m >= LIFT_REQUIRED_FRACTION * 0.03
    assert result.hold_s >= 2.0
    assert result.placement_dwell_s >= 2.0
    assert result.max_penetration_m <= 0.004
    assert result.path_seed in {"rest", "turned", "midpoint"}
    assert result.failed_gate is None
    steps = np.diff(result.times_s)
    assert steps == pytest.approx(0.002, abs=1e-9), "phase timing runs on the model's own clock"
    assert result.duration_s == pytest.approx(result.times_s[-1])


def test_the_recorded_controls_replay_to_the_recorded_states(certified, jaw) -> None:
    result, recorder = certified
    _, _, _, scene = jaw
    record = recorder.finish()
    assert len(record.arrays["time_s"]) == len(result.times_s)
    assert replay_physics(scene.model, record)["agrees"]


def test_a_transfer_that_fails_at_runtime_is_typed_and_measured(fixed_world) -> None:
    robot, effector, frame, scene = setup_body(FAILING_BODY, *fixed_world)
    result = attempt_transfer(robot.manifest, scene, effector, frame)
    assert not result.certified
    assert result.failed_gate in {"grasp_not_achieved", "object_not_lifted", "hold_not_sustained", "object_not_carried", "not_transported", "not_released", "placement_unstable", "excessive_penetration"}
    assert result.phases, "the attempt ran; it did not refuse"
    assert len(result.times_s) == len(result.qpos) == len(result.object_position_m)
    for violation in result.violations:
        assert isinstance(violation.measured, float) and isinstance(violation.limit, float)
        assert violation.detail


def test_an_infeasible_body_is_refused_typed_before_anything_moves(fixed_world) -> None:
    robot, effector, frame, scene = setup_body(INFEASIBLE_BODY, *fixed_world)
    result = attempt_transfer(robot.manifest, scene, effector, frame)
    assert not result.certified
    assert result.failed_gate == "unreachable_path"
    assert result.phases == () and len(result.times_s) == 0
    assert result.violations[0].measured > ik.DEFAULT_TOLERANCE_M
