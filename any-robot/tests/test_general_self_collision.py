"""A body's own links are an obstacle the reference has to respect.

Three things are pinned here, and they were all broken at once on the same
motion. Ingest excluded every non-adjacent link pair from the self-collision
gate, so the gate could not fire on any body. Grounding solved waypoints with
joint limits as the only constraint, so a legal key could put a palm inside its
own forearm. And the failure then surfaced under the wrong names -- a saturated
motor and a finger shoved past its slide limit -- instead of as the collision
it was. The fixture program below is the actual generated dual-arm reach/return
from before the repair; it is data, not a per-robot branch in the source.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
from rigby_core.contracts import MotionProgramV2
from rigby_core.motion.compiler import compile_motion_program

from rigby_general.errors import GeneralFailureCode, GroundingError
from rigby_general.gates.certify import GateCode, GatePolicy, evaluate_gates, simulate
from rigby_general.grounding import ik
from rigby_general.grounding.grounder import (
    _SELF_CLEARANCE_FRACTION,
    _collision_guard,
    ground,
    self_clearance_m,
)
from rigby_general.ingest.integrity import _separable, check_rest_contacts
from rigby_general.morphology.graph import physics_may_collide
from rigby_general.morphology.measure import neutral_qpos
from rigby_general.pipeline import _structural_overlaps, ingest_robot
from rigby_general.planner import OfflineSchemaPlanner
from rigby_general.schema.inventory import afforded_entries, load_inventory


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"
ZOO_IDS = sorted(p.parent.name for p in ZOO_ROOT.glob("*/robot.urdf"))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "g05_composition"
PROMPT = "reach out as far as you can and then come back"


@pytest.fixture(scope="module")
def zoo() -> dict:
    return {
        robot_id: ingest_robot(ZOO_ROOT / robot_id / "robot.urdf", robot_id=robot_id)
        for robot_id in ZOO_IDS
    }


@pytest.fixture(scope="module")
def inventory():
    return load_inventory()


def _bimanual(zoo: dict):
    """The one public body with two arms on a shared torso, found by
    structure: two grasping chains, not a name."""
    for robot in zoo.values():
        if len(robot.morphology.grasping_effectors) >= 2 and len(robot.morphology.chains) >= 2:
            return robot
    pytest.skip("no bimanual body in the public zoo")


def _body_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


# -- ingest: what the gate is allowed to ignore --------------------------------


def test_exclusions_name_only_pairs_no_joint_can_separate(zoo) -> None:
    """Every excluded pair is inseparable; every separable pair stays gated."""

    for robot in zoo.values():
        model = robot.finalized.model
        rest = neutral_qpos(model)
        excluded = {tuple(sorted(p)) for p in robot.manifest.adjacent_collision_exclusions}
        for first, second in excluded:
            assert not _separable(model, rest, _body_id(model, first), _body_id(model, second)), (
                f"{robot.manifest.rig_id}: {first}/{second} can be pulled apart but is excluded"
            )
        guarded = {tuple(sorted(p)) for p in robot.morphology.self_collision_pairs}
        assert guarded - excluded, "the gate would have nothing left to check"


def test_an_excluded_pair_is_an_authored_overlap(zoo) -> None:
    """What ingest excludes is a fact about the upload -- two hulls authored
    inside one another at every pose the joints between them allow -- and it
    is the recomputed set, nothing wider. Before the repair every body
    excluded its complete non-adjacent set, most of which never meet."""

    for robot in zoo.values():
        model, morphology = robot.finalized.model, robot.morphology
        excluded = {tuple(sorted(p)) for p in robot.manifest.adjacent_collision_exclusions}
        _, recorded = check_rest_contacts(model, morphology)
        authored = {
            tuple(sorted(note.subject.split("|", 1)))
            for note in recorded
            if "|" in note.subject
        }
        structural = {tuple(sorted(p)) for p in _structural_overlaps(model)}
        assert excluded == authored | structural, robot.manifest.rig_id
        non_adjacent = {tuple(sorted(p)) for p in morphology.self_collision_pairs}
        assert len(excluded) < len(non_adjacent) // 2, "the gate must cover most pairs"


def test_pairs_physics_never_reports_are_not_guarded(zoo) -> None:
    """Two links welded to the same parent, or a link and the weld it hangs
    from, never produce a contact in MuJoCo; guarding them would push the
    solver away from geometry that cannot move apart."""

    for robot in zoo.values():
        model = robot.finalized.model
        for first, second in robot.morphology.self_collision_pairs:
            a, b = _body_id(model, first), _body_id(model, second)
            assert physics_may_collide(model, a, b), (robot.manifest.rig_id, first, second)
            assert int(model.body_weldid[a]) != int(model.body_weldid[b])


# -- certification: the gate fires on the motion that used to pass it ----------


def test_gate_reports_the_folded_wrist_as_a_collision(zoo) -> None:
    robot = _bimanual(zoo)
    model, manifest = robot.finalized.model, robot.manifest
    program = MotionProgramV2.model_validate_json(
        (FIXTURE / "dual_arm_program_before_guard.json").read_bytes()
    )
    assert program.rig_id == manifest.rig_id
    guard = _collision_guard(manifest, model)
    assert guard is not None
    # Static truth first: the reference itself puts links inside one another.
    deepest = 0.0
    for track in program.tracks:
        for key in track.keyframes:
            qpos = np.asarray(manifest.rest_qpos, dtype=float)
            for name, value in key.joint_values.items():
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qpos[int(model.jnt_qposadr[joint])] = value
            for _, _, depth in ik.penetrations(model, guard, qpos):
                deepest = max(deepest, depth)
    assert deepest > 0.02, "the frozen program should bury a link tens of millimetres deep"

    # Then the physical gate, over the first phase only to keep the test quick:
    # the fold begins inside the return leg, well before its end.
    trajectory = compile_motion_program(program, model, manifest, sample_hz=60)
    cut = int(np.searchsorted(trajectory.times_s, 22.0))
    short = replace(
        trajectory,
        times_s=trajectory.times_s[:cut],
        qpos=trajectory.qpos[:cut],
        qvel=trajectory.qvel[:cut],
        qacc=trajectory.qacc[:cut],
    )
    trace = simulate(model, manifest, short, site_name=program.tracks[0].target)
    codes = {v.code for v in evaluate_gates(model, manifest, trace, GatePolicy())}
    assert GateCode.SELF_COLLISION in codes
    assert trace.unexpected_contacts, "the collision has to be named, not inferred from torque"


# -- grounding: the solver keeps the body out of itself -----------------------


def test_clearance_is_never_below_the_solver_tolerance() -> None:
    assert self_clearance_m(0.0) == 2.0 * ik.DEFAULT_TOLERANCE_M
    assert self_clearance_m(0.1) == 2.0 * ik.DEFAULT_TOLERANCE_M
    assert self_clearance_m(10.0) == pytest.approx(_SELF_CLEARANCE_FRACTION * 10.0)
    # Strictly inside the certification gate's own tracking tolerance on every
    # public body: the gate, not the preference, decides whether a motion
    # stayed clear.
    assert _SELF_CLEARANCE_FRACTION < GatePolicy().tracking_error_fraction


def test_guard_covers_every_gated_pair(zoo) -> None:
    for robot in zoo.values():
        model, manifest = robot.finalized.model, robot.manifest
        guard = _collision_guard(manifest, model)
        assert guard is not None
        excluded = {tuple(sorted(p)) for p in manifest.adjacent_collision_exclusions}
        expected = {
            tuple(sorted((_body_id(model, a), _body_id(model, b))))
            for a, b in robot.morphology.self_collision_pairs
            if tuple(sorted((a, b))) not in excluded
        }
        assert set(guard.pairs) == expected
        assert guard.clearance_m == pytest.approx(
            self_clearance_m(manifest.morphology.scale.reach_radius_m)
        )


def test_canonical_reach_and_return_grounds_clear_of_self_contact(zoo, inventory) -> None:
    """The same prompt, every body: no key and no span penetrates, and the
    bimanual body's return no longer folds its wrist onto its forearm."""

    planner = OfflineSchemaPlanner(inventory)
    for robot in zoo.values():
        model, manifest = robot.finalized.model, robot.manifest
        program = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
        grounded = ground(program, manifest, model, inventory)
        guard = _collision_guard(manifest, model)
        trajectory = compile_motion_program(grounded.program, model, manifest, sample_hz=30)
        for qpos in trajectory.qpos:
            assert ik.penetrations(model, guard, qpos) == (), manifest.rig_id


def test_guard_routes_round_the_body_or_refuses_with_the_pair_named() -> None:
    """A planar three-link arm with a post fixed to its own base.

    Three cases the unguarded solver gets wrong or right for the wrong reason:
    a target inside the post, which it reaches by running the hand through
    the post; a target beside the post, which it also reaches through the
    post although a clear configuration exists in the arm's redundancy; and
    a target far from the post, where nothing should change at all.
    """

    model = mujoco.MjModel.from_xml_string(
        """<mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="base" pos="0 0 0">
              <geom name="post" type="box" size="0.03 0.03 0.05" pos="0.35 0.20 0"/>
              <body name="upper" pos="0 0 0">
                <joint name="shoulder" type="hinge" axis="0 0 1" range="-3 3"/>
                <geom name="upper_geom" type="capsule" size="0.01" fromto="0 0 0 0.25 0 0"/>
                <body name="fore" pos="0.25 0 0">
                  <joint name="elbow" type="hinge" axis="0 0 1" range="-3 3"/>
                  <geom name="fore_geom" type="capsule" size="0.01" fromto="0 0 0 0.25 0 0"/>
                  <body name="hand" pos="0.25 0 0">
                    <joint name="wrist" type="hinge" axis="0 0 1" range="-3 3"/>
                    <geom name="hand_geom" type="capsule" size="0.01" fromto="0 0 0 0.25 0 0"/>
                    <site name="tip" pos="0.25 0 0" size="0.005"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>"""
    )
    base = _body_id(model, "base")
    pairs = tuple(
        sorted((min(base, _body_id(model, name)), max(base, _body_id(model, name))) for name in ("fore", "hand"))
    )
    guard = ik.CollisionGuard(pairs=pairs, clearance_m=0.02)
    joints = ("shoulder", "elbow", "wrist")
    seed = np.zeros(3)

    inside = np.array([[0.35, 0.20, 0.0]])
    unguarded = ik.solve_site_path(model, "tip", joints, inside, seed_qpos=seed)
    assert ik.penetrations(model, guard, unguarded.qpos[0]), "the unguarded solve runs the hand through the post"
    with pytest.raises(ik.IkFailure) as failure:
        ik.solve_site_path(model, "tip", joints, inside, seed_qpos=seed, guard=guard)
    assert failure.value.collision is not None and "base" in failure.value.collision
    assert failure.value.residual_m > ik.DEFAULT_TOLERANCE_M

    beside = np.array([[0.25, 0.40, 0.0]])
    through = ik.solve_site_path(model, "tip", joints, beside, seed_qpos=seed)
    assert ik.penetrations(model, guard, through.qpos[0]), "unguarded, the arm reaches beside the post through it"
    around = ik.solve_site_path(model, "tip", joints, beside, seed_qpos=seed, guard=guard)
    assert around.worst_residual_m <= ik.DEFAULT_TOLERANCE_M
    assert ik.penetrations(model, guard, around.qpos[0]) == ()

    far = np.array([[0.50, 0.30, 0.0]])
    free = ik.solve_site_path(model, "tip", joints, far, seed_qpos=seed)
    guarded = ik.solve_site_path(model, "tip", joints, far, seed_qpos=seed, guard=guard)
    assert np.allclose(free.qpos, guarded.qpos, atol=1e-12)


def test_grounding_types_a_self_colliding_request(zoo, inventory) -> None:
    """When the only way to a point is through the body, the refusal says so."""

    robot = _bimanual(zoo)
    model, manifest = robot.finalized.model, robot.manifest
    planner = OfflineSchemaPlanner(inventory)
    program = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
    # Start the body with its wrist already folded onto its forearm: the start
    # state itself is refused, typed, before any waypoint is solved.
    fixture = MotionProgramV2.model_validate_json(
        (FIXTURE / "dual_arm_program_before_guard.json").read_bytes()
    )
    guard = _collision_guard(manifest, model)
    folded, deepest = None, 0.0
    for key in fixture.tracks[0].keyframes:
        candidate = np.asarray(manifest.rest_qpos, dtype=float).copy()
        for name, value in key.joint_values.items():
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            candidate[int(model.jnt_qposadr[joint])] = value
        depth = max((d for _, _, d in ik.penetrations(model, guard, candidate)), default=0.0)
        if depth > deepest:
            folded, deepest = candidate, depth
    assert folded is not None and deepest > 0.02
    started_inside = manifest.model_copy(update={"rest_qpos": tuple(float(v) for v in folded)})
    with pytest.raises(GroundingError) as refusal:
        ground(program, started_inside, model, inventory)
    assert refusal.value.code is GeneralFailureCode.UNGROUNDABLE
    assert refusal.value.details["measurement"] == "start_state.self_collision"
    assert refusal.value.details["penetration_m"] > 0.0


def test_start_state_outside_a_limit_is_refused_before_planning(zoo, inventory) -> None:
    robot = next(iter(zoo.values()))
    model, manifest = robot.finalized.model, robot.manifest
    planner = OfflineSchemaPlanner(inventory)
    program = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
    dof = manifest.dofs[0]
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)
    start = np.asarray(manifest.rest_qpos, dtype=float).copy()
    start[int(model.jnt_qposadr[joint])] = dof.maximum + 0.5
    with pytest.raises(GroundingError) as refusal:
        ground(program, manifest.model_copy(update={"rest_qpos": tuple(start)}), model, inventory)
    assert refusal.value.details["measurement"] == "start_state.joint_limit"
    assert refusal.value.details["joint"] == dof.name


def test_a_displaced_start_is_where_the_motion_begins(zoo, inventory) -> None:
    """A body that does not start at its neutral pose starts where it is: the
    first reference sample is the start state, not a teleport to home."""

    robot = next(iter(zoo.values()))
    model, manifest = robot.finalized.model, robot.manifest
    planner = OfflineSchemaPlanner(inventory)
    program = planner.plan(PROMPT, afforded=afforded_entries(inventory, robot.morphology))
    start = np.asarray(manifest.rest_qpos, dtype=float).copy()
    moved = [d for d in manifest.dofs if d.maximum - d.minimum > 0.5][0]
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, moved.joint)
    address = int(model.jnt_qposadr[joint])
    start[address] = float(np.clip(start[address] + 0.2, moved.minimum, moved.maximum))
    displaced = manifest.model_copy(update={"rest_qpos": tuple(float(v) for v in start)})
    grounded = ground(program, displaced, model, inventory)
    first = grounded.program.tracks[0].keyframes[0].joint_values
    last = grounded.program.tracks[0].keyframes[-1].joint_values
    assert first[moved.name] == pytest.approx(start[address], abs=1e-9)
    assert last[moved.name] == pytest.approx(start[address], abs=1e-9)
