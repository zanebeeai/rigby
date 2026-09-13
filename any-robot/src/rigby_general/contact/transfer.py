"""Acquire an object, hold it, carry it, set it down and let go -- and prove each.

The grasp probe stops at the lift. A transfer is the whole contact family G06
asks for: approach, guarded descent, force-driven closure, lift, a sustained
hold, transport into a destination region, a guarded lowering, release, retreat
and a dwell during which an independent evaluator decides whether the object
is resting inside the region with nothing from the robot touching it.

Everything here runs on the model's native physics clock -- phase timing,
closure integration and the recorded timestamps are ``MjData.time`` -- and the
arm's path is solved with the same self-collision guard the grounder uses, so
a hand that would have to pass through its own forearm to reach the object is
refused, typed, before anything moves. The object is never written: its free
joint evolves only through ``mj_step``. The scene is checked for the things
that would make a hold meaningless -- an equality constraint, an actuator on
the object, a contact exclusion involving it -- and any of them is a refusal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from rigby_core.simulation.controller import ControlTarget
from rigby_core.simulation.recording import STATE_SPEC, PhysicsRecorder

from ..contracts import EffectorV1, RobotAssetManifestV1, SiteSemantic
from ..gates.control import ComputedTorqueController, ControllerConfig
from ..grounding import ik
from ..grounding.grounder import _collision_guard
from ..grounding.workspace import WorkspaceFrame
from ..scenes.block import GraspScene, block_qpos_address
from ..scenes.environment import OBJECT_PREFIX, SUPPORT_PREFIX, EnvironmentV1
from .closure import ClosureController, GripState
from .grasp import TRAVERSE_MARGIN, _hand_facing, _scene_rest_qpos, effector_grasp_site
from .placement import PlacementEvaluator, PlacementGoal


PHASES = ("approach", "turn", "descend", "close", "lift", "hold", "carry", "lower", "release", "retreat", "dwell")

HOVER_HEIGHTS = 1.75
"""Clearance above an object's top face at which the approach hovers, in
object heights; the same figure the authored-world task planner uses."""
LIFT_HEIGHTS = 2.5
"""How far the object is raised before it travels, in object heights."""
PLACE_STANDOFF_M = 0.006
"""Allowance between the fingertips and a support: how far above its resting
height the object is commanded before release, and how far above the bench
the fingers end when the grasp point is placed. Six millimetres rather than
two because a hand that arrives a few degrees off vertical -- the five-axis
arm cannot always do better -- swings a fingertip lower on one side."""
MIN_PHASE_S = {
    "approach": 1.0, "turn": 0.8, "descend": 0.8, "close": 0.6, "lift": 1.0, "hold": 2.0,
    "carry": 1.2, "lower": 0.8, "release": 0.5, "retreat": 0.8, "dwell": 2.0,
}
CLOSE_TIMEOUT_S = 2.5
"""A closure that has not settled on the object within this is not going to."""
DWELL_MARGIN_S = 0.5
"""How much longer than the required dwell the arm waits, still, after release."""
SETTLE_S = 0.25
"""Physics run before the attempt so an authored object has come to rest."""
CARRY_OFFSET_FRACTION = 2.5
"""Of the gripper's aperture: how far the object may sit from the grasp
centre while held before it counts as departed rather than carried."""
LIFT_REQUIRED_FRACTION = 0.8
"""Of the object's height: the least a lift has to raise it to count."""
HOLD_DROP_FRACTION = 0.5
"""Of the required lift: how far the object may sag during the hold."""
FACING_TOLERANCE_RAD = np.radians(15.0)
"""How far from the requested facing the planned hand may be at the end of
the turn and at the grasp. From the rest pose every body arrives within ten
degrees; from a folded posture the solver can leave the hand forty-five
degrees over, and a hand that descends tilted closes its fingers beside
the object. That is refused before motion, typed, so a composition can
insert a transition instead of executing a grasp that cannot close."""


@dataclass(frozen=True, slots=True)
class TransferViolation:
    code: str
    detail: str
    measured: float
    limit: float


@dataclass(frozen=True, slots=True)
class TransferScene:
    """A grasp scene that also knows where the object is going."""

    scene: GraspScene
    destination_m: np.ndarray
    """Where the object's centre rests once placed: on the destination top."""
    destination_top_m: float
    goal: PlacementGoal
    environment_id: str
    object_name: str

    @property
    def model(self) -> mujoco.MjModel:
        return self.scene.model


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    name: str
    start_s: float
    end_s: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class TransferStart:
    """Where a transfer begins when it continues a world another leaf left:
    the full physical state and the clock, so the recorded trace stays one
    continuous physics run and nothing is reset between attempts."""

    qpos: np.ndarray
    qvel: np.ndarray
    time_s: float
    state: np.ndarray | None = None
    """The full integration state (positions, velocities, actuator state,
    solver warm start, clock) the previous leaf recorded last. Restoring it
    exactly is what lets the recorded controls replay across the boundary:
    a fresh solver warm start would integrate to a slightly different
    state than the record shows."""


@dataclass(frozen=True, slots=True)
class TransferResult:
    certified: bool
    violations: tuple[TransferViolation, ...]
    phases: tuple[PhaseRecord, ...]
    duration_s: float
    lift_height_m: float
    hold_s: float
    carry_offset_max_m: float
    peak_force_n: float
    max_penetration_m: float
    opposition_achieved: bool
    placement_dwell_s: float
    placement_success: bool
    placed_inside: bool
    released: bool
    unexpected_contacts: tuple[tuple[str, str], ...]
    robot_fixture_contacts: tuple[tuple[str, str], ...]
    collision_policy: dict
    path_seed: str = ""
    """Which restart seed the path was solved from: rest, turned or midpoint."""
    times_s: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    qpos: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    ctrl: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    demand: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    object_position_m: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))
    grip_force_n: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    interrupted: bool = False
    """Stopped from outside before the sequence ended; the phases that ran
    are recorded and the object was never written."""
    final_qvel: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    final_time_s: float = 0.0
    final_state: np.ndarray | None = field(repr=False, default=None)

    @property
    def failed_gate(self) -> str | None:
        return self.violations[0].code if self.violations else None

    @property
    def executed(self) -> bool:
        return len(self.times_s) > 0

    def continuation(self) -> "TransferStart | None":
        """The state a following leaf starts from, or ``None`` if nothing moved."""

        if not self.executed:
            return None
        return TransferStart(qpos=np.array(self.qpos[-1], dtype=float), qvel=np.array(self.final_qvel, dtype=float), time_s=self.final_time_s,
                             state=None if self.final_state is None else np.array(self.final_state, dtype=float))


def transfer_scene_from_environment(
    manifest: RobotAssetManifestV1,
    mjcf_xml: str,
    environment: EnvironmentV1,
    *,
    object_name: str,
    destination_fixture: str,
    goal: PlacementGoal,
    asset_root=None,
) -> TransferScene:
    """The authored world with its target renamed for the grasp machinery,
    plus the destination read off the named fixture."""

    from .task import build_task_scene

    scene = build_task_scene(manifest, mjcf_xml, environment, object_name, asset_root=asset_root)
    fixture = next(f for f in environment.fixtures if f.name == destination_fixture)
    top = float(fixture.position_m[2] + fixture.size_m[2])
    half = scene.block_half_extent_m
    destination = np.array([fixture.position_m[0], fixture.position_m[1], top + half], dtype=float)
    return TransferScene(scene=scene, destination_m=destination, destination_top_m=top, goal=goal,
                         environment_id=environment.environment_id, object_name=object_name)


def collision_policy(model: mujoco.MjModel, object_body: str = "scene_block") -> dict:
    """What the scene does that could fake a hold; every entry must be clean."""

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scene_block_free")
    object_actuators = [
        index for index in range(model.nu)
        if int(model.actuator_trntype[index]) == int(mujoco.mjtTrn.mjTRN_JOINT) and int(model.actuator_trnid[index, 0]) == joint
    ]
    excluded_with_object = []
    for index in range(model.nexclude):
        signature = int(model.exclude_signature[index])
        first, second = signature >> 16, signature & 0xFFFF
        if body in (first, second):
            excluded_with_object.append([
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second),
            ])
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
    return {
        "equality_constraints": int(model.neq),
        "object_actuators": len(object_actuators),
        "contact_exclusions_with_object": excluded_with_object,
        "contact_exclusions_total": int(model.nexclude),
        "object_geoms_collidable": all(int(model.geom_contype[g]) or int(model.geom_conaffinity[g]) for g in geoms),
        "object_state_writes": 0,
    }


def _path(scene: TransferScene, frame: WorkspaceFrame, home: np.ndarray, offset_m: float = 0.0, standoff_m: float = 0.0,
          phase_range: tuple[str, str] = (PHASES[0], PHASES[-1]), object_position_m: np.ndarray | None = None) -> tuple[list[np.ndarray], dict[str, tuple[int, int]]]:
    """Waypoints, and which consecutive pair each moving phase travels.

    ``object_position_m`` is where the object is believed to be -- what a
    sensor reported -- when that is not where the world was authored to
    put it; the object is then taken to rest on whatever is under it.

    ``phase_range`` names the first and last phase that will run. The
    waypoint list begins at ``home`` -- where the grasp point is now -- and
    continues from the end of the first moving phase in the range, so a
    transfer that resumes mid-sequence (a place skill after a carry) solves
    only the spans it will travel, from where the arm actually is.

    The reach envelope was measured for the grasp centre; the point being
    driven is the grasp point, ``offset_m`` further along a hand that points
    down. Heights are therefore clamped as the centre would see them: a hover
    the centre can reach is one the point can reach with the hand vertical.

    ``standoff_m`` raises the grasp point above the object's centre by however
    far the fingers reach past that point, so a hand whose gripping surfaces
    are longer than the object is tall closes on it with the lower part of
    the fingers instead of driving the fingertips into the bench.
    """

    grasp = scene.scene
    half = grasp.block_half_extent_m
    height = 2.0 * half
    up = np.asarray(frame.up, dtype=float) * offset_m
    lift_up = np.asarray(frame.up, dtype=float) * standoff_m

    def clamp(point: np.ndarray) -> np.ndarray:
        return frame.clamp_rising(point + up) - up

    if object_position_m is None:
        source = np.asarray(grasp.block_position_m, dtype=float)
        support_top = grasp.support_height_m
    else:
        source = np.asarray(object_position_m, dtype=float)
        support_top = float(source[2]) - half
    above_source = clamp(np.array([source[0], source[1], support_top + height + HOVER_HEIGHTS * height]) + lift_up)
    at_source = source + lift_up
    lifted = clamp(np.array([source[0], source[1], source[2] + LIFT_HEIGHTS * height]) + lift_up)
    destination = np.asarray(scene.destination_m, dtype=float)
    above_destination = clamp(np.array([destination[0], destination[1], max(lifted[2], destination[2] + LIFT_HEIGHTS * height + standoff_m)]))
    at_destination = np.array([destination[0], destination[1], destination[2] + PLACE_STANDOFF_M]) + lift_up
    retreat = above_destination.copy()
    # The hover appears twice: the approach reaches it however the straight
    # line from home leads, and the turn span then brings the hand to face
    # the object with the position held. Without that stationary turn the
    # hand rotated from 60 degrees off vertical to vertical while descending,
    # and a finger swinging through that arc clipped a cube that sat a few
    # millimetres off the nominal spot and knocked it away.
    points = [home, above_source, above_source.copy(), at_source, lifted, above_destination, at_destination, retreat]
    spans = {"approach": (0, 1), "turn": (1, 2), "descend": (2, 3), "lift": (3, 4), "carry": (4, 5), "lower": (5, 6), "retreat": (6, 7)}
    first, last = phase_range
    if first == PHASES[0]:
        return points, spans
    moving = [name for name in PHASES[PHASES.index(first): PHASES.index(last) + 1] if name in spans]
    if not moving:
        return [home], {}
    start_index = spans[moving[0]][0]
    trimmed = [home] + points[start_index + 1:]
    shift = start_index
    return trimmed, {name: (a - shift, b - shift) for name, (a, b) in spans.items() if name in moving}


def restart_seeds(model, frame: WorkspaceFrame, joints: tuple[str, ...], rest: np.ndarray, target: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Configurations to start a solve from, in order: the rest pose; the rest
    pose with any joint whose axis is the frame's up turned to face the
    target's bearing; the middle of every joint's range.

    A local solver goes where its seed points it. From a straight-up rest pose
    the first swing toward a target beside the body can fold the upper arm
    through the base, and from there the guard rightly refuses; turned to face
    the target first, the same arm reaches out instead. The midpoint is the
    conventional escape from a singular rest pose.
    """

    seeds = [("rest", np.array(rest, dtype=float))]
    data = mujoco.MjData(model)
    data.qpos[:] = rest
    mujoco.mj_kinematics(model, data)
    up = np.asarray(frame.up, dtype=float)
    bearing = np.asarray(target, dtype=float) - np.asarray(frame.origin, dtype=float)
    bearing = bearing - up * float(bearing @ up)
    if float(np.linalg.norm(bearing)) > 1e-6:
        bearing = bearing / float(np.linalg.norm(bearing))
        out = np.asarray(frame.out, dtype=float)
        azimuth = float(np.arctan2(float(np.cross(out, bearing) @ up), float(out @ bearing)))
        turned = np.array(rest, dtype=float)
        for name in joints:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            axis = np.array(data.xmat[int(model.jnt_bodyid[joint])], dtype=float).reshape(3, 3) @ np.asarray(model.jnt_axis[joint], dtype=float)
            if abs(float(axis @ up)) > 0.95:
                address = int(model.jnt_qposadr[joint])
                low, high = model.jnt_range[joint]
                turned[address] = float(np.clip(turned[address] + np.sign(float(axis @ up)) * azimuth, low, high))
                seeds.append(("turned", turned))
                break
    midpoint = np.array(rest, dtype=float)
    for name in joints:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[joint]
        midpoint[int(model.jnt_qposadr[joint])] = 0.5 * (float(low) + float(high))
    seeds.append(("midpoint", midpoint))
    return seeds


def _joint_path(model, site: str, joints: tuple[str, ...], points: list[np.ndarray], seeds: list[tuple[str, np.ndarray]], guard, per_span: int, facing=None, facing_from_span: int = 1) -> tuple[np.ndarray, list[int], str]:
    """Solve the densified path span by span from the first seed that
    succeeds; return joint rows, the row index of each waypoint and the seed.

    The facing is asked for from ``facing_from_span`` on: the approach from
    home crosses the workspace wherever the straight line leads, and holding
    the hand vertical along all of it can make an intermediate point
    unreachable -- the jaw arm was refused 53 mm short of a point halfway
    down from its rest pose. The hand turns to face the object in the turn
    span at the hover, where the position is held and the wrist is free, and
    stays vertical through the descent, the lift, the carry and the lowering.
    """

    dense: list[np.ndarray] = [points[0]]
    marks = [0]
    for start, end in zip(points, points[1:]):
        for step in range(1, per_span + 1):
            dense.append(start + (end - start) * (step / per_span))
        marks.append(len(dense) - 1)
    failure: ik.IkFailure | None = None
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    for label, seed in seeds:
        rows: list[np.ndarray] = []
        current = np.array(seed, dtype=float)
        try:
            for span in range(len(points) - 1):
                first = 0 if span == 0 else marks[span] + 1
                targets = np.asarray(dense[first: marks[span + 1] + 1])
                if facing is not None and span == facing_from_span:
                    # The turn: the position is held and the requested axis
                    # is swung from where the hand points now to where it
                    # must point, one small rotation per dense target, so
                    # every row is a converged solution a few degrees from
                    # the last. Asking for the final facing in one go gave a
                    # joint-space blend between two configurations that
                    # share only the site position, and the fingers of a two
                    # metre arm swept fifty millimetres through the cube.
                    probe = mujoco.MjData(model)
                    probe.qpos[:] = current
                    mujoco.mj_kinematics(model, probe)
                    now = np.array(probe.site_xmat[site_id], dtype=float).reshape(3, 3) @ np.asarray(facing[0], dtype=float)
                    for step, target in enumerate(targets, start=1):
                        axis = _slerp(now, np.asarray(facing[1], dtype=float), step / len(targets))
                        solution = ik.solve_site_path(model, site, joints, target[None, :], seed_qpos=current, guard=guard, facing=(facing[0], axis))
                        rows.extend(solution.qpos)
                        current = solution.qpos[-1]
                    continue
                solution = ik.solve_site_path(
                    model, site, joints, targets, seed_qpos=current, guard=guard,
                    facing=facing if span >= facing_from_span else None,
                )
                rows.extend(solution.qpos)
                current = solution.qpos[-1]
        except ik.IkFailure as error:
            failure = failure or error
            continue
        return np.asarray(rows), marks, label
    assert failure is not None
    raise failure


def _slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """A unit vector ``fraction`` of the way from ``start`` to ``end`` along
    the great circle; antiparallel inputs turn through any perpendicular."""

    a = start / max(float(np.linalg.norm(start)), 1e-12)
    b = end / max(float(np.linalg.norm(end)), 1e-12)
    cosine = float(np.clip(a @ b, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-6:
        return b
    if angle > np.pi - 1e-3:
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(a, helper)
        perpendicular /= float(np.linalg.norm(perpendicular))
        return np.cos(fraction * angle) * a + np.sin(fraction * angle) * perpendicular
    return (np.sin((1.0 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)


def attempt_transfer(
    manifest: RobotAssetManifestV1,
    scene: TransferScene,
    effector: EffectorV1,
    frame: WorkspaceFrame,
    *,
    recorder: PhysicsRecorder | None = None,
    per_span: int = 6,
    resume: TransferStart | None = None,
    should_stop: Callable[[float], bool] | None = None,
    phase_range: tuple[str, str] = (PHASES[0], PHASES[-1]),
    object_position_m: np.ndarray | None = None,
    on_step: Callable[[mujoco.MjData], None] | None = None,
) -> TransferResult:
    """Run one transfer and gate every phase of it.

    ``object_position_m`` is the believed position of the object the path
    is planned to, when a sensor rather than the world's authoring says
    where it is. ``on_step`` is called with the data after every physics
    step and once before the first: where a monitor samples its sensors,
    and where a protocol applies the external forces or moves the
    occluders it declares, all of which the record keeps as user input.

    ``phase_range`` runs a contiguous part of the sequence -- approach
    through carry as an acquisition that ends holding the object, lower
    through dwell as a placement that begins holding it -- and gates only
    what those phases can establish: a placement is not asked whether it
    lifted, an acquisition is not asked whether it released.

    With ``resume`` the transfer continues the world another leaf left: the
    arm begins where it is, the object where it lies, the clock where it
    stood, and nothing settles or resets. ``should_stop`` is asked on every
    physics step with the simulation time; answering true ends the
    transfer where it is, typed ``interrupted``, with the phases that ran.
    """

    model = scene.model
    grasp = scene.scene
    policy = collision_policy(model)
    violations: list[TransferViolation] = []
    if policy["equality_constraints"]:
        violations.append(TransferViolation("hidden_weld", "the scene contains an equality constraint", float(policy["equality_constraints"]), 0.0))
    if policy["object_actuators"]:
        violations.append(TransferViolation("object_actuator", "the object is actuated", float(policy["object_actuators"]), 0.0))
    if policy["contact_exclusions_with_object"]:
        violations.append(TransferViolation("object_collision_exclusion", "a contact exclusion names the object", float(len(policy["contact_exclusions_with_object"])), 0.0))
    if not policy["object_geoms_collidable"]:
        violations.append(TransferViolation("object_not_collidable", "the object cannot make contact", 0.0, 1.0))
    if violations:
        # A world that could fake a hold is refused before anything moves;
        # a certified transfer through it would prove nothing.
        return _refused(violations, None, policy)

    arm_joints = ik.chain_joint_names(model, frame.figure_site, exclude=frozenset(effector.grip_joints))
    rest = _scene_rest_qpos(model, manifest) if resume is None else np.array(resume.qpos, dtype=float)
    guard = _collision_guard(manifest, model)
    solve_site = next(
        (s.name for s in manifest.morphology.sites if s.semantic is SiteSemantic.GRASP_POINT and s.name.startswith(effector.chain_id)),
        frame.figure_site,
    )
    home_data = mujoco.MjData(model)
    home_data.qpos[:] = rest
    mujoco.mj_kinematics(model, home_data)
    home = np.array(home_data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, solve_site)], dtype=float)
    facing = _hand_facing(manifest, effector)
    offset = _grasp_offset_m(manifest, effector)
    standoff = grasp_standoff_m(model, manifest, effector, solve_site, grasp.block_half_extent_m)
    first_phase, last_phase = phase_range
    if first_phase not in PHASES or last_phase not in PHASES or PHASES.index(first_phase) > PHASES.index(last_phase):
        raise ValueError(f"phase range {phase_range} is not a contiguous part of {PHASES}")
    order = list(PHASES[PHASES.index(first_phase): PHASES.index(last_phase) + 1])
    points, spans = _path(scene, frame, home, offset, standoff, phase_range=phase_range, object_position_m=object_position_m)
    downward = (facing, -np.asarray(frame.up, dtype=float)) if facing is not None else None

    seed_target = points[spans["descend"][1]] if "descend" in spans else points[-1]
    # Restart seeds are configurations the arm may be placed in before the
    # first recorded state; a transfer that continues a world another skill
    # left must solve from where the arm actually stands, since a path whose
    # first row is another solution of the same point would ask the
    # controller to jump to it.
    seeds = restart_seeds(model, frame, arm_joints, rest, seed_target) if (first_phase == PHASES[0] and resume is None) else [("rest", np.array(rest, dtype=float))]
    facing_from_span = 1 if "turn" in spans else 0
    try:
        path, marks, seed_used = _joint_path(model, solve_site, arm_joints, points, seeds, guard, per_span, facing=downward, facing_from_span=facing_from_span)
    except ik.IkFailure as error:
        code = "self_collision_path" if error.collision is not None else "unreachable_path"
        return _refused(violations, TransferViolation(code, str(error)[:300], float(error.residual_m), 0.0), policy)
    if downward is not None and "descend" in spans:
        worst = max(facing_angle(model, solve_site, path[marks[index]], downward[0], downward[1]) for index in spans["descend"])
        if worst > FACING_TOLERANCE_RAD:
            return _refused(violations, TransferViolation("facing_unmet", f"the planned hand is {np.degrees(worst):.1f} degrees from the requested facing at the hover or the grasp", float(worst), float(FACING_TOLERANCE_RAD)), policy)

    controller = ComputedTorqueController(model, ControllerConfig())
    closure = ClosureController(model, manifest, effector, object_geoms=frozenset({"scene_block_geom"}))
    evaluator = PlacementEvaluator(model, scene.goal, object_geom="scene_block_geom", object_joint="scene_block_free")

    arm_adr = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in arm_joints])
    grip_adr = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in effector.grip_joints])
    block_adr = block_qpos_address(model)
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "scene_block_geom")
    fixture_geoms = {g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(SUPPORT_PREFIX)}
    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, effector_grasp_site(manifest, effector))
    world_bodies = {b for b in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith((SUPPORT_PREFIX, OBJECT_PREFIX, "scene_"))} | {0}
    adjacency = {(min(b, int(model.body_parentid[b])), max(b, int(model.body_parentid[b]))) for b in range(1, model.nbody)}
    # The members of one effector closing on each other -- jaws meeting on air
    # -- are the gripper working, not the body colliding with itself.
    members = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in effector.member_bodies]
    for first in members:
        for second in members:
            if first >= 0 and second >= 0 and first != second:
                adjacency.add((min(first, second), max(first, second)))
    for first, second in manifest.adjacent_collision_exclusions:
        a, b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first), mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, second)
        if a >= 0 and b >= 0:
            adjacency.add((min(a, b), max(a, b)))

    # Phase durations from the joint travel each phase asks for, at the
    # declared joint speeds with the traverse margin, never below the floors.
    speeds = np.array([max(abs(float(next((d.velocity_limit for d in manifest.dofs if d.joint == n), 0.0) or 0.0)), 1e-3) for n in arm_joints])
    durations: dict[str, float] = {}
    for name, (start, end) in spans.items():
        travel = np.abs(np.diff(path[marks[start]: marks[end] + 1][:, arm_adr], axis=0)).sum(axis=0)
        durations[name] = max(MIN_PHASE_S[name], float(np.max(travel / speeds)) * TRAVERSE_MARGIN)
    # The dwell phase outlasts the required dwell by a margin: the evaluator
    # starts counting a step after the object first qualifies, and a phase
    # exactly as long as the requirement ends two milliseconds short of it.
    durations.update({"close": CLOSE_TIMEOUT_S, "hold": MIN_PHASE_S["hold"], "release": MIN_PHASE_S["release"], "dwell": scene.goal.dwell_s + DWELL_MARGIN_S})
    durations = {name: durations[name] for name in order}

    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    if resume is None:
        data.qpos[:] = rest
        data.qpos[arm_adr] = path[0][arm_adr]
        mujoco.mj_forward(model, data)
        for _ in range(int(round(SETTLE_S / dt))):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    elif resume.state is not None:
        # The arm is where it is; the controller tracks the path from there.
        # The forward pass fills in positions and contacts but replaces the
        # solver warm start, which is then restored from the record.
        mujoco.mj_setState(model, data, np.asarray(resume.state, dtype=float), STATE_SPEC)
        mujoco.mj_forward(model, data)
        mujoco.mj_setState(model, data, np.asarray(resume.state, dtype=float), STATE_SPEC)
    else:
        data.qpos[:] = rest
        data.qvel[:] = np.asarray(resume.qvel, dtype=float)
        data.time = float(resume.time_s)
        mujoco.mj_forward(model, data)
    if on_step is not None:
        on_step(data)
    start_position = np.array(data.qpos[block_adr: block_adr + 3], dtype=float)
    start_height = float(start_position[2])
    nudge_limit = 0.25 * grasp.block_half_extent_m
    required_lift = LIFT_REQUIRED_FRACTION * 2.0 * grasp.block_half_extent_m

    times: list[float] = []
    qpos_log: list[np.ndarray] = []
    ctrl_log: list[np.ndarray] = []
    demand_log: list[np.ndarray] = []
    object_log: list[np.ndarray] = []
    force_log: list[float] = []
    phases: list[PhaseRecord] = []
    unexpected: set[tuple[str, str]] = set()
    fixture_contacts: set[tuple[str, str]] = set()
    peak_force = peak_penetration = 0.0
    opposition = False
    peak_height = start_height
    max_carry_offset = 0.0
    hold_supported_s = 0.0
    lift_height = 0.0
    target = mujoco.MjData(model)
    phase_index = 0
    phase_start = float(data.time)
    frozen_arm = path[0][arm_adr].copy()
    descent_hold: np.ndarray | None = None
    lower_hold: np.ndarray | None = None
    release_started: float | None = None
    outcome_note = ""
    interrupted = False

    def arm_target_for(name: str, progress: float) -> np.ndarray:
        start, end = spans[name]
        lo, hi = marks[start], marks[end]
        index = lo + progress * (hi - lo)
        low = int(np.clip(np.floor(index), lo, hi))
        high = int(np.clip(low + 1, lo, hi))
        blend = float(np.clip(index - low, 0.0, 1.0))
        return (1.0 - blend) * path[low][arm_adr] + blend * path[high][arm_adr]

    while phase_index < len(order):
        name = order[phase_index]
        now = float(data.time)
        if should_stop is not None and should_stop(now):
            phases.append(PhaseRecord(name, phase_start, now, "interrupted"))
            interrupted = True
            break
        elapsed = now - phase_start
        duration = durations[name]
        progress = float(np.clip(elapsed / duration, 0.0, 1.0)) if duration > 0 else 1.0
        closing = name in ("close", "lift", "hold", "carry", "lower") or (name == "descend" and descent_hold is not None)
        report = closure.step(data, dt, closing=closing)
        opposition = opposition or report.opposition_satisfied
        if report.state is GripState.HOLDING:
            peak_force = max(peak_force, report.peak_force_n)
            peak_penetration = max(peak_penetration, report.max_penetration_m)
        block_position = np.array(data.qpos[block_adr: block_adr + 3], dtype=float)

        advance = False
        if name in ("approach", "turn"):
            planned = arm_target_for(name, progress)
            advance = progress >= 1.0
        elif name == "descend":
            if descent_hold is None and float(np.linalg.norm(block_position - start_position)) > nudge_limit:
                descent_hold = arm_target_for(name, progress)
            planned = descent_hold if descent_hold is not None else arm_target_for(name, progress)
            advance = progress >= 1.0 or descent_hold is not None
        elif name == "close":
            planned = frozen_arm
            if closure.settled and report.opposition_satisfied:
                advance = True
            elif progress >= 1.0:
                outcome_note = "closure never settled on the object"
                advance = True
        elif name == "lift":
            planned = arm_target_for(name, progress)
            advance = progress >= 1.0
        elif name == "hold":
            planned = frozen_arm
            if report.holding and float(block_position[2]) - start_height >= required_lift * HOLD_DROP_FRACTION:
                hold_supported_s += dt
            advance = progress >= 1.0
        elif name == "carry":
            planned = arm_target_for(name, progress)
            advance = progress >= 1.0
        elif name == "lower":
            touching = any(
                object_geom in (int(c.geom1), int(c.geom2)) and (int(c.geom1) in fixture_geoms or int(c.geom2) in fixture_geoms) and float(c.dist) <= 0.0
                for c in (data.contact[i] for i in range(data.ncon))
            )
            if lower_hold is None and touching and progress > 0.2:
                lower_hold = arm_target_for(name, progress)
            planned = lower_hold if lower_hold is not None else arm_target_for(name, progress)
            advance = progress >= 1.0 or lower_hold is not None
        elif name == "release":
            planned = frozen_arm
            advance = progress >= 1.0
        elif name == "retreat":
            planned = arm_target_for(name, progress)
            advance = progress >= 1.0
        else:  # dwell
            planned = frozen_arm
            evaluator.assess(data)
            advance = progress >= 1.0

        target.qpos[:] = rest
        target.qpos[arm_adr] = planned
        if closing:
            target.qpos[grip_adr] = data.qpos[grip_adr]
        else:
            commanded = closure.commanded_qpos()
            target.qpos[grip_adr] = [commanded.get(n, float(data.qpos[a])) for n, a in zip(effector.grip_joints, grip_adr)]
        target.qpos[block_adr: block_adr + 7] = data.qpos[block_adr: block_adr + 7]
        command = controller.compute(data, ControlTarget(qpos=np.array(target.qpos), qvel=np.zeros(model.nv), qacc=np.zeros(model.nv)))
        for joint_name, force in (closure.force_commands() if closing else {}).items():
            actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_motor")
            if actuator >= 0:
                command[actuator] = float(np.clip(force, model.actuator_forcerange[actuator][0], model.actuator_forcerange[actuator][1]))

        times.append(now)
        qpos_log.append(np.array(data.qpos, dtype=float))
        ctrl_log.append(np.array(command, dtype=float))
        demand_log.append(np.array(controller.last_demand, dtype=float))
        object_log.append(block_position)
        force_log.append(float(report.peak_force_n))
        peak_height = max(peak_height, float(block_position[2]))
        if opposition and grasp_site >= 0:
            max_carry_offset = max(max_carry_offset, float(np.linalg.norm(np.array(data.site_xpos[grasp_site], dtype=float) - block_position)))
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            if first == second:
                continue
            pair = (min(first, second), max(first, second))
            names = tuple(sorted((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first) or "?", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second) or "?")))
            if first in world_bodies or second in world_bodies:
                robot_side = second if first in world_bodies else first
                world_side = first if first in world_bodies else second
                if robot_side not in world_bodies and int(model.geom_bodyid[object_geom]) != world_side:
                    fixture_contacts.add(names)
                continue
            if pair in adjacency:
                continue
            unexpected.add(names)
        if recorder is not None:
            recorded = recorder.rows["time_s"]
            if resume is not None and recorded and now <= recorded[-1] + 1e-12:
                # Continuing from the state and instant another leaf recorded
                # last: that sample's command was never applied, this one is.
                if now < recorded[-1] - 1e-12:
                    raise ValueError("a transfer cannot resume before the last recorded sample")
                recorder.amend_last_action(command, demand=controller.last_demand)
            else:
                recorder.capture(data, command, control_time_s=now, demand=controller.last_demand)

        if advance:
            phases.append(PhaseRecord(name, phase_start, now, outcome_note if name == "close" else ""))
            if name == "descend":
                frozen_arm = planned.copy()
            elif name == "close":
                lift_height = 0.0
            elif name == "lift":
                lift_height = float(block_position[2]) - start_height
                frozen_arm = planned.copy()
            elif name == "lower":
                frozen_arm = planned.copy()
                release_started = now
            elif name == "retreat":
                frozen_arm = planned.copy()
            phase_index += 1
            phase_start = now
            if phase_index >= len(order):
                break
        data.ctrl[:] = command
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(data)

    final_state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=np.float64)
    mujoco.mj_getState(model, data, final_state, STATE_SPEC)
    final = evaluator.last
    lift = peak_height - start_height
    ran = set(order)
    if interrupted:
        violations.append(TransferViolation("interrupted", f"stopped from outside during {phases[-1].name} at {phases[-1].end_s:.3f} s", float(phases[-1].end_s), 0.0))
    if not opposition and ("close" in ran or "lower" in ran):
        violations.append(TransferViolation("grasp_not_achieved", "opposing members never both made contact with the object", 0.0, 1.0))
    if "lift" in ran and lift < required_lift:
        violations.append(TransferViolation("object_not_lifted", "the object never came off its support", lift, required_lift))
    if "hold" in ran and hold_supported_s + 1e-9 < MIN_PHASE_S["hold"]:
        violations.append(TransferViolation("hold_not_sustained", "the object was not held continuously through the hold", hold_supported_s, MIN_PHASE_S["hold"]))
    carry_limit = CARRY_OFFSET_FRACTION * (effector.max_aperture_m or 0.05)
    if opposition and max_carry_offset > carry_limit:
        violations.append(TransferViolation("object_not_carried", "the object left the gripper instead of being carried", max_carry_offset, carry_limit))
    if "dwell" in ran:
        if final is None or not final.whole_geometry_inside:
            violations.append(TransferViolation("not_transported", "the object did not end inside the destination region", 0.0, 1.0))
        if final is not None and not final.released:
            violations.append(TransferViolation("not_released", "the robot was still touching the object at the end", final.maximum_robot_normal_force_n, 0.0))
        if not evaluator.success:
            violations.append(TransferViolation("placement_unstable", "the object did not rest inside the region, released and still, for the dwell", evaluator.dwell_s, scene.goal.dwell_s))
    elif "carry" in ran and last_phase == "carry" and not (opposition and closure.state is GripState.HOLDING):
        violations.append(TransferViolation("hold_not_sustained", "the acquisition ended without the object held in opposition", 0.0, 1.0))
    if peak_penetration > closure.config.max_penetration_m:
        violations.append(TransferViolation("excessive_penetration", "the members were inside the object rather than around it", peak_penetration, closure.config.max_penetration_m))
    if unexpected:
        first, second = sorted(unexpected)[0]
        violations.append(TransferViolation("self_collision", f"{first} and {second} collided", float(len(unexpected)), 0.0))

    return TransferResult(
        certified=not violations, violations=tuple(violations), phases=tuple(phases),
        duration_s=float(times[-1]) if times else 0.0, lift_height_m=lift, hold_s=hold_supported_s,
        carry_offset_max_m=max_carry_offset, peak_force_n=peak_force, max_penetration_m=peak_penetration,
        opposition_achieved=opposition, placement_dwell_s=evaluator.dwell_s, placement_success=evaluator.success,
        placed_inside=bool(final is not None and final.whole_geometry_inside), released=bool(final is not None and final.released),
        unexpected_contacts=tuple(sorted(unexpected)), robot_fixture_contacts=tuple(sorted(fixture_contacts)),
        collision_policy=policy, path_seed=seed_used, times_s=np.asarray(times), qpos=np.asarray(qpos_log), ctrl=np.asarray(ctrl_log),
        demand=np.asarray(demand_log), object_position_m=np.asarray(object_log), grip_force_n=np.asarray(force_log),
        interrupted=interrupted, final_qvel=np.array(data.qvel, dtype=float), final_time_s=float(data.time), final_state=final_state,
    )


def facing_angle(model, site: str, qpos: np.ndarray, local_axis: np.ndarray, world_axis: np.ndarray) -> float:
    """Radians between the site's local axis at ``qpos`` and the world axis."""

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    rotation = np.array(data.site_xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)], dtype=float).reshape(3, 3)
    pointed = rotation @ np.asarray(local_axis, dtype=float)
    return float(np.arccos(np.clip(pointed @ np.asarray(world_axis, dtype=float) / max(np.linalg.norm(pointed) * np.linalg.norm(world_axis), 1e-12), -1.0, 1.0)))


def _extent_along(model, data, bodies: tuple[str, ...], origin: np.ndarray, direction: np.ndarray) -> float:
    """How far the geoms of ``bodies`` reach past ``origin`` along ``direction``."""

    worst = 0.0
    for name in bodies:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            continue
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) != body:
                continue
            centre = np.array(data.geom_xpos[geom], dtype=float)
            rotation = np.array(data.geom_xmat[geom], dtype=float).reshape(3, 3)
            size = np.array(model.geom_size[geom], dtype=float)
            kind = int(model.geom_type[geom])
            if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                half = size[:3]
            elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                half = np.array([size[0]] * 3)
            elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                half = np.array([size[0], size[0], size[1] + (size[0] if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0)])
            else:
                half = np.array([float(model.geom_rbound[geom])] * 3)
            reach = float(np.sum(np.abs(rotation.T @ direction) * half))
            worst = max(worst, float((centre - origin) @ direction) + reach)
    return worst


def grasp_standoff_m(model, manifest: RobotAssetManifestV1, effector: EffectorV1, grasp_site: str, half_extent_m: float) -> float:
    """How far above an object's centre the grasp point has to sit so the
    fingers end level with the object's bottom face rather than inside its
    support: their reach past the grasp point along the hand's own pointing
    axis, less the object's half height, plus a 2 mm allowance."""

    facing = _hand_facing(manifest, effector)
    if facing is None:
        return 0.0
    data = mujoco.MjData(model)
    data.qpos[:] = _scene_rest_qpos(model, manifest)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, grasp_site)
    if site < 0:
        return 0.0
    origin = np.array(data.site_xpos[site], dtype=float)
    direction = np.array(data.site_xmat[site], dtype=float).reshape(3, 3) @ facing
    below = _extent_along(model, data, effector.member_bodies, origin, direction)
    return max(below - half_extent_m, 0.0) + PLACE_STANDOFF_M


def _grasp_offset_m(manifest: RobotAssetManifestV1, effector: EffectorV1) -> float:
    """How far the grasp point lies beyond the grasp centre along the hand."""

    def find(semantic):
        return next((s for s in manifest.morphology.sites if s.semantic is semantic and s.name.startswith(effector.chain_id)), None)

    point, centre = find(SiteSemantic.GRASP_POINT), find(SiteSemantic.GRASP_CENTER)
    if point is None or centre is None:
        return 0.0
    return float(np.linalg.norm(np.array([point.position_m.x - centre.position_m.x, point.position_m.y - centre.position_m.y, point.position_m.z - centre.position_m.z])))


def _refused(violations: list[TransferViolation], refusal: TransferViolation | None, policy: dict) -> TransferResult:
    return TransferResult(
        certified=False, violations=tuple([*violations, *([refusal] if refusal is not None else [])]), phases=(), duration_s=0.0, lift_height_m=0.0,
        hold_s=0.0, carry_offset_max_m=0.0, peak_force_n=0.0, max_penetration_m=0.0, opposition_achieved=False,
        placement_dwell_s=0.0, placement_success=False, placed_inside=False, released=True,
        unexpected_contacts=(), robot_fixture_contacts=(), collision_policy=policy,
    )
