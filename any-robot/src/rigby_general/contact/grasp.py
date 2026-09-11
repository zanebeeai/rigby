"""Pick a block up, and prove it was picked up rather than attached.

The motion is planned as a schema chain like any other -- approach above the
object, descend onto it, close, lift -- but the closure step cannot be planned,
because nothing knows where the jaws will first touch. So the arm follows a
joint trajectory while the gripper is driven by :mod:`.closure` from measured
contact force, and the two run in the same rollout.

The gates are the point of the exercise. "The gripper closed" and "the object is
held" are different claims, and only the second is worth storing:

``GRASP_NOT_ACHIEVED``  opposing members never both made contact
``OBJECT_NOT_LIFTED``   the block never rose off its support
``OBJECT_DROPPED``      it rose and then fell back
``EXCESSIVE_PENETRATION`` the jaws were inside it rather than around it
``HIDDEN_WELD``         the model contains an equality constraint

The last is structural and is checked even though this package never creates one.
The v2 non-goals list "no fake grasp attachment, welded object, or non-contact
teleport" -- and a weld is exactly what makes a broken grasp look perfect, so it
is worth asserting rather than assuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import mujoco
import numpy as np

from ..contracts import EffectorV1, RobotAssetManifestV1, SiteSemantic
from ..gates.control import ComputedTorqueController, ControllerConfig
from ..grounding import ik
from ..morphology import measure
from ..grounding.workspace import WorkspaceFrame
from ..scenes.block import GraspScene, block_qpos_address


SETTLE_STEPS = 60
"""Steps of physics run before the attempt, so the scene starts at rest.

A quarter of a second. Enough that an authored object has come to rest on what
it was placed on; short of the 240 that let objects roll far enough to change
the task (thirty-two holds against thirty-seven)."""

PHYSICS_HZ = 240

# Phase durations as fractions of the whole attempt. Closure gets the largest
# share because it is the only phase that waits on something it cannot predict.
APPROACH_FRACTION = 0.28
DESCEND_FRACTION = 0.18
CLOSE_FRACTION = 0.24
LIFT_FRACTION = 0.30

TRAVERSE_MARGIN = 3.0
"""How much longer than the bare joint-speed minimum a motion is given.

The path is not traversed one joint at a time and the phases overlap, so the
sum of per-joint travel over the fastest each may turn is a floor rather than a
schedule."""

MAX_DURATION_S = 30.0
"""A ceiling, so an arm that measures as very slow does not run forever."""

FACING_GAIN = 0.05

"""How hard the seed is nudged toward closing across a face."""
"""How hard the seed is nudged toward facing its approach.

Small on purpose, and swept rather than argued for: 0.4 and 0.15 both come out
behind, 0.05 takes the authored worlds from two held to three. It is a bias on
the seed, not an objective in the solve -- the path solver already pulls toward
its seed in the null space, so a light touch here is carried along the whole
trajectory, while anything heavier fights the position task it is supposed to
be deferring to."""

LIFT_HEIGHT_FRACTION = 2.5
"""Of the block's half-extent: high enough that a lift is unambiguous."""

MIN_LIFT_FRACTION = 0.8

CARRY_OFFSET_FRACTION = 2.5
"""Of the gripper's aperture. Generous on purpose -- the grasp centre sits on
the palm and the block's origin is at its own centre, so even a perfect grip
leaves a real offset between them. What this rejects is not an imperfect hold
but an object that has departed."""


@dataclass(frozen=True, slots=True)
class GraspViolation:
    code: str
    detail: str
    measured: float
    limit: float


@dataclass(frozen=True, slots=True)
class GraspResult:
    certified: bool
    violations: tuple[GraspViolation, ...]
    lift_height_m: float
    final_height_m: float
    peak_force_n: float
    max_penetration_m: float
    opposition_achieved: bool
    duration_s: float
    qpos: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    times_s: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    carry_offset_m: float = 0.0


    @property
    def failed_gate(self) -> str | None:
        return self.violations[0].code if self.violations else None


def _waypoints(
    scene: GraspScene, frame: WorkspaceFrame, standoff_m: float = 0.0
) -> list[np.ndarray]:
    """Home, above the block, onto it, and back up with it.

    ``standoff_m`` is how far short of the block's centre the descent stops. It
    is zero for the derived scene; see the sweep recorded at the call site. The
    environment path uses it, where object heights are authored rather than
    derived from the jaw.
    """

    block = scene.block_position_m
    above = np.array([block[0], block[1], scene.approach_height_m])
    at_block = np.array([block[0], block[1], block[2] + standoff_m])
    # Lift generously, then pull the target back inside the measured envelope.
    # The two constraints pull opposite ways and both are real: a lift that only
    # just clears the required height fails whenever tracking lags, and a lift
    # commanded past the arm's reach fails outright. Swept on the authored
    # worlds, raising the multiple from 2.5 to 5.0 took held grasps from 1 of 13
    # to 3 of 13; unclamped, the same change cost the derived probe its only
    # working grasp, because that block sits at 0.55 of reach and 5x its height
    # is outside the envelope. Clamping keeps both.
    lifted = frame.clamp_rising(
        np.array(
            [
                block[0],
                block[1],
                block[2] + scene.lift_fraction * scene.block_half_extent_m * 2.0,
            ]
        )
    )
    # The hover is derived from the object's size, not requested by anyone, so
    # it gets pulled inside reach the same way the lift does. Left unclamped it
    # is the waypoint that fails: the object itself is admitted -- it passed the
    # envelope check before anything moved -- and then the path to it is refused
    # for a point above it that nobody asked for. Twelve attempts died that way,
    # `unreachable_object` on robots that can plainly reach the block.
    return [frame.home, frame.clamp_rising(above), at_block, lifted]


def _body_geoms(model: "mujoco.MjModel", body_name: str) -> tuple[int, ...]:
    """Collidable geoms of a named body, or all of them if none collide."""

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body < 0:
        return ()
    start = int(model.body_geomadr[body])
    count = int(model.body_geomnum[body])
    geoms = tuple(range(start, start + count))
    solid = tuple(g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g])
    return solid or geoms


def _hand_facing(manifest, effector) -> "np.ndarray | None":
    """Which way the hand faces, in the frame of the site the IK drives.

    From the grasp centre to the grasp point: the wrist at one end, the place an
    object is held at the other. Measured per robot from geometry the analyser
    already derived, rather than read off an axis of a frame nobody chose.
    """

    def _find(semantic):
        return next(
            (
                s
                for s in manifest.morphology.sites
                if s.semantic is semantic and s.name.startswith(effector.chain_id)
            ),
            None,
        )

    point, centre = _find(SiteSemantic.GRASP_POINT), _find(SiteSemantic.GRASP_CENTER)
    if point is None or centre is None:
        return None
    offset = np.array(
        [
            point.position_m.x - centre.position_m.x,
            point.position_m.y - centre.position_m.y,
            point.position_m.z - centre.position_m.z,
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(offset))
    return offset / norm if norm > 1e-6 else None


def _face_the_approach(
    model,
    site_name: str,
    joint_names: tuple[str, ...],
    seed: np.ndarray,
    facing_local: np.ndarray,
    approach_axis: np.ndarray,
) -> np.ndarray:
    """Turn the wrist toward the approach without moving the hand.

    Steps taken purely in the null space of the position task, so the point the
    solver was asked for stays where it is and only the spare freedom is spent.
    The result seeds the real solve: the path solver already pulls toward its
    seed in the null space, so biasing the seed biases the whole trajectory
    without adding a second objective to a loop whose convergence is delicate --
    every attempt to do that directly, co-equal or null-space, at weights from
    0.05 to 0.5, made things worse.

    Why it is needed: the point being solved for is offset from the wrist *in
    the hand's own frame* -- the KUKA's grasp point lies 64 mm along its palm's
    +X, the jaw arm's 113 mm along its palm's +Z. With the facing unconstrained
    that offset lands wherever the wrist happens to point.
    """

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names
    ]
    dof_adr = [int(model.jnt_dofadr[j]) for j in joint_ids]
    qpos_adr = [int(model.jnt_qposadr[j]) for j in joint_ids]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0 or not joint_ids:
        return seed

    data = mujoco.MjData(model)
    current = np.array(seed, dtype=float)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    lower = np.array([model.jnt_range[j][0] for j in joint_ids])
    upper = np.array([model.jnt_range[j][1] for j in joint_ids])

    for _ in range(60):
        data.qpos[:] = current
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rotation = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        twist = np.cross(rotation @ facing_local, approach_axis)
        if float(np.linalg.norm(twist)) < 1e-3:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        position = jacp[:, dof_adr]
        projector = np.eye(len(joint_ids)) - np.linalg.pinv(
            position, rcond=1e-3
        ) @ position
        step = projector @ (
            FACING_GAIN * (np.linalg.pinv(jacr[:, dof_adr], rcond=1e-3) @ twist)
        )
        step = np.clip(step, -0.1, 0.1)
        if float(np.max(np.abs(step))) < 1e-6:
            break
        current = current.copy()
        current[qpos_adr] = np.clip(current[qpos_adr] + step, lower, upper)
    return current


def attempt_grasp(
    manifest: RobotAssetManifestV1,
    scene: GraspScene,
    effector: EffectorV1,
    frame: WorkspaceFrame,
    *,
    duration_s: float = 6.0,
) -> GraspResult:
    """Run one pick attempt and gate the result."""

    from .closure import ClosureController, GripState

    model = scene.model
    violations: list[GraspViolation] = []

    if model.neq > 0:
        violations.append(
            GraspViolation(
                "hidden_weld",
                "the scene contains an equality constraint; a grasp proven with "
                "one proves nothing",
                measured=float(model.neq),
                limit=0.0,
            )
        )

    arm_joints = ik.chain_joint_names(
        model,
        frame.figure_site,
        exclude=frozenset(effector.grip_joints),
    )
    rest = _scene_rest_qpos(model, manifest)

    # KNOWN DEFECT, and the cause of most of the remaining failures. This solves
    # the approach for the grasp site's *position* only. Nothing constrains the
    # wrist, so the hand arrives at the right point in whatever orientation the
    # solver reached it in -- measured on the authored worlds, the jaw shows up
    # 166 degrees from straight down, which is to say very nearly upside-down,
    # with the grasp centre 91 to 200 mm off the block. A jaw pointing at the
    # ceiling cannot close on something under it, and no amount of closure or
    # lift tuning repairs that.
    #
    # The fix is an orientation objective on the approach axis -- v2's refinement
    # layer already has SiteOrientationObjective -- so the hand is required to
    # point along the approach direction as well as arrive at the point. Not done
    # here; it changes every grasp trajectory and wants its own change.
    #
    # Standoff stays at zero, and that is a measured result rather than an
    # oversight. Sweeping it at 0, 0.5, 1.0, 1.5 and 2.0 times the block's
    # half-extent certified 1, 0, 0, 0 and 0 grippers respectively: any daylight
    # left between the jaw and the block means the fingers never reach it. The
    # palm still leads the fingers down, but backing off does not help, because
    # the fingers have to be at the object to hold it.
    # No standoff. The grasp centre now sits halfway along the gripping
    # surfaces rather than at the members' origins, so driving it to the object
    # already puts the object between the jaws -- in whatever orientation the
    # hand happens to arrive in, which a vertical offset here could never do.
    # Solve for the grasp *point* -- where the gripping surfaces meet an object
    # -- while the workspace frame stays anchored on the grasp centre.
    #
    # Those were one site doing both jobs, and it sits on the members' origins,
    # which are their joints: the root of the jaws. Driving it to the block left
    # the block at the base of the fingers, or beneath them on a long-fingered
    # hand, and the fingers pressed down on it instead of closing around it --
    # opposition satisfied, 56 N of contact, and the block still on the table
    # afterwards. Moving the site would have moved every reach envelope and
    # every authored world with it, so morphology now derives both and this
    # picks the one it needs. Robots without a gripper have no grasp point and
    # fall back to the frame's own site.
    solve_site = next(
        (
            site.name
            for site in manifest.morphology.sites
            if site.semantic is SiteSemantic.GRASP_POINT
            and site.name.startswith(effector.chain_id)
        ),
        frame.figure_site,
    )
    waypoints = list(_waypoints(scene, frame))
    # The path starts wherever the solved site already is. `frame.home` is a
    # pose for the frame's own site, and asking a different one to reach it is a
    # target nothing put there -- it read as `unreachable_object` on four
    # robots that can plainly reach the block.
    if solve_site != frame.figure_site:
        home = mujoco.MjData(model)
        home.qpos[:] = rest
        mujoco.mj_kinematics(model, home)
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, solve_site)
        if home_id >= 0:
            waypoints[0] = np.array(home.site_xpos[home_id], dtype=float)
    # Bias the seed so the hand faces its approach; see `_face_the_approach`.
    facing = _hand_facing(manifest, effector)
    seed = rest
    if facing is not None:
        try:
            reached = ik.solve_site_path(
                model,
                solve_site,
                arm_joints,
                np.asarray([waypoints[-2]]),
                seed_qpos=rest,
            )
            seed = _face_the_approach(
                model,
                solve_site,
                arm_joints,
                reached.qpos[0],
                facing,
                -np.asarray(frame.up, dtype=float),
            )

        except ik.IkFailure:
            seed = rest

    points = ik.densify(waypoints, per_span=6)
    # Where the hand is at the object, located in the densified path rather than
    # assumed to sit at a fixed fraction of it.
    grasp_target = np.asarray(waypoints[-2], dtype=float)
    grasp_index = int(
        np.argmin([float(np.linalg.norm(np.asarray(p) - grasp_target)) for p in points])
    )
    try:
        solution = ik.solve_site_path(
            model,
            solve_site,
            arm_joints,
            np.asarray(points),
            seed_qpos=seed,
        )
    except ik.IkFailure as first_error:
        # Retry from the middle of every joint's range before giving up.
        #
        # A rest pose is chosen to be a pose the robot can *hold*, and that is
        # not the same as one it can be solved from. `neutral_qpos` walks out of
        # the clamped-zero pose only until self-penetration clears, which on the
        # Panda is almost immediately: it rests at [0, 0, 0, -0.06, 0, 0, 0],
        # arm straight, which is a singularity. The Jacobian there is rank
        # deficient and the solver diverges -- a 101 mm step comes back with a
        # 703 mm residual, on seven of its eight pairings.
        #
        # The midpoint of the joint box is the furthest a configuration can be
        # from every limit at once, and it is the conventional escape from a
        # singular seed. Costs one extra solve, and only on the path that was
        # about to be refused anyway.
        midpoint = np.array(rest, dtype=float)
        for name in arm_joints:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                continue
            low, high = model.jnt_range[joint]
            midpoint[int(model.jnt_qposadr[joint])] = 0.5 * (float(low) + float(high))
        try:
            solution = ik.solve_site_path(
                model,
                solve_site,
                arm_joints,
                np.asarray(points),
                seed_qpos=midpoint,
            )
        except ik.IkFailure:
            return GraspResult(
                certified=False,
                violations=(
                    GraspViolation(
                        "unreachable_object",
                        str(first_error),
                        first_error.residual_m,
                        0.0,
                    ),
                ),
                lift_height_m=0.0,
                final_height_m=scene.block_position_m[2],
                peak_force_n=0.0,
                max_penetration_m=0.0,
                opposition_achieved=False,
                duration_s=0.0,
            )

    controller = ComputedTorqueController(model, ControllerConfig())
    closure = ClosureController(
        model,
        manifest,
        effector,
        object_geoms=frozenset({"scene_block_geom"}),
    )

    data = mujoco.MjData(model)
    data.qpos[:] = rest
    mujoco.mj_forward(model, data)

    arm_adr = np.array(
        [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in arm_joints
        ]
    )
    grip_adr = np.array(
        [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in effector.grip_joints
        ]
    )
    block_adr = block_qpos_address(model)
    start_height = float(data.qpos[block_adr + 2])

    # Give the motion as long as this arm actually needs for it.
    #
    # The duration was a fixed six seconds whatever the path or the body, so an
    # arm asked to cross more of its workspace than it can in that time is asked
    # for a speed it does not have -- and a stiff tracker answers the resulting
    # error with everything the actuators can produce. The Panda's command was
    # clipped on 76% of steps against limits that are its real ones. Its joints
    # declare how fast they turn; dividing the path they have to cover by that
    # is the time the motion takes, and stretching to it is the difference
    # between a trajectory and a demand.
    travel = np.abs(np.diff(solution.qpos[:, arm_adr], axis=0)).sum(axis=0)
    speeds = np.array(
        [
            max(
                abs(
                    float(
                        next(
                            (d.velocity_limit for d in manifest.dofs if d.joint == name),
                            0.0,
                        )
                        or 0.0
                    )
                ),
                1e-3,
            )
            for name in arm_joints
        ]
    )
    needed = float(np.max(travel / speeds)) * TRAVERSE_MARGIN
    duration_s = max(duration_s, min(needed, MAX_DURATION_S))

    steps = int(duration_s * PHYSICS_HZ)
    dt = 1.0 / PHYSICS_HZ
    approach_end = int(steps * APPROACH_FRACTION)
    descend_end = approach_end + int(steps * DESCEND_FRACTION)
    close_end = descend_end + int(steps * CLOSE_FRACTION)

    # The arm follows the solved path over approach and descent, holds still
    # while the gripper closes, then follows it back up.
    arm_schedule = _arm_schedule(
        solution.qpos[:, arm_adr],
        steps,
        approach_end,
        descend_end,
        close_end,
        grasp_index,
    )

    # Begin on the trajectory, not beside it.
    #
    # The simulation started at the rest pose while the solved path starts
    # wherever the solver put its first waypoint, and the two are not the same
    # once the seed is biased. That difference is a step input, and a stiff
    # tracking gain turns a step into an impossible demand: the Panda's command
    # was clipped against its actuator limits on 100% of steps, overshooting by
    # 1.6e5 N*m, so the arm hung 199 mm below the block having never tracked
    # anything. Placing the arm on the first solved pose costs nothing -- it is
    # the pose the trajectory was going to ask for on the very next tick.
    data.qpos[arm_adr] = arm_schedule[0]
    mujoco.mj_forward(model, data)

    qpos_log = np.zeros((steps + 1, model.nq), dtype=float)
    times = np.zeros(steps + 1, dtype=float)
    peak_force = 0.0
    peak_penetration = 0.0
    opposition = False
    peak_height = start_height
    # How far the block ever gets from the grasp centre once the jaws have it.
    # A carried object stays within the hand; a launched one does not.
    grasp_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, effector_grasp_site(manifest, effector)
    )
    max_carry_offset = 0.0

    # A guarded descent: stop going down the moment the hand touches something,
    # and close from there.
    #
    # The descent drove blindly to its target, and on a hand whose palm leads
    # its fingers that means driving *through* the object: measured on the
    # SO-ARM101, the block departs at step 639 and the jaws do not begin to
    # close until 662, so it is gone 23 steps before anything grips. The gates
    # then report a lift of 1.4 m and a carry offset of 34 m, which is a block
    # batted across the cell rather than dropped.
    #
    # Touch is the honest stopping condition, and it needs nothing about the
    # shape of the hand: whatever the geometry, the moment the object starts
    # moving is the moment continuing to descend can only make it worse.
    contact_hold: int | None = None
    # Where the object was when the guard *armed*, not where it was authored.
    #
    # A scene settles: an object placed on a bench drops the last fraction of a
    # millimetre -- or, for a heavy one on a scaled bench, rather more. Measured
    # from the authored position, that settling is already banked before the
    # guard starts watching, so it fires on its first step and freezes the arm
    # where it stands. The Beetlebot's crate moves 19.5 mm in the first 22 steps
    # of 1441, against a 13 mm threshold; the guard armed at step 403 and tripped
    # at once, and the hand spent the whole attempt hovering 137 mm above a block
    # it never descended to. The heavier the object the worse it is, which is why
    # this hid behind every other failure until the grip was strong enough to be
    # given a heavy one.
    # Let the world come to rest before the attempt is measured.
    #
    # An object authored onto a bench is not quite on it, and it drops the
    # difference under gravity while the arm is still crossing the room. The
    # guard measures the object moving, so that settling is charged to the hand:
    # the Beetlebot's crate travels 19.5 mm in the first 22 steps against a
    # 13 mm threshold, and the guard -- which arms at step 403 -- fires on its
    # very first look and freezes the arm where it stands. It spent every
    # attempt hovering 137 mm above a block it never descended to. Heavier
    # objects settle further, so this stayed hidden until the grippers were
    # given enough force to be handed heavy ones.
    #
    # Re-baselining the guard when it arms also works and costs a hold; letting
    # the scene settle first is the same fix applied to the cause, and a cell
    # that is still moving when the robot is asked to work in it was never the
    # thing being tested.
    for _ in range(SETTLE_STEPS):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    start_position = np.array(data.qpos[block_adr : block_adr + 3], dtype=float)
    # A quarter of the block's half-extent: far enough that settling contact
    # does not trip it, close enough that the object is still where it was.
    # Swept -- 0.05, 0.10 and 0.50 all come out behind.
    nudge_limit = 0.25 * scene.block_half_extent_m

    target = mujoco.MjData(model)
    for step in range(steps + 1):
        if (
            contact_hold is None
            and approach_end <= step < descend_end
            and float(
                np.linalg.norm(
                    np.array(data.qpos[block_adr : block_adr + 3], dtype=float)
                    - start_position
                )
            )
            > nudge_limit
        ):
            contact_hold = step

        closing = step >= descend_end or contact_hold is not None
        report = closure.step(data, dt, closing=closing)
        opposition = opposition or report.opposition_satisfied
        # Measured only while the grip is actually holding. The peak over the
        # whole attempt is dominated by the moment the jaw first touches down --
        # a millisecond impact transient, which is a fact about the approach
        # rather than about the grasp. "Are the jaws inside the object" is a
        # question about the steady state, and asking it of the transient
        # rejects perfectly good grasps for the sound the landing made.
        if report.state is GripState.HOLDING:
            peak_force = max(peak_force, report.peak_force_n)
            peak_penetration = max(peak_penetration, report.max_penetration_m)

        target.qpos[:] = rest
        # Once the descent has been stopped by contact, the arm waits there
        # rather than continuing into a target the object now occupies. It
        # resumes when the close phase begins, so the last of the descent is
        # made against jaws that are already closing rather than against open
        # ones -- pausing on touch and then finishing gently, which is what
        # turned four of these attempts into holds.
        # Lift from where the hand actually is, not from where the descent had
        # planned to end.
        #
        # The guarded descent stops the arm on contact, and the retreat then
        # interpolated from the *planned* grasp pose -- so at the moment the
        # lift begins the target jumps to a pose the arm never reached. Both the
        # Panda and the SO-ARM101 establish a grip during the close (148 and 106
        # steps of it) and lose opposition on every one of the 433 lift steps:
        # the jump shakes the block out before it leaves the bench.
        if contact_hold is None:
            planned = arm_schedule[min(step, steps - 1)]
        elif step < close_end:
            planned = arm_schedule[contact_hold]
        else:
            share = (step - close_end) / max(steps - close_end, 1)
            planned = (1.0 - share) * arm_schedule[contact_hold] + share * (
                arm_schedule[steps - 1]
            )
        target.qpos[arm_adr] = planned
        # On the way in, drive the hand *open* -- as a position, not a nudge.
        #
        # Opening was left to a release force of a quarter of the travel force,
        # and on most of the fleet that is not enough to move the fingers at all
        # within the approach. Measured at the moment the descent begins, the
        # EEZYbotARM arrives with its jaws fully shut, the jaw arm 68% closed,
        # the Beetlebot and compact arm around 40%. A gripper that reaches the
        # block already closed cannot straddle it: it knocks it off the bench
        # instead, which is the 15 m of "lift" and 18 m of carry offset the
        # gates then report as a drop. Opening the hand on the way in is what
        # any real approach does, and it costs nothing to command.
        #
        # Once closing starts the joints are targeted at their current value
        # again, so the force command below is the only thing driving them and
        # the object decides where the fingers stop.
        if closing:
            target.qpos[grip_adr] = data.qpos[grip_adr]
        else:
            commanded = closure.commanded_qpos()
            target.qpos[grip_adr] = [
                commanded.get(name, float(data.qpos[address]))
                for name, address in zip(effector.grip_joints, grip_adr)
            ]
        target.qpos[block_adr : block_adr + 7] = data.qpos[block_adr : block_adr + 7]

        from rigby_core.simulation.controller import ControlTarget

        command = controller.compute(
            data,
            ControlTarget(
                qpos=np.array(target.qpos),
                qvel=np.zeros(model.nv),
                qacc=np.zeros(model.nv),
            ),
        )
        # While opening, the position loop above owns the fingers; a force
        # command here would only fight it.
        for name, force in (closure.force_commands() if closing else {}).items():
            actuator = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor"
            )
            if actuator >= 0:
                command[actuator] = float(
                    np.clip(
                        force,
                        model.actuator_forcerange[actuator][0],
                        model.actuator_forcerange[actuator][1],
                    )
                )

        qpos_log[step] = data.qpos
        times[step] = step * dt
        peak_height = max(peak_height, float(data.qpos[block_adr + 2]))
        if opposition and grasp_site >= 0:
            offset = np.linalg.norm(
                np.array(data.site_xpos[grasp_site], dtype=float)
                - np.array(data.qpos[block_adr : block_adr + 3], dtype=float)
            )
            max_carry_offset = max(max_carry_offset, float(offset))

        if step < steps:
            data.ctrl[:] = command
            mujoco.mj_step(model, data)

    final_height = float(data.qpos[block_adr + 2])
    lift = peak_height - start_height
    required = MIN_LIFT_FRACTION * scene.block_half_extent_m * 2.0

    if not opposition:
        violations.append(
            GraspViolation(
                "grasp_not_achieved",
                "opposing members never both made contact with the block",
                measured=0.0,
                limit=1.0,
            )
        )
    if lift < required:
        violations.append(
            GraspViolation(
                "object_not_lifted",
                "the block never came off its support",
                measured=lift,
                limit=required,
            )
        )
    elif final_height < start_height + required * 0.5:
        violations.append(
            GraspViolation(
                "object_dropped",
                "the block was lifted and then fell",
                measured=final_height - start_height,
                limit=required * 0.5,
            )
        )
    # A launched block satisfies "it went up" perfectly well. The lift gates
    # bound the height from below only, so a block batted across the room by a
    # closing jaw passed them -- 8.4 m of "lift" with zero grip force, certified.
    # What distinguishes carrying from launching is that a carried object stays
    # in the hand, so that is what gets measured.
    carry_limit = CARRY_OFFSET_FRACTION * (effector.max_aperture_m or 0.05)
    if opposition and max_carry_offset > carry_limit:
        violations.append(
            GraspViolation(
                "object_not_carried",
                "the block left the gripper instead of being carried by it",
                measured=max_carry_offset,
                limit=carry_limit,
            )
        )
    if peak_penetration > closure.config.max_penetration_m:
        violations.append(
            GraspViolation(
                "excessive_penetration",
                "the jaws were inside the block rather than around it",
                measured=peak_penetration,
                limit=closure.config.max_penetration_m,
            )
        )

    return GraspResult(
        certified=not violations,
        violations=tuple(violations),
        lift_height_m=lift,
        carry_offset_m=max_carry_offset,
        final_height_m=final_height,
        peak_force_n=peak_force,
        max_penetration_m=peak_penetration,
        opposition_achieved=opposition,
        duration_s=duration_s,
        qpos=qpos_log,
        times_s=times,
    )


def effector_grasp_site(manifest, effector) -> str:
    """The site to measure carriage from: the grasp centre, else the tip."""

    preferred = [
        site.name
        for site in manifest.morphology.sites
        if site.name in effector.site_names
        and site.semantic.value == "grasp_center"
    ]
    if preferred:
        return preferred[0]
    fallback = [
        site.name
        for site in manifest.morphology.sites
        if site.name in effector.site_names and site.semantic.value == "tip"
    ]
    return fallback[0] if fallback else next(iter(effector.site_names))


def _arm_schedule(
    path: np.ndarray,
    steps: int,
    approach_end: int,
    descend_end: int,
    close_end: int,
    grasp_index: int,
) -> np.ndarray:
    """Interpolate the solved path across the phases, holding during closure.

    ``grasp_index`` is where in the solved path the hand is at the object, and
    it is where the jaws close. This used to be 60% of the way from the end of
    the approach to the end of the *whole path* -- and the path ends with the
    lift, so the hold landed around four fifths along, past the object and
    already rising with it. The jaws closed above the block: on the KUKA, 33 mm
    high, its fingers 73 and 84 mm from a block they open 90 mm for, driving
    through their whole travel and touching nothing.
    """

    count = len(path)
    approach_share = max(1, int(count * 0.5))
    held_index = float(np.clip(grasp_index, approach_share - 1, count - 1))
    schedule = np.zeros((steps, path.shape[1]), dtype=float)

    for step in range(steps):
        if step < approach_end:
            fraction = step / max(approach_end, 1)
            index = fraction * (approach_share - 1)
        elif step < descend_end:
            fraction = (step - approach_end) / max(descend_end - approach_end, 1)
            index = (approach_share - 1) + fraction * (held_index - (approach_share - 1))
        elif step < close_end:
            index = held_index
        else:
            fraction = (step - close_end) / max(steps - close_end, 1)
            index = held_index + fraction * (count - 1 - held_index)

        low = int(np.clip(np.floor(index), 0, count - 1))
        high = int(np.clip(low + 1, 0, count - 1))
        blend = float(np.clip(index - low, 0.0, 1.0))
        schedule[step] = (1.0 - blend) * path[low] + blend * path[high]
    return schedule


def _scene_rest_qpos(
    model: mujoco.MjModel, manifest: RobotAssetManifestV1
) -> np.ndarray:
    """The robot's rest pose, with the scene's own state left as compiled.

    The rest pose is *measured*, not clamped from zero.

    This used to set every joint to `min(max(0, lower), upper)` -- the zero pose,
    pushed inside the range -- which is the exact defect ingest already fixes and
    records: the Panda's fourth joint has range [-3.1416, 0], so zero *is* its
    limit and the arm folds through itself. `neutral_qpos` walks toward the
    joint-range midpoint and descends on penetration to find a pose the robot can
    actually hold, and ingest has called it since. The grasp path reimplemented
    the old rule, so every attempt began from the folded pose: the Panda's solver
    could not move off it, failing the second waypoint of nineteen by 709 mm on
    seven of its eight pairings.

    Only the robot's own joints are taken from it. Whatever the scene put in the
    block's free joint stays exactly as compiled.
    """

    measured = measure.neutral_qpos(model)
    qpos = np.array(model.qpos0, dtype=float)
    for dof in manifest.dofs:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)
        if joint < 0:  # pragma: no cover - defensive
            continue
        address = int(model.jnt_qposadr[joint])
        qpos[address] = float(measured[address])
    return qpos
