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

import mujoco
import numpy as np

from ..contracts import EffectorV1, RobotAssetManifestV1
from ..gates.control import ComputedTorqueController, ControllerConfig
from ..grounding import ik
from ..grounding.workspace import WorkspaceFrame
from ..scenes.block import GraspScene, block_qpos_address


PHYSICS_HZ = 240

# Phase durations as fractions of the whole attempt. Closure gets the largest
# share because it is the only phase that waits on something it cannot predict.
APPROACH_FRACTION = 0.28
DESCEND_FRACTION = 0.18
CLOSE_FRACTION = 0.24
LIFT_FRACTION = 0.30

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


def _waypoints(scene: GraspScene, frame: WorkspaceFrame) -> list[np.ndarray]:
    """Home, above the block, onto it, and back up with it."""

    block = scene.block_position_m
    above = np.array([block[0], block[1], scene.approach_height_m])
    at_block = np.array([block[0], block[1], block[2]])
    lifted = np.array(
        [
            block[0],
            block[1],
            block[2] + LIFT_HEIGHT_FRACTION * scene.block_half_extent_m * 2.0,
        ]
    )
    return [frame.home, above, at_block, lifted]


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

    points = ik.densify(_waypoints(scene, frame), per_span=6)
    try:
        solution = ik.solve_site_path(
            model, frame.figure_site, arm_joints, np.asarray(points), seed_qpos=rest
        )
    except ik.IkFailure as error:
        return GraspResult(
            certified=False,
            violations=(
                GraspViolation("unreachable_object", str(error), error.residual_m, 0.0),
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

    steps = int(duration_s * PHYSICS_HZ)
    dt = 1.0 / PHYSICS_HZ
    approach_end = int(steps * APPROACH_FRACTION)
    descend_end = approach_end + int(steps * DESCEND_FRACTION)
    close_end = descend_end + int(steps * CLOSE_FRACTION)

    # The arm follows the solved path over approach and descent, holds still
    # while the gripper closes, then follows it back up.
    arm_schedule = _arm_schedule(
        solution.qpos[:, arm_adr], steps, approach_end, descend_end, close_end
    )

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

    target = mujoco.MjData(model)
    for step in range(steps + 1):
        closing = step >= descend_end
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
        target.qpos[arm_adr] = arm_schedule[min(step, steps - 1)]
        # The gripper's joints are targeted at their *current* value, so the
        # tracking loop contributes nothing there and the force command below is
        # the only thing driving them.
        target.qpos[grip_adr] = data.qpos[grip_adr]
        target.qpos[block_adr : block_adr + 7] = data.qpos[block_adr : block_adr + 7]

        from rigby_v2.simulation.controller import ControlTarget

        command = controller.compute(
            data,
            ControlTarget(
                qpos=np.array(target.qpos),
                qvel=np.zeros(model.nv),
                qacc=np.zeros(model.nv),
            ),
        )
        for name, force in closure.force_commands().items():
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
    path: np.ndarray, steps: int, approach_end: int, descend_end: int, close_end: int
) -> np.ndarray:
    """Interpolate the solved path across the phases, holding during closure."""

    count = len(path)
    approach_share = max(1, int(count * 0.5))
    schedule = np.zeros((steps, path.shape[1]), dtype=float)

    for step in range(steps):
        if step < approach_end:
            fraction = step / max(approach_end, 1)
            index = fraction * (approach_share - 1)
        elif step < descend_end:
            fraction = (step - approach_end) / max(descend_end - approach_end, 1)
            index = (approach_share - 1) + fraction * (count - 1 - approach_share + 1) * 0.6
        elif step < close_end:
            index = (approach_share - 1) + (count - 1 - approach_share + 1) * 0.6
        else:
            fraction = (step - close_end) / max(steps - close_end, 1)
            held = (approach_share - 1) + (count - 1 - approach_share + 1) * 0.6
            index = held + fraction * (count - 1 - held)

        low = int(np.clip(np.floor(index), 0, count - 1))
        high = int(np.clip(low + 1, 0, count - 1))
        blend = float(np.clip(index - low, 0.0, 1.0))
        schedule[step] = (1.0 - blend) * path[low] + blend * path[high]
    return schedule


def _scene_rest_qpos(
    model: mujoco.MjModel, manifest: RobotAssetManifestV1
) -> np.ndarray:
    """The robot's rest pose, with the scene's own state left as compiled."""

    qpos = np.array(model.qpos0, dtype=float)
    for dof in manifest.dofs:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)
        if joint < 0:  # pragma: no cover - defensive
            continue
        address = int(model.jnt_qposadr[joint])
        qpos[address] = min(max(0.0, dof.minimum), dof.maximum)
    return qpos
