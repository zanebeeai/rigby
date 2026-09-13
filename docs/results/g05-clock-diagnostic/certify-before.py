"""Simulate a planned trajectory and decide whether it may be certified.

The gate set here is not the v2 set with the humanoid parts deleted. It is the
set that describes a bolted-down arm. ``FALL``, ``PELVIS_DRIFT``, ``FOOT_SLIP``
and ``SupportFoot`` have no referent on a robot with no feet, and keeping them
would mean either passing them vacuously -- which reads as evidence and is not --
or failing every robot for lacking a pelvis.

What replaces them is the claim that actually matters for a fixed base:

``BASE_DRIFT``      the base did not move. A bolted arm that shifts is either
                    mis-modelled or being pushed by its own motion.
``TRACKING_ERROR``  the simulated arm went where the plan said. This is the
                    gate that keeps the library honest: a plan that compiles and
                    a robot that follows it are different claims, and only the
                    second one is worth storing.
``SELF_COLLISION``  no two non-adjacent links passed through each other.

Simulation uses :class:`~rigby_general.gates.control.InertiaScaledController`
rather than the v2 controller. The v2 *runtime* is unusable here for a structural
reason -- it requires a free root joint to track and a fixed-base robot has none
-- and the v2 *controller* for a subtler one: its single global proportional gain
is tuned for a 74 kg humanoid, and across this zoo the effective joint inertias
span four orders of magnitude. See that module for the measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import mujoco
import numpy as np
from rigby_core.motion.trajectory import CandidateTrajectoryV1

from ..contracts import RobotAssetManifestV1
from .control import ComputedTorqueController, ControllerConfig


PHYSICS_HZ = 240


class GateCode(StrEnum):
    NONFINITE_TRACE = "nonfinite_trace"
    JOINT_POSITION_LIMIT = "joint_position_limit"
    VELOCITY_LIMIT = "velocity_limit"
    ACTUATOR_EFFORT_LIMIT = "actuator_effort_limit"
    BASE_DRIFT = "base_drift"
    SELF_COLLISION = "self_collision"
    TRACKING_ERROR = "tracking_error"
    REPEAT_DISAGREEMENT = "repeat_disagreement"


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """Thresholds, all of them relative to something measured about the robot.

    ``max_tracking_error_m`` is the exception and is deliberately absolute-ish:
    it scales with the robot's own characteristic link length, so a desktop arm
    is not judged by the standard of a two-metre one.
    """

    max_base_drift_m: float = 1e-4
    tracking_error_fraction: float = 0.035
    """Of the chain's measured reach radius.

    Reach is the right denominator because it is the scale the motion spans. A
    characteristic *link* length is an arbitrary internal dimension -- on a long
    arm made of few long links it is generous, and on a short arm made of many
    short ones it is punishing, for no reason connected to how well the arm
    followed its plan.

    Three and a half percent is where a well-behaved motion actually lands with
    this controller, measured across the zoo, with enough margin that a marginal
    pass is not a coin flip. It is loose enough to admit ordinary servo lag and
    far too tight to admit a motion that went somewhere else: the failures this
    gate caught during development -- a jammed shoulder, a teleport to an
    unreachable first waypoint, a wrist flipping through its null space -- all
    exceeded it by one to two orders of magnitude, not by a few percent."""

    limit_margin: float = 1e-2
    """About half a degree. A joint commanded to sit exactly on its own limit --
    a gripper held fully open, say -- will be pushed a hair past it by contact
    and gravity, and calling that a defect would fail every grasp there is."""
    effort_margin_fraction: float = 0.02
    settle_fraction: float = 0.08
    """Ignore tracking during this opening fraction, while the PD loop catches up
    from rest. Judging the very first instants would fail every motion for the
    controller not having started yet."""

    repeats: int = 3

    def __post_init__(self) -> None:
        if self.repeats < 3:
            raise ValueError(
                "Certification needs at least three repeats; two cannot "
                "distinguish agreement from coincidence"
            )


@dataclass(frozen=True, slots=True)
class GateViolation:
    code: GateCode
    detail: str
    measured: float
    limit: float
    subject: str = ""


@dataclass(frozen=True, slots=True)
class RolloutTrace:
    times_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    demand: np.ndarray
    """Torque the controller asked for, before the actuator limit clamped it.

    Separate from ``ctrl`` on purpose. ``ctrl`` is what the actuator delivered,
    which by construction never exceeds its own limit -- comparing it against
    that limit is a test that cannot fail, and for a long time did not.
    """
    tracking_error_m: np.ndarray
    base_drift_m: float
    unexpected_contacts: tuple[tuple[str, str], ...]

    def content_hash(self) -> str:
        from rigby_core.hashing import sha256_bytes

        payload = b"".join(
            np.ascontiguousarray(array, dtype=np.float64).tobytes()
            for array in (self.times_s, self.qpos, self.qvel, self.ctrl)
        )
        return sha256_bytes(payload)


@dataclass(frozen=True, slots=True)
class CertificationResult:
    certified: bool
    violations: tuple[GateViolation, ...]
    trace: RolloutTrace
    repeats: int
    replay_hashes: tuple[str, ...]

    @property
    def failed_gate(self) -> str | None:
        return self.violations[0].code.value if self.violations else None


def simulate(
    model: mujoco.MjModel,
    manifest: RobotAssetManifestV1,
    trajectory: CandidateTrajectoryV1,
    *,
    site_name: str | None = None,
) -> RolloutTrace:
    """Roll the planned trajectory forward under the certified controller."""

    controller = ComputedTorqueController(model, ControllerConfig())
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(manifest.rest_qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, manifest.morphology.base_body
    )
    base_start = np.array(data.xpos[base_id], dtype=float)

    site_id = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_name
        else -1
    )

    adjacency = {
        (min(body, int(model.body_parentid[body])), max(body, int(model.body_parentid[body])))
        for body in range(1, model.nbody)
    }
    # Pairs the manifest declares un-gateable. Ingest records a pair here when no
    # joint motion can separate the two links -- their collision hulls are
    # authored intersecting, which is ordinary in real robot descriptions. Such a
    # pair is in contact in every frame of every motion, so gating it does not
    # detect a collision, it just rejects the robot.
    for first_name, second_name in manifest.adjacent_collision_exclusions:
        first_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first_name)
        second_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, second_name)
        if first_id >= 0 and second_id >= 0:
            adjacency.add((min(first_id, second_id), max(first_id, second_id)))

    duration = float(trajectory.times_s[-1])
    steps = max(1, int(round(duration * PHYSICS_HZ)))
    dt = 1.0 / PHYSICS_HZ

    times = np.zeros(steps + 1, dtype=float)
    qpos = np.zeros((steps + 1, model.nq), dtype=float)
    qvel = np.zeros((steps + 1, model.nv), dtype=float)
    ctrl = np.zeros((steps + 1, model.nu), dtype=float)
    demand = np.zeros((steps + 1, model.nu), dtype=float)
    tracking = np.zeros(steps + 1, dtype=float)
    base_drift = 0.0
    contacts: set[tuple[str, str]] = set()

    scratch = mujoco.MjData(model)

    for step in range(steps + 1):
        now = min(step * dt, duration)
        target = trajectory.sample(now)
        command = controller.compute(data, target)

        times[step] = now
        qpos[step] = data.qpos
        qvel[step] = data.qvel
        ctrl[step] = command
        demand[step] = controller.last_demand

        if site_id >= 0:
            # Where the plan wanted the effector, versus where it actually is.
            scratch.qpos[:] = target.qpos
            mujoco.mj_kinematics(model, scratch)
            tracking[step] = float(
                np.linalg.norm(
                    np.array(scratch.site_xpos[site_id]) - np.array(data.site_xpos[site_id])
                )
            )

        base_drift = max(
            base_drift,
            float(np.linalg.norm(np.array(data.xpos[base_id], dtype=float) - base_start)),
        )

        for index in range(data.ncon):
            contact = data.contact[index]
            first = int(model.geom_bodyid[contact.geom1])
            second = int(model.geom_bodyid[contact.geom2])
            pair = (min(first, second), max(first, second))
            if pair in adjacency or first == second:
                continue
            contacts.add(
                tuple(
                    sorted(
                        (
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first) or "?",
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second) or "?",
                        )
                    )
                )
            )

        if step < steps:
            data.ctrl[:] = command
            mujoco.mj_step(model, data)

    return RolloutTrace(
        times_s=times,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        demand=demand,
        tracking_error_m=tracking,
        base_drift_m=base_drift,
        unexpected_contacts=tuple(sorted(contacts)),
    )


def evaluate_gates(
    model: mujoco.MjModel,
    manifest: RobotAssetManifestV1,
    trace: RolloutTrace,
    policy: GatePolicy,
) -> tuple[GateViolation, ...]:
    violations: list[GateViolation] = []

    for name, array in (
        ("qpos", trace.qpos),
        ("qvel", trace.qvel),
        ("ctrl", trace.ctrl),
        ("demand", trace.demand),
    ):
        if not np.all(np.isfinite(array)):
            violations.append(
                GateViolation(
                    GateCode.NONFINITE_TRACE,
                    f"{name} contains a non-finite value",
                    measured=float("nan"),
                    limit=0.0,
                    subject=name,
                )
            )
    if violations:
        return tuple(violations)

    for dof in manifest.dofs:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, dof.joint)
        if joint < 0:  # pragma: no cover - manifest and model agree by construction
            continue
        address = int(model.jnt_qposadr[joint])
        values = trace.qpos[:, address]
        low = float(values.min())
        high = float(values.max())
        if low < dof.minimum - policy.limit_margin:
            violations.append(
                GateViolation(
                    GateCode.JOINT_POSITION_LIMIT,
                    f"{dof.name} went below its lower limit",
                    measured=low,
                    limit=dof.minimum,
                    subject=dof.name,
                )
            )
        if high > dof.maximum + policy.limit_margin:
            violations.append(
                GateViolation(
                    GateCode.JOINT_POSITION_LIMIT,
                    f"{dof.name} went above its upper limit",
                    measured=high,
                    limit=dof.maximum,
                    subject=dof.name,
                )
            )

        dof_address = int(model.jnt_dofadr[joint])
        peak = float(np.abs(trace.qvel[:, dof_address]).max())
        if peak > dof.velocity_limit * (1.0 + policy.effort_margin_fraction):
            violations.append(
                GateViolation(
                    GateCode.VELOCITY_LIMIT,
                    f"{dof.name} exceeded its velocity limit",
                    measured=peak,
                    limit=dof.velocity_limit,
                    subject=dof.name,
                )
            )

    for index in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or f"u{index}"
        # An unlimited actuator is one whose source declared no torque limit.
        # There is nothing to test it against, and inventing a threshold here
        # would quietly turn "unknown" into "passed".
        if not model.actuator_forcelimited[index]:
            continue
        peak = float(np.abs(trace.demand[:, index]).max())
        limit = float(model.actuator_forcerange[index][1])
        if limit > 0.0 and peak > limit * (1.0 + policy.effort_margin_fraction):
            share = float((np.abs(trace.demand[:, index]) > limit).mean())
            violations.append(
                GateViolation(
                    GateCode.ACTUATOR_EFFORT_LIMIT,
                    f"{name} demanded more force than it has, for "
                    f"{share * 100.0:.0f}% of the motion",
                    measured=peak,
                    limit=limit,
                    subject=name,
                )
            )

    if trace.base_drift_m > policy.max_base_drift_m:
        violations.append(
            GateViolation(
                GateCode.BASE_DRIFT,
                "the base moved; a bolted-down arm must not",
                measured=trace.base_drift_m,
                limit=policy.max_base_drift_m,
                subject=manifest.morphology.base_body,
            )
        )

    if trace.unexpected_contacts:
        first, second = trace.unexpected_contacts[0]
        violations.append(
            GateViolation(
                GateCode.SELF_COLLISION,
                f"{first} and {second} collided",
                measured=float(len(trace.unexpected_contacts)),
                limit=0.0,
                subject=f"{first}|{second}",
            )
        )

    if trace.tracking_error_m.size:
        start = int(policy.settle_fraction * trace.tracking_error_m.size)
        worst = float(trace.tracking_error_m[start:].max())
        allowed = (
            policy.tracking_error_fraction
            * manifest.morphology.scale.reach_radius_m
        )
        if worst > allowed:
            violations.append(
                GateViolation(
                    GateCode.TRACKING_ERROR,
                    "the arm did not go where the plan said it would",
                    measured=worst,
                    limit=allowed,
                    subject="effector",
                )
            )

    return tuple(violations)


def certify(
    model: mujoco.MjModel,
    manifest: RobotAssetManifestV1,
    trajectory: CandidateTrajectoryV1,
    *,
    site_name: str | None = None,
    policy: GatePolicy | None = None,
) -> CertificationResult:
    """Simulate, gate, and repeat -- a result is only certified if all three agree.

    Repeating is not redundancy. A trajectory that certifies once and differs on
    the next run is not reproducible, and a library entry that cannot be
    reproduced is a claim nobody can check.
    """

    resolved = policy or GatePolicy()
    traces = [
        simulate(model, manifest, trajectory, site_name=site_name)
        for _ in range(resolved.repeats)
    ]
    hashes = tuple(trace.content_hash() for trace in traces)

    violations = list(evaluate_gates(model, manifest, traces[0], resolved))
    if len(set(hashes)) != 1:
        violations.append(
            GateViolation(
                GateCode.REPEAT_DISAGREEMENT,
                f"{len(set(hashes))} of {len(hashes)} replays disagreed",
                measured=float(len(set(hashes))),
                limit=1.0,
                subject="replay",
            )
        )

    return CertificationResult(
        certified=not violations,
        violations=tuple(violations),
        trace=traces[0],
        repeats=resolved.repeats,
        replay_hashes=hashes,
    )
