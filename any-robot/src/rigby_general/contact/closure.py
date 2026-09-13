"""Close a gripper on what is actually there, not on a pre-computed pose.

A grasp cannot be planned open-loop. The object's exact position, the friction,
the moment the jaws first touch -- none of these are known until contact happens,
and a trajectory that commands the fingers to a fixed closed pose either stops
short of the object or drives straight through it. This is the reason
``rigby-mjco-sim`` grows a whole ``grip_feedback`` module rather than adding a
few keyframes.

The approach here is that module's, generalised to an arbitrary gripper:

*Drive closure until contact, then hold.* Each closure joint advances toward its
closed limit at a bounded rate until its member reports contact force, then holds
at the force it found. Nothing commands a final finger position, because nothing
knows one.

*Require opposition before declaring a grasp.* Contact on one side is a nudge;
contact on opposing sides is a grip. The opposition groups come from the
morphology measurement, where they were derived from which way each member
travels during closure -- so a two-finger jaw and a three-finger tripod are
handled by the same rule without either being special-cased.

*Never weld.* The v2 non-goals list "no fake grasp attachment, welded object, or
non-contact teleport", and the same applies here: the object is held by measured
contact forces or it is not held.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import mujoco
import numpy as np

from ..contracts import EffectorV1, RobotAssetManifestV1
from ..morphology import measure


CLOSURE_LIMITS_HOLD = True
"""Whether a compiled scene's grip joints get limit constraints that hold
against the closure's own force. The compiler's default limit is soft
(solref 0.02 s): a 60 N finger actuator drove the jaw arm's fingers 1.3 cm
past their range and through one another when a lost hold snapped the jaw
shut, and no later grasp could open them (G15's first campaign). False
reproduces the soft limits for the comparison."""
LIMIT_TIMECONST_S = 0.005
"""The limit constraint's time constant: at least twice the 2 ms step."""
LIMIT_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
OPENING_SIZED = True
"""Whether a jaw opens as wide as the object needs rather than as wide as it
can. A jaw open to its limit is a sweep: the long arm's fingers stand 17 cm
apart across the outer faces, and descending on a 3 cm cube they came down
on the cube 8 cm away (G15's first campaign). With the opening sized from
the object's width, the measured aperture at closed and a clearance per
side, the fingers stand where the grasp needs them and nowhere else. False
reproduces the full opening for the comparison."""


def hold_closure_limits(model: mujoco.MjModel, manifest: RobotAssetManifestV1) -> tuple[str, ...]:
    """Stiffen the limit constraints of every declared grip joint in a
    compiled model so the closure cannot drive its fingers past their range;
    returns the joints changed. A no-op while ``CLOSURE_LIMITS_HOLD`` is off."""

    if not CLOSURE_LIMITS_HOLD:
        return ()
    changed = []
    for effector in manifest.morphology.grasping_effectors:
        for joint_name in effector.grip_joints:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint < 0 or not model.jnt_limited[joint]:
                continue
            model.jnt_solref[joint] = (LIMIT_TIMECONST_S, 1.0)
            model.jnt_solimp[joint] = LIMIT_SOLIMP
            changed.append(joint_name)
    return tuple(changed)


def carry_joint_limits(source: mujoco.MjModel, target: mujoco.MjModel) -> None:
    """Copy every named joint's limit constraint parameters from one compiled
    model to another (a scene recompiled with a body added keeps the limits
    the first compile was given)."""

    for joint in range(source.njnt):
        name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_JOINT, joint)
        other = mujoco.mj_name2id(target, mujoco.mjtObj.mjOBJ_JOINT, name) if name else -1
        if other >= 0:
            target.jnt_solref[other] = source.jnt_solref[joint]
            target.jnt_solimp[other] = source.jnt_solimp[joint]


class GripState(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    HOLDING = "holding"
    RELEASING = "releasing"


TRAVEL_MARGIN = 4.0
"""How much more than the bare minimum a finger is driven with while it travels.

Enough to cover damping and stiction the model does not carry; not so much that
the finger arrives as a hammer."""

TRAVEL_ACCELERATION = 2.0
"""Reference angular acceleration, rad/s^2, for sizing the inertial term."""

UNDECLARED_CEILING_MARGIN = 40.0
"""Stand-in ceiling, in multiples of the measured travel torque, for a joint
that declares no actuator limit at all.

Generous -- a grip legitimately needs far more than it takes to move the finger
-- but finite, which is the whole point."""


@dataclass(frozen=True, slots=True)
class ClosureConfig:
    """How hard to squeeze, derived from the object rather than the gripper.

    The previous policy commanded a fraction of the *actuator's* force limit, on
    the reasoning that absolute newtons would be as body-dependent as absolute
    metres. That instinct is right and was applied to the wrong quantity. How
    hard you must hold something is a fact about the something: its weight, the
    friction at the contact, and how many fingers oppose. A gripper with strong
    actuators does not need to squeeze harder to hold the same block -- it just
    can.

    Measured across the zoo, a fraction of the actuator limit commanded 77x to
    283x the force physics asks for. The one robot that certified was the least
    over-squeezed of the five, at 77x; the others crushed through the block
    (19 mm of penetration on a 40 mm block) or failed for want of a grip.
    """

    closure_rate_per_s: float = 0.55
    """Fraction of the joint's span the finger advances per second while closing.

    Closing is rate-limited *position* control, and holding is force control.
    Driving a force through the approach was what crushed the block: the fingers
    accumulated depth during the travel, and switching to a gentle hold force
    afterwards did not retract them, because force control has nothing pulling
    back. Measured across the zoo, penetration ran 10-22 mm on blocks 18-48 mm
    across. Advancing by position means the finger stops where the object is."""

    closing_force_fraction: float = 0.12
    """Of the weakest closure actuator's limit -- what it takes to *move* the
    finger.

    This one is legitimately gripper-derived, and it is a different quantity from
    the one below. Advancing a finger against its own damping, armature and
    weight is a fact about the mechanism; holding an object once the finger has
    arrived is a fact about the object. Commanding the holding force from the
    start means a light object never gets gripped at all, because 0.6 N will not
    move a jaw that needs 5 N to travel."""

    grip_safety_factor: float = 80.0
    """Multiplier on the statically required force.

    Static equilibrium is the floor, not the answer. The lift accelerates the
    object, MuJoCo's friction cone is a limit rather than a promise, and the
    contact normal is rarely exactly perpendicular to gravity. Eight covers all
    three with room to spare and still lands two orders of magnitude below what
    the old policy asked for."""

    actuator_ceiling_fraction: float = 0.5
    """Never command more than half of what the joint can produce, whatever the
    object asks for. A grip that needs more than this is a grip this gripper
    should be reporting it cannot make."""

    contact_detection_fraction: float = 0.05
    """Of the required grip force -- the threshold at which a member counts as
    touching. Scaled to the grip rather than the actuator, for the same reason
    the grip is."""

    min_contact_force_n: float = 0.005
    release_force_fraction: float = 0.25
    hold_release_s: float = 0.10
    """How long opposition must stay lost before a grip is treated as gone.

    Contact force through a lift is not smooth: the object shifts a little in
    the jaws, a normal dips under the detection threshold for a step or two, and
    without any hysteresis the controller reads that as "no longer holding",
    goes back to CLOSING and advances the fingers into an object it already has.
    Measured on the compact arm, a single attempt flipped state six times and
    spent 20 steps holding; the block was gripped and then worried loose. A grip
    ends when the object is gone, not when one reading dips."""

    hold_duration_s: float = 0.15
    max_penetration_m: float = 0.004
    opening_clearance_m: float | None = 0.015
    """How far each gripping surface stands from the object before the
    closure, when the opening is sized to it (None: open to the limit)."""


@dataclass(frozen=True, slots=True)
class ClosureReport:
    state: GripState
    contacted_members: tuple[str, ...]
    opposition_satisfied: bool
    peak_force_n: float
    max_penetration_m: float
    closure_fraction: float
    required_force_n: float = 0.0
    commanded_force_n: float = 0.0

    @property
    def holding(self) -> bool:
        return self.state is GripState.HOLDING and self.opposition_satisfied


class ClosureController:
    """Drives one gripper's closure joints from measured contact."""

    def __init__(
        self,
        model: mujoco.MjModel,
        manifest: RobotAssetManifestV1,
        effector: EffectorV1,
        *,
        object_geoms: frozenset[str],
        config: ClosureConfig | None = None,
    ) -> None:
        self.config = config or ClosureConfig()
        self._model = model
        self._effector = effector
        self._joints = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in effector.grip_joints
        )
        if any(joint < 0 for joint in self._joints):
            raise KeyError(f"gripper {effector.name!r} names a joint the model lacks")

        self._qpos_adr = np.array(
            [int(model.jnt_qposadr[joint]) for joint in self._joints]
        )
        self._ranges = np.array(
            [
                (float(model.jnt_range[joint][0]), float(model.jnt_range[joint][1]))
                for joint in self._joints
            ]
        )

        self._object_geoms = frozenset(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in object_geoms
        ) - {-1}
        self._member_geoms = {
            body: frozenset(_body_geoms(model, body)) for body in effector.member_bodies
        }
        self._groups = effector.opposition_groups

        opposing_count = max(len(effector.opposition_groups), 2)
        payload_kg = float(manifest.morphology.scale.payload_kg)

        # What it takes to *move the finger*, measured from the mechanism.
        #
        # Gravity and the joint's own inertia are facts about this hand and are
        # already in the model, so they can be asked instead of assumed.
        probe = mujoco.MjData(model)
        probe.qpos[:] = measure.neutral_qpos(model)
        mujoco.mj_forward(model, probe)
        probe.qvel[:] = 0.0
        # One joint at a time. Accelerating every degree of freedom at once adds
        # the coupling between them, which a finger closing on its own never
        # pays: it inflated the travel force enough to cost three holds and put
        # the jaw arm through its block.
        probe.qacc[:] = 0.0
        gravity = np.zeros(model.nv, dtype=float)
        mujoco.mj_rne(model, probe, 1, gravity)
        loaded = np.zeros(model.nv, dtype=float)
        needed = 0.0
        for joint in self._joints:
            dof = int(model.jnt_dofadr[joint])
            probe.qacc[:] = 0.0
            probe.qacc[dof] = TRAVEL_ACCELERATION
            mujoco.mj_rne(model, probe, 1, loaded)
            inertial = abs(float(loaded[dof] - gravity[dof]))
            needed = max(needed, abs(float(gravity[dof])) + inertial)
        probe.qacc[:] = 0.0

        # Where nothing is declared, the ceiling is measured rather than
        # invented or abandoned. Substituting 1.0 capped these grippers at half
        # a newton; removing the cap entirely let them squeeze without bound and
        # the solver blew up -- 14 kN of contact through 52 mm of penetration on
        # the Beetlebot. What the mechanism can be asked for is a fact about it,
        # and the same reading that sizes the travel force bounds this too.
        # ...and consistent with what the same robot is credited with carrying.
        #
        # Two fallbacks for undeclared hardware disagreed. Payload comes from
        # the robot's own mass, so the Beetlebot is credited with 97 g; the grip
        # ceiling came from the torque needed to swing its claws, which gave it
        # 0.05 N. A world fitted to the first is impossible for the second: it
        # was handed a 104 mm, 97 g block, asked for 27 N of squeeze, allowed
        # 0.016 N, and never touched the object once in 1441 steps. A hand that
        # can carry a payload can hold it.
        carried = (
            self.config.grip_safety_factor
            * float(payload_kg)
            * 9.81
            / max(1.4 * opposing_count, 1e-6)
        ) / max(self.config.actuator_ceiling_fraction, 1e-6)
        undeclared = max(
            needed * UNDECLARED_CEILING_MARGIN, carried, 1e-6
        )
        forces = [
            float(abs(model.jnt_actfrcrange[joint][1]))
            if model.jnt_actfrclimited[joint]
            else undeclared
            for joint in self._joints
        ]
        self._limits = np.array(forces, dtype=float)
        weakest = max(min(forces), 1e-6)

        # What this particular object needs held. Read off the scene, so it is a
        # fact about the thing being grasped and not about the hand.
        self.object_mass_kg, self.object_friction = _object_properties(
            model, self._object_geoms
        )
        opposing = max(len(effector.opposition_groups), 2)
        required = (
            self.config.grip_safety_factor
            * self.object_mass_kg
            * 9.81
            / max(self.object_friction * opposing, 1e-6)
        )
        self.required_force_n = float(required)
        self.squeeze_force_n = float(
            min(required, self.config.actuator_ceiling_fraction * weakest)
        )
        self.force_saturated = bool(required > self.squeeze_force_n)
        self.contact_force_n = max(
            self.config.min_contact_force_n,
            self.config.contact_detection_fraction * self.squeeze_force_n,
        )
        # A fraction of the declared limit is a fraction of a number nobody
        # checked: the SO-ARM101 declares effort="10" on every joint and the
        # 5-DOF SG90 arm declares 1000, so the "gentle" travel force came out
        # hundreds of newtons and the jaws batted the block across the cell --
        # 34 kN of contact and 129 m of carry offset on one attempt. The
        # declared limit still caps it: an actuator cannot exceed itself.
        self.closing_force_n = float(
            min(
                max(needed * TRAVEL_MARGIN, self.config.min_contact_force_n),
                self.config.closing_force_fraction * weakest,
            )
        )

        # Which end of each joint's range closes the gripper -- measured during
        # the closure sweep, not assumed. The Franka hand closes toward its lower
        # limit, and hardcoding the upper one opens the jaw when it means to grip.
        if effector.closes_toward_upper:
            self._closed_end = self._ranges[:, 1]
            self._open_end = self._ranges[:, 0]
        else:
            self._closed_end = self._ranges[:, 0]
            self._open_end = self._ranges[:, 1]
        self._closing_sign = np.sign(self._closed_end - self._open_end)

        # The jaw opens as wide as the object needs. The aperture at the
        # closed end was measured at ingest (it may be negative, for fingers
        # that interleave); each slide finger moves half the aperture, so
        # the travel from closed that stands the surfaces a clearance off the
        # object's faces follows, clamped to the joint's own range.
        self.opening_sized = False
        self.opening_aperture_m: float | None = None
        width = _object_width(model, self._object_geoms)
        slides = all(int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_SLIDE) for joint in self._joints)
        if OPENING_SIZED and self.config.opening_clearance_m is not None and effector.min_aperture_m is not None and width is not None and slides and len(self._joints) == 2:
            target = width + 2.0 * float(self.config.opening_clearance_m)
            travel = max(0.0, (target - float(effector.min_aperture_m)) / 2.0)
            span = np.abs(self._closed_end - self._open_end)
            travel = np.minimum(travel, span)
            self._open_end = self._closed_end - self._closing_sign * travel
            self.opening_sized = True
            self.opening_aperture_m = float(effector.min_aperture_m) + 2.0 * float(np.min(travel))

        self._closure = 0.0
        self._held_for = 0.0
        self._lost_for = 0.0
        self.state = GripState.OPEN
        # Where the fingers have been *told* to be. Advances while closing,
        # freezes the moment opposition is found, and is what the tracking loop
        # follows until the grip takes over.
        self._commanded = np.array(self._open_end, dtype=float)


    def commanded_qpos(self) -> dict[str, float]:
        """Where the closure joints are being told to go while they travel.

        Only meaningful in OPEN, CLOSING and RELEASING. Once holding, the
        object decides where the fingers are and the force command takes over.
        """

        return {
            name: float(self._commanded[index])
            for index, name in enumerate(self._effector.grip_joints)
        }

    def force_commands(self) -> dict[str, float]:
        """Torque per closure joint: a squeeze, a release, or a gentle opening.

        Commanding a *force* rather than a position is the whole point. A closed
        position command means that the instant a jaw touches, a millimetre of
        position error becomes whatever torque the actuator can produce --
        measured at 50 N against a 7 N target, with the jaws several millimetres
        inside the block. Backing the position off does not fix it; the loop just
        oscillates between crushing and dropping. With a force command the object
        itself decides where the fingers stop, which is what a grasp actually is.
        """

        direction = self._closing_sign
        if self.state is GripState.RELEASING:
            magnitude = -self.config.release_force_fraction * self.closing_force_n
        elif self.state is GripState.OPEN:
            magnitude = -self.config.release_force_fraction * self.closing_force_n * 0.5
        elif self.state is GripState.HOLDING:
            # Arrived. From here the object decides, so squeeze only as hard as
            # the object needs.
            magnitude = self.squeeze_force_n
        else:
            # Still travelling. Drive the finger, not the object.
            magnitude = self.closing_force_n
        return {
            name: float(direction[index] * magnitude)
            for index, name in enumerate(self._effector.grip_joints)
        }

    def observe(self, data: mujoco.MjData) -> tuple[dict[str, float], float, float]:
        """Per-member contact force against the object, plus worst penetration."""

        model = self._model
        forces = {body: 0.0 for body in self._member_geoms}
        penetration = 0.0
        buffer = np.zeros(6, dtype=float)

        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self._object_geoms:
                member_geom = second
            elif second in self._object_geoms:
                member_geom = first
            else:
                continue

            for body, geoms in self._member_geoms.items():
                if member_geom in geoms:
                    mujoco.mj_contactForce(model, data, index, buffer)
                    forces[body] += float(abs(buffer[0]))
                    penetration = max(penetration, max(0.0, -float(contact.dist)))
                    break

        return forces, penetration, max(forces.values(), default=0.0)

    def step(self, data: mujoco.MjData, dt: float, *, closing: bool) -> ClosureReport:
        forces, penetration, peak = self.observe(data)
        contacted = tuple(
            sorted(body for body, force in forces.items() if force >= self.contact_force_n)
        )
        opposition = self._opposition_satisfied(contacted)

        if not closing:
            self.state = GripState.RELEASING
            self._held_for = 0.0
            self._commanded = np.array(self._open_end, dtype=float)
        elif opposition or (
            self.state is GripState.HOLDING
            and self._lost_for < self.config.hold_release_s
        ):
            if opposition:
                self._lost_for = 0.0
            else:
                self._lost_for += dt
            self.state = GripState.HOLDING
            self._held_for += dt
            # Frozen: the object is where the fingers stopped, and commanding
            # them further in is how a grasp becomes a crush.
        else:
            self.state = GripState.CLOSING
            self._held_for = 0.0
            self._lost_for = 0.0
            span = self._closed_end - self._open_end
            step = self.config.closure_rate_per_s * dt * span
            self._commanded = self._commanded + step
            beyond = (self._commanded - self._closed_end) * np.sign(span)
            self._commanded = np.where(
                beyond > 0.0, self._closed_end, self._commanded
            )
        # Reported rather than commanded: with force control the fingers sit
        # wherever the object stopped them, and that is the measurement.
        span = self._closed_end - self._open_end
        travelled = np.asarray(data.qpos, dtype=float)[self._qpos_adr] - self._open_end
        closed = float(
            np.mean(np.abs(travelled / np.where(np.abs(span) < 1e-9, 1.0, span)))
        )

        return ClosureReport(
            state=self.state,
            contacted_members=contacted,
            opposition_satisfied=opposition,
            peak_force_n=peak,
            required_force_n=self.required_force_n,
            commanded_force_n=self.squeeze_force_n,
            max_penetration_m=penetration,
            closure_fraction=closed,
        )

    def _opposition_satisfied(self, contacted: tuple[str, ...]) -> bool:
        """Contact on one side is a nudge; contact on both is a grip."""

        if not self._groups:
            return False
        touched = set(contacted)
        return all(bool(touched & set(group)) for group in self._groups)

    @property
    def settled(self) -> bool:
        return self._held_for >= self.config.hold_duration_s


def _object_width(model: mujoco.MjModel, object_geoms: frozenset[int]) -> float | None:
    """The object's width across its widest horizontal box extent, or None
    when it is not made of boxes: what a jaw has to open past."""

    widths = []
    for geom in object_geoms:
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            return None
        widths.append(2.0 * float(max(model.geom_size[geom][0], model.geom_size[geom][1])))
    return max(widths) if widths else None


def _object_properties(
    model: mujoco.MjModel, object_geoms: frozenset[int]
) -> tuple[float, float]:
    """The grasped object's mass and sliding friction, read from the scene.

    Mass is summed over the bodies that own the object's geoms, so a multi-part
    object is weighed whole. Friction is the smallest of its geoms' sliding
    coefficients -- the slipperiest surface is the one that decides whether a
    grip holds.
    """

    if not object_geoms:
        return 0.0, 1.0
    bodies = {int(model.geom_bodyid[geom]) for geom in object_geoms}
    mass = sum(float(model.body_mass[body]) for body in bodies)
    friction = min(float(model.geom_friction[geom][0]) for geom in object_geoms)
    return mass, max(friction, 1e-6)


def _body_geoms(model: mujoco.MjModel, body_name: str) -> tuple[int, ...]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body < 0:
        return ()
    start = int(model.body_geomadr[body])
    count = int(model.body_geomnum[body])
    return tuple(range(start, start + count)) if start >= 0 else ()
