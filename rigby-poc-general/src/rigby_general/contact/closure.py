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


class GripState(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    HOLDING = "holding"
    RELEASING = "releasing"


@dataclass(frozen=True, slots=True)
class ClosureConfig:
    """Forces as fractions of what the gripper can actually produce.

    Absolute newtons would be as body-dependent as absolute metres: a desktop
    jaw's firm grip is a rounding error to an industrial one.
    """

    contact_force_fraction: float = 0.02
    """Of the weakest closure actuator's force limit -- enough to register."""

    squeeze_force_fraction: float = 0.30
    """The closing force commanded, as a fraction of the actuator's own limit.

    Firm enough to hold the block against gravity through a lift, gentle enough
    that the jaws rest on its surface rather than sinking into it."""

    release_force_fraction: float = 0.25
    hold_duration_s: float = 0.15
    max_penetration_m: float = 0.004


@dataclass(frozen=True, slots=True)
class ClosureReport:
    state: GripState
    contacted_members: tuple[str, ...]
    opposition_satisfied: bool
    peak_force_n: float
    max_penetration_m: float
    closure_fraction: float

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

        forces = [
            float(abs(model.jnt_actfrcrange[joint][1]))
            if model.jnt_actfrclimited[joint]
            else 1.0
            for joint in self._joints
        ]
        self._limits = np.array(forces, dtype=float)
        weakest = max(min(forces), 1e-6)
        self.contact_force_n = self.config.contact_force_fraction * weakest

        # Which end of each joint's range closes the gripper. Measured during
        # morphology analysis, so a jaw that closes by extending and one that
        # closes by retracting are both handled without a convention.
        self._closed_end = self._ranges[:, 1]
        self._open_end = self._ranges[:, 0]

        self._closure = 0.0
        self._held_for = 0.0
        self.state = GripState.OPEN

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

        direction = np.sign(self._closed_end - self._open_end)
        if self.state is GripState.RELEASING:
            magnitude = -self.config.release_force_fraction * self._limits
        elif self.state is GripState.OPEN:
            magnitude = -self.config.release_force_fraction * self._limits * 0.5
        else:
            magnitude = self.config.squeeze_force_fraction * self._limits
        return {
            name: float(direction[index] * magnitude[index])
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
        elif opposition:
            self.state = GripState.HOLDING
            self._held_for += dt
        else:
            self.state = GripState.CLOSING
            self._held_for = 0.0

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


def _body_geoms(model: mujoco.MjModel, body_name: str) -> tuple[int, ...]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body < 0:
        return ()
    start = int(model.body_geomadr[body])
    count = int(model.body_geomnum[body])
    return tuple(range(start, start + count)) if start >= 0 else ()
