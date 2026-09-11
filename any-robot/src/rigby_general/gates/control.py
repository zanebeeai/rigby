"""Computed-torque tracking, which needs no per-robot tuning at all.

``rigby_core``'s ``InverseDynamicsPDController`` is structurally morphology-neutral
but carries one scalar proportional gain for every joint, chosen for a 74 kg
humanoid. Across this zoo the effective joint inertias span four orders of
magnitude, and a single number cannot serve both ends: applied to a 2.4 kg
desktop arm it is so overstiff the simulated arm is flung rather than steered
(measured peaks of 191 rad/s against a 3.2 rad/s limit).

The obvious repair -- measure each joint's inertia and size its gains from that
-- turns out to be a trap, and an instructive one. Inertia is not a property of a
joint but of a *configuration*: a base yaw carrying a folded arm has almost none
about its own axis and a great deal with the arm extended. Gains fixed at rest
leave it hopelessly soft everywhere else; gains fixed at the worst case make it
unstable at rest. There is no single correct value to find.

So do not look for one. Computed torque asks the model for the mass matrix at
the configuration the robot is *currently* in, and lets it supply the scaling:

    tau = M(q) * (qacc_desired + 2*zeta*omega*e_dot + omega^2*e) + C(q, qdot) + g(q)

The closed-loop error then obeys ``e_ddot + 2*zeta*omega*e_dot + omega^2*e = 0``
-- a second-order system with the chosen damping and natural frequency,
identical on every joint of every robot regardless of size. The only two
parameters left are dimensionless, and they describe how the *error* should decay
rather than anything about the body. That is the same move the schema layer makes
above: say the thing that does not depend on the body, and let a measurement
supply the rest.

One ``mj_inverse`` call evaluates the whole right-hand side, because inverse
dynamics is exactly ``M(q)*qacc + C(q,qdot) + g(q)`` when handed the current
state and a desired acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import mujoco
import numpy as np

from rigby_core.simulation.controller import ControlTarget

from ..morphology import measure



# Chosen from a measured sweep across the zoo, not from a rule of thumb. Raising
# the bandwidth tightens tracking (40 mm at 6 Hz, 14 mm at 20 Hz) but the loop is
# stepped at 240 Hz, and by 20 Hz omega*dt has reached 0.52 -- far enough into
# the discrete regime that the response overshoots its own reference velocity by
# more than a quarter. 10 Hz keeps omega*dt near 0.26 and lands between the two.
DEFAULT_NATURAL_FREQUENCY_HZ = 14.0
DEFAULT_DAMPING_RATIO = 1.2

TRACKING_REFERENCE_RAD = 0.05
"""The tracking error a joint should be able to answer without clipping.

About three degrees. Swept against the authored worlds at the 6 Hz floor: 0.05
holds thirty-four, 0.1 thirty, 0.3 twenty-eight."""

MIN_BANDWIDTH_RAD_S = 2.0 * np.pi * 6.0
"""However weak an arm measures, a floor on how slack its tracking may go.

Without one the EEZYbotARM -- 0.176 N*m of SG90 -- is slowed until it cannot
hold what it has picked up, losing all three of its holds while the Panda gains
two. Swept: a 6 Hz floor holds thirty-four, 9 Hz thirty-three, 4 Hz thirty-one."""


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """How the tracking error should decay. Both parameters are dimensionless."""

    natural_frequency_hz: float = DEFAULT_NATURAL_FREQUENCY_HZ
    damping_ratio: float = DEFAULT_DAMPING_RATIO

    def __post_init__(self) -> None:
        if self.natural_frequency_hz <= 0.0:
            raise ValueError("natural frequency must be positive")
        if self.damping_ratio <= 0.0:
            raise ValueError("damping ratio must be positive")

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.natural_frequency_hz


class ComputedTorqueController:
    """Model-based tracking with configuration-dependent scaling.

    Inverse dynamics runs in a scratch ``MjData``, following the rule the v2
    controller sets: the live simulation state changes only through ``mj_step``.
    """

    def __init__(
        self, model: mujoco.MjModel, config: ControllerConfig | None = None
    ) -> None:
        if model.nu == 0:
            raise ValueError("model has no actuators to command")

        self._model = model
        self.config = config or ControllerConfig()
        self._scratch = mujoco.MjData(model)

        dofs: list[int] = []
        qpos_adr: list[int] = []
        for actuator in range(model.nu):
            if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(f"actuator {actuator} is not a joint transmission")
            joint = int(model.actuator_trnid[actuator, 0])
            if model.jnt_type[joint] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(f"actuator {actuator} drives a multi-DOF joint")
            dofs.append(int(model.jnt_dofadr[joint]))
            qpos_adr.append(int(model.jnt_qposadr[joint]))

        self._dofs = np.asarray(dofs, dtype=int)
        self._qpos_adr = np.asarray(qpos_adr, dtype=int)
        self._force_low = np.array(model.actuator_forcerange[:, 0], dtype=float)
        self._force_high = np.array(model.actuator_forcerange[:, 1], dtype=float)
        self._limited = np.array(model.actuator_forcelimited, dtype=bool)

        # The bandwidth an arm can actually back up, not the same number for
        # every arm.
        #
        # 14 Hz was swept across the zoo and is right for the bodies it was
        # swept on. It is not a property of the controller, though; it is a
        # claim about the actuators, and `omega**2 * I * error` is the torque
        # that claim costs. On the SO-ARM101 -- 0.63 kg, 10 N*m declared -- it
        # cost 242 N*m, twenty-four times what the arm has. The controller
        # clipped on 1290 of 1441 steps, tracked 139 degrees behind, and closed
        # on air 183 mm from a block admission had already confirmed reachable.
        # Clamping then feeds itself: the clip causes the lag, the lag raises
        # the demand, and the demand clips harder.
        #
        # So the gain is priced against the torque available. Only downward: an
        # arm that can back 14 Hz keeps it.
        self.omega = self.config.omega
        reference = np.zeros(model.nv, dtype=float)
        probe = mujoco.MjData(model)
        probe.qpos[:] = measure.neutral_qpos(model)
        mujoco.mj_forward(model, probe)
        probe.qvel[:] = 0.0
        for index, dof in enumerate(self._dofs):
            if not self._limited[index]:
                continue
            ceiling = abs(float(self._force_high[index]))
            if ceiling <= 0.0:
                continue
            probe.qacc[:] = 0.0
            probe.qacc[int(dof)] = 1.0
            mujoco.mj_rne(model, probe, 0, reference)
            inertia = abs(float(reference[int(dof)]))
            if inertia <= 1e-12:
                continue
            affordable = math.sqrt(
                ceiling / (inertia * TRACKING_REFERENCE_RAD)
            )
            self.omega = min(self.omega, max(affordable, MIN_BANDWIDTH_RAD_S))
        probe.qacc[:] = 0.0

        self.last_demand = np.zeros(model.nu, dtype=float)
        """Torque asked for on the last ``compute``, before clamping."""

    def compute(self, data: mujoco.MjData, target: ControlTarget) -> np.ndarray:
        model = self._model
        omega = self.omega
        damping = 2.0 * self.config.damping_ratio * omega

        position_error = np.zeros(model.nv, dtype=float)
        position_error[self._dofs] = (
            np.asarray(target.qpos, dtype=float)[self._qpos_adr]
            - np.asarray(data.qpos, dtype=float)[self._qpos_adr]
        )
        velocity_error = np.zeros(model.nv, dtype=float)
        velocity_error[self._dofs] = (
            np.asarray(target.qvel, dtype=float)[self._dofs]
            - np.asarray(data.qvel, dtype=float)[self._dofs]
        )

        desired = (
            np.asarray(target.qacc, dtype=float)
            + damping * velocity_error
            + (omega * omega) * position_error
        )

        # Evaluated at the *current* configuration and velocity, which is what
        # makes the mass matrix supply the right scaling here and now rather than
        # at some pose the robot is not in.
        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = data.qvel
        scratch.qacc[:] = desired
        mujoco.mj_inverse(model, scratch)

        command = np.asarray(scratch.qfrc_inverse, dtype=float)[self._dofs]
        # Kept before the clamp, because the clamp is exactly the information
        # worth having. A caller that only ever sees the clipped value cannot
        # tell a joint comfortably inside its limit from one pinned against it
        # for the whole motion, and those are opposite situations.
        self.last_demand = command.copy()
        return np.where(
            self._limited,
            np.clip(command, self._force_low, self._force_high),
            command,
        )


# Retained under the previous name so existing imports keep working.
InertiaScaledController = ComputedTorqueController
