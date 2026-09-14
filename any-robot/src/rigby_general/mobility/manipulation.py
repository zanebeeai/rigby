"""A mobile body's manipulator: a numerical reach on the declared grasp site, a jaw that opens and closes, and a hold read from contacts.

The manipulator is whatever the mobility declaration names: the dog's
and the biped's arms (a yaw and two or three pitch hinges under a
parallel jaw on two slides), the octopus's front tentacles (four yaw
and pitch pairs under a pincer on two hinges). The reach is a damped
least-squares solve on the grasp site's position, in the limb's own
joints and nothing else, from the base pose the body has now; the
jaw's targets come from the joint ranges; a hold is two fingers in
contact with the object and the object moving with the site. Nothing
here writes the object's state or the root.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .ingest import MobileBody


@dataclass
class ReachSolution:
    joints: dict[str, float]
    residual_m: float
    iterations: int
    reached: bool


class Manipulator:
    """One declared manipulator on a mobile body, bound to a compiled model (the body alone or the body in a world)."""

    TOLERANCE_M = 0.004
    ITERATIONS = 300

    def __init__(self, body: MobileBody, model: mujoco.MjModel, manipulator: dict) -> None:
        self.body = body
        self.model = model
        self.declaration = manipulator
        self.limb = manipulator["limb"]
        self.grip_joints = tuple(manipulator["grip_joints"])
        self.fingers = tuple(manipulator["fingers"])
        self.joints = tuple(j for j in body.declaration["limbs"][self.limb] if j not in self.grip_joints)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, manipulator["grasp_site"])
        if self.site < 0:
            raise ValueError(f"no grasp site {manipulator['grasp_site']!r}")
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        self.qpos_adr = [int(model.jnt_qposadr[j]) for j in self.joint_ids]
        self.dof_adr = [int(model.jnt_dofadr[j]) for j in self.joint_ids]
        self.ranges = np.array([model.jnt_range[j] for j in self.joint_ids], dtype=float)
        self.grip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.grip_joints]
        self.grip_ranges = np.array([model.jnt_range[j] for j in self.grip_ids], dtype=float)
        self.finger_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f) for f in self.fingers]
        self.finger_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in self.finger_bodies]
        self.actuator_of = {}
        for a in range(model.nu):
            joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a][0]))
            self.actuator_of[joint] = a
        self.slides = bool(model.jnt_type[self.grip_ids[0]] == mujoco.mjtJoint.mjJNT_SLIDE)

    # -- the jaw ---------------------------------------------------------------------------------------
    def open_targets(self) -> dict[str, float]:
        """The grip joints at the open end of their travel: the lower range for the slides and for the pincer hinges alike (a positive hinge angle swings a finger inward)."""

        return {j: float(r[0]) for j, r in zip(self.grip_joints, self.grip_ranges)}

    def closed_targets(self, object_width_m: float | None = None) -> dict[str, float]:
        """The grip joints commanded past the object, to the upper end of their travel; the servos' force limits do the squeezing."""

        return {j: float(r[1]) for j, r in zip(self.grip_joints, self.grip_ranges)}

    # -- the reach -------------------------------------------------------------------------------------
    def site_position(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(data.site_xpos[self.site], dtype=float)

    def current(self, data: mujoco.MjData) -> dict[str, float]:
        return {j: float(data.qpos[a]) for j, a in zip(self.joints, self.qpos_adr)}

    def site_position_of(self, data: mujoco.MjData, joints: dict[str, float]) -> np.ndarray:
        """Where the site would be with the limb's joints at `joints` and the rest of the body as it is: the servos' commanded place."""

        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = data.qpos
        for j, a in zip(self.joints, self.qpos_adr):
            if j in joints:
                scratch.qpos[a] = joints[j]
        mujoco.mj_kinematics(self.model, scratch)
        return np.array(scratch.site_xpos[self.site], dtype=float)

    def solve(self, data: mujoco.MjData, target: np.ndarray, *, seed: dict[str, float] | None = None, down: float = 0.0, damping: float = 0.02, step: float = 0.6, single_start: bool = False, want: np.ndarray | None = None) -> ReachSolution:
        """Joint values that put the grasp site at `target` (world), from the body's pose now, moving the limb's joints only.

        `down` weights a secondary aim: the site's approach axis (the jaw's
        -z for an arm, the pincer's +x for a tentacle) pointing along `want`
        (the floor's way by default), so a jaw comes down on an object from
        above and a pincer can come at it from the side. The solve starts from
        several limb configurations (where it is, the seed, straight, the
        middle of the ranges, and each of those with the first joint turned
        to the target's bearing), solves the position first and then refines
        with the approach axis, and keeps the best. Solved on a scratch copy
        of the state; the physics state is not touched."""

        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, scratch)
        target = np.asarray(target, dtype=float)
        current = np.array([scratch.qpos[a] for a in self.qpos_adr])
        starts = [current]
        if seed:
            starts.append(np.array([seed.get(j, current[i]) for i, j in enumerate(self.joints)]))
        if single_start:
            # a fine move keeps the limb on the branch it is on: only the seed (the targets it is tracking) or where it is
            starts = starts[-1:]
        else:
            starts.append(np.clip(np.zeros(len(self.joints)), self.ranges[:, 0], self.ranges[:, 1]))
            starts.append(self.ranges.mean(axis=1))
            # the first joint turned towards the target, as seen from that joint's body frame
            first = self.joint_ids[0]
            body = int(self.model.jnt_bodyid[first])
            anchor = scratch.xpos[body]
            rotation = scratch.xmat[body].reshape(3, 3)
            local = rotation.T @ (target - anchor)
            bearing = float(np.arctan2(local[1], local[0]))
            for base in list(starts):
                turned = base.copy()
                turned[0] = float(np.clip(base[0] + bearing, self.ranges[0, 0], self.ranges[0, 1]))
                starts.append(turned)
        best = None
        want = np.array([0.0, 0.0, -1.0]) if want is None else np.asarray(want, dtype=float) / max(1e-9, float(np.linalg.norm(want)))
        for start in starts:
            candidates = []
            q, residual, orientation, iterations = self._descend(scratch, start, target, down=0.0, damping=damping, step=step, iterations=self.ITERATIONS // 2, want=want)
            candidates.append((residual + (0.02 * orientation if down > 0.0 else 0.0), q, residual, orientation, iterations))
            if down > 0.0:
                q2, residual2, orientation2, more = self._descend(scratch, q, target, down=down, damping=damping, step=step, iterations=self.ITERATIONS // 2, want=want)
                candidates.append((residual2 + 0.02 * orientation2, q2, residual2, orientation2, iterations + more))
            for score, q, residual, orientation, iterations in candidates:
                if best is None or score < best[0]:
                    best = (score, q, residual, iterations, orientation)
            if best[2] < self.TOLERANCE_M and (down == 0.0 or best[4] < 0.15):
                break
        _, q, residual, iterations, _ = best
        return ReachSolution(joints={j: float(v) for j, v in zip(self.joints, q)}, residual_m=residual, iterations=iterations, reached=residual < self.TOLERANCE_M)

    def _descend(self, scratch: mujoco.MjData, start: np.ndarray, target: np.ndarray, *, down: float, damping: float, step: float, iterations: int, want: np.ndarray | None = None) -> tuple[np.ndarray, float, float, int]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        approach_axis = np.array([0.0, 0.0, -1.0]) if self.slides else np.array([1.0, 0.0, 0.0])
        want = np.array([0.0, 0.0, -1.0]) if want is None else want
        q = np.clip(np.asarray(start, dtype=float), self.ranges[:, 0], self.ranges[:, 1])
        orientation = 0.0
        residual = float("inf")
        count = 0
        for count in range(1, iterations + 1):
            for a, value in zip(self.qpos_adr, q):
                scratch.qpos[a] = value
            mujoco.mj_kinematics(self.model, scratch)
            mujoco.mj_comPos(self.model, scratch)
            error = target - scratch.site_xpos[self.site]
            residual = float(np.linalg.norm(error))
            mujoco.mj_jacSite(self.model, scratch, jacp, jacr, self.site)
            J = jacp[:, self.dof_adr]
            rhs = error
            if down > 0.0:
                rotation = scratch.site_xmat[self.site].reshape(3, 3)
                axis = rotation @ approach_axis
                orientation_error = np.cross(axis, want)
                orientation = float(np.linalg.norm(orientation_error))
                J = np.vstack([J, down * jacr[:, self.dof_adr]])
                rhs = np.concatenate([error, down * orientation_error])
                if not self.slides:
                    # a pincer's fingers swing about its z axis: that axis vertical (either way up) opens them across the object, not over and under it
                    hinge = rotation @ np.array([0.0, 0.0, 1.0])
                    upright = np.array([0.0, 0.0, 1.0 if hinge[2] >= 0.0 else -1.0])
                    roll_error = np.cross(hinge, upright)
                    orientation += float(np.linalg.norm(roll_error))
                    J = np.vstack([J, down * jacr[:, self.dof_adr]])
                    rhs = np.concatenate([rhs, down * roll_error])
            if residual < self.TOLERANCE_M * 0.5 and (down == 0.0 or orientation < 0.1):
                break
            dq = J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(J.shape[0]), rhs)
            q = np.clip(q + step * dq, self.ranges[:, 0], self.ranges[:, 1])
        for a, value in zip(self.qpos_adr, q):
            scratch.qpos[a] = value
        mujoco.mj_kinematics(self.model, scratch)
        residual = float(np.linalg.norm(target - scratch.site_xpos[self.site]))
        if down > 0.0:
            rotation = scratch.site_xmat[self.site].reshape(3, 3)
            axis = rotation @ approach_axis
            orientation = float(np.linalg.norm(np.cross(axis, want)))
            if not self.slides:
                hinge = rotation @ np.array([0.0, 0.0, 1.0])
                orientation += float(np.linalg.norm(np.cross(hinge, np.array([0.0, 0.0, 1.0 if hinge[2] >= 0.0 else -1.0]))))
        return q, residual, orientation, count

    # -- the hold --------------------------------------------------------------------------------------
    def finger_contacts(self, data: mujoco.MjData, object_body: int) -> tuple[bool, bool]:
        """Whether each finger touches the object now."""

        touching = [False] * len(self.finger_bodies)
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(self.model.geom_bodyid[g1]), int(self.model.geom_bodyid[g2])
            for k, finger in enumerate(self.finger_bodies):
                if (b1 == finger and b2 == object_body) or (b2 == finger and b1 == object_body):
                    touching[k] = True
        return tuple(touching)

    def holding(self, data: mujoco.MjData, object_body: int, *, max_offset_m: float = 0.06) -> bool:
        """Both fingers on the object and the object within reach of the grasp site."""

        contacts = self.finger_contacts(data, object_body)
        offset = float(np.linalg.norm(data.xpos[object_body] - data.site_xpos[self.site]))
        return all(contacts) and offset < max_offset_m


def manipulators_of(body: MobileBody, model: mujoco.MjModel) -> list[Manipulator]:
    return [Manipulator(body, model, m) for m in body.declaration["manipulators"]]


__all__ = ["Manipulator", "ReachSolution", "manipulators_of"]
