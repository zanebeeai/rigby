"""Independent placement truth for a transferred object.

Kept apart from the controller on purpose: nothing in :mod:`.transfer` reads
this module's verdicts to decide what to do. The predicates are the ones the
G02 benchmark evaluator registers -- the object's whole geometry inside the
destination region, no robot touching it and no positive normal force from a
robot geom, linear and angular speed below the registered limits, all of it
continuously for the dwell -- applied to an environment model, where robot
bodies carry the uploaded names and world bodies carry the environment
prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..scenes.environment import OBJECT_PREFIX, SUPPORT_PREFIX


@dataclass(frozen=True, slots=True)
class PlacementGoal:
    """Where a transferred object has to end up, and how still."""

    region_minimum_m: tuple[float, float, float]
    region_maximum_m: tuple[float, float, float]
    dwell_s: float = 2.0
    maximum_linear_speed_mps: float = 0.01
    maximum_angular_speed_radps: float = 0.1

    def __post_init__(self) -> None:
        low, high = np.asarray(self.region_minimum_m), np.asarray(self.region_maximum_m)
        if not np.all(high > low):
            raise ValueError("a placement region needs positive extent on every axis")
        if self.dwell_s <= 0 or self.maximum_linear_speed_mps <= 0 or self.maximum_angular_speed_radps <= 0:
            raise ValueError("dwell and speed limits must be positive")

    def scaled(self, factor: float) -> "PlacementGoal":
        """The same region for a world whose lengths were scaled by ``factor``."""

        return PlacementGoal(
            region_minimum_m=tuple(float(v) * factor for v in self.region_minimum_m),
            region_maximum_m=tuple(float(v) * factor for v in self.region_maximum_m),
            dwell_s=self.dwell_s,
            maximum_linear_speed_mps=self.maximum_linear_speed_mps,
            maximum_angular_speed_radps=self.maximum_angular_speed_radps,
        )


@dataclass(frozen=True, slots=True)
class PlacementSample:
    time_s: float
    whole_geometry_inside: bool
    released: bool
    maximum_robot_normal_force_n: float
    linear_speed_mps: float
    angular_speed_radps: float

    @property
    def conditions_met(self) -> bool:
        return self.whole_geometry_inside and self.released


def _is_world_body(model: mujoco.MjModel, body: int) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
    return body == 0 or name.startswith(SUPPORT_PREFIX) or name.startswith(OBJECT_PREFIX) or name.startswith("scene_")


class PlacementEvaluator:
    """Assess one object against a goal on every physics step it is shown."""

    def __init__(self, model: mujoco.MjModel, goal: PlacementGoal, *, object_geom: str, object_joint: str) -> None:
        self.model = model
        self.goal = goal
        self.geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_geom)
        self.joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, object_joint)
        if self.geom < 0 or self.joint < 0:
            raise KeyError("the placement evaluator needs the object's geom and free joint")
        self._eligible_since: float | None = None
        self._last_time: float | None = None
        self._snapshot = mujoco.MjData(model)
        self.dwell_s = 0.0
        self.success = False
        self.last: PlacementSample | None = None

    def assess(self, data: mujoco.MjData) -> PlacementSample:
        model, goal = self.model, self.goal
        now = float(data.time)
        if self._last_time is not None and now - self._last_time > model.opt.timestep + 1e-9:
            self._eligible_since = None  # A gap in observation restarts the dwell.
        mujoco.mj_copyData(self._snapshot, model, data)
        mujoco.mj_forward(model, self._snapshot)
        sample = self._snapshot
        centre = np.asarray(sample.geom_xpos[self.geom])
        half = np.abs(np.asarray(sample.geom_xmat[self.geom]).reshape(3, 3)) @ np.asarray(model.geom_size[self.geom])
        low, high = np.asarray(goal.region_minimum_m), np.asarray(goal.region_maximum_m)
        inside = bool(np.all(centre - half >= low) and np.all(centre + half <= high))
        dof = int(model.jnt_dofadr[self.joint])
        linear = float(np.linalg.norm(sample.qvel[dof : dof + 3]))
        angular = float(np.linalg.norm(sample.qvel[dof + 3 : dof + 6]))
        robot_contact = False
        maximum_normal = 0.0
        force = np.zeros(6)
        for index in range(sample.ncon):
            contact = sample.contact[index]
            if self.geom not in (int(contact.geom1), int(contact.geom2)):
                continue
            other = int(contact.geom2 if int(contact.geom1) == self.geom else contact.geom1)
            if _is_world_body(model, int(model.geom_bodyid[other])):
                continue
            if contact.efc_address >= 0:
                mujoco.mj_contactForce(model, sample, index, force)
                maximum_normal = max(maximum_normal, float(force[0]))
            robot_contact |= float(contact.dist) <= 0.0 or (contact.efc_address >= 0 and float(force[0]) > 0.0)
        released = not robot_contact
        eligible = inside and released and linear <= goal.maximum_linear_speed_mps and angular <= goal.maximum_angular_speed_radps
        if not eligible:
            self._eligible_since = None
        elif self._eligible_since is None:
            self._eligible_since = now
        self.dwell_s = 0.0 if self._eligible_since is None else now - self._eligible_since
        self.success = self.success or (eligible and self.dwell_s >= goal.dwell_s)
        self._last_time = now
        self.last = PlacementSample(now, inside, released, maximum_normal, linear, angular)
        return self.last
