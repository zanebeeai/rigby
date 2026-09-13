"""Privileged goal evaluation; this module is never passed to the acting API."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from rigby_core.observations import EvaluatorTruth, ObjectPose, PredicateValue

from .world import BenchmarkRefusal, CompiledBenchmarkWorld, ROBOT_PREFIX, WORLD_PREFIX, verify_world


def inspection_copy(model, data):
    """Derive current positions on a copy, preserving live solver/controller state."""
    snapshot = mujoco.MjData(model)
    mujoco.mj_copyData(snapshot, model, data)
    mujoco.mj_forward(model, snapshot)
    return snapshot


@dataclass(frozen=True)
class GoalAssessment:
    time_s: float
    conditions_met: bool
    dwell_elapsed_s: float
    root_success: bool
    objects: tuple[dict, ...]
    truth: EvaluatorTruth


class IndependentEvaluator:
    def __init__(self, compiled: CompiledBenchmarkWorld, *, expected_world_sha256: str):
        verify_world(compiled, expected_world_sha256)
        self.compiled = compiled
        self.expected_world_sha256 = expected_world_sha256
        self._eligible_since: float | None = None
        self._last_time: float | None = None

    def evaluate(self, data: mujoco.MjData) -> GoalAssessment:
        verify_world(self.compiled, self.expected_world_sha256)
        model, goal = self.compiled.model, self.compiled.world.goal
        now = float(data.time)
        if not np.isfinite(now) or now < 0 or (self._last_time is not None and now <= self._last_time):
            raise BenchmarkRefusal("invalid_evaluation_clock", "Evaluator needs strictly increasing actual physics timestamps")
        if self._last_time is not None and now-self._last_time > model.opt.timestep + 1e-9:
            # Never infer continuous satisfaction from two distant snapshots.
            # A missed physics interval restarts the required dwell interval.
            self._eligible_since = None
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise BenchmarkRefusal("nonfinite_state", "Nonfinite state cannot satisfy a goal")
        sample = inspection_copy(model, data)
        rows, poses = [], []
        low, high = np.asarray(goal.target.minimum_m), np.asarray(goal.target.maximum_m)
        for name in goal.object_names:
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, WORLD_PREFIX+name+"_geom")
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WORLD_PREFIX+name)
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, WORLD_PREFIX+name+"_free")
            if min(geom, body, joint) < 0:
                raise BenchmarkRefusal("missing_goal_object", name)
            center = sample.geom_xpos[geom]
            half_width = np.abs(sample.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
            contained = bool(np.all(center-half_width >= low) and np.all(center+half_width <= high))
            v = int(model.jnt_dofadr[joint])
            linear = float(np.linalg.norm(sample.qvel[v:v+3]))
            angular = float(np.linalg.norm(sample.qvel[v+3:v+6]))
            robot_contact = False
            maximum_robot_normal_force = 0.0
            for index, contact in enumerate(sample.contact):
                if geom not in (contact.geom1, contact.geom2):
                    continue
                other = int(contact.geom2 if contact.geom1 == geom else contact.geom1)
                other_body = int(model.geom_bodyid[other])
                other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body) or ""
                if not other_name.startswith(ROBOT_PREFIX):
                    continue
                # Positive-distance contacts can still support an object when
                # collision margins are enabled. Decode the actual normal force
                # rather than equating geometric separation with release.
                force = np.zeros(6)
                if contact.efc_address >= 0:
                    mujoco.mj_contactForce(model, sample, index, force)
                if not np.isfinite(force).all() or not np.isfinite(contact.dist):
                    raise BenchmarkRefusal("nonfinite_contact", "Invalid contact evidence cannot prove release")
                maximum_robot_normal_force = max(maximum_robot_normal_force, float(force[0]))
                # Zero is deliberately conservative: no positive force threshold
                # can silently permit a small robot to keep holding an object.
                robot_contact |= contact.dist <= 0 or force[0] > 0
            valid = contained and not robot_contact and linear <= goal.maximum_linear_speed_mps and angular <= goal.maximum_angular_speed_radps
            rows.append({"object": name, "whole_geometry_inside": contained, "released": not robot_contact,
                         "maximum_robot_normal_force_n": maximum_robot_normal_force,
                         "linear_speed_mps": linear, "angular_speed_radps": angular, "conditions_met": valid})
            poses.append(ObjectPose(object_id=name, position_xyz=tuple(float(x) for x in sample.xpos[body]),
                                    orientation_wxyz=tuple(float(x) for x in sample.xquat[body])))
        eligible = all(row["conditions_met"] for row in rows)
        if not eligible:
            self._eligible_since = None
        elif self._eligible_since is None:
            self._eligible_since = now
        dwell = 0.0 if self._eligible_since is None else now-self._eligible_since
        success = eligible and dwell >= goal.dwell_s
        self._last_time = now
        truth = EvaluatorTruth(time_s=now, object_poses=tuple(poses),
                               success_labels=(PredicateValue(name="root_success", satisfied=success),),
                               simulator_state=tuple(float(x) for x in np.concatenate((sample.qpos, sample.qvel))))
        return GoalAssessment(now, eligible, dwell, success, tuple(rows), truth)
