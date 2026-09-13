"""Independent labels from the full recorded state.

The labeler reads what the evaluator may not: every recorded position,
velocity and contact. Its rules are the physical facts the predicates name,
written without reference to the evaluator's thresholds: opposition is
contact force on every opposing group; held is opposition throughout the
window with the object off every support; moving with the robot is the
object's displacement matching the grasp point's while the grasp point
travels, held or pushed; stably placed is the placement evaluator's own conditions on
every sample of the window; area clear is the object's whole geometry
outside the region throughout; reachable is the transfer's own solver
placing the grasp point above and at the object, hand facing down, within
the self-collision guard, from where the arm stands or from its restart
seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..contact.grasp import _hand_facing, _scene_rest_qpos
from ..contact.transfer import HOVER_HEIGHTS, grasp_standoff_m, restart_seeds
from ..contracts import EffectorV1
from ..grounding import ik
from ..grounding.grounder import _collision_guard
from .episode import Episode
from .sensors import frame_for


ORACLE_CONTACT_N = 1e-3
"""Positive normal force on a member, in newtons: physical contact."""
ORACLE_TRAVEL_M = 0.01
"""How far the grasp point must move over the window for the object to be
said to move with it."""
ORACLE_MATCH_FRACTION = 0.5
"""How much of that travel the object's displacement may differ by: an
object that moved half as far as the hand that holds it is sliding in it,
not moving with it."""


@dataclass(frozen=True)
class Label:
    predicate: str
    value: bool
    detail: dict


class OracleLabeler:
    def __init__(self, episode: Episode) -> None:
        self.episode = episode
        self._solver: dict[str, tuple] = {}

    def _indices(self, start_s: float, end_s: float) -> range:
        times = self.episode.times
        lo = int(np.searchsorted(times, start_s - 1e-9, side="left"))
        hi = int(np.searchsorted(times, end_s + 1e-9, side="right"))
        return range(max(lo, 0), max(hi, lo + 1))

    def opposition(self, effector: EffectorV1, index: int) -> bool:
        groups = self.episode.group_forces(effector, index)
        return len(groups) >= 2 and all(f >= ORACLE_CONTACT_N for f in groups)

    def opposition_established(self, effector: EffectorV1, now_s: float, window_s: float) -> Label:
        indices = self._indices(now_s - window_s, now_s)
        opposed = [self.opposition(effector, i) for i in indices]
        fraction = sum(opposed) / len(opposed)
        return Label("opposition_established", fraction >= 0.9, {"opposed_fraction": fraction})

    def held(self, effector: EffectorV1, now_s: float, window_s: float) -> Label:
        indices = self._indices(now_s - window_s, now_s)
        opposed = all(self.opposition(effector, i) for i in indices)
        support, _, _ = self.episode.object_contacts(indices[-1])
        return Label("held", opposed and not support, {"opposition_throughout": opposed, "on_support": support})

    def moving_with_robot(self, effector: EffectorV1, now_s: float, window_s: float) -> Label:
        indices = self._indices(now_s - window_s, now_s)
        site = self.episode.grasp_site(effector)
        first, last = indices[0], indices[-1]
        effector_start = np.array(self.episode.kinematics(first).site_xpos[site], dtype=float)
        effector_end = np.array(self.episode.kinematics(last).site_xpos[site], dtype=float)
        object_start, object_end = self.episode.object_position(first), self.episode.object_position(last)
        travel = float(np.linalg.norm(effector_end - effector_start))
        mismatch = float(np.linalg.norm((object_end - object_start) - (effector_end - effector_start)))
        # Kinematic on purpose: an object the hand pushes along moves with the
        # robot as surely as one it carries. Whether it is held is ``held``.
        return Label("moving_with_robot", travel >= ORACLE_TRAVEL_M and mismatch <= ORACLE_MATCH_FRACTION * travel, {"effector_travel_m": travel, "mismatch_m": mismatch})

    def stably_placed(self, now_s: float, window_s: float) -> Label:
        episode, goal = self.episode, self.episode.goal
        low, high = np.asarray(goal.region_minimum_m), np.asarray(goal.region_maximum_m)
        worst_speed = 0.0
        for i in self._indices(now_s - window_s, now_s):
            lo, hi = episode.object_extent(i)
            inside = bool(np.all(lo >= low) and np.all(hi <= high))
            _, robot, _ = episode.object_contacts(i)
            linear, angular = episode.object_velocity(i)
            worst_speed = max(worst_speed, float(np.linalg.norm(linear)))
            if not inside or robot or float(np.linalg.norm(linear)) > goal.maximum_linear_speed_mps or float(np.linalg.norm(angular)) > goal.maximum_angular_speed_radps:
                return Label("stably_placed", False, {"inside": inside, "robot_contact": robot, "peak_speed_mps": worst_speed})
        return Label("stably_placed", True, {"inside": True, "robot_contact": False, "peak_speed_mps": worst_speed})

    def area_clear(self, now_s: float, window_s: float) -> Label:
        goal = self.episode.goal
        low, high = np.asarray(goal.region_minimum_m), np.asarray(goal.region_maximum_m)
        for i in self._indices(now_s - window_s, now_s):
            lo, hi = self.episode.object_extent(i)
            overlap = bool(np.all(hi >= low) and np.all(lo <= high))
            if overlap:
                return Label("area_clear", False, {"overlap": True})
        return Label("area_clear", True, {"overlap": False})

    def _solver_for(self, effector: EffectorV1) -> tuple:
        if effector.chain_id not in self._solver:
            episode = self.episode
            model, manifest = episode.model, episode.manifest
            site = episode.grasp_site(effector)
            site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site)
            joints = ik.chain_joint_names(model, site_name, exclude=frozenset(effector.grip_joints))
            guard = _collision_guard(manifest, model)
            facing = _hand_facing(manifest, effector)
            standoff = grasp_standoff_m(model, manifest, effector, site_name, float(episode.object_half_extent_m[2]))
            frame = frame_for(episode, effector)
            rest = _scene_rest_qpos(model, manifest)
            self._solver[effector.chain_id] = (site_name, joints, guard, facing, standoff, frame, rest)
        return self._solver[effector.chain_id]

    def reachable(self, effector: EffectorV1, now_s: float) -> Label:
        """The transfer's own solver: the grasp point at the hover above the
        object and at the object, hand facing down, from where the arm stands
        and then from the restart seeds."""

        episode = self.episode
        site_name, joints, guard, facing, standoff, frame, rest = self._solver_for(effector)
        index = episode.index_at(now_s)
        centre = episode.object_position(index)
        height = 2.0 * float(episode.object_half_extent_m[2])
        up = np.asarray(frame.up, dtype=float)
        targets = np.array([centre + up * (height + HOVER_HEIGHTS * height + standoff), centre + up * standoff])
        downward = (facing, -up) if facing is not None else None
        current = np.array(episode.record.arrays["qpos"][index], dtype=float)
        seeds = [("current", current)] + restart_seeds(episode.model, frame, joints, rest, targets[-1])
        failure = ""
        for label, seed in seeds:
            try:
                ik.solve_site_path(episode.model, site_name, joints, targets, seed_qpos=seed, guard=guard, facing=downward)
                return Label("reachable", True, {"seed": label, "distance_m": float(np.linalg.norm(centre - frame.origin))})
            except ik.IkFailure as error:
                failure = failure or f"{label}: {str(error)[:120]}"
        return Label("reachable", False, {"failure": failure, "distance_m": float(np.linalg.norm(centre - frame.origin))})

    def label(self, predicate: str, effector: EffectorV1, now_s: float, window_s: float) -> Label:
        if predicate == "reachable":
            return self.reachable(effector, now_s)
        if predicate == "opposition_established":
            return self.opposition_established(effector, now_s, window_s)
        if predicate == "held":
            return self.held(effector, now_s, window_s)
        if predicate == "moving_with_robot":
            return self.moving_with_robot(effector, now_s, window_s)
        if predicate == "stably_placed":
            return self.stably_placed(now_s, window_s)
        if predicate == "area_clear":
            return self.area_clear(now_s, window_s)
        raise KeyError(predicate)
