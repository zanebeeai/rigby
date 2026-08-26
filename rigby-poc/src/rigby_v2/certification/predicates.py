"""Measured object-state predicates evaluated from authoritative traces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import degrees
from typing import Protocol

import mujoco
import numpy as np
import numpy.typing as npt

from rigby_v2.contracts import SceneManifestV2, SceneStatePredicateV1
from rigby_v2.simulation.runtime import SimulationResult

from .models import ContactPair


class PredicateSupportError(ValueError):
    """The model does not expose state required by a predicate."""


@dataclass(frozen=True)
class KinematicTrace:
    times_s: npt.NDArray[np.float64]
    body_positions: dict[str, npt.NDArray[np.float64]]
    body_rotations: dict[str, npt.NDArray[np.float64]]
    joint_positions: dict[str, npt.NDArray[np.float64]]
    site_positions: dict[str, npt.NDArray[np.float64]]
    subtree_com: dict[str, npt.NDArray[np.float64]]


@dataclass(frozen=True)
class ObjectStateContext:
    model: mujoco.MjModel
    simulation: SimulationResult
    kinematics: KinematicTrace


@dataclass(frozen=True)
class PredicateEvaluation:
    name: str
    passed: bool
    message: str
    observed: float | int | str | None = None
    required: float | int | str | None = None


class ObjectStatePredicate(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(self, context: ObjectStateContext) -> PredicateEvaluation: ...


@dataclass(frozen=True)
class ManifestStatePredicate:
    """Evaluate the state predicate declared by a compiled object pack.

    Joint targets use their final generalized coordinate.  Semantic-site
    targets use maximum vertical displacement from their initial measured
    pose, which makes pack lift thresholds independent of world placement.
    """

    specification: SceneStatePredicateV1

    @property
    def name(self) -> str:
        return f"{self.specification.object_id}.{self.specification.name}"

    def evaluate(self, context: ObjectStateContext) -> PredicateEvaluation:
        target = self.specification.target
        if target in context.kinematics.joint_positions:
            values = context.kinematics.joint_positions[target]
            observed = float(values[-1])
            if self.specification.units == "degrees":
                observed = degrees(observed)
        elif target in context.kinematics.site_positions:
            if self.specification.units != "meters":
                raise PredicateSupportError(
                    f"semantic site predicate {self.name!r} requires meter units"
                )
            positions = context.kinematics.site_positions[target]
            observed = float(np.max(positions[:, 2] - positions[0, 2]))
        else:
            raise PredicateSupportError(
                f"predicate target {target!r} is absent from the compiled model"
            )

        required = self.specification.values
        operator = self.specification.operator
        if operator == "ge":
            passed = observed >= required[0]
            requirement = f">={required[0]}"
        elif operator == "le":
            passed = observed <= required[0]
            requirement = f"<={required[0]}"
        elif operator == "between":
            passed = required[0] <= observed <= required[1]
            requirement = f"[{required[0]},{required[1]}]"
        else:
            raise PredicateSupportError(
                f"predicate operator {operator!r} is not supported for compiled packs"
            )
        return PredicateEvaluation(
            name=self.name,
            passed=passed,
            message=(
                "measured object state satisfies the pack predicate"
                if passed
                else "measured object state failed the pack predicate"
            ),
            observed=observed,
            required=requirement,
        )


def predicates_from_manifest(
    scene: SceneManifestV2,
) -> tuple[ManifestStatePredicate, ...]:
    return tuple(ManifestStatePredicate(item) for item in scene.state_predicates)


def _window_indices(times: npt.NDArray[np.float64], start_s: float, end_s: float | None) -> npt.NDArray[np.int64]:
    end = float(times[-1]) if end_s is None else end_s
    return np.flatnonzero((times >= start_s) & (times <= end))


def _object_effector_contacts(
    context: ObjectStateContext,
    object_geoms: frozenset[str],
    effector_geoms: frozenset[str],
    frame_index: int,
    minimum_force_n: float,
) -> set[ContactPair]:
    found: set[ContactPair] = set()
    frame = context.simulation.trace.contacts[frame_index]
    for contact in frame.contacts:
        names = {contact.geom1_name, contact.geom2_name}
        if (
            names & object_geoms
            and names & effector_geoms
            and contact.normal_force_n >= minimum_force_n
        ):
            found.add(ContactPair.of(contact.geom1_name, contact.geom2_name))
    return found


@dataclass(frozen=True)
class GraspPredicate:
    object_geoms: frozenset[str]
    effector_geoms: frozenset[str]
    start_s: float = 0.0
    end_s: float | None = None
    minimum_force_n: float = 0.1
    min_distinct_effectors: int = 2
    name: str = "grasp"

    def evaluate(self, context: ObjectStateContext) -> PredicateEvaluation:
        indices = _window_indices(context.kinematics.times_s, self.start_s, self.end_s)
        best = 0
        for index in indices:
            pairs = _object_effector_contacts(
                context,
                self.object_geoms,
                self.effector_geoms,
                int(index),
                self.minimum_force_n,
            )
            effectors = {
                name
                for pair in pairs
                for name in (pair.first, pair.second)
                if name in self.effector_geoms
            }
            best = max(best, len(effectors))
        return PredicateEvaluation(
            name=self.name,
            passed=best >= self.min_distinct_effectors,
            message="measured simultaneous effector contacts satisfy grasp" if best >= self.min_distinct_effectors else "insufficient measured grasp contacts",
            observed=best,
            required=self.min_distinct_effectors,
        )


@dataclass(frozen=True)
class LiftPredicate:
    object_body: str
    min_rise_m: float
    complete_by_s: float | None = None
    name: str = "lift"

    def evaluate(self, context: ObjectStateContext) -> PredicateEvaluation:
        try:
            positions = context.kinematics.body_positions[self.object_body]
        except KeyError as exc:
            raise PredicateSupportError(f"object body {self.object_body!r} is missing") from exc
        indices = _window_indices(context.kinematics.times_s, 0.0, self.complete_by_s)
        rise = float(np.max(positions[indices, 2]) - positions[0, 2]) if len(indices) else 0.0
        return PredicateEvaluation(
            name=self.name,
            passed=rise >= self.min_rise_m,
            message="object reached required measured lift" if rise >= self.min_rise_m else "object did not reach required lift",
            observed=rise,
            required=self.min_rise_m,
        )


@dataclass(frozen=True)
class HoldPredicate:
    object_body: str
    start_s: float
    end_s: float
    min_height_above_initial_m: float
    max_vertical_drift_m: float = 0.03
    name: str = "hold"

    def evaluate(self, context: ObjectStateContext) -> PredicateEvaluation:
        try:
            z = context.kinematics.body_positions[self.object_body][:, 2]
        except KeyError as exc:
            raise PredicateSupportError(f"object body {self.object_body!r} is missing") from exc
        indices = _window_indices(context.kinematics.times_s, self.start_s, self.end_s)
        if not len(indices):
            return PredicateEvaluation(self.name, False, "hold interval has no samples")
        held = z[indices]
        clearance = float(np.min(held) - z[0])
        drift = float(np.max(held) - np.min(held))
        passed = clearance >= self.min_height_above_initial_m and drift <= self.max_vertical_drift_m
        return PredicateEvaluation(
            name=self.name,
            passed=passed,
            message="measured object height remained stable" if passed else "object height failed hold bounds",
            observed=f"clearance={clearance:.6g}, drift={drift:.6g}",
            required=f"clearance>={self.min_height_above_initial_m}, drift<={self.max_vertical_drift_m}",
        )


@dataclass(frozen=True)
class ReleasePredicate:
    object_geoms: frozenset[str]
    effector_geoms: frozenset[str]
    start_s: float
    settle_duration_s: float = 0.1
    minimum_force_n: float = 0.05
    name: str = "release"

    def evaluate(self, context: ObjectStateContext) -> PredicateEvaluation:
        final_time = float(context.kinematics.times_s[-1])
        settle_start = max(self.start_s, final_time - self.settle_duration_s)
        indices = _window_indices(context.kinematics.times_s, settle_start, None)
        touching = sum(
            bool(
                _object_effector_contacts(
                    context,
                    self.object_geoms,
                    self.effector_geoms,
                    int(index),
                    self.minimum_force_n,
                )
            )
            for index in indices
        )
        return PredicateEvaluation(
            name=self.name,
            passed=touching == 0,
            message="no measured effector contact remains" if touching == 0 else "effector contact persists after release",
            observed=touching,
            required=0,
        )


class JointComparator(StrEnum):
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    NEAR = "near"


@dataclass(frozen=True)
class ArticulatedCompletionPredicate:
    joint_name: str
    target: float
    tolerance: float
    comparator: JointComparator = JointComparator.NEAR
    name: str = "articulated_completion"

    def evaluate(self, context: ObjectStateContext) -> PredicateEvaluation:
        try:
            values = context.kinematics.joint_positions[self.joint_name]
        except KeyError as exc:
            raise PredicateSupportError(f"joint {self.joint_name!r} is missing") from exc
        actual = float(values[-1])
        if self.comparator is JointComparator.AT_LEAST:
            passed = actual >= self.target - self.tolerance
        elif self.comparator is JointComparator.AT_MOST:
            passed = actual <= self.target + self.tolerance
        else:
            passed = abs(actual - self.target) <= self.tolerance
        return PredicateEvaluation(
            name=self.name,
            passed=passed,
            message="articulated state reached target" if passed else "articulated state missed target",
            observed=actual,
            required=f"{self.comparator.value}:{self.target}±{self.tolerance}",
        )
