"""Inputs to an experiment, frozen before body-dependent execution.

The public development protocol is separate from the existing sealed release
benchmark. These contracts neither load nor reveal that benchmark's identities.
"""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import Field, FiniteFloat, model_validator
from rigby_core.contracts import Contract
from rigby_core.hashing import content_hash

from ..scenes.environment import EnvironmentV1

Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Quaternion = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
Mode = Literal["strict_fixed_world", "capability_normalized"]


class RegionV1(Contract):
    minimum_m: Vector3
    maximum_m: Vector3

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if any(a >= b for a, b in zip(self.minimum_m, self.maximum_m)):
            raise ValueError("Region bounds must increase on every axis")
        return self


class RootGoalV1(Contract):
    predicate: Literal["all_objects_placed"] = "all_objects_placed"
    object_names: tuple[str, ...] = Field(min_length=1)
    target: RegionV1
    dwell_s: FiniteFloat = Field(default=2.0, gt=0)
    maximum_linear_speed_mps: FiniteFloat = Field(default=0.01, gt=0)
    maximum_angular_speed_radps: FiniteFloat = Field(default=0.1, gt=0)
    require_release: Literal[True] = True
    release_rule: Literal["no_robot_touch_or_positive_normal_force"] = "no_robot_touch_or_positive_normal_force"
    containment: Literal["whole_geometry"] = "whole_geometry"

    @model_validator(mode="after")
    def unique_objects(self) -> Self:
        if len(set(self.object_names)) != len(self.object_names):
            raise ValueError("Goal objects must be unique")
        return self


class StartZoneV1(Contract):
    base_position_m: Vector3 = (0, 0, 0)
    base_quaternion_wxyz: Quaternion = (1, 0, 0, 0)
    position_tolerance_m: FiniteFloat = Field(default=0.001, ge=0)
    orientation_tolerance_rad: FiniteFloat = Field(default=0.001, ge=0)
    initialization: Literal["declared_body_rest_pose"] = "declared_body_rest_pose"

    @model_validator(mode="after")
    def unit_rotation(self) -> Self:
        if abs(sum(x*x for x in self.base_quaternion_wxyz) - 1) > 1e-9:
            raise ValueError("Start-zone quaternion must be unit length")
        return self


class SensorPolicyV1(Contract):
    mode: Literal["declared_sensors", "fully_observed_diagnostic"] = "declared_sensors"
    joint_positions: bool = True
    joint_velocities: bool = True
    rgb: bool = True
    depth: bool = False
    camera_source: Literal["fixed_world", "declared_body_mount"] = "fixed_world"
    width: int = Field(default=96, ge=8, le=2048)
    height: int = Field(default=96, ge=8, le=2048)
    frequency_hz: int = Field(default=10, ge=1, le=500)
    noise: Literal["ideal_simulated_sensors"] = "ideal_simulated_sensors"
    missing_data: Literal["refuse"] = "refuse"


class PhysicsV1(Contract):
    timestep_s: FiniteFloat = Field(default=0.002, gt=0, le=0.01)
    gravity_mps2: Vector3 = (0, 0, -9.81)
    integrator: Literal["Euler", "implicitfast"] = "implicitfast"
    solver: Literal["Newton"] = "Newton"
    jacobian: Literal["dense"] = "dense"
    iterations: int = Field(default=100, ge=1)
    tolerance: FiniteFloat = Field(default=1e-10, gt=0)
    contact_timeconst_s: FiniteFloat = Field(default=0.004, gt=0)

    @model_validator(mode="after")
    def resolved_contact(self) -> Self:
        if self.contact_timeconst_s < 2 * self.timestep_s:
            raise ValueError("Contact time constant must resolve at least two physics steps")
        return self


class LightingV1(Contract):
    position_m: Vector3 = (0, -1, 3)
    direction: Vector3 = (0, 0, -1)
    diffuse: Vector3 = (0.8, 0.8, 0.8)
    ambient: Vector3 = (0.3, 0.3, 0.3)
    specular: Vector3 = (0.1, 0.1, 0.1)


class WorldCameraV1(Contract):
    position_m: Vector3 = (1.8, -2.5, 1.8)
    xyaxes: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] = (0.8115, 0.5843, 0, -0.2496, 0.3467, 0.9042)
    fovy_degrees: FiniteFloat = Field(default=50, gt=0, lt=180)


class BenchmarkWorldV1(Contract):
    schema_version: Literal["benchmark.world.v1"] = "benchmark.world.v1"
    environment: EnvironmentV1
    physics: PhysicsV1 = Field(default_factory=PhysicsV1)
    lighting: LightingV1 = Field(default_factory=LightingV1)
    camera: WorldCameraV1 = Field(default_factory=WorldCameraV1)
    start_zone: StartZoneV1 = Field(default_factory=StartZoneV1)
    sensor_policy: SensorPolicyV1 = Field(default_factory=SensorPolicyV1)
    goal: RootGoalV1
    reference_length_m: FiniteFloat = Field(default=0.5, gt=0)
    floor_half_width_m: FiniteFloat = Field(default=3, gt=0)

    @model_validator(mode="after")
    def complete_geometry(self) -> Self:
        content_hash(self)  # Reject nonfinite values in legacy environment models.
        items = (*self.environment.fixtures, *self.environment.objects)
        names = [x.name for x in items]
        if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", n) for n in names):
            raise ValueError("World elements require unique portable names")
        if any(any(x <= 0 for x in item.size_m) for item in items):
            raise ValueError("World half-extents must be positive")
        if set(self.goal.object_names) - {x.name for x in self.environment.objects}:
            raise ValueError("The root goal names a missing world object")
        if self.environment.robot_mount_m != self.start_zone.base_position_m:
            raise ValueError("Environment mount must equal the frozen start zone")
        return self


class TrialBudgetV1(Contract):
    maximum_attempts: int = Field(default=3, ge=1)
    maximum_simulation_s: FiniteFloat = Field(default=120, gt=0)
    maximum_wall_s: FiniteFloat = Field(default=300, gt=0)
    retries_consume_budget: Literal[True] = True


class PerturbationsV1(Contract):
    distribution: Literal["independent_uniform"] = "independent_uniform"
    object_translation_half_width_m: Vector3 = (0.02, 0.02, 0)
    mass_multiplier_range: tuple[FiniteFloat, FiniteFloat] = (0.8, 1.2)
    friction_multiplier_range: tuple[FiniteFloat, FiniteFloat] = (0.8, 1.2)
    draw_order: Literal["sorted_object_names_xyz_mass_friction"] = "sorted_object_names_xyz_mass_friction"
    invalid_world_draw: Literal["record_refusal_no_redraw"] = "record_refusal_no_redraw"

    @model_validator(mode="after")
    def ranges(self) -> Self:
        if any(x < 0 for x in self.object_translation_half_width_m):
            raise ValueError("Translation half widths cannot be negative")
        for low, high in (self.mass_multiplier_range, self.friction_multiplier_range):
            if low <= 0 or low > high:
                raise ValueError("Multiplier bounds must be positive and ordered")
        return self


class PublicSplitV1(Contract):
    split_id: str
    purpose: Literal["development", "validation", "engineering_test"]
    seeds: tuple[int, ...] = Field(min_length=1)
    visibility: Literal["public_engineering_only"] = "public_engineering_only"

    @model_validator(mode="after")
    def valid_seeds(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds) or any(s < 0 or s >= 2**32 for s in self.seeds):
            raise ValueError("Seeds must be unique unsigned 32-bit integers")
        return self


class BenchmarkProtocolV1(Contract):
    schema_version: Literal["benchmark.protocol.v1"] = "benchmark.protocol.v1"
    protocol_id: str = Field(min_length=1)
    modes: tuple[Mode, ...] = ("strict_fixed_world", "capability_normalized")
    world: BenchmarkWorldV1
    budget: TrialBudgetV1 = Field(default_factory=TrialBudgetV1)
    perturbations: PerturbationsV1 = Field(default_factory=PerturbationsV1)
    splits: tuple[PublicSplitV1, ...] = Field(min_length=1)
    metrics: tuple[str, ...] = ("root_success", "elapsed_simulation_s", "elapsed_wall_s", "attempts", "typed_refusal", "coverage", "feasible_success_rate", "all_attempted_success_rate")
    feasibility_rule: Literal["independent_geometry_and_mechanics_before_scoring"] = "independent_geometry_and_mechanics_before_scoring"
    controller_failure_is_infeasibility: Literal[False] = False
    excluded_bodies_stay_in_coverage: Literal[True] = True
    normalization_rule: Literal["uniform_length_and_cubic_mass_preserve_density"] = "uniform_length_and_cubic_mass_preserve_density"
    semantic_substitutions: Literal["forbidden"] = "forbidden"
    sealed_release_policy: Literal["existing_manifests_untouched_independent_custodian_required"] = "existing_manifests_untouched_independent_custodian_required"
    confirmatory_split_rule: Literal["disjoint_topology_provenance_scene_family_and_prompt_template"] = "disjoint_topology_provenance_scene_family_and_prompt_template"

    @model_validator(mode="after")
    def disjoint_splits(self) -> Self:
        ids = [s.split_id for s in self.splits]
        seeds = [s for split in self.splits for s in split.seeds]
        if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
            raise ValueError("Split identifiers and seeds must be disjoint")
        if len(set(self.modes)) != len(self.modes) or not self.modes:
            raise ValueError("Declare distinct supported modes")
        if self.world.goal.dwell_s >= self.budget.maximum_simulation_s:
            raise ValueError("A dwell interval must fit inside the trial budget")
        return self
