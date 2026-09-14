"""Contracts for mobile bodies: what a floating-base robot declares, and what ingest measured of it.

The fixed-base manifest (``rigby_general.contracts.RobotAssetManifestV1``)
refuses a free joint on purpose: its gates prove a base stayed put. A mobile
body's base is meant to move, so it gets its own manifest here, measured
the same way -- nothing in it is copied from a name, every physical claim
is read off the compiled model or off a settling test -- and validated
against what the body's author declared: which members are meant to bear
its weight, which stances are meant to be statically stable, how tall it
stands. Where the declaration and the measurement disagree, the manifest
says so and the body is not accepted.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from ..contracts import Contract


class BaseKind(StrEnum):
    LEGGED = "legged"
    WHEELED = "wheeled"
    CRAWLING = "crawling"


class MobileJointV1(Contract):
    name: str = Field(min_length=1)
    kind: Literal["hinge", "slide"]
    body: str = Field(min_length=1)
    limb: str = Field(min_length=1)
    role: Literal["leg", "wheel", "arm", "grip", "tentacle"]
    minimum: float | None = None
    maximum: float | None = None
    """None for a wheel that spins without limit."""
    effort_limit: float = Field(gt=0.0)
    velocity_limit: float = Field(gt=0.0)
    actuator: str = Field(min_length=1)
    actuator_kind: Literal["position", "velocity"]
    rest: float


class LimbV1(Contract):
    limb_id: str = Field(min_length=1)
    role: Literal["leg", "wheel_leg", "tentacle", "arm"]
    joints: tuple[str, ...] = Field(min_length=1)
    tip_body: str = Field(min_length=1)
    length_m: float = Field(ge=0.0)
    """Sum of the link offsets from the limb's first joint to its tip."""


class SupportMemberV1(Contract):
    body: str = Field(min_length=1)
    limb: str | None = None
    geoms: tuple[str, ...] = Field(min_length=1)
    friction: float = Field(gt=0.0)
    """The sliding friction of its colliders (the smallest, if several)."""
    touch_sensor: str | None = None


class ManipulatorV1(Contract):
    limb: str = Field(min_length=1)
    grasp_site: str = Field(min_length=1)
    grip_joints: tuple[str, ...] = Field(min_length=1)
    fingers: tuple[str, ...] = Field(min_length=1)
    aperture_m: float = Field(gt=0.0)
    """The fingers' separation at the open end of their travel, measured."""
    reach_m: float = Field(gt=0.0)
    """The grasp site's distance from the limb's first joint with the limb stretched along its links: the sum of the body offsets from that joint's body to the site, plus the site's own offset; measured."""


class StanceMeasurementV1(Contract):
    """What a settling test found: the body dropped two centimetres onto a
    level floor in the stance, every position servo holding it, for the
    declared duration."""

    stance: str = Field(min_length=1)
    declared_statically_stable: bool
    settled: bool
    """Came to rest: the base slower than the threshold over the last second and its height steady."""
    upright: bool
    """The base's tilt from its initial orientation stayed under the threshold."""
    statically_stable: bool
    """settled and upright, with at least three support contacts (or one broad one) at rest."""
    base_height_m: float
    tilt_deg: float = Field(ge=0.0)
    support_contacts: tuple[str, ...]
    """Bodies in contact with the floor at rest, measured."""
    undeclared_contacts: tuple[str, ...]
    """Bodies in contact with the floor at rest that the author did not declare as support members."""
    support_polygon_area_m2: float = Field(ge=0.0)
    duration_s: float = Field(gt=0.0)


class RecoveryTrialV1(Contract):
    """The body released from a perturbed initial state in its working
    stance, servos holding: does it come back to the stance on its own?"""

    trial_id: str = Field(min_length=1)
    perturbation: str = Field(min_length=1)
    roll_deg: float
    pitch_deg: float
    drop_m: float = Field(ge=0.0)
    recovered: bool
    final_tilt_deg: float = Field(ge=0.0)
    final_height_m: float
    supported: bool
    """Whether the manoeuvre is one the body is meant to survive without a controller; an unsupported one that fails is a finding, not a defect."""


class MobileBodyManifestV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    robot_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    base_kind: BaseKind
    base_body: str = Field(min_length=1)
    root_joint: str = Field(min_length=1)
    model_sha256: str = Field(min_length=64, max_length=64)
    mass_kg: float = Field(gt=0.0)
    joints: tuple[MobileJointV1, ...] = Field(min_length=1)
    limbs: tuple[LimbV1, ...] = Field(min_length=1)
    support_members: tuple[SupportMemberV1, ...] = Field(min_length=1)
    manipulators: tuple[ManipulatorV1, ...] = ()
    stances: dict[str, dict[str, float]]
    working_stance: str = Field(min_length=1)
    expected_standing_height_m: float = Field(gt=0.0)
    sensors: tuple[str, ...] = ()
    sensor_names: tuple[str, ...] = ()
    """The compiled model's sensors, by name."""
    footprint_m: tuple[float, float]
    """The body's extent across x and y in its working stance, measured."""
    traction: dict[str, object] = Field(default_factory=dict)
    provenance: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def internally_consistent(self) -> Self:
        joint_names = {j.name for j in self.joints}
        if len(joint_names) != len(self.joints):
            raise ValueError("joint names must be unique")
        for limb in self.limbs:
            missing = set(limb.joints) - joint_names
            if missing:
                raise ValueError(f"limb {limb.limb_id!r} names unknown joints {sorted(missing)}")
        if self.working_stance not in self.stances:
            raise ValueError(f"working stance {self.working_stance!r} is not a declared stance")
        for name, stance in self.stances.items():
            unknown = set(stance) - joint_names
            if unknown:
                raise ValueError(f"stance {name!r} names unknown joints {sorted(unknown)}")
        for manipulator in self.manipulators:
            if set(manipulator.grip_joints) - joint_names:
                raise ValueError(f"manipulator on {manipulator.limb!r} names unknown grip joints")
        return self


class MobileIntegrityReportV1(Contract):
    robot_id: str = Field(min_length=1)
    passed: bool
    violations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    body_count: int = Field(ge=1)
    joint_count: int = Field(ge=1)
    actuator_count: int = Field(ge=1)
    collider_count: int = Field(ge=1)
    free_joints: int = Field(ge=0)
    mass_kg: float = Field(gt=0.0)
    inertia_checked_bodies: int = Field(ge=0)
    stances: tuple[StanceMeasurementV1, ...] = ()
    recoveries: tuple[RecoveryTrialV1, ...] = ()


__all__ = ["BaseKind", "LimbV1", "ManipulatorV1", "MobileBodyManifestV1", "MobileIntegrityReportV1", "MobileJointV1", "RecoveryTrialV1", "StanceMeasurementV1", "SupportMemberV1"]
