"""Typed inputs and outcomes for deterministic simulation certification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from rigby_core.contracts import ContactEdgeV2, CoordinateSystem
from rigby_v2.simulation.runtime import SimulationRequest, SimulationResult

if TYPE_CHECKING:
    from .predicates import ObjectStatePredicate, PredicateEvaluation


class CertificationOutcome(StrEnum):
    CERTIFIED = "certified"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    SIMULATION_FAILED = "simulation_failed"
    ALL_CANDIDATES_REJECTED = "all_candidates_rejected"


class GateCode(StrEnum):
    NONFINITE_TRACE = "nonfinite_trace"
    TRACE_MISSING = "trace_missing"
    JOINT_POSITION_LIMIT = "joint_position_limit"
    VELOCITY_LIMIT = "velocity_limit"
    ACCELERATION_LIMIT = "acceleration_limit"
    ACTUATOR_EFFORT_LIMIT = "actuator_effort_limit"
    JOINT_EFFORT_LIMIT = "joint_effort_limit"
    JOINT_POWER_LIMIT = "joint_power_limit"
    PENETRATION = "penetration"
    UNEXPECTED_CONTACT = "unexpected_contact"
    CONTACT_ORDER = "contact_order"
    CONTACT_BREAK = "contact_break"
    CONTACT_DROPOUT = "contact_dropout"
    CONTACT_WINDOW = "contact_window"
    CONTACT_SLIP = "contact_slip"
    FALL = "fall"
    PELVIS_DRIFT = "pelvis_drift"
    FOOT_DRIFT = "foot_drift"
    FOOT_SLIP = "foot_slip"
    HIDDEN_WELD = "hidden_weld"
    MOCAP_BODY = "mocap_body"
    TELEPORT = "teleport"
    OBJECT_PREDICATE = "object_predicate"
    REPEAT_DISAGREEMENT = "repeat_disagreement"
    EXPORT_REIMPORT = "export_reimport"
    UNSUPPORTED_MODEL = "unsupported_model"
    UNSUPPORTED_PREDICATE = "unsupported_predicate"
    ROTATION_NORMALIZATION = "rotation_normalization"
    ASSET_BINDING = "asset_binding"
    COORDINATE_BINDING = "coordinate_binding"


@dataclass(frozen=True, order=True)
class ContactPair:
    first: str
    second: str

    def __post_init__(self) -> None:
        if not self.first or not self.second or self.first == self.second:
            raise ValueError("contact pairs require two distinct geom names")
        if self.first > self.second:
            first, second = self.second, self.first
            object.__setattr__(self, "first", first)
            object.__setattr__(self, "second", second)

    @classmethod
    def of(cls, first: str, second: str) -> "ContactPair":
        return cls(first=first, second=second)


@dataclass(frozen=True)
class SupportFoot:
    body_name: str
    geom_names: frozenset[str]

    def __post_init__(self) -> None:
        if not self.body_name or not self.geom_names:
            raise ValueError("support foot requires a body and at least one geom")


@dataclass(frozen=True)
class GateViolation:
    code: GateCode
    message: str
    actual: float | str | None = None
    limit: float | str | None = None
    time_s: float | None = None


@dataclass(frozen=True)
class ExportValidation:
    passed: bool
    message: str = ""


class ExportReimportValidator(Protocol):
    def __call__(self, result: SimulationResult) -> ExportValidation | bool: ...


@dataclass(frozen=True)
class CertificationPolicy:
    pelvis_body: str = "pelvis"
    torso_body: str = "torso"
    support_feet: tuple[SupportFoot, ...] = ()
    ground_geom_names: frozenset[str] = frozenset({"floor"})
    allowed_contact_pairs: frozenset[ContactPair] | None = None
    expected_contact_order: tuple[ContactPair, ...] = ()
    min_contact_force_n: float = 0.05
    contact_window_tolerance_s: float = 1.0 / 240.0
    unplanned_contact_persistence_s: float = 0.02
    max_penetration_m: float = 0.002
    joint_position_tolerance: float = 1e-4
    default_joint_velocity_limit: float = 20.0
    default_joint_acceleration_limit: float = 500.0
    default_actuator_effort_limit: float = 200.0
    default_joint_effort_limit: float = 300.0
    default_joint_power_limit: float = 2_000.0
    joint_velocity_limits: Mapping[str, float] = field(default_factory=dict)
    joint_acceleration_limits: Mapping[str, float] = field(default_factory=dict)
    actuator_effort_limits: Mapping[str, float] = field(default_factory=dict)
    joint_effort_limits: Mapping[str, float] = field(default_factory=dict)
    joint_power_limits: Mapping[str, float] = field(default_factory=dict)
    max_pelvis_drop_m: float = 0.20
    max_pelvis_drift_m: float = 0.25
    max_torso_tilt_deg: float = 50.0
    max_foot_drift_m: float = 0.08
    max_foot_slip_m: float = 0.025
    teleport_tolerance_m_or_rad: float = 1e-5
    repeat_count: int = 3
    require_export_reimport: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.max_penetration_m,
            self.joint_position_tolerance,
            self.default_joint_velocity_limit,
            self.default_joint_acceleration_limit,
            self.default_actuator_effort_limit,
            self.default_joint_effort_limit,
            self.default_joint_power_limit,
            self.max_pelvis_drop_m,
            self.max_pelvis_drift_m,
            self.max_torso_tilt_deg,
            self.max_foot_drift_m,
            self.max_foot_slip_m,
            self.teleport_tolerance_m_or_rad,
            self.contact_window_tolerance_s,
            self.unplanned_contact_persistence_s,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("certification limits must be positive")
        if self.min_contact_force_n < 0:
            raise ValueError("minimum contact force cannot be negative")
        if self.repeat_count < 3:
            raise ValueError("certification requires at least three repeats")


@dataclass(frozen=True)
class CandidateCertificationRequest:
    simulation: SimulationRequest
    policy: CertificationPolicy
    predicates: tuple["ObjectStatePredicate", ...] = ()
    export_reimport: ExportReimportValidator | None = None
    planned_contacts: tuple[ContactEdgeV2, ...] = ()
    expected_model_hash: str | None = None
    rig_asset_hash: str | None = None
    scene_rig_asset_hash: str | None = None
    rig_coordinate_system: CoordinateSystem | None = None
    scene_coordinate_system: CoordinateSystem | None = None


@dataclass(frozen=True)
class CertificationResult:
    outcome: CertificationOutcome
    candidate_id: str
    violations: tuple[GateViolation, ...]
    predicate_evaluations: tuple["PredicateEvaluation", ...]
    simulation_runs: tuple[SimulationResult, ...]

    @property
    def certified(self) -> bool:
        return self.outcome is CertificationOutcome.CERTIFIED


@dataclass(frozen=True)
class CandidateSelectionResult:
    outcome: CertificationOutcome
    selected_candidate_id: str | None
    candidates: tuple[CertificationResult, ...]
