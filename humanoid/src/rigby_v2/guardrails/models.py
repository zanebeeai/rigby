"""Stable typed outcomes for deterministic capability and asset validation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from rigby_core.contracts import Contract
from rigby_core.hashing import content_hash, validate_sha256


class FirewallOutcome(StrEnum):
    ALLOW = "allow"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"
    INVALID_ASSET = "invalid_asset"


class FirewallReason(StrEnum):
    SUPPORTED = "supported"
    DEFORMABLE_SIMULATION = "deformable_simulation"
    FLUID_SIMULATION = "fluid_simulation"
    MULTI_CHARACTER = "multi_character"
    DIRECT_ROBOT_EXECUTION = "direct_robot_execution"
    UNREACHABLE_TARGET = "unreachable_target"
    CONTRADICTORY_CONSTRAINTS = "contradictory_constraints"
    PATH_TRAVERSAL = "path_traversal"
    PLUGIN_INJECTION = "plugin_injection"
    NONFINITE_OR_INVALID_INERTIA = "nonfinite_or_invalid_inertia"
    RAW_CONCAVE_DYNAMIC_COLLIDER = "raw_concave_dynamic_collider"


class CapabilityDecisionV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    firewall_version: Literal["rigby-capability-firewall.v1"] = (
        "rigby-capability-firewall.v1"
    )
    prompt_sha256: str
    outcome: FirewallOutcome
    reason: FirewallReason
    rule_id: str = Field(min_length=1)
    matched_facts: tuple[str, ...] = ()
    decision_hash: str = ""

    @model_validator(mode="after")
    def seal(self) -> Self:
        validate_sha256(self.prompt_sha256)
        if (self.outcome is FirewallOutcome.ALLOW) != (
            self.reason is FirewallReason.SUPPORTED
        ):
            raise ValueError("only supported decisions may be allowed")
        expected = content_hash(self.model_dump(mode="json", exclude={"decision_hash"}))
        if self.decision_hash and self.decision_hash != expected:
            raise ValueError("capability decision hash does not match its content")
        object.__setattr__(self, "decision_hash", expected)
        return self


class AssetIntakeV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    asset_paths: tuple[str, ...] = ()
    native_plugins: tuple[str, ...] = ()
    mass_kg: float | None = None
    inertia_diagonal_kg_m2: tuple[float, float, float] | None = None
    dynamic: bool = False
    collision_representation: Literal[
        "primitive",
        "convex_decomposition",
        "convex_mesh",
        "raw_concave_mesh",
    ] = "primitive"


class CapabilityFirewallError(RuntimeError):
    """Raised before a model boundary when deterministic intake rejects."""

    def __init__(self, decision: CapabilityDecisionV1) -> None:
        super().__init__(f"{decision.outcome.value}: {decision.reason.value}")
        self.decision = decision
        self.outcome = decision.outcome
        self.reason = decision.reason

