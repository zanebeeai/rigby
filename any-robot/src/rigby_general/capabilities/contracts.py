"""Evidence-qualified capabilities; source assertions are not physical success."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, JsonValue, field_serializer, model_validator
from rigby_core.contracts import Contract
from rigby_core.hashing import content_hash

from ..contracts import RobotAssetManifestV1


class CapabilityFactV1(Contract):
    fact_id: str
    subject: str
    value: JsonValue
    units: str | None = None
    basis: Literal["source_declared", "compiled_model", "measured_kinematics", "simulation_assumption", "unverified"]
    method: str
    uncertainty: str

    @field_serializer("value")
    def serialize_value(self, value):
        # Core contracts freeze JSON arrays as tuples after validation. Convert
        # those immutable containers back to JSON arrays for this recursive type.
        def plain(item):
            if isinstance(item, dict):
                return {key: plain(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [plain(child) for child in item]
            return item
        return plain(value)

    @model_validator(mode="after")
    def finite(self) -> Self:
        content_hash(self)
        return self


class BodyCapabilityV1(Contract):
    capability: str
    subject: str = "body"
    status: Literal["observed_in_model", "unknown", "unsupported_by_runtime"]
    enabled: bool = False
    evidence_ids: tuple[str, ...]
    condition: str

    @model_validator(mode="after")
    def no_unproven_enablement(self) -> Self:
        if self.enabled and self.status != "observed_in_model":
            raise ValueError("Unproven capabilities cannot be enabled")
        return self


class BodyCapabilityManifestV1(Contract):
    schema_version: Literal["body.capabilities.v1"] = "body.capabilities.v1"
    intake_profile: Literal["structural_urdf.v1"] = "structural_urdf.v1"
    source_urdf_sha256: str
    canonical_urdf_sha256: str
    package_sha256: str
    assets: dict[str, str]
    link_aliases: dict[str, str]
    joint_aliases: dict[str, str]
    robot: RobotAssetManifestV1
    facts: tuple[CapabilityFactV1, ...] = Field(min_length=1)
    capabilities: tuple[BodyCapabilityV1, ...] = Field(min_length=1)
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def evidence_references(self) -> Self:
        identifiers = {f.fact_id for f in self.facts}
        if len(identifiers) != len(self.facts):
            raise ValueError("Fact identifiers must be unique")
        for capability in self.capabilities:
            if not capability.evidence_ids or set(capability.evidence_ids) - identifiers:
                raise ValueError("Every capability must cite recorded evidence or an explicit evidence gap")
        content_hash(self)
        return self

    def capability_hash(self) -> str:
        """Exclude original labels and source bytes, retain all scientific facts.

        Runtime IDs are structural, so complete robot geometry/morphology,
        controls, sensors, evidence and limitations remain in the comparison.
        """
        return content_hash({"profile": self.intake_profile, "canonical_urdf_sha256": self.canonical_urdf_sha256,
            "assets": self.assets, "robot": self.robot, "facts": self.facts,
            "capabilities": self.capabilities, "limitations": self.limitations})
