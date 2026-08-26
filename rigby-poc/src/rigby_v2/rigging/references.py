"""Audited external modeling references that cannot become Rigby's identity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from ..contracts import Contract


REFERENCE_ROOT = Path(__file__).resolve().parents[3] / "assets" / "v2" / "references"


class ReferenceLicensePolicyV1(Contract):
    repository_license_is_not_asset_license: Literal[True] = True
    per_asset_license_required: Literal[True] = True
    pin_upstream_commit_before_import: Literal[True] = True
    attribution_required_when_applicable: Literal[True] = True


class ReferenceAssetLicenseV1(Contract):
    asset_id: str = Field(min_length=1)
    upstream_path: str = Field(min_length=1)
    upstream_commit: str = Field(min_length=7)
    license_id: str = Field(min_length=1)
    license_source_url: str = Field(min_length=1)
    bundled: bool = False


class ModelingReferenceV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    record_version: str = Field(pattern=r"^\d{4}\.\d{2}\.\d{2}$")
    reference_id: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    role: Literal["modeling_and_testing_reference_only"]
    canonical_identity_allowed: Literal[False] = False
    bundled_assets: tuple[str, ...] = ()
    reference_scope: tuple[str, ...] = Field(min_length=1)
    license_policy: ReferenceLicensePolicyV1
    asset_licenses: tuple[ReferenceAssetLicenseV1, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def reference_only_and_license_complete(self) -> Self:
        if self.bundled_assets:
            raise ValueError("reference-only records cannot bundle upstream assets")
        ids = [item.asset_id for item in self.asset_licenses]
        if len(ids) != len(set(ids)):
            raise ValueError("reference asset license IDs must be unique")
        if any(item.bundled for item in self.asset_licenses):
            raise ValueError(
                "Menagerie assets cannot be bundled by a reference-only record"
            )
        return self


def load_menagerie_reference(
    path: Path = REFERENCE_ROOT / "mujoco_menagerie.json",
) -> ModelingReferenceV1:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    try:
        payload = path.read_bytes()
        expected = checksum_path.read_text(encoding="ascii").strip().split()[0].lower()
    except (OSError, IndexError) as error:
        raise ValueError("Menagerie reference record or checksum is missing") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError("Menagerie reference record failed checksum verification")
    return ModelingReferenceV1.model_validate_json(payload)
