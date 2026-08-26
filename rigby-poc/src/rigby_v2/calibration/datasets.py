"""License, namespace, and expert-review enforcement for research datasets."""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field

from rigby_v2.contracts import Contract
from rigby_v2.flywheel.schemas import DatasetNamespacePolicyV1
from rigby_v2.hashing import validate_sha256


DATASET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "v2" / "datasets"


class DatasetUse(StrEnum):
    SIGN_LANGUAGE_RESEARCH = "sign_language_research"
    LINGUISTIC_MOTION_ANALYSIS = "linguistic_motion_analysis"
    GENERIC_GESTURE = "generic_gesture"
    OBJECT_CONTACT_RESEARCH = "object_contact_research"
    ARTICULATED_INTERACTION_ANALYSIS = "articulated_interaction_analysis"
    INTERACTION_RETARGETING_RESEARCH = "interaction_retargeting_research"
    PRODUCTION_ANIMATION = "production_animation"


class ClaimCategory(StrEnum):
    GENERIC_GESTURE_CAPABILITY = "generic_gesture_capability"
    SIGNING_CAPABILITY = "signing_capability"
    ASL_CAPABILITY = "asl_capability"
    PRODUCTION_READY = "production_ready"
    OBJECT_CONTACT_RESEARCH_RESULT = "object_contact_research_result"


class DatasetManifest(Contract):
    schema_version: str = "1.0"
    dataset_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    namespace_policy: DatasetNamespacePolicyV1
    permitted_uses: tuple[DatasetUse, ...] = Field(min_length=1)
    prohibited_uses: tuple[DatasetUse, ...] = ()
    claims_requiring_deaf_expert_review: tuple[ClaimCategory, ...] = ()
    prohibited_claims: tuple[ClaimCategory, ...] = ()
    input_role: str = Field(min_length=1)
    media_downloaded: bool = False


class DatasetUseRequest(Contract):
    dataset_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    intended_use: DatasetUse
    claim_categories: tuple[ClaimCategory, ...] = ()
    research_context: bool
    deaf_expert_review_id: str | None = None
    deaf_expert_review_artifact_sha256: str | None = None


class DatasetAuthorization(Contract):
    dataset_id: str
    namespace: str
    intended_use: DatasetUse
    research_only: bool
    expert_review_verified: bool
    expert_review_artifact_sha256: str | None = None


class DatasetPolicyError(ValueError):
    pass


@lru_cache(maxsize=16)
def load_dataset_manifest(dataset_id: str, *, root: Path = DATASET_ROOT) -> DatasetManifest:
    if not dataset_id or any(token in dataset_id for token in ("/", "\\", "..")):
        raise DatasetPolicyError("invalid dataset ID")
    safe_root = root.resolve()
    path = (safe_root / f"{dataset_id}.json").resolve()
    try:
        path.relative_to(safe_root)
    except ValueError as error:
        raise DatasetPolicyError("dataset manifest path escapes registry") from error
    if not path.is_file():
        raise DatasetPolicyError(f"unknown dataset: {dataset_id}")
    manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.dataset_id != dataset_id:
        raise DatasetPolicyError("dataset manifest ID does not match filename")
    if manifest.media_downloaded:
        raise DatasetPolicyError("dataset registry must not embed or claim downloaded media")
    return manifest


def load_dataset_registry(*, root: Path = DATASET_ROOT) -> dict[str, DatasetManifest]:
    index_path = root / "registry.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    identifiers = data.get("datasets")
    if not isinstance(identifiers, list) or not identifiers:
        raise DatasetPolicyError("dataset registry index is invalid")
    manifests = {identifier: load_dataset_manifest(identifier, root=root) for identifier in identifiers}
    if len(manifests) != len(identifiers):
        raise DatasetPolicyError("dataset registry contains duplicate IDs")
    return manifests


def authorize_dataset_use(
    request: DatasetUseRequest,
    *,
    root: Path = DATASET_ROOT,
) -> DatasetAuthorization:
    manifest = load_dataset_manifest(request.dataset_id, root=root)
    policy = manifest.namespace_policy
    if request.namespace != policy.namespace:
        raise DatasetPolicyError(
            f"dataset {manifest.dataset_id} is confined to namespace {policy.namespace!r}"
        )
    if policy.research_only and not request.research_context:
        raise DatasetPolicyError("research-only dataset cannot be used outside a research context")
    if request.intended_use not in manifest.permitted_uses:
        raise DatasetPolicyError(
            f"use {request.intended_use.value!r} is not permitted for {manifest.dataset_id}"
        )
    if request.intended_use in manifest.prohibited_uses:
        raise DatasetPolicyError(f"use {request.intended_use.value!r} is explicitly prohibited")
    prohibited = set(request.claim_categories) & set(manifest.prohibited_claims)
    if prohibited:
        raise DatasetPolicyError(
            "prohibited dataset claim categories: "
            + ", ".join(sorted(value.value for value in prohibited))
        )
    review_required = bool(
        set(request.claim_categories) & set(manifest.claims_requiring_deaf_expert_review)
    )
    review_hash = None
    if review_required:
        if not request.deaf_expert_review_id or not request.deaf_expert_review_artifact_sha256:
            raise DatasetPolicyError(
                "signing capability claims require a documented Deaf expert review artifact"
            )
        try:
            review_hash = validate_sha256(request.deaf_expert_review_artifact_sha256)
        except ValueError as error:
            raise DatasetPolicyError("Deaf expert review artifact hash is invalid") from error
    return DatasetAuthorization(
        dataset_id=manifest.dataset_id,
        namespace=policy.namespace,
        intended_use=request.intended_use,
        research_only=policy.research_only,
        expert_review_verified=review_hash is not None,
        expert_review_artifact_sha256=review_hash,
    )
