from __future__ import annotations

import pytest

from rigby_v2.calibration import (
    ClaimCategory,
    DatasetPolicyError,
    DatasetUse,
    DatasetUseRequest,
    authorize_dataset_use,
    load_dataset_manifest,
    load_dataset_registry,
)


def _request(
    dataset_id: str,
    namespace: str,
    intended_use: DatasetUse,
    *,
    claims: tuple[ClaimCategory, ...] = (),
    research: bool = True,
    review: str | None = None,
    review_hash: str | None = None,
) -> DatasetUseRequest:
    return DatasetUseRequest(
        dataset_id=dataset_id,
        namespace=namespace,
        intended_use=intended_use,
        claim_categories=claims,
        research_context=research,
        deaf_expert_review_id=review,
        deaf_expert_review_artifact_sha256=review_hash,
    )


def test_registry_contains_policy_only_manifests_without_downloaded_media() -> None:
    registry = load_dataset_registry()
    assert set(registry) == {"signavatars", "how2sign", "humoto", "omniretarget"}
    assert all(manifest.namespace_policy.research_only for manifest in registry.values())
    assert all(not manifest.media_downloaded for manifest in registry.values())
    assert registry["signavatars"].namespace_policy.namespace == "research.language.sign"
    assert registry["how2sign"].namespace_policy.namespace == "research.language.sign"
    assert registry["humoto"].namespace_policy.namespace == "research.object_contact"
    assert registry["omniretarget"].namespace_policy.namespace == "research.object_contact"


@pytest.mark.parametrize("dataset_id", ("signavatars", "how2sign"))
def test_sign_data_is_language_namespaced_and_never_generic_gesture(
    dataset_id: str,
) -> None:
    authorized = authorize_dataset_use(
        _request(
            dataset_id,
            "research.language.sign",
            DatasetUse.SIGN_LANGUAGE_RESEARCH,
        )
    )
    assert authorized.research_only
    with pytest.raises(DatasetPolicyError, match="not permitted"):
        authorize_dataset_use(
            _request(
                dataset_id,
                "research.language.sign",
                DatasetUse.GENERIC_GESTURE,
                review="deaf-review-1",
            )
        )
    with pytest.raises(DatasetPolicyError, match="confined"):
        authorize_dataset_use(
            _request(dataset_id, "generic.gesture", DatasetUse.SIGN_LANGUAGE_RESEARCH)
        )


@pytest.mark.parametrize("claim", (ClaimCategory.SIGNING_CAPABILITY, ClaimCategory.ASL_CAPABILITY))
def test_signing_claims_require_documented_deaf_expert_review(
    claim: ClaimCategory,
) -> None:
    request = _request(
        "signavatars",
        "research.language.sign",
        DatasetUse.SIGN_LANGUAGE_RESEARCH,
        claims=(claim,),
    )
    with pytest.raises(DatasetPolicyError, match="Deaf expert review"):
        authorize_dataset_use(request)
    approved = authorize_dataset_use(
        request.model_copy(
            update={
                "deaf_expert_review_id": "deaf-review-2026-01",
                "deaf_expert_review_artifact_sha256": "d" * 64,
            }
        )
    )
    assert approved.expert_review_verified
    assert approved.expert_review_artifact_sha256 == "d" * 64


def test_deaf_review_id_without_bound_artifact_hash_is_not_evidence() -> None:
    with pytest.raises(DatasetPolicyError, match="review artifact"):
        authorize_dataset_use(
            _request(
                "signavatars",
                "research.language.sign",
                DatasetUse.SIGN_LANGUAGE_RESEARCH,
                claims=(ClaimCategory.SIGNING_CAPABILITY,),
                review="unbound-review-id",
            )
        )
    with pytest.raises(DatasetPolicyError, match="hash is invalid"):
        authorize_dataset_use(
            _request(
                "signavatars",
                "research.language.sign",
                DatasetUse.SIGN_LANGUAGE_RESEARCH,
                claims=(ClaimCategory.SIGNING_CAPABILITY,),
                review="review-id",
                review_hash="not-a-hash",
            )
        )


def test_research_only_data_rejects_nonresearch_context() -> None:
    with pytest.raises(DatasetPolicyError, match="research-only"):
        authorize_dataset_use(
            _request(
                "how2sign",
                "research.language.sign",
                DatasetUse.LINGUISTIC_MOTION_ANALYSIS,
                research=False,
            )
        )


@pytest.mark.parametrize(
    ("dataset_id", "allowed_use"),
    (
        ("humoto", DatasetUse.OBJECT_CONTACT_RESEARCH),
        ("omniretarget", DatasetUse.INTERACTION_RETARGETING_RESEARCH),
    ),
)
def test_interaction_sources_are_object_contact_inputs_only(
    dataset_id: str,
    allowed_use: DatasetUse,
) -> None:
    assert authorize_dataset_use(
        _request(dataset_id, "research.object_contact", allowed_use)
    ).intended_use is allowed_use
    with pytest.raises(DatasetPolicyError, match="not permitted"):
        authorize_dataset_use(
            _request(dataset_id, "research.object_contact", DatasetUse.GENERIC_GESTURE)
        )
    with pytest.raises(DatasetPolicyError, match="prohibited dataset claim"):
        authorize_dataset_use(
            _request(
                dataset_id,
                "research.object_contact",
                allowed_use,
                claims=(ClaimCategory.SIGNING_CAPABILITY,),
            )
        )


def test_dataset_id_cannot_traverse_registry() -> None:
    with pytest.raises(DatasetPolicyError, match="invalid"):
        load_dataset_manifest("../signavatars")
