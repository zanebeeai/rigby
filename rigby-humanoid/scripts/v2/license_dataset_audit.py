from __future__ import annotations

import json
from pathlib import Path

from rigby_v2.calibration import (
    ClaimCategory,
    DatasetPolicyError,
    DatasetUse,
    DatasetUseRequest,
    authorize_dataset_use,
    load_dataset_registry,
)
from rigby_v2.config import RuntimeSettings
from rigby_v2.hashing import hash_file
from rigby_v2.release_ops import seal_release_evidence


ROOT = Path(__file__).resolve().parents[2]


def _must_reject(request: DatasetUseRequest) -> str:
    try:
        authorize_dataset_use(request)
    except DatasetPolicyError as error:
        return str(error)
    raise RuntimeError(f"dataset policy unexpectedly authorized {request}")


def main() -> int:
    registry = load_dataset_registry()
    if set(registry) != {"signavatars", "how2sign", "humoto", "omniretarget"}:
        raise RuntimeError("dataset registry does not match the audited release set")
    if any(manifest.media_downloaded for manifest in registry.values()):
        raise RuntimeError("release registry claims downloaded research media")

    rejected: dict[str, str] = {}
    for dataset_id in ("signavatars", "how2sign"):
        namespace = registry[dataset_id].namespace_policy.namespace
        rejected[f"{dataset_id}:generic"] = _must_reject(
            DatasetUseRequest(
                dataset_id=dataset_id,
                namespace=namespace,
                intended_use=DatasetUse.GENERIC_GESTURE,
                research_context=True,
            )
        )
        rejected[f"{dataset_id}:production"] = _must_reject(
            DatasetUseRequest(
                dataset_id=dataset_id,
                namespace=namespace,
                intended_use=DatasetUse.PRODUCTION_ANIMATION,
                research_context=False,
            )
        )
        rejected[f"{dataset_id}:unreviewed_sign_claim"] = _must_reject(
            DatasetUseRequest(
                dataset_id=dataset_id,
                namespace=namespace,
                intended_use=DatasetUse.SIGN_LANGUAGE_RESEARCH,
                claim_categories=(ClaimCategory.SIGNING_CAPABILITY,),
                research_context=True,
            )
        )

    research_authorizations = {}
    for dataset_id in ("humoto", "omniretarget"):
        manifest = registry[dataset_id]
        use = (
            DatasetUse.OBJECT_CONTACT_RESEARCH
            if dataset_id == "humoto"
            else DatasetUse.INTERACTION_RETARGETING_RESEARCH
        )
        authorization = authorize_dataset_use(
            DatasetUseRequest(
                dataset_id=dataset_id,
                namespace=manifest.namespace_policy.namespace,
                intended_use=use,
                claim_categories=(ClaimCategory.OBJECT_CONTACT_RESEARCH_RESULT,),
                research_context=True,
            )
        )
        research_authorizations[dataset_id] = authorization.model_dump(mode="json")
        rejected[f"{dataset_id}:production"] = _must_reject(
            DatasetUseRequest(
                dataset_id=dataset_id,
                namespace=manifest.namespace_policy.namespace,
                intended_use=DatasetUse.PRODUCTION_ANIMATION,
                research_context=False,
            )
        )

    reference_path = ROOT / "assets" / "v2" / "references" / "mujoco_menagerie.json"
    expected_reference_hash = (
        reference_path.with_suffix(".json.sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    if hash_file(reference_path) != expected_reference_hash:
        raise RuntimeError("Menagerie reference policy checksum mismatch")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference["role"] != "modeling_and_testing_reference_only"
        or reference["canonical_identity_allowed"] is not False
        or reference["bundled_assets"] != []
        or reference["license_policy"]["per_asset_license_required"] is not True
    ):
        raise RuntimeError("Menagerie reference/license policy is unsafe")

    payload = {
        "dataset_manifest_sha256": {
            dataset_id: hash_file(
                ROOT / "assets" / "v2" / "datasets" / f"{dataset_id}.json"
            )
            for dataset_id in sorted(registry)
        },
        "media_downloaded": False,
        "sign_language_namespace": "research.language.sign",
        "generic_or_production_sign_use_rejected": True,
        "deaf_expert_review_obligation": (
            "a hash-bound Deaf expert review is mandatory before any signing/ASL "
            "claim; this release makes no signing/ASL claim and uses no sign media"
        ),
        "object_contact_research_authorizations": research_authorizations,
        "rejected_policy_cases": rejected,
        "menagerie_reference_policy_sha256": expected_reference_hash,
        "menagerie_assets_bundled": 0,
        "menagerie_canonical_identity_allowed": False,
    }
    settings = RuntimeSettings.from_env()
    evidence = seal_release_evidence(
        settings.project_root / "artifacts-v2" / "release-evidence",
        requirement_id="license_dataset_review",
        passed=True,
        payload=payload,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "evidence": evidence.artifact_path,
                "evidence_sha256": evidence.artifact_sha256,
                **payload,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
