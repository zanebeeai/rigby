from __future__ import annotations

from dataclasses import asdict

import pytest

from rigby_v2.hashing import canonical_json
from rigby_v2.library import (
    COSMOS_EMBED1_SPEC,
    SIGLIP2_SPEC,
    EmbeddingModelIdentity,
    EmbeddingProviderError,
    EmbeddingProviderFailureCode,
    ModelCacheError,
    SigLIP2KeyframeProvider,
    cosmos_blocker_report,
    prefetch_snapshot,
    snapshot_directory,
    snapshot_tree_sha256,
    verify_snapshot,
)
from rigby_v2.library.model_cache import MANIFEST_NAME, _build_manifest

pytestmark = pytest.mark.fast


def _sealed_fake_snapshot(tmp_path):
    root = snapshot_directory(SIGLIP2_SPEC, tmp_path)
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"test-weights")
    manifest = _build_manifest(SIGLIP2_SPEC, root)
    (root.parent / MANIFEST_NAME).write_text(
        canonical_json(manifest) + "\n", encoding="utf-8", newline="\n"
    )
    return root, manifest


def test_official_model_identities_are_exact_immutable_commits() -> None:
    assert SIGLIP2_SPEC.repository == "google/siglip2-base-patch16-224"
    assert SIGLIP2_SPEC.revision == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert SIGLIP2_SPEC.license_id == "Apache-2.0"
    assert SIGLIP2_SPEC.dimensions == 768
    assert COSMOS_EMBED1_SPEC.repository == "nvidia/Cosmos-Embed1-336p"
    assert COSMOS_EMBED1_SPEC.revision == "0e8a28f7bf370f2dcb3b9b61d23e167d0d6b0e6f"


def test_full_snapshot_tree_is_sealed_and_tampering_fails_closed(tmp_path) -> None:
    root, manifest = _sealed_fake_snapshot(tmp_path)
    resolved, verified = verify_snapshot(SIGLIP2_SPEC, tmp_path)
    assert resolved == root.resolve()
    assert canonical_json(verified) == canonical_json(manifest)
    assert verified["snapshot"]["tree_sha256"] == snapshot_tree_sha256(root)
    assert verified["model"] == asdict(SIGLIP2_SPEC)

    (root / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelCacheError, match="differs"):
        verify_snapshot(SIGLIP2_SPEC, tmp_path)


def test_snapshot_tree_rejects_symbolic_links(tmp_path) -> None:
    root, _ = _sealed_fake_snapshot(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not-model-owned")
    try:
        (root / "linked.bin").symlink_to(outside)
    except OSError:
        pytest.skip("symbolic-link creation is unavailable on this Windows host")

    with pytest.raises(ModelCacheError, match="symbolic links"):
        verify_snapshot(SIGLIP2_SPEC, tmp_path)


def test_cosmos_is_explicitly_unavailable_and_never_substituted(tmp_path) -> None:
    with pytest.raises(ModelCacheError, match="disabled"):
        prefetch_snapshot(COSMOS_EMBED1_SPEC, tmp_path)
    report = cosmos_blocker_report()
    assert report["status"] == "unavailable"
    assert report["substitution_allowed"] is False
    assert report["model"]["repository"] == "nvidia/Cosmos-Embed1-336p"


def test_runtime_does_not_download_when_verified_cache_is_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RIGBY_V2_MODEL_CACHE", str(tmp_path))
    identity = EmbeddingModelIdentity(
        repository=SIGLIP2_SPEC.repository,
        revision=SIGLIP2_SPEC.revision,
        snapshot_sha256="a" * 64,
        preprocessing_version=SIGLIP2_SPEC.preprocessing_version,
        dimensions=SIGLIP2_SPEC.dimensions,
    )
    provider = SigLIP2KeyframeProvider(identity)
    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed([[[0, 0, 0]]])
    assert captured.value.code is EmbeddingProviderFailureCode.MODEL_NOT_CACHED
