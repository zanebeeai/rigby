from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from rigby_v2.flywheel.schemas import RetrievalIndex
from rigby_v2.library import (
    COSMOS_EMBED1_REPOSITORY,
    SIGLIP2_REPOSITORY,
    CosmosEmbed1TextProvider,
    CosmosEmbed1VideoProvider,
    EmbeddingModelIdentity,
    EmbeddingProviderError,
    EmbeddingProviderFailureCode,
    SigLIP2KeyframeProvider,
    build_pinned_embedding_router,
    snapshot_tree_sha256,
)

pytestmark = pytest.mark.fast


MODEL_HASH = "a" * 64
REVISION = "b" * 40


def _identity(repository: str, *, dimensions: int = 4, digest: str = MODEL_HASH):
    return EmbeddingModelIdentity(
        repository=repository,
        revision=REVISION,
        snapshot_sha256=digest,
        preprocessing_version="test-preprocessing-v1",
        dimensions=dimensions,
    )


@dataclass
class FakeBackend:
    snapshot_sha256: str = MODEL_HASH
    dimensions: int = 4
    calls: list[tuple[str, int, str]] = field(default_factory=list)

    def encode(self, modality, payloads, *, device):
        self.calls.append((modality, len(payloads), device))
        rows = []
        for index, payload in enumerate(payloads):
            signal = float(np.asarray(payload).mean()) if not isinstance(payload, str) else len(payload)
            rows.append([signal + 1.0, index + 1.0, 2.0, 3.0])
        return np.asarray(rows, dtype=np.float64)


def test_cosmos_text_and_video_are_pinned_batched_and_deterministically_normalized() -> None:
    backend = FakeBackend()
    loader_calls = []

    def loader(identity, device, modality):
        loader_calls.append((identity.version_identity, device, modality))
        return backend

    text = CosmosEmbed1TextProvider(
        _identity(COSMOS_EMBED1_REPOSITORY),
        device="cuda:1",
        batch_size=2,
        backend_loader=loader,
    )
    vectors = text.embed_batch(("one", "two words", "three words here"))
    assert len(vectors) == 3
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert backend.calls == [("text", 2, "cuda:1"), ("text", 1, "cuda:1")]
    assert len(loader_calls) == 1
    assert text.embed("repeat") == text.embed("repeat")

    video_backend = FakeBackend()
    video = CosmosEmbed1VideoProvider(
        _identity(COSMOS_EMBED1_REPOSITORY),
        batch_size=4,
        backend_loader=lambda *_: video_backend,
    )
    result = video.embed(np.ones((8, 2, 2, 3), dtype=np.uint8))
    assert len(result) == 4
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert video_backend.calls == [("video", 1, "cpu")]


def test_siglip_keyframes_are_encoded_in_batches_then_mean_pooled() -> None:
    backend = FakeBackend()
    provider = SigLIP2KeyframeProvider(
        _identity(SIGLIP2_REPOSITORY),
        batch_size=2,
        backend_loader=lambda *_: backend,
    )
    first = [np.zeros((2, 2, 3)), np.ones((2, 2, 3))]
    second = np.full((2, 2, 3), 2.0)
    vectors = provider.embed_batch((first, second))
    assert len(vectors) == 2
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])
    assert backend.calls == [("image", 2, "cpu"), ("image", 1, "cpu")]


@pytest.mark.parametrize(
    ("loader", "code"),
    (
        (
            lambda *_: (_ for _ in ()).throw(ImportError("missing", name="transformers")),
            EmbeddingProviderFailureCode.OPTIONAL_DEPENDENCY_MISSING,
        ),
        (
            lambda *_: FakeBackend(snapshot_sha256="f" * 64),
            EmbeddingProviderFailureCode.MODEL_IDENTITY_MISMATCH,
        ),
    ),
)
def test_provider_fails_closed_with_typed_loader_and_identity_errors(loader, code) -> None:
    provider = CosmosEmbed1TextProvider(
        _identity(COSMOS_EMBED1_REPOSITORY), backend_loader=loader
    )
    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed("hello")
    assert captured.value.code is code


def test_provider_rejects_invalid_payload_and_invalid_backend_output() -> None:
    provider = CosmosEmbed1TextProvider(
        _identity(COSMOS_EMBED1_REPOSITORY), backend_loader=lambda *_: FakeBackend()
    )
    with pytest.raises(EmbeddingProviderError) as captured:
        provider.embed(" ")
    assert captured.value.code is EmbeddingProviderFailureCode.INVALID_PAYLOAD

    class ZeroBackend(FakeBackend):
        def encode(self, modality, payloads, *, device):
            return np.zeros((len(payloads), self.dimensions))

    zero = CosmosEmbed1TextProvider(
        _identity(COSMOS_EMBED1_REPOSITORY), backend_loader=lambda *_: ZeroBackend()
    )
    with pytest.raises(EmbeddingProviderError) as captured:
        zero.embed("hello")
    assert captured.value.code is EmbeddingProviderFailureCode.INVALID_OUTPUT


def test_router_binds_model_revision_hash_and_real_provider_types() -> None:
    cosmos_backend = FakeBackend()
    siglip_backend = FakeBackend()
    router = build_pinned_embedding_router(
        cosmos_identity=_identity(COSMOS_EMBED1_REPOSITORY),
        siglip2_identity=_identity(SIGLIP2_REPOSITORY),
        descriptor_spec_sha256="c" * 64,
        descriptor_dimensions=4,
        cosmos_backend_loader=lambda *_: cosmos_backend,
        siglip2_backend_loader=lambda *_: siglip_backend,
    )
    assert router.configs[RetrievalIndex.TEXT].model_name.endswith(f"@{REVISION}")
    assert router.configs[RetrievalIndex.KEYFRAME].model_name.endswith(f"@{REVISION}")
    assert router.configs[RetrievalIndex.TEXT].preprocessing_version.endswith(":text")
    assert router.configs[RetrievalIndex.VIDEO].preprocessing_version.endswith(":video")
    assert router.configs[RetrievalIndex.KEYFRAME].preprocessing_version.endswith(":keyframes")
    assert isinstance(router.providers[RetrievalIndex.TEXT], CosmosEmbed1TextProvider)
    assert isinstance(router.providers[RetrievalIndex.VIDEO], CosmosEmbed1VideoProvider)
    assert isinstance(router.providers[RetrievalIndex.KEYFRAME], SigLIP2KeyframeProvider)


def test_snapshot_tree_hash_binds_paths_bytes_and_identity(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.bin").write_bytes(b"weights")
    first = snapshot_tree_sha256(tmp_path)
    assert first == snapshot_tree_sha256(tmp_path)
    (weights / "model.bin").write_bytes(b"changed")
    assert snapshot_tree_sha256(tmp_path) != first

    identity = _identity(COSMOS_EMBED1_REPOSITORY)
    assert len(identity.version_identity) == 64
    changed = EmbeddingModelIdentity(
        repository=identity.repository,
        revision="d" * 40,
        snapshot_sha256=identity.snapshot_sha256,
        preprocessing_version=identity.preprocessing_version,
        dimensions=identity.dimensions,
    )
    assert changed.version_identity != identity.version_identity
