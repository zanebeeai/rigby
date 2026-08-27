"""Pinned, offline-only embedding providers for the five-index library.

The production adapters deliberately do not download weights.  A model must
already exist in the Hugging Face cache at an exact revision and the complete
snapshot digest must match the release configuration before inference starts.
Tests and air-gapped deployments can inject an equivalent backend loader.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from rigby_v2.flywheel.schemas import RetrievalIndex
from rigby_v2.hashing import canonical_json_bytes, sha256_bytes, validate_sha256

from .embeddings import DeterministicDescriptorEmbedder, EmbeddingRouter, standard_namespace_configs
from .model_cache import (
    COSMOS_EMBED1_SPEC,
    SIGLIP2_SPEC,
    ModelCacheError,
    snapshot_tree_sha256,
    verify_snapshot,
)


COSMOS_EMBED1_REPOSITORY = COSMOS_EMBED1_SPEC.repository
SIGLIP2_REPOSITORY = SIGLIP2_SPEC.repository


class EmbeddingProviderFailureCode(StrEnum):
    OPTIONAL_DEPENDENCY_MISSING = "optional_dependency_missing"
    MODEL_NOT_CACHED = "model_not_cached"
    MODEL_IDENTITY_MISMATCH = "model_identity_mismatch"
    INVALID_PAYLOAD = "invalid_payload"
    BACKEND_FAILED = "backend_failed"
    INVALID_OUTPUT = "invalid_output"


class EmbeddingProviderError(RuntimeError):
    """Machine-readable, fail-closed provider failure."""

    def __init__(
        self,
        code: EmbeddingProviderFailureCode,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class EmbeddingModelIdentity:
    repository: str
    revision: str
    snapshot_sha256: str
    preprocessing_version: str
    dimensions: int

    def __post_init__(self) -> None:
        if not self.repository or "/" not in self.repository:
            raise ValueError("model repository must be an owner/name identifier")
        if len(self.revision) != 40 or any(value not in "0123456789abcdef" for value in self.revision.lower()):
            raise ValueError("model revision must be a pinned 40-character commit digest")
        object.__setattr__(self, "revision", self.revision.lower())
        object.__setattr__(self, "snapshot_sha256", validate_sha256(self.snapshot_sha256))
        if not self.preprocessing_version or self.dimensions <= 0:
            raise ValueError("preprocessing version and positive dimensions are required")

    @property
    def version_identity(self) -> str:
        payload = {
            "repository": self.repository,
            "revision": self.revision,
            "snapshot_sha256": self.snapshot_sha256,
            "preprocessing_version": self.preprocessing_version,
            "dimensions": self.dimensions,
        }
        return sha256_bytes(canonical_json_bytes(payload))


class EmbeddingInferenceBackend(Protocol):
    snapshot_sha256: str
    dimensions: int

    def encode(
        self,
        modality: str,
        payloads: Sequence[object],
        *,
        device: str,
    ) -> np.ndarray: ...


BackendLoader = Callable[[EmbeddingModelIdentity, str, str], EmbeddingInferenceBackend]


class HuggingFaceOfflineBackend:
    """Concrete local-cache backend shared by Cosmos-Embed1 and SigLIP2."""

    def __init__(self, identity: EmbeddingModelIdentity, device: str, modality: str) -> None:
        try:
            spec = {
                SIGLIP2_SPEC.repository: SIGLIP2_SPEC,
                COSMOS_EMBED1_SPEC.repository: COSMOS_EMBED1_SPEC,
            }[identity.repository]
            if identity.revision != spec.revision:
                raise ModelCacheError("identity revision is not an admitted immutable revision")
            snapshot, _ = verify_snapshot(spec)
        except (KeyError, ModelCacheError) as error:
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.MODEL_NOT_CACHED,
                "the pinned embedding snapshot is not present and verified; runtime downloads are disabled",
                details={"repository": identity.repository, "revision": identity.revision},
            ) from error
        self.snapshot_sha256 = snapshot_tree_sha256(snapshot)
        self.dimensions = identity.dimensions
        trust_remote_code = identity.repository == COSMOS_EMBED1_REPOSITORY
        try:
            transformers = importlib.import_module("transformers")
            self._torch = importlib.import_module("torch")
        except ImportError as error:
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.OPTIONAL_DEPENDENCY_MISSING,
                "embedding inference requires the optional torch, transformers, and huggingface_hub packages",
                details={"missing_module": getattr(error, "name", None)},
            ) from error
        try:
            self._processor = transformers.AutoProcessor.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )
            self._model = transformers.AutoModel.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            ).to(device)
            self._model.eval()
        except Exception as error:
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.BACKEND_FAILED,
                "the pinned embedding model could not be loaded",
                details={"repository": identity.repository, "modality": modality},
            ) from error

    @staticmethod
    def _pooled(output: Any) -> Any:
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state.mean(dim=1)
        if isinstance(output, (tuple, list)) and output:
            value = output[0]
            return value.mean(dim=1) if getattr(value, "ndim", 0) == 3 else value
        return output

    def encode(
        self,
        modality: str,
        payloads: Sequence[object],
        *,
        device: str,
    ) -> np.ndarray:
        processor_key = {"text": "text", "video": "videos", "image": "images"}[modality]
        try:
            inputs = self._processor(
                **{processor_key: list(payloads)},
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            feature_method = getattr(self._model, f"get_{modality}_features", None)
            with self._torch.inference_mode():
                output = feature_method(**inputs) if feature_method else self._pooled(self._model(**inputs))
            return np.asarray(output.detach().to("cpu", dtype=self._torch.float32).numpy())
        except EmbeddingProviderError:
            raise
        except Exception as error:
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.BACKEND_FAILED,
                "embedding inference failed",
                details={"modality": modality, "batch_size": len(payloads)},
            ) from error


def _default_backend_loader(
    identity: EmbeddingModelIdentity, device: str, modality: str
) -> EmbeddingInferenceBackend:
    return HuggingFaceOfflineBackend(identity, device, modality)


class _PinnedEmbeddingProvider:
    def __init__(
        self,
        identity: EmbeddingModelIdentity,
        modality: str,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        backend_loader: BackendLoader | None = None,
    ) -> None:
        if modality not in {"text", "video", "image"}:
            raise ValueError("unsupported embedding modality")
        if not device.strip() or batch_size <= 0:
            raise ValueError("device and positive batch size are required")
        self.identity = identity
        self.modality = modality
        self.device = device
        self.batch_size = batch_size
        self._loader = backend_loader or _default_backend_loader
        self._backend: EmbeddingInferenceBackend | None = None

    def _load_backend(self) -> EmbeddingInferenceBackend:
        if self._backend is None:
            try:
                backend = self._loader(self.identity, self.device, self.modality)
            except EmbeddingProviderError:
                raise
            except ImportError as error:
                raise EmbeddingProviderError(
                    EmbeddingProviderFailureCode.OPTIONAL_DEPENDENCY_MISSING,
                    "embedding backend dependency is unavailable",
                    details={"missing_module": getattr(error, "name", None)},
                ) from error
            except Exception as error:
                raise EmbeddingProviderError(
                    EmbeddingProviderFailureCode.BACKEND_FAILED,
                    "embedding backend loader failed",
                    details={"backend_error": type(error).__name__},
                ) from error
            try:
                observed = validate_sha256(backend.snapshot_sha256)
            except (AttributeError, ValueError) as error:
                raise EmbeddingProviderError(
                    EmbeddingProviderFailureCode.MODEL_IDENTITY_MISMATCH,
                    "embedding backend did not provide a valid snapshot digest",
                ) from error
            if observed != self.identity.snapshot_sha256 or backend.dimensions != self.identity.dimensions:
                raise EmbeddingProviderError(
                    EmbeddingProviderFailureCode.MODEL_IDENTITY_MISMATCH,
                    "loaded embedding backend does not match the pinned identity",
                    details={
                        "expected_sha256": self.identity.snapshot_sha256,
                        "observed_sha256": observed,
                        "expected_dimensions": self.identity.dimensions,
                        "observed_dimensions": backend.dimensions,
                    },
                )
            self._backend = backend
        return self._backend

    @staticmethod
    def _normalize(values: np.ndarray, expected_rows: int, dimensions: int) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (expected_rows, dimensions) or np.any(~np.isfinite(matrix)):
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.INVALID_OUTPUT,
                "embedding backend returned a wrong-shaped or non-finite matrix",
                details={"shape": tuple(matrix.shape), "expected": (expected_rows, dimensions)},
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.INVALID_OUTPUT,
                "embedding backend returned a zero vector",
            )
        return matrix / norms

    def _validate_payloads(self, payloads: Sequence[object]) -> None:
        if not payloads:
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.INVALID_PAYLOAD,
                "embedding batch cannot be empty",
            )
        if self.modality == "text" and any(not isinstance(value, str) or not value.strip() for value in payloads):
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.INVALID_PAYLOAD,
                "text embeddings require nonempty strings",
            )
        if self.modality == "video" and any(
            value is None or (hasattr(value, "__len__") and len(value) == 0)  # type: ignore[arg-type]
            for value in payloads
        ):
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.INVALID_PAYLOAD,
                "video embeddings require at least one frame",
            )

    def embed_batch(self, payloads: Sequence[object]) -> tuple[tuple[float, ...], ...]:
        self._validate_payloads(payloads)
        backend = self._load_backend()
        matrices = []
        for start in range(0, len(payloads), self.batch_size):
            chunk = payloads[start : start + self.batch_size]
            try:
                values = backend.encode(self.modality, chunk, device=self.device)
            except EmbeddingProviderError:
                raise
            except Exception as error:
                raise EmbeddingProviderError(
                    EmbeddingProviderFailureCode.BACKEND_FAILED,
                    "embedding backend failed while encoding a batch",
                    details={"modality": self.modality, "batch_size": len(chunk)},
                ) from error
            matrices.append(self._normalize(values, len(chunk), self.identity.dimensions))
        matrix = np.concatenate(matrices, axis=0)
        return tuple(tuple(float(value) for value in row) for row in matrix)

    def embed(self, payload: object) -> tuple[float, ...]:
        return self.embed_batch((payload,))[0]


class CosmosEmbed1TextProvider(_PinnedEmbeddingProvider):
    def __init__(self, identity: EmbeddingModelIdentity, **kwargs: object) -> None:
        if identity.repository != COSMOS_EMBED1_REPOSITORY:
            raise ValueError("Cosmos text provider requires the pinned Cosmos-Embed1 repository")
        super().__init__(identity, "text", **kwargs)  # type: ignore[arg-type]


class CosmosEmbed1VideoProvider(_PinnedEmbeddingProvider):
    def __init__(self, identity: EmbeddingModelIdentity, **kwargs: object) -> None:
        if identity.repository != COSMOS_EMBED1_REPOSITORY:
            raise ValueError("Cosmos video provider requires the pinned Cosmos-Embed1 repository")
        super().__init__(identity, "video", **kwargs)  # type: ignore[arg-type]


class SigLIP2KeyframeProvider(_PinnedEmbeddingProvider):
    def __init__(self, identity: EmbeddingModelIdentity, **kwargs: object) -> None:
        if identity.repository != SIGLIP2_REPOSITORY:
            raise ValueError("SigLIP2 provider requires the pinned SigLIP2 repository")
        super().__init__(identity, "image", **kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _frames(payload: object) -> tuple[object, ...]:
        if isinstance(payload, np.ndarray) and payload.ndim in {2, 3}:
            return (payload,)
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            frames = tuple(payload)
            if frames:
                return frames
        raise EmbeddingProviderError(
            EmbeddingProviderFailureCode.INVALID_PAYLOAD,
            "keyframe embeddings require one or more image frames",
        )

    def embed_batch(self, payloads: Sequence[object]) -> tuple[tuple[float, ...], ...]:
        if not payloads:
            raise EmbeddingProviderError(
                EmbeddingProviderFailureCode.INVALID_PAYLOAD, "embedding batch cannot be empty"
            )
        frame_groups = tuple(self._frames(payload) for payload in payloads)
        flattened = tuple(frame for group in frame_groups for frame in group)
        encoded = super().embed_batch(flattened)
        matrix = np.asarray(encoded, dtype=np.float64)
        pooled = []
        offset = 0
        for group in frame_groups:
            value = matrix[offset : offset + len(group)].mean(axis=0)
            offset += len(group)
            norm = float(np.linalg.norm(value))
            if norm <= 0:
                raise EmbeddingProviderError(
                    EmbeddingProviderFailureCode.INVALID_OUTPUT,
                    "mean-pooled keyframe embedding is zero",
                )
            pooled.append(tuple(float(item) for item in value / norm))
        return tuple(pooled)


def build_pinned_embedding_router(
    *,
    cosmos_identity: EmbeddingModelIdentity,
    siglip2_identity: EmbeddingModelIdentity,
    descriptor_spec_sha256: str,
    descriptor_dimensions: int = 256,
    device: str = "cpu",
    batch_size: int = 8,
    cosmos_backend_loader: BackendLoader | None = None,
    siglip2_backend_loader: BackendLoader | None = None,
) -> EmbeddingRouter:
    """Build the production five-channel router from pinned model identities."""

    if cosmos_identity.repository != COSMOS_EMBED1_REPOSITORY:
        raise ValueError("Cosmos identity uses the wrong repository")
    if siglip2_identity.repository != SIGLIP2_REPOSITORY:
        raise ValueError("SigLIP2 identity uses the wrong repository")
    configs = standard_namespace_configs(
        cosmos_model_sha256=cosmos_identity.snapshot_sha256,
        siglip2_model_sha256=siglip2_identity.snapshot_sha256,
        descriptor_spec_sha256=descriptor_spec_sha256,
        cosmos_dimensions=cosmos_identity.dimensions,
        siglip2_dimensions=siglip2_identity.dimensions,
        descriptor_dimensions=descriptor_dimensions,
        cosmos_revision=cosmos_identity.revision,
        siglip2_revision=siglip2_identity.revision,
        cosmos_text_preprocessing=cosmos_identity.preprocessing_version + ":text",
        cosmos_video_preprocessing=cosmos_identity.preprocessing_version + ":video",
        siglip2_preprocessing=siglip2_identity.preprocessing_version + ":keyframes",
    )
    descriptor = DeterministicDescriptorEmbedder(descriptor_dimensions)
    providers = {
        RetrievalIndex.TEXT: CosmosEmbed1TextProvider(
            cosmos_identity,
            device=device,
            batch_size=batch_size,
            backend_loader=cosmos_backend_loader,
        ),
        RetrievalIndex.VIDEO: CosmosEmbed1VideoProvider(
            cosmos_identity,
            device=device,
            batch_size=batch_size,
            backend_loader=cosmos_backend_loader,
        ),
        RetrievalIndex.KEYFRAME: SigLIP2KeyframeProvider(
            siglip2_identity,
            device=device,
            batch_size=batch_size,
            backend_loader=siglip2_backend_loader,
        ),
        RetrievalIndex.PROGRAM: descriptor,
        RetrievalIndex.MOTION: descriptor,
    }
    return EmbeddingRouter(configs, providers)
