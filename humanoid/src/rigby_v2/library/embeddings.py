"""Versioned embedding configuration with pluggable, no-download providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np

from rigby_v2.flywheel.schemas import RetrievalIndex
from rigby_core.hashing import canonical_json

from .models import DistanceMetric, EmbeddingNamespaceConfig


class EmbeddingProvider(Protocol):
    def embed(self, payload: object) -> Sequence[float]: ...


@dataclass(frozen=True)
class RetrievalQuery:
    prompt: str
    structured_program: object
    task_space_motion: object
    video: object
    salient_keyframes: object

    def payload(self, index: RetrievalIndex) -> object:
        return {
            RetrievalIndex.TEXT: self.prompt,
            RetrievalIndex.PROGRAM: self.structured_program,
            RetrievalIndex.MOTION: self.task_space_motion,
            RetrievalIndex.VIDEO: self.video,
            RetrievalIndex.KEYFRAME: self.salient_keyframes,
        }[index]


class DeterministicDescriptorEmbedder:
    """Feature-hashing embedder for structured program and phase/contact data."""

    def __init__(self, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, payload: object) -> Sequence[float]:
        canonical = canonical_json(payload)
        vector = np.zeros(self.dimensions, dtype=np.float64)
        tokens = canonical.replace("{", " ").replace("}", " ").replace(",", " ").split()
        for token in tokens or [canonical]:
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class EmbeddingRouter:
    def __init__(
        self,
        configs: Mapping[RetrievalIndex, EmbeddingNamespaceConfig],
        providers: Mapping[RetrievalIndex, EmbeddingProvider],
    ) -> None:
        required = set(RetrievalIndex)
        if set(configs) != required or set(providers) != required:
            raise ValueError("all five independent retrieval indexes must be configured")
        self.configs = dict(configs)
        self.providers = dict(providers)

    def embed(self, index: RetrievalIndex, payload: object) -> tuple[float, ...]:
        values = np.asarray(self.providers[index].embed(payload), dtype=np.float64)
        config = self.configs[index]
        if values.shape != (config.dimensions,) or np.any(~np.isfinite(values)):
            raise ValueError(
                f"{index.value} embedder must return {config.dimensions} finite values"
            )
        if config.distance_metric is DistanceMetric.COSINE:
            norm = float(np.linalg.norm(values))
            if norm == 0:
                raise ValueError(f"{index.value} embedder returned a zero cosine vector")
            values = values / norm
        return tuple(float(value) for value in values)


def standard_namespace_configs(
    *,
    cosmos_model_sha256: str,
    siglip2_model_sha256: str,
    descriptor_spec_sha256: str,
    cosmos_dimensions: int = 768,
    siglip2_dimensions: int = 768,
    descriptor_dimensions: int = 256,
    cosmos_revision: str | None = None,
    siglip2_revision: str | None = None,
    cosmos_text_preprocessing: str = "cosmos-embed1-text-v1",
    cosmos_video_preprocessing: str = "cosmos-embed1-8frame-336p-v1",
    siglip2_preprocessing: str = "rigby-salient-hand-crop-v1",
) -> dict[RetrievalIndex, EmbeddingNamespaceConfig]:
    """Pin the five namespaces without loading any model weights."""

    def config(
        index: RetrievalIndex,
        namespace: str,
        model: str,
        digest: str,
        dimensions: int,
        preprocessing: str,
    ) -> EmbeddingNamespaceConfig:
        return EmbeddingNamespaceConfig(
            index=index,
            namespace=namespace,
            version="1.0",
            model_name=model,
            model_sha256=digest,
            preprocessing_version=preprocessing,
            dimensions=dimensions,
        )

    return {
        RetrievalIndex.TEXT: config(
            RetrievalIndex.TEXT,
            "cosmos_embed1_text",
            "nvidia/Cosmos-Embed1-336p" + (f"@{cosmos_revision}" if cosmos_revision else ""),
            cosmos_model_sha256,
            cosmos_dimensions,
            cosmos_text_preprocessing,
        ),
        RetrievalIndex.VIDEO: config(
            RetrievalIndex.VIDEO,
            "cosmos_embed1_video",
            "nvidia/Cosmos-Embed1-336p" + (f"@{cosmos_revision}" if cosmos_revision else ""),
            cosmos_model_sha256,
            cosmos_dimensions,
            cosmos_video_preprocessing,
        ),
        RetrievalIndex.KEYFRAME: config(
            RetrievalIndex.KEYFRAME,
            "siglip2_salient_keyframe",
            "google/siglip2-base-patch16-224" + (f"@{siglip2_revision}" if siglip2_revision else ""),
            siglip2_model_sha256,
            siglip2_dimensions,
            siglip2_preprocessing,
        ),
        RetrievalIndex.PROGRAM: config(
            RetrievalIndex.PROGRAM,
            "rigby_program_descriptor",
            "rigby/deterministic-program-descriptor",
            descriptor_spec_sha256,
            descriptor_dimensions,
            "motion-program-v2-feature-hash-v1",
        ),
        RetrievalIndex.MOTION: config(
            RetrievalIndex.MOTION,
            "rigby_phase_contact_descriptor",
            "rigby/deterministic-phase-contact-descriptor",
            descriptor_spec_sha256,
            descriptor_dimensions,
            "phase-contact-task-space-v1",
        ),
    }
