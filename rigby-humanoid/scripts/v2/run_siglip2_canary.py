from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import sys
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from rigby_v2.hashing import canonical_json, canonical_json_bytes, hash_file
from rigby_v2.library import (
    SIGLIP2_SPEC,
    EmbeddingModelIdentity,
    SigLIP2KeyframeProvider,
    manifest_path,
    verify_snapshot,
)


_FRAME_NAME = re.compile(r"^frames/[0-9]{6}\.png$")
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_FRAME_BYTES = 16 * 1024 * 1024


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_verified_frame(
    archive_path: Path, expected_archive_sha256: str, frame_name: str
) -> tuple[bytes, str]:
    archive_path = archive_path.resolve()
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ValueError("source archive must be a regular file")
    if archive_path.stat().st_size > _MAX_ARCHIVE_BYTES:
        raise ValueError("source archive exceeds the canary size limit")
    if hash_file(archive_path) != expected_archive_sha256.lower():
        raise ValueError("source archive does not match its content-addressed hash")
    if not _FRAME_NAME.fullmatch(frame_name):
        raise ValueError("frame name must use the canonical frames/NNNNNN.png form")
    with zipfile.ZipFile(archive_path) as archive:
        matches = [item for item in archive.infolist() if item.filename == frame_name]
        if len(matches) != 1:
            raise ValueError("source archive must contain exactly one requested frame")
        item = matches[0]
        if item.file_size <= 0 or item.file_size > _MAX_FRAME_BYTES:
            raise ValueError("source frame size is outside the safe range")
        frame_bytes = archive.read(item)
    if not frame_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("source frame is not a PNG")
    return frame_bytes, _sha256_bytes(frame_bytes)


def _write_new(path: Path, document: dict[str, object]) -> str:
    path = path.resolve()
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = canonical_json(document) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hash_file(path)
    seal = path.with_suffix(path.suffix + ".sha256")
    with seal.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeat-identical SigLIP2 inference on one verified MuJoCo "
            "task-closeup frame. Runtime registry access remains disabled."
        )
    )
    parser.add_argument("source_archive", type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--frame", default="frames/000150.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _, manifest = verify_snapshot(SIGLIP2_SPEC)
    tree_sha256 = str(manifest["snapshot"]["tree_sha256"])
    frame_bytes, frame_sha256 = _load_verified_frame(
        arguments.source_archive,
        arguments.archive_sha256,
        arguments.frame,
    )
    with Image.open(io.BytesIO(frame_bytes)) as source:
        source.load()
        image = np.asarray(source.convert("RGB"), dtype=np.uint8)

    identity = EmbeddingModelIdentity(
        repository=SIGLIP2_SPEC.repository,
        revision=SIGLIP2_SPEC.revision,
        snapshot_sha256=tree_sha256,
        preprocessing_version=SIGLIP2_SPEC.preprocessing_version,
        dimensions=SIGLIP2_SPEC.dimensions,
    )
    started = perf_counter()
    provider = SigLIP2KeyframeProvider(
        identity,
        device=arguments.device,
        batch_size=1,
    )
    first = provider.embed(image)
    second = provider.embed(image)
    elapsed = perf_counter() - started
    first_values = np.asarray(first, dtype=np.float64)
    second_values = np.asarray(second, dtype=np.float64)
    max_delta = float(np.max(np.abs(first_values - second_values)))
    norm = float(np.linalg.norm(first_values))
    if max_delta != 0.0 or abs(norm - 1.0) > 1e-12:
        raise RuntimeError("SigLIP2 canary was not repeat-identical and normalized")

    import torch
    import transformers

    document: dict[str, object] = {
        "schema_version": "rigby.siglip2_keyframe_canary.v1",
        "status": "passed",
        "model": {
            "repository": SIGLIP2_SPEC.repository,
            "revision": SIGLIP2_SPEC.revision,
            "snapshot_tree_sha256": tree_sha256,
            "snapshot_manifest_sha256": hash_file(manifest_path(SIGLIP2_SPEC)),
            "preprocessing_version": SIGLIP2_SPEC.preprocessing_version,
            "dimensions": SIGLIP2_SPEC.dimensions,
            "license_id": SIGLIP2_SPEC.license_id,
        },
        "source": {
            "kind": "mujoco_task_closeup_raw_frame",
            "archive_sha256": arguments.archive_sha256.lower(),
            "frame_name": arguments.frame,
            "frame_sha256": frame_sha256,
        },
        "inference": {
            "device": arguments.device,
            "repeat_count": 2,
            "maximum_repeat_delta": max_delta,
            "l2_norm": norm,
            "vector_sha256": _sha256_bytes(canonical_json_bytes(first)),
            "vector": list(first),
            "elapsed_seconds": elapsed,
            "network_at_runtime": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    evidence_sha256 = _write_new(arguments.output, document)
    print(
        json.dumps(
            {
                "passed": True,
                "evidence": str(arguments.output.resolve()),
                "evidence_sha256": evidence_sha256,
                "vector_sha256": document["inference"]["vector_sha256"],  # type: ignore[index]
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
