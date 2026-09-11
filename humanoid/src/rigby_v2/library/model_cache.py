"""Immutable, explicitly prefetched model snapshots for library inference.

Runtime inference never contacts a model registry.  Operators must first run
this module's ``prefetch`` command, which downloads an exact commit into a
dedicated directory and seals every regular file in a provenance manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from rigby_core.hashing import canonical_json, canonical_json_bytes, hash_file, sha256_bytes


MANIFEST_NAME: Final = "rigby-model-manifest.v1.json"
DEFAULT_CACHE_ROOT: Final = Path.home() / ".cache" / "rigby_v2" / "models"


@dataclass(frozen=True, slots=True)
class PinnedModelSpec:
    key: str
    repository: str
    revision: str
    dimensions: int
    license_id: str
    license_url: str
    source_url: str
    preprocessing_version: str
    runtime_code: str


SIGLIP2_SPEC: Final = PinnedModelSpec(
    key="siglip2-base-patch16-224",
    repository="google/siglip2-base-patch16-224",
    revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    dimensions=768,
    license_id="Apache-2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0",
    source_url="https://huggingface.co/google/siglip2-base-patch16-224",
    preprocessing_version="transformers-auto-processor-siglip2-224-v1",
    runtime_code="transformers_builtin",
)

COSMOS_EMBED1_SPEC: Final = PinnedModelSpec(
    key="cosmos-embed1-336p",
    repository="nvidia/Cosmos-Embed1-336p",
    revision="0e8a28f7bf370f2dcb3b9b61d23e167d0d6b0e6f",
    dimensions=768,
    license_id="NVIDIA-Open-Model-License",
    license_url="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    source_url="https://huggingface.co/nvidia/Cosmos-Embed1-336p",
    preprocessing_version="cosmos-embed1-official-8frame-336p-v1",
    runtime_code="repository_custom_code",
)

MODEL_SPECS: Final = {item.key: item for item in (SIGLIP2_SPEC, COSMOS_EMBED1_SPEC)}


class ModelCacheError(RuntimeError):
    """A snapshot is unavailable, incomplete, or different from its seal."""


def _cache_root(value: Path | None = None) -> Path:
    configured = os.environ.get("RIGBY_V2_MODEL_CACHE")
    return (value or (Path(configured) if configured else DEFAULT_CACHE_ROOT)).resolve()


def snapshot_directory(spec: PinnedModelSpec, cache_root: Path | None = None) -> Path:
    return _cache_root(cache_root) / spec.key / spec.revision / "snapshot"


def manifest_path(spec: PinnedModelSpec, cache_root: Path | None = None) -> Path:
    return snapshot_directory(spec, cache_root).parent / MANIFEST_NAME


def snapshot_tree_entries(root: Path) -> tuple[dict[str, object], ...]:
    """Return the complete, sorted tree of model-owned regular files."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise ModelCacheError(f"model snapshot directory does not exist: {resolved}")
    entries: list[dict[str, object]] = []
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise ModelCacheError("model snapshots cannot contain symbolic links")
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        # Hugging Face local_dir metadata is an implementation cache, not part
        # of the upstream commit.  It is neither loaded nor admitted as model
        # content; all actual snapshot files remain covered.
        if relative.startswith(".cache/huggingface/"):
            continue
        entries.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": hash_file(path)}
        )
    if not entries:
        raise ModelCacheError("model snapshot contains no regular model files")
    return tuple(entries)


def snapshot_tree_sha256(root: Path) -> str:
    return sha256_bytes(canonical_json_bytes(snapshot_tree_entries(root)))


def _build_manifest(spec: PinnedModelSpec, root: Path) -> dict[str, object]:
    entries = snapshot_tree_entries(root)
    return {
        "schema_version": "rigby.model_snapshot_manifest.v1",
        "model": asdict(spec),
        "snapshot": {
            "file_count": len(entries),
            "total_bytes": sum(int(item["size_bytes"]) for item in entries),
            "tree_sha256": sha256_bytes(canonical_json_bytes(entries)),
            "files": entries,
        },
        "provenance": {
            "registry": "huggingface_hub",
            "immutable_revision": spec.revision,
            "network_at_runtime": False,
            "license_metadata_source": spec.source_url,
        },
    }


def verify_snapshot(
    spec: PinnedModelSpec, cache_root: Path | None = None
) -> tuple[Path, dict[str, object]]:
    root = snapshot_directory(spec, cache_root)
    seal_path = manifest_path(spec, cache_root)
    if not seal_path.is_file():
        raise ModelCacheError(f"model snapshot is not sealed: {seal_path}")
    try:
        manifest = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCacheError(f"model snapshot manifest cannot be read: {seal_path}") from error
    expected_model = asdict(spec)
    if manifest.get("schema_version") != "rigby.model_snapshot_manifest.v1":
        raise ModelCacheError("unsupported model snapshot manifest schema")
    if manifest.get("model") != expected_model:
        raise ModelCacheError("model snapshot provenance does not match the pinned model spec")
    observed = snapshot_tree_entries(root)
    snapshot = manifest.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("files") != list(observed):
        raise ModelCacheError("model snapshot file tree differs from its sealed manifest")
    digest = sha256_bytes(canonical_json_bytes(observed))
    if snapshot.get("tree_sha256") != digest:
        raise ModelCacheError("model snapshot tree digest differs from its sealed manifest")
    return root, manifest


def prefetch_snapshot(
    spec: PinnedModelSpec,
    cache_root: Path | None = None,
    *,
    replace: bool = False,
) -> dict[str, object]:
    """Noninteractively download and seal one exact public model commit."""

    if spec.repository == COSMOS_EMBED1_SPEC.repository:
        raise ModelCacheError(
            "Cosmos-Embed1 prefetch is disabled pending platform/resource and license approval"
        )
    destination = snapshot_directory(spec, cache_root)
    seal_path = manifest_path(spec, cache_root)
    if destination.exists() or seal_path.exists():
        if not replace:
            _, manifest = verify_snapshot(spec, cache_root)
            return manifest
        shutil.rmtree(destination.parent)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=spec.repository,
            revision=spec.revision,
            local_dir=destination,
            token=False,
        )
        manifest = _build_manifest(spec, destination)
        seal_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
        verify_snapshot(spec, cache_root)
        return manifest
    except Exception:
        if destination.parent.exists():
            shutil.rmtree(destination.parent)
        raise


def resource_report() -> dict[str, object]:
    """Record measured local capacity used for fail-closed admission decisions."""

    try:
        import psutil

        ram = int(psutil.virtual_memory().total)
    except ImportError:
        ram = 0
    gpu: dict[str, object] = {"available": False}
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "available": True,
                "name": properties.name,
                "total_bytes": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
    except ImportError:
        pass
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "system_ram_bytes": ram,
        "gpu": gpu,
    }


def cosmos_blocker_report() -> dict[str, object]:
    return {
        "schema_version": "rigby.model_admission_blocker.v1",
        "model": asdict(COSMOS_EMBED1_SPEC),
        "status": "unavailable",
        "measured_resources": resource_report(),
        "reasons": [
            "official model card lists Linux as the tested operating system; this host is Windows",
            "official model card lists Ampere and Hopper as tested GPU architectures; this host is Ada",
            "the 1B model requires repository custom code and the immutable repository security scan is incomplete",
            "the NVIDIA Open Model License requires an explicit project license review",
            "safe headroom for 8-frame 336p inference is not established on 8 GiB VRAM and 16 GB RAM",
        ],
        "substitution_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prefetch", "verify"):
        item = sub.add_parser(command)
        item.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
        item.add_argument("--cache-root", type=Path)
        if command == "prefetch":
            item.add_argument("--replace", action="store_true")
    blocker = sub.add_parser("cosmos-blocker")
    blocker.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "cosmos-blocker":
        report = cosmos_blocker_report()
        rendered = canonical_json(report) + "\n"
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered, encoding="utf-8", newline="\n")
        else:
            print(rendered, end="")
        return 0
    spec = MODEL_SPECS[arguments.model]
    try:
        if arguments.command == "prefetch":
            manifest = prefetch_snapshot(
                spec, arguments.cache_root, replace=arguments.replace
            )
        else:
            _, manifest = verify_snapshot(spec, arguments.cache_root)
    except ModelCacheError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
