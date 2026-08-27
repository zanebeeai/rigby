from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .contracts import ArtifactRefV1
from .errors import ArtifactIntegrityError
from .hashing import canonical_json_bytes, hash_file, sha256_bytes, validate_sha256


@runtime_checkable
class ArtifactStore(Protocol):
    def put_bytes(
        self,
        value: bytes,
        *,
        media_type: str = "application/octet-stream",
        filename: str | None = None,
    ) -> ArtifactRefV1: ...

    def put_file(
        self,
        source: Path,
        *,
        media_type: str = "application/octet-stream",
        filename: str | None = None,
    ) -> ArtifactRefV1: ...

    def resolve(self, reference: ArtifactRefV1, *, verify: bool = True) -> Path: ...

    def read_bytes(self, reference: ArtifactRefV1, *, verify: bool = True) -> bytes: ...


class ContentAddressedArtifactStore:
    """Immutable SHA-256 object store with atomic, deduplicated writes."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.objects = self.root / "objects" / "sha256"
        self.objects.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        digest = validate_sha256(digest)
        path = self.objects / digest[:2] / digest[2:]
        if self.objects not in path.resolve().parents:
            raise ArtifactIntegrityError("Artifact path escaped the store root")
        return path

    def put_bytes(
        self,
        value: bytes,
        *,
        media_type: str = "application/octet-stream",
        filename: str | None = None,
    ) -> ArtifactRefV1:
        digest = sha256_bytes(value)
        destination = self._path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self._verify_path(destination, digest, len(value))
        else:
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent, prefix=".incoming-", delete=False
                ) as stream:
                    temporary_name = stream.name
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.replace(temporary_name, destination)
                    temporary_name = None
                except OSError:
                    if not destination.exists():
                        raise
                    self._verify_path(destination, digest, len(value))
            finally:
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
        return ArtifactRefV1(
            sha256=digest,
            size_bytes=len(value),
            media_type=media_type,
            filename=filename,
        )

    def put_json(self, value: Any, *, filename: str | None = None) -> ArtifactRefV1:
        return self.put_bytes(
            canonical_json_bytes(value),
            media_type="application/json",
            filename=filename,
        )

    def put_file(
        self,
        source: Path,
        *,
        media_type: str = "application/octet-stream",
        filename: str | None = None,
    ) -> ArtifactRefV1:
        size_bytes = source.stat().st_size
        digest = hash_file(source)
        destination = self._path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self._verify_path(destination, digest, size_bytes)
        else:
            temporary_name: str | None = None
            try:
                with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
                    mode="wb", dir=destination.parent, prefix=".incoming-", delete=False
                ) as output_stream:
                    temporary_name = output_stream.name
                    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                try:
                    os.replace(temporary_name, destination)
                    temporary_name = None
                except OSError:
                    if not destination.exists():
                        raise
                    self._verify_path(destination, digest, size_bytes)
            finally:
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
        return ArtifactRefV1(
            sha256=digest,
            size_bytes=size_bytes,
            media_type=media_type,
            filename=filename or source.name,
        )

    def _verify_path(self, path: Path, digest: str, size_bytes: int) -> None:
        actual_size = path.stat().st_size
        actual_hash = hash_file(path)
        if actual_size != size_bytes or actual_hash != digest:
            raise ArtifactIntegrityError(
                "Artifact content does not match its reference",
                details={
                    "expected_sha256": digest,
                    "actual_sha256": actual_hash,
                    "expected_size": size_bytes,
                    "actual_size": actual_size,
                },
            )

    def resolve(self, reference: ArtifactRefV1, *, verify: bool = True) -> Path:
        path = self._path(reference.sha256)
        if not path.is_file():
            raise FileNotFoundError(reference.sha256)
        if verify:
            self._verify_path(path, reference.sha256, reference.size_bytes)
        return path

    def read_bytes(self, reference: ArtifactRefV1, *, verify: bool = True) -> bytes:
        return self.resolve(reference, verify=verify).read_bytes()

    def exists(self, reference: ArtifactRefV1, *, verify: bool = False) -> bool:
        try:
            self.resolve(reference, verify=verify)
            return True
        except (FileNotFoundError, ArtifactIntegrityError):
            return False
