"""Deterministic ZIP transport for the existing evidence bundle contract.

Archives add no evidence schema: they contain the exact manifest, seal and
payload files. Retain both returned archive hashes and original manifest
digests externally. ZIP bytes are deterministic for the same Python/zlib
compression implementation; unpacked bundle bytes are independent of it.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import zipfile
from pathlib import Path

from .evidence import (
    EvidenceIntegrityError,
    _CONTROL_FILES,
    _SHA256,
    _parse_canonical,
    _publish,
    _reject_link_ancestors,
    _validate_paths,
    verify_bundle,
    write_bundle,
)


def _digest(value: str, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value.lower()):
        raise EvidenceIntegrityError(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _new_destination(path: Path) -> Path:
    path = Path(path).absolute()
    _reject_link_ancestors(path)
    if os.path.lexists(path):
        raise EvidenceIntegrityError(f"Evidence destination already exists: {path}")
    if not path.parent.is_dir():
        raise EvidenceIntegrityError("Evidence destination parent must be an existing directory")
    return path


def pack_bundle(root: Path, archive: Path, expected_digest: str | None = None) -> str:
    """Verify and archive a bundle without replacing an existing archive.

    Returns the SHA-256 of the complete ZIP. The archive's parent must exist
    and the archive must be outside the source bundle. Publication is atomic.
    """
    temporary: Path | None = None
    try:
        root = Path(root).absolute()
        manifest = verify_bundle(root, expected_digest)
        archive = _new_destination(archive)
        if archive.resolve().is_relative_to(root.resolve()):
            raise EvidenceIntegrityError("An archive cannot be created inside its source bundle")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{archive.name}.staging-", dir=archive.parent)
        temporary = Path(temporary_name)
        os.close(descriptor)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
            for name in sorted(set(manifest["files"]) | _CONTROL_FILES):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                output.writestr(
                    info, root.joinpath(*name.split("/")).read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED, compresslevel=9,
                )
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        _publish(temporary, archive)
        temporary = None
        return digest
    except (OSError, TypeError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        if isinstance(exc, EvidenceIntegrityError):
            raise
        raise EvidenceIntegrityError(f"Cannot pack evidence bundle: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _archive_payloads(source: zipfile.ZipFile, expected_digest: str) -> tuple[dict[str, bytes], dict]:
    entries = source.infolist()
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)):
        raise EvidenceIntegrityError("Duplicate ZIP entry names are forbidden")
    if not _CONTROL_FILES <= set(names):
        raise EvidenceIntegrityError("Archive manifest.json or seal.json is missing")
    _validate_paths(set(names) - _CONTROL_FILES)
    for entry in entries:
        kind = stat.S_IFMT(entry.external_attr >> 16)
        if (
            entry.orig_filename != entry.filename
            or entry.is_dir()
            or entry.external_attr & 0x10
            or kind not in (0, stat.S_IFREG)
            or entry.flag_bits & 1
        ):
            raise EvidenceIntegrityError(f"Unsafe ZIP entry type or name: {entry.orig_filename!r}")
    raw = source.read("manifest.json")
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise EvidenceIntegrityError("Archive manifest differs from the trusted external digest")
    manifest = _parse_canonical(raw, "manifest.json")
    seal = _parse_canonical(source.read("seal.json"), "seal.json")
    if seal != {"sha256": expected_digest}:
        raise EvidenceIntegrityError("Archive manifest does not match its seal")
    records = manifest.get("files")
    if type(records) is not dict or not records:
        raise EvidenceIntegrityError("Archive manifest must describe at least one payload")
    _validate_paths(records)
    if set(names) != set(records) | _CONTROL_FILES:
        raise EvidenceIntegrityError("Archive inventory differs from the manifest")
    for name, record in records.items():
        if (
            type(record) is not dict
            or type(record.get("bytes")) is not int
            or record["bytes"] < 0
            or source.getinfo(name).file_size != record["bytes"]
        ):
            raise EvidenceIntegrityError(f"Archive payload size differs from the manifest: {name}")
    return {name: source.read(name) for name in records}, manifest


def unpack_bundle(
    archive: Path, destination: Path, *, archive_sha256: str, expected_digest: str
) -> str:
    """Verify two external digests, then atomically publish the original bundle.

    Hashes the archive before opening ZIP. Entries are never extracted using
    their paths: payload bytes pass through ``write_bundle`` and its portable
    path contract in a private temporary bundle. The complete original
    manifest digest is verified before the destination becomes visible.
    """
    try:
        archive_sha256 = _digest(archive_sha256, "Archive hash")
        expected_digest = _digest(expected_digest, "Manifest digest")
        destination = _new_destination(destination)
        archive = Path(archive).absolute()
        _reject_link_ancestors(archive)
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != archive_sha256:
                raise EvidenceIntegrityError("Archive hash differs from the trusted archive hash")
            stream.seek(0)
            with zipfile.ZipFile(stream, "r") as source:
                payloads, manifest = _archive_payloads(source, expected_digest)
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}.unpacking-", dir=destination.parent) as work:
            staged = Path(work) / "bundle"
            digest = write_bundle(staged, payloads, manifest.get("metadata"))
            if digest != expected_digest:
                raise EvidenceIntegrityError("Reconstructed payloads or manifest differ from the trusted external digest")
            verify_bundle(staged, expected_digest)
            _publish(staged, destination)
        return digest
    except (
        OSError, TypeError, ValueError, RecursionError, RuntimeError,
        zipfile.BadZipFile, zipfile.LargeZipFile,
    ) as exc:
        if isinstance(exc, EvidenceIntegrityError):
            raise
        raise EvidenceIntegrityError(f"Cannot unpack evidence bundle: {exc}") from exc
