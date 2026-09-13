"""Portable, atomically published evidence bundles with explicit trust anchors.

A bundle's files are immutable by convention: this API never overwrites a
destination, and verification checks the complete inventory and every byte.
The seal detects accidental corruption, but an attacker can replace the files
and reseal the entire bundle. Retain the returned manifest digest in a trusted
external record and pass it as ``expected_digest`` to bind that provenance.

Payload roles belong to the caller. Paths use ASCII POSIX-relative names that
are safe on Windows, macOS and Linux; metadata is ordinary finite JSON. The
destination's parent must exist. Publication is atomic and refuses an existing
destination, including an empty directory. Concurrent hostile filesystem
mutation during verification is outside this file-format contract.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path


SCHEMA = "rigby.evidence/1"
_CONTROL_FILES = {"manifest.json", "seal.json"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DEVICES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


class EvidenceIntegrityError(ValueError):
    """Evidence is malformed, unsafe, incomplete or inconsistent with its seal."""


def _validate_json(value: object) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for child in value:
            _validate_json(child)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():
            _validate_json(child)
        return
    raise EvidenceIntegrityError("Metadata and manifests require finite JSON values and string keys")


def _canonical(value: object) -> bytes:
    _validate_json(value)
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceIntegrityError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_canonical(raw: bytes, name: str) -> dict[str, object]:
    parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if type(parsed) is not dict or _canonical(parsed) != raw:
        raise EvidenceIntegrityError(f"{name} must be a canonical JSON object")
    return parsed


def _path_parts(name: str) -> tuple[str, ...]:
    if type(name) is not str or not name:
        raise EvidenceIntegrityError("Payload paths must be nonempty strings")
    parts = tuple(name.split("/"))
    for part in parts:
        if (
            not part
            or part in (".", "..")
            or len(part) > 255
            or part[-1] in " ."
            or any(ord(char) < 32 or ord(char) > 126 or char in '<>:"\\|?*' for char in part)
            or part.split(".", 1)[0].upper() in _DEVICES
        ):
            raise EvidenceIntegrityError(f"Unsafe or nonportable payload path: {name!r}")
    if parts[0].lower() in _CONTROL_FILES:
        raise EvidenceIntegrityError(f"Reserved bundle path: {name!r}")
    return parts


def _validate_paths(names: object) -> set[str]:
    nodes: dict[str, str] = {}
    files: set[str] = set()
    directories: set[str] = set()
    for name in names:
        parts = _path_parts(name)
        for depth in range(1, len(parts) + 1):
            prefix = "/".join(parts[:depth])
            alias = prefix.lower()
            if alias in nodes and nodes[alias] != prefix:
                raise EvidenceIntegrityError(f"Case-aliased bundle paths: {nodes[alias]!r}, {prefix!r}")
            nodes[alias] = prefix
            if depth < len(parts):
                directories.add(prefix)
        if name in files:
            raise EvidenceIntegrityError(f"Duplicate payload path: {name!r}")
        files.add(name)
    if files & directories:
        raise EvidenceIntegrityError("A payload path cannot also be a directory")
    return directories


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _reject_link_ancestors(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if _is_link(info):
            raise EvidenceIntegrityError(f"Symlinks and reparse points are forbidden: {candidate}")


def _publish(staging: Path, destination: Path) -> None:
    """Rename without replacement, including a racing empty destination."""
    if os.name == "nt":
        os.rename(staging, destination)  # Windows rename never replaces a destination.
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staging), os.fsencode(destination), 4)
    else:
        raise EvidenceIntegrityError("This platform lacks atomic no-replace directory publication")
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def write_bundle(destination: Path, payloads: dict[str, bytes], metadata: dict[str, object]) -> str:
    """Publish a new bundle and return its canonical manifest's SHA-256 digest.

    ``manifest.json`` contains ``schema``, caller ``metadata`` and ``files``;
    each file entry records ``sha256`` and ``bytes``. ``seal.json`` contains
    the manifest's ``sha256``. Neither self hash authenticates a resealed bundle.
    """
    staging: Path | None = None
    try:
        destination = Path(destination).absolute()
        _reject_link_ancestors(destination)
        if os.path.lexists(destination):
            raise EvidenceIntegrityError(f"Evidence destination already exists: {destination}")
        if not destination.parent.is_dir():
            raise EvidenceIntegrityError("Evidence destination parent must be an existing directory")
        if type(payloads) is not dict or not payloads:
            raise EvidenceIntegrityError("An evidence bundle requires at least one payload")
        if type(metadata) is not dict:
            raise EvidenceIntegrityError("Evidence metadata must be a JSON object")
        _validate_paths(payloads)
        if any(type(payload) is not bytes for payload in payloads.values()):
            raise EvidenceIntegrityError("Every payload must be bytes")
        manifest = {
            "schema": SCHEMA,
            "metadata": metadata,
            "files": {
                name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                for name, payload in payloads.items()
            },
        }
        manifest_bytes = _canonical(manifest)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        for name, payload in payloads.items():
            target = staging.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        (staging / "seal.json").write_bytes(_canonical({"sha256": digest}))
        _publish(staging, destination)
        staging = None
        return digest
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        if isinstance(exc, EvidenceIntegrityError):
            raise
        raise EvidenceIntegrityError(f"Cannot write evidence bundle: {exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def _inventory(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                info = entry.stat(follow_symlinks=False)
                if _is_link(info):
                    raise EvidenceIntegrityError(f"Symlink or reparse point in bundle: {relative}")
                if stat.S_ISDIR(info.st_mode):
                    directories.add(relative)
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    files.add(relative)
                else:
                    raise EvidenceIntegrityError(f"Nonregular entry in bundle: {relative}")
    return files, directories


def verify_bundle(root: Path, expected_digest: str | None = None) -> dict:
    """Check every file against the seal and an optional trusted external digest.

    Returns the validated manifest; raises ``EvidenceIntegrityError`` on any
    missing, extra, malformed or modified content. Expected digests may use
    either hex case. Without an external digest, a consistently resealed
    replacement is valid content but carries no authenticated provenance.
    """
    try:
        root = Path(root).absolute()
        _reject_link_ancestors(root)
        if not root.is_dir():
            raise EvidenceIntegrityError("Evidence bundle root must be an existing directory")
        actual_files, actual_directories = _inventory(root)
        if not _CONTROL_FILES <= actual_files:
            raise EvidenceIntegrityError("Evidence manifest.json or seal.json is missing")
        manifest_bytes = (root / "manifest.json").read_bytes()
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        seal = _parse_canonical((root / "seal.json").read_bytes(), "seal.json")
        if set(seal) != {"sha256"} or seal["sha256"] != digest:
            raise EvidenceIntegrityError("Evidence manifest does not match its seal")
        if expected_digest is not None:
            if type(expected_digest) is not str or not _SHA256.fullmatch(expected_digest.lower()):
                raise EvidenceIntegrityError("Expected digest must be a 64-character SHA-256 hex digest")
            if digest != expected_digest.lower():
                raise EvidenceIntegrityError("Evidence manifest differs from the trusted external digest")
        manifest = _parse_canonical(manifest_bytes, "manifest.json")
        if set(manifest) != {"schema", "metadata", "files"} or manifest["schema"] != SCHEMA:
            raise EvidenceIntegrityError("Unsupported or malformed evidence manifest")
        if type(manifest["metadata"]) is not dict:
            raise EvidenceIntegrityError("Evidence metadata must be a JSON object")
        records = manifest["files"]
        if type(records) is not dict or not records:
            raise EvidenceIntegrityError("Evidence manifest must describe at least one payload")
        expected_directories = _validate_paths(records)
        _validate_paths(actual_files - _CONTROL_FILES)
        if actual_files != set(records) | _CONTROL_FILES or actual_directories != expected_directories:
            raise EvidenceIntegrityError("Evidence file/directory inventory differs from the manifest")
        for name, record in records.items():
            if (
                type(record) is not dict
                or set(record) != {"sha256", "bytes"}
                or type(record["bytes"]) is not int
                or record["bytes"] < 0
                or type(record["sha256"]) is not str
                or not _SHA256.fullmatch(record["sha256"])
            ):
                raise EvidenceIntegrityError(f"Malformed evidence file record: {name}")
            digest = hashlib.sha256()
            size = 0
            with root.joinpath(*name.split("/")).open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
            if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise EvidenceIntegrityError(f"Evidence payload bytes/hash mismatch: {name}")
        return manifest
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        if isinstance(exc, EvidenceIntegrityError):
            raise
        raise EvidenceIntegrityError(f"Cannot verify evidence bundle: {exc}") from exc
