"""ZIP transport must preserve externally anchored evidence before publication."""

from __future__ import annotations

import hashlib
import json
import stat
import warnings
import zipfile
from pathlib import Path

import pytest

from rigby_core import evidence_archive
from rigby_core.evidence import EvidenceIntegrityError, verify_bundle, write_bundle
from rigby_core.evidence_archive import pack_bundle, unpack_bundle


PAYLOADS = {
    "trace/states.json": b'{"qpos":[1,2,3]}\n' * 4000,
    "model/robot.xml": b"<mujoco/>",
    "world/scene.json": b'{"seed":17}',
    "outcome.json": b'{"success":false}',
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def packed(tmp_path: Path):
    root = tmp_path / "source"
    manifest_digest = write_bundle(root, PAYLOADS, {"seed": 17})
    archive = tmp_path / "evidence.zip"
    archive_digest = pack_bundle(root, archive, manifest_digest)
    return root, archive, manifest_digest, archive_digest


def altered_archive(archive: Path, mutation) -> str:
    with zipfile.ZipFile(archive) as source:
        entries = [(entry, source.read(entry)) for entry in source.infolist()]
    mutation(entries)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Deliberate duplicate-entry corruption.
        with zipfile.ZipFile(archive, "w") as output:
            for entry, payload in entries:
                output.writestr(entry, payload)
    return sha(archive)


def test_deterministic_compressed_round_trip_preserves_all_bundle_bytes(packed, tmp_path: Path) -> None:
    root, archive, manifest_digest, archive_digest = packed
    other = tmp_path / "second.zip"
    assert pack_bundle(root, other) == archive_digest == sha(archive)
    assert other.read_bytes() == archive.read_bytes()
    assert archive.stat().st_size < sum(len(value) for value in PAYLOADS.values()) / 10
    with zipfile.ZipFile(archive) as source:
        assert source.namelist() == sorted(set(PAYLOADS) | {"manifest.json", "seal.json"})
        assert all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in source.infolist())
        assert all(entry.compress_type == zipfile.ZIP_DEFLATED for entry in source.infolist())
    destination = tmp_path / "restored"
    assert unpack_bundle(
        archive, destination, archive_sha256=archive_digest.upper(), expected_digest=manifest_digest.upper()
    ) == manifest_digest
    assert verify_bundle(destination, manifest_digest) == verify_bundle(root, manifest_digest)
    for name in [*PAYLOADS, "manifest.json", "seal.json"]:
        assert (destination / name).read_bytes() == (root / name).read_bytes()


def test_archive_hash_is_checked_before_zip_is_opened(packed, tmp_path: Path, monkeypatch) -> None:
    _, archive, manifest_digest, archive_digest = packed
    archive.write_bytes(archive.read_bytes() + b"tampered")

    def must_not_open(*args, **kwargs):
        pytest.fail("ZIP parser was invoked before archive hash validation")

    monkeypatch.setattr(evidence_archive.zipfile, "ZipFile", must_not_open)
    with pytest.raises(EvidenceIntegrityError, match="Archive hash"):
        unpack_bundle(archive, tmp_path / "restored", archive_sha256=archive_digest, expected_digest=manifest_digest)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["evidence.zip", "source"]


@pytest.mark.parametrize("mutation", ["payload", "resealed-payload", "traversal", "extra", "duplicate", "case-alias", "symlink", "directory"])
def test_rehashed_archive_corruption_never_publishes_a_destination(packed, tmp_path: Path, mutation: str) -> None:
    _, archive, manifest_digest, _ = packed

    def corrupt(entries):
        if mutation in {"payload", "resealed-payload"}:
            index = next(index for index, (entry, _) in enumerate(entries) if entry.filename == "outcome.json")
            entry, data = entries[index]
            replacement = b"X" + data[1:]
            entries[index] = (entry, replacement)
            if mutation == "resealed-payload":
                manifest_index = next(i for i, (entry, _) in enumerate(entries) if entry.filename == "manifest.json")
                info, raw = entries[manifest_index]
                manifest = json.loads(raw)
                manifest["files"]["outcome.json"]["sha256"] = hashlib.sha256(replacement).hexdigest()
                changed = canonical(manifest)
                entries[manifest_index] = (info, changed)
                seal_index = next(i for i, (entry, _) in enumerate(entries) if entry.filename == "seal.json")
                entries[seal_index] = (entries[seal_index][0], canonical({"sha256": hashlib.sha256(changed).hexdigest()}))
        elif mutation == "duplicate":
            entries.append(entries[0])
        else:
            name = {
                "traversal": "../escaped", "extra": "extra.bin", "case-alias": "Outcome.json",
                "symlink": "link", "directory": "directory/",
            }[mutation]
            info = zipfile.ZipInfo(name)
            if mutation == "symlink":
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            entries.append((info, b"outside"))

    archive_digest = altered_archive(archive, corrupt)
    with pytest.raises(EvidenceIntegrityError):
        unpack_bundle(archive, tmp_path / "restored", archive_sha256=archive_digest, expected_digest=manifest_digest)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["evidence.zip", "source"]


def test_rehashed_archive_with_an_invalid_seal_is_rejected(packed, tmp_path: Path) -> None:
    _, archive, manifest_digest, _ = packed

    def corrupt(entries):
        index = next(i for i, (entry, _) in enumerate(entries) if entry.filename == "seal.json")
        entries[index] = (entries[index][0], canonical({"sha256": "0" * 64}))

    archive_digest = altered_archive(archive, corrupt)
    with pytest.raises(EvidenceIntegrityError, match="seal"):
        unpack_bundle(archive, tmp_path / "restored", archive_sha256=archive_digest, expected_digest=manifest_digest)
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("target_kind", ["file", "empty-directory"])
def test_existing_destinations_and_archives_are_not_overwritten(packed, tmp_path: Path, target_kind: str) -> None:
    root, archive, manifest_digest, archive_digest = packed
    original = archive.read_bytes()
    with pytest.raises(EvidenceIntegrityError, match="already exists"):
        pack_bundle(root, archive)
    assert archive.read_bytes() == original
    destination = tmp_path / "restored"
    if target_kind == "file":
        destination.write_bytes(b"keep")
    else:
        destination.mkdir()
    with pytest.raises(EvidenceIntegrityError, match="already exists"):
        unpack_bundle(archive, destination, archive_sha256=archive_digest, expected_digest=manifest_digest)
    if target_kind == "file":
        assert destination.read_bytes() == b"keep"
    else:
        assert list(destination.iterdir()) == []


def test_failed_publication_cleans_the_private_bundle(packed, tmp_path: Path, monkeypatch) -> None:
    _, archive, manifest_digest, archive_digest = packed

    def fail_publication(staged, destination):
        assert verify_bundle(staged, manifest_digest)
        raise OSError("simulated publication failure")

    monkeypatch.setattr(evidence_archive, "_publish", fail_publication)
    with pytest.raises(EvidenceIntegrityError, match="simulated publication failure"):
        unpack_bundle(archive, tmp_path / "restored", archive_sha256=archive_digest, expected_digest=manifest_digest)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["evidence.zip", "source"]


def test_pack_checks_input_integrity_and_does_not_modify_the_bundle(packed, tmp_path: Path) -> None:
    root, _, manifest_digest, _ = packed
    with pytest.raises(EvidenceIntegrityError, match="inside"):
        pack_bundle(root, root / "inside.zip")
    assert verify_bundle(root, manifest_digest)
    (root / "outcome.json").write_bytes(b"modified")
    with pytest.raises(EvidenceIntegrityError, match="mismatch"):
        pack_bundle(root, tmp_path / "bad.zip", manifest_digest)
    assert not (tmp_path / "bad.zip").exists()


@pytest.mark.parametrize("field", ["archive_sha256", "expected_digest"])
def test_both_external_hashes_are_required_and_validated(packed, tmp_path: Path, field: str) -> None:
    _, archive, manifest_digest, archive_digest = packed
    hashes = {"archive_sha256": archive_digest, "expected_digest": manifest_digest}
    hashes[field] = None
    with pytest.raises(EvidenceIntegrityError, match="SHA-256"):
        unpack_bundle(archive, tmp_path / "restored", **hashes)
    assert not (tmp_path / "restored").exists()
