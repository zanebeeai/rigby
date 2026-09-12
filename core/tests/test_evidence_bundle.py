"""Evidence integrity checks real bundles, including consistently resealed edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rigby_core import evidence
from rigby_core.evidence import EvidenceIntegrityError, verify_bundle, write_bundle


PAYLOADS = {
    "trace/states.json": b'{"qpos":[1,2,3]}',
    "model/robot.xml": b"<mujoco/>",
    "world/scene.json": b'{"seed":17}',
    "outcome.json": b'{"success":false}',
    "empty.bin": b"",
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def reseal(root: Path, raw: bytes) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    (root / "manifest.json").write_bytes(raw)
    (root / "seal.json").write_bytes(canonical({"sha256": digest}))
    return digest


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "bundle"
    digest = write_bundle(root, PAYLOADS, {"commit": "abc", "seed": 17, "label": "démonstration"})
    return root, digest


def test_round_trip_is_canonical_and_order_independent(bundle, tmp_path: Path) -> None:
    root, digest = bundle
    manifest = verify_bundle(root, digest.upper())
    assert manifest["schema"] == "rigby.evidence/1"
    assert manifest["metadata"] == {"commit": "abc", "seed": 17, "label": "démonstration"}
    assert manifest["files"]["empty.bin"] == {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    for name, payload in PAYLOADS.items():
        assert (root / name).read_bytes() == payload
        assert manifest["files"][name] == {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    assert (root / "manifest.json").read_bytes() == canonical(manifest)
    reordered = write_bundle(
        tmp_path / "other", dict(reversed(list(PAYLOADS.items()))),
        {"label": "démonstration", "seed": 17, "commit": "abc"},
    )
    assert digest == reordered


@pytest.mark.parametrize("name", list(PAYLOADS)[:4])
@pytest.mark.parametrize("mutation", ["replace", "truncate", "delete"])
def test_modified_or_missing_roles_are_rejected(bundle, name: str, mutation: str) -> None:
    root, digest = bundle
    target = root / name
    if mutation == "delete":
        target.unlink()
    elif mutation == "truncate":
        target.write_bytes(target.read_bytes()[:-1])
    else:
        original = target.read_bytes()
        target.write_bytes(b"X" + original[1:])  # Same size: size alone cannot verify it.
    with pytest.raises(EvidenceIntegrityError):
        verify_bundle(root, digest)


@pytest.mark.parametrize("addition", ["extra.bin", ".hidden", "trace/extra.bin", "directory"])
def test_complete_inventory_rejects_extra_content(bundle, addition: str) -> None:
    root, _ = bundle
    if addition == "directory":
        (root / addition).mkdir()
    else:
        (root / addition).write_bytes(b"extra")
    with pytest.raises(EvidenceIntegrityError, match="inventory"):
        verify_bundle(root)


def test_untrusted_resealing_cannot_defeat_an_external_digest(bundle) -> None:
    root, trusted = bundle
    manifest = verify_bundle(root)
    replacement = b'{"success":true}'
    (root / "outcome.json").write_bytes(replacement)
    manifest["files"]["outcome.json"] = {"bytes": len(replacement), "sha256": hashlib.sha256(replacement).hexdigest()}
    manifest["metadata"]["commit"] = "forged"
    replacement_digest = reseal(root, canonical(manifest))
    assert replacement_digest != trusted
    assert verify_bundle(root)["metadata"]["commit"] == "forged"
    with pytest.raises(EvidenceIntegrityError, match="trusted external digest"):
        verify_bundle(root, trusted)


@pytest.mark.parametrize("control", ["manifest.json", "seal.json"])
@pytest.mark.parametrize("mutation", ["delete", "replace"])
def test_control_file_tampering_is_rejected(bundle, control: str, mutation: str) -> None:
    root, _ = bundle
    if mutation == "delete":
        (root / control).unlink()
    else:
        (root / control).write_bytes(b"{}")
    with pytest.raises(EvidenceIntegrityError):
        verify_bundle(root)


@pytest.mark.parametrize("path", [
    "", "/absolute", "../escape", "a/../escape", "a/./b", "a//b", "a/", "C:/file", "C:file",
    "a\\b", "\\\\server\\share", "a:b", "a?b", "a*b", "a<b", 'a"b', "a|b", "a\nb",
    "trailing.", "trailing ", "NUL", "aux.txt", "com1.log", "LPT9", "CONIN$", "CONOUT$",
    "robot/PRN/file", "résumé.json", "a" * 256,
    "manifest.json", "MANIFEST.JSON", "seal.json", "Seal.json", "manifest.json/child",
])
def test_unsafe_paths_fail_without_creating_anything(tmp_path: Path, path: str) -> None:
    with pytest.raises(EvidenceIntegrityError):
        write_bundle(tmp_path / "bundle", {path: b"payload"}, {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("paths", [
    {"A": b"a", "a": b"b"},
    {"Robot/a": b"a", "robot/b": b"b"},
    {"a": b"a", "a/b": b"b"},
])
def test_case_aliases_and_file_directory_collisions_are_rejected(tmp_path: Path, paths) -> None:
    with pytest.raises(EvidenceIntegrityError):
        write_bundle(tmp_path / "bundle", paths, {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("metadata", [{"value": float("nan")}, {1: "key"}, {"value": {1, 2}}, []])
def test_non_json_metadata_is_rejected(tmp_path: Path, metadata) -> None:
    with pytest.raises(EvidenceIntegrityError):
        write_bundle(tmp_path / "bundle", PAYLOADS, metadata)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payloads", [{}, {"a": "text"}, {"a": bytearray(b"a")}, {1: b"a"}])
def test_payload_contract_is_enforced(tmp_path: Path, payloads) -> None:
    with pytest.raises(EvidenceIntegrityError):
        write_bundle(tmp_path / "bundle", payloads, {})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", [
    "schema", "extra", "metadata", "files-list", "files-empty", "record-extra", "record-list",
    "bytes-negative", "bytes-bool", "bytes-string", "hash-invalid", "path-escape", "path-case-alias",
])
def test_even_resealed_malformed_manifests_are_rejected(bundle, mutation: str) -> None:
    root, _ = bundle
    manifest = verify_bundle(root)
    record = manifest["files"]["outcome.json"]
    if mutation == "schema":
        manifest["schema"] = "rigby.evidence/99"
    elif mutation == "extra":
        manifest["extra"] = True
    elif mutation == "metadata":
        manifest["metadata"] = []
    elif mutation == "files-list":
        manifest["files"] = []
    elif mutation == "files-empty":
        manifest["files"] = {}
    elif mutation == "record-extra":
        record["ignored"] = True
    elif mutation == "record-list":
        manifest["files"]["outcome.json"] = []
    elif mutation.startswith("bytes-"):
        record["bytes"] = {"bytes-negative": -1, "bytes-bool": True, "bytes-string": "16"}[mutation]
    elif mutation == "hash-invalid":
        record["sha256"] = "not-a-digest"
    elif mutation == "path-escape":
        manifest["files"]["../outside"] = record.copy()
    elif mutation == "path-case-alias":
        manifest["files"]["Outcome.json"] = record.copy()
    reseal(root, canonical(manifest))
    with pytest.raises(EvidenceIntegrityError):
        verify_bundle(root)


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"a":1,"a":2}', b'{"value":NaN}', b'\xff', b' {} '])
def test_invalid_or_noncanonical_json_is_rejected_even_if_resealed(bundle, raw: bytes) -> None:
    root, _ = bundle
    reseal(root, raw)
    with pytest.raises(EvidenceIntegrityError):
        verify_bundle(root)


@pytest.mark.parametrize("digest", ["bad", "0" * 64, "g" * 64, 1])
def test_invalid_or_wrong_external_digest_is_rejected(bundle, digest) -> None:
    root, _ = bundle
    with pytest.raises(EvidenceIntegrityError):
        verify_bundle(root, digest)


@pytest.mark.parametrize("existing", ["directory", "empty-directory", "file"])
def test_existing_destination_is_never_clobbered(tmp_path: Path, existing: str) -> None:
    root = tmp_path / "bundle"
    if existing == "file":
        root.write_bytes(b"keep")
    else:
        root.mkdir()
        if existing == "directory":
            (root / "keep").write_bytes(b"keep")
    with pytest.raises(EvidenceIntegrityError, match="already exists"):
        write_bundle(root, PAYLOADS, {})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bundle"]
    if existing == "file":
        assert root.read_bytes() == b"keep"
    elif existing == "directory":
        assert (root / "keep").read_bytes() == b"keep"
    else:
        assert list(root.iterdir()) == []


def test_racing_destination_is_not_replaced_and_staging_is_cleaned(tmp_path: Path, monkeypatch) -> None:
    publish = evidence._publish

    def create_racing_destination(staging, destination):
        destination.mkdir()
        publish(staging, destination)

    monkeypatch.setattr(evidence, "_publish", create_racing_destination)
    with pytest.raises(EvidenceIntegrityError):
        write_bundle(tmp_path / "bundle", PAYLOADS, {})
    assert [path.name for path in tmp_path.iterdir()] == ["bundle"]
    assert list((tmp_path / "bundle").iterdir()) == []


def test_failure_halfway_through_writing_leaves_no_visible_or_staged_bundle(tmp_path: Path, monkeypatch) -> None:
    write_bytes = Path.write_bytes

    def fail_on_model(path, data):
        if path.name == "robot.xml":
            raise OSError("simulated full disk")
        return write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_on_model)
    with pytest.raises(EvidenceIntegrityError, match="simulated full disk"):
        write_bundle(tmp_path / "bundle", PAYLOADS, {})
    assert list(tmp_path.iterdir()) == []


def make_symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        pytest.skip(f"OS does not permit creating symlinks in this test process: {exc}")


def test_symlink_payload_is_rejected_even_when_bytes_match(bundle, tmp_path: Path) -> None:
    root, _ = bundle
    original = root / "outcome.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    make_symlink(original, outside)
    with pytest.raises(EvidenceIntegrityError, match="Symlink"):
        verify_bundle(root)


def test_symlink_directory_and_root_are_rejected(bundle, tmp_path: Path) -> None:
    root, _ = bundle
    alias = tmp_path / "alias"
    make_symlink(alias, root, directory=True)
    with pytest.raises(EvidenceIntegrityError, match="Symlink"):
        verify_bundle(alias)
    with pytest.raises(EvidenceIntegrityError, match="Symlink"):
        write_bundle(alias / "new", PAYLOADS, {})
    assert not (root / "new").exists()


def test_missing_root_and_parent_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(EvidenceIntegrityError, match="existing directory"):
        verify_bundle(tmp_path / "missing")
    with pytest.raises(EvidenceIntegrityError, match="parent"):
        write_bundle(tmp_path / "missing" / "bundle", PAYLOADS, {})
    assert list(tmp_path.iterdir()) == []
