"""`ContentAddressedArtifactStore`: the object store both product stacks write to.

Untested before this file. It is 168 lines that every simulation result, every
piece of evidence and every staged rig passes through, and its contract is
strong enough to be worth pinning: writes are atomic and deduplicated, reads are
verified by default, and a corrupted object is an error rather than a value.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rigby_core.artifacts import ArtifactStore, ContentAddressedArtifactStore
from rigby_core.contracts import ArtifactRefV1
from rigby_core.errors import ArtifactIntegrityError
from rigby_core.hashing import canonical_json_bytes, sha256_bytes


@pytest.fixture
def store(tmp_path: Path) -> ContentAddressedArtifactStore:
    return ContentAddressedArtifactStore(tmp_path / "artifacts")


def test_it_satisfies_the_protocol_its_consumers_type_against() -> None:
    assert isinstance(ContentAddressedArtifactStore, type)
    assert issubclass(ContentAddressedArtifactStore, ArtifactStore)


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------


def test_bytes_round_trip_and_the_reference_describes_them(store) -> None:
    payload = b"the quick brown fox"
    reference = store.put_bytes(payload, media_type="text/plain", filename="fox.txt")

    assert reference.sha256 == sha256_bytes(payload)
    assert reference.size_bytes == len(payload)
    assert reference.media_type == "text/plain"
    assert reference.filename == "fox.txt"
    assert store.read_bytes(reference) == payload
    assert store.exists(reference)


def test_an_empty_payload_is_a_real_artifact(store) -> None:
    """Zero bytes is a value, not a missing artifact."""

    reference = store.put_bytes(b"")
    assert reference.size_bytes == 0
    assert store.read_bytes(reference) == b""
    assert store.exists(reference, verify=True)


def test_a_file_round_trips_and_defaults_its_filename(tmp_path: Path, store) -> None:
    source = tmp_path / "source.bin"
    payload = bytes(range(256)) * 10
    source.write_bytes(payload)

    reference = store.put_file(source)
    assert reference.sha256 == sha256_bytes(payload)
    assert reference.size_bytes == len(payload)
    assert reference.filename == "source.bin"
    assert store.read_bytes(reference) == payload


def test_put_json_is_canonical_so_key_order_cannot_fork_an_artifact(store) -> None:
    first = store.put_json({"b": 1, "a": 2})
    second = store.put_json({"a": 2, "b": 1})

    assert first.sha256 == second.sha256
    assert first.media_type == "application/json"
    assert store.read_bytes(first) == canonical_json_bytes({"a": 2, "b": 1})


# --------------------------------------------------------------------------
# deduplication and layout
# --------------------------------------------------------------------------


def test_the_same_bytes_are_stored_once(store) -> None:
    payload = b"duplicate me"
    first = store.put_bytes(payload, filename="one.txt")
    second = store.put_bytes(payload, filename="two.txt")

    assert first.sha256 == second.sha256
    assert store.resolve(first) == store.resolve(second)
    stored = [path for path in store.objects.rglob("*") if path.is_file()]
    assert len(stored) == 1, stored


def test_a_rewrite_of_existing_content_verifies_rather_than_overwrites(store) -> None:
    """Re-putting known bytes must not rewrite the object underneath a reader."""

    payload = b"stable"
    reference = store.put_bytes(payload)
    path = store.resolve(reference)
    before = path.stat().st_mtime_ns

    store.put_bytes(payload)
    assert path.stat().st_mtime_ns == before


def test_objects_are_sharded_by_digest_prefix(store) -> None:
    reference = store.put_bytes(b"shard me")
    path = store.resolve(reference)
    assert path.parent.name == reference.sha256[:2]
    assert path.name == reference.sha256[2:]


def test_no_partial_file_is_left_behind_after_a_write(store) -> None:
    store.put_bytes(b"a")
    store.put_bytes(b"b")
    leftovers = [path.name for path in store.objects.rglob(".incoming-*")]
    assert not leftovers, leftovers


# --------------------------------------------------------------------------
# integrity: the reason the store exists
# --------------------------------------------------------------------------


def test_corrupted_content_is_an_error_and_not_a_value(store) -> None:
    reference = store.put_bytes(b"original content")
    store.resolve(reference).write_bytes(b"tampered content!")

    with pytest.raises(ArtifactIntegrityError):
        store.read_bytes(reference)
    with pytest.raises(ArtifactIntegrityError):
        store.resolve(reference)


def test_a_truncated_object_is_caught_by_size_as_well_as_by_hash(store) -> None:
    reference = store.put_bytes(b"0123456789")
    store.resolve(reference).write_bytes(b"01234")

    with pytest.raises(ArtifactIntegrityError):
        store.resolve(reference)


def test_verification_can_be_skipped_but_is_on_by_default(store) -> None:
    """Opt-out, never opt-in. A caller who says nothing gets the check."""

    reference = store.put_bytes(b"original")
    store.resolve(reference).write_bytes(b"tampered")

    assert store.resolve(reference, verify=False).is_file()
    with pytest.raises(ArtifactIntegrityError):
        store.resolve(reference)


def test_exists_reports_corruption_as_absence_only_when_asked_to_verify(store) -> None:
    reference = store.put_bytes(b"original")
    store.resolve(reference).write_bytes(b"tampered")

    assert store.exists(reference) is True
    assert store.exists(reference, verify=True) is False


def test_a_reference_to_content_that_was_never_stored_is_missing(store) -> None:
    absent = ArtifactRefV1(sha256="a" * 64, size_bytes=1)
    assert store.exists(absent) is False
    with pytest.raises(FileNotFoundError):
        store.resolve(absent)


def test_a_digest_shaped_like_a_path_traversal_is_refused(store) -> None:
    """The digest is interpolated into a path, so it is validated as a digest.

    `validate_sha256` is what stands between a caller-supplied string and the
    filesystem; without it `../` in a reference would address outside the store.
    """

    with pytest.raises(ValueError, match="64-character SHA-256"):
        store.resolve(ArtifactRefV1.model_construct(sha256="../" * 21 + "x", size_bytes=1))


def test_two_stores_over_the_same_root_see_each_others_objects(tmp_path: Path) -> None:
    """Content addressing, so a second process is not a second store."""

    root = tmp_path / "shared"
    writer = ContentAddressedArtifactStore(root)
    reference = writer.put_bytes(b"shared payload")

    reader = ContentAddressedArtifactStore(root)
    assert reader.read_bytes(reference) == b"shared payload"
