"""`canonical_json` and the digests built on it.

Every content-addressed store, every corpus `expected.json`, every sealed
response and the artifact store itself resolve to `sha256` over the bytes this
module produces. It had no tests: `core/` shipped 5,034 lines behind a single
58-line import-graph guard, and this is the module underneath all of it.

The properties below are the ones a caller is entitled to assume. Determinism
across processes is the load-bearing one -- a digest that depends on dict
insertion order, on `PYTHONHASHSEED`, or on the platform is not a digest.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import pytest
from pydantic import BaseModel
from rigby_core.hashing import (
    canonical_json,
    canonical_json_bytes,
    content_hash,
    hash_file,
    sha256_bytes,
    validate_sha256,
)


class Colour(Enum):
    RED = "red"


class Point(BaseModel):
    y: int
    x: int


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_key_order_does_not_change_the_digest() -> None:
    """The whole point. Two dicts that differ only by insertion order are one value."""

    first = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    second = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json(first) == canonical_json(second)
    assert content_hash(first) == content_hash(second)


def test_the_digest_is_stable_across_processes() -> None:
    """A fresh interpreter with a different hash seed must agree.

    Set randomization explicitly rather than trusting the default: this is the
    failure that only appears on someone else's machine, months later, as a
    corpus digest that will not reproduce.
    """

    program = (
        "from rigby_core.hashing import content_hash;"
        "print(content_hash({'b': [1, 2, {'d': 4, 'c': 3}], 'a': 'x'}))"
    )
    digests = set()
    for seed in ("0", "524287"):
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        digests.add(completed.stdout.strip())
    assert len(digests) == 1, f"digest moved with PYTHONHASHSEED: {digests}"
    assert digests.pop() == content_hash({"b": [1, 2, {"d": 4, "c": 3}], "a": "x"})


def test_a_set_hashes_by_value_not_by_iteration_order() -> None:
    assert content_hash({1, 2, 3}) == content_hash({3, 1, 2})


# --------------------------------------------------------------------------
# the normalizations, each of which is a decision
# --------------------------------------------------------------------------


def test_negative_zero_is_folded_into_zero() -> None:
    """`-0.0 == 0.0` in Python but they serialize differently, so they are folded.

    Without this an IK solve that happened to produce `-0.0` on one platform and
    `0.0` on another would publish two digests for one motion.
    """

    assert canonical_json(-0.0) == canonical_json(0.0) == "0.0"
    assert content_hash({"v": -0.0}) == content_hash({"v": 0.0})


def test_aware_datetimes_normalize_to_utc_and_naive_ones_are_refused() -> None:
    same_instant = [
        datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
        datetime(2026, 8, 28, 14, 0, tzinfo=timezone(timedelta(hours=2))),
    ]
    assert content_hash(same_instant[0]) == content_hash(same_instant[1])
    assert canonical_json(same_instant[0]) == '"2026-08-28T12:00:00.000000Z"'

    with pytest.raises(ValueError, match="timezone-aware"):
        # naive on purpose: being refused is the assertion
        canonical_json(datetime(2026, 8, 28, 12, 0))  # noqa: DTZ001


def test_dates_paths_enums_and_models_all_reduce_to_plain_json() -> None:
    assert canonical_json(date(2026, 8, 28)) == '"2026-08-28"'
    assert canonical_json(Path("a") / "b") == '"a/b"'
    assert canonical_json(Colour.RED) == '"red"'
    # Field order in the model declaration must not leak into the digest.
    assert canonical_json(Point(x=1, y=2)) == '{"x":1,"y":2}'


def test_a_path_hashes_the_same_on_every_platform() -> None:
    """`as_posix`, so a Windows checkout does not produce a second digest.

    Same class of defect as the CRLF one that took ~100 tests red on the Windows
    runner: a platform detail reaching a hash. See the repository .gitattributes.
    """

    assert canonical_json(Path("a/b/c")) == '"a/b/c"'
    assert "\\" not in canonical_json(Path("a") / "b" / "c")


def test_bytes_are_tagged_rather_than_decoded() -> None:
    """Base64 under a marker key, so bytes cannot collide with a string."""

    assert canonical_json(b"hi") == '{"$bytes_base64":"aGk="}'
    assert content_hash(b"hi") != content_hash("aGk=")


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_refused(value: float) -> None:
    """They have no JSON representation; `allow_nan` would emit invalid JSON."""

    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json({"v": value})


def test_non_string_mapping_keys_are_refused() -> None:
    with pytest.raises(TypeError, match="string keys"):
        canonical_json({1: "a"})


def test_an_unsupported_type_is_refused_rather_than_stringified() -> None:
    """A silent `str()` fallback would make two different objects one digest."""

    with pytest.raises(TypeError, match="Unsupported canonical value"):
        canonical_json(object())


# --------------------------------------------------------------------------
# the digest helpers
# --------------------------------------------------------------------------


def test_canonical_json_bytes_is_utf8_of_canonical_json() -> None:
    value = {"k": "ünïcø∂e"}
    assert canonical_json_bytes(value) == canonical_json(value).encode("utf-8")
    # ensure_ascii=False, so the text is not escaped into ASCII first.
    assert "ü" in canonical_json(value)


def test_content_hash_is_sha256_of_those_bytes() -> None:
    value = {"a": [1, 2, 3]}
    assert content_hash(value) == sha256_bytes(canonical_json_bytes(value))


def test_hash_file_matches_sha256_of_the_whole_file_across_chunk_sizes(tmp_path: Path) -> None:
    """Chunking is an implementation detail and must not reach the digest."""

    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 40
    path.write_bytes(payload)

    expected = sha256_bytes(payload)
    assert hash_file(path) == expected
    for chunk_size in (1, 7, 4096, len(payload) * 2):
        assert hash_file(path, chunk_size=chunk_size) == expected


def test_hash_file_of_an_empty_file_is_the_empty_digest(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert hash_file(path) == sha256_bytes(b"")


def test_validate_sha256_normalizes_case_and_rejects_everything_else() -> None:
    digest = "A" * 64
    assert validate_sha256(digest) == "a" * 64

    for bad in ("a" * 63, "a" * 65, "g" * 64, "", "../../etc/passwd"):
        with pytest.raises(ValueError, match="64-character SHA-256"):
            validate_sha256(bad)
