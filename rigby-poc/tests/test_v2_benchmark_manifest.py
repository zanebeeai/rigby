from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

from rigby_v2.benchmark import (
    BenchmarkCaseKind,
    BenchmarkFamily,
    DEFAULT_SPEC_PATH,
    load_benchmark_manifest,
)
from rigby_v2.errors import ArtifactIntegrityError

pytestmark = pytest.mark.fast


def test_sealed_manifest_expands_to_exact_release_matrix_with_stable_hashes() -> None:
    first = load_benchmark_manifest()
    second = load_benchmark_manifest()
    assert first == second
    assert first.content_hash() == second.content_hash()
    assert len(first.cases) == 400
    assert len({case.case_id for case in first.cases}) == 400
    assert len({case.case_hash for case in first.cases}) == 400
    supported = [case for case in first.cases if case.kind is BenchmarkCaseKind.SUPPORTED]
    adversarial = [
        case for case in first.cases if case.kind is BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL
    ]
    assert len(supported) == 300
    assert len(adversarial) == 100
    assert Counter(case.family for case in supported) == {
        family: 75 for family in BenchmarkFamily
    }
    assert all(case.family is None for case in adversarial)


def test_spec_byte_tamper_fails_external_checksum(tmp_path) -> None:
    path = tmp_path / "benchmark_spec.json"
    path.write_bytes(DEFAULT_SPEC_PATH.read_bytes().replace(b"friendly greeting", b"altered greeting"))
    path.with_suffix(".json.sha256").write_text(
        DEFAULT_SPEC_PATH.with_suffix(".json.sha256").read_text(encoding="ascii"),
        encoding="ascii",
    )
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        load_benchmark_manifest(path)


def test_rechecks_expanded_hash_even_if_attacker_rewrites_file_checksum(tmp_path) -> None:
    path = tmp_path / "benchmark_spec.json"
    document = json.loads(DEFAULT_SPEC_PATH.read_text(encoding="utf-8"))
    document["supported_grids"][0]["actions"][0] = "Tampered {entity} action."
    encoded = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    path.write_bytes(encoded)
    path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n", encoding="ascii"
    )
    with pytest.raises(ArtifactIntegrityError, match="Expanded benchmark"):
        load_benchmark_manifest(path)


def test_case_content_hash_rejects_tampering() -> None:
    case = load_benchmark_manifest().cases[0]
    data = case.model_dump(mode="python")
    data["prompt"] = "changed after sealing"
    with pytest.raises(ValueError, match="case hash"):
        type(case).model_validate(data)
