from __future__ import annotations

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.benchmark import (
    BenchmarkCaseKind,
    BenchmarkCaseV1,
    ExpectedOutcome,
    evaluate_adversarial_cases,
    load_benchmark_manifest,
    run_adversarial_benchmark_100,
)


def test_adversarial_benchmark_100_seals_content_addressed_evidence(tmp_path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    manifest = load_benchmark_manifest()
    attempt = run_adversarial_benchmark_100(artifacts=store, manifest=manifest)

    assert attempt.sealed
    assert len(attempt.results) == 100
    assert len(attempt.evidence) == 100
    assert len({item.sha256 for item in attempt.evidence}) == 100
    assert attempt.critical_false_accepts == 0
    assert attempt.typed_outcome_mismatches == 0
    assert attempt.model_calls == 0
    assert attempt.seal is not None
    assert attempt.seal.release_evidence_id == "adversarial_benchmark_100"
    assert attempt.seal.case_count == 100
    assert attempt.seal.model_calls == 0
    assert attempt.seal_artifact is not None
    assert store.read_bytes(attempt.seal_artifact)
    assert all(store.read_bytes(reference) for reference in attempt.evidence)


def test_seal_is_withheld_on_a_rule_based_false_accept(tmp_path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "cas")
    case = BenchmarkCaseV1(
        case_id="adversarial-near-miss-001",
        kind=BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL,
        prompt="Open the rigid modeled drawer.",
        expected_outcome=ExpectedOutcome.UNSUPPORTED,
        seed=7,
        tags=("adversarial", "synthetic"),
    )
    attempt = evaluate_adversarial_cases(
        (case,),
        artifacts=store,
        benchmark_manifest_hash="0" * 64,
        require_exact_100=False,
    )

    assert not attempt.sealed
    assert attempt.critical_false_accepts == 1
    assert attempt.typed_outcome_mismatches == 1
    assert attempt.seal is None
    assert attempt.seal_artifact is None
