from __future__ import annotations

import numpy as np

from rigby_core.errors import FailureCode, RigbyV2Error
from rigby_core.hashing import content_hash

from .models import (
    AdversarialCaseResultV1,
    BenchmarkCaseKind,
    BenchmarkManifestV1,
    BenchmarkMetricsV1,
    BenchmarkRunV1,
    ExpectedOutcome,
    ObservedOutcome,
    ReleaseGateCheckV1,
    ReleaseGateReportV1,
    SupportedCaseResultV1,
)


BOOTSTRAP_SAMPLES = 10_000
RATE_TOLERANCE = 1e-12


class BenchmarkEvaluationError(RigbyV2Error):
    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(FailureCode.INVALID_CONTRACT, message, details=details)


def _paired_bootstrap_ci(
    deltas: np.ndarray, *, seed: int, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float]:
    if deltas.ndim != 1 or len(deltas) == 0 or np.any(~np.isfinite(deltas)):
        raise ValueError("paired bootstrap requires a finite one-dimensional sample")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch_size = 500
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = generator.integers(0, len(deltas), size=(stop - start, len(deltas)))
        means[start:stop] = np.mean(deltas[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return float(low), float(high)


def _typed_outcome_matches(expected: ExpectedOutcome, observed: ObservedOutcome) -> bool:
    return expected.value == observed.value


def _check(name: str, passed: bool, measured: object, requirement: str) -> ReleaseGateCheckV1:
    if not isinstance(measured, (float, int, bool, str)):
        measured = str(measured)
    return ReleaseGateCheckV1(
        name=name, passed=bool(passed), measured=measured, requirement=requirement
    )


def evaluate_release_gate(
    manifest: BenchmarkManifestV1,
    run: BenchmarkRunV1,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> ReleaseGateReportV1:
    if bootstrap_samples < 1_000:
        raise ValueError("release evaluation requires at least 1,000 bootstrap samples")
    manifest_hash = manifest.content_hash()
    if run.manifest_hash != manifest_hash:
        raise BenchmarkEvaluationError(
            "Benchmark run is bound to a different manifest",
            details={"expected": manifest_hash, "actual": run.manifest_hash},
        )
    cases = {case.case_id: case for case in manifest.cases}
    results = {result.case_id: result for result in run.results}
    if set(results) != set(cases):
        raise BenchmarkEvaluationError(
            "Benchmark run must contain exactly one result for every sealed case",
            details={
                "missing": sorted(set(cases) - set(results)),
                "unexpected": sorted(set(results) - set(cases)),
            },
        )
    for case_id, result in results.items():
        case = cases[case_id]
        if result.case_hash != case.case_hash:
            raise BenchmarkEvaluationError(
                f"Result {case_id!r} is bound to the wrong immutable case hash"
            )
        if case.kind is BenchmarkCaseKind.SUPPORTED and not isinstance(
            result, SupportedCaseResultV1
        ):
            raise BenchmarkEvaluationError(f"Supported case {case_id!r} has adversarial result type")
        if case.kind is BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL and not isinstance(
            result, AdversarialCaseResultV1
        ):
            raise BenchmarkEvaluationError(f"Adversarial case {case_id!r} has supported result type")

    supported = [
        results[case.case_id]
        for case in manifest.cases
        if case.kind is BenchmarkCaseKind.SUPPORTED
    ]
    adversarial_pairs = [
        (case, results[case.case_id])
        for case in manifest.cases
        if case.kind is BenchmarkCaseKind.UNSUPPORTED_ADVERSARIAL
    ]
    assert all(isinstance(result, SupportedCaseResultV1) for result in supported)
    supported_results: list[SupportedCaseResultV1] = supported  # type: ignore[assignment]
    false_accepts = sum(
        result.observed_outcome is ObservedOutcome.CERTIFIED
        for _, result in adversarial_pairs
    )
    typed_outcomes = sum(
        _typed_outcome_matches(case.expected_outcome, result.observed_outcome)
        for case, result in adversarial_pairs
    )
    certified_rate = np.mean(
        [result.observed_outcome is ObservedOutcome.CERTIFIED for result in supported_results]
    )
    fingertip_values = [
        result.max_fingertip_error_m
        for result in supported_results
        if result.max_fingertip_error_m is not None
    ]
    penetration_values = [
        result.max_penetration_m
        for result in supported_results
        if result.max_penetration_m is not None
    ]
    missing_measurements = 2 * len(supported_results) - len(fingertip_values) - len(
        penetration_values
    )
    max_fingertip = max(fingertip_values, default=0.0)
    max_penetration = max(penetration_values, default=0.0)
    repeat_rate = np.mean(
        [
            len(result.repeat_task_outcomes) >= 3
            and len(set(result.repeat_task_outcomes)) == 1
            for result in supported_results
        ]
    )
    vlm_agreement = np.mean([result.vlm_human_agrees for result in supported_results])
    judge_consistency = np.mean(
        [result.judge_order_consistent for result in supported_results]
    )
    all_results = list(results.values())
    critical_judge_fa = np.mean(
        [result.critical_judge_false_accept for result in all_results]
    )
    precision_at_3 = sum(
        result.retrieval_relevant_top3 for result in supported_results
    ) / (3.0 * len(supported_results))
    recall_at_5 = np.mean(
        [
            min(result.retrieval_relevant_top5, result.retrieval_total_relevant)
            / result.retrieval_total_relevant
            for result in supported_results
        ]
    )
    deltas = np.asarray(
        [
            float(result.rag_on_success) - float(result.rag_off_success)
            for result in supported_results
        ],
        dtype=np.float64,
    )
    rag_lift = float(np.mean(deltas))
    bootstrap_seed = int(
        content_hash(
            {
                "manifest_hash": manifest_hash,
                "result_set_hash": run.result_set_hash,
                "bootstrap_samples": bootstrap_samples,
            }
        )[:16],
        16,
    )
    ci_low, ci_high = _paired_bootstrap_ci(
        deltas, seed=bootstrap_seed, samples=bootstrap_samples
    )
    p95_seconds = float(
        np.quantile(
            [result.local_pipeline_seconds for result in supported_results],
            0.95,
            method="linear",
        )
    )
    metrics = BenchmarkMetricsV1(
        supported_count=len(supported_results),
        adversarial_count=len(adversarial_pairs),
        critical_deterministic_false_accepts=false_accepts,
        adversarial_typed_outcome_rate=typed_outcomes / len(adversarial_pairs),
        supported_certified_winner_rate=float(certified_rate),
        max_fingertip_error_m=float(max_fingertip),
        max_penetration_m=float(max_penetration),
        repeat_task_outcome_identical_rate=float(repeat_rate),
        vlm_human_agreement_rate=float(vlm_agreement),
        judge_order_consistency_rate=float(judge_consistency),
        critical_judge_false_accept_rate=float(critical_judge_fa),
        retrieval_precision_at_3=float(precision_at_3),
        retrieval_recall_at_5=float(recall_at_5),
        rag_success_lift=rag_lift,
        rag_lift_ci95_low=ci_low,
        rag_lift_ci95_high=ci_high,
        local_pipeline_p95_seconds=p95_seconds,
        missing_physics_measurements=missing_measurements,
    )
    checks = (
        _check("benchmark_matrix", len(supported_results) == 300 and len(adversarial_pairs) == 100, f"{len(supported_results)}+{len(adversarial_pairs)}", "exactly 300 supported + 100 adversarial"),
        _check("critical_deterministic_false_accepts", false_accepts == 0, false_accepts, "zero"),
        _check("adversarial_typed_outcomes", typed_outcomes == 100, typed_outcomes / 100.0, "100% expected typed outcomes"),
        _check("supported_certified_winner_rate", certified_rate + RATE_TOLERANCE >= 0.95, float(certified_rate), ">= 0.95"),
        _check("physics_measurements_complete", missing_measurements == 0, missing_measurements, "zero missing"),
        _check("max_fingertip_error_m", max_fingertip < 0.008, float(max_fingertip), "< 0.008"),
        _check("max_penetration_m", max_penetration < 0.002, float(max_penetration), "< 0.002"),
        _check("repeat_task_outcome_identical", repeat_rate == 1.0, float(repeat_rate), "1.0"),
        _check("vlm_human_agreement", vlm_agreement + RATE_TOLERANCE >= 0.80, float(vlm_agreement), ">= 0.80"),
        _check("judge_order_consistency", judge_consistency + RATE_TOLERANCE >= 0.95, float(judge_consistency), ">= 0.95"),
        _check("critical_judge_false_accept_rate", critical_judge_fa <= 0.01 + RATE_TOLERANCE, float(critical_judge_fa), "<= 0.01"),
        _check("retrieval_precision_at_3", precision_at_3 + RATE_TOLERANCE >= 0.90, float(precision_at_3), ">= 0.90"),
        _check("retrieval_recall_at_5", recall_at_5 + RATE_TOLERANCE >= 0.80, float(recall_at_5), ">= 0.80"),
        _check("rag_success_lift", rag_lift + RATE_TOLERANCE >= 0.05, rag_lift, ">= 0.05"),
        _check("rag_lift_ci95_low", ci_low > 0.0, ci_low, "> 0.0"),
        _check("local_pipeline_p95_seconds", p95_seconds <= 90.0 + RATE_TOLERANCE, p95_seconds, "<= 90.0 for five 10s candidates, VLM excluded"),
    )
    return ReleaseGateReportV1(
        release_allowed=all(check.passed for check in checks),
        manifest_hash=manifest_hash,
        result_set_hash=run.result_set_hash,
        metrics=metrics,
        checks=checks,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
