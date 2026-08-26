from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rigby_v2.benchmark import (
    AdversarialCaseResultV1,
    BenchmarkCaseKind,
    BenchmarkEvaluationError,
    BenchmarkRunV1,
    ExpectedOutcome,
    ObservedOutcome,
    SupportedCaseResultV1,
    evaluate_release_gate,
    load_benchmark_manifest,
)


ENVIRONMENT_HASH = "e" * 64


def passing_run() -> tuple[object, BenchmarkRunV1]:
    manifest = load_benchmark_manifest()
    results = []
    supported_index = 0
    for case in manifest.cases:
        if case.kind is BenchmarkCaseKind.SUPPORTED:
            results.append(
                SupportedCaseResultV1(
                    case_id=case.case_id,
                    case_hash=case.case_hash,
                    observed_outcome=ObservedOutcome.CERTIFIED,
                    max_fingertip_error_m=0.007,
                    max_penetration_m=0.0015,
                    repeat_task_outcomes=("success", "success", "success"),
                    vlm_human_agrees=True,
                    judge_order_consistent=True,
                    retrieval_relevant_top3=3,
                    retrieval_relevant_top5=5,
                    retrieval_total_relevant=5,
                    rag_off_success=supported_index < 270,
                    rag_on_success=True,
                    local_pipeline_seconds=80.0,
                )
            )
            supported_index += 1
        else:
            observed = {
                ExpectedOutcome.UNSUPPORTED: ObservedOutcome.UNSUPPORTED,
                ExpectedOutcome.INFEASIBLE: ObservedOutcome.INFEASIBLE,
                ExpectedOutcome.INVALID_ASSET: ObservedOutcome.INVALID_ASSET,
            }[case.expected_outcome]
            results.append(
                AdversarialCaseResultV1(
                    case_id=case.case_id,
                    case_hash=case.case_hash,
                    observed_outcome=observed,
                )
            )
    run = BenchmarkRunV1(
        run_id="passing-run",
        manifest_hash=manifest.content_hash(),
        environment_hash=ENVIRONMENT_HASH,
        completed_at=datetime(2026, 8, 10, 23, tzinfo=UTC),
        results=tuple(results),
    )
    return manifest, run


def _replace_results(
    run: BenchmarkRunV1, replacements: dict[int, dict[str, object]]
) -> BenchmarkRunV1:
    results = list(run.results)
    for index, update in replacements.items():
        data = results[index].model_dump(mode="python", exclude={"result_hash"})
        data.update(update)
        results[index] = type(results[index]).model_validate(data)
    return BenchmarkRunV1(
        run_id=run.run_id + "-changed",
        manifest_hash=run.manifest_hash,
        environment_hash=run.environment_hash,
        completed_at=run.completed_at,
        results=tuple(results),
    )


def _supported_indices(run: BenchmarkRunV1) -> list[int]:
    return [index for index, item in enumerate(run.results) if item.result_kind == "supported"]


def _adversarial_indices(run: BenchmarkRunV1) -> list[int]:
    return [index for index, item in enumerate(run.results) if item.result_kind == "adversarial"]


def _gate(report, name: str):  # type: ignore[no-untyped-def]
    return next(check for check in report.checks if check.name == name)


def test_passing_release_report_is_reproducible_and_audited() -> None:
    manifest, run = passing_run()
    first = evaluate_release_gate(manifest, run)
    second = evaluate_release_gate(manifest, run)
    assert first == second
    assert first.report_hash == second.report_hash
    assert first.release_allowed
    assert all(check.passed for check in first.checks)
    assert first.metrics.supported_count == 300
    assert first.metrics.adversarial_count == 100
    assert first.metrics.rag_success_lift == pytest.approx(0.10)
    assert first.metrics.rag_lift_ci95_low > 0
    assert first.metrics.local_pipeline_p95_seconds == pytest.approx(80.0)
    assert first.bootstrap_samples == 10_000


def test_critical_adversarial_false_accept_is_derived_and_blocks_release() -> None:
    manifest, run = passing_run()
    adversarial = _adversarial_indices(run)[0]
    changed = _replace_results(
        run, {adversarial: {"observed_outcome": ObservedOutcome.CERTIFIED}}
    )
    report = evaluate_release_gate(manifest, changed)
    assert report.metrics.critical_deterministic_false_accepts == 1
    assert not _gate(report, "critical_deterministic_false_accepts").passed
    assert not report.release_allowed


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    (
        ("max_fingertip_error_m", 0.008, "max_fingertip_error_m"),
        ("max_penetration_m", 0.002, "max_penetration_m"),
        ("repeat_task_outcomes", ("success", "failure", "success"), "repeat_task_outcome_identical"),
        ("max_fingertip_error_m", None, "physics_measurements_complete"),
        ("local_pipeline_seconds", 90.1, "local_pipeline_p95_seconds"),
    ),
)
def test_strict_physics_repeat_and_runtime_boundaries_fail(
    field: str, value: object, gate: str
) -> None:
    manifest, run = passing_run()
    supported = _supported_indices(run)
    indices = supported if field == "local_pipeline_seconds" else supported[:1]
    changed = _replace_results(run, {index: {field: value} for index in indices})
    report = evaluate_release_gate(manifest, changed, bootstrap_samples=2_000)
    assert not _gate(report, gate).passed
    assert not report.release_allowed


def test_rate_thresholds_are_inclusive_only_where_specified() -> None:
    manifest, run = passing_run()
    supported = _supported_indices(run)
    adversarial = _adversarial_indices(run)
    # Exactly 95% supported winners, 80% VLM agreement, 95% order
    # consistency, and 1% critical judge false accepts all meet the gates.
    replacements: dict[int, dict[str, object]] = {}
    for index in supported[:15]:
        replacements.setdefault(index, {})["observed_outcome"] = ObservedOutcome.REJECTED
        replacements[index]["judge_order_consistent"] = False
    for index in supported[:60]:
        replacements.setdefault(index, {})["vlm_human_agrees"] = False
    for index in adversarial[:4]:
        replacements.setdefault(index, {})["critical_judge_false_accept"] = True
    boundary = _replace_results(run, replacements)
    report = evaluate_release_gate(manifest, boundary, bootstrap_samples=2_000)
    assert _gate(report, "supported_certified_winner_rate").passed
    assert _gate(report, "vlm_human_agreement").passed
    assert _gate(report, "judge_order_consistency").passed
    assert _gate(report, "critical_judge_false_accept_rate").passed

    below = _replace_results(
        boundary,
        {
            supported[15]: {
                "observed_outcome": ObservedOutcome.REJECTED,
                "judge_order_consistent": False,
            },
            supported[60]: {"vlm_human_agrees": False},
            adversarial[4]: {"critical_judge_false_accept": True},
        },
    )
    failed = evaluate_release_gate(manifest, below, bootstrap_samples=2_000)
    assert not _gate(failed, "supported_certified_winner_rate").passed
    assert not _gate(failed, "vlm_human_agreement").passed
    assert not _gate(failed, "judge_order_consistency").passed
    assert not _gate(failed, "critical_judge_false_accept_rate").passed


def test_retrieval_thresholds_and_paired_rag_confidence_are_hard_gates() -> None:
    manifest, run = passing_run()
    supported = _supported_indices(run)
    # Exactly P@3=.90 and R@5=.80 pass.
    replacements = {
        index: {
            "retrieval_relevant_top3": 2 if offset < 90 else 3,
            "retrieval_relevant_top5": 4,
            "retrieval_total_relevant": 5,
        }
        for offset, index in enumerate(supported)
    }
    boundary = _replace_results(run, replacements)
    report = evaluate_release_gate(manifest, boundary, bootstrap_samples=2_000)
    assert report.metrics.retrieval_precision_at_3 == pytest.approx(0.90)
    assert report.metrics.retrieval_recall_at_5 == pytest.approx(0.80)
    assert _gate(report, "retrieval_precision_at_3").passed
    assert _gate(report, "retrieval_recall_at_5").passed

    below = _replace_results(
        boundary,
        {
            supported[90]: {"retrieval_relevant_top3": 2},
            supported[0]: {"retrieval_relevant_top3": 2, "retrieval_relevant_top5": 3},
        },
    )
    failed = evaluate_release_gate(manifest, below, bootstrap_samples=2_000)
    assert not _gate(failed, "retrieval_precision_at_3").passed
    assert not _gate(failed, "retrieval_recall_at_5").passed

    # Mean paired lift is exactly +5pp, but high paired variance makes the
    # bootstrap lower bound non-positive, so the empirical claim is rejected.
    rag_updates = {}
    for offset, index in enumerate(supported):
        if offset < 157:
            rag_updates[index] = {"rag_off_success": False, "rag_on_success": True}
        elif offset < 299:
            rag_updates[index] = {"rag_off_success": True, "rag_on_success": False}
        else:
            rag_updates[index] = {"rag_off_success": True, "rag_on_success": True}
    noisy = _replace_results(run, rag_updates)
    noisy_report = evaluate_release_gate(manifest, noisy)
    assert noisy_report.metrics.rag_success_lift == pytest.approx(0.05)
    assert _gate(noisy_report, "rag_success_lift").passed
    assert noisy_report.metrics.rag_lift_ci95_low <= 0.0
    assert not _gate(noisy_report, "rag_lift_ci95_low").passed


def test_run_and_case_binding_tamper_are_rejected() -> None:
    manifest, run = passing_run()
    with pytest.raises(ValueError, match="result-set hash"):
        BenchmarkRunV1.model_validate(
            {**run.model_dump(mode="python"), "result_set_hash": "a" * 64}
        )

    sealed_result = run.results[0]
    result_data = sealed_result.model_dump(mode="python")
    result_data["max_fingertip_error_m"] = 0.5
    with pytest.raises(ValueError, match="result hash"):
        type(sealed_result).model_validate(result_data)

    changed = _replace_results(run, {0: {"case_hash": "b" * 64}})
    with pytest.raises(BenchmarkEvaluationError, match="wrong immutable case hash"):
        evaluate_release_gate(manifest, changed, bootstrap_samples=1_000)

    wrong_manifest_run = BenchmarkRunV1(
        run_id="wrong-manifest",
        manifest_hash="c" * 64,
        environment_hash=run.environment_hash,
        completed_at=run.completed_at,
        results=run.results,
    )
    with pytest.raises(BenchmarkEvaluationError, match="different manifest"):
        evaluate_release_gate(manifest, wrong_manifest_run, bootstrap_samples=1_000)
