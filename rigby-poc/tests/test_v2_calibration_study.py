from __future__ import annotations

from dataclasses import replace

import pytest

from rigby_v2.calibration import (
    CalibrationStudy,
    CalibrationStudyError,
    SUPPORTED_ACTION_FAMILIES,
    build_synthetic_completed_study,
    compute_calibration_metrics,
    construct_calibration_blueprint,
    validate_calibration_study,
)
from rigby_v2.flywheel.schemas import CalibrationRatingV1, DefectKind, HumanComparisonPairV1


def test_blueprint_is_deterministic_blinded_and_seeded() -> None:
    first = construct_calibration_blueprint(seed=73)
    repeated = construct_calibration_blueprint(seed=73)
    changed = construct_calibration_blueprint(seed=74)
    assert first == repeated
    assert first != changed
    assert len(first.assignments) == 200
    assert all(value.left_record_id.startswith("blind_") for value in first.assignments)
    assert all(value.right_record_id.startswith("blind_") for value in first.assignments)
    assert all(len({rater.rater_id_hash for rater in value.raters}) == 3 for value in first.assignments)


def test_completed_study_meets_size_family_rater_order_and_defect_requirements() -> None:
    study = build_synthetic_completed_study(seed=91)
    result = validate_calibration_study(study)
    assert result.pair_count == 200
    assert result.rating_count == 600
    assert result.pair_counts_by_family == {
        family: 40 for family in SUPPORTED_ACTION_FAMILIES
    }
    assert result.presentation_counts == {"left_right": 300, "right_left": 300}
    assert set(result.defect_counts) == set(DefectKind)
    assert all(count > 0 for count in result.defect_counts.values())
    assert result.reversal_twin_groups == 20
    assert all(len(pair.ratings) == 3 for pair in study.pairs)


def test_study_rejects_too_few_pairs_and_unbalanced_presentation() -> None:
    study = build_synthetic_completed_study(seed=21)
    shortened = CalibrationStudy(
        seed=study.seed,
        pairs=study.pairs[:-1],
        ground_truth={pair.pair_id: study.ground_truth[pair.pair_id] for pair in study.pairs[:-1]},
    )
    with pytest.raises(CalibrationStudyError, match="at least 200"):
        validate_calibration_study(shortened)

    pairs = list(study.pairs)
    original = pairs[0]
    ratings = tuple(
        CalibrationRatingV1(
            rater_id_hash=rating.rater_id_hash,
            presentation_order="left_right",
            verdict=rating.verdict,
            rubric=rating.rubric,
        )
        for rating in original.ratings
    )
    pairs[0] = HumanComparisonPairV1(
        pair_id=original.pair_id,
        action_family=original.action_family,
        left_record_id=original.left_record_id,
        right_record_id=original.right_record_id,
        defects=original.defects,
        ratings=ratings,
    )
    unbalanced = replace(study, pairs=tuple(pairs))
    with pytest.raises(CalibrationStudyError, match="globally balanced"):
        validate_calibration_study(unbalanced)


def test_metrics_are_deterministic_and_include_confidence_intervals() -> None:
    study = build_synthetic_completed_study(
        seed=2026, correct_preference_probability=0.99
    )
    first = compute_calibration_metrics(study, bootstrap_samples=300)
    repeated = compute_calibration_metrics(study, bootstrap_samples=300)
    assert first == repeated
    assert first.fleiss_kappa > 0.8
    assert first.majority_agreement_rate > 0.95
    assert first.order_consistency_rate >= 0.95
    assert first.critical_false_accept_rate <= 0.03
    assert first.passes_release_thresholds
    assert first.critical_decision_count > 0
    assert first.reversal_twin_count == 20
    for interval in (
        first.fleiss_kappa_ci,
        first.majority_agreement_ci,
        first.order_consistency_ci,
        first.critical_false_accept_ci,
    ):
        assert interval.lower <= interval.upper
        assert interval.confidence == 0.95


def test_order_metric_normalizes_screen_reversal_to_record_identity() -> None:
    perfect = build_synthetic_completed_study(
        seed=55, correct_preference_probability=1.0
    )
    metrics = compute_calibration_metrics(perfect, bootstrap_samples=100)
    assert metrics.order_consistency_rate == 1.0
    assert metrics.critical_false_accept_rate == 0.0


def test_critical_false_accept_metric_detects_unsafe_choices() -> None:
    unsafe = build_synthetic_completed_study(
        seed=56,
        correct_preference_probability=1.0,
        critical_false_accept_probability=1.0,
    )
    metrics = compute_calibration_metrics(unsafe, bootstrap_samples=100)
    assert metrics.critical_false_accept_rate == 1.0
    assert metrics.critical_false_accept_ci.lower > 0.98
    assert not metrics.passes_release_thresholds
