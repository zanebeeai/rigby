"""Reproducible agreement, order-bias, and safety calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from rigby_v2.flywheel.schemas import DefectKind, HumanComparisonPairV1

from .study import CalibrationStudy, selected_record, validate_calibration_study


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float = 0.95


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    fleiss_kappa: float
    fleiss_kappa_ci: ConfidenceInterval
    majority_agreement_rate: float
    majority_agreement_ci: ConfidenceInterval
    order_consistency_rate: float
    order_consistency_ci: ConfidenceInterval
    critical_false_accept_rate: float
    critical_false_accept_ci: ConfidenceInterval
    critical_decision_count: int
    reversal_twin_count: int

    @property
    def passes_release_thresholds(self) -> bool:
        return (
            self.fleiss_kappa >= 0.60
            and self.order_consistency_rate >= 0.95
            and self.critical_false_accept_rate <= 0.01
        )


def _wilson(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    if total <= 0:
        return ConfidenceInterval(0.0, 1.0, confidence)
    z = float(norm.ppf(0.5 + confidence / 2))
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return ConfidenceInterval(max(0.0, float(center - radius)), min(1.0, float(center + radius)), confidence)


def _rating_category(pair: HumanComparisonPairV1, rating_index: int) -> int:
    rating = pair.ratings[rating_index]
    chosen = selected_record(pair, rating)
    if chosen == pair.left_record_id:
        return 0
    if chosen == pair.right_record_id:
        return 1
    return {"tie": 2, "abstain": 3, "both_fail": 4}[rating.verdict]


def _fleiss_kappa(pairs: tuple[HumanComparisonPairV1, ...]) -> float:
    if not pairs:
        return 0.0
    counts = np.zeros((len(pairs), 5), dtype=float)
    for row, pair in enumerate(pairs):
        for rating_index in range(3):
            counts[row, _rating_category(pair, rating_index)] += 1
    raters = 3
    pair_agreement = (np.sum(counts * counts, axis=1) - raters) / (raters * (raters - 1))
    observed = float(np.mean(pair_agreement))
    proportions = np.sum(counts, axis=0) / (len(pairs) * raters)
    expected = float(np.sum(proportions * proportions))
    return 1.0 if np.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def _bootstrap_kappa(
    pairs: tuple[HumanComparisonPairV1, ...],
    *,
    seed: int,
    samples: int,
) -> ConfidenceInterval:
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = tuple(pairs[item] for item in rng.integers(0, len(pairs), len(pairs)))
        values[index] = _fleiss_kappa(selected)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return ConfidenceInterval(float(lower), float(upper))


def _consensus(pair: HumanComparisonPairV1) -> str | None:
    choices = [selected_record(pair, rating) for rating in pair.ratings]
    counts = {
        record: choices.count(record)
        for record in (pair.left_record_id, pair.right_record_id)
    }
    winner, count = max(counts.items(), key=lambda item: item[1])
    return winner if count >= 2 else None


def compute_calibration_metrics(
    study: CalibrationStudy,
    *,
    bootstrap_samples: int = 1000,
) -> CalibrationMetrics:
    validate_calibration_study(study)
    if bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    majority_pairs = 0
    for pair in study.pairs:
        categories = [_rating_category(pair, index) for index in range(3)]
        if max(categories.count(value) for value in set(categories)) >= 2:
            majority_pairs += 1

    reversal_groups: dict[tuple[str, str], list[HumanComparisonPairV1]] = {}
    for pair in study.pairs:
        if DefectKind.ORDER_REVERSAL in pair.defects:
            reversal_groups.setdefault(
                tuple(sorted((pair.left_record_id, pair.right_record_id))), []
            ).append(pair)
    consistent = sum(
        _consensus(group[0]) is not None and _consensus(group[0]) == _consensus(group[1])
        for group in reversal_groups.values()
    )

    critical_decisions = 0
    false_accepts = 0
    for pair in study.pairs:
        rejected = set(study.ground_truth[pair.pair_id].critical_reject_record_ids)
        if not rejected:
            continue
        for rating in pair.ratings:
            critical_decisions += 1
            false_accepts += selected_record(pair, rating) in rejected

    kappa = _fleiss_kappa(study.pairs)
    return CalibrationMetrics(
        fleiss_kappa=kappa,
        fleiss_kappa_ci=_bootstrap_kappa(
            study.pairs,
            seed=study.seed ^ 0xB00757A9,
            samples=bootstrap_samples,
        ),
        majority_agreement_rate=majority_pairs / len(study.pairs),
        majority_agreement_ci=_wilson(majority_pairs, len(study.pairs)),
        order_consistency_rate=consistent / len(reversal_groups),
        order_consistency_ci=_wilson(consistent, len(reversal_groups)),
        critical_false_accept_rate=(false_accepts / critical_decisions),
        critical_false_accept_ci=_wilson(false_accepts, critical_decisions),
        critical_decision_count=critical_decisions,
        reversal_twin_count=len(reversal_groups),
    )
