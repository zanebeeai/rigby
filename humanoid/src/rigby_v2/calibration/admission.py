"""Admission test for learned motion in-betweening."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from scipy.stats import beta, binomtest

from .metrics import ConfidenceInterval


class HumanOutcome(StrEnum):
    LEARNED_PREFERRED = "learned_preferred"
    BASELINE_PREFERRED = "baseline_preferred"
    TIE = "tie"
    BOTH_FAIL = "both_fail"


@dataclass(frozen=True, slots=True)
class InbetweeningTrial:
    trial_id: str
    human_outcome: HumanOutcome
    hard_anchor_errors_m: Mapping[str, float]
    hard_anchor_tolerance_m: float
    physics_gates: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hard_anchor_errors_m", MappingProxyType(dict(self.hard_anchor_errors_m))
        )
        object.__setattr__(self, "physics_gates", MappingProxyType(dict(self.physics_gates)))
        if not self.trial_id:
            raise ValueError("trial ID is required")
        if self.hard_anchor_tolerance_m < 0 or not self.hard_anchor_errors_m:
            raise ValueError("hard-anchor evidence and a nonnegative tolerance are required")
        if any(value < 0 for value in self.hard_anchor_errors_m.values()):
            raise ValueError("hard-anchor errors must be nonnegative")
        if not self.physics_gates:
            raise ValueError("every trial requires physics-gate evidence")


@dataclass(frozen=True, slots=True)
class InbetweeningAdmissionDecision:
    admitted: bool
    learned_wins: int
    baseline_wins: int
    ties_or_both_fail: int
    decisive_trials: int
    learned_preference_rate: float
    learned_preference_ci: ConfidenceInterval
    one_sided_p_value: float
    hard_anchors_preserved: bool
    all_physics_gates_passed: bool
    reasons: tuple[str, ...]


def _clopper_pearson(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    alpha = 1 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(
        beta.ppf(1 - alpha / 2, successes + 1, total - successes)
    )
    return ConfidenceInterval(lower=lower, upper=upper, confidence=confidence)


def evaluate_inbetweening_admission(
    trials: tuple[InbetweeningTrial, ...],
    *,
    alpha: float = 0.05,
    minimum_decisive_trials: int = 30,
) -> InbetweeningAdmissionDecision:
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and 0.5")
    identifiers = [trial.trial_id for trial in trials]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("in-betweening trial IDs must be unique")
    learned_wins = sum(
        trial.human_outcome is HumanOutcome.LEARNED_PREFERRED for trial in trials
    )
    baseline_wins = sum(
        trial.human_outcome is HumanOutcome.BASELINE_PREFERRED for trial in trials
    )
    decisive = learned_wins + baseline_wins
    ties = len(trials) - decisive
    preference_rate = learned_wins / decisive if decisive else 0.0
    interval = (
        _clopper_pearson(learned_wins, decisive)
        if decisive
        else ConfidenceInterval(0.0, 1.0)
    )
    p_value = (
        float(binomtest(learned_wins, decisive, p=0.5, alternative="greater").pvalue)
        if decisive
        else 1.0
    )
    anchors_preserved = all(
        max(trial.hard_anchor_errors_m.values()) <= trial.hard_anchor_tolerance_m
        for trial in trials
    )
    physics_passed = all(all(trial.physics_gates.values()) for trial in trials)
    reasons: list[str] = []
    if decisive < minimum_decisive_trials:
        reasons.append(f"requires at least {minimum_decisive_trials} decisive human trials")
    if p_value >= alpha or interval.lower <= 0.5:
        reasons.append("learned human preference is not statistically better than baseline")
    if not anchors_preserved:
        reasons.append("one or more hard anchors exceeded tolerance")
    if not physics_passed:
        reasons.append("one or more deterministic physics gates failed")
    return InbetweeningAdmissionDecision(
        admitted=not reasons,
        learned_wins=learned_wins,
        baseline_wins=baseline_wins,
        ties_or_both_fail=ties,
        decisive_trials=decisive,
        learned_preference_rate=preference_rate,
        learned_preference_ci=interval,
        one_sided_p_value=p_value,
        hard_anchors_preserved=anchors_preserved,
        all_physics_gates_passed=physics_passed,
        reasons=tuple(reasons),
    )

