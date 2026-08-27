from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from rigby_v2.flywheel.schemas import (
    AnonymousJudgeEvaluationV1,
    BestOfFiveSelectionV1,
    CameraEvidenceV1,
    CandidateEvidenceV1,
    CandidateSetV1,
    JudgePassV1,
    SelectionOutcome,
)
from rigby_core.hashing import content_hash


@dataclass(frozen=True)
class AnonymousEvidenceView:
    """Evidence boundary exposed to judges; candidate identity is absent."""

    anonymous_id: str
    trace_sha256: str
    cameras: tuple[CameraEvidenceV1, ...]
    metrics: dict[str, float]


@runtime_checkable
class AnonymousJudge(Protocol):
    judge_id: str

    def evaluate(
        self,
        pass_id: str,
        prompt: str,
        evidence: tuple[AnonymousEvidenceView, ...],
    ) -> Sequence[AnonymousJudgeEvaluationV1]: ...


@runtime_checkable
class EvidenceRepairer(Protocol):
    def repair(
        self,
        candidate_id: str,
        evidence: CandidateEvidenceV1,
        diagnostics: dict[str, object],
    ) -> CandidateEvidenceV1 | None: ...


_DIMENSIONS = (
    "semantic_fidelity",
    "physical_plausibility",
    "contact_quality",
    "timing_energy",
    "whole_body_quality",
    "visual_clarity",
)


def _rank_key(evaluation: AnonymousJudgeEvaluationV1) -> tuple[float, float, float]:
    return (
        evaluation.scores.weakest,
        evaluation.scores.aggregate,
        -evaluation.uncertainty,
    )


def _pass_winner(judge_pass: JudgePassV1) -> str | None:
    accepted = [
        evaluation
        for evaluation in judge_pass.evaluations
        if evaluation.independently_clears_rubric
    ]
    if not accepted:
        return None
    return max(accepted, key=_rank_key).anonymous_id


def _pass_margin(judge_pass: JudgePassV1) -> float:
    accepted = sorted(
        (
            evaluation
            for evaluation in judge_pass.evaluations
            if evaluation.independently_clears_rubric
        ),
        key=_rank_key,
        reverse=True,
    )
    if len(accepted) < 2:
        return float("inf")
    first, second = accepted[:2]
    weakest_margin = first.scores.weakest - second.scores.weakest
    if abs(weakest_margin) > 1e-12:
        return weakest_margin
    return first.scores.aggregate - second.scores.aggregate


def _dimension_shift(
    first: AnonymousJudgeEvaluationV1,
    second: AnonymousJudgeEvaluationV1,
) -> float:
    return max(
        abs(float(getattr(first.scores, dimension)) - float(getattr(second.scores, dimension)))
        for dimension in _DIMENSIONS
    )


class BestOfFiveOrchestrator:
    MAX_TOTAL_CALLS = 4

    def __init__(
        self,
        judge: AnonymousJudge,
        *,
        fallback_judge: AnonymousJudge | None = None,
        repairer: EvidenceRepairer | None = None,
        uncertainty_limit: float = 0.5,
    ) -> None:
        if not 0.0 <= uncertainty_limit <= 1.0:
            raise ValueError("uncertainty_limit must lie in [0, 1]")
        self.judge = judge
        self.fallback_judge = fallback_judge or judge
        self.repairer = repairer
        self.uncertainty_limit = uncertainty_limit

    @staticmethod
    def _validate_evidence(
        candidates: CandidateSetV1,
        evidence: Sequence[CandidateEvidenceV1],
    ) -> tuple[dict[str, CandidateEvidenceV1], dict[str, str]]:
        candidate_ids = {candidate.candidate_id for candidate in candidates.candidates}
        by_candidate = {item.candidate_id: item for item in evidence}
        if len(evidence) != 5 or set(by_candidate) != candidate_ids:
            raise ValueError("Selection requires exactly one evidence bundle per candidate")
        anonymous_to_candidate = {
            item.anonymous_id: item.candidate_id for item in evidence
        }
        if len(anonymous_to_candidate) != 5:
            raise ValueError("Evidence anonymous IDs must be unique")
        return by_candidate, anonymous_to_candidate

    @staticmethod
    def _order(anonymous_ids: Sequence[str], seed_material: str, pass_number: int) -> tuple[str, ...]:
        ordered = sorted(anonymous_ids)
        seed = int(content_hash({"seed": seed_material, "pass": pass_number})[:16], 16)
        random.Random(seed).shuffle(ordered)
        return tuple(ordered)

    @staticmethod
    def _view(item: CandidateEvidenceV1) -> AnonymousEvidenceView:
        return AnonymousEvidenceView(
            anonymous_id=item.anonymous_id,
            trace_sha256=item.trace.sha256,
            cameras=item.cameras,
            metrics=dict(item.metrics),
        )

    def _judge_pass(
        self,
        judge: AnonymousJudge,
        pass_number: int,
        candidate_set: CandidateSetV1,
        evidence_by_candidate: dict[str, CandidateEvidenceV1],
        anonymous_to_candidate: dict[str, str],
        previous_order: tuple[str, ...] | None = None,
    ) -> JudgePassV1:
        pass_id = f"judge-pass-{pass_number}"
        order = self._order(
            tuple(anonymous_to_candidate), candidate_set.semantic_plan.content_hash(), pass_number
        )
        if previous_order is not None and order == previous_order:
            order = order[1:] + order[:1]
        views = tuple(
            self._view(evidence_by_candidate[anonymous_to_candidate[anonymous_id]])
            for anonymous_id in order
        )
        evaluations = tuple(
            judge.evaluate(pass_id, candidate_set.semantic_plan.prompt, views)
        )
        # Physics and contract gates are authoritative.  A judge may inspect a
        # failed rollout for diagnostics, but it can never vote that rollout
        # into the certified library.  Absence of the metric keeps this layer
        # backwards-compatible with pre-gate evidence fixtures; production
        # runners always emit an explicit 0.0 or 1.0.
        metrics_by_id = {view.anonymous_id: view.metrics for view in views}
        evaluations = tuple(
            evaluation.model_copy(
                update={
                    "independently_clears_rubric": False,
                    "rationale": (
                        "Deterministic certification veto: " + evaluation.rationale
                    ),
                }
            )
            if metrics_by_id.get(evaluation.anonymous_id, {}).get(
                "deterministic_gates_passed"
            )
            == 0.0
            else evaluation
            for evaluation in evaluations
        )
        return JudgePassV1(
            pass_id=pass_id,
            judge_id=judge.judge_id,
            presentation_order=order,
            evaluations=evaluations,
        )

    @staticmethod
    def _fallback_reasons(
        first: JudgePassV1, second: JudgePassV1
    ) -> list[str]:
        reasons: list[str] = []
        if _pass_winner(first) != _pass_winner(second):
            reasons.append("winner_changed")
        first_by_id = {item.anonymous_id: item for item in first.evaluations}
        second_by_id = {item.anonymous_id: item for item in second.evaluations}
        maximum_shift = max(
            _dimension_shift(first_by_id[identifier], second_by_id[identifier])
            for identifier in first_by_id
        )
        if maximum_shift > 1.0:
            reasons.append("dimension_shift_over_one")
        if min(_pass_margin(first), _pass_margin(second)) < 0.25:
            reasons.append("top_margin_below_0_25")
        return reasons

    def select(
        self,
        candidate_set: CandidateSetV1,
        evidence: Sequence[CandidateEvidenceV1],
    ) -> BestOfFiveSelectionV1:
        evidence_by_candidate, anonymous_to_candidate = self._validate_evidence(
            candidate_set, evidence
        )
        passes = [
            self._judge_pass(
                self.judge,
                1,
                candidate_set,
                evidence_by_candidate,
                anonymous_to_candidate,
            )
        ]
        passes.append(
            self._judge_pass(
                self.judge,
                2,
                candidate_set,
                evidence_by_candidate,
                anonymous_to_candidate,
                previous_order=passes[0].presentation_order,
            )
        )
        call_count = 2
        repair_calls = 0
        fallback_reasons = self._fallback_reasons(passes[0], passes[1])
        fallback_used = False

        initial_winners = [_pass_winner(item) for item in passes]
        if fallback_reasons and call_count < self.MAX_TOTAL_CALLS:
            passes.append(
                self._judge_pass(
                    self.fallback_judge,
                    3,
                    candidate_set,
                    evidence_by_candidate,
                    anonymous_to_candidate,
                    previous_order=passes[-1].presentation_order,
                )
            )
            call_count += 1
            fallback_used = True
        elif initial_winners == [None, None] and self.repairer is not None:
            # One bounded repair plus one blinded post-repair judge consumes
            # the remaining two calls. It can establish recovery evidence but
            # cannot force selection without repeated support.
            all_evaluations = [evaluation for item in passes for evaluation in item.evaluations]
            best_rejected = max(all_evaluations, key=_rank_key)
            candidate_id = anonymous_to_candidate[best_rejected.anonymous_id]
            repair_calls = 1
            call_count += 1
            replacement = self.repairer.repair(
                candidate_id,
                evidence_by_candidate[candidate_id],
                {"reason": "all_candidates_rejected", "judge_passes": 2},
            )
            if replacement is not None:
                if (
                    replacement.candidate_id != candidate_id
                    or replacement.anonymous_id != best_rejected.anonymous_id
                ):
                    raise ValueError("Repair must preserve candidate and anonymous identity")
                evidence_by_candidate[candidate_id] = replacement
            if call_count < self.MAX_TOTAL_CALLS:
                passes.append(
                    self._judge_pass(
                        self.fallback_judge,
                        3,
                        candidate_set,
                        evidence_by_candidate,
                        anonymous_to_candidate,
                        previous_order=passes[-1].presentation_order,
                    )
                )
                call_count += 1

        winners = [_pass_winner(item) for item in passes]
        nonnull = [winner for winner in winners if winner is not None]
        winner_counts = Counter(nonnull)
        selected_anonymous: str | None = None
        if len(passes) == 2 and not fallback_reasons and winners[0] == winners[1]:
            selected_anonymous = winners[0]
        elif len(passes) >= 3 and repair_calls == 0 and winner_counts:
            candidate, count = winner_counts.most_common(1)[0]
            if count >= 2 and winners[-1] == candidate:
                selected_anonymous = candidate

        if selected_anonymous is not None:
            selected_evaluations = [
                evaluation
                for item in passes
                for evaluation in item.evaluations
                if evaluation.anonymous_id == selected_anonymous
                and evaluation.independently_clears_rubric
            ]
            if (
                len(selected_evaluations) < 2
                or max(item.uncertainty for item in selected_evaluations)
                > self.uncertainty_limit
                or _pass_margin(passes[-1]) < 0.25
            ):
                selected_anonymous = None

        any_clears = any(
            evaluation.independently_clears_rubric
            for item in passes
            for evaluation in item.evaluations
        )
        if selected_anonymous is not None:
            outcome = SelectionOutcome.SELECTED
            selected_candidate_id = anonymous_to_candidate[selected_anonymous]
        elif not any_clears:
            outcome = SelectionOutcome.ALL_CANDIDATES_REJECTED
            selected_candidate_id = None
        else:
            outcome = SelectionOutcome.UNCERTAIN
            selected_candidate_id = None
        return BestOfFiveSelectionV1(
            outcome=outcome,
            selected_candidate_id=selected_candidate_id,
            judge_passes=tuple(passes),
            fallback_used=fallback_used,
            repair_calls=repair_calls,
            total_judge_and_repair_calls=call_count,
            diagnostics={
                "fallback_reasons": fallback_reasons,
                "pass_winners": winners,
                "weakest_dimension_first": True,
                "forced_winner": False,
            },
        )
