from __future__ import annotations

from dataclasses import dataclass

from rigby_v2.contracts import ArtifactRefV1
from rigby_v2.flywheel.schemas import (
    AnonymousJudgeEvaluationV1,
    CameraEvidenceV1,
    CandidateEvidenceV1,
    EvidenceCameraRole,
    RubricScoresV1,
    SelectionOutcome,
)
from rigby_v2.selection import BestOfFiveOrchestrator, generate_candidate_set

from test_v2_selection_generation import semantic_plan
import pytest

pytestmark = pytest.mark.fast


IDENTITY = tuple(float(value) for value in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
INTRINSICS = (100.0, 0.0, 79.5, 0.0, 100.0, 59.5, 0.0, 0.0, 1.0)


def _artifact(index: int) -> ArtifactRefV1:
    digit = "0123456789abcdef"[index % 16]
    return ArtifactRefV1(sha256=digit * 64, size_bytes=10)


def evidence_set() -> tuple[CandidateEvidenceV1, ...]:
    candidates = generate_candidate_set(semantic_plan()).candidates
    result = []
    for index, candidate in enumerate(candidates):
        cameras = tuple(
            CameraEvidenceV1(
                camera_id=f"anon-{index}-{role.value}",
                role=role,
                raw_frames=_artifact(index + 1),
                video=_artifact(index + 6),
                timestamps_s=(0.0, 1 / 30),
                world_from_camera=(IDENTITY, IDENTITY),
                intrinsics=INTRINSICS,
            )
            for role in EvidenceCameraRole
        )
        result.append(
            CandidateEvidenceV1(
                anonymous_id=f"anon-{index}",
                candidate_id=candidate.candidate_id,
                trace=_artifact(index + 11),
                cameras=cameras,
                metrics={"candidate_index": float(index)},
            )
        )
    return tuple(result)


def _scores(weakest: float, other: float | None = None) -> RubricScoresV1:
    other = weakest if other is None else other
    return RubricScoresV1(
        semantic_fidelity=weakest,
        physical_plausibility=other,
        contact_quality=other,
        timing_energy=other,
        whole_body_quality=other,
        visual_clarity=other,
    )


@dataclass
class ScriptedJudge:
    scripts: list[dict[str, tuple[RubricScoresV1, bool, float]]]
    judge_id: str = "fake-judge"

    def __post_init__(self) -> None:
        self.calls = 0
        self.orders: list[tuple[str, ...]] = []

    def evaluate(self, pass_id, prompt, evidence):  # type: ignore[no-untyped-def]
        del pass_id
        assert prompt == "reach and press the button"
        assert all(not hasattr(item, "candidate_id") for item in evidence)
        self.orders.append(tuple(item.anonymous_id for item in evidence))
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return tuple(
            AnonymousJudgeEvaluationV1(
                anonymous_id=item.anonymous_id,
                scores=script[item.anonymous_id][0],
                independently_clears_rubric=script[item.anonymous_id][1],
                uncertainty=script[item.anonymous_id][2],
                rationale="deterministic fake evaluation",
            )
            for item in evidence
        )


def _script(
    accepted: dict[str, RubricScoresV1], *, uncertainty: float = 0.1
) -> dict[str, tuple[RubricScoresV1, bool, float]]:
    return {
        f"anon-{index}": (
            accepted.get(f"anon-{index}", _scores(1.0)),
            f"anon-{index}" in accepted,
            uncertainty,
        )
        for index in range(5)
    }


def test_two_randomized_passes_rank_weakest_dimension_before_aggregate() -> None:
    # anon-0 has a much higher average but a weak 2.0 dimension. anon-1 wins
    # because its minimum is 3.0, proving aggregate cannot hide one bad axis.
    script = _script({"anon-0": _scores(2.0, 5.0), "anon-1": _scores(3.0)})
    judge = ScriptedJudge([script, script])
    candidates = generate_candidate_set(semantic_plan())
    result = BestOfFiveOrchestrator(judge).select(candidates, evidence_set())
    assert result.outcome is SelectionOutcome.SELECTED
    assert result.selected_candidate_id == candidates.candidates[1].candidate_id
    assert result.total_judge_and_repair_calls == 2
    assert judge.orders[0] != judge.orders[1]
    assert set(judge.orders[0]) == {f"anon-{index}" for index in range(5)}


def test_winner_change_invokes_fallback_and_uses_its_tie_break() -> None:
    first = _script({"anon-0": _scores(4.0), "anon-1": _scores(3.0)})
    second = _script({"anon-0": _scores(3.0), "anon-1": _scores(4.0)})
    fallback_script = _script({"anon-0": _scores(2.5), "anon-1": _scores(4.5)})
    primary = ScriptedJudge([first, second], judge_id="primary")
    fallback = ScriptedJudge([fallback_script], judge_id="fallback")
    candidates = generate_candidate_set(semantic_plan())
    result = BestOfFiveOrchestrator(primary, fallback_judge=fallback).select(
        candidates, evidence_set()
    )
    assert result.fallback_used
    assert "winner_changed" in result.diagnostics["fallback_reasons"]
    assert result.outcome is SelectionOutcome.SELECTED
    assert result.selected_candidate_id == candidates.candidates[1].candidate_id
    assert result.total_judge_and_repair_calls == 3


def test_low_margin_and_dimension_shift_trigger_fallback() -> None:
    first = _script({"anon-0": _scores(3.2), "anon-1": _scores(3.1)})
    second = _script({"anon-0": _scores(4.4), "anon-1": _scores(4.3)})
    fallback_script = _script({"anon-0": _scores(4.5), "anon-1": _scores(3.0)})
    result = BestOfFiveOrchestrator(
        ScriptedJudge([first, second]),
        fallback_judge=ScriptedJudge([fallback_script]),
    ).select(generate_candidate_set(semantic_plan()), evidence_set())
    assert result.fallback_used
    assert set(result.diagnostics["fallback_reasons"]) == {
        "dimension_shift_over_one",
        "top_margin_below_0_25",
    }
    assert result.outcome is SelectionOutcome.SELECTED


def test_all_rejected_is_never_forced_into_a_winner() -> None:
    rejected = _script({})
    result = BestOfFiveOrchestrator(ScriptedJudge([rejected, rejected])).select(
        generate_candidate_set(semantic_plan()), evidence_set()
    )
    assert result.outcome is SelectionOutcome.ALL_CANDIDATES_REJECTED
    assert result.selected_candidate_id is None
    assert result.diagnostics["forced_winner"] is False


class OneRepair:
    def __init__(self) -> None:
        self.calls = 0

    def repair(self, candidate_id, evidence, diagnostics):  # type: ignore[no-untyped-def]
        assert diagnostics["reason"] == "all_candidates_rejected"
        assert evidence.candidate_id == candidate_id
        self.calls += 1
        return CandidateEvidenceV1.model_validate(
            {**evidence.model_dump(mode="python"), "metrics": {"repaired": 1.0}}
        )


class RepairAwareJudge(ScriptedJudge):
    def evaluate(self, pass_id, prompt, evidence):  # type: ignore[no-untyped-def]
        if any(item.metrics.get("repaired") == 1.0 for item in evidence):
            repaired = next(item.anonymous_id for item in evidence if item.metrics.get("repaired") == 1.0)
            self.scripts.append(_script({repaired: _scores(4.0)}))
        return super().evaluate(pass_id, prompt, evidence)


def test_one_bounded_repair_and_post_repair_judge_respect_four_call_cap() -> None:
    rejected = _script({})
    judge = RepairAwareJudge([rejected, rejected])
    repair = OneRepair()
    result = BestOfFiveOrchestrator(judge, repairer=repair).select(
        generate_candidate_set(semantic_plan()), evidence_set()
    )
    assert repair.calls == 1
    assert result.repair_calls == 1
    assert result.total_judge_and_repair_calls == 4
    assert len(result.judge_passes) == 3
    assert result.outcome is SelectionOutcome.UNCERTAIN
    assert result.selected_candidate_id is None


def test_high_uncertainty_returns_uncertain_even_with_same_winner() -> None:
    script = _script({"anon-0": _scores(4.5)}, uncertainty=0.9)
    result = BestOfFiveOrchestrator(ScriptedJudge([script, script])).select(
        generate_candidate_set(semantic_plan()), evidence_set()
    )
    assert result.outcome is SelectionOutcome.UNCERTAIN
    assert result.selected_candidate_id is None


def test_deterministic_gate_failure_vetoes_judge_winner() -> None:
    evidence = list(evidence_set())
    evidence[0] = evidence[0].model_copy(
        update={"metrics": {"deterministic_gates_passed": 0.0}}
    )
    script = _script({"anon-0": _scores(5.0), "anon-1": _scores(4.0)})
    candidates = generate_candidate_set(semantic_plan())
    result = BestOfFiveOrchestrator(ScriptedJudge([script, script])).select(
        candidates, evidence
    )
    assert result.outcome is SelectionOutcome.SELECTED
    assert result.selected_candidate_id == candidates.candidates[1].candidate_id
    vetoed = [
        evaluation
        for judge_pass in result.judge_passes
        for evaluation in judge_pass.evaluations
        if evaluation.anonymous_id == "anon-0"
    ]
    assert vetoed and all(not evaluation.independently_clears_rubric for evaluation in vetoed)
