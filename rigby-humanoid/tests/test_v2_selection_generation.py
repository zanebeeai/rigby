from __future__ import annotations

import pytest

from rigby_v2.contracts import (
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
)
from rigby_v2.flywheel.schemas import PathFamily, SemanticPlanV1
from rigby_v2.motion import MotionCompilationError, MotionFailureReason
from rigby_v2.selection import (
    CandidateGenerationError,
    CandidateGenerationReason,
    DeterministicCandidateCompiler,
    generate_candidate_set,
    structural_fingerprint,
)

pytestmark = pytest.mark.fast


def semantic_plan() -> SemanticPlanV1:
    program = MotionProgramV2(
        program_id="base-program",
        source_text="reach and press the button",
        duration_s=1.0,
        rig_id="rig-v1",
        scene_id="button-scene",
        seed=17,
        phases=(
            MotionPhaseV2(
                phase_id="action", kind=PhaseKind.ACTION, start_s=0, end_s=1, energy=0.5
            ),
        ),
        tracks=(
            MotionTrackV2(
                track_id="reach",
                target="right_hand",
                owner="right_arm",
                keyframes=(
                    MotionKeyframeV2(time_s=0, joint_values={"shoulder": 0.0}),
                    MotionKeyframeV2(time_s=1, joint_values={"shoulder": 0.8}),
                ),
            ),
        ),
    )
    return SemanticPlanV1(
        plan_id="plan-1",
        prompt=program.source_text,
        base_program=program,
        retrieval_release="N",
    )


def test_one_plan_produces_exactly_five_structurally_diverse_candidates() -> None:
    result = generate_candidate_set(semantic_plan())
    assert len(result.candidates) == 5
    assert result.total_compile_attempts == 5
    assert result.repair_rounds == 0
    assert {item.variation.path_family for item in result.candidates} == set(PathFamily)
    assert len({structural_fingerprint(item.program) for item in result.candidates}) == 5
    assert len({item.candidate_id for item in result.candidates}) == 5
    assert all(item.semantic_plan_hash == result.semantic_plan.content_hash() for item in result.candidates)
    route_signatures = {
        tuple(
            (keyframe.time_s, tuple(sorted(keyframe.joint_values.items())))
            for keyframe in item.program.tracks[0].keyframes
        )
        for item in result.candidates
    }
    assert len(route_signatures) == 5
    assert all(len(item.program.tracks[0].keyframes) == 3 for item in result.candidates)


class FailOnceCompiler(DeterministicCandidateCompiler):
    def __init__(self) -> None:
        self.compile_calls = 0
        self.repair_calls = 0

    def compile(self, plan, variation, attempt):  # type: ignore[no-untyped-def]
        self.compile_calls += 1
        if self.compile_calls == 1:
            raise MotionCompilationError(MotionFailureReason.IK_INFEASIBLE, "unreachable")
        return super().compile(plan, variation, attempt)

    def repair(self, plan, variation, attempt, failure):  # type: ignore[no-untyped-def]
        self.repair_calls += 1
        return super().repair(plan, variation, attempt, failure)


def test_generation_uses_at_most_one_bounded_repair_round() -> None:
    compiler = FailOnceCompiler()
    result = generate_candidate_set(semantic_plan(), compiler)
    assert len(result.candidates) == 5
    assert result.total_compile_attempts == 6
    assert result.repair_rounds == 1
    assert compiler.repair_calls == 1
    assert result.candidates[0].repaired is True


class DuplicateCompiler(DeterministicCandidateCompiler):
    def compile(self, plan, variation, attempt):  # type: ignore[no-untyped-def]
        del variation, attempt
        return plan.base_program


def test_structural_duplicates_are_rejected_until_twenty_attempt_cap() -> None:
    with pytest.raises(CandidateGenerationError) as failure:
        generate_candidate_set(semantic_plan(), DuplicateCompiler())
    assert failure.value.reason is CandidateGenerationReason.INSUFFICIENT_DIVERSE_CANDIDATES
    assert failure.value.details["attempts"] == 20


class AlwaysFailCompiler(DeterministicCandidateCompiler):
    def __init__(self) -> None:
        self.repair_calls = 0

    def compile(self, plan, variation, attempt):  # type: ignore[no-untyped-def]
        del plan, variation, attempt
        raise RuntimeError("compile failed")

    def repair(self, plan, variation, attempt, failure):  # type: ignore[no-untyped-def]
        del plan, variation, attempt, failure
        self.repair_calls += 1
        raise RuntimeError("repair failed")


def test_failed_generation_never_exceeds_one_repair_or_twenty_compiles() -> None:
    compiler = AlwaysFailCompiler()
    with pytest.raises(CandidateGenerationError) as failure:
        generate_candidate_set(semantic_plan(), compiler)
    assert failure.value.details["attempts"] == 20
    assert compiler.repair_calls == 1
