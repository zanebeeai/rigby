import os
from pathlib import Path

import pytest

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.contracts import (
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
)
from rigby_v2.flywheel.schemas import SemanticPlanV1
from rigby_v2.mujoco_executor import MujocoCertifiedExecutor, MujocoExecutionCancelled
from rigby_v2.rigging import stage_canonical_rig
from rigby_v2.scenes import stage_object_pack_scene
from rigby_v2.selection import generate_candidate_set

pytestmark = pytest.mark.medium


def test_concrete_executor_runs_real_certification_and_evidence(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "drawer",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    program = MotionProgramV2(
        program_id="neutral-drawer-attempt",
        source_text="Stand neutrally without opening the drawer.",
        duration_s=0.2,
        rig_id=rig.manifest.rig_id,
        scene_id=scene.manifest.scene_id,
        seed=23,
        phases=(
            MotionPhaseV2(
                phase_id="setup",
                kind=PhaseKind.SETUP,
                start_s=0.0,
                end_s=0.05,
            ),
            MotionPhaseV2(
                phase_id="action",
                kind=PhaseKind.ACTION,
                start_s=0.05,
                end_s=0.15,
            ),
            MotionPhaseV2(
                phase_id="recovery",
                kind=PhaseKind.RECOVERY,
                start_s=0.15,
                end_s=0.2,
            ),
        ),
        tracks=(
            MotionTrackV2(
                track_id="balanced-spine",
                target="spine_flex",
                owner="posture",
                keyframes=(
                    MotionKeyframeV2(
                        time_s=0.0, joint_values={"spine_flex": 0.0}, hard=True
                    ),
                    MotionKeyframeV2(
                        time_s=0.1, joint_values={"spine_flex": 0.01}, hard=True
                    ),
                    MotionKeyframeV2(
                        time_s=0.2, joint_values={"spine_flex": 0.0}, hard=True
                    ),
                ),
            ),
        ),
    )
    plan = SemanticPlanV1(
        plan_id="neutral-drawer-plan",
        prompt=program.source_text,
        base_program=program,
        retrieval_release="release-1",
    )
    executor = MujocoCertifiedExecutor(
        rig=rig,
        scene=scene,
        artifacts=artifacts,
    )
    candidates = generate_candidate_set(plan, executor.candidate_compiler).candidates

    results = executor.execute_many(candidates)
    result = results[0]

    assert tuple(item.candidate_id for item in results) == tuple(
        item.candidate_id for item in candidates
    )
    assert not result.certified
    assert result.repeat_count == 3
    assert result.failure_code in {
        "joint_position_limit",
        "velocity_limit",
        "object_predicate",
        "repeat_disagreement",
    }
    assert result.failed_predicate == "drawer_unit.opened"
    assert len(result.evidence.cameras) == 3
    assert all(camera.fps == 30 for camera in result.evidence.cameras)
    assert all(
        run.diagnostics["execution_backend"] == "closed_loop_process_compatible"
        and int(run.diagnostics["process_id"]) != os.getpid()
        for certification in executor._certification_results.values()
        for run in certification.simulation_runs
    )

    files_before_cancel = tuple(sorted(path for path in artifacts.root.rglob("*") if path.is_file()))
    with pytest.raises(MujocoExecutionCancelled):
        executor.execute_many(candidates, cancel_check=lambda: True)
    assert tuple(sorted(path for path in artifacts.root.rglob("*") if path.is_file())) == files_before_cancel
