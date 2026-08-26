from __future__ import annotations

from pathlib import Path

import mujoco

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.certification import (
    CandidateCertificationRequest,
    CertificationEngine,
    CertificationOutcome,
    policy_from_manifests,
)
from rigby_v2.contracts import (
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
)
from rigby_v2.motion import TimingProfile, compile_motion_program
from rigby_v2.rigging import validate_simulated_glb_round_trip
from rigby_v2.rigging.manifest_builder import stage_canonical_rig
from rigby_v2.scenes import stage_object_pack_scene
from rigby_v2.simulation import (
    LinearKeyframeTrajectory,
    SimulationConfig,
    SimulationRequest,
)
from rigby_v2.worker import (
    _canonical_standing_config,
    _map_generalized_state,
    _project_qpos_to_rig,
)


def test_program_to_mjz_scene_to_three_run_certification_vertical(tmp_path: Path) -> None:
    artifacts = ContentAddressedArtifactStore(tmp_path / "artifacts")
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "drawer",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    rig_xml = artifacts.read_bytes(rig.manifest.mjcf).decode("utf-8")
    rig_model = mujoco.MjModel.from_xml_string(rig_xml)
    scene_model = scene.compiled.load_model()
    program = MotionProgramV2(
        program_id="certified-neutral-motion",
        source_text="Maintain a balanced neutral stance in front of the drawer.",
        duration_s=0.2,
        rig_id=rig.manifest.rig_id,
        scene_id=scene.manifest.scene_id,
        seed=17,
        phases=(
            MotionPhaseV2(
                phase_id="setup",
                kind=PhaseKind.SETUP,
                start_s=0.0,
                end_s=0.05,
                energy=0.3,
            ),
            MotionPhaseV2(
                phase_id="action",
                kind=PhaseKind.ACTION,
                start_s=0.05,
                end_s=0.10,
                energy=0.7,
            ),
            MotionPhaseV2(
                phase_id="hold",
                kind=PhaseKind.HOLD,
                start_s=0.10,
                end_s=0.15,
            ),
            MotionPhaseV2(
                phase_id="recovery",
                kind=PhaseKind.RECOVERY,
                start_s=0.15,
                end_s=0.20,
                energy=0.2,
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
                            time_s=0.10, joint_values={"spine_flex": 0.0}, hard=True
                        ),
                    MotionKeyframeV2(
                        time_s=0.20, joint_values={"spine_flex": 0.0}, hard=True
                    ),
                ),
            ),
        ),
    )
    compiled = compile_motion_program(
        program,
        rig_model,
        rig.manifest,
        timing_profile=TimingProfile.ENERGETIC,
    )
    persisted = compiled.to_contract(rig_id=rig.manifest.rig_id)
    assert persisted.program_hash == program.content_hash()
    assert persisted.rig_hash == rig.reference.sha256
    assert persisted.samples[1].qacc

    qpos, qvel = _map_generalized_state(
        rig_model, scene_model, compiled.qpos, compiled.qvel
    )
    _, qacc = _map_generalized_state(
        rig_model, scene_model, compiled.qpos, compiled.qacc
    )
    request = SimulationRequest(
        model_xml=None,
        model_mjz=scene.compiled.mjz_bytes,
        trajectory=LinearKeyframeTrajectory(
            times_s=compiled.times_s,
            qpos=qpos,
            qvel=qvel,
            qacc=qacc,
        ),
        config=SimulationConfig(
            duration_s=program.duration_s,
            free_root_joint_name="pelvis_free",
            standing=_canonical_standing_config(scene_model),
        ),
        initial_qpos=qpos[0],
        initial_qvel=qvel[0],
        request_id=compiled.candidate_id,
    )
    source_glb = artifacts.read_bytes(rig.manifest.visual_asset)

    def export_reimport(result):
        return validate_simulated_glb_round_trip(
            source_glb=source_glb,
            model_xml=rig_xml,
            rig=rig.manifest,
            times_s=result.trace.times_s,
            qpos=_project_qpos_to_rig(
                scene_model, rig_model, result.trace.qpos
            ),
        )

    certification = CertificationEngine().certify(
        CandidateCertificationRequest(
            simulation=request,
            policy=policy_from_manifests(rig.manifest, scene.manifest),
            export_reimport=export_reimport,
        )
    )

    assert certification.outcome is CertificationOutcome.CERTIFIED
    assert len(certification.simulation_runs) == 3
    assert not certification.violations
    assert all(run.model_hash == scene.compiled.mjz_sha256 for run in certification.simulation_runs)
    assert all(
        max(0.0, -contact.distance_m) <= 0.002
        for run in certification.simulation_runs
        for frame in run.trace.contacts
        for contact in frame.contacts
    )
