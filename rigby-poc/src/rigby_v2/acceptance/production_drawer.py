"""Production-compiler physical drawer pull acceptance.

Only canonical-human actuator targets are authored.  The drawer coordinate is
never present in a target trajectory and can change only through MuJoCo
contact dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.certification import (
    CandidateCertificationRequest,
    CertificationEngine,
    CertificationResult,
    ContactPair,
    GraspPredicate,
    ReleasePredicate,
    RobustnessCertificationResult,
    certify_robustness,
)
from rigby_v2.contracts import (
    ContactEdgeV2,
    InterpolationKind,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    Vec3,
)
from rigby_v2.flywheel.schemas import CandidateProposalV1, CandidateVariationV1, PathFamily
from rigby_v2.mujoco_executor import MujocoCertifiedExecutor
from rigby_v2.rigging import stage_canonical_rig
from rigby_v2.scenes import stage_object_pack_scene
from rigby_v2.simulation import NativeMujocoRuntime, SimulationResult


@dataclass(frozen=True, slots=True)
class ProductionDrawerCase:
    executor: MujocoCertifiedExecutor
    proposal: CandidateProposalV1


@dataclass(frozen=True, slots=True)
class DrawerDiagnostics:
    production_compile_succeeded: bool
    terminal_travel_m: float
    maximum_travel_m: float
    contact_samples: int
    first_contact_s: float | None
    last_contact_s: float | None
    maximum_contact_force_n: float
    maximum_penetration_m: float
    maximum_global_penetration_m: float
    contact_geoms: tuple[str, ...]
    object_actuator_count: int
    object_target_excursion_m: float
    qpos_writes_after_initialization: int


@dataclass(frozen=True, slots=True)
class DrawerCertification:
    request: CandidateCertificationRequest
    baseline: CertificationResult
    robustness: RobustnessCertificationResult | None
    diagnostics: DrawerDiagnostics

    @property
    def robust(self) -> bool:
        return self.robustness is not None and self.robustness.robust


def _palm_keyframe(time_s: float, site: tuple[float, float, float]) -> MotionKeyframeV2:
    site = np.asarray(site, dtype=np.float64)
    return MotionKeyframeV2(
        time_s=time_s,
        position=Vec3(x=float(site[0]), y=float(site[1]), z=float(site[2])),
        hard=True,
    )


def _finger_values(scale: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for finger in ("index", "middle", "ring", "little"):
        for segment, degrees in (("mcp", 70), ("pip", 85), ("dip", 55)):
            values[f"left_{finger}_{segment}"] = float(np.deg2rad(degrees) * scale)
    for name, degrees in (
        ("thumb_opposition", 35),
        ("thumb_mcp", 50),
        ("thumb_pip", 55),
        ("thumb_dip", 40),
    ):
        values[f"left_{name}"] = float(np.deg2rad(degrees) * scale)
    return values


def _drawer_clearance_shape() -> dict[str, float]:
    values = _finger_values(0.7)
    thumb = _finger_values(0.25)
    for name in ("thumb_opposition", "thumb_mcp", "thumb_pip", "thumb_dip"):
        values[f"left_{name}"] = thumb[f"left_{name}"]
    return values


def _drawer_pregrasp_shape() -> dict[str, float]:
    values = _finger_values(0.4)
    thumb = _finger_values(0.1)
    for name in ("thumb_opposition", "thumb_mcp", "thumb_pip", "thumb_dip"):
        values[f"left_{name}"] = thumb[f"left_{name}"]
    return values


def build_production_drawer_program(*, rig_id: str, scene_id: str) -> MotionProgramV2:
    # Approach above the bar, descend into the rear clearance, establish a
    # shallow physical hook, pull 235 mm, hold, lift clear, and retreat.
    targets = (
        (0.0, (0.300, -0.480, 1.380)),
        (1.0, (0.300, -0.480, 1.380)),
        (3.0, (0.300, -0.445, 1.263)),
        (3.5, (0.300, -0.445, 1.263)),
        (4.7, (0.300, -0.220, 1.263)),
        (5.4, (0.300, -0.220, 1.263)),
        (6.0, (0.300, -0.015, 1.263)),
        (6.3, (0.550, -0.015, 1.400)),
    )
    return MotionProgramV2(
        program_id="production-drawer-open",
        source_text=(
            "Approach the drawer handle clear, form a measured rear-palm hook, "
            "pull the passive drawer open by physical contact, hold, release, and retreat."
        ),
        duration_s=6.9,
        rig_id=rig_id,
        scene_id=scene_id,
        seed=20260811,
        phases=(
            MotionPhaseV2(phase_id="approach", kind=PhaseKind.SETUP, start_s=0.0, end_s=1.9),
            MotionPhaseV2(
                phase_id="contact", kind=PhaseKind.ACTION, start_s=1.9, end_s=3.5, energy=0.0
            ),
            MotionPhaseV2(
                phase_id="pull", kind=PhaseKind.ACTION, start_s=3.5, end_s=4.7, energy=0.0
            ),
            MotionPhaseV2(phase_id="hold", kind=PhaseKind.HOLD, start_s=4.7, end_s=5.4),
            MotionPhaseV2(phase_id="release", kind=PhaseKind.RELEASE, start_s=5.4, end_s=6.0),
            MotionPhaseV2(phase_id="recovery", kind=PhaseKind.RECOVERY, start_s=6.0, end_s=6.9),
        ),
        tracks=(
            MotionTrackV2(
                track_id="left-palm-drawer-hook",
                target="left_palm",
                owner="left_arm",
                interpolation=InterpolationKind.SLERP,
                keyframes=tuple(_palm_keyframe(time_s, center) for time_s, center in targets),
            ),
            MotionTrackV2(
                track_id="left-hand-drawer-clearance",
                target="left_hand_joints",
                owner="left_hand",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=tuple(
                    MotionKeyframeV2(
                        time_s=time_s,
                        joint_values=(
                            _drawer_pregrasp_shape()
                            if index < 3
                            else (
                                _drawer_clearance_shape()
                                if index < len(targets) - 2
                                else _finger_values(0.0)
                            )
                        ),
                        hard=True,
                    )
                    for index, (time_s, _) in enumerate(targets)
                ),
            ),
        ),
        contacts=(
            ContactEdgeV2(
                contact_id="left-palm-handle-hook",
                body_a="left_lower_arm",
                body_b="obj__drawer_unit__drawer__handle",
                start_s=1.95,
                end_s=5.8,
                max_slip_m=0.3,
                max_penetration_m=0.002,
            ),
        ),
        metadata={"task_space_sample_hz": 4, "object_target_channels": 0},
    )


def build_production_drawer_case(artifact_root: Path) -> ProductionDrawerCase:
    artifacts = ContentAddressedArtifactStore(artifact_root)
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "drawer", profile="medium", rig_reference=rig.reference, artifacts=artifacts
    )
    program = build_production_drawer_program(
        rig_id=rig.manifest.rig_id, scene_id=scene.manifest.scene_id
    )
    return ProductionDrawerCase(
        executor=MujocoCertifiedExecutor(rig=rig, scene=scene, artifacts=artifacts),
        proposal=CandidateProposalV1(
            candidate_id="production-drawer-open",
            semantic_plan_hash="d" * 64,
            variation=CandidateVariationV1(
                retrieval_seed=20260811,
                path_family=PathFamily.CONTACT_FIRST,
                timing_style="neutral",
                energy=0.35,
                body_participation=("left_arm", "left_hand"),
                contact_strategy="rear_palm_handle_hook",
            ),
            program=program,
            compile_attempt=2,
        ),
    )


def _enriched_request(case: ProductionDrawerCase) -> CandidateCertificationRequest:
    request = case.executor._prepare_certification_request(case.proposal)  # noqa: SLF001
    standing = request.simulation.config.standing
    assert standing is not None
    allowed_pairs = request.policy.allowed_contact_pairs
    assert allowed_pairs is not None
    task_pairs = frozenset(
        {
            ContactPair.of(
                "left_lower_arm_collision", "obj__drawer_unit__drawer__handle"
            ),
            ContactPair.of(
                "left_index_proximal_collision", "left_thumb_distal_collision"
            ),
            ContactPair.of(
                "left_middle_middle_collision", "left_ring_proximal_collision"
            ),
            ContactPair.of("left_palm_collision", "left_thumb_middle_collision"),
        }
    )
    return replace(
        request,
        simulation=replace(
            request.simulation,
            config=replace(
                request.simulation.config,
                standing=replace(standing, posture_kp=70.0, posture_kd=26.0),
            ),
        ),
        policy=replace(
            request.policy,
            allowed_contact_pairs=allowed_pairs | task_pairs,
            contact_window_tolerance_s=0.15,
            unplanned_contact_persistence_s=0.06,
        ),
        predicates=request.predicates
        + (
            GraspPredicate(
                object_geoms=frozenset({"obj__drawer_unit__drawer__handle"}),
                effector_geoms=frozenset({"left_palm_collision"}),
                start_s=1.90,
                end_s=5.45,
                min_distinct_effectors=1,
                name="measured_handle_hook_contact",
            ),
            ReleasePredicate(
                object_geoms=frozenset({"obj__drawer_unit__drawer__handle"}),
                effector_geoms=frozenset({"left_palm_collision"}),
                start_s=6.5,
                settle_duration_s=0.4,
                name="measured_handle_release",
            ),
        ),
    )


def _diagnostics(case: ProductionDrawerCase, result: SimulationResult) -> DrawerDiagnostics:
    model = case.executor.scene_model
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj__drawer_unit__drawer_slide")
    address = int(model.jnt_qposadr[joint])
    selected = [
        (frame.time_s, contact)
        for frame in result.trace.contacts
        for contact in frame.contacts
        if "obj__drawer_unit__drawer__handle" in (contact.geom1_name, contact.geom2_name)
        and any(name.startswith("left_") for name in (contact.geom1_name, contact.geom2_name))
    ]
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(model.nu)
    )
    trajectory = _enriched_request(case).simulation.trajectory
    object_targets = np.asarray(trajectory.qpos)[:, address]
    return DrawerDiagnostics(
        production_compile_succeeded=True,
        terminal_travel_m=float(result.trace.qpos[-1, address]),
        maximum_travel_m=float(np.max(result.trace.qpos[:, address])),
        contact_samples=len(selected),
        first_contact_s=None if not selected else float(selected[0][0]),
        last_contact_s=None if not selected else float(selected[-1][0]),
        maximum_contact_force_n=max((item.normal_force_n for _, item in selected), default=0.0),
        maximum_penetration_m=max(
            (max(0.0, -item.distance_m) for _, item in selected), default=0.0
        ),
        maximum_global_penetration_m=max(
            (
                max(0.0, -item.distance_m)
                for frame in result.trace.contacts
                for item in frame.contacts
            ),
            default=0.0,
        ),
        contact_geoms=tuple(
            sorted(
                {
                    item.geom1_name if item.geom1_name != "obj__drawer_unit__drawer__handle" else item.geom2_name
                    for _, item in selected
                }
            )
        ),
        object_actuator_count=sum("drawer" in name for name in actuator_names),
        object_target_excursion_m=float(np.ptp(object_targets)),
        qpos_writes_after_initialization=int(result.diagnostics["qpos_writes_after_initialization"]),
    )


def run_production_drawer_once(case: ProductionDrawerCase) -> tuple[SimulationResult, DrawerDiagnostics]:
    request = _enriched_request(case)
    result = NativeMujocoRuntime().simulate(request.simulation)
    return result, _diagnostics(case, result)


def certify_production_drawer(case: ProductionDrawerCase) -> DrawerCertification:
    request = _enriched_request(case)
    engine = CertificationEngine(NativeMujocoRuntime())
    baseline = engine.certify(request)
    diagnostics = _diagnostics(case, baseline.simulation_runs[0])
    robustness = (
        certify_robustness(request, engine=engine, baseline=baseline)
        if baseline.certified
        else None
    )
    return DrawerCertification(request, baseline, robustness, diagnostics)
