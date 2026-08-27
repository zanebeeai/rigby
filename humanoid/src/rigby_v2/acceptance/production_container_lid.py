"""Bounded production-path attempt for the passive container lid pack.

The lid and its free base never appear in a control target.  The only authored
channels are canonical-human position actuators; all object motion therefore
comes from measured MuJoCo contact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.certification import (
    CandidateCertificationRequest,
    GraspPredicate,
    ReleasePredicate,
)
from rigby_core.contracts import (
    ContactEdgeV2,
    CoordinateFrame,
    InterpolationKind,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    PhaseKind,
    Quaternion,
    QuaternionConvention,
    QuaternionMeaning,
    QuaternionOrder,
    Vec3,
)
from rigby_v2.flywheel.schemas import (
    CandidateProposalV1,
    CandidateVariationV1,
    PathFamily,
)
from rigby_v2.mujoco_executor import MujocoCertifiedExecutor
from rigby_v2.rigging import stage_canonical_rig
from rigby_v2.scenes import stage_object_pack_scene
from rigby_v2.simulation import NativeMujocoRuntime, SimulationResult


_WORLD_WXYZ = QuaternionConvention(
    order=QuaternionOrder.WXYZ,
    frame=CoordinateFrame.WORLD,
    meaning=QuaternionMeaning.ABSOLUTE,
)
_TIMES = (0.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0)
_ANGLES_DEG = (0.0, 0.0, 0.0, 35.0, 70.0, 80.0, 80.0, 80.0, 80.0)
_RIGHT_ARM = (
    "right_shoulder_flex",
    "right_shoulder_abduct",
    "right_shoulder_twist",
    "right_elbow_flex",
    "right_forearm_twist",
    "right_wrist_flex",
    "right_wrist_deviation",
)
_LEFT_ARM = tuple(name.replace("right_", "left_") for name in _RIGHT_ARM)

# Sequential, joint-limit-safe solutions of the actual medium model at the
# 0/35/70/80-degree knob poses.  They are seeds only; the production task-space
# compiler must independently meet every hard world-space palm objective.
_RIGHT_SEEDS = (
    np.asarray((-2.37902094, -0.25984691, 1.09548745, 0.08942358, -0.97967765, -0.00063353, -0.61086278)),
    np.asarray((-2.02898932, -0.63614335, 1.13902770, 1.31666177, -0.49927223, 0.29527558, -0.61085773)),
    np.asarray((-1.85767731, -0.81909601, 1.33114896, 2.09273606, -0.24714983, 0.18464587, -0.61086424)),
    np.asarray((-1.81824073, -0.86304335, 1.40054873, 2.30604863, -0.15361621, 0.10352942, -0.61086424)),
)
_LEFT_CLEAR = np.asarray((-0.41576186, 0.69264934, -1.745329, 0.66092530, -1.09800887, -1.15773373, 0.610864))
_LEFT_CONTACT = np.asarray((-0.25009795, 0.63299891, -1.745329, 0.76864636, -1.06573682, -1.29675829, 0.610864))


@dataclass(frozen=True, slots=True)
class ProductionContainerLidCase:
    executor: MujocoCertifiedExecutor
    proposal: CandidateProposalV1


@dataclass(frozen=True, slots=True)
class ContainerLidDiagnostics:
    production_compile_succeeded: bool
    terminal_lid_angle_deg: float
    maximum_lid_angle_deg: float
    knob_contact_samples: int
    knob_contact_effectors: tuple[str, ...]
    first_knob_contact_s: float | None
    last_knob_contact_s: float | None
    maximum_knob_force_n: float
    base_contact_samples: int
    maximum_penetration_m: float
    object_actuator_count: int
    object_target_excursion_rad: float
    qpos_writes_after_initialization: int


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)))


def _palm_site(side: str, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    offset = np.asarray((-0.005 if side == "left" else 0.005, 0.043, 0.0))
    return center - rotation @ offset


def _pose_frame(time_s: float, position: np.ndarray, rotation: np.ndarray) -> MotionKeyframeV2:
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
    return MotionKeyframeV2(
        time_s=time_s,
        position=Vec3(x=float(position[0]), y=float(position[1]), z=float(position[2])),
        rotation=Quaternion(
            values=tuple(float(value) for value in quaternion),
            convention=_WORLD_WXYZ,
        ),
        hard=True,
    )


def _joint_frame(time_s: float, names: tuple[str, ...], values: np.ndarray) -> MotionKeyframeV2:
    return MotionKeyframeV2(
        time_s=time_s,
        joint_values={name: float(value) for name, value in zip(names, values, strict=True)},
        hard=True,
    )


def _finger_values(scale: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for finger in ("index", "middle", "ring", "little"):
        for segment, degrees in (("mcp", 70), ("pip", 85), ("dip", 55)):
            values[f"right_{finger}_{segment}"] = float(np.deg2rad(degrees) * scale)
    for name, degrees in (("thumb_opposition", 35), ("thumb_mcp", 50), ("thumb_pip", 55), ("thumb_dip", 40)):
        values[f"right_{name}"] = float(np.deg2rad(degrees) * scale)
    return values


def _right_targets() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    # The calibrated pack's closed knob is (-.08, -.48, 1.17).  Its local +Z
    # normal follows the passive -X hinge.  The 42 mm center separation is the
    # sum of the knob radius and palm half-thickness less 1 mm contact depth.
    knob_closed = np.asarray((-0.08, -0.48, 1.17))
    hinge = np.asarray((0.0, -0.34 + 0.14, 0.905 + 0.21))
    relative = knob_closed - hinge
    targets: list[tuple[np.ndarray, np.ndarray]] = []
    for index, angle_deg in enumerate(_ANGLES_DEG):
        rotation = _rotation_x(-np.deg2rad(angle_deg))
        knob = hinge + rotation @ relative
        normal = rotation[:, 2]
        separation = 0.092 if index == 0 else 0.12 if index >= 7 else 0.042
        targets.append((_palm_site("right", knob - separation * normal, rotation), rotation))
    return tuple(targets)


def build_production_container_lid_program(*, rig_id: str, scene_id: str) -> MotionProgramV2:
    right_targets = _right_targets()
    left_clear = _palm_site("left", np.asarray((0.331, -0.34, 1.005)), np.eye(3))
    left_contact = _palm_site("left", np.asarray((0.261, -0.34, 1.005)), np.eye(3))
    left_positions = (left_clear, left_contact, left_contact, left_contact, left_contact, left_contact, left_contact, left_clear, left_clear)
    right_seed_indices = (0, 0, 0, 1, 2, 3, 3, 3, 3)
    opened = _finger_values(0.0)
    closed = _finger_values(0.35)
    return MotionProgramV2(
        program_id="production-container-lid-open",
        source_text=(
            "Clear bimanual approach, base stabilization, articulated right-hand knob contact, "
            "passive hinge opening through 80 degrees, release, and settle."
        ),
        duration_s=12.0,
        rig_id=rig_id,
        scene_id=scene_id,
        seed=20260811,
        phases=(
            MotionPhaseV2(phase_id="approach", kind=PhaseKind.SETUP, start_s=0.0, end_s=2.0),
            MotionPhaseV2(phase_id="contact", kind=PhaseKind.ACTION, start_s=2.0, end_s=3.0),
            MotionPhaseV2(phase_id="open", kind=PhaseKind.ACTION, start_s=3.0, end_s=8.0),
            MotionPhaseV2(phase_id="hold", kind=PhaseKind.HOLD, start_s=8.0, end_s=9.0),
            MotionPhaseV2(phase_id="release", kind=PhaseKind.RELEASE, start_s=9.0, end_s=10.0),
            MotionPhaseV2(phase_id="settle", kind=PhaseKind.RECOVERY, start_s=10.0, end_s=12.0),
        ),
        tracks=(
            MotionTrackV2(
                track_id="left-palm-base-stabilize",
                target="left_palm",
                owner="left_arm",
                interpolation=InterpolationKind.SLERP,
                keyframes=tuple(_pose_frame(time_s, position, np.eye(3)) for time_s, position in zip(_TIMES, left_positions, strict=True)),
            ),
            MotionTrackV2(
                track_id="left-arm-reachable-seed",
                target="left_arm_joints",
                owner="left_arm",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=tuple(_joint_frame(time_s, _LEFT_ARM, _LEFT_CLEAR if index in (0, 7, 8) else _LEFT_CONTACT) for index, time_s in enumerate(_TIMES)),
            ),
            MotionTrackV2(
                track_id="right-palm-knob-arc",
                target="right_palm",
                owner="right_arm",
                interpolation=InterpolationKind.SLERP,
                keyframes=tuple(_pose_frame(time_s, position, rotation) for time_s, (position, rotation) in zip(_TIMES, right_targets, strict=True)),
            ),
            MotionTrackV2(
                track_id="right-arm-reachable-seed",
                target="right_arm_joints",
                owner="right_arm",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=tuple(_joint_frame(time_s, _RIGHT_ARM, _RIGHT_SEEDS[seed_index]) for time_s, seed_index in zip(_TIMES, right_seed_indices, strict=True)),
            ),
            MotionTrackV2(
                track_id="right-articulated-knob-grasp",
                target="right_hand_joints",
                owner="right_hand",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=tuple(MotionKeyframeV2(time_s=time_s, joint_values=opened if index in (0, 1, 7, 8) else closed, hard=True) for index, time_s in enumerate(_TIMES)),
            ),
        ),
        contacts=(
            ContactEdgeV2(contact_id="left-base-stabilize", body_a="left_hand", body_b="obj__container__base", start_s=2.0, end_s=9.0, max_penetration_m=0.002),
            ContactEdgeV2(contact_id="right-knob-open", body_a="right_hand", body_b="obj__container__lid_knob", start_s=2.0, end_s=9.0, max_penetration_m=0.002),
        ),
        metadata={"task_space_sample_hz": 1, "object_target_channels": 0},
    )


def build_production_container_lid_case(artifact_root: Path) -> ProductionContainerLidCase:
    artifacts = ContentAddressedArtifactStore(artifact_root)
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene("container_lid", profile="medium", rig_reference=rig.reference, artifacts=artifacts)
    program = build_production_container_lid_program(rig_id=rig.manifest.rig_id, scene_id=scene.manifest.scene_id)
    proposal = CandidateProposalV1(
        candidate_id="production-container-lid-open",
        semantic_plan_hash="c" * 64,
        variation=CandidateVariationV1(
            retrieval_seed=20260811,
            path_family=PathFamily.CONTACT_FIRST,
            timing_style="neutral",
            energy=0.3,
            body_participation=("left_arm", "right_arm", "right_hand"),
            contact_strategy="stabilize_base_and_follow_knob_arc",
        ),
        program=program,
        compile_attempt=2,
    )
    return ProductionContainerLidCase(
        executor=MujocoCertifiedExecutor(rig=rig, scene=scene, artifacts=artifacts),
        proposal=proposal,
    )


_KNOB_GEOMS = frozenset({"obj__container__lid_knob__knob_neck", "obj__container__lid_knob__knob_grip"})
_RIGHT_EFFECTORS = frozenset(
    {"right_palm_collision"}
    | {f"right_{finger}_{segment}_collision" for finger in ("thumb", "index", "middle", "ring", "little") for segment in ("proximal", "middle", "distal")}
)


def enriched_request(case: ProductionContainerLidCase) -> CandidateCertificationRequest:
    request = case.executor._prepare_certification_request(case.proposal)  # noqa: SLF001
    return replace(
        request,
        predicates=request.predicates
        + (
            GraspPredicate(object_geoms=_KNOB_GEOMS, effector_geoms=_RIGHT_EFFECTORS, start_s=1.8, end_s=9.2, min_distinct_effectors=1, name="measured_articulated_knob_contact"),
            ReleasePredicate(object_geoms=_KNOB_GEOMS, effector_geoms=_RIGHT_EFFECTORS, start_s=10.0, settle_duration_s=1.0, name="measured_knob_release"),
        ),
    )


def run_authoritative_container_lid_once(case: ProductionContainerLidCase) -> tuple[SimulationResult, ContainerLidDiagnostics]:
    request = enriched_request(case)
    result = NativeMujocoRuntime().simulate(request.simulation)
    model = case.executor.scene_model
    hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj__container__lid_hinge")
    hinge_address = int(model.jnt_qposadr[hinge_id])
    selected = [
        (frame.time_s, contact)
        for frame in result.trace.contacts
        for contact in frame.contacts
        if bool(_KNOB_GEOMS & {contact.geom1_name, contact.geom2_name})
        and bool(_RIGHT_EFFECTORS & {contact.geom1_name, contact.geom2_name})
    ]
    base_selected = [
        contact
        for frame in result.trace.contacts
        for contact in frame.contacts
        if "left_palm_collision" in (contact.geom1_name, contact.geom2_name)
        and any(name.startswith("obj__container__base__") for name in (contact.geom1_name, contact.geom2_name))
    ]
    actuator_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or "" for index in range(model.nu))
    target_hinge = np.asarray(request.simulation.trajectory.qpos)[:, hinge_address]
    hinge = np.asarray(result.trace.qpos)[:, hinge_address]
    return result, ContainerLidDiagnostics(
        production_compile_succeeded=True,
        terminal_lid_angle_deg=float(np.rad2deg(hinge[-1])),
        maximum_lid_angle_deg=float(np.rad2deg(np.max(hinge))),
        knob_contact_samples=len(selected),
        knob_contact_effectors=tuple(sorted({name for _, contact in selected for name in (contact.geom1_name, contact.geom2_name) if name in _RIGHT_EFFECTORS})),
        first_knob_contact_s=None if not selected else float(selected[0][0]),
        last_knob_contact_s=None if not selected else float(selected[-1][0]),
        maximum_knob_force_n=max((contact.normal_force_n for _, contact in selected), default=0.0),
        base_contact_samples=len(base_selected),
        maximum_penetration_m=max((max(0.0, -contact.distance_m) for frame in result.trace.contacts for contact in frame.contacts), default=0.0),
        object_actuator_count=sum("container" in name for name in actuator_names),
        object_target_excursion_rad=float(np.ptp(target_hinge)),
        qpos_writes_after_initialization=int(result.diagnostics["qpos_writes_after_initialization"]),
    )
