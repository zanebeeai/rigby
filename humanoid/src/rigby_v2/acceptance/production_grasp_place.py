"""Production-path grasp/place program and authoritative trace diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from rigby_core.artifacts import ContentAddressedArtifactStore
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
_APPROACH_PALM = np.asarray((0.535493, -0.232108, 1.38))
_HOVER_PALM = np.asarray((0.535493, -0.232108, 1.38))
# The block's declared grasp affordance is rotated so its 52 mm narrow axis is
# collinear with the measured thumb-to-middle-finger vector.  The reachable
# identity palm pose then puts both distal capsules 1 mm into opposite faces.
_PREGRASP_PALM = np.asarray((0.535493, -0.232108, 1.218))
_APPROACH_ROTATION = (0.984807753, 0.0, 0.173648178, 0.0)
_PREGRASP_ROTATION = (0.984807753, 0.0, 0.173648178, 0.0)
_TIMES = (0.0, 2.0, 4.0, 6.0, 12.0, 24.0, 28.0, 32.0, 33.0, 37.0, 41.0, 48.0)
_LEFT_ARM_NAMES = (
    "left_shoulder_flex",
    "left_shoulder_abduct",
    "left_shoulder_twist",
    "left_elbow_flex",
    "left_forearm_twist",
    "left_wrist_flex",
    "left_wrist_deviation",
)
_ARM_APPROACH = np.asarray(
    (-0.6484347830, -0.3796896849, -0.4482929258, 1.8097088998,
     -0.2367108297, -0.6758072734, 0.4024239000)
)
_ARM_HOVER = np.asarray(
    (-0.6484347830, -0.3796896849, -0.4482929258, 1.8097088998,
     -0.2367108297, -0.6758072734, 0.4024239000)
)
_ARM_PREGRASP = np.asarray(
    (-0.1484722503, -0.1534537402, -0.5330779584, 1.5750848161,
     -0.2268300103, -0.9714775384, 0.3509993074)
)
_ARM_LIFT = np.asarray(
    (-0.6125193443, -0.3650239216, -0.4595658710, 1.8051861660,
     -0.2400076613, -0.7072884682, 0.3948198026)
)
_ARM_CARRY = np.asarray(
    (-0.6125193443, -0.3650239216, -0.4595658710, 1.8051861660,
     -0.2400076613, -0.7072884682, 0.3948198026)
)
_ARM_LOWER = np.asarray(
    (-0.1484722503, -0.1534537402, -0.5330779584, 1.5750848161,
     -0.2268300103, -0.9714775384, 0.3509993074)
)
_ARM_RETREAT = np.asarray(
    (-0.6493810375, -0.3778656870, -0.4500087845, 1.8097089047,
     -0.2402484801, -0.6752607465, 0.3995430070)
)


@dataclass(frozen=True, slots=True)
class ProductionGraspPlaceCase:
    executor: MujocoCertifiedExecutor
    proposal: CandidateProposalV1


@dataclass(frozen=True, slots=True)
class GraspPlaceDiagnostics:
    production_compile_succeeded: bool
    max_penetration_m: float
    max_simultaneous_digit_contacts: int
    contacting_digits: tuple[str, ...]
    first_digit_contact_s: float | None
    last_digit_contact_s: float | None
    max_digit_force_n: float
    object_rise_m: float
    object_carry_m: float
    final_object_position_m: tuple[float, float, float]


def _palm_keyframe(
    time_s: float,
    position: np.ndarray,
    rotation: tuple[float, float, float, float],
) -> MotionKeyframeV2:
    return MotionKeyframeV2(
        time_s=time_s,
        position=Vec3(x=float(position[0]), y=float(position[1]), z=float(position[2])),
        rotation=Quaternion(values=rotation, convention=_WORLD_WXYZ),
        hard=True,
    )


def _position_keyframe(time_s: float, position: np.ndarray) -> MotionKeyframeV2:
    return MotionKeyframeV2(
        time_s=time_s,
        position=Vec3(
            x=float(position[0]), y=float(position[1]), z=float(position[2])
        ),
        hard=True,
    )


def _finger_values(scale: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for finger in ("index", "middle", "ring", "little"):
        for segment, degrees in (("mcp", 70), ("pip", 85), ("dip", 55)):
            values[f"left_{finger}_{segment}"] = float(
                np.deg2rad(degrees) * scale
            )
    for name, degrees in (
        ("thumb_opposition", 35),
        ("thumb_mcp", 50),
        ("thumb_pip", 55),
        ("thumb_dip", 40),
    ):
        values[f"left_{name}"] = float(np.deg2rad(degrees) * scale)
    return values


def build_production_grasp_place_program(
    *,
    rig_id: str,
    scene_id: str,
) -> MotionProgramV2:
    offsets = (
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.asarray((0.0, 0.0, 0.15)),
        np.asarray((0.0, 0.0, 0.15)),
        np.asarray((0.0, 0.0, 0.15)),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    )
    palm_frames = []
    for index, (time_s, offset) in enumerate(zip(_TIMES, offsets, strict=True)):
        retreat = index in (0, len(_TIMES) - 1)
        hover = index == 1
        palm_frames.append(
            _palm_keyframe(
                time_s,
                (
                    _APPROACH_PALM
                    if retreat
                    else _HOVER_PALM
                    if hover
                    else _PREGRASP_PALM
                )
                + offset,
                _APPROACH_ROTATION if retreat else _PREGRASP_ROTATION,
            )
        )
    opened = _finger_values(0.0)
    closed = _finger_values(0.2)
    finger_frames = tuple(
        MotionKeyframeV2(
            time_s=time_s,
            joint_values=opened if index in (0, 1, 2, 10, 11) else closed,
            hard=True,
        )
        for index, time_s in enumerate(_TIMES)
    )
    arm_poses = (
        _ARM_APPROACH,
        _ARM_HOVER,
        _ARM_PREGRASP,
        _ARM_PREGRASP,
        _ARM_PREGRASP,
        _ARM_PREGRASP,
        _ARM_LIFT,
        _ARM_LIFT,
        _ARM_CARRY,
        _ARM_LOWER,
        _ARM_LOWER,
        _ARM_RETREAT,
    )
    arm_frames = tuple(
        MotionKeyframeV2(
            time_s=time_s,
            joint_values={
                name: float(value)
                for name, value in zip(_LEFT_ARM_NAMES, pose, strict=True)
            },
            hard=True,
        )
        for time_s, pose in zip(_TIMES, arm_poses, strict=True)
    )
    thumb_hover = np.asarray((0.605228, -0.254108, 1.356699))
    thumb_pregrasp = np.asarray((0.605227, -0.254108, 1.194700))
    thumb_hook = np.asarray((0.601431, -0.251200, 1.200000))
    index_hover = np.asarray((0.666917, -0.220108, 1.338920))
    index_pregrasp = np.asarray((0.666916, -0.220108, 1.176921))
    index_hook = np.asarray((0.649451, -0.220108, 1.175184))
    thumb_targets = (
        thumb_hover, thumb_hover, thumb_pregrasp, thumb_hook, thumb_hook,
        thumb_hook, thumb_hook + (0.0, 0.0, 0.15),
        thumb_hook + (0.0, 0.0, 0.15), thumb_hook + (0.0, 0.0, 0.15),
        thumb_hook, thumb_pregrasp, thumb_hover,
    )
    index_targets = (
        index_hover, index_hover, index_pregrasp, index_hook, index_hook,
        index_hook, index_hook + (0.0, 0.0, 0.15),
        index_hook + (0.0, 0.0, 0.15), index_hook + (0.0, 0.0, 0.15),
        index_hook, index_pregrasp, index_hover,
    )
    return MotionProgramV2(
        program_id="production-grasp-place-block",
        source_text=(
            "Collision-aware task-space palm approach, articulated finger "
            "preshape and closure, lift, hold, carry, place, release, and retreat."
        ),
        duration_s=48.0,
        rig_id=rig_id,
        scene_id=scene_id,
        seed=20260811,
        phases=(
            MotionPhaseV2(
                phase_id="approach",
                kind=PhaseKind.SETUP,
                start_s=0.0,
                end_s=2.0,
            ),
            MotionPhaseV2(
                phase_id="grasp",
                kind=PhaseKind.ACTION,
                start_s=2.0,
                end_s=24.0,
            ),
            MotionPhaseV2(
                phase_id="lift",
                kind=PhaseKind.ACTION,
                start_s=24.0,
                end_s=28.0,
            ),
            MotionPhaseV2(
                phase_id="hold",
                kind=PhaseKind.HOLD,
                start_s=28.0,
                end_s=33.0,
            ),
            MotionPhaseV2(
                phase_id="lower",
                kind=PhaseKind.HOLD,
                start_s=33.0,
                end_s=37.0,
            ),
            MotionPhaseV2(
                phase_id="release",
                kind=PhaseKind.RELEASE,
                start_s=37.0,
                end_s=45.0,
            ),
            MotionPhaseV2(
                phase_id="settle",
                kind=PhaseKind.RECOVERY,
                start_s=45.0,
                end_s=48.0,
            ),
        ),
        tracks=(
            MotionTrackV2(
                track_id="left-palm-task-space",
                target="left_palm",
                owner="left_arm",
                interpolation=InterpolationKind.SLERP,
                keyframes=tuple(palm_frames),
            ),
            MotionTrackV2(
                track_id="left-arm-reachable-seed",
                target="left_arm_joints",
                owner="left_arm",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=arm_frames,
            ),
            MotionTrackV2(
                track_id="left-articulated-finger-closure",
                target="left_hand_joints",
                owner="left_hand",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=finger_frames,
            ),
            MotionTrackV2(
                track_id="left-thumb-underhook",
                target="left_thumb_tip",
                owner="left_hand",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=tuple(
                    _position_keyframe(time_s, position)
                    for time_s, position in zip(_TIMES, thumb_targets, strict=True)
                ),
            ),
            MotionTrackV2(
                track_id="left-index-underhook",
                target="left_index_tip",
                owner="left_hand",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=tuple(
                    _position_keyframe(time_s, position)
                    for time_s, position in zip(_TIMES, index_targets, strict=True)
                ),
            ),
        ),
        contacts=(
            ContactEdgeV2(
                contact_id="left-opposed-digits-block",
                body_a="left_hand",
                body_b="obj__block__body",
                start_s=2.6125,
                end_s=44.7208333333,
                max_slip_m=0.01,
                max_penetration_m=0.002,
            ),
        ),
        metadata={"task_space_sample_hz": 1},
    )


def build_production_grasp_place_case(
    artifacts: ContentAddressedArtifactStore,
) -> ProductionGraspPlaceCase:
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "grasp_place_block",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    program = build_production_grasp_place_program(
        rig_id=rig.manifest.rig_id,
        scene_id=scene.manifest.scene_id,
    )
    proposal = CandidateProposalV1(
        candidate_id="production-grasp-place-block",
        semantic_plan_hash="a" * 64,
        variation=CandidateVariationV1(
            retrieval_seed=20260811,
            path_family=PathFamily.HAND_LED,
            timing_style="neutral",
            energy=0.35,
            body_participation=("left_arm", "left_hand"),
            contact_strategy="opposed_thumb_and_fingers",
        ),
        program=program,
        compile_attempt=1,
    )
    return ProductionGraspPlaceCase(
        executor=MujocoCertifiedExecutor(
            rig=rig,
            scene=scene,
            artifacts=artifacts,
        ),
        proposal=proposal,
    )


def run_authoritative_diagnostic(
    case: ProductionGraspPlaceCase,
) -> tuple[SimulationResult, GraspPlaceDiagnostics]:
    request = case.executor._prepare_certification_request(case.proposal)  # noqa: SLF001
    result = NativeMujocoRuntime().simulate(request.simulation)
    model = case.executor.scene_model
    object_geom = "obj__block__body__box"
    digit_geoms = {
        f"left_{finger}_{segment}_collision"
        for finger in ("thumb", "index", "middle", "ring", "little")
        for segment in ("proximal", "middle", "distal")
    }
    contacting: set[str] = set()
    first: float | None = None
    last: float | None = None
    maximum_simultaneous = 0
    maximum_force = 0.0
    for frame in result.trace.contacts:
        frame_digits: set[str] = set()
        for contact in frame.contacts:
            names = {contact.geom1_name, contact.geom2_name}
            hits = names & digit_geoms
            if object_geom in names and hits and contact.normal_force_n >= 0.1:
                frame_digits.update(hits)
                maximum_force = max(maximum_force, contact.normal_force_n)
        if frame_digits:
            first = frame.time_s if first is None else first
            last = frame.time_s
            contacting.update(frame_digits)
            maximum_simultaneous = max(maximum_simultaneous, len(frame_digits))
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "obj__block__body"
    )
    data = mujoco.MjData(model)
    positions = []
    for qpos in result.trace.qpos:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[body_id].copy())
    xyz = np.asarray(positions)
    penetration = max(
        (
            max(0.0, -contact.distance_m)
            for frame in result.trace.contacts
            for contact in frame.contacts
        ),
        default=0.0,
    )
    diagnostics = GraspPlaceDiagnostics(
        production_compile_succeeded=True,
        max_penetration_m=penetration,
        max_simultaneous_digit_contacts=maximum_simultaneous,
        contacting_digits=tuple(sorted(contacting)),
        first_digit_contact_s=first,
        last_digit_contact_s=last,
        max_digit_force_n=maximum_force,
        object_rise_m=float(np.max(xyz[:, 2] - xyz[0, 2])),
        object_carry_m=float(
            np.max(np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1))
        ),
        final_object_position_m=tuple(float(value) for value in xyz[-1]),
    )
    return result, diagnostics
