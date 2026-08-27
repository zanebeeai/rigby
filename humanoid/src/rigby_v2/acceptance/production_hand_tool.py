"""Production-path physical hand-tool acceptance and measured diagnostics."""

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
_APPROACH = np.asarray((0.63, -0.163, 1.36), dtype=np.float64)
_GRASP = np.asarray((0.61, -0.293, 1.265), dtype=np.float64)
_APPROACH_ROTATION = (0.9999999984, 0.0000017318, 0.0000501183, -0.0000275376)
_GRASP_ROTATION = (0.9999999984, 0.0000016761, 0.0000501260, -0.0000275985)


@dataclass(frozen=True, slots=True)
class ProductionHandToolCase:
    executor: MujocoCertifiedExecutor
    proposal: CandidateProposalV1
    authored_attempt: int


@dataclass(frozen=True, slots=True)
class HandToolDiagnostics:
    max_penetration_m: float
    max_simultaneous_digit_contacts: int
    contacting_digits: tuple[str, ...]
    first_digit_contact_s: float | None
    last_digit_contact_s: float | None
    grasp_coverage: float
    object_rise_m: float
    max_head_support_force_n: float
    strike_while_opposed_grasp_n: float
    final_hand_contact: bool
    final_support_contact: bool


def _palm_frame(
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


def build_production_hand_tool_program(
    *,
    rig_id: str,
    scene_id: str,
    authored_attempt: int = 1,
) -> MotionProgramV2:
    if authored_attempt not in (1, 2):
        raise ValueError("hand-tool acceptance is bounded to two authored attempts")
    # Attempt two uses the shorter task-space route that remained feasible in
    # the production nonlinear refinement after attempt one's compile rejection.
    lift_height = 0.15
    strike_height = -0.015 if authored_attempt == 1 else 0.0
    grasp = _GRASP.copy()
    if authored_attempt == 1:
        times = (0.0, 2.0, 4.0, 9.0, 11.0, 13.0, 14.0, 15.0, 17.0, 19.0, 21.0)
        offsets = (
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.asarray((0.0, 0.0, lift_height)),
            np.asarray((0.0, 0.0, lift_height)),
            np.asarray((0.0, 0.0, 0.08)),
            np.asarray((0.0, 0.0, strike_height)),
            np.asarray((0.0, 0.0, 0.04)),
            np.zeros(3),
            np.zeros(3),
        )
        open_indices = (0, 1, 9, 10)
        retreat_indices = (0, 1, len(times) - 1)
        contact_end = 17.0
        phases = (
            MotionPhaseV2(
                phase_id="approach", kind=PhaseKind.SETUP, start_s=0.0, end_s=4.0
            ),
            MotionPhaseV2(
                phase_id="grasp", kind=PhaseKind.ACTION, start_s=4.0, end_s=9.0
            ),
            MotionPhaseV2(
                phase_id="lift-strike",
                kind=PhaseKind.ACTION,
                start_s=9.0,
                end_s=14.0,
                energy=0.65,
            ),
            MotionPhaseV2(
                phase_id="rebound-hold",
                kind=PhaseKind.HOLD,
                start_s=14.0,
                end_s=17.0,
            ),
            MotionPhaseV2(
                phase_id="release", kind=PhaseKind.RELEASE, start_s=17.0, end_s=21.0
            ),
        )
    else:
        times = (0.0, 2.0, 4.0, 10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0)
        offsets = (
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.asarray((0.0, 0.0, lift_height)),
            np.asarray((0.0, 0.0, lift_height)),
            np.asarray((-0.12, 0.0, lift_height)),
            np.asarray((-0.12, 0.0, strike_height)),
            np.asarray((-0.12, 0.0, strike_height)),
            np.zeros(3),
        )
        open_indices = (0, 1, 8, 9)
        retreat_indices = (0, len(times) - 1)
        contact_end = 16.0
        phases = (
            MotionPhaseV2(
                phase_id="approach", kind=PhaseKind.SETUP, start_s=0.0, end_s=2.0
            ),
            MotionPhaseV2(
                phase_id="grasp-lift-strike",
                kind=PhaseKind.ACTION,
                start_s=2.0,
                end_s=10.0,
                energy=0.65,
            ),
            MotionPhaseV2(
                phase_id="lift-carry-strike",
                kind=PhaseKind.HOLD,
                start_s=10.0,
                end_s=15.0,
            ),
            MotionPhaseV2(
                phase_id="release", kind=PhaseKind.RELEASE, start_s=15.0, end_s=18.0
            ),
        )
    palm_frames = tuple(
        _palm_frame(
            time_s,
            _APPROACH if index in retreat_indices else grasp + offset,
            _APPROACH_ROTATION
            if index in retreat_indices
            else _GRASP_ROTATION,
        )
        for index, (time_s, offset) in enumerate(zip(times, offsets, strict=True))
    )
    opened = _finger_values(0.0)
    closure = 0.72 if authored_attempt == 1 else 0.50
    closed = _finger_values(closure)
    finger_frames = tuple(
        MotionKeyframeV2(
            time_s=time_s,
            joint_values=opened if index in open_indices else closed,
            hard=True,
        )
        for index, time_s in enumerate(times)
    )
    return MotionProgramV2(
        program_id=f"production-hand-tool-attempt-{authored_attempt}",
        source_text=(
            "Approach a free hammer from clear space, close opposing articulated "
            "digits around its handle, lift and hold, strike the support with the "
            "head, rebound, lower, release, and retreat."
        ),
        duration_s=times[-1],
        rig_id=rig_id,
        scene_id=scene_id,
        seed=20260811 + authored_attempt,
        phases=phases,
        tracks=(
            MotionTrackV2(
                track_id="left-palm-tool-route",
                target="left_palm",
                owner="left_arm",
                interpolation=InterpolationKind.SLERP,
                keyframes=palm_frames,
            ),
            MotionTrackV2(
                track_id="left-articulated-tool-grasp",
                target="left_hand_joints",
                owner="left_hand",
                interpolation=InterpolationKind.QUINTIC,
                keyframes=finger_frames,
            ),
        ),
        contacts=(
            ContactEdgeV2(
                contact_id="thumb-handle",
                body_a="left_thumb_distal",
                body_b="obj__hammer__body__handle",
                start_s=4.0,
                end_s=contact_end,
                max_slip_m=0.02,
            ),
            ContactEdgeV2(
                contact_id="index-handle",
                body_a="left_index_distal",
                body_b="obj__hammer__body__handle",
                start_s=4.0,
                end_s=contact_end,
                max_slip_m=0.02,
            ),
        ),
        metadata={"task_space_sample_hz": 1, "authored_attempt": authored_attempt},
    )


def build_production_hand_tool_case(
    artifacts: ContentAddressedArtifactStore,
    *,
    authored_attempt: int = 1,
) -> ProductionHandToolCase:
    rig = stage_canonical_rig("medium", artifacts=artifacts)
    scene = stage_object_pack_scene(
        "hand_tool",
        profile="medium",
        rig_reference=rig.reference,
        artifacts=artifacts,
    )
    program = build_production_hand_tool_program(
        rig_id=rig.manifest.rig_id,
        scene_id=scene.manifest.scene_id,
        authored_attempt=authored_attempt,
    )
    proposal = CandidateProposalV1(
        candidate_id=f"production-hand-tool-attempt-{authored_attempt}",
        semantic_plan_hash=str(authored_attempt) * 64,
        variation=CandidateVariationV1(
            retrieval_seed=program.seed,
            path_family=PathFamily.HAND_LED,
            timing_style="neutral",
            energy=0.55,
            body_participation=("left_arm", "left_hand"),
            contact_strategy="opposed_thumb_and_fingers",
        ),
        program=program,
        compile_attempt=authored_attempt,
    )
    return ProductionHandToolCase(
        executor=MujocoCertifiedExecutor(rig=rig, scene=scene, artifacts=artifacts),
        proposal=proposal,
        authored_attempt=authored_attempt,
    )


def run_authoritative_hand_tool_diagnostic(
    case: ProductionHandToolCase,
) -> tuple[SimulationResult, HandToolDiagnostics]:
    request = case.executor._prepare_certification_request(case.proposal)  # noqa: SLF001
    result = NativeMujocoRuntime().simulate(request.simulation)
    model = case.executor.scene_model
    object_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "obj__hammer__body"
    )
    data = mujoco.MjData(model)
    positions = []
    for qpos in result.trace.qpos:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[object_body].copy())
    xyz = np.asarray(positions)
    digits = {
        f"left_{finger}_{segment}_collision"
        for finger in ("thumb", "index", "middle", "ring", "little")
        for segment in ("proximal", "middle", "distal")
    }
    touching_frames = 0
    maximum_digits = 0
    contacting: set[str] = set()
    first: float | None = None
    last: float | None = None
    maximum_strike = 0.0
    grasped_strike = 0.0
    final_hand = False
    final_support = False
    penetration = 0.0
    for frame in result.trace.contacts:
        frame_digits: set[str] = set()
        frame_strike = 0.0
        for contact in frame.contacts:
            names = {contact.geom1_name, contact.geom2_name}
            penetration = max(penetration, max(0.0, -contact.distance_m))
            hit_digits = names & digits
            if "obj__hammer__body__handle" in names and hit_digits and contact.normal_force_n >= 0.1:
                frame_digits.update(hit_digits)
            if {
                "obj__hammer__body__head",
                "support__tool_bench",
            } == names:
                frame_strike = max(frame_strike, contact.normal_force_n)
        opposed = any("thumb" in item for item in frame_digits) and any(
            "thumb" not in item for item in frame_digits
        )
        if frame_digits:
            first = frame.time_s if first is None else first
            last = frame.time_s
            touching_frames += 1
            contacting.update(frame_digits)
            maximum_digits = max(maximum_digits, len(frame_digits))
        maximum_strike = max(maximum_strike, frame_strike)
        if opposed:
            grasped_strike = max(grasped_strike, frame_strike)
        if frame.time_s >= 20.0:
            final_hand = final_hand or bool(frame_digits)
            final_support = final_support or frame_strike > 0.05
    return result, HandToolDiagnostics(
        max_penetration_m=penetration,
        max_simultaneous_digit_contacts=maximum_digits,
        contacting_digits=tuple(sorted(contacting)),
        first_digit_contact_s=first,
        last_digit_contact_s=last,
        grasp_coverage=touching_frames / len(result.trace.contacts),
        object_rise_m=float(np.max(xyz[:, 2] - xyz[0, 2])),
        max_head_support_force_n=maximum_strike,
        strike_while_opposed_grasp_n=grasped_strike,
        final_hand_contact=final_hand,
        final_support_contact=final_support,
    )
