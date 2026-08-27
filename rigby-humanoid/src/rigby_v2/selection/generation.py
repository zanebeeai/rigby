from __future__ import annotations

import math
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from rigby_v2.contracts import (
    ContactEdgeV2,
    InterpolationKind,
    MotionKeyframeV2,
    MotionPhaseV2,
    MotionProgramV2,
    MotionTrackV2,
    Quaternion,
    QuaternionOrder,
    Vec3,
)
from rigby_v2.errors import FailureCode, RigbyV2Error
from rigby_v2.flywheel.schemas import (
    CandidateProposalV1,
    CandidateSetV1,
    CandidateVariationV1,
    PathFamily,
    SemanticPlanV1,
)
from rigby_v2.hashing import content_hash
from rigby_v2.motion.rotations import slerp, squad


_TIME_EPSILON_S = 1e-8
_MAX_TASK_ROUTE_OFFSET_M = 0.018
_MAX_TASK_ROTATION_OFFSET_RAD = math.radians(3.0)


def _signed_code(value: str) -> float:
    """Stable, bounded code used only to place a variation on a route axis."""

    weighted = sum((index + 1) * ord(character) for index, character in enumerate(value))
    return ((weighted % 41) - 20) / 20.0


def _position_array(value: Vec3) -> np.ndarray:
    return np.asarray((value.x, value.y, value.z), dtype=np.float64)


def _position_contract(value: np.ndarray) -> Vec3:
    return Vec3(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def _wxyz(value: Quaternion) -> np.ndarray:
    components = np.asarray(value.values, dtype=np.float64)
    if value.convention.order is QuaternionOrder.XYZW:
        return components[[3, 0, 1, 2]]
    return components


def _quaternion_contract(value: np.ndarray, template: Quaternion) -> Quaternion:
    normalized = np.asarray(value, dtype=np.float64)
    normalized /= float(np.linalg.norm(normalized))
    if template.convention.order is QuaternionOrder.XYZW:
        normalized = normalized[[1, 2, 3, 0]]
    return Quaternion(
        values=tuple(float(component) for component in normalized),
        convention=template.convention,
    )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _rotate_quaternion(
    value: Quaternion,
    *,
    axis: np.ndarray,
    angle_rad: float,
) -> Quaternion:
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12 or abs(angle_rad) <= 1e-12:
        return value
    axis /= axis_norm
    half_angle = 0.5 * float(
        np.clip(angle_rad, -_MAX_TASK_ROTATION_OFFSET_RAD, _MAX_TASK_ROTATION_OFFSET_RAD)
    )
    delta = np.asarray((math.cos(half_angle), *(axis * math.sin(half_angle))))
    return _quaternion_contract(_quaternion_multiply(delta, _wxyz(value)), value)


def _track_segment(track: MotionTrackV2, time_s: float) -> tuple[int, int, float]:
    times = np.asarray([keyframe.time_s for keyframe in track.keyframes], dtype=np.float64)
    if len(times) == 1:
        return 0, 0, 0.0
    right = int(np.searchsorted(times, time_s, side="right"))
    right = min(max(right, 1), len(times) - 1)
    left = right - 1
    progress = float(
        np.clip((time_s - times[left]) / (times[right] - times[left]), 0.0, 1.0)
    )
    return left, right, progress


def _sample_task_track(
    track: MotionTrackV2, time_s: float
) -> tuple[Vec3 | None, Quaternion | None]:
    """Sample the unvaried authored route for immutable boundary anchors."""

    left, right, progress = _track_segment(track, time_s)
    left_keyframe = track.keyframes[left]
    right_keyframe = track.keyframes[right]
    position: Vec3 | None = None
    if left_keyframe.position is not None and right_keyframe.position is not None:
        start = _position_array(left_keyframe.position)
        end = _position_array(right_keyframe.position)
        # Same zero-velocity/acceleration quintic used by the motion compiler.
        blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
        position = _position_contract(start + blend * (end - start))
    rotation: Quaternion | None = None
    if left_keyframe.rotation is not None and right_keyframe.rotation is not None:
        if left == right:
            rotation = left_keyframe.rotation
        elif track.interpolation is InterpolationKind.SQUAD:
            previous = track.keyframes[max(0, left - 1)].rotation or left_keyframe.rotation
            following = track.keyframes[min(len(track.keyframes) - 1, right + 1)].rotation or right_keyframe.rotation
            rotation = squad(
                previous,
                left_keyframe.rotation,
                right_keyframe.rotation,
                following,
                progress,
            )
        else:
            rotation = slerp(left_keyframe.rotation, right_keyframe.rotation, progress)
    return position, rotation


def _route_frame(start: Vec3 | None, end: Vec3 | None, track_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if start is None or end is None:
        direction = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        distance = 0.0
    else:
        displacement = _position_array(end) - _position_array(start)
        distance = float(np.linalg.norm(displacement))
        if distance <= 1e-10:
            direction = np.eye(3, dtype=np.float64)[track_index % 3]
        else:
            direction = displacement / distance
    reference = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    if abs(float(np.dot(reference, direction))) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    lateral = np.cross(reference, direction)
    lateral /= float(np.linalg.norm(lateral))
    normal = np.cross(direction, lateral)
    normal /= float(np.linalg.norm(normal))
    return direction, lateral, normal, distance


def _task_route_offset(
    variation: CandidateVariationV1,
    *,
    start: Vec3 | None,
    end: Vec3 | None,
    track_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    direction, lateral, normal, distance = _route_frame(start, end, track_index)
    path_coefficients = {
        PathFamily.DIRECT: (0.00, 0.00, 0.00),
        PathFamily.ARC: (0.00, 1.00, 0.35),
        PathFamily.BODY_LED: (-0.25, -0.45, -0.35),
        PathFamily.HAND_LED: (0.10, 0.30, 0.90),
        PathFamily.CONTACT_FIRST: (0.45, -0.70, 0.20),
    }[variation.path_family]
    body_code = _signed_code("|".join(variation.body_participation))
    contact_code = _signed_code(variation.contact_strategy)
    seed_code = ((variation.retrieval_seed % 43) - 21) / 21.0
    route_scale = min(_MAX_TASK_ROUTE_OFFSET_M, max(0.006, 0.16 * distance))
    forward = path_coefficients[0] + 0.10 * contact_code + 0.035 * seed_code
    sideways = path_coefficients[1] + 0.22 * body_code + 0.06 * seed_code
    vertical = path_coefficients[2] + 0.16 * contact_code - 0.04 * seed_code
    energy_gain = 0.85 + 0.30 * variation.energy
    offset = energy_gain * route_scale * (
        forward * direction + sideways * lateral + vertical * normal
    )
    offset_norm = float(np.linalg.norm(offset))
    if offset_norm > _MAX_TASK_ROUTE_OFFSET_M:
        offset *= _MAX_TASK_ROUTE_OFFSET_M / offset_norm
    rotation_axis = direction + 0.35 * body_code * lateral + 0.25 * contact_code * normal
    path_rotation = {
        PathFamily.DIRECT: 0.00,
        PathFamily.ARC: 0.60,
        PathFamily.BODY_LED: -0.45,
        PathFamily.HAND_LED: 0.35,
        PathFamily.CONTACT_FIRST: -0.75,
    }[variation.path_family]
    rotation_angle = energy_gain * math.radians(2.2) * (
        path_rotation + 0.25 * body_code + 0.18 * contact_code + 0.07 * seed_code
    )
    return offset, rotation_axis, float(
        np.clip(
            rotation_angle,
            -_MAX_TASK_ROTATION_OFFSET_RAD,
            _MAX_TASK_ROTATION_OFFSET_RAD,
        )
    )


def _task_midpoint_time(
    start_s: float,
    end_s: float,
    variation: CandidateVariationV1,
) -> float:
    timing_fraction = {
        "energetic": 0.38,
        "neutral": 0.50,
        "relaxed": 0.62,
    }[variation.timing_style]
    contact_code = _signed_code(variation.contact_strategy)
    fraction = float(np.clip(timing_fraction + 0.035 * contact_code, 0.25, 0.75))
    return start_s + fraction * (end_s - start_s)


def _vary_task_track(
    track: MotionTrackV2,
    contacts: tuple[ContactEdgeV2, ...],
    variation: CandidateVariationV1,
    track_index: int,
) -> MotionTrackV2:
    """Add bounded route points while retaining every authored/semantic anchor."""

    if len(track.keyframes) < 2 or any(keyframe.joint_values for keyframe in track.keyframes):
        return track
    positions_complete = all(keyframe.position is not None for keyframe in track.keyframes)
    rotations_complete = all(keyframe.rotation is not None for keyframe in track.keyframes)
    if not positions_complete and not rotations_complete:
        return track
    start_s = track.keyframes[0].time_s
    end_s = track.keyframes[-1].time_s
    anchors: dict[float, MotionKeyframeV2] = {
        keyframe.time_s: keyframe for keyframe in track.keyframes
    }
    # Contact creation/break times are immutable route anchors.  Sampling the
    # original track here prevents a varied curve from moving their target pose.
    for contact in contacts:
        for boundary_s in (contact.start_s, contact.end_s):
            if not start_s + _TIME_EPSILON_S < boundary_s < end_s - _TIME_EPSILON_S:
                continue
            if any(abs(existing - boundary_s) <= _TIME_EPSILON_S for existing in anchors):
                continue
            position, rotation = _sample_task_track(track, boundary_s)
            anchors[boundary_s] = MotionKeyframeV2(
                time_s=boundary_s,
                position=position,
                rotation=rotation,
            )
    ordered_anchors = [anchors[time_s] for time_s in sorted(anchors)]
    varied: list[MotionKeyframeV2] = []
    for anchor_index, (left, right) in enumerate(zip(ordered_anchors, ordered_anchors[1:])):
        varied.append(left)
        if right.time_s - left.time_s <= 2e-6:
            continue
        midpoint_s = _task_midpoint_time(left.time_s, right.time_s, variation)
        position, rotation = _sample_task_track(track, midpoint_s)
        offset, rotation_axis, rotation_angle = _task_route_offset(
            variation,
            start=left.position,
            end=right.position,
            track_index=track_index + anchor_index,
        )
        if position is not None:
            position = _position_contract(_position_array(position) + offset)
        if rotation is not None:
            rotation = _rotate_quaternion(
                rotation,
                axis=rotation_axis,
                angle_rad=rotation_angle,
            )
        varied.append(
            MotionKeyframeV2(
                time_s=midpoint_s,
                position=position,
                rotation=rotation,
            )
        )
    varied.append(ordered_anchors[-1])
    return track.model_copy(update={"keyframes": tuple(varied)})


class CandidateGenerationReason(StrEnum):
    COMPILE_FAILED = "compile_failed"
    INSUFFICIENT_DIVERSE_CANDIDATES = "insufficient_diverse_candidates"
    REPAIR_FAILED = "repair_failed"


class CandidateGenerationError(RigbyV2Error):
    def __init__(
        self,
        reason: CandidateGenerationReason,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            FailureCode.ALL_CANDIDATES_REJECTED,
            message,
            details={"generation_reason": reason.value, **(details or {})},
        )
        self.reason = reason


@runtime_checkable
class CandidateCompiler(Protocol):
    def compile(
        self,
        plan: SemanticPlanV1,
        variation: CandidateVariationV1,
        attempt: int,
    ) -> MotionProgramV2: ...

    def repair(
        self,
        plan: SemanticPlanV1,
        variation: CandidateVariationV1,
        attempt: int,
        failure: Exception,
    ) -> MotionProgramV2: ...


class DeterministicCandidateCompiler:
    """Learned-model-free default that specializes one semantic program."""

    @staticmethod
    def _program(
        plan: SemanticPlanV1, variation: CandidateVariationV1, *, repaired: bool
    ) -> MotionProgramV2:
        base = plan.base_program
        phases = tuple(
            MotionPhaseV2.model_validate(
                {
                    **phase.model_dump(mode="python"),
                    "energy": min(1.0, max(0.0, 0.45 * phase.energy + 0.55 * variation.energy)),
                }
            )
            for phase in base.phases
        )
        metadata = dict(base.metadata)
        metadata["candidate_variation"] = variation.model_dump(mode="json")
        metadata["bounded_repair"] = repaired
        path_offset = {
            PathFamily.DIRECT: 0.0,
            PathFamily.ARC: 0.7,
            PathFamily.BODY_LED: -0.5,
            PathFamily.HAND_LED: 0.35,
            PathFamily.CONTACT_FIRST: -0.8,
        }[variation.path_family]
        timing_fraction = {
            "energetic": 0.38,
            "neutral": 0.50,
            "relaxed": 0.62,
        }[variation.timing_style]
        contact_bias = _signed_code(variation.contact_strategy)
        body_bias = _signed_code("|".join(variation.body_participation))
        seed_bias = ((variation.retrieval_seed % 29) - 14) / 100.0
        tracks: list[MotionTrackV2] = []
        for track_index, track in enumerate(base.tracks):
            if (
                len(track.keyframes) >= 2
                and not any(keyframe.joint_values for keyframe in track.keyframes)
            ):
                tracks.append(
                    _vary_task_track(track, base.contacts, variation, track_index)
                )
                continue
            if len(track.keyframes) < 2 or not all(
                keyframe.joint_values for keyframe in track.keyframes
            ):
                tracks.append(track)
                continue
            first, last = track.keyframes[0], track.keyframes[-1]
            shared = sorted(set(first.joint_values) & set(last.joint_values))
            if not shared or last.time_s - first.time_s <= 1e-9:
                tracks.append(track)
                continue
            midpoint_time = first.time_s + timing_fraction * (
                last.time_s - first.time_s
            )
            # Contact strategy changes when the trajectory commits while path,
            # body participation, and retrieval seed change its spatial route.
            midpoint_time += contact_bias * 0.04 * (last.time_s - first.time_s)
            midpoint_time = min(last.time_s - 1e-6, max(first.time_s + 1e-6, midpoint_time))
            midpoint_values: dict[str, float] = {}
            route_bias = 0.02 * (
                path_offset + body_bias + contact_bias + seed_bias
            )
            for joint_index, name in enumerate(shared):
                alpha = (midpoint_time - first.time_s) / (last.time_s - first.time_s)
                baseline = (1.0 - alpha) * first.joint_values[name] + alpha * last.joint_values[name]
                direction = -1.0 if (track_index + joint_index) % 2 else 1.0
                midpoint_values[name] = baseline + direction * route_bias
            midpoint = MotionKeyframeV2(
                time_s=midpoint_time,
                joint_values=midpoint_values,
            )
            retained = tuple(
                keyframe
                for keyframe in track.keyframes[1:-1]
                if abs(keyframe.time_s - midpoint_time) > 1e-6
            )
            tracks.append(
                track.model_copy(
                    update={
                        "keyframes": tuple(
                            sorted(
                                (first, *retained, midpoint, last),
                                key=lambda keyframe: keyframe.time_s,
                            )
                        )
                    }
                )
            )
        return MotionProgramV2.model_validate(
            {
                **base.model_dump(mode="python"),
                "program_id": f"{base.program_id}:{variation.path_family.value}:{variation.retrieval_seed}",
                "seed": variation.retrieval_seed,
                "phases": phases,
                "tracks": tuple(tracks),
                "metadata": metadata,
            }
        )

    def compile(
        self,
        plan: SemanticPlanV1,
        variation: CandidateVariationV1,
        attempt: int,
    ) -> MotionProgramV2:
        del attempt
        return self._program(plan, variation, repaired=False)

    def repair(
        self,
        plan: SemanticPlanV1,
        variation: CandidateVariationV1,
        attempt: int,
        failure: Exception,
    ) -> MotionProgramV2:
        del attempt, failure
        return self._program(plan, variation, repaired=True)


_BODY_PARTICIPATION = (
    ("arms", "hands"),
    ("torso", "arms", "hands"),
    ("pelvis", "torso", "arms", "hands"),
    ("gaze", "arms", "hands"),
    ("whole_body", "hands"),
)
_CONTACT_STRATEGIES = (
    "direct_then_close",
    "preshape_then_contact",
    "contact_first_stabilize",
    "bimanual_support",
    "task_normal_aligned",
)
_TIMING_STYLES = ("neutral", "relaxed", "energetic", "neutral", "energetic")


def variation_for_attempt(plan: SemanticPlanV1, attempt: int) -> CandidateVariationV1:
    if not 1 <= attempt <= 20:
        raise ValueError("candidate attempt must lie in [1, 20]")
    index = attempt - 1
    families = tuple(PathFamily)
    cycle, family_index = divmod(index, len(families))
    energy = 0.25 + 0.12 * family_index + 0.025 * cycle
    return CandidateVariationV1(
        retrieval_seed=plan.base_program.seed + 104729 * attempt,
        path_family=families[family_index],
        timing_style=_TIMING_STYLES[(family_index + cycle) % len(_TIMING_STYLES)],
        energy=min(0.95, energy),
        body_participation=_BODY_PARTICIPATION[(family_index + 2 * cycle) % 5],
        contact_strategy=_CONTACT_STRATEGIES[(family_index + 3 * cycle) % 5],
    )


def structural_fingerprint(program: MotionProgramV2) -> str:
    """Ignore IDs/text/metadata while retaining actual authored structure."""

    return content_hash(
        {
            "duration_s": program.duration_s,
            "rig_id": program.rig_id,
            "scene_id": program.scene_id,
            "phases": [phase.model_dump(mode="json") for phase in program.phases],
            "tracks": [track.model_dump(mode="json") for track in program.tracks],
            "contacts": [contact.model_dump(mode="json") for contact in program.contacts],
            "assertions": [assertion.model_dump(mode="json") for assertion in program.assertions],
        }
    )


def generate_candidate_set(
    plan: SemanticPlanV1,
    compiler: CandidateCompiler | None = None,
    *,
    max_attempts: int = 20,
) -> CandidateSetV1:
    if not 5 <= max_attempts <= 20:
        raise ValueError("max_attempts must lie in [5, 20]")
    compiler = compiler or DeterministicCandidateCompiler()
    proposals: list[CandidateProposalV1] = []
    fingerprints: set[str] = set()
    failures: list[dict[str, object]] = []
    attempts = 0
    repair_rounds = 0
    pending_repair: tuple[CandidateVariationV1, Exception] | None = None

    while attempts < max_attempts and len(proposals) < 5:
        attempts += 1
        repaired = pending_repair is not None
        if repaired:
            variation, failure = pending_repair
            pending_repair = None
            try:
                program = compiler.repair(plan, variation, attempts, failure)
            except Exception as error:
                failures.append({"attempt": attempts, "error": str(error), "repaired": True})
                continue
        else:
            variation = variation_for_attempt(plan, attempts)
            try:
                program = compiler.compile(plan, variation, attempts)
            except Exception as error:
                failures.append({"attempt": attempts, "error": str(error), "repaired": False})
                if repair_rounds == 0 and attempts < max_attempts:
                    repair_rounds = 1
                    pending_repair = (variation, error)
                continue
        if program.rig_id != plan.base_program.rig_id or program.scene_id != plan.base_program.scene_id:
            failures.append(
                {"attempt": attempts, "error": "compiler changed rig or scene", "repaired": repaired}
            )
            continue
        fingerprint = structural_fingerprint(program)
        if fingerprint in fingerprints:
            failures.append(
                {"attempt": attempts, "error": "structural_duplicate", "repaired": repaired}
            )
            continue
        fingerprints.add(fingerprint)
        proposal_hash = content_hash(
            {
                "plan": plan.content_hash(),
                "variation": variation.content_hash(),
                "structure": fingerprint,
            }
        )
        proposals.append(
            CandidateProposalV1(
                candidate_id=f"candidate-{proposal_hash[:20]}",
                semantic_plan_hash=plan.content_hash(),
                variation=variation,
                program=program,
                compile_attempt=attempts,
                repaired=repaired,
            )
        )
    if len(proposals) != 5:
        raise CandidateGenerationError(
            CandidateGenerationReason.INSUFFICIENT_DIVERSE_CANDIDATES,
            f"Only {len(proposals)} structurally distinct candidates compiled in {attempts} attempts",
            details={"attempts": attempts, "failures": failures},
        )
    return CandidateSetV1(
        semantic_plan=plan,
        candidates=tuple(proposals),
        total_compile_attempts=attempts,
        repair_rounds=repair_rounds,
    )
