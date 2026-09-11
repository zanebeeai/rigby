"""Bounded embodied inspection and body-region callback repair.

The final visual judge should never be the first component to notice that a
hand missed an object.  This module inspects the exact posed rig geometry used
by the renderer, assigns a failure to the nearest responsible body region, and
recompiles only bounded parameter changes for that region.  It is deterministic
and makes no model/API calls.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import rig_kinematics
from .motion_states import state_for, states
from .models import ClipFrame, ClipResult, Hand, Intent, MotionProgram, PrimitiveKind, SceneManifest


MASTER_INSPECTION_PROTOCOL = "first_person_embodiment_inspector_v1"
MASTER_CALLBACK_PROTOCOL = "embodied_master_region_callback_v1"
MAX_MASTER_CALLBACKS = 4
PALM_SOCKET_ERROR_LIMIT_M = 0.035
PALM_NORMAL_MINIMUM_ALIGNMENT = 0.78
DIGIT_ORDER_TOLERANCE_M = -0.004
THUMB_CLEARANCE_MINIMUM_M = 0.006


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _phase_frame(clip: ClipResult, kind: PrimitiveKind) -> ClipFrame | None:
    ranges = clip.metrics.get("phase_ranges_s")
    if not isinstance(ranges, list) or not clip.frames:
        return None
    phase = next(
        (
            item
            for item in ranges
            if isinstance(item, dict) and item.get("kind") == kind.value
        ),
        None,
    )
    if not isinstance(phase, dict):
        return None
    endpoint = float(phase.get("end_s", 0.0))
    return min(clip.frames, key=lambda frame: abs(frame.time_s - endpoint))


def _socket_geometry(
    program: MotionProgram,
    scene: SceneManifest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str, str] | None:
    object_id = next(
        (primitive.object_id for primitive in program.primitives if primitive.object_id),
        None,
    )
    socket_id = next(
        (primitive.socket_id for primitive in program.primitives if primitive.socket_id),
        None,
    )
    target = scene.object_by_id(object_id or "")
    if target is None or socket_id is None:
        return None
    socket = next((value for value in target.sockets if value.id == socket_id), None)
    if socket is None:
        return None
    object_rotation = Rotation.from_quat(target.transform.rotation.as_list())
    socket_position = np.asarray(target.transform.translation.as_list(), dtype=float)
    socket_position += object_rotation.apply(socket.transform.translation.as_list())
    approach_normal = object_rotation.apply(socket.approach_normal.as_list())
    approach_normal /= max(float(np.linalg.norm(approach_normal)), 1e-12)
    lateral_axis = object_rotation.apply([1.0, 0.0, 0.0])
    lateral_axis /= max(float(np.linalg.norm(lateral_axis)), 1e-12)
    return (
        socket_position,
        approach_normal,
        lateral_axis,
        float(socket.grasp_span_m),
        target.id,
        socket.id,
    )


def _hand_geometry(
    frame: ClipFrame,
    hand: Hand,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    side = hand.value
    prefix = side
    kinematics = rig_kinematics()
    positions = kinematics.canonical_positions(frame.bones)
    tips = kinematics.fingertip_positions(frame.bones, side)
    palm_points = [
        positions[f"{prefix}Hand"],
        positions[f"{prefix}IndexProximal"],
        positions[f"{prefix}MiddleProximal"],
        positions[f"{prefix}RingProximal"],
        positions[f"{prefix}LittleProximal"],
    ]
    palm_center = np.mean(palm_points, axis=0)
    across = (
        positions[f"{prefix}IndexProximal"]
        - positions[f"{prefix}LittleProximal"]
    )
    along = (
        positions[f"{prefix}MiddleProximal"]
        - positions[f"{prefix}Hand"]
    )
    palm_normal = np.cross(across, along)
    palm_normal /= max(float(np.linalg.norm(palm_normal)), 1e-12)
    return palm_center, palm_normal, positions, tips


def _architecture_check(program: MotionProgram, clip: ClipResult) -> dict[str, Any]:
    """Whether a body-part descent was authored and routed cleanly.

    Skipped rather than failed where no descent exists. This gate reads a plan
    the compiler on this branch does not produce, and a gate that reports
    ``fail`` for a stage that was never attempted would route every geometric
    repair to ``whole_body`` and starve the arm, wrist and digit stages that
    actually diagnose the miss.
    """
    plan = getattr(program, "body_part_plan", None)
    if plan is None:
        return {
            "id": "architecture_route",
            "body_region": "whole_body",
            "stage": "primitive_descent",
            "status": "skip",
            "passed": True,
            "severity": 0.0,
            "measurements": {"defined_move_count": 0, "routing_failures": []},
            "feedback": None,
        }
    failures = clip.metrics.get("body_part_primitive_plan_failures", [])
    valid = bool(clip.metrics.get("body_part_primitive_plan_valid", False))
    return {
        "id": "architecture_route",
        "body_region": "whole_body",
        "stage": "primitive_descent",
        "status": "pass" if valid else "fail",
        "passed": valid,
        "severity": 0.0 if valid else 1.0,
        "measurements": {
            "defined_move_count": len(plan),
            "routing_failures": list(failures) if isinstance(failures, list) else [],
        },
        "feedback": None if valid else "rebuild the complete body-part primitive descent",
    }


def _digit_chain_check(
    program: MotionProgram,
    frame: ClipFrame,
    socket_position: np.ndarray,
    lateral_axis: np.ndarray,
    grasp_span_m: float,
) -> dict[str, Any]:
    palm_center, _normal, positions, tips = _hand_geometry(frame, program.hand)
    side = program.hand.value
    chain_digits = ("Index", "Middle", "Ring", "Little")
    chain_axis = (
        positions[f"{side}IndexProximal"]
        - positions[f"{side}LittleProximal"]
    )
    chain_axis /= max(float(np.linalg.norm(chain_axis)), 1e-12)
    minimum_order_gap = float("inf")
    inversion_count = 0
    for level in ("Proximal", "Intermediate", "Distal", "tip"):
        projections = [
            float(
                np.dot(
                    tips[digit.lower()]
                    if level == "tip"
                    else positions[f"{side}{digit}{level}"],
                    chain_axis,
                )
            )
            for digit in chain_digits
        ]
        gaps = [
            projections[index] - projections[index + 1]
            for index in range(len(projections) - 1)
        ]
        minimum_order_gap = min(minimum_order_gap, *gaps)
        inversion_count += sum(gap < DIGIT_ORDER_TOLERANCE_M for gap in gaps)

    thumb_projection = float(np.dot(tips["thumb"] - socket_position, lateral_axis))
    finger_projections = [
        float(np.dot(tips[digit] - socket_position, lateral_axis))
        for digit in ("index", "middle", "ring", "little")
    ]
    finger_projection = float(np.median(finger_projections))
    opposing_span = abs(thumb_projection - finger_projection)
    finger_center = np.mean(
        [tips[digit] for digit in ("index", "middle", "ring", "little")],
        axis=0,
    )
    opposition_segment = finger_center - tips["thumb"]
    segment_length_squared = float(np.dot(opposition_segment, opposition_segment))
    segment_alpha = float(
        np.clip(
            np.dot(socket_position - tips["thumb"], opposition_segment)
            / max(segment_length_squared, 1e-12),
            0.0,
            1.0,
        )
    )
    nearest_opposition_point = tips["thumb"] + segment_alpha * opposition_segment
    opposition_socket_error = nearest_opposition_point - socket_position
    opposition_socket_distance = float(np.linalg.norm(opposition_socket_error))
    parent_translation = -opposition_socket_error
    opposed = bool(
        math.sqrt(segment_length_squared) >= grasp_span_m * 0.30
        and opposition_socket_distance <= max(0.012, grasp_span_m * 0.30)
    )
    nearest_other_tip = min(
        float(np.linalg.norm(tips["thumb"] - tips[digit]))
        for digit in ("index", "middle", "ring", "little")
    )
    passed = bool(
        inversion_count == 0
        and nearest_other_tip >= THUMB_CLEARANCE_MINIMUM_M
        and opposed
    )
    severity = max(
        0.0,
        -minimum_order_gap / 0.02,
        (THUMB_CLEARANCE_MINIMUM_M - nearest_other_tip)
        / THUMB_CLEARANCE_MINIMUM_M,
        0.0 if opposed else 1.0,
    )
    return {
        "id": "digit_chain_and_opposition",
        "body_region": f"{side}_digits",
        "stage": PrimitiveKind.CLOSE.value,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "severity": float(severity),
        "measurements": {
            "minimum_digit_order_gap_m": minimum_order_gap,
            "digit_chain_inversion_count": inversion_count,
            "nearest_thumb_to_other_tip_m": nearest_other_tip,
            "thumb_socket_lateral_m": thumb_projection,
            "finger_socket_lateral_m": finger_projection,
            "opposing_span_m": opposing_span,
            "thumb_to_finger_center_m": math.sqrt(segment_length_squared),
            "opposition_segment_socket_distance_m": opposition_socket_distance,
            "opposed_about_socket": opposed,
            "requested_parent_translation_m": parent_translation.tolist(),
            "palm_center_m": palm_center.tolist(),
        },
        "feedback": (
            None
            if passed
            else "regenerate digit curl, splay, and thumb opposition without moving the accepted arm or wrist"
        ),
    }


def inspect_embodied_candidate(
    program: MotionProgram,
    clip: ClipResult,
    scene: SceneManifest,
) -> dict[str, Any]:
    """Inspect one compiled candidate in parent-before-child order."""

    architecture = _architecture_check(program, clip)
    checks: list[dict[str, Any]] = [architecture]
    if not architecture["passed"]:
        return {
            "protocol": MASTER_INSPECTION_PROTOCOL,
            "active_for_every_supported_input": True,
            "passed": False,
            "checks": checks,
            "first_failure": architecture,
        }

    socket = _socket_geometry(program, scene)
    if program.intent != Intent.GRAB or socket is None:
        continuity_checks = clip.metrics.get("universal_stage_checks", [])
        failed_continuity = next(
            (
                check
                for check in continuity_checks
                if isinstance(check, dict) and check.get("passed") is False
            ),
            None,
        )
        if failed_continuity is not None:
            region = str(failed_continuity.get("layer", "final_motion"))
            check = {
                "id": "universal_stage_continuity",
                "body_region": region,
                "stage": failed_continuity.get("primitive"),
                "status": "fail",
                "passed": False,
                "severity": 1.0,
                "measurements": failed_continuity.get("measurements", {}),
                "feedback": failed_continuity.get("feedback_to_primitive"),
            }
            checks.append(check)
            return {
                "protocol": MASTER_INSPECTION_PROTOCOL,
                "active_for_every_supported_input": True,
                "passed": False,
                "checks": checks,
                "first_failure": check,
            }
        return {
            "protocol": MASTER_INSPECTION_PROTOCOL,
            "active_for_every_supported_input": True,
            "passed": True,
            "checks": checks,
            "first_failure": None,
        }

    socket_position, approach_normal, lateral_axis, grasp_span, object_id, socket_id = socket
    contact_frame = (
        _phase_frame(clip, PrimitiveKind.CONTACT)
        or _phase_frame(clip, PrimitiveKind.PRESHAPE)
        or _phase_frame(clip, PrimitiveKind.REACH)
    )
    if contact_frame is None:
        missing = {
            "id": "object_socket_alignment",
            "body_region": f"{program.hand.value}_arm",
            "stage": "contact",
            "status": "fail",
            "passed": False,
            "severity": 1.0,
            "measurements": {"reason": "no inspectable object-contact frame"},
            "feedback": "regenerate the object approach lifecycle",
        }
        checks.append(missing)
        return {
            "protocol": MASTER_INSPECTION_PROTOCOL,
            "active_for_every_supported_input": True,
            "passed": False,
            "checks": checks,
            "first_failure": missing,
        }

    palm_center, palm_normal, _positions, _tips = _hand_geometry(
        contact_frame,
        program.hand,
    )
    position_error = palm_center - socket_position
    position_distance = float(np.linalg.norm(position_error))
    position_passed = position_distance <= PALM_SOCKET_ERROR_LIMIT_M
    position_check = {
        "id": "object_socket_alignment",
        "body_region": f"{program.hand.value}_arm",
        "stage": PrimitiveKind.CONTACT.value,
        "status": "pass" if position_passed else "fail",
        "passed": position_passed,
        "severity": max(
            0.0,
            (position_distance - PALM_SOCKET_ERROR_LIMIT_M)
            / PALM_SOCKET_ERROR_LIMIT_M,
        ),
        "measurements": {
            "object_id": object_id,
            "socket_id": socket_id,
            "palm_center_m": palm_center.tolist(),
            "socket_center_m": socket_position.tolist(),
            "palm_minus_socket_m": position_error.tolist(),
            "palm_socket_distance_m": position_distance,
            "maximum_distance_m": PALM_SOCKET_ERROR_LIMIT_M,
        },
        "feedback": (
            None
            if position_passed
            else "call back to the arm and translate its wrist target so the rendered palm, not merely the IK wrist, reaches the object socket"
        ),
    }
    checks.append(position_check)
    if not position_passed:
        return {
            "protocol": MASTER_INSPECTION_PROTOCOL,
            "active_for_every_supported_input": True,
            "passed": False,
            "checks": checks,
            "first_failure": position_check,
        }

    normal_alignment = abs(float(np.dot(palm_normal, approach_normal)))
    normal_angle = math.degrees(math.acos(float(np.clip(normal_alignment, -1.0, 1.0))))
    wrist_passed = normal_alignment >= PALM_NORMAL_MINIMUM_ALIGNMENT
    wrist_check = {
        "id": "palm_surface_orientation",
        "body_region": f"{program.hand.value}_wrist",
        "stage": PrimitiveKind.CONTACT.value,
        "status": "pass" if wrist_passed else "fail",
        "passed": wrist_passed,
        "severity": max(
            0.0,
            (PALM_NORMAL_MINIMUM_ALIGNMENT - normal_alignment)
            / PALM_NORMAL_MINIMUM_ALIGNMENT,
        ),
        "measurements": {
            "absolute_normal_alignment": normal_alignment,
            "normal_error_deg": normal_angle,
            "minimum_alignment": PALM_NORMAL_MINIMUM_ALIGNMENT,
        },
        "feedback": (
            None
            if wrist_passed
            else "keep the accepted arm target and regenerate only wrist pitch, yaw, and roll to face the object socket"
        ),
    }
    checks.append(wrist_check)
    if not wrist_passed:
        return {
            "protocol": MASTER_INSPECTION_PROTOCOL,
            "active_for_every_supported_input": True,
            "passed": False,
            "checks": checks,
            "first_failure": wrist_check,
        }

    close_frame = _phase_frame(clip, PrimitiveKind.CLOSE)
    if close_frame is not None:
        digit_check = _digit_chain_check(
            program,
            close_frame,
            socket_position,
            lateral_axis,
            grasp_span,
        )
        checks.append(digit_check)
        if not digit_check["passed"]:
            return {
                "protocol": MASTER_INSPECTION_PROTOCOL,
                "active_for_every_supported_input": True,
                "passed": False,
                "checks": checks,
                "first_failure": digit_check,
            }

    coupled = clip.metrics.get("render_physics_pose_coupled") is True
    opposing_contacts = clip.metrics.get("opposing_contacts") is True
    lift_height = float(clip.metrics.get("lift_height_m", 0.0))
    requested_lift = max(
        (
            primitive.parameters.lift_height_m
            for primitive in program.primitives
            if primitive.kind == PrimitiveKind.LIFT
        ),
        default=0.0,
    )
    recover_support_ratio = float(
        clip.metrics.get("recover_support_ratio", 0.0)
    )
    maximum_penetration = float(clip.metrics.get("max_penetration_m", 0.0))
    dynamics_passed = bool(
        coupled
        and opposing_contacts
        and lift_height >= requested_lift - 0.015
        and recover_support_ratio >= 0.80
        and maximum_penetration < 0.006
    )
    dynamics_check = {
        "id": "embodied_contact_dynamics",
        "body_region": (
            f"{program.hand.value}_digits"
            if not opposing_contacts
            else f"{program.hand.value}_arm"
        ),
        "stage": "continuous_contact_through_recover",
        "status": "pass" if dynamics_passed else "fail",
        "passed": dynamics_passed,
        "severity": max(
            0.0 if coupled else 1.0,
            0.0 if opposing_contacts else 1.0,
            (requested_lift - 0.015 - lift_height)
            / max(requested_lift, 0.01),
            (0.80 - recover_support_ratio) / 0.80,
            (maximum_penetration - 0.006) / 0.006,
        ),
        "measurements": {
            "render_physics_pose_coupled": coupled,
            "opposing_contacts": opposing_contacts,
            "opposing_contact_ratio": clip.metrics.get(
                "opposing_contact_ratio",
                0.0,
            ),
            "lift_height_m": lift_height,
            "requested_lift_height_m": requested_lift,
            "recover_support_ratio": recover_support_ratio,
            "maximum_penetration_m": maximum_penetration,
            "collision_body_count": clip.metrics.get("physics_model", {}).get(
                "collision_body_count",
                0,
            ),
        },
        "feedback": (
            None
            if dynamics_passed
            else "the renderer-driven palm and finger collision bodies did not maintain a real opposing grasp through lift, hold, and recovery"
        ),
    }
    checks.append(dynamics_check)
    if not dynamics_passed:
        return {
            "protocol": MASTER_INSPECTION_PROTOCOL,
            "active_for_every_supported_input": True,
            "passed": False,
            "checks": checks,
            "first_failure": dynamics_check,
        }

    return {
        "protocol": MASTER_INSPECTION_PROTOCOL,
        "active_for_every_supported_input": True,
        "passed": True,
        "checks": checks,
        "first_failure": None,
    }


def _update_object_phases(
    program: MotionProgram,
    updates: dict[str, float],
    *,
    regions: set[PrimitiveKind],
) -> MotionProgram:
    repaired = program.model_copy(deep=True)
    for primitive in repaired.primitives:
        if primitive.object_id and primitive.kind in regions:
            primitive.parameters = primitive.parameters.model_copy(update=updates)
    return repaired


def _state_owning(gate_id: str):
    """The state a gate is checked in, if any state claims it."""
    for state in states().values():
        if state.runs(gate_id):
            return state
    return None


def scope_proposals_to_active_regions(
    proposals: list[tuple[Any, dict[str, Any]]],
    inspection: dict[str, Any],
) -> list[tuple[Any, dict[str, Any]]]:
    """Drop repairs aimed at a region the failing state is not solving.

    Two reasons, and the second matters more than the saving. A repair aimed at
    an inert region costs a full compile and simulation to establish that a part
    which did not move still has not moved. And during ``lift`` the digits are
    inert *because they are holding the object*: re-solving them there changes
    the grip that is at that moment carrying the block, which is how a candidate
    that was holding it drops it.

    If scoping would reject everything the proposals are returned unchanged. A
    filter that can empty the repair set turns a fixable candidate into an
    unfixable one, which is worse than the cost it saves.
    """
    failure = inspection.get("first_failure")
    if not isinstance(failure, dict):
        return proposals
    state = _state_owning(str(failure.get("id", "")))
    if state is None or not state.active_regions:
        return proposals
    kept = [
        (program, callback)
        for program, callback in proposals
        if state.solves(str(callback.get("target_body_region", "")))
    ]
    return kept or proposals


def body_region_repair_candidates(
    program: MotionProgram,
    inspection: dict[str, Any],
    *,
    attempt: int,
) -> list[tuple[MotionProgram, dict[str, Any]]]:
    """Return bounded repairs for only the first failed body region."""

    failure = inspection.get("first_failure")
    if not isinstance(failure, dict):
        return []
    body_region = str(failure.get("body_region", ""))
    measurements = failure.get("measurements")
    measurements = measurements if isinstance(measurements, dict) else {}
    all_object_phases = {
        PrimitiveKind.REACH,
        PrimitiveKind.PRESHAPE,
        PrimitiveKind.CONTACT,
        PrimitiveKind.CLOSE,
        PrimitiveKind.LIFT,
        PrimitiveKind.HOLD,
    }
    context = {
        "master_prompt": program.source_text,
        "nearest_parent_context_weight": 1.0,
        "previous_parent_context_weight": 0.55,
        "callback_attempt": attempt + 1,
        "maximum_callback_attempts": MAX_MASTER_CALLBACKS,
    }

    if body_region.endswith("_arm") and failure.get("id") == "object_socket_alignment":
        error = measurements.get("palm_minus_socket_m")
        if not isinstance(error, list) or len(error) != 3:
            return []
        gain = 0.85 if attempt == 0 else 1.0
        sample = next(
            (
                primitive
                for primitive in program.primitives
                if primitive.object_id and primitive.kind == PrimitiveKind.CONTACT
            ),
            None,
        )
        if sample is None:
            return []
        updates = {
            "lateral_offset": _clamp(
                sample.parameters.lateral_offset - float(error[0]) / 0.08 * gain,
                -1.0,
                1.0,
            ),
            "arm_height": _clamp(
                sample.parameters.arm_height - float(error[1]) / 0.08 * gain,
                -1.0,
                1.0,
            ),
            "arm_depth": _clamp(
                sample.parameters.arm_depth - float(error[2]) / 0.08 * gain,
                -1.0,
                1.0,
            ),
        }
        repaired = _update_object_phases(
            program,
            updates,
            regions=all_object_phases,
        )
        return [
            (
                repaired,
                {
                    "protocol": MASTER_CALLBACK_PROTOCOL,
                    "reason": "rendered_palm_missed_object_socket",
                    "target_body_region": body_region,
                    "preserved_regions": ["head", "trunk", "legs"],
                    "invalidated_descendants": [
                        body_region.replace("_arm", "_forearm"),
                        body_region.replace("_arm", "_wrist"),
                        body_region.replace("_arm", "_palm"),
                        body_region.replace("_arm", "_digits"),
                    ],
                    "parameter_updates": updates,
                    "feedback": failure.get("feedback"),
                    "context": context,
                },
            )
        ]

    if body_region.endswith("_wrist"):
        candidates: list[tuple[MotionProgram, dict[str, Any]]] = []
        for name in ("wrist_pitch", "wrist_yaw", "wrist_roll"):
            for delta in (-0.24, 0.24):
                repaired = program.model_copy(deep=True)
                for primitive in repaired.primitives:
                    if primitive.object_id and primitive.kind in all_object_phases:
                        value = _clamp(
                            float(getattr(primitive.parameters, name)) + delta,
                            -1.0,
                            1.0,
                        )
                        primitive.parameters = primitive.parameters.model_copy(
                            update={name: value}
                        )
                candidates.append(
                    (
                        repaired,
                        {
                            "protocol": MASTER_CALLBACK_PROTOCOL,
                            "reason": "palm_surface_not_aligned_to_object",
                            "target_body_region": body_region,
                            "preserved_regions": [
                                "head",
                                "trunk",
                                body_region.replace("_wrist", "_arm"),
                                body_region.replace("_wrist", "_forearm"),
                            ],
                            "invalidated_descendants": [
                                body_region.replace("_wrist", "_palm"),
                                body_region.replace("_wrist", "_digits"),
                            ],
                            "parameter_updates": {name: delta},
                            "feedback": failure.get("feedback"),
                            "context": context,
                        },
                    )
                )
        return candidates

    if body_region.endswith("_digits"):
        # A distal grip can be internally valid yet remain entirely on one
        # side of the object.  Curling harder cannot solve that geometry: the
        # digit controller must call back through palm/wrist to translate the
        # arm, then retry the unchanged digit ordering against the new frame.
        opposed = measurements.get("opposed_about_socket") is True
        parent_translation = measurements.get("requested_parent_translation_m")
        if (
            not opposed
            and isinstance(parent_translation, list)
            and len(parent_translation) == 3
        ):
            sample = next(
                (
                    primitive
                    for primitive in program.primitives
                    if primitive.object_id
                    and primitive.kind == PrimitiveKind.CONTACT
                ),
                None,
            )
            if sample is not None:
                gain = 0.90
                updates = {
                    "lateral_offset": _clamp(
                        sample.parameters.lateral_offset
                        + float(parent_translation[0]) / 0.08 * gain,
                        -1.0,
                        1.0,
                    ),
                    "arm_height": _clamp(
                        sample.parameters.arm_height
                        + float(parent_translation[1]) / 0.08 * gain,
                        -1.0,
                        1.0,
                    ),
                    "arm_depth": _clamp(
                        sample.parameters.arm_depth
                        + float(parent_translation[2]) / 0.08 * gain,
                        -1.0,
                        1.0,
                    ),
                }
                repaired = _update_object_phases(
                    program,
                    updates,
                    regions=all_object_phases,
                )
                parent_region = body_region.replace("_digits", "_arm")
                return [
                    (
                        repaired,
                        {
                            "protocol": MASTER_CALLBACK_PROTOCOL,
                            "reason": "digits_cannot_oppose_without_parent_translation",
                            "source_body_region": body_region,
                            "target_body_region": parent_region,
                            "preserved_regions": ["head", "trunk", "legs"],
                            "invalidated_descendants": [
                                parent_region.replace("_arm", "_forearm"),
                                parent_region.replace("_arm", "_wrist"),
                                parent_region.replace("_arm", "_palm"),
                                body_region,
                            ],
                            "parameter_updates": updates,
                            "feedback": (
                                "the digits exhausted their local opposition range; "
                                "translate the arm parent and retry the digit stage"
                            ),
                            "context": context,
                        },
                    )
                ]
        updates = {
            "finger_curl": 0.56,
            "finger_splay": 0.08,
            "thumb_opposition": 1.0,
        }
        repaired = _update_object_phases(
            program,
            updates,
            regions={
                PrimitiveKind.PRESHAPE,
                PrimitiveKind.CONTACT,
                PrimitiveKind.CLOSE,
                PrimitiveKind.LIFT,
                PrimitiveKind.HOLD,
            },
        )
        return [
            (
                repaired,
                {
                    "protocol": MASTER_CALLBACK_PROTOCOL,
                    "reason": "digit_chain_or_thumb_opposition_failed",
                    "target_body_region": body_region,
                    "preserved_regions": [
                        "head",
                        "trunk",
                        body_region.replace("_digits", "_arm"),
                        body_region.replace("_digits", "_forearm"),
                        body_region.replace("_digits", "_wrist"),
                        body_region.replace("_digits", "_palm"),
                    ],
                    "invalidated_descendants": [],
                    "parameter_updates": updates,
                    "feedback": failure.get("feedback"),
                    "context": context,
                },
            )
        ]
    return []


def inspection_score(inspection: dict[str, Any]) -> float:
    checks = inspection.get("checks")
    if not isinstance(checks, list):
        return float("inf")
    return float(
        sum(
            float(check.get("severity", 1.0))
            for check in checks
            if isinstance(check, dict) and check.get("passed") is False
        )
    )


def inspection_progress(inspection: dict[str, Any]) -> tuple[int, float]:
    """Rank parent-first progress before the residual error at that level."""

    checks = inspection.get("checks")
    if not isinstance(checks, list):
        return (-1, float("-inf"))
    first_failed_index = len(checks)
    for index, check in enumerate(checks):
        if isinstance(check, dict) and check.get("passed") is False:
            first_failed_index = index
            break
    return (first_failed_index, -inspection_score(inspection))


def run_embodied_master_controller(
    program: MotionProgram,
    clip: ClipResult,
    scene: SceneManifest,
    compile_candidate: Callable[[MotionProgram], ClipResult],
    *,
    maximum_callbacks: int = MAX_MASTER_CALLBACKS,
) -> tuple[MotionProgram, ClipResult, dict[str, Any], list[dict[str, Any]]]:
    """Inspect, call back one region, recompile, and repeat until satisfied."""

    current_program = program
    current_clip = clip
    inspection = inspect_embodied_candidate(current_program, current_clip, scene)
    callbacks: list[dict[str, Any]] = []
    for attempt in range(maximum_callbacks):
        if inspection.get("passed") is True:
            break
        proposals = scope_proposals_to_active_regions(
            body_region_repair_candidates(
                current_program,
                inspection,
                attempt=attempt,
            ),
            inspection,
        )
        if not proposals:
            break
        baseline_score = inspection_score(inspection)
        baseline_progress = inspection_progress(inspection)
        best: tuple[MotionProgram, ClipResult, dict[str, Any], dict[str, Any]] | None = None
        best_progress = baseline_progress
        for proposal, callback in proposals:
            proposal_clip = compile_candidate(proposal)
            proposal_inspection = inspect_embodied_candidate(
                proposal,
                proposal_clip,
                scene,
            )
            proposal_score = inspection_score(proposal_inspection)
            proposal_progress = inspection_progress(proposal_inspection)
            if proposal_progress > best_progress:
                best = (proposal, proposal_clip, proposal_inspection, callback)
                best_progress = proposal_progress
                if proposal_inspection.get("passed") is True:
                    break
        if best is None:
            break
        current_program, current_clip, inspection, callback = best
        callbacks.append(
            {
                **callback,
                "callback_attempt": attempt + 1,
                "before_score": baseline_score,
                "after_score": inspection_score(inspection),
                "before_stage_index": baseline_progress[0],
                "after_stage_index": best_progress[0],
                "reinspection_passed": inspection.get("passed") is True,
            }
        )
    return current_program, current_clip, inspection, callbacks
