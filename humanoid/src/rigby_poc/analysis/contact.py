"""Intra-hand fingertip contact and gaze-tracking metrics.

Moved verbatim from ``compiler._intra_hand_contact_metrics`` and
``compiler._append_intra_hand_contact_failures``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..kinematics import rig_kinematics
from ..models import ClipFrame, MotionProgram
from .context import AnalysisContext
from .contract import (
    ANATOMY,
    CONTRACT,
    CheckResult,
    binary_check,
    lower_bound_check,
    upper_bound_check,
)
from .rig import EGO_NEUTRAL_GAZE, identity_bones


def intra_hand_contact_metrics(
    frames: list[ClipFrame],
    phase_ranges: list[dict[str, float | str]],
    program: MotionProgram,
    *,
    ctx: AnalysisContext | None = None,
) -> dict[str, Any]:
    """Fingertip-contact and gaze metrics for one clip.

    Pass ``ctx`` — whose ``frames`` must be the ``frames`` argument — and every
    world quantity is sliced out of that frame's single forward-kinematics
    evaluation, shared with every other check. Without it the hierarchy is
    walked here instead, once per contact-phase frame for the fingertips and
    once per gaze frame for the head rotation: the same values, at roughly one
    extra full pass each.

    The rest-head rotation below is deliberately *not* served from the context.
    It is taken at the identity pose, which is not a clip frame, so there is no
    frame index to cache it under.
    """

    contacts = [
        primitive
        for primitive in program.primitives
        if primitive.intra_hand_contact is not None
    ]
    gaze_primitives = [
        primitive
        for primitive in program.primitives
        if primitive.gaze_target is not None
    ]
    if not contacts and not gaze_primitives:
        return {}
    ranges = {
        str(item.get("label")): (float(item["start_s"]), float(item["end_s"]))
        for item in phase_ranges
    }
    kinematics = ctx.kinematics if ctx is not None else rig_kinematics()
    contact_records: list[dict[str, Any]] = []
    observed_order: list[str] = []
    release_separations: list[float] = []

    def fingertips(index: int, hand: str) -> dict[str, np.ndarray]:
        if ctx is not None:
            return ctx.fingertip_positions(index, hand)
        return kinematics.fingertip_positions(frames[index].bones, hand)

    for primitive in contacts:
        contact = primitive.intra_hand_contact
        assert contact is not None
        interval = ranges.get(primitive.label or "")
        # Indices, not frames: the same interval predicate to the same
        # tolerance, but the index is what addresses the shared per-frame cache.
        phase_indices = (
            [
                index
                for index, frame in enumerate(frames)
                if interval[0] - 1e-8 <= frame.time_s <= interval[1] + 1e-8
            ]
            if interval is not None
            else []
        )
        closest: tuple[float, int, dict[str, np.ndarray]] | None = None
        for index in phase_indices:
            tips = fingertips(index, contact.hand.value)
            distance = float(
                np.linalg.norm(
                    tips[contact.driver_digit.value]
                    - tips[contact.target_digit.value]
                )
            )
            if closest is None or distance < closest[0]:
                closest = (distance, index, tips)
        minimum_distance = closest[0] if closest is not None else float("inf")
        non_target_distance = (
            min(
                float(
                    np.linalg.norm(
                        closest[2][contact.driver_digit.value] - position
                    )
                )
                for digit, position in closest[2].items()
                if digit
                not in {
                    contact.driver_digit.value,
                    contact.target_digit.value,
                }
            )
            if closest is not None
            else 0.0
        )
        passed = bool(
            minimum_distance <= contact.maximum_distance_m
            and minimum_distance < non_target_distance
        )
        if passed:
            observed_order.append(contact.target_digit.value)
        contact_records.append(
            {
                "driver_digit": contact.driver_digit.value,
                "target_digit": contact.target_digit.value,
                "phase": primitive.label,
                "minimum_distance_m": minimum_distance,
                "maximum_distance_m": contact.maximum_distance_m,
                "nearest_other_fingertip_m": non_target_distance,
                "contact_time_s": (
                    frames[closest[1]].time_s if closest is not None else None
                ),
                "passed": passed,
            }
        )

        release_label = (primitive.label or "").replace("_touch_", "_release_", 1)
        release_interval = ranges.get(release_label)
        if release_interval is not None:
            release_index = min(
                range(len(frames)),
                key=lambda item: abs(frames[item].time_s - release_interval[1]),
            )
            release_tips = fingertips(release_index, contact.hand.value)
            release_separations.append(
                float(
                    np.linalg.norm(
                        release_tips[contact.driver_digit.value]
                        - release_tips[contact.target_digit.value]
                    )
                )
            )

    # Shared with the visibility sampler's ego camera through the context: the
    # identity pose has no frame index to memoise under, so without one owner
    # the two would evaluate the same rest hierarchy twice per composite clip.
    rest_head_rotation = (
        ctx.rest_head_transform[1]
        if ctx is not None
        else kinematics.canonical_world_rotation(identity_bones(), "head")
    )
    gaze_records: list[dict[str, Any]] = []
    for primitive in gaze_primitives:
        gaze = primitive.gaze_target
        assert gaze is not None
        interval = ranges.get(primitive.label or "")
        if interval is None:
            continue
        index = min(
            range(len(frames)),
            key=lambda item: abs(frames[item].time_s - interval[1]),
        )
        frame = frames[index]
        positions = (
            ctx.world_positions[index]
            if ctx is not None
            else kinematics.canonical_positions(frame.bones)
        )
        if gaze.hand is not None:
            target = positions[f"{gaze.hand.value}Hand"]
        else:
            transform = frame.objects.get(gaze.object_id or "")
            if transform is None:
                continue
            target = np.asarray(transform.translation.as_list(), dtype=float)
        head_rotation = (
            ctx.world_rotation(index, "head")
            if ctx is not None
            else kinematics.canonical_world_rotation(frame.bones, "head")
        )
        head_delta = head_rotation @ rest_head_rotation.T
        forward = head_delta @ EGO_NEUTRAL_GAZE
        direction = target - positions["head"]
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        angle = math.degrees(
            math.acos(float(np.clip(np.dot(forward, direction), -1.0, 1.0)))
        )
        gaze_records.append(
            {
                "phase": primitive.label,
                "target_hand": gaze.hand.value if gaze.hand is not None else None,
                "target_object": gaze.object_id,
                "angle_deg": angle,
                "maximum_angle_deg": gaze.maximum_angle_deg,
                "passed": angle <= gaze.maximum_angle_deg,
            }
        )

    return {
        "intra_hand_contact_expected_order": [
            primitive.intra_hand_contact.target_digit.value
            for primitive in contacts
            if primitive.intra_hand_contact is not None
        ],
        "intra_hand_contact_observed_order": observed_order,
        "intra_hand_contact_count": sum(
            1 for record in contact_records if record["passed"]
        ),
        "intra_hand_contact_records": contact_records,
        "intra_hand_minimum_release_separation_m": min(
            release_separations,
            default=0.0,
        ),
        "gaze_target_records": gaze_records,
        "gaze_max_endpoint_angle_deg": max(
            (float(record["angle_deg"]) for record in gaze_records),
            default=0.0,
        ),
    }


def _contact_assertion(program: MotionProgram):
    return next(
        (
            assertion
            for assertion in program.assertions
            if assertion.name == "ordered_intra_hand_contacts"
        ),
        None,
    )


def _gaze_assertion(program: MotionProgram):
    return next(
        (
            assertion
            for assertion in program.assertions
            if assertion.name == "gaze_tracks_active_hand"
        ),
        None,
    )


def intra_hand_contact_failures(
    program: MotionProgram,
    metrics: dict[str, Any],
) -> list[str]:
    """The contact/gaze structural failures, in the compiler's exact order."""

    failures: list[str] = []
    contact_assertion = _contact_assertion(program)
    if contact_assertion is not None:
        expected_count = int(round(contact_assertion.threshold or 0.0))
        observed_count = int(metrics.get("intra_hand_contact_count", 0))
        expected_order = metrics.get("intra_hand_contact_expected_order", [])
        observed_order = metrics.get("intra_hand_contact_observed_order", [])
        if observed_count != expected_count or observed_order != expected_order:
            failures.append(
                "ordered fingertip contacts were not completed in the requested sequence"
            )
        if float(metrics.get("intra_hand_minimum_release_separation_m", 0.0)) < 0.025:
            failures.append(
                "thumb does not visibly separate between successive fingertip contacts"
            )
    gaze_assertion = _gaze_assertion(program)
    if gaze_assertion is not None and float(
        metrics.get("gaze_max_endpoint_angle_deg", float("inf"))
    ) > float(gaze_assertion.threshold or 12.0):
        failures.append("head/camera gaze does not track the requested hand")
    return failures


def intra_hand_contact_checks(
    program: MotionProgram,
    metrics: dict[str, Any],
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    contact_assertion = _contact_assertion(program)
    if contact_assertion is not None:
        expected_count = int(round(contact_assertion.threshold or 0.0))
        observed_count = int(metrics.get("intra_hand_contact_count", 0))
        expected_order = metrics.get("intra_hand_contact_expected_order", [])
        observed_order = metrics.get("intra_hand_contact_observed_order", [])
        ordered = (
            observed_count == expected_count and observed_order == expected_order
        )
        missed = abs(expected_count - observed_count)
        # A wrong order with the right count is still a total failure of the
        # requested sequence, so it saturates rather than scoring zero.
        order_severity = (
            0.0
            if ordered
            else (min(1.0, missed / max(expected_count, 1)) if missed else 1.0)
        )
        checks.append(
            binary_check(
                "contract.contact.ordered_intra_hand",
                CONTRACT,
                passed=ordered,
                measured={
                    "expected_order": list(expected_order),
                    "observed_order": list(observed_order),
                    "expected_count": expected_count,
                    "observed_count": observed_count,
                },
                threshold=float(expected_count),
                severity=order_severity,
                detail="ordered fingertip contacts were not completed in the requested sequence",
            )
        )
        checks.append(
            lower_bound_check(
                "contract.contact.release_separation",
                CONTRACT,
                float(metrics.get("intra_hand_minimum_release_separation_m", 0.0)),
                0.025,
                scale=0.025,
                detail="thumb does not visibly separate between successive fingertip contacts",
            )
        )
    gaze_assertion = _gaze_assertion(program)
    if gaze_assertion is not None:
        threshold = float(gaze_assertion.threshold or 12.0)
        checks.append(
            upper_bound_check(
                "anatomy.gaze.tracks_target",
                ANATOMY,
                float(metrics.get("gaze_max_endpoint_angle_deg", float("inf"))),
                threshold,
                scale=threshold,
                detail="head/camera gaze does not track the requested hand",
            )
        )
    return checks
