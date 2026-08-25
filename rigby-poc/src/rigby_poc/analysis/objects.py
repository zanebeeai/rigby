"""The object-handoff metric pass, moved out of ``_compile_object_handoff``.

Unlike the five paths moved before it, this one is **not a relocation**. Plan 02
§1.1 claims no metric is accumulated inside a per-frame loop; that holds for
whole body, composite, gesture, strike and grab — each was moved intact and
verified byte-identical — and is false here. ``object_steps`` accumulates per
frame and the ownership transitions are stamped with frame times from inside the
same loop. So every generation-accumulated value is re-derived from what the
clip actually shows, and the equivalence harness is what says the derivation is
right rather than merely plausible.

One value is carried rather than derived: ``object_lifecycle``. Its transitions
fire on authored phase kinds at fixed progress fractions, so reconstructing them
would mean recomputing ``elapsed + progress * phase_duration_s`` from
``phase_ranges_s`` — whose bounds are running sums, which is exactly the
arithmetic that moved the composite presentation window by an ulp in 02c. It is
already a published metric, so carrying it costs no new key; the compiler simply
writes it before the pass rather than after.

Everything else follows from the lifecycle plus the frames:

* ownership over time — source from ``source_attached`` until
  ``receiver_attached``, receiver thereafter;
* the object's own track, for ``object_max_step_m``;
* the attachment offset per frame, from ``arm_landmarks`` on the owning hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import Hand, ObjectAction
from .context import AnalysisContext
from .gesture import arm_landmarks
from .safety import clip_contract_violations, safety_metrics as _safety_metrics

#: The reference the compiler compares ``object_max_step_m`` against. A step
#: larger than this is a teleport rather than motion.
MAX_STEP_REFERENCE_M = 0.08

_SOURCE_ATTACHED = "source_attached"
_DUAL_CONTACT = "dual_contact"
_RECEIVER_ATTACHED = "receiver_attached"


def _state_time(lifecycle: list[dict[str, Any]], state: str) -> float | None:
    for entry in lifecycle:
        if entry.get("state") == state:
            return float(entry["time_s"])
    return None


def carried_object_id(ctx: AnalysisContext) -> str | None:
    """The object the handoff moves: the one whose transform is not constant.

    Derived rather than read off the program because the program names an
    object id only when the planner supplied one, while every scene object
    appears in every frame.
    """

    for object_id, track in ctx.object_tracks.items():
        first = track[0].translation.as_list()
        if any(item.translation.as_list() != first for item in track):
            return object_id
    return next(iter(ctx.object_tracks), None)


def handoff_metrics(ctx: AnalysisContext) -> dict[str, Any]:
    """Every metric the handoff path emits, bar the carry-overs."""

    frames = ctx.frames
    program = ctx.program
    lifecycle = list(ctx.compiled_metric("object_lifecycle", []) or [])

    source = program.hand
    receiver = next((hand for hand in program.hands if hand != source), None)

    source_attachment_time_s = _state_time(lifecycle, _SOURCE_ATTACHED)
    receiver_contact_time_s = _state_time(lifecycle, _DUAL_CONTACT)
    transfer_time_s = _state_time(lifecycle, _RECEIVER_ATTACHED)

    object_id = carried_object_id(ctx)
    attachment_offsets: dict[Hand, list[np.ndarray]] = {source: []}
    if receiver is not None:
        attachment_offsets[receiver] = []
    object_steps: list[float] = []

    previous: np.ndarray | None = None
    for frame in frames:
        transform = frame.objects.get(object_id or "")
        if transform is None:
            continue
        position = np.asarray(transform.translation.as_list(), dtype=float)
        if previous is not None:
            object_steps.append(float(np.linalg.norm(position - previous)))
        previous = position

        if source_attachment_time_s is None or frame.time_s < source_attachment_time_s:
            continue
        owner = (
            receiver
            if transfer_time_s is not None
            and frame.time_s >= transfer_time_s
            and receiver is not None
            else source
        )
        _, _, wrist, hand_world = arm_landmarks(frame, owner)
        attachment_offsets[owner].append(hand_world.inv().apply(position - wrist))

    max_step = max(object_steps, default=0.0)
    slip_by_hand = {
        hand.value: (
            max(
                float(np.linalg.norm(offset - offsets[0]))
                for offset in offsets
            )
            if offsets
            else float("inf")
        )
        for hand, offsets in attachment_offsets.items()
    }
    dual_contact_duration = (
        transfer_time_s - receiver_contact_time_s
        if transfer_time_s is not None and receiver_contact_time_s is not None
        else 0.0
    )
    # The compiler's `owner` after the loop is whichever hand last took the
    # object, which is the receiver exactly when the transfer fired.
    owner = receiver if transfer_time_s is not None else source

    safety = _safety_metrics(frames)
    structural_failures: list[str] = []
    if source_attachment_time_s is None:
        structural_failures.append("handoff source never secured the object")
    if receiver_contact_time_s is None:
        structural_failures.append("handoff receiver never contacted the object")
    if transfer_time_s is None:
        structural_failures.append("object ownership never transferred")
    if dual_contact_duration < 0.08:
        structural_failures.append("handoff lacks a stable dual-hand overlap")
    if owner != receiver:
        structural_failures.append("receiver did not retain the object")
    if any(value > 0.005 for value in slip_by_hand.values()):
        structural_failures.append("object slipped relative to an owning palm")
    if max_step > MAX_STEP_REFERENCE_M:
        structural_failures.append(
            "handoff object trajectory contains a teleport-sized step"
        )
    if safety["nan_count"]:
        structural_failures.append("clip contains non-finite transforms")
    if clip_contract_violations(safety, allow_root_motion=False):
        structural_failures.append("clip exceeds a joint limit")
    if safety["discontinuities"]:
        structural_failures.append(
            f"clip contains {safety['discontinuities']} rotational discontinuities"
        )

    metrics: dict[str, Any] = {}
    metrics.update(safety)
    metrics.update(
        {
            "object_action": ObjectAction.HANDOFF.value,
            "active_hands": [
                source.value,
                receiver.value if receiver is not None else source.value,
            ],
            "handoff_source_hand": source.value,
            "handoff_receiver_hand": (
                receiver.value if receiver is not None else None
            ),
            "handoff_source_attachment_time_s": source_attachment_time_s,
            "handoff_receiver_contact_time_s": receiver_contact_time_s,
            "handoff_transfer_time_s": transfer_time_s,
            "handoff_dual_contact_duration_s": dual_contact_duration,
            "handoff_receiver_retained": owner == receiver,
            "handoff_attachment_slip_m": max(slip_by_hand.values()),
            "handoff_attachment_slip_by_hand_m": slip_by_hand,
            "object_max_step_m": max_step,
            "object_max_step_reference_m": MAX_STEP_REFERENCE_M,
            "structural_failures": structural_failures,
            "structural_valid": not structural_failures,
        }
    )
    return metrics


__all__ = ["MAX_STEP_REFERENCE_M", "carried_object_id", "handoff_metrics"]
