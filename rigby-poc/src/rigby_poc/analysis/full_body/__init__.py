"""The whole-body metric pass, moved out of ``_compile_full_body``.

``full_body_metrics(ctx)`` runs the same blocks in the same order the compiler
ran them, because several of them read keys an earlier block wrote — the burpee
block reads the root height envelope, the push-up block republishes the plank
range, and the structural checks read almost everything. Order is part of the
behaviour, so it is preserved literally rather than tidied into an unordered set
of independent analyzers.

Only three things had to change to make the pass run on a finished clip:

* the primitive selectors were hoisted into :class:`~.selectors.FullBodyPass`,
  where both the metric blocks and the structural checks can share them;
* ``current_yaw`` is re-derived from the program rather than carried over — see
  :func:`~.root.commanded_root_yaw_rad`;
* the commanded IK support targets are read from ``metrics``, because the
  compiler now persists them. They are the one input here that genuinely cannot
  be recovered from frames.
"""

from __future__ import annotations

from typing import Any

from ...models import BodyAction
from ..contract import AnalyzerEntry
from ..safety import safety_metrics
from ..semantic import semantic_cycle_metrics
from .balance import angular_metrics, final_balance_metrics, support_metrics
from .climb import climb_metrics
from .dance import dance_metrics
from .exercises import (
    burpee_metrics,
    crawl_metrics,
    jumping_jack_metrics,
    lunge_metrics,
    push_up_metrics,
    single_leg_metrics,
    sit_up_metrics,
    squat_metrics,
)
from .failures import full_body_failures
from .ground import airborne_metrics, ground_metrics
from .obstacle import obstacle_metrics
from .posture import posture_metrics
from .root import commanded_root_yaw_rad, root_metrics
from .rotation import rotation_metrics
from .selectors import FullBodyPass


def full_body_metrics(ctx: "Any") -> dict[str, Any]:
    """Every metric the whole-body compile path emits, bar the carry-overs.

    Excluded are ``phase_ranges_s``, ``body_actions``, ``root_motion_enabled``
    and the two support-constraint records: those are written by the compiler
    before the measurement pass begins and are inputs to it, not outputs of it.
    """

    fb = FullBodyPass(ctx)
    metrics: dict[str, Any] = {}

    metrics.update(safety_metrics(fb.frames, allow_root_motion=True))
    root_metrics(fb, metrics)

    jumping_jack_metrics(fb, metrics)
    burpee_metrics(fb, metrics)
    squat_metrics(fb, metrics)
    lunge_metrics(fb, metrics)
    single_leg_metrics(fb, metrics)
    sit_up_metrics(fb, metrics)
    crawl_metrics(fb, metrics)
    push_up_metrics(fb, metrics)

    ground_metrics(fb, metrics)
    climb_metrics(fb, metrics)
    dance_metrics(fb, metrics)
    rotation_metrics(fb, metrics)
    obstacle_metrics(fb, metrics)
    horizontal_contact_count = posture_metrics(fb, metrics)

    airborne_metrics(fb, metrics)
    support_metrics(fb, metrics)
    angular_metrics(fb, metrics)
    final_balance_metrics(fb, metrics)

    metrics.update(semantic_cycle_metrics(fb.frames, fb.phase_ranges, fb.program))

    failures = full_body_failures(fb, metrics, horizontal_contact_count)
    metrics["structural_failures"] = failures
    metrics["structural_valid"] = not failures
    return metrics


def _body_entry(note: str) -> AnalyzerEntry:
    return AnalyzerEntry(full_body_metrics, "analysis.full_body", "02b", note=note)


# Which block each action gates. Five of the twelve members gate none of their
# own: they contribute only to the shared root, ground, support, angular and
# balance blocks, which are ported too. Registering them against the same pass
# keeps that honest — the action is analysed, it simply has no bespoke metrics.
BODY_ENTRIES = {
    BodyAction.HOLD: _body_entry("shared blocks only"),
    BodyAction.STEP: _body_entry("shared blocks only"),
    BodyAction.WALK: _body_entry("shared blocks only; gates the obstacle block"),
    BodyAction.RUN: _body_entry("shared blocks only; gates the obstacle block"),
    BodyAction.TURN: _body_entry("shared blocks only; drives final_root_yaw_deg"),
    BodyAction.CROUCH: _body_entry("squat_* labels"),
    BodyAction.JUMP: _body_entry("jumping jacks and burpee_* labels"),
    BodyAction.KICK: _body_entry("shared blocks only"),
    BodyAction.DANCE: _body_entry("dance beats, sway and alternating lifts"),
    BodyAction.CLIMB: _body_entry("needs the persisted climb support targets"),
    BodyAction.ROTATE: _body_entry("cartwheel, floor roll and airborne spin"),
    BodyAction.POSE: _body_entry(
        "lunge, single-leg, sit-up, crawl, push-up and horizontal support"
    ),
}


__all__ = [
    "BODY_ENTRIES",
    "FullBodyPass",
    "commanded_root_yaw_rad",
    "full_body_metrics",
]
