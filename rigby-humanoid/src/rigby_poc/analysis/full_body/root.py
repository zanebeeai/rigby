"""Root travel, height envelope, and the commanded turn total.

Three of these four keys are plain observations of the hips track. The fourth,
``final_root_yaw_deg``, is *commanded* rather than achieved: the compiler
accumulates it from the authored turn primitives and never reads it back off the
rig. Plan 02 §1.4 listed it as unrecoverable, but it is a pure function of the
program — see :func:`commanded_root_yaw_rad`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...models import BodyAction, BodyTarget, MotionProgram, PrimitiveKind
from .selectors import FullBodyPass


def commanded_root_yaw_rad(program: MotionProgram) -> float:
    """Reproduce the compiler's accumulated turn intent, exactly.

    ``_compile_full_body`` carries a ``current_yaw`` accumulator through its
    generation loop and advances it by ``radians(turn_degrees)`` on every TURN
    primitive that is not a recovery. Nothing else writes it, and it starts at
    zero, so the total is a fold over the program in primitive order. The loop
    below is that fold with the same operations in the same order, which is what
    makes it bit-identical rather than merely close.

    Reproducing it beats persisting it: a mutation that rewrites the authored
    turn is then visible to the analyzer, where a carried-over number would make
    the metric an echo of the request.
    """

    current_yaw = 0.0
    for primitive in program.primitives:
        target = primitive.body or BodyTarget(
            action=BodyAction.HOLD,
            cycles=0.0,
            intensity=0.0,
        )
        if target.action == BodyAction.TURN and primitive.kind != PrimitiveKind.RECOVER:
            current_yaw = current_yaw + math.radians(target.turn_degrees)
    return current_yaw


def root_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    hips_positions = fb.hips_positions
    current_yaw = commanded_root_yaw_rad(fb.program)

    segment_lengths = np.linalg.norm(
        np.diff(hips_positions[:, [0, 2]], axis=0),
        axis=1,
    )
    metrics["root_path_length_m"] = float(np.sum(segment_lengths))
    metrics["root_displacement_m"] = float(
        np.linalg.norm(
            hips_positions[-1, [0, 2]] - hips_positions[0, [0, 2]]
        )
    )
    metrics["root_vertical_min_m"] = float(np.min(hips_positions[:, 1]))
    metrics["root_vertical_max_m"] = float(np.max(hips_positions[:, 1]))
    metrics["final_root_yaw_deg"] = math.degrees(current_yaw)


__all__ = ["commanded_root_yaw_rad", "root_metrics"]
