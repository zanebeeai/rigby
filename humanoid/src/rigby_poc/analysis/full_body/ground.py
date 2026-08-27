"""Ground plane, per-frame toe clearance, and the airborne frame count.

The ground plane is the lowest toe of the rig standing in its neutral pose, not
``y = 0``: the source rig's rest pose sits slightly above the origin, so every
clearance metric is measured against that offset. ``AnalysisContext`` derives it
once and shares it, which is why the two blocks below read it rather than
rebuilding the neutral pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...models import Hand
from .selectors import FullBodyPass


def ground_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    ground_height = fb.ground_height
    toe_clearances = fb.toe_clearances

    metrics["ground_height_m"] = ground_height
    metrics["ground_penetration_m"] = max(
        0.0,
        -min(float(np.min(values)) for values in toe_clearances.values()),
    )
    metrics["maximum_foot_clearance_m"] = max(
        float(np.max(values)) for values in toe_clearances.values()
    )


def airborne_metrics(fb: FullBodyPass, metrics: dict[str, Any]) -> None:
    toe_clearances = fb.toe_clearances

    metrics["airborne_frame_count"] = int(
        sum(
            left > 0.028 and right > 0.028
            for left, right in zip(
                toe_clearances[Hand.LEFT.value],
                toe_clearances[Hand.RIGHT.value],
                strict=True,
            )
        )
    )


__all__ = ["airborne_metrics", "ground_metrics"]
