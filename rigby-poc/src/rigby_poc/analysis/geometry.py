"""Small pure geometric primitives shared by the checks."""

from __future__ import annotations

import numpy as np


def line_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Return the closest centerline distance between two finite 3D segments."""

    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    aa = float(np.dot(first, first))
    ab = float(np.dot(first, second))
    bb = float(np.dot(second, second))
    ao = float(np.dot(first, offset))
    bo = float(np.dot(second, offset))
    denominator = aa * bb - ab * ab
    epsilon = 1e-10
    if denominator < epsilon:
        first_alpha = 0.0
        second_alpha = float(np.clip(bo / max(bb, epsilon), 0.0, 1.0))
    else:
        first_alpha = float(
            np.clip((ab * bo - bb * ao) / denominator, 0.0, 1.0)
        )
        second_alpha = float(
            np.clip((aa * bo - ab * ao) / denominator, 0.0, 1.0)
        )
        # Clamping one parameter changes the optimum of the other. One
        # coordinate-descent refinement is exact for the remaining segment.
        first_alpha = float(
            np.clip(
                (ab * second_alpha - ao) / max(aa, epsilon),
                0.0,
                1.0,
            )
        )
        second_alpha = float(
            np.clip(
                (ab * first_alpha + bo) / max(bb, epsilon),
                0.0,
                1.0,
            )
        )
    delta = (
        offset + first_alpha * first - second_alpha * second
    )
    return float(np.linalg.norm(delta))
