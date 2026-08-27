"""Finger curl read back off a finished pose.

Pure, and misfiled in ``compiler.py`` until 02c: it reads one frame's bone
rotations and normalises each digit's flexion against the range that digit can
reach. No generation state, no scene, no program.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..models import ClipFrame


# The segment whose flexion stands in for the whole digit. The thumb's
# metacarpal carries its opposition, so it is measured there rather than at a
# proximal phalanx, and it saturates sooner -- hence the smaller divisor below.
_CURL_SEGMENT = {
    "thumb": "ThumbMetacarpal",
    "index": "IndexProximal",
    "middle": "MiddleProximal",
    "ring": "RingProximal",
    "little": "LittleProximal",
}


def curl_values_from_frame(frame: ClipFrame, hand_value: str) -> dict[str, float]:
    """Normalised 0..1 curl per digit for one hand in one frame."""

    segment = _CURL_SEGMENT
    result: dict[str, float] = {}
    for digit, suffix in segment.items():
        quat = frame.bones[f"{hand_value}{suffix}"].rotation
        curl_angle = abs(float(Rotation.from_quat(quat.as_list()).as_euler("xyz")[0]))
        result[digit] = float(
            np.clip(curl_angle / (0.95 if digit == "thumb" else 1.15), 0.0, 1.0)
        )
    return result


__all__ = ["curl_values_from_frame"]
