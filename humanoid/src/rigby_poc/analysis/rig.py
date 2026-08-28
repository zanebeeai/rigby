"""Rig-profile access shared by every analysis check.

These used to live in ``compiler.py``. They are pure reads of committed
configuration and carry no generation state, so the analysis layer owns them
and the compiler imports them back.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..models import BonePose, Quat

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RIG_PROFILE = PROJECT_ROOT / "config" / "rig_profiles" / "mesh2motion-human-vrm1.json"

# Neutral egocentric gaze direction of the calibrated head, matching
# ``frontend/src/camera.ts``.
EGO_NEUTRAL_GAZE = np.asarray([0.0, -0.65, 1.0], dtype=float)
EGO_NEUTRAL_GAZE /= np.linalg.norm(EGO_NEUTRAL_GAZE)

# Head-local offset the renderer hangs the ego camera off, matching
# ``frontend/src/camera.ts`` NEUTRAL_EYE_OFFSET -- which now imports it from
# ``config/camera.v1.json`` rather than retyping it, as this copy does.
# ``test_camera_config.py`` pins both against the config so they cannot drift.
#
# This offset composes onto the rig's *measured* rest head position. The value it
# replaced folded the offset into a rest head rounded to four decimals, which put
# the implied offset 0.024 mm from the renderer's -- 0.028 px of a 1600x900
# capture, and enough to flip ``active_hand_visibility_fraction`` at frame 0 of
# every strike and gesture clip.
EGO_EYE_OFFSET_M = np.asarray([0.0, 0.04, 0.11], dtype=float)


def rig_profile() -> dict[str, Any]:
    return json.loads(RIG_PROFILE.read_text(encoding="utf-8"))


def identity_pose() -> dict[str, Quat]:
    return {key: Quat() for key in rig_profile()["bone_map"]}


def identity_bones() -> dict[str, BonePose]:
    return {name: BonePose(rotation=value) for name, value in identity_pose().items()}


@lru_cache(maxsize=1)
def canonical_bone_names() -> tuple[str, ...]:
    """Every canonical bone the rig profile maps, in profile order."""

    return tuple(rig_profile()["bone_map"])
