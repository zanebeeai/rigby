"""Opt-in compile behaviours, switched by environment variable.

Two of the closed-loop branch's compiler changes alter every clip they touch:
the angular rate limiter re-times motion, and the grab gaze re-aims the neck
and head each frame. Either one changes corpus digests, so landing them on
by default would have required a re-bless on every platform at the moment
of merge, with nobody able to review the new motion. They land off. Set the
variable to ``1`` to turn one on, re-bless the corpus, and commit the two
together; that is a change of its own with its own review.

    RIGBY_ANGULAR_RATE_LIMIT=1   bound joint speed after authoring (not strikes)
    RIGBY_GRAB_GAZE=1            aim neck and head at the object during a grab
"""

from __future__ import annotations

import os


def _on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def angular_rate_limit_enabled() -> bool:
    return _on("RIGBY_ANGULAR_RATE_LIMIT")


def grab_gaze_enabled() -> bool:
    return _on("RIGBY_GRAB_GAZE")
