"""Planning templates: how to break a task into things that must become true.

The planner recognises what is being asked, loads the template for it, and turns
each stage into a Talmy product carrying a goal the body can sense. Execution
stays with the per-frame selector.

That split is the point. A template says *what must become true and in what
order* -- never which primitive to run at 1.4 seconds -- so a new skill teaches
planning without touching the controller, and skills become something you can
write down and upload rather than code you have to add.

It also removes the last piece of task knowledge that was hiding in Python. The
grab decomposition used to live in ``closed_loop.decompose`` as four hand-written
steps with bespoke error functions, which meant the only way to teach a second
skill was to write a second function.

GOAL KINDS

Each is a predicate over :class:`~rigby_poc.closed_loop.Sensing`, plus a
continuous error the selector minimises toward it. They are deliberately few:
a goal a body cannot sense is a goal nothing can verify, which is how a hand
closing on empty air came to be reported as a grasp.

``hand_within``            the fingertips are within a distance of the object
``aperture_encapsulates``  the opening stands square to the palm AND contains it
``opposition_pairs``       N thumb-finger pairs loaded on opposing faces
``figure_raised``          the OBJECT is higher than it started
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_SKILLS = Path(__file__).resolve().parents[2] / "config" / "skills"


class SkillError(ValueError):
    """A template is missing, malformed, or names a goal nothing can sense."""


#: Every goal kind a template may name. A template naming anything else is
#: rejected at load rather than silently producing a stage that never completes.
GOAL_KINDS = (
    "hand_within",
    "aperture_encapsulates",
    # A SHAPE, which can be approached, rather than a force reading that only
    # appears once the grip already exists. ``opposition_pairs`` never read
    # above zero across fifteen runs: before contact it was a constant, so
    # nothing could work toward it, and after contact the stage was over.
    "thumb_and_finger_touch",
    "palm_presented",
    "object_between_digits",
    "whole_hand_c",
    "thumb_finger_c",
    "opposition_pairs",
    "figure_raised",
)

#: Part groups a stage may act with, resolved per hand at planning time.
PART_GROUPS = ("arm", "digits", "gaze", "torso")


@lru_cache(maxsize=8)
def skill_template(name: str) -> dict[str, Any]:
    """Load one planning template."""
    path = _SKILLS / f"{name}.v1.json"
    if not path.is_file():
        raise SkillError(f"no planning template for {name!r} at {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    stages = document.get("stages")
    if not stages:
        raise SkillError(f"{name} declares no stages")
    for stage in stages:
        if stage.get("goal") not in GOAL_KINDS:
            raise SkillError(
                f"{name}.{stage.get('name')} names goal {stage.get('goal')!r}, "
                f"which nothing can sense; known goals are {list(GOAL_KINDS)}"
            )
        unknown = set(stage.get("acts_with", ())) - set(PART_GROUPS)
        if unknown:
            raise SkillError(
                f"{name}.{stage.get('name')} acts with unknown groups {sorted(unknown)}"
            )
        if not stage.get("note"):
            raise SkillError(
                f"{name}.{stage.get('name')} has no note; a stage whose purpose is "
                "unexplained cannot be reviewed when it stops completing"
            )
    return document


@lru_cache(maxsize=1)
def skill_names() -> tuple[str, ...]:
    if not _SKILLS.is_dir():
        return ()
    return tuple(sorted(p.stem.removesuffix(".v1") for p in _SKILLS.glob("*.v1.json")))


def resolve(text: str) -> str | None:
    """Which skill a command asks for, longest alias winning."""
    lowered = f" {text.lower().strip()} "
    best: tuple[int, str] | None = None
    for name in skill_names():
        for alias in skill_template(name).get("aliases", []):
            if f" {alias} " in lowered or lowered.strip().startswith(alias):
                if best is None or len(alias) > best[0]:
                    best = (len(alias), name)
    return best[1] if best else None


def aperture_orthogonality_deg(bones: dict[str, Any], side: str) -> float:
    """How square the thumb-finger apertures stand to the palm, in degrees.

    90 is an opening the object can sit inside; 0 is an aperture lying flat in
    the palm, where "containing" the object means resting against it.

    This is the quantity the whole grasp turned on and nothing was measuring: at
    the authored fist the four apertures stand at 19, 19, 25 and 63 degrees, and
    no wrist placement changes that -- the quads and the palm are both built
    from the hand, so moving the wrist turns both together. Only the thumb can
    change it, which is why the template asks for the thumb to come inward
    before anything closes.
    """
    from .grasp_aperture import _palm_normal, aperture_quads

    normal = _palm_normal(bones, side)
    angles: list[float] = []
    for quad in aperture_quads(bones, side).values():
        corners = quad.corners
        face = np.cross(corners[1] - corners[0], corners[3] - corners[0])
        size = float(np.linalg.norm(face))
        if size < 1e-12:
            continue
        cosine = abs(float(np.dot(face / size, normal)))
        angles.append(float(np.degrees(np.arccos(min(1.0, cosine)))))
    return float(np.mean(angles)) if angles else 0.0


def opposition_pairs(
    contact_force_n: dict[str, float], min_force_n: float
) -> int:
    """How many thumb-finger pairs are loaded at once.

    A pair is the thumb and one finger both carrying load, which is the shape a
    grasp has to have: the thumb works against the fingers, and a finger loaded
    without it is pressing the object into free space.
    """
    thumb = contact_force_n.get("thumb", 0.0) >= min_force_n
    if not thumb:
        return 0
    return sum(
        1
        for finger in ("index", "middle", "ring", "little")
        if contact_force_n.get(finger, 0.0) >= min_force_n
    )
