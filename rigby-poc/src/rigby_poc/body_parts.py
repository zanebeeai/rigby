"""A movement vocabulary written in the units of the range-of-motion document.

``config/rom.v1.json`` says how far each of the rig's 156 DOFs may rotate. It
does not say what any of them are *for*: there is no vocabulary of named
operations, so a planner has no finite action space per body part and the
capability description that does exist -- ``FINGERS``, ``SEGMENTS``,
``HAND_SHAPES`` -- is a wall of Python literals with no unit and no source.

``config/body_parts.v1.json`` supplies the missing half, and the design decision
that makes the two halves agree is that **a move is authored as a signed
fraction of a DOF's own ROM range, never as an angle**. ``{"flexion": 0.6}``
means "60% of the way to this joint's typical flexion limit". A move therefore
cannot name an out-of-range pose -- not because a check rejects it, but because
the vocabulary is written in the envelope's units and has no way to express one.

The rotations come out through :func:`analysis.anatomy.frame.compose`, the same
anatomical frames ``rom_checks`` decomposes with, so a pose authored here and a
pose measured there are talking about the same axes.

Two structures sit on top of the per-part moves:

``move_sets``
    Every part's finite, named vocabulary. All three DOFs of every part are
    reachable, so this spans the rig's articulation rather than a convenient
    corner of it.

``sequences``
    A grouped action as a **timed** composition, not a single pose. ``grip`` is
    the reason the structure exists: a real grasp is thumb-then-fingers, and
    collapsing it to one keyframe is what lets index and middle reach a free
    object first and push it out of the hand before the thumb arrives.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .analysis.anatomy.frame import AnatomicalFrame, DofAngles, all_frames, compose
from .analysis.anatomy.rom import rom_limit
from .models import Quat

CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "body_parts.v1.json"
)

_DOFS = ("flexion", "abduction", "twist")


class BodyPartError(LookupError):
    """A part, move or sequence the catalog does not define."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid body-part catalog: {message}")


def _validate(document: dict[str, Any]) -> None:
    parts = document["parts"]
    move_sets = document["move_sets"]
    _require(set(parts) == set(move_sets), "every part needs exactly one move set")

    frames = all_frames()
    for name, part in parts.items():
        parent = part["parent"]
        _require(
            parent is None or parent in parts,
            f"{name!r} has unknown parent {parent!r}",
        )
        for bone in part["bones"]:
            _require(bone in frames, f"{name!r} names bone {bone!r} with no frame")
            for dof in _DOFS:
                # Raises RomError if the pair is absent, which is the point: a
                # part cannot claim a DOF the range-of-motion document does not
                # carry a limit for.
                rom_limit(bone, dof)

        moves = move_sets[name]
        _require(bool(moves), f"{name!r} has an empty move set")
        _require("hold" in moves, f"{name!r} has no hold move")
        reachable: set[str] = set()
        for move, spec in moves.items():
            for bone, dofs in spec.items():
                _require(
                    bone in part["bones"],
                    f"{name}.{move} drives {bone!r}, which is not one of its bones",
                )
                for dof, value in dofs.items():
                    _require(dof in _DOFS, f"{name}.{move} names unknown dof {dof!r}")
                    _require(
                        isinstance(value, (int, float)) and -1.0 <= value <= 1.0,
                        f"{name}.{move}.{dof} is {value!r}; moves are fractions in [-1, 1]",
                    )
                    reachable.add(dof)
        _require(
            reachable == set(_DOFS),
            f"{name!r} never reaches {sorted(set(_DOFS) - reachable)}; "
            "a part whose vocabulary cannot address a DOF has that DOF only in theory",
        )

    for name, sequence in document["sequences"].items():
        steps = sequence["steps"]
        _require(bool(steps), f"sequence {name!r} has no steps")
        times = [float(step["at"]) for step in steps]
        _require(
            times == sorted(times) and times[-1] == 1.0,
            f"sequence {name!r} must be ordered and end at 1.0",
        )
        for step in steps:
            for part, move in step["moves"].items():
                _require(part in parts, f"sequence {name!r} names unknown part {part!r}")
                _require(
                    move in move_sets[part],
                    f"sequence {name!r} gives {part!r} undefined move {move!r}",
                )


@lru_cache(maxsize=1)
def body_part_catalog() -> dict[str, Any]:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    _validate(document)
    return document


def parts() -> dict[str, Any]:
    return body_part_catalog()["parts"]


def moves_for(part: str) -> dict[str, Any]:
    catalog = body_part_catalog()
    if part not in catalog["move_sets"]:
        raise BodyPartError(f"no such body part: {part}")
    return catalog["move_sets"][part]


def move_angles_deg(
    part: str, move: str, amplitude: float = 1.0
) -> dict[str, dict[str, float]]:
    """Resolve one move to anatomical degrees per bone, scaled by ``amplitude``.

    A positive fraction scales the DOF's typical upper bound and a negative one
    its lower bound, so the result is inside the envelope by construction. The
    scaled fraction is clamped to [-1, 1] for the same reason: an amplitude may
    ask for more of a move than the move names, but never for more than the
    range of motion allows.

    **Why moves have a magnitude at all.** Without one, each move is a fixed
    pose and a part's reachable set is the handful of poses somebody wrote down.
    Measured on the arm: over all 990 combinations of shoulder, elbow and wrist
    moves, the closest the hand could bring its fingertips to a block on the
    table was 17.4 cm. Not close enough to grasp, and no selector can fix that
    -- a perfect chooser picking perfectly from 990 unreachable poses is still
    17.4 cm short. Letting the same 990 combinations carry a magnitude brings
    the best within 2.4 cm, because the reachable set stops being a lattice.

    ``amplitude`` of 1.0 reproduces the authored move exactly, so every existing
    caller is unaffected.
    """
    moves = moves_for(part)
    if move not in moves:
        raise BodyPartError(f"{part} has no move {move!r}")
    resolved: dict[str, dict[str, float]] = {}
    for bone, dofs in moves[move].items():
        angles: dict[str, float] = {}
        for dof, fraction in dofs.items():
            low, high = rom_limit(bone, dof).typical_deg
            scaled = max(-1.0, min(1.0, float(fraction) * float(amplitude)))
            angles[dof] = scaled * (high if scaled >= 0 else -low)
        resolved[bone] = angles
    return resolved


def _quaternion(angles: dict[str, float], frame: AnatomicalFrame) -> Quat:
    xyzw = compose(
        DofAngles(
            flexion_rad=math.radians(angles.get("flexion", 0.0)),
            abduction_rad=math.radians(angles.get("abduction", 0.0)),
            twist_rad=math.radians(angles.get("twist", 0.0)),
        ),
        frame,
    )
    return Quat(
        x=float(xyzw[0]), y=float(xyzw[1]), z=float(xyzw[2]), w=float(xyzw[3])
    )


def move_rotations(part: str, move: str, amplitude: float = 1.0) -> dict[str, Quat]:
    """One move as local delta rotations, ready to drop into a clip frame."""
    frames = all_frames()
    return {
        bone: _quaternion(angles, frames[bone])
        for bone, angles in move_angles_deg(part, move).items()
    }


def sequence_angles_deg(name: str, progress: float) -> dict[str, dict[str, float]]:
    """A timed sequence sampled at ``progress`` in ``[0, 1]``.

    Steps are interpolated in **angle** space rather than quaternion space. The
    catalog's authored value for a DOF is a scalar, so interpolating it is
    exact; going through quaternions first and blending those would make the
    midpoint depend on the composition order of three axes.
    """
    catalog = body_part_catalog()
    if name not in catalog["sequences"]:
        raise BodyPartError(f"no such sequence: {name}")
    steps = catalog["sequences"][name]["steps"]
    alpha = float(np.clip(progress, 0.0, 1.0))

    upper = next((i for i, s in enumerate(steps) if float(s["at"]) >= alpha), len(steps) - 1)
    lower = max(0, upper - 1)
    span = float(steps[upper]["at"]) - float(steps[lower]["at"])
    blend = 0.0 if span <= 1e-12 else (alpha - float(steps[lower]["at"])) / span

    def angles_of(step: dict[str, Any]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for part, move in step["moves"].items():
            for bone, angles in move_angles_deg(part, move).items():
                out[bone] = dict(angles)
        return out

    first, second = angles_of(steps[lower]), angles_of(steps[upper])
    resolved: dict[str, dict[str, float]] = {}
    for bone in set(first) | set(second):
        a, b = first.get(bone, {}), second.get(bone, {})
        resolved[bone] = {
            dof: (1.0 - blend) * a.get(dof, 0.0) + blend * b.get(dof, 0.0)
            for dof in set(a) | set(b)
        }
    return resolved


def sequence_rotations(name: str, progress: float) -> dict[str, Quat]:
    frames = all_frames()
    return {
        bone: _quaternion(angles, frames[bone])
        for bone, angles in sequence_angles_deg(name, progress).items()
    }


def sequence_names() -> tuple[str, ...]:
    return tuple(body_part_catalog()["sequences"])
