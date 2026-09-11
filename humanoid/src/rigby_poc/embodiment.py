"""What a body offers a controller, read from a manifest rather than compiled in.

URDF says where a robot's links and joints are. This says what it can be ASKED
to do and what it can be asked ABOUT -- which is the layer a language model
actually plans in, and the layer that has to change when the robot does. A
different machine ships a different manifest against the same solver names, or
the same manifest against different solvers, and the control loop is untouched.

Two things the manifest carries that code alone kept getting wrong:

AMPLITUDE MEANING. Four separate failures came from a caller assuming the wrong
one. reach_to below 0.7 moved the hand AWAY because its amplitude was a fraction
of a path; close_grip's is a target force; open_grip's is a shape; lift's is a
rate, and at full amplitude it commanded 15 cm in a single decision and tore the
object out of the hand. A number between 0 and 1 does not describe itself.

SENSOR PROVENANCE. Every metric names the instrument that could supply it, so a
quantity nothing could measure cannot quietly become a goal, and a body cannot
be handed knowledge it has no way to obtain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONFIG = Path(__file__).resolve().parents[2] / "config"


class EmbodimentError(ValueError):
    """The manifest is missing, malformed, or describes something unbuildable."""


@dataclass(frozen=True)
class PrimitiveSpec:
    """One thing the body can be asked to do."""

    name: str
    solver: str
    parts: tuple[str, ...]
    amplitude: str
    describes: str
    refuses_when: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricSpec:
    """One thing the body can be asked about."""

    name: str
    wants: float | None
    sensor: str
    describes: str
    targetable: bool = True


@lru_cache(maxsize=4)
def manifest(name: str = "embodiment") -> dict[str, Any]:
    path = _CONFIG / f"{name}.v1.json"
    if not path.is_file():
        raise EmbodimentError(f"no embodiment manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def part_groups(side: str, name: str = "embodiment") -> dict[str, tuple[str, ...]]:
    """The named part groups, resolved for one side of the body."""
    return {
        group: tuple(entry.replace("{side}", side) for entry in members)
        for group, members in manifest(name)["part_groups"].items()
    }


@lru_cache(maxsize=8)
def primitive_specs(side: str, name: str = "embodiment") -> tuple[PrimitiveSpec, ...]:
    """Every primitive the manifest declares, expanded per digit where asked."""
    groups = part_groups(side, name)
    out: list[PrimitiveSpec] = []
    for entry in manifest(name)["primitives"]:
        expand = entry.get("per")
        members = groups.get(expand, ()) if expand else ()
        if expand:
            for part in members:
                digit = part.rsplit("_", 1)[-1]
                out.append(PrimitiveSpec(
                    name=entry["name"].replace("{digit}", digit),
                    solver=entry["solver"],
                    parts=(part,),
                    amplitude=entry["amplitude"],
                    describes=entry["describes"].replace("one digit", f"the {digit}"),
                    refuses_when=tuple(entry.get("refuses_when", ())),
                ))
            continue
        out.append(PrimitiveSpec(
            name=entry["name"],
            solver=entry["solver"],
            parts=groups.get(entry["parts"], ()),
            amplitude=entry["amplitude"],
            describes=entry["describes"],
            refuses_when=tuple(entry.get("refuses_when", ())),
        ))
    return tuple(out)


@lru_cache(maxsize=4)
def metric_specs(name: str = "embodiment") -> tuple[MetricSpec, ...]:
    return tuple(
        MetricSpec(
            name=entry["name"],
            wants=entry.get("wants"),
            sensor=entry["sensor"],
            describes=entry["describes"],
            targetable=bool(entry.get("targetable", True)),
        )
        for entry in manifest(name)["metrics"]
    )


def check(side: str = "right", name: str = "embodiment") -> list[str]:
    """Everything the manifest declares that the code cannot honour.

    Run this after changing either. A manifest that names a solver nothing
    provides, or a metric nothing can read, is a promise to a controller that
    will be broken at the worst possible moment -- and silently, which is how
    most of this project's time was spent.
    """
    from .closed_loop import METRICS, _register_metrics, super_primitives
    from .models import Hand

    _register_metrics()
    hand = Hand.LEFT if side == "left" else Hand.RIGHT
    built = {p.name for p in super_primitives(hand)}
    complaints: list[str] = []

    for spec in primitive_specs(side, name):
        if spec.name not in built:
            complaints.append(
                f"primitive {spec.name!r} is declared but not built")
        if not spec.parts:
            complaints.append(f"primitive {spec.name!r} acts with no parts")
        if not spec.amplitude:
            complaints.append(
                f"primitive {spec.name!r} does not say what its amplitude means")
    for name_ in sorted(built - {s.name for s in primitive_specs(side, name)}):
        complaints.append(f"primitive {name_!r} is built but not declared")

    for spec in metric_specs(name):
        if spec.name not in METRICS:
            complaints.append(f"metric {spec.name!r} is declared but not readable")
        if spec.targetable and spec.wants is None and not spec.name.endswith("_m"):
            complaints.append(
                f"metric {spec.name!r} is targetable but says nothing about what it wants")
    for name_ in sorted(set(METRICS) - {s.name for s in metric_specs(name)}):
        complaints.append(f"metric {name_!r} is readable but not declared")
    return complaints
