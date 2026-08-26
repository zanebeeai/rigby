"""Enumerate the primitives a body affords.

The count is a product, not a list: every schema the robot's measured
capabilities admit, crossed with every degree of remove that means something for
it. That is the payoff of keeping Path and Manner in separate slots -- a closed
inventory of two dozen schemas spans a library of a hundred-odd primitives per
robot without anyone authoring a single one of them.

Manner is deliberately *not* part of the cross product. A primitive is baked at
neutral manner and modulated at bind time, because manner is a co-event that
rides on a path rather than a different path. Multiplying it in would inflate the
bake by a factor of two thousand and store the same motion over and over at
different speeds.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import RobotMorphologyV1
from ..schema.inventory import SchemaEntry, SchemaInventory, afforded_entries
from ..schema.program import (
    BoundaryCondition,
    Dimensionality,
    MannerV1,
    MotionSchemaProgramV1,
    RegionV1,
    Remove,
    RoleBindingV1,
    SegmentV1,
    Deixis,
    ReferenceFrame,
)


# ``coincident`` means "at the Ground itself". With the chain's own root as
# Ground that is a point inside the robot, so it is excluded rather than baked
# and immediately failed.
BAKED_REMOVES: tuple[Remove, ...] = (
    Remove.ADJACENT,
    Remove.PROXIMAL,
    Remove.MEDIAL,
    Remove.DISTAL,
)


@dataclass(frozen=True, slots=True)
class BindingCandidate:
    """One primitive to attempt: a schema, a region, and nothing else."""

    entry_id: str
    schema_key: str
    segment_key: str
    remove: Remove
    program: MotionSchemaProgramV1

    @property
    def label(self) -> str:
        return f"{self.entry_id}@{self.remove.value}"


def _frame_for(entry: SchemaEntry) -> ReferenceFrame:
    """Deictic schemas need a frame with a viewpoint; the rest read absolutely."""

    deixis = entry.schema.deixis
    if deixis is not None and deixis is not Deixis.NEUTRAL:
        return ReferenceFrame.INTRINSIC
    return ReferenceFrame.ABSOLUTE


def _boundary_for(entry: SchemaEntry) -> BoundaryCondition:
    contour = entry.schema.contour
    if contour is not None and contour.value == "oscillating":
        return BoundaryCondition.DIRECTION_REVERSED
    if entry.schema.stative is not None:
        return BoundaryCondition.DWELL
    return BoundaryCondition.TERMINUS


def build_candidate(entry: SchemaEntry, remove: Remove) -> BindingCandidate:
    segment = SegmentV1(
        segment_id="primitive",
        motion_schema=entry.schema,
        figure=RoleBindingV1(role=entry.figure_role),
        ground=RoleBindingV1(role=entry.ground_role),
        region=RegionV1(remove=remove, dimensionality=Dimensionality.POINT),
        frame=_frame_for(entry),
        manner=MannerV1(),
        boundary=_boundary_for(entry),
    )
    program = MotionSchemaProgramV1(
        program_id=f"primitive-{entry.entry_id}-{remove.value}",
        source_text=f"{entry.gloss}, {remove.value}",
        segments=(segment,),
    )
    return BindingCandidate(
        entry_id=entry.entry_id,
        schema_key=entry.canonical_key,
        segment_key=segment.canonical_key,
        remove=remove,
        program=program,
    )


def enumerate_bindings(
    inventory: SchemaInventory,
    morphology: RobotMorphologyV1,
    *,
    contact_scene: bool = False,
    removes: tuple[Remove, ...] = BAKED_REMOVES,
) -> tuple[BindingCandidate, ...]:
    """Every primitive this robot could hold, in a stable order.

    Sorted so two bakes of the same robot attempt the same things in the same
    sequence, which is what makes a partially completed bake comparable to a
    complete one.
    """

    entries = afforded_entries(inventory, morphology, contact_scene=contact_scene)
    candidates = [
        build_candidate(entry, remove)
        for entry in sorted(entries, key=lambda item: item.entry_id)
        for remove in removes
    ]
    return tuple(candidates)


def afforded_schema_ids(
    inventory: SchemaInventory,
    morphology: RobotMorphologyV1,
    *,
    contact_scene: bool = False,
) -> frozenset[str]:
    return frozenset(
        entry.entry_id
        for entry in afforded_entries(
            inventory, morphology, contact_scene=contact_scene
        )
    )
