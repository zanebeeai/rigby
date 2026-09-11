"""Bind a schema program to one robot's certified primitives.

This is the prompt-time counterpart of the bake. The planner has said what to do
in body-neutral terms; the binder decides whether *this* robot has a certified
primitive for each segment, and answers with either a motion or a typed refusal
naming the reason.

Retrieval is exact-match on the segment key. A schema binding is a discrete
symbol -- a Path, a Figure role, a Ground role, a region, a frame -- so looking
one up is a dictionary lookup, not a nearest-neighbour search over an embedding.
That is worth stating because it is easy to reach for a vector store by reflex
here, and it would add a model dependency, a similarity threshold, and a class of
silent near-miss failures, all to answer a question that has an exact answer.

When a segment has no certified primitive the binder consults the recorded
failures from the bake. Having measured, once, exactly why this body could not
do that thing is what lets the answer be "your arm cannot reach that far" rather
than "unsupported".
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from ..contracts import RobotAssetManifestV1
from ..errors import GeneralFailureCode, RigbyGeneralError
from ..grounding import GroundedProgram, ground
from ..primitives.library import (
    BindingFailure,
    PrimitiveRecord,
    refusal_reason,
    retrieve,
    retrieve_by_schema,
)
from ..schema.inventory import SchemaInventory
from ..schema.program import MotionSchemaProgramV1, Remove, SegmentV1


@dataclass(frozen=True, slots=True)
class SegmentBinding:
    segment_id: str
    segment_key: str
    entry_id: str
    record: PrimitiveRecord | None
    substituted: bool = False
    """True when a neighbouring region stood in for the one that was asked for."""


@dataclass(frozen=True, slots=True)
class BoundMotion:
    """A grounded, ready-to-compile motion plus how it was arrived at."""

    grounded: GroundedProgram
    bindings: tuple[SegmentBinding, ...]
    robot_id: str
    program: MotionSchemaProgramV1
    """The schema program as *bound*, which may differ from the one planned when a
    neighbouring region stood in. This is the one that was grounded."""

    @property
    def used_certified_primitives(self) -> int:
        return sum(1 for binding in self.bindings if binding.record is not None)

    @property
    def substitutions(self) -> int:
        return sum(1 for binding in self.bindings if binding.substituted)


class UnbindableSegment(RigbyGeneralError):
    """No certified primitive covers this segment, and here is why."""

    def __init__(
        self,
        robot_id: str,
        segment: SegmentV1,
        failure: BindingFailure | None,
    ) -> None:
        if failure is not None:
            detail = (
                f"{robot_id} has no certified {failure.entry_id!r} at "
                f"{failure.remove}: it failed at the {failure.stage.value} stage "
                f"-- {failure.detail}"
            )
            details = {
                "segment_key": segment.canonical_key,
                "stage": failure.stage.value,
                "failure_code": failure.failure_code,
                "failed_gate": failure.failed_gate,
                "measurements": failure.measurements,
            }
        else:
            detail = (
                f"{robot_id} has no certified primitive for "
                f"{segment.canonical_key!r}, and none was attempted"
            )
            details = {"segment_key": segment.canonical_key}
        super().__init__(GeneralFailureCode.UNAFFORDED_SCHEMA, detail, details=details)


def bind(
    schema_program: MotionSchemaProgramV1,
    manifest: RobotAssetManifestV1,
    model: mujoco.MjModel,
    inventory: SchemaInventory,
    records: tuple[PrimitiveRecord, ...],
    failures: tuple[BindingFailure, ...],
    *,
    allow_region_substitution: bool = True,
    seed: int = 0,
) -> BoundMotion:
    """Check every segment against the library, then ground the whole program.

    Each segment must be covered before anything is grounded. Grounding first and
    discovering halfway through that the third segment is impossible would mean
    reporting a failure about a motion that was already partly built.
    """

    bindings: list[SegmentBinding] = []
    rewritten: list[SegmentV1] = []
    for segment in schema_program.segments:
        entry = inventory.by_binding_key(
            f"{segment.motion_schema.canonical_key}"
            f"|{segment.figure.role.value}->{segment.ground.role.value}"
        )
        record = retrieve(records, segment.canonical_key, manner=segment.manner)
        substituted = False

        if record is None and allow_region_substitution:
            # The schema certified at some other remove. A reach that is proven
            # at medium range and asked for at long range is a better answer than
            # a refusal -- but it is a *different* motion from the one requested,
            # so it is reported rather than passed off as an exact match.
            nearby = retrieve_by_schema(records, segment.motion_schema.canonical_key)
            candidates = [
                item
                for item in nearby
                if item.figure_role == segment.figure.role.value
                and item.ground_role == segment.ground.role.value
            ]
            if candidates:
                record = candidates[-1]
                substituted = True

        if record is None:
            raise UnbindableSegment(
                manifest.rig_id,
                segment,
                refusal_reason(failures, segment.canonical_key),
            )

        bindings.append(
            SegmentBinding(
                segment_id=segment.segment_id,
                segment_key=segment.canonical_key,
                entry_id=entry.entry_id,
                record=record,
                substituted=substituted,
            )
        )

        # When a region was substituted, the segment has to be rewritten to the
        # one that actually certified. Grounding the region originally asked for
        # while reporting the neighbour as the primitive used would be a plain
        # falsehood: the motion performed would be the uncertified one, and the
        # certified label would be attached to something else entirely.
        if substituted:
            segment = segment.model_copy(
                update={
                    "region": segment.region.model_copy(
                        update={"remove": Remove(record.remove)}
                    )
                }
            )
        rewritten.append(segment)

    bound_program = schema_program.model_copy(update={"segments": tuple(rewritten)})
    # Ground at the pace the library actually certified at. A primitive that only
    # passed after the bake slowed it down is a primitive whose certificate is
    # *about* the slower version, and re-grounding it at the nominal pace answers
    # with a motion nothing ever certified -- it fails the same velocity gate the
    # bake already worked around, which is how a certified gesture came back
    # refused. One scale covers the program because grounding applies one.
    pace = max(
        (
            float(binding.record.measurements.get("duration_scale", 1.0))
            for binding in bindings
            if binding.record is not None
        ),
        default=1.0,
    )
    grounded = ground(
        bound_program,
        manifest,
        model,
        inventory,
        seed=seed,
        duration_scale=max(pace, 1.0),
    )
    return BoundMotion(
        grounded=grounded,
        bindings=tuple(bindings),
        robot_id=manifest.rig_id,
        program=bound_program,
    )


def coverage_report(
    records: tuple[PrimitiveRecord, ...], failures: tuple[BindingFailure, ...]
) -> dict[str, object]:
    """What this robot can and cannot do, for the operator and for the planner."""

    certified: dict[str, list[str]] = {}
    for record in records:
        certified.setdefault(record.entry_id, []).append(record.remove)

    refused: dict[str, list[str]] = {}
    for failure in failures:
        if failure.entry_id in certified:
            continue
        refused.setdefault(failure.entry_id, []).append(
            failure.failed_gate or failure.failure_code
        )

    return {
        "certified": {key: sorted(value) for key, value in sorted(certified.items())},
        "refused": {key: sorted(set(value)) for key, value in sorted(refused.items())},
        "certified_count": len(records),
        "schema_count": len(certified),
    }
