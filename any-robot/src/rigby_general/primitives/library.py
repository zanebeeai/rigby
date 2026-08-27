"""The per-robot certified primitive library, and its refusal boundary.

Two things live here, and the second is as much a product as the first.

A **certified primitive** is a grounded schema binding that compiled, simulated,
and passed every gate for its morphology class. Retrieval over these is
exact-match on a discrete key, not nearest-neighbour over an embedding: a schema
binding *is* a symbol, and looking it up needs no model. That also sidesteps the
blocked Cosmos-Embed1 dependency entirely for the core path -- vector search
belongs where meanings are fuzzy, which is prompt text, not schema keys.

A **binding failure** is a primitive the robot could not perform, recorded with
the stage it died at and the gate it broke. The set of them is the robot's
declared refusal boundary, and the planner consults it so that asking for
something this body cannot do returns a typed refusal naming the reason rather
than a fabricated motion. The bake produces the capability and its limits in one
pass; storing only the successes would throw away half of what was learned.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rigby_core.hashing import content_hash

from ..schema.program import MannerV1


class BakeStage(StrEnum):
    """Where a binding attempt died. Ordered by how early the failure was."""

    AFFORDANCE = "affordance"
    GROUNDING = "grounding"
    COMPILATION = "compilation"
    SIMULATION = "simulation"
    CERTIFICATION = "certification"


@dataclass(frozen=True, slots=True)
class PrimitiveRecord:
    """One certified primitive."""

    robot_id: str
    entry_id: str
    schema_key: str
    segment_key: str
    remove: str
    figure_role: str
    ground_role: str
    figure_site: str
    duration_s: float
    waypoints: int
    program: dict[str, Any]
    program_sha256: str
    trace_sha256: str
    measurements: dict[str, float]
    inventory_sha256: str
    base_tree_sha256: str

    @property
    def record_id(self) -> str:
        return content_hash(
            {"robot": self.robot_id, "segment": self.segment_key}
        )[:32]

    def to_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "robot_id": self.robot_id,
            "entry_id": self.entry_id,
            "schema_key": self.schema_key,
            "segment_key": self.segment_key,
            "remove": self.remove,
            "figure_role": self.figure_role,
            "ground_role": self.ground_role,
            "figure_site": self.figure_site,
            "duration_s": self.duration_s,
            "waypoints": self.waypoints,
            "program": self.program,
            "program_sha256": self.program_sha256,
            "trace_sha256": self.trace_sha256,
            "measurements": self.measurements,
            "inventory_sha256": self.inventory_sha256,
            "base_tree_sha256": self.base_tree_sha256,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PrimitiveRecord":
        fields = {
            key: payload[key]
            for key in (
                "robot_id",
                "entry_id",
                "schema_key",
                "segment_key",
                "remove",
                "figure_role",
                "ground_role",
                "figure_site",
                "duration_s",
                "waypoints",
                "program",
                "program_sha256",
                "trace_sha256",
                "measurements",
                "inventory_sha256",
                "base_tree_sha256",
            )
        }
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class BindingFailure:
    """A primitive this robot cannot perform, and why."""

    robot_id: str
    entry_id: str
    schema_key: str
    segment_key: str
    remove: str
    stage: BakeStage
    failure_code: str
    detail: str
    failed_gate: str | None = None
    measurements: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "entry_id": self.entry_id,
            "schema_key": self.schema_key,
            "segment_key": self.segment_key,
            "remove": self.remove,
            "stage": self.stage.value,
            "failure_code": self.failure_code,
            "detail": self.detail,
            "failed_gate": self.failed_gate,
            "measurements": self.measurements,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BindingFailure":
        return cls(
            robot_id=payload["robot_id"],
            entry_id=payload["entry_id"],
            schema_key=payload["schema_key"],
            segment_key=payload["segment_key"],
            remove=payload["remove"],
            stage=BakeStage(payload["stage"]),
            failure_code=payload["failure_code"],
            detail=payload["detail"],
            failed_gate=payload.get("failed_gate"),
            measurements=payload.get("measurements", {}),
        )


@dataclass(frozen=True, slots=True)
class BakeSummary:
    robot_id: str
    attempted: int
    certified: int
    elapsed_seconds: float
    budget_seconds: int
    max_attempts: int
    complete: bool
    inventory_sha256: str
    base_tree_sha256: str
    afforded_entry_ids: tuple[str, ...]
    covered_entry_ids: tuple[str, ...]

    @property
    def yield_fraction(self) -> float:
        return self.certified / self.attempted if self.attempted else 0.0

    @property
    def schema_coverage(self) -> float:
        """Fraction of afforded schemas with at least one certified primitive.

        The headline number for requirement ``bake_coverage``: a library that
        certifies many variants of one schema and none of another has not covered
        the robot, however high its raw yield.
        """

        if not self.afforded_entry_ids:
            return 1.0
        return len(set(self.covered_entry_ids)) / len(set(self.afforded_entry_ids))

    def to_json(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "attempted": self.attempted,
            "certified": self.certified,
            "yield_fraction": round(self.yield_fraction, 4),
            "schema_coverage": round(self.schema_coverage, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "budget_seconds": self.budget_seconds,
            "max_attempts": self.max_attempts,
            "complete": self.complete,
            "inventory_sha256": self.inventory_sha256,
            "base_tree_sha256": self.base_tree_sha256,
            "afforded_entry_ids": list(self.afforded_entry_ids),
            "covered_entry_ids": sorted(set(self.covered_entry_ids)),
        }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


class PrimitiveLibrary:
    """Filesystem-backed store, one directory per robot."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def directory(self, robot_id: str) -> Path:
        return self.root / robot_id / "primitives"

    def write(
        self,
        robot_id: str,
        records: list[PrimitiveRecord],
        failures: list[BindingFailure],
        summary: BakeSummary,
    ) -> Path:
        directory = self.directory(robot_id)
        directory.mkdir(parents=True, exist_ok=True)

        _atomic_write(
            directory / "primitives.json",
            json.dumps(
                [record.to_json() for record in records], indent=2, sort_keys=True
            ).encode("utf-8"),
        )
        _atomic_write(
            directory / "failures.json",
            json.dumps(
                [failure.to_json() for failure in failures], indent=2, sort_keys=True
            ).encode("utf-8"),
        )
        _atomic_write(
            directory / "bake.json",
            json.dumps(summary.to_json(), indent=2, sort_keys=True).encode("utf-8"),
        )
        return directory

    def load(self, robot_id: str) -> tuple[PrimitiveRecord, ...]:
        path = self.directory(robot_id) / "primitives.json"
        if not path.is_file():
            return ()
        return tuple(
            PrimitiveRecord.from_json(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        )

    def load_failures(self, robot_id: str) -> tuple[BindingFailure, ...]:
        path = self.directory(robot_id) / "failures.json"
        if not path.is_file():
            return ()
        return tuple(
            BindingFailure.from_json(item)
            for item in json.loads(path.read_text(encoding="utf-8"))
        )

    def load_summary(self, robot_id: str) -> BakeSummary | None:
        path = self.directory(robot_id) / "bake.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return BakeSummary(
            robot_id=payload["robot_id"],
            attempted=payload["attempted"],
            certified=payload["certified"],
            elapsed_seconds=payload["elapsed_seconds"],
            budget_seconds=payload["budget_seconds"],
            max_attempts=payload["max_attempts"],
            complete=payload["complete"],
            inventory_sha256=payload["inventory_sha256"],
            base_tree_sha256=payload["base_tree_sha256"],
            afforded_entry_ids=tuple(payload["afforded_entry_ids"]),
            covered_entry_ids=tuple(payload["covered_entry_ids"]),
        )

    def has_library(self, robot_id: str) -> bool:
        return (self.directory(robot_id) / "primitives.json").is_file()


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def retrieve(
    records: tuple[PrimitiveRecord, ...],
    segment_key: str,
    *,
    manner: MannerV1 | None = None,
) -> PrimitiveRecord | None:
    """Exact match on the segment key. No embedding, no nearest neighbour.

    A schema binding is a discrete symbol, so the right lookup is a dictionary
    one. ``manner`` is accepted for symmetry with the fuzzy case but does not
    affect the match: primitives are baked neutral, and manner is applied when
    the primitive is bound.
    """

    for record in records:
        if record.segment_key == segment_key:
            return record
    return None


def retrieve_by_schema(
    records: tuple[PrimitiveRecord, ...], schema_key: str
) -> tuple[PrimitiveRecord, ...]:
    """Every certified region for one schema, nearest region first."""

    order = {"adjacent": 0, "proximal": 1, "medial": 2, "distal": 3, "coincident": 4}
    matches = [record for record in records if record.schema_key == schema_key]
    return tuple(sorted(matches, key=lambda item: order.get(item.remove, 9)))


def refusal_reason(
    failures: tuple[BindingFailure, ...], segment_key: str
) -> BindingFailure | None:
    """Why this body cannot do that, if it was tried and failed."""

    for failure in failures:
        if failure.segment_key == segment_key:
            return failure
    return None
