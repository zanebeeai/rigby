"""Load and validate the sealed schema inventory.

The inventory is data, not code, and it is hash-pinned exactly as
``benchmark_spec.json`` is in v2. That is deliberate: the closed class is the
system's whole vocabulary, and a silent edit to it would change what every baked
primitive means without changing a line of source.

Affordance is decided here too, and only from measurement. A robot is admitted to
a schema because something on it was observed to close, or because a person
confirmed which way it faces -- never because of what a link is called.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from ..contracts import FrameSource, RobotMorphologyV1
from .program import (
    BindingRole,
    Conformation,
    Contour,
    Deixis,
    PathVector,
    SchemaKind,
    SegmentSchemaV1,
    Stative,
)


INVENTORY_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "general" / "schema_inventory.v1.json"
)


class Requirement(StrEnum):
    GRASPING_EFFECTOR = "grasping_effector"
    TWO_GRASPING_EFFECTORS = "two_grasping_effectors"
    SENSOR = "sensor"
    CONFIRMED_FRAME = "confirmed_frame"
    CONTACT_SCENE = "contact_scene"


@dataclass(frozen=True, slots=True)
class SchemaEntry:
    entry_id: str
    schema: SegmentSchemaV1
    figure_role: BindingRole
    ground_role: BindingRole
    gloss: str
    requires: frozenset[Requirement]

    @property
    def canonical_key(self) -> str:
        return self.schema.canonical_key

    @property
    def binding_key(self) -> str:
        """Schema plus roles: what actually identifies a motion.

        A path schema alone does not. "Reach to a point" and "hand it across"
        share the very same Path -- to a point, straight -- and differ only in
        what the Figure is located against. Talmy's decomposition taken
        literally, which is why the retrieval key carries the roles.
        """

        return (
            f"{self.schema.canonical_key}"
            f"|{self.figure_role.value}->{self.ground_role.value}"
        )

    @property
    def needs_contact(self) -> bool:
        return Requirement.CONTACT_SCENE in self.requires


@dataclass(frozen=True, slots=True)
class SchemaInventory:
    version: str
    inventory_id: str
    sha256: str
    entries: tuple[SchemaEntry, ...]
    remove_fractions: dict[str, float]
    manner_axes: tuple[str, ...]

    def by_id(self, entry_id: str) -> SchemaEntry:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise KeyError(f"unknown schema entry: {entry_id}")

    def by_binding_key(self, binding_key: str) -> SchemaEntry:
        for entry in self.entries:
            if entry.binding_key == binding_key:
                return entry
        raise KeyError(f"unknown schema binding: {binding_key}")

    @property
    def free_space_entries(self) -> tuple[SchemaEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.needs_contact)

    def remove_fraction(self, remove: str) -> float:
        try:
            return self.remove_fractions[remove]
        except KeyError as error:
            raise KeyError(f"no remove fraction for {remove!r}") from error


def capabilities_of(
    morphology: RobotMorphologyV1, *, contact_scene: bool = False
) -> frozenset[Requirement]:
    """What this robot was *measured* to be able to do."""

    capabilities: set[Requirement] = set()
    grasping = morphology.grasping_effectors
    if grasping:
        capabilities.add(Requirement.GRASPING_EFFECTOR)
    if len(grasping) >= 2:
        capabilities.add(Requirement.TWO_GRASPING_EFFECTORS)
    if any(effector.kind.value == "sensor" for effector in morphology.effectors):
        capabilities.add(Requirement.SENSOR)
    if morphology.intrinsic_frame.source is FrameSource.OPERATOR_CONFIRMED:
        capabilities.add(Requirement.CONFIRMED_FRAME)
    if contact_scene:
        capabilities.add(Requirement.CONTACT_SCENE)
    return frozenset(capabilities)


def afforded_entries(
    inventory: SchemaInventory,
    morphology: RobotMorphologyV1,
    *,
    contact_scene: bool = False,
) -> tuple[SchemaEntry, ...]:
    """The schemas this body can be asked for, and no others.

    The complement is just as important: schemas a robot does *not* afford become
    its refusal boundary, so asking a one-armed robot for a handover returns a
    typed unsupported result instead of an invented motion.
    """

    capabilities = capabilities_of(morphology, contact_scene=contact_scene)
    return tuple(
        entry for entry in inventory.entries if entry.requires <= capabilities
    )


def unafforded_reasons(
    inventory: SchemaInventory,
    morphology: RobotMorphologyV1,
    entry_id: str,
    *,
    contact_scene: bool = False,
) -> tuple[str, ...]:
    """Exactly which capabilities a robot lacks for one schema."""

    entry = inventory.by_id(entry_id)
    capabilities = capabilities_of(morphology, contact_scene=contact_scene)
    return tuple(sorted(item.value for item in entry.requires - capabilities))


def _parse_entry(payload: dict, *, kind: SchemaKind) -> SchemaEntry:
    if kind is SchemaKind.PATH:
        schema = SegmentSchemaV1(
            kind=SchemaKind.PATH,
            vector=PathVector(payload["vector"]),
            conformation=Conformation(payload["conformation"]),
            deixis=Deixis(payload["deixis"]),
            contour=Contour(payload["contour"]),
        )
    else:
        schema = SegmentSchemaV1(
            kind=SchemaKind.STATIVE, stative=Stative(payload["stative"])
        )
    return SchemaEntry(
        entry_id=str(payload["id"]),
        schema=schema,
        figure_role=BindingRole(payload["figure"]),
        ground_role=BindingRole(payload["ground"]),
        gloss=str(payload["gloss"]),
        requires=frozenset(Requirement(item) for item in payload.get("requires", ())),
    )


@lru_cache(maxsize=4)
def load_inventory(path: Path = INVENTORY_PATH) -> SchemaInventory:
    """Read the sealed inventory, verifying its sidecar digest when present."""

    resolved = Path(path).resolve()
    payload = resolved.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()

    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
        if expected != digest:
            raise ValueError(
                "The schema inventory does not match its sealed digest. Every "
                "primitive ever baked was certified against the sealed version; "
                "re-seal deliberately rather than by accident."
            )

    data = json.loads(payload)
    entries = [
        _parse_entry(item, kind=SchemaKind.PATH) for item in data["path_schemas"]
    ] + [_parse_entry(item, kind=SchemaKind.STATIVE) for item in data["statives"]]

    ids = [entry.entry_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("schema inventory entry ids must be unique")
    keys = [entry.binding_key for entry in entries]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(
            f"two inventory entries describe the same binding: {duplicates}. "
            "Retrieval keys on schema plus roles, so a duplicate would make two "
            "primitives indistinguishable."
        )

    fractions = {str(key): float(value) for key, value in data["remove_fractions"].items()}
    if not all(0.0 <= value <= 1.0 for value in fractions.values()):
        raise ValueError("remove fractions must lie within the reachable radius")

    return SchemaInventory(
        version=str(data["schema_version"]),
        inventory_id=str(data["inventory_id"]),
        sha256=digest,
        entries=tuple(entries),
        remove_fractions=fractions,
        manner_axes=tuple(item["name"] for item in data["manner_axes"]),
    )
