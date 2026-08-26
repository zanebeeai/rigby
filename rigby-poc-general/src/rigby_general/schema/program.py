"""The magnitude-neutral motion schema IR.

This is the layer that makes an arbitrary robot possible. It sits *above*
``MotionProgramV2``: a language model authors a ``MotionSchemaProgramV1``, and
``rigby_general.grounding`` deterministically turns that into an ordinary
``MotionProgramV2`` using one particular robot's measured scale. Everything
downstream of the grounder -- compile, simulate, certify, render, judge -- is the
certified v2 pipeline, unchanged.

Three properties are load-bearing, and each traces to a specific result:

**No metric values anywhere.** Talmy (1983) observes that closed-class spatial
terms are topological: magnitude-neutral, shape-idealized. The same *in* serves a
thimble and a volcano. So this IR has no float fields at all -- a fact asserted
structurally by ``assert_metric_free`` rather than left to reviewer discipline.
A model that cannot express "0.42 m" cannot guess a reach it has no way to know.

**Path and Manner are separate slots.** Talmy (1975) decomposes a motion event
into Figure, Motion, Path and Ground, plus the co-events Manner and Cause, and
notes that English expresses Path and Manner separately ("run *in*", "limp
*across*"). Their separability in the grammar is evidence they are separate in the
underlying representation -- so a primitive here is a *product*
(Figure x Path x Ground x Manner), not an atom like "wave". A small closed
inventory then spans a large behaviour space, and the per-robot work collapses
from authoring vocabulary to computing bindings.

**Semantics live at segment boundaries.** Tversky & Lee (1998) find that route
directions and sketch maps share one skeleton: segments punctuated by
reorientations, with descriptive detail concentrated at the action points rather
than along the progressions. So a program is a segment chain, each boundary
marked by the constraint that changes there, and the compiler need only be exact
at the nodes.
"""

from __future__ import annotations

import types
import typing
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, model_validator
from rigby_v2.contracts import Contract
from rigby_v2.hashing import content_hash


SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------
# The closed class
# --------------------------------------------------------------------------


class PathVector(StrEnum):
    """Talmy's Path vector: the deep sense of arrival, traversal, or departure."""

    TO = "to"
    VIA = "via"
    FROM = "from"
    TOWARD = "toward"
    AWAY = "away"


class Conformation(StrEnum):
    """The geometry the Ground contributes to the Path.

    Idealized, per Talmy: a Ground enters the schema as a point, a line, a
    surface, a volume, a boundary, or an axis -- never as its actual shape.
    """

    POINT = "point"
    LINE = "line"
    SURFACE = "surface"
    VOLUME = "volume"
    BOUNDARY = "boundary"
    AXIS = "axis"


class Deixis(StrEnum):
    """Direction relative to the deictic centre (the robot's own front)."""

    NEUTRAL = "neutral"
    HITHER = "hither"
    THITHER = "thither"


class Contour(StrEnum):
    """Path shape, in the coarse categories language actually distinguishes.

    Tversky & Lee found two path categories sufficed to reconstruct route maps:
    straight and curved. Circular and oscillating are added because a robot can
    be asked for them explicitly and they are structurally distinct.
    """

    STRAIGHT = "straight"
    ARCED = "arced"
    CIRCULAR = "circular"
    OSCILLATING = "oscillating"


class Stative(StrEnum):
    """Non-translational schemas: Talmy's BELOC, plus force dynamics."""

    ORIENT = "orient"
    HOLD = "hold"
    APPLY_FORCE = "apply_force"


class SchemaKind(StrEnum):
    PATH = "path"
    STATIVE = "stative"


class Remove(StrEnum):
    """Degree of remove between Figure and Ground.

    Ordinal and magnitude-neutral: ``distal`` means far *for this body*, which is
    exactly what the grounder resolves against the measured reach radius.
    """

    COINCIDENT = "coincident"
    ADJACENT = "adjacent"
    PROXIMAL = "proximal"
    MEDIAL = "medial"
    DISTAL = "distal"


class Dimensionality(StrEnum):
    POINT = "point"
    LINE = "line"
    PLANE = "plane"
    VOLUME = "volume"


class ReferenceFrame(StrEnum):
    """Which frame the schema is read in.

    ``INTRINSIC`` requires an operator-confirmed front direction; a program that
    uses it against a robot whose frame is only ``DERIVED`` is refused by the
    grounder rather than silently resolved against a guess.
    """

    INTRINSIC = "intrinsic"
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    GROUND_INTRINSIC = "ground_intrinsic"


class BoundaryCondition(StrEnum):
    """What changes at the node that ends this segment."""

    CONTACT_MADE = "contact_made"
    CONTACT_BROKEN = "contact_broken"
    DIRECTION_REVERSED = "direction_reversed"
    ORIENTATION_SET = "orientation_set"
    DWELL = "dwell"
    TERMINUS = "terminus"


class BindingRole(StrEnum):
    """Body-neutral Figure and Ground roles.

    Talmy's assignment criteria -- Figure is the smaller, movable, to-be-located
    element; Ground is the larger, more permanent reference -- apply directly to a
    kinematic tree, where the base is the ultimate Ground and every distal link is
    a Figure relative to its parent.

    These names are what make requirement ``schema_invariance`` checkable: they
    carry no robot-specific information, so the same prompt must produce the same
    roles on every robot.
    """

    PRIMARY_EFFECTOR = "primary_effector"
    SECONDARY_EFFECTOR = "secondary_effector"
    BASE = "base"
    SENSOR = "sensor"
    SELF = "self"
    TARGET_OBJECT = "target_object"
    SECONDARY_OBJECT = "secondary_object"
    SUPPORT_SURFACE = "support_surface"
    WORLD_GROUND = "world_ground"
    GRAVITY_AXIS = "gravity_axis"
    FRONT_AXIS = "front_axis"
    LATERAL_AXIS = "lateral_axis"


class Concurrency(StrEnum):
    SEQUENCE = "sequence"
    CONCURRENT_INDEPENDENT = "concurrent_independent"
    CONCURRENT_COUPLED = "concurrent_coupled"


class Coupling(StrEnum):
    """How two concurrent segments constrain one another.

    Each maps onto an objective the certified v2 IK refinement already supports;
    ``MAINTAIN_OFFSET`` in particular is ``RelativeSitePositionObjective``, which
    is how bimanual coordination is expressed without any new solver.
    """

    MAINTAIN_OFFSET = "maintain_offset"
    MAINTAIN_CONTACT = "maintain_contact"
    MIRROR = "mirror"


# --------------------------------------------------------------------------
# Schema components
# --------------------------------------------------------------------------


class SegmentSchemaV1(Contract):
    """A path schema or a stative schema.

    Modelled as one class with a discriminating ``kind`` and nullable members
    rather than a union, because the planner response must survive OpenAI's
    strict JSON-Schema conversion -- the same reason
    ``rigby_v2.planner.semantic`` mirrors its contracts.
    """

    kind: SchemaKind
    vector: PathVector | None = None
    conformation: Conformation | None = None
    deixis: Deixis | None = None
    contour: Contour | None = None
    stative: Stative | None = None

    @model_validator(mode="after")
    def members_match_kind(self) -> Self:
        path_members = (self.vector, self.conformation, self.deixis, self.contour)
        if self.kind is SchemaKind.PATH:
            if any(member is None for member in path_members):
                raise ValueError(
                    "A path schema requires vector, conformation, deixis, and contour"
                )
            if self.stative is not None:
                raise ValueError("A path schema cannot also name a stative")
        else:
            if self.stative is None:
                raise ValueError("A stative schema requires a stative")
            if any(member is not None for member in path_members):
                raise ValueError("A stative schema cannot carry path members")
        return self

    @property
    def canonical_key(self) -> str:
        """The primitive-library lookup key.

        A discrete symbol, which is why primitive retrieval is exact-match and
        does not need a text embedding model.
        """

        if self.kind is SchemaKind.PATH:
            assert self.vector and self.conformation and self.deixis and self.contour
            return (
                f"path:{self.vector.value}.{self.conformation.value}"
                f".{self.deixis.value}.{self.contour.value}"
            )
        assert self.stative
        return f"stative:{self.stative.value}"


class MannerV1(Contract):
    """Talmy's Manner co-event, as ordinals relative to this robot's neutral.

    Every axis is an integer in [-2, 2] meaning "relative to what is neutral for
    this body". ``speed=+2`` on a fast arm and on a slow one denote different
    metre-per-second values and the same *meaning*, which is precisely the
    magnitude neutrality that lets one program serve both.

    ``repetition_count`` is the deliberate exception. Talmy's magnitude neutrality
    is about spatial extent; cardinality is not neutral -- "three times" is exact
    in any language and on any body. Recording it as an integer is therefore
    linguistically correct and leaks no body knowledge.
    """

    speed: int = Field(default=0, ge=-2, le=2)
    effort: int = Field(default=0, ge=-2, le=2)
    smoothness: int = Field(default=0, ge=-2, le=2)
    rhythm: int = Field(default=0, ge=-2, le=2)
    amplitude: int = Field(default=0, ge=-2, le=2)
    repetition: int = Field(default=0, ge=-2, le=2)
    precision: int = Field(default=0, ge=-2, le=2)
    repetition_count: int | None = Field(default=None, ge=1, le=64)

    @property
    def is_neutral(self) -> bool:
        return self == NEUTRAL_MANNER

    def distance(self, other: "MannerV1") -> int:
        """L1 distance over the ordinal axes; used to rank retrieved primitives."""

        return sum(
            abs(getattr(self, axis) - getattr(other, axis)) for axis in MANNER_AXES
        )


MANNER_AXES: tuple[str, ...] = (
    "speed",
    "effort",
    "smoothness",
    "rhythm",
    "amplitude",
    "repetition",
    "precision",
)

NEUTRAL_MANNER = MannerV1()


class RegionV1(Contract):
    """Where the Figure ends up relative to the Ground, topologically."""

    remove: Remove
    dimensionality: Dimensionality = Dimensionality.POINT


class RoleBindingV1(Contract):
    """A Figure or Ground, named by role and optionally bound to a robot site.

    ``role`` is authored by the planner and is body-neutral. ``site`` and
    ``object_id`` are filled in later by ``rigby_general.binding`` once a robot is
    known. The invariance hash drops both, so the same prompt on a Franka and on
    an ALOHA must hash identically.
    """

    role: BindingRole
    site: str | None = None
    object_id: str | None = None

    @model_validator(mode="after")
    def object_roles_carry_object_ids(self) -> Self:
        object_roles = (
            BindingRole.TARGET_OBJECT,
            BindingRole.SECONDARY_OBJECT,
            BindingRole.SUPPORT_SURFACE,
        )
        if self.object_id is not None and self.role not in object_roles:
            raise ValueError(f"Role {self.role.value!r} cannot bind a scene object")
        return self

    def role_normalized(self) -> dict[str, str]:
        return {"role": self.role.value}


class SegmentV1(Contract):
    """One schema binding: the unit of the primitive library.

    Segment granularity follows Tversky & Lee: the reusable chunk is the stretch
    between two reorientations, not a whole named action. "Pick up the block" is
    several segments; each segment is separately certifiable and separately
    reusable.
    """

    segment_id: str = Field(min_length=1)
    motion_schema: SegmentSchemaV1
    figure: RoleBindingV1
    ground: RoleBindingV1
    region: RegionV1
    frame: ReferenceFrame
    manner: MannerV1 = NEUTRAL_MANNER
    boundary: BoundaryCondition

    @model_validator(mode="after")
    def figure_and_ground_differ(self) -> Self:
        if self.figure.role is self.ground.role:
            raise ValueError(
                "Figure and Ground must be distinct roles; a thing cannot locate itself"
            )
        if self.motion_schema.kind is SchemaKind.PATH:
            if self.motion_schema.deixis is not Deixis.NEUTRAL and self.frame not in (
                ReferenceFrame.INTRINSIC,
                ReferenceFrame.RELATIVE,
            ):
                raise ValueError(
                    "Deictic paths need an intrinsic or viewer frame to point away from"
                )
        elif self.boundary is BoundaryCondition.DIRECTION_REVERSED:
            raise ValueError("A stative segment has no direction to reverse")
        return self

    @property
    def canonical_key(self) -> str:
        """Primitive-library key: schema, roles, region, and frame, no manner.

        Manner is excluded because a primitive is baked at neutral manner and
        modulated at bind time -- that is the whole point of keeping the co-event
        in its own slot.
        """

        return (
            f"{self.motion_schema.canonical_key}"
            f"|{self.figure.role.value}->{self.ground.role.value}"
            f"|{self.region.remove.value}.{self.region.dimensionality.value}"
            f"|{self.frame.value}"
        )

    def role_normalized(self) -> dict[str, Any]:
        return {
            "schema": self.motion_schema.canonical_key,
            "figure": self.figure.role_normalized(),
            "ground": self.ground.role_normalized(),
            "region": {
                "remove": self.region.remove.value,
                "dimensionality": self.region.dimensionality.value,
            },
            "frame": self.frame.value,
            "manner": self.manner.model_dump(mode="python"),
            "boundary": self.boundary.value,
        }


class SegmentLinkV1(Contract):
    """One edge of the concurrency lattice."""

    from_segment: str = Field(min_length=1)
    to_segment: str = Field(min_length=1)
    relation: Concurrency
    coupling: Coupling | None = None
    coupling_region: RegionV1 | None = None

    @model_validator(mode="after")
    def coupling_matches_relation(self) -> Self:
        if self.from_segment == self.to_segment:
            raise ValueError("A segment cannot link to itself")
        if self.relation is Concurrency.CONCURRENT_COUPLED:
            if self.coupling is None:
                raise ValueError("A coupled relation requires a coupling constraint")
        elif self.coupling is not None or self.coupling_region is not None:
            raise ValueError(
                f"Relation {self.relation.value!r} cannot carry a coupling constraint"
            )
        return self


class MotionSchemaProgramV1(Contract):
    """A body-neutral motion program: the planner's entire output.

    Contains no metric value, no joint name, and no link name. It is the same
    artifact for every robot that can be asked the same thing, which is what
    requirement ``schema_invariance`` measures.
    """

    schema_version: Literal["1.0"] = "1.0"
    program_id: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    segments: tuple[SegmentV1, ...] = Field(min_length=1, max_length=32)
    links: tuple[SegmentLinkV1, ...] = ()

    @model_validator(mode="after")
    def lattice_is_well_formed(self) -> Self:
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("Segment identifiers must be unique")
        known = set(ids)

        seen_edges: set[tuple[str, str]] = set()
        successors: dict[str, list[str]] = {segment_id: [] for segment_id in ids}
        for link in self.links:
            missing = {link.from_segment, link.to_segment} - known
            if missing:
                raise ValueError(f"Link names unknown segments: {sorted(missing)}")
            edge = (link.from_segment, link.to_segment)
            if edge in seen_edges:
                raise ValueError(f"Duplicate link {edge[0]!r} -> {edge[1]!r}")
            seen_edges.add(edge)
            if link.relation is Concurrency.SEQUENCE:
                successors[link.from_segment].append(link.to_segment)

        _reject_cycles(successors)

        if len(self.segments) > 1 and not self.links:
            raise ValueError(
                "A multi-segment program must say how its segments relate in time"
            )
        return self

    def role_normalized_payload(self) -> dict[str, Any]:
        """The body-independent content of this program.

        Drops ``program_id`` (a UUID), ``source_text`` (the prompt wording), and
        every site or object binding a robot supplied. Whatever survives is the
        claim the semantic layer actually made.
        """

        return {
            "schema_version": self.schema_version,
            "segments": [segment.role_normalized() for segment in self.segments],
            "links": sorted(
                (
                    {
                        "from": link.from_segment,
                        "to": link.to_segment,
                        "relation": link.relation.value,
                        "coupling": link.coupling.value if link.coupling else None,
                    }
                    for link in self.links
                ),
                key=lambda item: (item["from"], item["to"]),
            ),
        }

    def role_normalized_hash(self) -> str:
        """Requirement ``schema_invariance`` compares exactly this value.

        If the same prompt yields different hashes on two robots, the semantic
        layer has leaked body knowledge and the design is wrong.
        """

        return content_hash(self.role_normalized_payload())

    @property
    def canonical_keys(self) -> tuple[str, ...]:
        return tuple(segment.canonical_key for segment in self.segments)


def _reject_cycles(successors: dict[str, list[str]]) -> None:
    """Depth-first cycle detection over the SEQUENCE edges only."""

    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str, trail: tuple[str, ...]) -> None:
        if node in done:
            return
        if node in visiting:
            cycle = " -> ".join((*trail, node))
            raise ValueError(f"Segment sequence contains a cycle: {cycle}")
        visiting.add(node)
        for successor in successors[node]:
            visit(successor, (*trail, node))
        visiting.discard(node)
        done.add(node)

    for start in successors:
        visit(start, ())


# --------------------------------------------------------------------------
# The metric-freedom guarantee
# --------------------------------------------------------------------------


_METRIC_TYPES = (float,)


def assert_metric_free(model: type[Contract], *, _seen: set[type] | None = None) -> None:
    """Raise if any field reachable from ``model`` can hold a real number.

    This is the structural half of requirement ``grounding_soundness``. A planner
    physically cannot emit a reach, a duration, or a joint angle if the contract
    it must fill has nowhere to put one. Enforced by test, not by review.
    """

    seen = _seen if _seen is not None else set()
    if model in seen:
        return
    seen.add(model)

    for name, field in model.model_fields.items():
        for candidate in _annotation_types(field.annotation):
            if candidate in _METRIC_TYPES:
                raise TypeError(
                    f"{model.__name__}.{name} admits {candidate.__name__}; the schema IR "
                    "must be magnitude-neutral"
                )
            if isinstance(candidate, type) and issubclass(candidate, Contract):
                assert_metric_free(candidate, _seen=seen)


def _annotation_types(annotation: Any) -> tuple[Any, ...]:
    """Flatten an annotation into the concrete types it can hold."""

    if annotation is None or annotation is type(None):
        return ()
    origin = typing.get_origin(annotation)
    if origin is None:
        return (annotation,)
    if origin in (typing.Union, types.UnionType, tuple, list, set, frozenset, dict):
        collected: list[Any] = []
        for argument in typing.get_args(annotation):
            if argument is Ellipsis:
                continue
            collected.extend(_annotation_types(argument))
        return tuple(collected)
    if origin is Literal:
        return tuple(type(value) for value in typing.get_args(annotation))
    return (origin,)
