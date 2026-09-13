"""One neutral typed contract for leaves and composite skills.

A skill is the same kind of thing whether it is a controller against a
grounded contract or a tree of other skills: typed arguments, the
capabilities it requires, the predicates that must hold to begin
(initiation), throughout (invariants) and on success (effects), a
termination rule, a finite timeout, a bounded recovery, the resources it
owns while it runs, and the context its evidence is valid in. Nothing in the
contract names a limb, a digit or a joint: resources and predicates are
strings the library declares, and a body supplies their meaning.

Definitions live in a library as a directed acyclic graph -- a composite
refers to its children by identifier, and the same definition may appear
under many parents -- with loops represented only by a bounded RepeatUntil.
A library is expanded into a task tree for a particular root and set of
arguments; the tree is what an executor runs and what an execution record
refers back to by hash.

What the library refuses, at construction, is the set of things that make a
tree meaningless at run time: a cycle among definitions (including through
a recovery), a retry without a bound, a predicate nobody declared, a child
argument nobody bound, and a resource a child would own that its parent
does not, or that two parallel children would both own exclusively.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from ..contracts import Contract
from ..hashing import content_hash


IDENTIFIER = r"^[a-z][a-z0-9_]*$"
"""Skill, predicate and argument identifiers: lower-case, underscore-joined."""


class NodeKind(StrEnum):
    PRIMITIVE = "primitive"
    """A controller against a grounded contract."""
    SEQUENCE = "sequence"
    """Ordered children; every one must succeed."""
    SELECTOR = "selector"
    """Ranked alternatives satisfying the same effects; the first to succeed."""
    PARALLEL = "parallel"
    """Children that own disjoint resources; all must succeed."""
    REPEAT_UNTIL = "repeat_until"
    """One child, repeated within a budget until a predicate holds."""
    OBSERVE = "observe"
    """Collect declared evidence and update belief."""


COMPOSITE_KINDS = frozenset({NodeKind.SEQUENCE, NodeKind.SELECTOR, NodeKind.PARALLEL, NodeKind.REPEAT_UNTIL})


class Verdict(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    """No observation could decide; not a softer failure."""
    INTERRUPTED = "interrupted"
    """Stopped from outside before it could decide."""


class ArgumentType(StrEnum):
    STRING = "string"
    IDENTIFIER = "identifier"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


class ArgumentSpecV1(Contract):
    name: str = Field(pattern=IDENTIFIER)
    type: ArgumentType
    required: bool = True
    description: str = ""


class PredicateSpecV1(Contract):
    """A predicate the library declares. A reference to any other is refused."""

    name: str = Field(pattern=IDENTIFIER)
    parameters: tuple[str, ...] = ()
    description: str = ""

    @field_validator("parameters")
    @classmethod
    def parameter_names(cls, parameters: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(parameters)) != len(parameters):
            raise ValueError("predicate parameters must be distinct")
        for parameter in parameters:
            if not re.fullmatch(IDENTIFIER, parameter):
                raise ValueError(f"predicate parameter {parameter!r} is not an identifier")
        return parameters


class PredicateRefV1(Contract):
    """A declared predicate applied to arguments: ``$name`` binds a skill
    argument at expansion, anything else is a literal."""

    name: str = Field(pattern=IDENTIFIER)
    arguments: tuple[str, ...] = ()
    negate: bool = False

    @property
    def bound_names(self) -> tuple[str, ...]:
        return tuple(a[1:] for a in self.arguments if a.startswith("$"))


class ResourceMode(StrEnum):
    EXCLUSIVE = "exclusive"
    SHARED = "shared"


class ResourceClaimV1(Contract):
    """Something a running skill owns: a manipulator, a carried object, a
    support contact, a perception channel. ``$name`` binds an argument."""

    resource: str = Field(min_length=1)
    mode: ResourceMode = ResourceMode.EXCLUSIVE

    @property
    def bound_names(self) -> tuple[str, ...]:
        return tuple(m.group(1) for m in re.finditer(r"\$([a-z][a-z0-9_]*)", self.resource))


class OnUnknown(StrEnum):
    PROPAGATE = "propagate"
    """An undecidable predicate leaves the node undecided."""
    FAIL = "fail"
    """An undecidable predicate counts against the node."""


class TerminationRuleV1(Contract):
    """How a node's verdict is settled once its controller or children have
    reported. Success additionally requires the effects to hold, unless the
    definition says otherwise; an undecidable predicate is handled as
    ``on_unknown`` says."""

    require_effects: bool = True
    on_unknown: OnUnknown = OnUnknown.PROPAGATE


class RecoveryV1(Contract):
    """What happens after an attempt fails: an optional recovery skill, then
    another attempt, within a bound that includes the first attempt."""

    skill: str | None = Field(default=None, pattern=IDENTIFIER)
    max_attempts: int = Field(default=1, ge=1, le=16)


class RangeV1(Contract):
    quantity: str = Field(min_length=1)
    low: float
    high: float
    units: str = Field(min_length=1)

    @model_validator(mode="after")
    def finite_and_ordered(self) -> Self:
        if not (math.isfinite(self.low) and math.isfinite(self.high)) or self.high < self.low:
            raise ValueError("a validity range must be finite and increasing")
        return self


class ValidityContextV1(Contract):
    """Where the evidence for a skill was gathered, and so where its
    certificate means anything."""

    bodies: tuple[str, ...] = ()
    environments: tuple[str, ...] = ()
    operating_range: tuple[RangeV1, ...] = ()
    evidence: tuple[str, ...] = ()
    notes: str = ""


class ChildRefV1(Contract):
    """A child definition and how the parent's arguments reach it."""

    skill: str = Field(pattern=IDENTIFIER)
    bindings: dict[str, str] = {}
    rank: int = Field(default=0, ge=0)

    @field_validator("bindings")
    @classmethod
    def binding_names(cls, bindings: dict[str, str]) -> dict[str, str]:
        for key in bindings:
            if not re.fullmatch(IDENTIFIER, key):
                raise ValueError(f"binding target {key!r} is not an identifier")
        return bindings


class ObservationSpecV1(Contract):
    """What an Observe node establishes, and from what."""

    evidence: tuple[str, ...] = Field(min_length=1)
    source: str = Field(min_length=1)


class LoopSpecV1(Contract):
    """A bounded loop: stop when ``until`` holds, fail when the budget is
    spent or when ``progress`` stops holding after an attempt."""

    until: PredicateRefV1
    progress: PredicateRefV1 | None = None
    max_attempts: int = Field(ge=1, le=64)


class SkillDefinitionV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    skill_id: str = Field(pattern=IDENTIFIER)
    kind: NodeKind
    description: str = ""
    arguments: tuple[ArgumentSpecV1, ...] = ()
    requirements: tuple[str, ...] = ()
    """Capabilities the body must afford, as neutral names."""
    initiation: tuple[PredicateRefV1, ...] = ()
    invariants: tuple[PredicateRefV1, ...] = ()
    effects: tuple[PredicateRefV1, ...] = ()
    termination: TerminationRuleV1 = TerminationRuleV1()
    timeout_s: float = Field(gt=0.0)
    recovery: RecoveryV1 = RecoveryV1()
    resources: tuple[ResourceClaimV1, ...] = ()
    validity: ValidityContextV1 = ValidityContextV1()
    children: tuple[ChildRefV1, ...] = ()
    loop: LoopSpecV1 | None = None
    observation: ObservationSpecV1 | None = None
    controller: str | None = None
    """For a primitive: the controller family the body must realize."""

    @field_validator("timeout_s")
    @classmethod
    def finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("a skill needs a finite timeout")
        return value

    @field_validator("arguments")
    @classmethod
    def distinct_arguments(cls, arguments: tuple[ArgumentSpecV1, ...]) -> tuple[ArgumentSpecV1, ...]:
        names = [a.name for a in arguments]
        if len(set(names)) != len(names):
            raise ValueError("skill arguments must be distinct")
        return arguments

    @property
    def argument_names(self) -> frozenset[str]:
        return frozenset(a.name for a in self.arguments)

    @property
    def predicate_refs(self) -> tuple[PredicateRefV1, ...]:
        refs = [*self.initiation, *self.invariants, *self.effects]
        if self.loop is not None:
            refs.append(self.loop.until)
            if self.loop.progress is not None:
                refs.append(self.loop.progress)
        return tuple(refs)

    @model_validator(mode="after")
    def shape_matches_kind(self) -> Self:
        kind = self.kind
        if kind is NodeKind.PRIMITIVE:
            if not self.controller:
                raise ValueError(f"{self.skill_id}: a primitive names its controller family")
            if self.children or self.loop or self.observation:
                raise ValueError(f"{self.skill_id}: a primitive has no children, loop or observation")
        elif kind is NodeKind.OBSERVE:
            if self.observation is None:
                raise ValueError(f"{self.skill_id}: an observation says what it establishes")
            if self.children or self.loop or self.controller:
                raise ValueError(f"{self.skill_id}: an observation has no children, loop or controller")
        else:
            if self.controller or self.observation:
                raise ValueError(f"{self.skill_id}: a composite has no controller or observation")
            if kind is NodeKind.REPEAT_UNTIL:
                if len(self.children) != 1:
                    raise ValueError(f"{self.skill_id}: a loop repeats exactly one child")
                if self.loop is None:
                    raise ValueError(f"{self.skill_id}: a loop needs a bound and a predicate")
            else:
                if self.loop is not None:
                    raise ValueError(f"{self.skill_id}: only a loop carries a loop specification")
                minimum = 2 if kind is NodeKind.PARALLEL else 1
                if len(self.children) < minimum:
                    raise ValueError(f"{self.skill_id}: a {kind.value} needs at least {minimum} child(ren)")
            if kind is NodeKind.SELECTOR:
                ranks = [c.rank for c in self.children]
                if len(set(ranks)) != len(ranks):
                    raise ValueError(f"{self.skill_id}: selector alternatives need distinct ranks")
        declared = self.argument_names
        for ref in self.predicate_refs:
            for name in ref.bound_names:
                if name not in declared:
                    raise ValueError(f"{self.skill_id}: predicate {ref.name} binds undeclared argument ${name}")
        for claim in self.resources:
            for name in claim.bound_names:
                if name not in declared:
                    raise ValueError(f"{self.skill_id}: resource {claim.resource!r} binds undeclared argument ${name}")
        for child in self.children:
            for value in child.bindings.values():
                if value.startswith("$") and value[1:] not in declared:
                    raise ValueError(f"{self.skill_id}: child {child.skill} binds undeclared argument {value}")
        if self.recovery.skill == self.skill_id:
            raise ValueError(f"{self.skill_id}: a skill cannot be its own recovery")
        return self


def substitute(text: str, arguments: dict[str, str]) -> str:
    """Replace every ``$name`` in ``text`` with the bound argument."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in arguments:
            raise KeyError(name)
        return arguments[name]

    return re.sub(r"\$([a-z][a-z0-9_]*)", replace, text)


class TaskNodeV1(Contract):
    """One node of an expanded tree: a definition with its arguments bound."""

    node_id: str = Field(min_length=1)
    skill_id: str = Field(pattern=IDENTIFIER)
    kind: NodeKind
    depth: int = Field(ge=1)
    arguments: dict[str, str] = {}
    resources: tuple[str, ...] = ()
    """Exclusive resources this node owns, with arguments substituted."""
    shared_resources: tuple[str, ...] = ()
    children: tuple[TaskNodeV1, ...] = ()
    recovery: TaskNodeV1 | None = None

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()
        if self.recovery is not None:
            yield from self.recovery.walk()

    @property
    def max_depth(self) -> int:
        return max(node.depth for node in self.walk())


class TaskTreeV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    library_id: str = Field(min_length=1)
    library_sha256: str = Field(min_length=64, max_length=64)
    root: TaskNodeV1

    @property
    def max_depth(self) -> int:
        return self.root.max_depth

    def node(self, node_id: str) -> TaskNodeV1:
        for candidate in self.root.walk():
            if candidate.node_id == node_id:
                return candidate
        raise KeyError(node_id)


class SkillLibraryV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    library_id: str = Field(min_length=1)
    description: str = ""
    predicates: tuple[PredicateSpecV1, ...] = ()
    skills: tuple[SkillDefinitionV1, ...] = Field(min_length=1)

    # -- lookup ---------------------------------------------------------

    def skill(self, skill_id: str) -> SkillDefinitionV1:
        for definition in self.skills:
            if definition.skill_id == skill_id:
                return definition
        raise KeyError(skill_id)

    def predicate(self, name: str) -> PredicateSpecV1:
        for spec in self.predicates:
            if spec.name == name:
                return spec
        raise KeyError(name)

    # -- validation -----------------------------------------------------

    @model_validator(mode="after")
    def well_formed(self) -> Self:
        ids = [s.skill_id for s in self.skills]
        if len(set(ids)) != len(ids):
            raise ValueError("skill identifiers must be unique")
        names = [p.name for p in self.predicates]
        if len(set(names)) != len(names):
            raise ValueError("predicate names must be unique")
        known = {s.skill_id: s for s in self.skills}
        arity = {p.name: len(p.parameters) for p in self.predicates}

        successors: dict[str, list[str]] = {skill_id: [] for skill_id in ids}
        for definition in self.skills:
            for ref in definition.predicate_refs:
                if ref.name not in arity:
                    raise ValueError(f"{definition.skill_id}: unknown predicate {ref.name!r}")
                if len(ref.arguments) != arity[ref.name]:
                    raise ValueError(f"{definition.skill_id}: predicate {ref.name} takes {arity[ref.name]} argument(s), given {len(ref.arguments)}")
            if definition.recovery.skill is not None:
                if definition.recovery.skill not in known:
                    raise ValueError(f"{definition.skill_id}: unknown recovery skill {definition.recovery.skill!r}")
                successors[definition.skill_id].append(definition.recovery.skill)
            for child in definition.children:
                if child.skill not in known:
                    raise ValueError(f"{definition.skill_id}: unknown child skill {child.skill!r}")
                successors[definition.skill_id].append(child.skill)
                target = known[child.skill]
                child_arguments = {a.name: a for a in target.arguments}
                for key in child.bindings:
                    if key not in child_arguments:
                        raise ValueError(f"{definition.skill_id}: child {child.skill} has no argument {key!r}")
                for argument in target.arguments:
                    if argument.required and argument.name not in child.bindings:
                        raise ValueError(f"{definition.skill_id}: child {child.skill} argument {argument.name!r} is unbound")
                parent_arguments = {a.name: a for a in definition.arguments}
                for key, value in child.bindings.items():
                    if value.startswith("$"):
                        source = parent_arguments[value[1:]]
                        if source.type is not child_arguments[key].type:
                            raise ValueError(f"{definition.skill_id}: child {child.skill} argument {key!r} is {child_arguments[key].type.value}, bound to {source.type.value} ${source.name}")
                    else:
                        _check_literal(child_arguments[key].type, value, f"{definition.skill_id}: child {child.skill} argument {key!r}")
        _reject_cycles(successors)
        for definition in self.skills:
            if definition.kind in COMPOSITE_KINDS:
                _check_ownership(definition, known)
        return self

    # -- expansion ------------------------------------------------------

    def expand(self, root: str, arguments: dict[str, str] | None = None) -> TaskTreeV1:
        """A task tree for ``root`` with ``arguments`` bound; loops are kept
        as loops, not unrolled, and a definition reached twice appears twice."""

        definition = self.skill(root)
        bound = _bind_arguments(definition, dict(arguments or {}), f"root {root}")
        node = self._expand(definition, bound, "0", 1, ())
        return TaskTreeV1(library_id=self.library_id, library_sha256=self.content_hash(), root=node)

    def _expand(self, definition: SkillDefinitionV1, arguments: dict[str, str], node_id: str, depth: int, trail: tuple[str, ...]) -> TaskNodeV1:
        if definition.skill_id in trail:
            raise ValueError("expansion revisits " + " -> ".join((*trail, definition.skill_id)))
        trail = (*trail, definition.skill_id)
        children = []
        ordered = sorted(definition.children, key=lambda c: c.rank) if definition.kind is NodeKind.SELECTOR else definition.children
        for index, child in enumerate(ordered):
            target = self.skill(child.skill)
            child_arguments = {key: substitute(value, arguments) if value.startswith("$") else value for key, value in child.bindings.items()}
            child_arguments = _bind_arguments(target, child_arguments, f"{definition.skill_id} -> {child.skill}")
            children.append(self._expand(target, child_arguments, f"{node_id}.{index}", depth + 1, trail))
        recovery = None
        if definition.recovery.skill is not None:
            target = self.skill(definition.recovery.skill)
            recovery_arguments = _bind_arguments(target, {a.name: arguments[a.name] for a in target.arguments if a.name in arguments}, f"{definition.skill_id} recovery")
            recovery = self._expand(target, recovery_arguments, f"{node_id}.r", depth + 1, trail)
        exclusive = tuple(substitute(c.resource, arguments) for c in definition.resources if c.mode is ResourceMode.EXCLUSIVE)
        shared = tuple(substitute(c.resource, arguments) for c in definition.resources if c.mode is ResourceMode.SHARED)
        return TaskNodeV1(node_id=node_id, skill_id=definition.skill_id, kind=definition.kind, depth=depth, arguments=arguments,
                          resources=exclusive, shared_resources=shared, children=tuple(children), recovery=recovery)

    def definition_depth(self, root: str) -> int:
        """Nesting depth of the deepest child under ``root``, root counted as 1."""

        def depth(skill_id: str, trail: tuple[str, ...]) -> int:
            definition = self.skill(skill_id)
            if not definition.children:
                return 1
            return 1 + max(depth(c.skill, (*trail, skill_id)) for c in definition.children)

        return depth(root, ())


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _check_literal(kind: ArgumentType, value: str, where: str) -> None:
    try:
        parse_argument(kind, value)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def parse_argument(kind: ArgumentType, value: str) -> Any:
    """A bound argument's value in its declared type."""

    if kind is ArgumentType.STRING:
        return value
    if kind is ArgumentType.IDENTIFIER:
        if not re.fullmatch(r"^[A-Za-z_][A-Za-z0-9_:\-./]*$", value):
            raise ValueError(f"{value!r} is not an identifier")
        return value
    if kind is ArgumentType.INTEGER:
        if not re.fullmatch(r"^[+-]?\d+$", value):
            raise ValueError(f"{value!r} is not an integer")
        return int(value)
    if kind is ArgumentType.NUMBER:
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"{value!r} is not a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{value!r} is not finite")
        return number
    if kind is ArgumentType.BOOLEAN:
        if value not in ("true", "false"):
            raise ValueError(f"{value!r} is not a boolean")
        return value == "true"
    raise ValueError(f"unknown argument type {kind}")


def _bind_arguments(definition: SkillDefinitionV1, given: dict[str, str], where: str) -> dict[str, str]:
    declared = {a.name: a for a in definition.arguments}
    for key in given:
        if key not in declared:
            raise ValueError(f"{where}: no argument {key!r}")
    for argument in definition.arguments:
        if argument.name not in given:
            if argument.required:
                raise ValueError(f"{where}: argument {argument.name!r} is unbound")
            continue
        _check_literal(argument.type, given[argument.name], f"{where}: argument {argument.name!r}")
    return {name: given[name] for name in sorted(given)}


def _reject_cycles(successors: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(node: str, trail: tuple[str, ...]) -> None:
        if node in done:
            return
        if node in visiting:
            raise ValueError("skill definitions form a cycle: " + " -> ".join((*trail, node)))
        visiting.add(node)
        for successor in successors[node]:
            visit(successor, (*trail, node))
        visiting.discard(node)
        done.add(node)

    for start in successors:
        visit(start, ())


def _check_ownership(definition: SkillDefinitionV1, known: dict[str, SkillDefinitionV1]) -> None:
    """A child owns nothing its parent does not; parallel children own
    nothing exclusively in common. Checked on the resource text with the
    parent's argument names substituted through the bindings, so a claim on
    ``effector:$effector`` in a child bound to the parent's ``$effector`` is
    the parent's claim, and two children bound to the same argument clash."""

    parent_claims = {c.resource: c.mode for c in definition.resources}
    seen_exclusive: dict[str, str] = {}
    for child in definition.children:
        target = known[child.skill]
        rewrite = {key: value for key, value in child.bindings.items()}
        for claim in target.resources:
            text = claim.resource
            for name in claim.bound_names:
                if name in rewrite:
                    text = text.replace(f"${name}", rewrite[name])
            owned = parent_claims.get(text)
            if owned is None:
                raise ValueError(f"{definition.skill_id}: child {child.skill} owns {text!r}, which its parent does not")
            if claim.mode is ResourceMode.EXCLUSIVE and owned is not ResourceMode.EXCLUSIVE:
                raise ValueError(f"{definition.skill_id}: child {child.skill} owns {text!r} exclusively, which its parent only shares")
            if definition.kind is NodeKind.PARALLEL and claim.mode is ResourceMode.EXCLUSIVE:
                if text in seen_exclusive:
                    raise ValueError(f"{definition.skill_id}: parallel children {seen_exclusive[text]} and {child.skill} both own {text!r} exclusively")
                seen_exclusive[text] = child.skill


def library_digest(library: SkillLibraryV1) -> str:
    return content_hash(library)
