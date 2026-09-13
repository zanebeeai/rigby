"""Whether the state one skill ends in is a state the next may begin in.

A composition of two certified skills is not certified by their two
certificates. What the first leaves behind -- where the joints are and how
fast they move, whether something is held and by what, how old the belief
is, which resources are still owned, what is touching what -- is a boundary
state, and the second skill's initiation set says which boundary states it
admits. Checking the one against the other is what this module does, and
it does it on neutral terms: joint names are data the body supplies, the
contact mode is one of three words, resources and objects are strings.

A violation is typed, measured against its limit, and classed by what can
repair it. A joint inside its range but inside the margin, or a joint past
its range, is repaired by a guarded move to the admissible region; a
residual velocity by settling; a stale belief by observing. A contact-mode
or ownership mismatch is not repaired by any path: a skill that expects to
be holding an object cannot be given one by moving smoothly, and a skill
that expects free hands cannot be satisfied by carrying the object along.
Those are refused as composition failures, and the planner says so rather
than proposing a motion that would appear to bridge them.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from ..contracts import Contract


class ContactMode(StrEnum):
    """The manipulator's relation to the object. Where the object rests is
    a separate fact of the boundary (``resting_on``), because a skill that
    begins free may still require the object to be resting somewhere."""

    FREE = "free"
    """Nothing is held in opposition."""
    HOLDING = "holding"
    """The object is held in opposition."""


class JointStateV1(Contract):
    name: str = Field(min_length=1)
    position: float
    velocity: float
    minimum: float
    maximum: float
    velocity_limit: float = Field(gt=0.0)

    @model_validator(mode="after")
    def finite(self) -> Self:
        for value in (self.position, self.velocity, self.minimum, self.maximum, self.velocity_limit):
            if not math.isfinite(value):
                raise ValueError(f"joint {self.name}: every value must be finite")
        if self.maximum <= self.minimum:
            raise ValueError(f"joint {self.name}: the range must be increasing")
        return self

    @property
    def range(self) -> float:
        return self.maximum - self.minimum

    @property
    def excess(self) -> float:
        """How far outside the range the position lies; zero inside."""

        return max(self.minimum - self.position, self.position - self.maximum, 0.0)


class BoundaryStateV1(Contract):
    """What a skill left behind, measured."""

    schema_version: Literal["1.0"] = "1.0"
    time_s: float = Field(ge=0.0)
    joints: tuple[JointStateV1, ...] = Field(min_length=1)
    contact_mode: ContactMode
    held: dict[str, str] = {}
    """Object held, by the resource holding it."""
    resting_on: dict[str, str] = {}
    """Object resting, by the support it rests on."""
    belief_age_s: float = Field(ge=0.0)
    """Physics time since the last observation the belief rests on."""
    owned: tuple[str, ...] = ()
    """Resources still owned by the skill that ended."""
    manipulator: str = Field(min_length=1)
    """The resource this boundary is measured for."""

    def joint(self, name: str) -> JointStateV1:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(name)


class InitiationSetV1(Contract):
    """The boundary states a skill admits."""

    schema_version: Literal["1.0"] = "1.0"
    skill_id: str = Field(min_length=1)
    limit_margin_fraction: float = Field(default=0.02, ge=0.0, lt=0.5)
    """Of each joint's range: how far inside its limits a joint must be."""
    speed_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    """Of each joint's velocity limit: how still the body must be."""
    contact_mode: ContactMode
    required_held: dict[str, str] = {}
    """Objects that must be held, by the resource that must hold them."""
    forbidden_held: bool = True
    """Whether holding anything not required is a mismatch."""
    required_resting: tuple[str, ...] = ()
    """Objects that must be resting on a support when the skill begins."""
    max_belief_age_s: float = Field(gt=0.0)
    required_resources: tuple[str, ...] = ()
    """Resources the skill will own; owned by anything else is a conflict."""


class Repair(StrEnum):
    NONE = "none"
    JOINT_MOVE = "joint_move"
    """A guarded move of the joints into the admissible region."""
    SETTLE = "settle"
    """Hold still until the residual velocity is under the limit."""
    OBSERVE = "observe"
    """Refresh the belief."""
    CONTACT_CHANGE = "contact_change"
    """Not a path: a skill that changes what is held or rests must run."""
    RELEASE_RESOURCE = "release_resource"
    """Not a path: the owner must release the resource first."""


REPAIRABLE_BY_PATH = frozenset({Repair.JOINT_MOVE, Repair.SETTLE, Repair.OBSERVE})


class BoundaryViolationV1(Contract):
    code: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    measured: float
    limit: float
    repair: Repair
    detail: str = ""

    @property
    def repairable_by_path(self) -> bool:
        return self.repair in REPAIRABLE_BY_PATH


class BoundaryVerdictV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    compatible: bool
    violations: tuple[BoundaryViolationV1, ...] = ()
    repair: Repair
    """What must happen before the next skill may begin: none when
    compatible; the path repair when every violation admits one; a contact
    change or a resource release when any violation admits nothing else."""
    joint_targets: dict[str, float] = {}
    """For a joint move: where each offending joint must go, the nearest
    admissible position."""

    @property
    def rejected(self) -> bool:
        return not self.compatible and self.repair not in REPAIRABLE_BY_PATH


def check_boundary(terminal: BoundaryStateV1, initiation: InitiationSetV1) -> BoundaryVerdictV1:
    """Every way ``terminal`` fails ``initiation``, each typed and measured,
    and the one thing that must happen before the next skill begins."""

    violations: list[BoundaryViolationV1] = []
    targets: dict[str, float] = {}
    for joint in terminal.joints:
        margin = initiation.limit_margin_fraction * joint.range
        low, high = joint.minimum + margin, joint.maximum - margin
        if joint.excess > 0.0:
            violations.append(BoundaryViolationV1(code="joint_beyond_limit", subject=joint.name, measured=joint.position,
                                                  limit=joint.minimum if joint.position < joint.minimum else joint.maximum, repair=Repair.JOINT_MOVE,
                                                  detail=f"{joint.excess:.4f} past the range"))
            targets[joint.name] = low if joint.position < joint.minimum else high
        elif joint.position < low or joint.position > high:
            violations.append(BoundaryViolationV1(code="joint_inside_margin", subject=joint.name, measured=joint.position,
                                                  limit=low if joint.position < low else high, repair=Repair.JOINT_MOVE,
                                                  detail=f"within {initiation.limit_margin_fraction:.0%} of range of a limit"))
            targets[joint.name] = low if joint.position < low else high
        speed_limit = initiation.speed_fraction * joint.velocity_limit
        if abs(joint.velocity) > speed_limit:
            violations.append(BoundaryViolationV1(code="velocity_too_high", subject=joint.name, measured=abs(joint.velocity), limit=speed_limit, repair=Repair.SETTLE))
    if terminal.contact_mode is not initiation.contact_mode:
        violations.append(BoundaryViolationV1(code="contact_mode_mismatch", subject=terminal.manipulator, measured=0.0, limit=0.0, repair=Repair.CONTACT_CHANGE,
                                              detail=f"ends {terminal.contact_mode.value}, the next skill begins {initiation.contact_mode.value}"))
    for obj, resource in initiation.required_held.items():
        holder = terminal.held.get(obj)
        if holder != resource:
            violations.append(BoundaryViolationV1(code="ownership_mismatch", subject=obj, measured=0.0, limit=0.0, repair=Repair.CONTACT_CHANGE,
                                                  detail=f"{obj} must be held by {resource}; held by {holder or 'nothing'}"))
    if initiation.forbidden_held:
        for obj, holder in terminal.held.items():
            if obj not in initiation.required_held:
                violations.append(BoundaryViolationV1(code="ownership_mismatch", subject=obj, measured=0.0, limit=0.0, repair=Repair.CONTACT_CHANGE,
                                                      detail=f"{obj} is held by {holder}; the next skill expects it not held"))
    for obj in initiation.required_resting:
        if obj not in terminal.resting_on:
            violations.append(BoundaryViolationV1(code="object_not_resting", subject=obj, measured=0.0, limit=0.0, repair=Repair.CONTACT_CHANGE,
                                                  detail=f"{obj} is not resting on a support; a skill that sets it down must run first"))
    if terminal.belief_age_s > initiation.max_belief_age_s:
        violations.append(BoundaryViolationV1(code="belief_stale", subject="belief", measured=terminal.belief_age_s, limit=initiation.max_belief_age_s, repair=Repair.OBSERVE))
    for resource in initiation.required_resources:
        if resource in terminal.owned:
            violations.append(BoundaryViolationV1(code="resource_conflict", subject=resource, measured=0.0, limit=0.0, repair=Repair.RELEASE_RESOURCE,
                                                  detail=f"{resource} is still owned by the skill that ended"))
    if not violations:
        return BoundaryVerdictV1(compatible=True, repair=Repair.NONE)
    repairs = {v.repair for v in violations}
    if repairs & {Repair.CONTACT_CHANGE, Repair.RELEASE_RESOURCE}:
        repair = Repair.CONTACT_CHANGE if Repair.CONTACT_CHANGE in repairs else Repair.RELEASE_RESOURCE
    elif Repair.JOINT_MOVE in repairs:
        repair = Repair.JOINT_MOVE
    elif Repair.SETTLE in repairs:
        repair = Repair.SETTLE
    else:
        repair = Repair.OBSERVE
    return BoundaryVerdictV1(compatible=False, violations=tuple(violations), repair=repair, joint_targets=targets if repair is Repair.JOINT_MOVE else {})


class TransitionCostV1(Contract):
    """What a repair spent: physics time, joint travel and the peak speed."""

    physics_s: float = Field(ge=0.0)
    joint_travel_rad: float = Field(ge=0.0)
    peak_speed_fraction: float = Field(ge=0.0)
    repairs: tuple[Repair, ...] = ()
