"""Immutable contracts for arbitrary-robot morphology, manifests, and primitives.

The vocabulary here is deliberately body-neutral. Nothing in this module names a
specific robot, link, or joint: every field is either *measured* from an uploaded
model or drawn from the closed schema inventory. That is what makes the same code
serve a parallel-jaw arm, a five-finger hand, and a bimanual platform.

Two deliberate departures from ``rigby_v2.contracts``:

``RobotAssetManifestV1`` is not a ``RigAssetManifestV1``
    ``RigAssetManifestV1.unique_rig_components`` raises when ``free_root`` is
    false, because v2 certifies a free-root humanoid. A fixed-base arm is bolted
    to the world and has no free joint, so it can never satisfy that validator.
    Rather than fabricate a free root -- which would defeat the ``BASE_DRIFT``
    gate that proves the base stayed put -- this module defines its own manifest
    and declares ``CompilerRigProtocol``: the exact members that
    ``rigby_v2.motion.compiler.compile_motion_program`` reads from a rig. The
    protocol is enforced by ``tests/test_general_compiler_protocol.py`` so that a
    change in the base tree surfaces as a failing test rather than a runtime
    ``AttributeError``.

``RobotSiteV1`` widens the site vocabulary
    ``SiteSpecV1.semantic`` is the human literal set
    ``fingertip|palm|foot|gaze|task``. Robots have tips, grasp centers, and
    sensors instead. ``to_rig_site_spec`` narrows a general site back down for
    the v2 code paths that still expect the human vocabulary.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator
from rigby_v2.contracts import (
    ArtifactRefV1,
    Contract,
    CoordinateSystem,
    DofSpecV1,
    SiteSpecV1,
    Vec3,
)


SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


class DirectionV1(Contract):
    """A unit direction expressed in the robot's world frame."""

    x: float
    y: float
    z: float

    @model_validator(mode="after")
    def finite_and_unit(self) -> Self:
        values = (self.x, self.y, self.z)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Direction components must be finite")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Direction must be a unit vector; norm was {norm:.9f}")
        return self

    def dot(self, other: "DirectionV1") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def is_orthogonal_to(self, other: "DirectionV1", *, tolerance: float = 1e-4) -> bool:
        return abs(self.dot(other)) <= tolerance


# --------------------------------------------------------------------------
# Joints
# --------------------------------------------------------------------------


class JointKind(StrEnum):
    """MuJoCo joint kinds a compiled robot model can present."""

    HINGE = "hinge"
    SLIDE = "slide"
    BALL = "ball"
    FREE = "free"


class JointRole(StrEnum):
    """Measured functional role of a joint within its chain.

    Roles are assigned by sweeping the joint and observing what happens to the
    chain tip, never by reading the joint's name.
    """

    MAJOR_POSITION = "major_position"
    """Sweeping it translates the tip across a large fraction of the reach."""

    WRIST_ORIENT = "wrist_orient"
    """Sweeping it rotates the tip while barely translating it."""

    GRIP = "grip"
    """Sweeping it changes the separation between opposing distal members."""

    REDUNDANT = "redundant"
    """Moves the tip, but within the null space of an already-spanned direction."""

    IMMOBILE = "immobile"
    """Range is degenerate, or the joint has no measurable effect on any tip."""


class RobotJointV1(Contract):
    """One actuated degree of freedom, measured from the compiled model."""

    name: str = Field(min_length=1)
    kind: JointKind
    body: str = Field(min_length=1)
    axis: DirectionV1
    minimum: float
    maximum: float
    velocity_limit: float = Field(gt=0.0)
    effort_limit: float = Field(gt=0.0)
    role: JointRole
    tip_translation_m: float = Field(ge=0.0)
    """Tip displacement across the joint's full range, holding others at rest."""

    tip_rotation_rad: float = Field(ge=0.0)
    """Tip reorientation across the joint's full range, holding others at rest."""

    @model_validator(mode="after")
    def increasing_range(self) -> Self:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("Joint limits must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("Joint maximum must exceed minimum")
        return self

    @property
    def span(self) -> float:
        return self.maximum - self.minimum

    def to_dof_spec(self, *, acceleration_limit: float = 500.0) -> DofSpecV1:
        """Narrow to the v2 DOF spec consumed by the motion compiler."""

        return DofSpecV1(
            name=self.name,
            joint=self.name,
            minimum=self.minimum,
            maximum=self.maximum,
            velocity_limit=self.velocity_limit,
            acceleration_limit=acceleration_limit,
            effort_limit=self.effort_limit,
        )


# --------------------------------------------------------------------------
# Sites
# --------------------------------------------------------------------------


class SiteSemantic(StrEnum):
    """Body-neutral site vocabulary.

    Compare ``SiteSpecV1.semantic``, which is anatomical. These names describe a
    site's *function* in a schema binding, so they survive a change of body.
    """

    TIP = "tip"
    GRASP_CENTER = "grasp_center"
    CONTACT = "contact"
    JOINT = "joint"
    BASE = "base"
    GAZE = "gaze"
    SENSOR = "sensor"
    TASK = "task"


_SITE_SEMANTIC_TO_RIG: dict[SiteSemantic, str] = {
    SiteSemantic.TIP: "task",
    SiteSemantic.GRASP_CENTER: "palm",
    SiteSemantic.CONTACT: "fingertip",
    SiteSemantic.JOINT: "task",
    SiteSemantic.BASE: "task",
    SiteSemantic.GAZE: "gaze",
    SiteSemantic.SENSOR: "gaze",
    SiteSemantic.TASK: "task",
}


class RobotSiteV1(Contract):
    """A derived semantic attachment point.

    Sites are produced by ``rigby_general.morphology``, never hand-authored. The
    ``derivation`` string records which rule produced this site, so a wrong site
    can be traced back to the rule that invented it.
    """

    name: str = Field(min_length=1)
    body: str = Field(min_length=1)
    semantic: SiteSemantic
    position_m: Vec3
    derivation: str = Field(min_length=1)

    def to_rig_site_spec(self) -> SiteSpecV1:
        """Narrow to the v2 human site vocabulary for legacy code paths."""

        return SiteSpecV1(
            name=self.name,
            body=self.body,
            semantic=_SITE_SEMANTIC_TO_RIG[self.semantic],
        )


# --------------------------------------------------------------------------
# Effectors and chains
# --------------------------------------------------------------------------


class EffectorKind(StrEnum):
    """Effector classification, decided by a behavioural test.

    The test sweeps candidate grip joints and measures whether distal member
    surfaces monotonically approach one another. Link naming (``gripper``,
    ``finger``, ``hand``) is a hint that may seed the search; geometry decides.
    """

    PARALLEL_JAW = "parallel_jaw"
    """Exactly two members that converge under a shared grip degree of freedom."""

    MULTIFINGER = "multifinger"
    """Three or more independently articulated converging members."""

    TOOL_TIP = "tool_tip"
    """A single rigid distal member with no closure degree of freedom."""

    SENSOR = "sensor"
    """A camera or range link; provides a gaze origin, never a grasp."""


class EffectorV1(Contract):
    name: str = Field(min_length=1)
    kind: EffectorKind
    chain_id: str = Field(min_length=1)
    tip_body: str = Field(min_length=1)
    member_bodies: tuple[str, ...] = ()
    """Distal members that participate in closure. Empty for tips and sensors."""

    grip_joints: tuple[str, ...] = ()
    opposition_groups: tuple[tuple[str, ...], ...] = ()
    """Member groups that must oppose one another for a stable grasp."""

    max_aperture_m: float | None = Field(default=None, gt=0.0)

    closes_toward_upper: bool = True
    """Which end of the grip joints' range closes this gripper.

    Measured during the closure sweep, where it falls out of which end produced
    the smaller separation. It has to be carried because it is not a convention:
    the Franka hand closes toward its *lower* limit, and a controller that
    assumes otherwise opens the jaw when it means to grip.
    """

    site_names: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def closure_evidence_matches_kind(self) -> Self:
        needs_closure = self.kind in (EffectorKind.PARALLEL_JAW, EffectorKind.MULTIFINGER)
        if needs_closure:
            if not self.grip_joints:
                raise ValueError(f"{self.kind.value} effector requires grip joints")
            if self.max_aperture_m is None:
                raise ValueError(f"{self.kind.value} effector requires a measured aperture")
            if len(self.opposition_groups) < 2:
                raise ValueError(
                    f"{self.kind.value} effector requires at least two opposition groups"
                )
            expected = 2 if self.kind is EffectorKind.PARALLEL_JAW else 3
            if len(self.member_bodies) < expected:
                raise ValueError(
                    f"{self.kind.value} effector requires at least "
                    f"{expected} converging members"
                )
        elif self.grip_joints or self.opposition_groups or self.max_aperture_m is not None:
            raise ValueError(f"{self.kind.value} effector cannot declare closure evidence")
        members = [body for group in self.opposition_groups for body in group]
        if len(members) != len(set(members)):
            raise ValueError("A member cannot belong to two opposition groups")
        if set(members) - set(self.member_bodies):
            raise ValueError("Opposition groups must reference declared member bodies")
        return self

    @property
    def can_grasp(self) -> bool:
        return self.kind in (EffectorKind.PARALLEL_JAW, EffectorKind.MULTIFINGER)


class KinematicChainV1(Contract):
    """A maximal serial path from the base body to one leaf."""

    chain_id: str = Field(min_length=1)
    bodies: tuple[str, ...] = Field(min_length=1)
    joints: tuple[str, ...] = Field(min_length=1)
    tip_body: str = Field(min_length=1)
    root_body: str = Field(min_length=1)
    """First body on the chain that actually moves. The shoulder, in arm terms."""

    reach_radius_m: float = Field(gt=0.0)
    """Furthest the tip was ever measured from the chain root, in any direction."""

    workspace_centroid_m: Vec3
    """Mean reachable tip position. Its horizontal bearing from ``root_body`` is
    the chain's *working direction* -- the way this arm is built to point.

    Unlike the robot's front, this needs no operator confirmation: it is a fact
    about where the mechanism can put its own tip. That is why free-space schemas
    can ground on any robot the moment it is uploaded, while deictic ones wait
    for a person."""

    working_direction: DirectionV1
    """The ``out`` axis of this chain's workspace frame, stored so that grounding
    reads the envelope below in the frame it was measured in."""

    reach_envelope_m: tuple[float, ...] = ()
    """Furthest reachable radius per direction, row-major over azimuth then
    elevation bins in the workspace frame.

    A reachable set is not a ball. Most arms extend furthest along one axis and
    barely at all along another, so treating ``reach_radius_m`` as uniform asks
    for targets in directions the arm cannot go -- and the IK solver then spends
    its whole budget failing to get there. Measuring the envelope by direction is
    what makes "distal" mean "as far as this arm goes *that way*"."""

    reach_inner_m: tuple[float, ...] = ()
    """The other side of the reachable shell, same bin layout as
    ``reach_envelope_m``: the closest the effector gets to the chain root
    on each bearing. A serial arm cannot fold onto its own shoulder, and on
    a real industrial arm that hole is 15-25% of maximum reach."""

    envelope_azimuth_bins: int = Field(default=0, ge=0)
    envelope_elevation_bins: int = Field(default=0, ge=0)

    positioning_dof: int = Field(ge=0)
    orienting_dof: int = Field(ge=0)

    @model_validator(mode="after")
    def tip_is_last_body(self) -> Self:
        if self.bodies[-1] != self.tip_body:
            raise ValueError("Chain tip must be the last body on the chain")
        if self.root_body not in self.bodies:
            raise ValueError("Chain root must be a body on the chain")
        expected = self.envelope_azimuth_bins * self.envelope_elevation_bins
        if len(self.reach_envelope_m) != expected:
            raise ValueError(
                "Reach envelope must hold exactly azimuth x elevation entries"
            )
        if any(value < 0.0 for value in self.reach_envelope_m):
            raise ValueError("Reach envelope radii cannot be negative")
        if len(set(self.bodies)) != len(self.bodies):
            raise ValueError("Chain bodies must be unique")
        if len(set(self.joints)) != len(self.joints):
            raise ValueError("Chain joints must be unique")
        return self


# --------------------------------------------------------------------------
# Frames, symmetry, scale
# --------------------------------------------------------------------------


class FrameSource(StrEnum):
    DERIVED = "derived"
    """Proposed by measurement; usable, but not yet confirmed by a person."""

    OPERATOR_CONFIRMED = "operator_confirmed"
    """A person accepted or corrected the proposal at upload time."""


class IntrinsicFrameV1(Contract):
    """The robot's own up / front / lateral axes.

    ``up`` is free -- it is the negation of gravity. ``front`` is genuinely
    ambiguous for a bare arm bolted to a table, so it is proposed from evidence
    and then confirmed by a person exactly once. Deixis (``HITHER``/``THITHER``)
    and any INTRINSIC-frame schema are meaningless without it, which is why
    ``FrameSource.DERIVED`` is recorded rather than silently trusted.
    """

    up: DirectionV1
    front: DirectionV1
    lateral: DirectionV1
    source: FrameSource
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    """Which proposals agreed, e.g. ``reach_centroid``, ``base_principal_axis``."""

    @model_validator(mode="after")
    def orthogonal_right_handed(self) -> Self:
        if not self.up.is_orthogonal_to(self.front):
            raise ValueError("Intrinsic up and front must be orthogonal")
        if not self.up.is_orthogonal_to(self.lateral):
            raise ValueError("Intrinsic up and lateral must be orthogonal")
        if not self.front.is_orthogonal_to(self.lateral):
            raise ValueError("Intrinsic front and lateral must be orthogonal")
        cross = (
            self.up.y * self.front.z - self.up.z * self.front.y,
            self.up.z * self.front.x - self.up.x * self.front.z,
            self.up.x * self.front.y - self.up.y * self.front.x,
        )
        handedness = (
            cross[0] * self.lateral.x
            + cross[1] * self.lateral.y
            + cross[2] * self.lateral.z
        )
        if handedness < 0.9:
            raise ValueError("Intrinsic frame must be right-handed: up x front = lateral")
        if self.source is FrameSource.OPERATOR_CONFIRMED and self.confidence < 1.0:
            raise ValueError("An operator-confirmed frame is certain by definition")
        return self


class SymmetryV1(Contract):
    """A measured mirror relationship between two chains."""

    plane_normal: DirectionV1
    plane_point_m: Vec3
    left_chain_id: str = Field(min_length=1)
    right_chain_id: str = Field(min_length=1)
    residual_m: float = Field(ge=0.0)
    """Worst per-body mismatch after reflection. Small means genuinely mirrored."""

    @model_validator(mode="after")
    def distinct_chains(self) -> Self:
        if self.left_chain_id == self.right_chain_id:
            raise ValueError("A chain cannot be its own mirror")
        return self


class RobotScaleV1(Contract):
    """Measured magnitudes. Every magnitude-neutral term grounds through these.

    This is the whole reason an arbitrary robot works: language commits to
    ``distal`` and ``fast``, not to metres and rad/s, so the semantic layer stays
    body-neutral and only this record is body-specific.
    """

    reach_radius_m: float = Field(gt=0.0)
    characteristic_length_m: float = Field(gt=0.0)
    neutral_speed_mps: float = Field(gt=0.0)
    base_footprint_m: float = Field(gt=0.0)
    payload_kg: float = Field(gt=0.0)
    total_mass_kg: float = Field(gt=0.0)
    workspace_centroid_m: Vec3
    reach_samples: int = Field(gt=0)
    """How many IK-feasible configurations the envelope was measured from."""

    @model_validator(mode="after")
    def lengths_are_ordered(self) -> Self:
        if self.characteristic_length_m > self.reach_radius_m:
            raise ValueError("Characteristic link length cannot exceed total reach")
        return self


# --------------------------------------------------------------------------
# Morphology
# --------------------------------------------------------------------------


class MorphologyClass(StrEnum):
    """Which gate set and controller a robot is entitled to.

    Anything outside the supported classes is refused at ingest with a typed
    failure rather than being simulated on gates that do not describe it.
    """

    FIXED_BASE_ARM = "fixed_base_arm"
    FIXED_BASE_BIMANUAL = "fixed_base_bimanual"
    DEXTEROUS_EFFECTOR = "dexterous_effector"
    """A hand with no arm: it closes, but it cannot place itself.

    Refusing these conflated two different things. The two-axis minimum exists to
    turn away a cart on a rail -- a mechanism that moves without positioning
    anything -- and a five-fingered hand tripped it for the opposite reason: it
    positions nothing because it has nothing to position *with*, while being
    exactly the part of a robot that does the holding.

    Admitted, and then honestly limited. Every path schema requires positioning,
    so a hand affords none of them; what it affords is the statives and the
    contact schemas, which is what a hand can actually be asked for."""

    UNSUPPORTED_FLOATING_BASE = "unsupported_floating_base"
    UNSUPPORTED_TOPOLOGY = "unsupported_topology"


SUPPORTED_MORPHOLOGY_CLASSES = frozenset(
    {
        MorphologyClass.FIXED_BASE_ARM,
        MorphologyClass.FIXED_BASE_BIMANUAL,
        MorphologyClass.DEXTEROUS_EFFECTOR,
    }
)


class RobotMorphologyV1(Contract):
    """Everything measured from an uploaded model, before any motion exists."""

    schema_version: Literal["1.0"] = "1.0"
    robot_id: str = Field(min_length=1)
    morphology_class: MorphologyClass
    base_body: str = Field(min_length=1)
    joints: tuple[RobotJointV1, ...] = Field(min_length=1)
    chains: tuple[KinematicChainV1, ...] = Field(min_length=1)
    effectors: tuple[EffectorV1, ...] = Field(min_length=1)
    sites: tuple[RobotSiteV1, ...] = Field(min_length=1)
    intrinsic_frame: IntrinsicFrameV1
    scale: RobotScaleV1
    symmetry: SymmetryV1 | None = None
    self_collision_pairs: tuple[tuple[str, str], ...] = ()
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)

    @model_validator(mode="after")
    def internally_consistent(self) -> Self:
        joint_names = {joint.name for joint in self.joints}
        if len(joint_names) != len(self.joints):
            raise ValueError("Joint names must be unique")

        chain_ids = {chain.chain_id for chain in self.chains}
        if len(chain_ids) != len(self.chains):
            raise ValueError("Chain identifiers must be unique")
        for chain in self.chains:
            missing = set(chain.joints) - joint_names
            if missing:
                raise ValueError(
                    f"Chain {chain.chain_id!r} names unknown joints: {sorted(missing)}"
                )

        site_names = {site.name for site in self.sites}
        if len(site_names) != len(self.sites):
            raise ValueError("Site names must be unique")

        effector_names = {effector.name for effector in self.effectors}
        if len(effector_names) != len(self.effectors):
            raise ValueError("Effector names must be unique")
        for effector in self.effectors:
            if effector.chain_id not in chain_ids:
                raise ValueError(
                    f"Effector {effector.name!r} names unknown chain {effector.chain_id!r}"
                )
            missing_sites = set(effector.site_names) - site_names
            if missing_sites:
                raise ValueError(
                    f"Effector {effector.name!r} names unknown sites: {sorted(missing_sites)}"
                )
            missing_grip = set(effector.grip_joints) - joint_names
            if missing_grip:
                raise ValueError(
                    f"Effector {effector.name!r} names unknown grip joints: "
                    f"{sorted(missing_grip)}"
                )

        if self.morphology_class is MorphologyClass.FIXED_BASE_BIMANUAL:
            if len({e.chain_id for e in self.effectors if e.can_grasp}) < 2:
                raise ValueError("A bimanual morphology requires two grasping chains")
        if self.symmetry is not None:
            # Symmetry is evidence, not a requirement. Plenty of real dual-arm
            # rigs are not mirrored; left and right can still be assigned from
            # the intrinsic lateral axis. When a mirror *is* measured it must at
            # least name chains that exist.
            if self.symmetry.left_chain_id not in chain_ids:
                raise ValueError("Symmetry left chain is not a known chain")
            if self.symmetry.right_chain_id not in chain_ids:
                raise ValueError("Symmetry right chain is not a known chain")

        for first, second in self.self_collision_pairs:
            if first == second:
                raise ValueError("A body cannot self-collide with itself")

        return self

    @property
    def is_supported(self) -> bool:
        return self.morphology_class in SUPPORTED_MORPHOLOGY_CLASSES

    @property
    def grasping_effectors(self) -> tuple[EffectorV1, ...]:
        return tuple(effector for effector in self.effectors if effector.can_grasp)

    def joint(self, name: str) -> RobotJointV1:
        for candidate in self.joints:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown joint: {name}")

    def site(self, name: str) -> RobotSiteV1:
        for candidate in self.sites:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown site: {name}")


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


@runtime_checkable
class CompilerRigProtocol(Protocol):
    """The exact rig surface ``compile_motion_program`` reads.

    ``rigby_v2.motion.compiler`` annotates its ``rig`` parameter as
    ``RigAssetManifestV1`` but touches only these four members. Declaring the
    real requirement lets a fixed-base robot -- which can never satisfy that
    class's free-root validator -- drive the certified v2 compiler unchanged.
    ``tests/test_general_compiler_protocol.py`` asserts the base tree still reads
    nothing else.
    """

    @property
    def rig_id(self) -> str: ...

    @property
    def dofs(self) -> tuple[DofSpecV1, ...]: ...

    @property
    def rest_qpos(self) -> tuple[float, ...]: ...

    def content_hash(self) -> str: ...


class RobotAssetManifestV1(Contract):
    """A compiled, hash-pinned robot ready to be simulated.

    Structurally satisfies :class:`CompilerRigProtocol`.
    """

    schema_version: Literal["1.0"] = "1.0"
    rig_id: str = Field(min_length=1)
    mjcf: ArtifactRefV1
    source_asset: ArtifactRefV1
    source_format: Literal["urdf", "mjcf"]
    fixed_base: bool = True
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)
    dofs: tuple[DofSpecV1, ...] = Field(min_length=1)
    actuator_order: tuple[str, ...] = Field(min_length=1)
    rest_qpos: tuple[float, ...] = Field(min_length=1)
    sites: tuple[RobotSiteV1, ...] = Field(min_length=1)
    morphology: RobotMorphologyV1
    adjacent_collision_exclusions: tuple[tuple[str, str], ...] = ()
    asset_licenses: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_with_morphology(self) -> Self:
        if self.rig_id != self.morphology.robot_id:
            raise ValueError("Manifest rig_id must match the morphology robot_id")

        dof_names = [dof.name for dof in self.dofs]
        if len(dof_names) != len(set(dof_names)):
            raise ValueError("DOF names must be unique")
        if len(self.actuator_order) != len(set(self.actuator_order)):
            raise ValueError("Actuator order cannot contain duplicates")
        if set(self.actuator_order) - set(dof_names):
            raise ValueError("Every actuator must drive a declared DOF")

        morphology_joints = {joint.name for joint in self.morphology.joints}
        if set(dof_names) != morphology_joints:
            raise ValueError(
                "Manifest DOFs must match the measured morphology joints exactly"
            )

        manifest_sites = {site.name for site in self.sites}
        morphology_sites = {site.name for site in self.morphology.sites}
        if manifest_sites != morphology_sites:
            raise ValueError(
                "Manifest sites must match the derived morphology sites exactly"
            )

        if not self.fixed_base:
            raise ValueError(
                "This package certifies fixed-base robots only; a floating base has no "
                "BASE_DRIFT gate and no balance controller here"
            )
        if not self.morphology.is_supported:
            raise ValueError(
                f"Morphology class {self.morphology.morphology_class.value!r} "
                "is not supported"
            )
        return self

    def rig_sites(self) -> tuple[SiteSpecV1, ...]:
        """Narrow all sites to the v2 human vocabulary for legacy code paths."""

        return tuple(site.to_rig_site_spec() for site in self.sites)
