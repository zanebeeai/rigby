"""Strict declarative object-pack contracts for safe scene composition."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]
MAX_POSITION_COMPONENT_M = 25.0
MAX_COLLISION_HALF_SIZE_M = 5.0
MAX_TOTAL_PARTS = 256
MAX_TOTAL_GEOMS = 256
MAX_TOTAL_SITES = 256
MAX_TOTAL_AFFORDANCES = 256
MAX_TOTAL_STATE_PREDICATES = 256


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _finite(values: tuple[float, ...], label: str) -> tuple[float, ...]:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} values must be finite")
    return values


def _bounded_position(values: Vec3, label: str) -> Vec3:
    _finite(values, label)
    if any(abs(value) > MAX_POSITION_COMPONENT_M for value in values):
        raise ValueError(
            f"{label} exceeds the {MAX_POSITION_COMPONENT_M:g} meter scene bound"
        )
    return values


class MaterialSpec(StrictModel):
    name: str = Field(min_length=1, max_length=64)
    friction: Vec3
    rgba: tuple[float, float, float, float] = (0.55, 0.55, 0.55, 1.0)

    @field_validator("friction")
    @classmethod
    def valid_friction(cls, value: Vec3) -> Vec3:
        _finite(value, "friction")
        if value[0] <= 0 or value[1] < 0 or value[2] < 0:
            raise ValueError(
                "friction requires positive sliding and nonnegative torsional/rolling values"
            )
        if value[0] > 5 or value[1] > 1 or value[2] > 1:
            raise ValueError("friction exceeds certified material bounds")
        return value

    @field_validator("rgba")
    @classmethod
    def valid_rgba(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        _finite(value, "rgba")
        if any(component < 0 or component > 1 for component in value):
            raise ValueError("rgba components must be between zero and one")
        return value


class CollisionKind(StrEnum):
    PRIMITIVE = "primitive"
    CONVEX_DECOMPOSITION = "convex_decomposition"


class CollisionGeomSpec(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    type: Literal["box", "sphere", "capsule", "cylinder", "convex_mesh"]
    size: tuple[float, ...] = ()
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat_wxyz: QuatWxyz = (1.0, 0.0, 0.0, 0.0)
    margin_m: float = Field(default=0.0, ge=0.0, le=0.01)
    contact_time_constant_s: float | None = Field(default=None, ge=0.005, le=0.1)
    contact_damping_ratio: float | None = Field(default=None, ge=0.1, le=5.0)
    asset_path: str | None = None
    asset_sha256: str | None = None

    @model_validator(mode="after")
    def valid_geometry(self) -> Self:
        if (self.contact_time_constant_s is None) != (
            self.contact_damping_ratio is None
        ):
            raise ValueError(
                "contact time constant and damping ratio must be declared together"
            )
        _finite(self.size, "size")
        _bounded_position(self.pos, "collision position")
        _finite(self.quat_wxyz, "quaternion")
        expected = {
            "box": 3,
            "sphere": 1,
            "capsule": 2,
            "cylinder": 2,
            "convex_mesh": 0,
        }[self.type]
        if len(self.size) != expected or any(value <= 0 for value in self.size):
            raise ValueError(
                f"{self.type} collision requires {expected} positive size values"
            )
        if any(value > MAX_COLLISION_HALF_SIZE_M for value in self.size):
            raise ValueError(
                f"collision size exceeds the {MAX_COLLISION_HALF_SIZE_M:g} meter bound"
            )
        norm = math.sqrt(sum(value * value for value in self.quat_wxyz))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("collision quaternion must be normalized WXYZ")
        if self.type == "convex_mesh":
            if self.asset_path is None or self.asset_sha256 is None:
                raise ValueError(
                    "convex mesh pieces require a content-addressed asset path and SHA-256"
                )
        elif self.asset_path is not None or self.asset_sha256 is not None:
            raise ValueError("primitive collision geometry cannot reference an asset")
        return self


class VisualMeshSpec(StrictModel):
    """A content-addressed mesh that can affect rendering, never physics."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    render_only: Literal[True] = True
    asset_path: str = Field(min_length=1, max_length=256)
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat_wxyz: QuatWxyz = (1.0, 0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)

    @model_validator(mode="after")
    def valid_render_mesh(self) -> Self:
        _bounded_position(self.pos, "visual mesh position")
        _finite(self.quat_wxyz, "visual mesh quaternion")
        _finite(self.scale, "visual mesh scale")
        norm = math.sqrt(sum(value * value for value in self.quat_wxyz))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("visual mesh quaternion must be normalized WXYZ")
        if any(value <= 0.0 or value > 100.0 for value in self.scale):
            raise ValueError("visual mesh scale must be positive and at most 100")
        return self


class JointSpec(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    type: Literal["hinge", "slide"]
    axis: Vec3
    range: tuple[float, float]
    damping: float = Field(default=0.2, ge=0, le=100)
    frictionloss: float = Field(default=0.02, ge=0, le=100)

    @model_validator(mode="after")
    def valid_joint(self) -> Self:
        _finite(self.axis, "joint axis")
        _finite(self.range, "joint range")
        norm = math.sqrt(sum(value * value for value in self.axis))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("joint axis must be a normalized unit vector")
        if self.range[1] <= self.range[0]:
            raise ValueError("joint range must be increasing")
        if self.type == "slide" and max(abs(value) for value in self.range) > 5:
            raise ValueError("slide joint range must be expressed in meters")
        if self.type == "hinge" and max(abs(value) for value in self.range) > 360:
            raise ValueError("hinge joint range must be expressed in degrees")
        return self


class PartSpec(StrictModel):
    part_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    parent: str | None = None
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat_wxyz: QuatWxyz = (1.0, 0.0, 0.0, 0.0)
    mass_kg: float = Field(gt=0, le=250)
    joint: JointSpec | None = None
    collision_kind: CollisionKind
    geoms: tuple[CollisionGeomSpec, ...] = Field(min_length=1, max_length=64)
    visual_meshes: tuple[VisualMeshSpec, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def valid_part(self) -> Self:
        _bounded_position(self.pos, "part position")
        _finite(self.quat_wxyz, "part quaternion")
        norm = math.sqrt(sum(value * value for value in self.quat_wxyz))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("part quaternion must be normalized WXYZ")
        names = [geom.name for geom in self.geoms]
        if len(names) != len(set(names)):
            raise ValueError("collision geom names must be unique within a part")
        visual_names = [mesh.name for mesh in self.visual_meshes]
        if len(visual_names) != len(set(visual_names)):
            raise ValueError("visual mesh names must be unique within a part")
        if set(names) & set(visual_names):
            raise ValueError(
                "collision and render-only geometry names must be distinct"
            )
        contains_mesh = any(geom.type == "convex_mesh" for geom in self.geoms)
        if self.collision_kind is CollisionKind.PRIMITIVE and contains_mesh:
            raise ValueError("primitive collision parts cannot contain mesh geometry")
        if (
            self.collision_kind is CollisionKind.CONVEX_DECOMPOSITION
            and not contains_mesh
        ):
            raise ValueError(
                "convex_decomposition requires explicit convex mesh pieces"
            )
        return self


class SiteKind(StrEnum):
    GRASP = "grasp"
    CONTACT = "contact"
    INSERTION = "insertion"
    SUPPORT = "support"
    STATE = "state"


class SemanticSiteSpec(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    part: str = Field(min_length=1)
    kind: SiteKind
    pos: Vec3
    size_m: float = Field(default=0.01, gt=0, le=0.1)

    @field_validator("pos")
    @classmethod
    def finite_position(cls, value: Vec3) -> Vec3:
        return _bounded_position(value, "site position")


class AffordanceSpec(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    kind: Literal[
        "grasp", "place", "pull", "push", "rotate", "strike", "lift", "open", "close"
    ]
    site: str = Field(min_length=1)
    allowed_effectors: tuple[Literal["left_palm", "right_palm", "both_palms"], ...] = (
        Field(min_length=1)
    )


class StatePredicateSpec(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    target: str = Field(min_length=1)
    operator: Literal["ge", "le", "between"]
    values: tuple[float, ...]
    units: Literal["meters", "degrees", "newtons", "boolean"]

    @model_validator(mode="after")
    def valid_predicate(self) -> Self:
        _finite(self.values, "predicate")
        required = 2 if self.operator == "between" else 1
        if len(self.values) != required:
            raise ValueError(f"{self.operator} predicate requires {required} values")
        if self.operator == "between" and self.values[1] <= self.values[0]:
            raise ValueError("between predicate bounds must increase")
        if self.units == "meters" and any(
            abs(value) > MAX_POSITION_COMPONENT_M for value in self.values
        ):
            raise ValueError("meter predicate exceeds the certified scene bound")
        if self.units == "degrees" and any(abs(value) > 360 for value in self.values):
            raise ValueError("degree predicate exceeds one full rotation")
        if self.units == "newtons" and any(
            abs(value) > 100_000 for value in self.values
        ):
            raise ValueError("force predicate exceeds the certified bound")
        if self.units == "boolean" and any(
            value not in {0.0, 1.0} for value in self.values
        ):
            raise ValueError("boolean predicates must use zero or one")
        return self


class ObjectSpec(StrictModel):
    object_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    pos: Vec3
    quat_wxyz: QuatWxyz = (1.0, 0.0, 0.0, 0.0)
    dynamic: bool
    articulated: bool
    mass_kg: float = Field(gt=0, le=500)
    material: MaterialSpec
    parts: tuple[PartSpec, ...] = Field(min_length=1, max_length=128)
    sites: tuple[SemanticSiteSpec, ...] = ()
    affordances: tuple[AffordanceSpec, ...] = ()
    state_predicates: tuple[StatePredicateSpec, ...] = ()

    @model_validator(mode="after")
    def valid_object(self) -> Self:
        _bounded_position(self.pos, "object position")
        _finite(self.quat_wxyz, "object quaternion")
        norm = math.sqrt(sum(value * value for value in self.quat_wxyz))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("object quaternion must be normalized WXYZ")
        part_names = [part.part_id for part in self.parts]
        if len(part_names) != len(set(part_names)):
            raise ValueError("part IDs must be unique")
        roots = [part for part in self.parts if part.parent is None]
        if len(roots) != 1 or roots[0].joint is not None:
            raise ValueError("objects require exactly one jointless root part")
        known = set(part_names)
        if any(
            part.parent not in known for part in self.parts if part.parent is not None
        ):
            raise ValueError("part parent must reference another part")
        for part in self.parts:
            seen = {part.part_id}
            parent = part.parent
            while parent is not None:
                if parent in seen:
                    raise ValueError("part hierarchy must be acyclic")
                seen.add(parent)
                parent = next(
                    candidate.parent
                    for candidate in self.parts
                    if candidate.part_id == parent
                )
        joints = {part.joint.name for part in self.parts if part.joint is not None}
        if self.articulated != bool(joints):
            raise ValueError("articulated objects require explicit joint metadata")
        if self.articulated and (
            not self.sites or not self.affordances or not self.state_predicates
        ):
            raise ValueError(
                "articulated objects require semantic sites, affordances, and state predicates"
            )
        if not math.isclose(
            sum(part.mass_kg for part in self.parts),
            self.mass_kg,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError("object mass_kg must equal the sum of part masses")
        site_names = {site.name for site in self.sites}
        if len(site_names) != len(self.sites) or any(
            site.part not in known for site in self.sites
        ):
            raise ValueError("semantic sites must be unique and reference known parts")
        if any(affordance.site not in site_names for affordance in self.affordances):
            raise ValueError("affordances must reference semantic sites")
        targets = site_names | joints
        if any(predicate.target not in targets for predicate in self.state_predicates):
            raise ValueError("state predicates must reference a semantic site or joint")
        return self


class SupportSpec(StrictModel):
    support_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    pos: Vec3
    size: Vec3
    material: MaterialSpec
    site_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")

    @model_validator(mode="after")
    def valid_support(self) -> Self:
        _bounded_position(self.pos, "support position")
        _finite(self.size, "support size")
        if any(value <= 0 or value > 10 for value in self.size):
            raise ValueError(
                "support half-sizes must be positive and at most 10 meters"
            )
        return self


class ObjectPackSpec(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    units: Literal["meters_degrees_kilograms"]
    pack_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(min_length=1, max_length=500)
    supports: tuple[SupportSpec, ...] = ()
    objects: tuple[ObjectSpec, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        names = [support.support_id for support in self.supports]
        names += [obj.object_id for obj in self.objects]
        if len(names) != len(set(names)):
            raise ValueError("support and object IDs must be globally unique")
        totals = {
            "parts": sum(len(obj.parts) for obj in self.objects),
            "collision geoms": sum(
                len(part.geoms) for obj in self.objects for part in obj.parts
            ),
            "visual meshes": sum(
                len(part.visual_meshes)
                for obj in self.objects
                for part in obj.parts
            ),
            "all geoms": sum(
                len(part.geoms) + len(part.visual_meshes)
                for obj in self.objects
                for part in obj.parts
            ),
            "semantic sites": sum(len(obj.sites) for obj in self.objects),
            "affordances": sum(len(obj.affordances) for obj in self.objects),
            "state predicates": sum(len(obj.state_predicates) for obj in self.objects),
        }
        limits = {
            "parts": MAX_TOTAL_PARTS,
            "collision geoms": MAX_TOTAL_GEOMS,
            "visual meshes": MAX_TOTAL_GEOMS,
            "all geoms": MAX_TOTAL_GEOMS,
            "semantic sites": MAX_TOTAL_SITES,
            "affordances": MAX_TOTAL_AFFORDANCES,
            "state predicates": MAX_TOTAL_STATE_PREDICATES,
        }
        exceeded = [
            f"{name}={totals[name]}>{limits[name]}"
            for name in totals
            if totals[name] > limits[name]
        ]
        if exceeded:
            raise ValueError(
                "object pack exceeds aggregate complexity limits: "
                + ", ".join(exceeded)
            )
        return self
