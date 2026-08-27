from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .hashing import canonical_json, content_hash, validate_sha256


def utc_now() -> datetime:
    return datetime.now(UTC)


class FrozenDict(dict[str, Any]):
    """A JSON-compatible mapping that cannot be changed after validation."""

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Contract mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class Contract(BaseModel):
    """Immutable, extra-forbidding base for every persisted public contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
    )

    def canonical_json(self) -> str:
        return canonical_json(self)

    def content_hash(self) -> str:
        return content_hash(self)

    @model_validator(mode="after")
    def deeply_immutable(self) -> Self:
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            frozen = _deep_freeze(value)
            if frozen is not value:
                object.__setattr__(self, field_name, frozen)
        return self


class Vec3(Contract):
    x: float
    y: float
    z: float

    @field_validator("x", "y", "z")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Vector components must be finite")
        return value


class QuaternionOrder(StrEnum):
    XYZW = "xyzw"
    WXYZ = "wxyz"


class CoordinateFrame(StrEnum):
    LOCAL = "local"
    WORLD = "world"


class QuaternionMeaning(StrEnum):
    ABSOLUTE = "absolute"
    REST_DELTA = "rest_delta"


class QuaternionConvention(Contract):
    order: QuaternionOrder
    frame: CoordinateFrame
    meaning: QuaternionMeaning


class Quaternion(Contract):
    values: tuple[float, float, float, float]
    convention: QuaternionConvention

    @field_validator("values")
    @classmethod
    def unit_and_finite(
        cls, values: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Quaternion components must be finite")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"Quaternion must be normalized; norm was {norm:.8f}")
        return values


class Pose(Contract):
    position: Vec3
    rotation: Quaternion


class ArtifactRefV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    sha256: str
    size_bytes: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    filename: str | None = None

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("filename")
    @classmethod
    def basename_only(cls, value: str | None) -> str | None:
        if value is not None and (
            "/" in value or "\\" in value or value in {".", ".."}
        ):
            raise ValueError("Artifact filename must be a basename")
        return value


class CoordinateSystem(Contract):
    up_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "+Z"
    forward_axis: Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"] = "-Y"
    right_handed: bool = True
    units: Literal["meters"] = "meters"

    @model_validator(mode="after")
    def distinct_axes(self) -> Self:
        if self.up_axis[-1] == self.forward_axis[-1]:
            raise ValueError("Up and forward axes must be orthogonal")
        return self


class PhaseKind(StrEnum):
    ANTICIPATION = "anticipation"
    SETUP = "setup"
    ATTACK = "attack"
    ACTION = "action"
    HOLD = "hold"
    RELEASE = "release"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


class MotionPhaseV2(Contract):
    phase_id: str = Field(min_length=1)
    kind: PhaseKind
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    energy: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def positive_interval(self) -> Self:
        if self.end_s <= self.start_s:
            raise ValueError("Phase end_s must be greater than start_s")
        return self


class InterpolationKind(StrEnum):
    QUINTIC = "quintic"
    SLERP = "slerp"
    SQUAD = "squad"
    STEP = "step"


class TrackOwnership(StrEnum):
    EXCLUSIVE = "exclusive"
    ADDITIVE = "additive"


class MotionKeyframeV2(Contract):
    time_s: float = Field(ge=0.0)
    position: Vec3 | None = None
    rotation: Quaternion | None = None
    joint_values: dict[str, float] = Field(default_factory=dict)
    hard: bool = False

    @field_validator("joint_values")
    @classmethod
    def finite_joints(cls, values: dict[str, float]) -> dict[str, float]:
        if not all(name and math.isfinite(value) for name, value in values.items()):
            raise ValueError("Joint names must be nonempty and values must be finite")
        return values

    @model_validator(mode="after")
    def has_target(self) -> Self:
        if self.position is None and self.rotation is None and not self.joint_values:
            raise ValueError(
                "A keyframe must specify position, rotation, or joint values"
            )
        return self


class MotionTrackV2(Contract):
    track_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    ownership: TrackOwnership = TrackOwnership.EXCLUSIVE
    priority: int = 0
    interpolation: InterpolationKind = InterpolationKind.QUINTIC
    keyframes: tuple[MotionKeyframeV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def monotonic_keyframes(self) -> Self:
        times = [keyframe.time_s for keyframe in self.keyframes]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("Track keyframe times must be strictly increasing")
        return self


class ContactEdgeV2(Contract):
    contact_id: str = Field(min_length=1)
    body_a: str = Field(min_length=1)
    body_b: str = Field(min_length=1)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    required: bool = True
    max_slip_m: float = Field(default=0.01, ge=0.0)
    max_penetration_m: float = Field(default=0.002, ge=0.0)

    @model_validator(mode="after")
    def valid_contact(self) -> Self:
        if self.body_a == self.body_b:
            raise ValueError("A contact edge requires two distinct bodies")
        if self.end_s <= self.start_s:
            raise ValueError("Contact end_s must be greater than start_s")
        return self


class AssertionSeverity(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class MotionAssertionV2(Contract):
    assertion_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    severity: AssertionSeverity = AssertionSeverity.HARD
    parameters: dict[str, Any] = Field(default_factory=dict)


class MotionProgramV2(Contract):
    schema_version: Literal["2.0"] = "2.0"
    program_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    source_text: str = Field(min_length=1)
    duration_s: float = Field(gt=0.0)
    rig_id: str = Field(min_length=1)
    scene_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    phases: tuple[MotionPhaseV2, ...] = Field(min_length=1)
    tracks: tuple[MotionTrackV2, ...] = Field(min_length=1)
    contacts: tuple[ContactEdgeV2, ...] = ()
    assertions: tuple[MotionAssertionV2, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        identifiers = [phase.phase_id for phase in self.phases]
        identifiers += [track.track_id for track in self.tracks]
        identifiers += [contact.contact_id for contact in self.contacts]
        identifiers += [assertion.assertion_id for assertion in self.assertions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Program identifiers must be unique")
        if any(phase.end_s > self.duration_s for phase in self.phases):
            raise ValueError("A phase extends beyond program duration")
        if any(
            keyframe.time_s > self.duration_s
            for track in self.tracks
            for keyframe in track.keyframes
        ):
            raise ValueError("A keyframe extends beyond program duration")
        if any(contact.end_s > self.duration_s for contact in self.contacts):
            raise ValueError("A contact extends beyond program duration")
        return self


class CollisionModel(StrEnum):
    PRIMITIVE = "primitive"
    CONVEX_DECOMPOSITION = "convex_decomposition"
    STATIC_MESH = "static_mesh"


class SceneJointV2(Contract):
    name: str = Field(min_length=1)
    kind: Literal["hinge", "slide", "ball", "free"]
    axis: Vec3 | None = None
    range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def valid_joint(self) -> Self:
        if self.kind in {"hinge", "slide"} and self.axis is None:
            raise ValueError("Hinge and slide joints require an axis")
        if self.range is not None and self.range[1] <= self.range[0]:
            raise ValueError("Joint range must be increasing")
        return self


class SceneObjectV2(Contract):
    object_id: str = Field(min_length=1)
    asset: ArtifactRefV1
    pose: Pose
    dynamic: bool
    articulated: bool = False
    mass_kg: float | None = Field(default=None, gt=0.0)
    collision_model: CollisionModel
    joints: tuple[SceneJointV2, ...] = ()
    affordances: tuple[str, ...] = ()
    material: str | None = None

    @model_validator(mode="after")
    def physically_complete(self) -> Self:
        if self.dynamic and self.mass_kg is None:
            raise ValueError("Dynamic objects require mass_kg")
        if self.dynamic and self.collision_model is CollisionModel.STATIC_MESH:
            raise ValueError("Dynamic objects cannot use raw concave static meshes")
        if self.articulated and not self.joints:
            raise ValueError("Articulated objects require joint metadata")
        return self


class SupportSurfaceV2(Contract):
    surface_id: str = Field(min_length=1)
    pose: Pose
    size_m: Vec3
    friction: tuple[float, float, float]


class CameraSpecV2(Contract):
    camera_id: str = Field(min_length=1)
    pose: Pose | None = None
    attached_body: str | None = None
    width: int = Field(default=960, gt=0)
    height: int = Field(default=540, gt=0)
    fov_y_degrees: float = Field(default=45.0, gt=0.0, lt=180.0)

    @model_validator(mode="after")
    def one_mount(self) -> Self:
        if (self.pose is None) == (self.attached_body is None):
            raise ValueError("Camera must specify exactly one of pose or attached_body")
        return self


class SceneAffordanceV1(Contract):
    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    site: str = Field(min_length=1)
    allowed_effectors: tuple[str, ...] = Field(min_length=1)


class SceneStatePredicateV1(Contract):
    object_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    target: str = Field(min_length=1)
    operator: Literal["ge", "le", "between", "near"]
    values: tuple[float, ...] = Field(min_length=1)
    units: str = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def finite_values(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Scene predicate values must be finite")
        return values

    @model_validator(mode="after")
    def valid_arity(self) -> Self:
        expected = 2 if self.operator == "between" else 1
        if len(self.values) != expected:
            raise ValueError(
                f"Scene predicate operator {self.operator!r} requires {expected} value(s)"
            )
        if self.operator == "between" and self.values[1] < self.values[0]:
            raise ValueError("Scene predicate interval must be increasing")
        return self


class SceneManifestV2(Contract):
    schema_version: Literal["2.0"] = "2.0"
    scene_id: str = Field(min_length=1)
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)
    rig_asset_hash: str
    objects: tuple[SceneObjectV2, ...] = ()
    supports: tuple[SupportSurfaceV2, ...] = ()
    cameras: tuple[CameraSpecV2, ...] = ()
    allowed_contact_pairs: tuple[tuple[str, str], ...] = ()
    pack_id: str | None = Field(default=None, min_length=1)
    compiled_mjcf: ArtifactRefV1 | None = None
    compiled_mjz: ArtifactRefV1 | None = None
    semantic_sites: dict[str, str] = Field(default_factory=dict)
    affordances: tuple[SceneAffordanceV1, ...] = ()
    state_predicates: tuple[SceneStatePredicateV1, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rig_asset_hash")
    @classmethod
    def valid_rig_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def unique_scene_names(self) -> Self:
        names = [item.object_id for item in self.objects]
        names += [item.surface_id for item in self.supports]
        names += [item.camera_id for item in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError(
                "Scene object, support, and camera identifiers must be unique"
            )
        if (self.compiled_mjcf is None) != (self.compiled_mjz is None):
            raise ValueError(
                "Compiled scenes require both MJCF and self-contained MJZ artifacts"
            )
        if self.compiled_mjcf is not None and self.pack_id is None:
            raise ValueError("Compiled scenes require a pack_id")
        affordance_names = [item.name for item in self.affordances]
        predicate_names = [item.name for item in self.state_predicates]
        if len(affordance_names) != len(set(affordance_names)):
            raise ValueError("Scene affordance names must be unique")
        if len(predicate_names) != len(set(predicate_names)):
            raise ValueError("Scene state predicate names must be unique")
        known_sites = set(self.semantic_sites.values())
        if any(item.site not in known_sites for item in self.affordances):
            raise ValueError("Scene affordances must target declared semantic sites")
        return self


class DofSpecV1(Contract):
    name: str = Field(min_length=1)
    joint: str = Field(min_length=1)
    minimum: float
    maximum: float
    velocity_limit: float = Field(gt=0.0)
    acceleration_limit: float = Field(default=500.0, gt=0.0)
    effort_limit: float = Field(gt=0.0)

    @model_validator(mode="after")
    def increasing_range(self) -> Self:
        if self.maximum <= self.minimum:
            raise ValueError("DOF maximum must exceed minimum")
        return self


class ColliderSpecV1(Contract):
    name: str = Field(min_length=1)
    body: str = Field(min_length=1)
    group: str = Field(min_length=1)
    shape: Literal["box", "capsule", "sphere", "ellipsoid", "cylinder", "convex_mesh"]


class SiteSpecV1(Contract):
    name: str = Field(min_length=1)
    body: str = Field(min_length=1)
    semantic: Literal["fingertip", "palm", "foot", "gaze", "task"]


class BodyInertiaSpecV1(Contract):
    name: str = Field(min_length=1)
    mass_kg: float = Field(gt=0.0)
    center_of_mass_m: Vec3
    diagonal_inertia_kg_m2: Vec3

    @model_validator(mode="after")
    def positive_inertia(self) -> Self:
        values = self.diagonal_inertia_kg_m2
        if min(values.x, values.y, values.z) <= 0.0:
            raise ValueError("Body diagonal inertia must be positive")
        return self


class RigAssetManifestV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    rig_id: str = Field(min_length=1)
    mjcf: ArtifactRefV1
    visual_asset: ArtifactRefV1 | None = None
    body_size_profile: Literal["small", "medium", "large", "custom"] = "medium"
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)
    free_root: bool = True
    dofs: tuple[DofSpecV1, ...] = Field(min_length=1)
    actuator_order: tuple[str, ...] = Field(min_length=1)
    colliders: tuple[ColliderSpecV1, ...] = ()
    bodies: tuple[BodyInertiaSpecV1, ...] = ()
    sites: tuple[SiteSpecV1, ...] = ()
    rest_qpos: tuple[float, ...] = Field(min_length=1)
    visual_skeleton_map: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    visual_body_map: dict[str, str] = Field(default_factory=dict)
    visual_node_map: dict[str, str] = Field(default_factory=dict)
    fixed_visual_bones: tuple[str, ...] = ()
    adjacent_collision_exclusions: tuple[tuple[str, str], ...] = ()
    asset_licenses: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_rig_components(self) -> Self:
        dof_names = [dof.name for dof in self.dofs]
        if len(dof_names) != len(set(dof_names)):
            raise ValueError("DOF names must be unique")
        if len(self.actuator_order) != len(set(self.actuator_order)):
            raise ValueError("Actuator order cannot contain duplicates")
        if not self.free_root:
            raise ValueError("Rigby v2 certification requires a free root")
        if set(self.fixed_visual_bones) - set(self.visual_node_map):
            raise ValueError("Fixed visual bones must resolve to source GLB nodes")
        if set(self.fixed_visual_bones) & set(self.visual_body_map):
            raise ValueError(
                "Fixed visual bones cannot claim independently simulated bodies"
            )
        return self


class SolverConfigV1(Contract):
    timestep_s: float = Field(default=1.0 / 240.0, gt=0.0)
    integrator: Literal["Euler", "implicit", "implicitfast", "RK4"] = "implicitfast"
    solver: Literal["PGS", "CG", "Newton"] = "Newton"
    iterations: int = Field(default=100, gt=0)
    tolerance: float = Field(default=1e-8, gt=0.0)


class CaptureConfigV1(Contract):
    fps: int = Field(default=30, gt=0)
    cameras: tuple[str, ...] = Field(min_length=1)
    width: int = Field(default=960, gt=0)
    height: int = Field(default=540, gt=0)


class PerturbationV1(Contract):
    name: str = Field(min_length=1)
    parameters: dict[str, float]

    @field_validator("parameters")
    @classmethod
    def finite_parameters(cls, values: dict[str, float]) -> dict[str, float]:
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Perturbation values must be finite")
        return values


class TrajectorySampleV1(Contract):
    time_s: float = Field(ge=0.0)
    qpos: tuple[float, ...] = Field(min_length=1)
    qvel: tuple[float, ...] = Field(min_length=1)
    qacc: tuple[float, ...] = ()

    @field_validator("qpos", "qvel", "qacc")
    @classmethod
    def finite_state(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Trajectory state must be finite")
        return values


class CandidateContactPlateauV1(Contract):
    contact_id: str = Field(min_length=1)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def increasing_interval(self) -> Self:
        if self.end_s <= self.start_s:
            raise ValueError("Candidate contact plateau must have positive duration")
        return self


class CandidateTrajectoryV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    candidate_id: str = Field(min_length=1)
    rig_id: str = Field(min_length=1)
    program_hash: str | None = None
    rig_hash: str | None = None
    quaternion_qpos_adrs: tuple[int, ...] = ()
    contact_plateaus: tuple[CandidateContactPlateauV1, ...] = ()
    samples: tuple[TrajectorySampleV1, ...] = Field(min_length=2)

    @field_validator("program_hash", "rig_hash")
    @classmethod
    def optional_hashes(cls, value: str | None) -> str | None:
        return validate_sha256(value) if value is not None else None

    @model_validator(mode="after")
    def consistent_samples(self) -> Self:
        times = [sample.time_s for sample in self.samples]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("Candidate sample times must be strictly increasing")
        if not math.isclose(times[0], 0.0, abs_tol=1e-12):
            raise ValueError("Candidate trajectory must begin at time zero")
        qpos_sizes = {len(sample.qpos) for sample in self.samples}
        qvel_sizes = {len(sample.qvel) for sample in self.samples}
        qacc_sizes = {len(sample.qacc) for sample in self.samples}
        if len(qpos_sizes) != 1 or len(qvel_sizes) != 1:
            raise ValueError(
                "Candidate samples must use consistent qpos/qvel dimensions"
            )
        if len(qacc_sizes) != 1 or next(iter(qacc_sizes)) not in {
            0,
            next(iter(qvel_sizes)),
        }:
            raise ValueError("Candidate qacc must be absent or match qvel dimensions")
        qpos_size = next(iter(qpos_sizes))
        for address in self.quaternion_qpos_adrs:
            if address < 0 or address + 4 > qpos_size:
                raise ValueError("Candidate quaternion address is outside qpos")
            for sample in self.samples:
                norm = math.sqrt(
                    sum(value * value for value in sample.qpos[address : address + 4])
                )
                if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(
                        "Candidate trajectory quaternions must be normalized"
                    )
        return self


class SimulationJobV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    job_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    program_hash: str
    scene_hash: str
    rig_hash: str
    candidate_hash: str
    program_artifact: ArtifactRefV1
    scene_artifact: ArtifactRefV1
    rig_artifact: ArtifactRefV1
    candidate_artifact: ArtifactRefV1
    solver: SolverConfigV1 = Field(default_factory=SolverConfigV1)
    capture: CaptureConfigV1
    seed: int = Field(ge=0)
    perturbations: tuple[PerturbationV1, ...] = ()
    submitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("program_hash", "scene_hash", "rig_hash", "candidate_hash")
    @classmethod
    def valid_content_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("submitted_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("submitted_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def artifacts_match_declared_hashes(self) -> Self:
        pairs = (
            ("program", self.program_hash, self.program_artifact.sha256),
            ("scene", self.scene_hash, self.scene_artifact.sha256),
            ("rig", self.rig_hash, self.rig_artifact.sha256),
            ("candidate", self.candidate_hash, self.candidate_artifact.sha256),
        )
        mismatched = [name for name, expected, actual in pairs if expected != actual]
        if mismatched:
            raise ValueError(
                "Artifact references do not match declared hashes: "
                + ", ".join(mismatched)
            )
        return self


class SubmitSimulationJobRequestV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    job: SimulationJobV1
    priority: int = 0
    idempotency_key: str | None = None

    @field_validator("idempotency_key")
    @classmethod
    def nonblank_key(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("idempotency_key cannot be blank")
        return value


class GateResultV1(Contract):
    gate: str = Field(min_length=1)
    passed: bool
    measured: float | bool | str | None = None
    threshold: float | bool | str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReproducibilityManifestV1(Contract):
    os: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    mujoco_version: str = Field(min_length=1)
    model_hash: str
    solver: SolverConfigV1
    seeds: tuple[int, ...] = Field(min_length=1)
    requested_capture: CaptureConfigV1 | None = None
    cameras: tuple[CameraSpecV2, ...]
    dependency_lock_hash: str

    @field_validator("model_hash", "dependency_lock_hash")
    @classmethod
    def valid_manifest_hash(cls, value: str) -> str:
        return validate_sha256(value)


class SimulationResultV1(Contract):
    schema_version: Literal["1.0"] = "1.0"
    result_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    job_id: str = Field(min_length=1)
    success: bool
    outcome: str = Field(min_length=1)
    trace: ArtifactRefV1
    videos: tuple[ArtifactRefV1, ...] = ()
    export: ArtifactRefV1 | None = None
    gates: tuple[GateResultV1, ...] = ()
    metrics: dict[str, float | int | bool | str] = Field(default_factory=dict)
    reproducibility: ReproducibilityManifestV1
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator("completed_at")
    @classmethod
    def aware_completion(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def consistent_gate_outcome(self) -> Self:
        if self.success and any(not gate.passed for gate in self.gates):
            raise ValueError(
                "A successful result cannot contain a failed certification gate"
            )
        return self
